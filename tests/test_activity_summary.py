"""Tests for the activity-summary computation, its storage, and its read endpoint.

One of these tests is a note rather than a check on behaviour:
:func:`test_the_summary_table_is_a_database_schema_follow_on` records where
``insights.activity_summary`` is declared. This repository contains no DDL — every
``insights.*`` table is created by ``database-schema``'s ``_INSIGHTS_STATEMENTS`` — so the
computation below is written against a table that repository must add, and the test fails
if anyone resolves that by introducing a ``CREATE TABLE`` here.
"""

from __future__ import annotations

import inspect
import re
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from insights import computations
from insights.computations import (
    ACTIVITY_SUMMARY_DEFAULT_DAYS,
    ACTIVITY_SUMMARY_TABLE,
    EVENT_TYPE_DIMENSION,
    POLICY_ID_DIMENSION,
    activity_summary_window,
    compute_and_store_activity_summary,
)
from insights.models import ActivitySummaryItem
from tests.conftest import TEST_CACHE_GENERATION


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable


DAY_ONE = datetime(2026, 9, 10, 9, 30, tzinfo=UTC)
DAY_TWO = datetime(2026, 9, 11, 14, 15, tzinfo=UTC)

SUBJECT = UUID("11111111-1111-1111-1111-111111111111")
OTHER_SUBJECT = UUID("22222222-2222-2222-2222-222222222222")


class _FakeEvent:
    """The three fields the summary reads off an event."""

    def __init__(self, event_type: str, subject_id: UUID, occurred_at: datetime) -> None:
        self.event_type = event_type
        self.subject_id = subject_id
        self.occurred_at = occurred_at


class _FakeImpression:
    """The four fields the summary reads off an impression."""

    def __init__(self, policy_id: str, subject_id: UUID, candidate_set_id: UUID, occurred_at: datetime) -> None:
        self.policy_id = policy_id
        self.subject_id = subject_id
        self.candidate_set_id = candidate_set_id
        self.occurred_at = occurred_at


def _reader(items: Iterable[Any]) -> Any:
    """Return a stand-in for a read-path async iterator yielding ``items``."""
    collected = list(items)

    def read(*_args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        read.calls.append(kwargs)  # type: ignore[attr-defined]

        async def iterate() -> AsyncIterator[Any]:
            for item in collected:
                yield item

        return iterate()

    read.calls = []  # type: ignore[attr-defined]
    return read


def _executed(pool: MagicMock) -> list[tuple[str, Any]]:
    """Return the ``(statement, params)`` pairs executed against the pool's cursor."""
    cursor = pool.connection().cursor()
    return [(call.args[0], call.args[1] if len(call.args) > 1 else None) for call in cursor.execute.await_args_list]


def _inserts(pool: MagicMock) -> list[Any]:
    """Return the parameters of every summary INSERT."""
    return [params for statement, params in _executed(pool) if "INSERT INTO insights.activity_summary" in statement]


def _normalized(statement: str) -> str:
    return re.sub(r"\s+", " ", statement).strip()


# ── The window ──────────────────────────────────────────────────────────────


def test_the_window_covers_whole_utc_days_including_today() -> None:
    """Seven days ending at the start of tomorrow, so today is in the window in full."""
    since, until = activity_summary_window(7, now=datetime(2026, 9, 14, 17, 45, tzinfo=UTC))

    assert since == datetime(2026, 9, 8, tzinfo=UTC)
    assert until == datetime(2026, 9, 15, tzinfo=UTC)
    assert (until - since).days == 7


def test_the_window_is_a_whole_day_at_its_minimum() -> None:
    """One day is today, from its own midnight."""
    since, until = activity_summary_window(1, now=datetime(2026, 9, 14, 0, 1, tzinfo=UTC))

    assert since == datetime(2026, 9, 14, tzinfo=UTC)
    assert until == datetime(2026, 9, 15, tzinfo=UTC)


def test_the_window_rejects_a_non_positive_span() -> None:
    """An empty window would silently summarise nothing."""
    with pytest.raises(ValueError, match="days must be positive"):
        activity_summary_window(0)


# ── The computation ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_events_are_counted_per_day_and_type_with_distinct_subjects(mock_pg_pool: MagicMock) -> None:
    """Two events from one subject are two records and one subject."""
    events = _reader(
        [
            _FakeEvent("recommendation.shown", SUBJECT, DAY_ONE),
            _FakeEvent("recommendation.shown", SUBJECT, DAY_ONE),
            _FakeEvent("recommendation.shown", OTHER_SUBJECT, DAY_ONE),
            _FakeEvent("recommendation.opened", SUBJECT, DAY_TWO),
        ]
    )
    with (
        patch.object(computations, "read_events", events),
        patch.object(computations, "read_impressions", _reader([])),
    ):
        written = await compute_and_store_activity_summary(AsyncMock(), mock_pg_pool)

    assert written == 2
    rows = _inserts(mock_pg_pool)
    assert rows[0] == (date(2026, 9, 10), EVENT_TYPE_DIMENSION, "recommendation.shown", 3, 2, None)
    assert rows[1] == (date(2026, 9, 11), EVENT_TYPE_DIMENSION, "recommendation.opened", 1, 1, None)


@pytest.mark.asyncio
async def test_impressions_are_counted_per_day_and_policy_with_distinct_candidate_sets(mock_pg_pool: MagicMock) -> None:
    """A policy's row carries its impression count, its subjects, and its candidate sets."""
    candidate_set = uuid4()
    impressions = _reader(
        [
            _FakeImpression("policy-a", SUBJECT, candidate_set, DAY_ONE),
            _FakeImpression("policy-a", OTHER_SUBJECT, candidate_set, DAY_ONE),
            _FakeImpression("policy-a", SUBJECT, uuid4(), DAY_ONE),
            _FakeImpression("policy-b", SUBJECT, uuid4(), DAY_ONE),
        ]
    )
    with (
        patch.object(computations, "read_events", _reader([])),
        patch.object(computations, "read_impressions", impressions),
    ):
        written = await compute_and_store_activity_summary(AsyncMock(), mock_pg_pool)

    assert written == 2
    rows = _inserts(mock_pg_pool)
    assert rows[0] == (date(2026, 9, 10), POLICY_ID_DIMENSION, "policy-a", 3, 2, 2)
    assert rows[1] == (date(2026, 9, 10), POLICY_ID_DIMENSION, "policy-b", 1, 1, 1)


@pytest.mark.asyncio
async def test_event_rows_carry_no_candidate_set_count(mock_pg_pool: MagicMock) -> None:
    """An event has no ranking decision, so its candidate-set count is null, not zero."""
    with (
        patch.object(computations, "read_events", _reader([_FakeEvent("collection.added", SUBJECT, DAY_ONE)])),
        patch.object(computations, "read_impressions", _reader([])),
    ):
        await compute_and_store_activity_summary(AsyncMock(), mock_pg_pool)

    assert _inserts(mock_pg_pool)[0][-1] is None


@pytest.mark.asyncio
async def test_both_reads_use_the_product_analytics_purpose(mock_pg_pool: MagicMock) -> None:
    """Counting what happened is product analytics, not model training."""
    events, impressions = _reader([]), _reader([])
    with (
        patch.object(computations, "read_events", events),
        patch.object(computations, "read_impressions", impressions),
    ):
        await compute_and_store_activity_summary(AsyncMock(), mock_pg_pool)

    assert events.calls[0]["purposes"] == ("product_analytics",)
    assert impressions.calls[0]["purposes"] == ("product_analytics",)


@pytest.mark.asyncio
async def test_both_reads_are_bounded_by_the_same_window(mock_pg_pool: MagicMock) -> None:
    """One window for both tables, so the two halves of a day's summary agree."""
    events, impressions = _reader([]), _reader([])
    with (
        patch.object(computations, "read_events", events),
        patch.object(computations, "read_impressions", impressions),
    ):
        await compute_and_store_activity_summary(AsyncMock(), mock_pg_pool, days=3)

    expected = activity_summary_window(3)
    assert (events.calls[0]["since"], events.calls[0]["until"]) == expected
    assert (impressions.calls[0]["since"], impressions.calls[0]["until"]) == expected


@pytest.mark.asyncio
async def test_the_computation_never_fetches_from_the_catalog_api(mock_pg_pool: MagicMock) -> None:
    """The only computation whose input is the database, not the promoted contract."""
    client = AsyncMock()
    with (
        patch.object(computations, "read_events", _reader([])),
        patch.object(computations, "read_impressions", _reader([])),
    ):
        await compute_and_store_activity_summary(client, mock_pg_pool)

    client.get.assert_not_awaited()


# ── Storage ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_window_is_replaced_transactionally(mock_pg_pool: MagicMock) -> None:
    """The delete is bounded to the recomputed window and precedes the inserts."""
    with (
        patch.object(computations, "read_events", _reader([_FakeEvent("collection.added", SUBJECT, DAY_ONE)])),
        patch.object(computations, "read_impressions", _reader([])),
    ):
        await compute_and_store_activity_summary(AsyncMock(), mock_pg_pool, days=7)

    statements = [_normalized(statement) for statement, _params in _executed(mock_pg_pool)]
    delete_index = next(i for i, s in enumerate(statements) if s.startswith("DELETE FROM insights.activity_summary"))
    insert_index = next(i for i, s in enumerate(statements) if s.startswith("INSERT INTO insights.activity_summary"))
    assert delete_index < insert_index
    assert "WHERE summary_date >= %s AND summary_date < %s" in statements[delete_index]

    since, until = activity_summary_window(7)
    assert _executed(mock_pg_pool)[delete_index][1] == (since.date(), until.date())
    mock_pg_pool.connection().transaction.assert_called_with()


@pytest.mark.asyncio
async def test_an_empty_window_is_still_rewritten(mock_pg_pool: MagicMock) -> None:
    """A revocation must remove the previous run's counts, not leave them standing.

    This is the deliberate difference from the catalog-derived computations, which leave the
    previous snapshot intact on an empty producer result.
    """
    with (
        patch.object(computations, "read_events", _reader([])),
        patch.object(computations, "read_impressions", _reader([])),
    ):
        written = await compute_and_store_activity_summary(AsyncMock(), mock_pg_pool)

    assert written == 0
    statements = [_normalized(statement) for statement, _params in _executed(mock_pg_pool)]
    assert any(s.startswith("DELETE FROM insights.activity_summary") for s in statements)
    assert not any(s.startswith("INSERT INTO insights.activity_summary") for s in statements)


# ── Lifecycle logging ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_successful_run_is_logged_under_its_own_insight_type(mock_pg_pool: MagicMock) -> None:
    """The computation runs under _record_lifecycle, keyed activity_summary."""
    with (
        patch.object(computations, "read_events", _reader([_FakeEvent("collection.added", SUBJECT, DAY_ONE)])),
        patch.object(computations, "read_impressions", _reader([])),
        patch.object(computations, "_log_computation", AsyncMock()) as mock_log,
    ):
        await compute_and_store_activity_summary(AsyncMock(), mock_pg_pool)

    mock_log.assert_awaited_once()
    assert mock_log.await_args.args[1] == "activity_summary"
    assert mock_log.await_args.args[2] == "completed"
    assert mock_log.await_args.args[4] == 1


@pytest.mark.asyncio
async def test_a_failed_run_is_logged_and_re_raised(mock_pg_pool: MagicMock) -> None:
    """A read failure is recorded as failed and still propagates to the scheduler."""

    def exploding_read(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        async def iterate() -> AsyncIterator[Any]:
            raise RuntimeError("consent join failed")
            yield  # pragma: no cover  # Makes the function an async generator.

        return iterate()

    with (
        patch.object(computations, "read_events", exploding_read),
        patch.object(computations, "_log_computation", AsyncMock()) as mock_log,
        pytest.raises(RuntimeError, match="consent join failed"),
    ):
        await compute_and_store_activity_summary(AsyncMock(), mock_pg_pool)

    assert mock_log.await_args.args[2] == "failed"
    assert mock_log.await_args.kwargs["error_message"] == "RuntimeError: consent join failed"


# ── The read endpoint ───────────────────────────────────────────────────────


def _summary_row(summary_date: date = date(2026, 9, 10), dimension: str = EVENT_TYPE_DIMENSION) -> tuple[Any, ...]:
    return (summary_date, dimension, "recommendation.shown", 42, 7, None)


def test_the_endpoint_returns_the_stored_rows(test_client: TestClient, mock_pg_pool: MagicMock) -> None:
    """The endpoint serves the summary; it does not recompute it."""
    mock_pg_pool.connection().cursor().fetchall.return_value = [
        _summary_row(),
        (date(2026, 9, 10), POLICY_ID_DIMENSION, "policy-a", 9, 4, 3),
    ]

    response = test_client.get("/api/insights/activity-summary")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["days"] == ACTIVITY_SUMMARY_DEFAULT_DAYS
    assert body["items"][0] == {
        "summary_date": "2026-09-10",
        "dimension": EVENT_TYPE_DIMENSION,
        "dimension_key": "recommendation.shown",
        "record_count": 42,
        "subject_count": 7,
        "candidate_set_count": None,
    }
    assert body["items"][1]["candidate_set_count"] == 3


def test_the_endpoint_reads_only_the_summary_table(test_client: TestClient, mock_pg_pool: MagicMock) -> None:
    """No request path touches activity.events or activity.impressions."""
    mock_pg_pool.connection().cursor().fetchall.return_value = []

    test_client.get("/api/insights/activity-summary")

    statement = _normalized(_executed(mock_pg_pool)[0][0])
    assert "FROM insights.activity_summary" in statement
    assert "activity.events" not in statement
    assert "activity.impressions" not in statement
    assert "ORDER BY summary_date DESC, dimension, dimension_key" in statement


def test_the_endpoint_bounds_the_read_by_the_requested_days(test_client: TestClient, mock_pg_pool: MagicMock) -> None:
    """The cutoff is the first of the requested whole days."""
    mock_pg_pool.connection().cursor().fetchall.return_value = []

    response = test_client.get("/api/insights/activity-summary", params={"days": 3})

    assert response.json()["days"] == 3
    expected = (datetime.now(UTC) - timedelta(days=2)).date()
    assert _executed(mock_pg_pool)[0][1] == (expected,)
    assert response.json()["since"] == expected.isoformat()


@pytest.mark.parametrize("days", [0, 91, -1])
def test_the_endpoint_rejects_an_out_of_range_span(test_client: TestClient, days: int) -> None:
    """The lookback is bounded, so one request cannot scan the whole table."""
    assert test_client.get("/api/insights/activity-summary", params={"days": days}).status_code == 422


def test_the_endpoint_is_unavailable_before_the_pool_is_up() -> None:
    """A request before startup finishes is a 503, not a crash."""
    import insights.insights as module

    previous = module._pool
    module._pool = None
    try:
        assert TestClient(module.app).get("/api/insights/activity-summary").status_code == 503
    finally:
        module._pool = previous


def test_the_endpoint_uses_cache_aside_within_one_generation(
    test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock
) -> None:
    """The generation read before the database read is the one written back to."""
    mock_pg_pool.connection().cursor().fetchall.return_value = [_summary_row()]

    response = test_client_with_cache.get("/api/insights/activity-summary")

    assert response.status_code == 200
    mock_cache.get.assert_awaited_once_with("insights:activity-summary:7", TEST_CACHE_GENERATION)
    assert mock_cache.set.await_args.args[0] == "insights:activity-summary:7"
    assert mock_cache.set.await_args.args[2] == TEST_CACHE_GENERATION


def test_a_cache_hit_skips_the_database(test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock) -> None:
    """A hit is served as-is and the summary table is not queried."""
    mock_cache.get.return_value = {"days": 7, "since": "2026-09-08", "items": [], "count": 0}

    response = test_client_with_cache.get("/api/insights/activity-summary")

    assert response.json()["since"] == "2026-09-08"
    assert _executed(mock_pg_pool) == []
    mock_cache.set.assert_not_awaited()


def test_the_cache_key_separates_spans(test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock) -> None:
    """Two lookbacks are two results, so one cannot be served for the other."""
    mock_pg_pool.connection().cursor().fetchall.return_value = []

    test_client_with_cache.get("/api/insights/activity-summary", params={"days": 3})
    test_client_with_cache.get("/api/insights/activity-summary", params={"days": 30})

    keys = [call.args[0] for call in mock_cache.get.await_args_list]
    assert keys == ["insights:activity-summary:3", "insights:activity-summary:30"]


def test_the_status_endpoint_reports_the_activity_summary(test_client: TestClient) -> None:
    """An operator sees whether the summary ran, alongside the other seven."""
    response = test_client.get("/api/insights/status")

    reported = [status["insight_type"] for status in response.json()["statuses"]]
    assert "activity_summary" in reported


# ── The model ───────────────────────────────────────────────────────────────


def test_the_model_defaults_the_candidate_set_count_to_null() -> None:
    """An event-type row is constructible without a candidate-set count."""
    item = ActivitySummaryItem(
        summary_date=date(2026, 9, 10),
        dimension=EVENT_TYPE_DIMENSION,
        dimension_key="recommendation.shown",
        record_count=1,
        subject_count=1,
    )

    assert item.candidate_set_count is None
    assert item.model_dump(mode="json")["summary_date"] == "2026-09-10"


# ── Table ownership ─────────────────────────────────────────────────────────


def test_the_summary_table_is_a_database_schema_follow_on() -> None:
    """`insights.activity_summary` must be declared in `database-schema`, not here.

    Every `insights.*` table this service reads and writes is created by that repository.
    This one is no different, and the filed follow-on is to add it to its
    `_INSIGHTS_STATEMENTS` alongside the seven that are already there:

        CREATE TABLE IF NOT EXISTS insights.activity_summary (
            summary_date        DATE NOT NULL,
            dimension           TEXT NOT NULL CHECK (dimension IN ('event_type', 'policy_id')),
            dimension_key       TEXT NOT NULL,
            record_count        BIGINT NOT NULL,
            subject_count       BIGINT NOT NULL,
            candidate_set_count BIGINT,
            computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (summary_date, dimension, dimension_key)
        )

    If this test ever fails because a `CREATE TABLE` appeared in this repository, the fix is
    to move it to `database-schema` rather than to relax the test.
    """
    for module in (computations, __import__("insights.insights", fromlist=["insights"])):
        source = inspect.getsource(module)
        body = source[source.index('"""', source.index('"""') + 3) :].upper()
        assert "CREATE TABLE" not in body, f"{module.__name__} declares DDL; insights tables belong to database-schema"
        assert "CREATE SCHEMA" not in body


def test_the_table_constant_matches_the_sql_the_module_issues() -> None:
    """The documented table name and the statements cannot drift apart."""
    source = inspect.getsource(computations)
    assert "INSERT INTO " + ACTIVITY_SUMMARY_TABLE in source
    assert "DELETE FROM " + ACTIVITY_SUMMARY_TABLE in source  # noqa: S608  # A source assertion, not a statement.
    assert ACTIVITY_SUMMARY_TABLE in inspect.getsource(__import__("insights.insights", fromlist=["insights"]))
