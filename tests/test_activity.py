"""Tests for the consent-aware activity read path.

The point these tests defend is ADR 0010's double check: a row is readable only when its
write-time ``consent_purposes`` snapshot carries the purpose *and* the subject's user still
holds an unrevoked grant for it. Three of the four combinations must yield nothing, and
each must yield nothing for its own reason, so the four-way table below is written out
rather than collapsed.
"""

from __future__ import annotations

import inspect
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest

from insights import activity
from insights.activity import (
    MODEL_TRAINING,
    PRODUCT_ANALYTICS,
    ConsentPurposeError,
    read_events,
    read_impressions,
    training_eligible_subjects,
)


if TYPE_CHECKING:
    from unittest.mock import MagicMock


SINCE = datetime(2026, 8, 1, tzinfo=UTC)
UNTIL = datetime(2026, 9, 1, tzinfo=UTC)

SUBJECT = UUID("11111111-1111-1111-1111-111111111111")
OTHER_SUBJECT = UUID("22222222-2222-2222-2222-222222222222")


def _connection(pool: MagicMock) -> Any:
    """Return the single mock connection the shared pool fixture hands out."""
    return pool.connection()


def _cursor(pool: MagicMock) -> Any:
    """Return the single mock cursor every connection in the fixture shares."""
    return _connection(pool).cursor()


def _statements(pool: MagicMock) -> list[str]:
    """Return every SQL statement executed, in order."""
    return [call.args[0] for call in _cursor(pool).execute.await_args_list]


def _params(pool: MagicMock) -> list[Any]:
    """Return every parameter tuple executed, in order."""
    return [call.args[1] for call in _cursor(pool).execute.await_args_list]


def _normalized(statement: str) -> str:
    """Collapse the whitespace of a statement so shape assertions stay readable."""
    return re.sub(r"\s+", " ", statement).strip()


def _event_row(
    *,
    event_id: UUID | None = None,
    event_type: str = "recommendation.shown",
    subject_id: UUID = SUBJECT,
    occurred_at: datetime | None = None,
    purposes: tuple[str, ...] = (MODEL_TRAINING,),
) -> tuple[Any, ...]:
    """Build one ``activity.events`` row in the column order the reader selects."""
    return (
        event_id or uuid4(),
        event_type,
        1,
        subject_id,
        None,
        occurred_at or SINCE,
        (occurred_at or SINCE) + timedelta(seconds=1),
        "catalog-api",
        list(purposes),
        None,
        None,
        "idem-1",
        {"item_id": "abc"},
    )


def _impression_row(
    *,
    impression_id: UUID | None = None,
    subject_id: UUID = SUBJECT,
    policy_id: str = "policy-a",
    candidate_set_id: UUID | None = None,
    occurred_at: datetime | None = None,
    purposes: tuple[str, ...] = (MODEL_TRAINING,),
) -> tuple[Any, ...]:
    """Build one ``activity.impressions`` row in the column order the reader selects."""
    return (
        impression_id or uuid4(),
        subject_id,
        "home_feed",
        policy_id,
        candidate_set_id or uuid4(),
        1,
        uuid4(),
        0.9,
        0.25,
        uuid4(),
        occurred_at or SINCE,
        (occurred_at or SINCE) + timedelta(seconds=1),
        list(purposes),
    )


def _grant_rows(*subjects: UUID) -> list[tuple[UUID]]:
    """Build the eligible-subject result the grant query returns."""
    return [(subject,) for subject in subjects]


async def _drain(iterator: Any) -> list[Any]:
    """Collect an async iterator into a list."""
    return [item async for item in iterator]


# ── The consent filter itself ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_eligible_subjects_joins_grants_and_excludes_revocations(mock_pg_pool: MagicMock) -> None:
    """The filter joins the two tables, checks revocation, and returns subjects only."""
    _cursor(mock_pg_pool).fetchall.return_value = _grant_rows(SUBJECT, OTHER_SUBJECT)

    async with _connection(mock_pg_pool) as conn:
        eligible = await training_eligible_subjects(conn, MODEL_TRAINING)

    assert eligible == {SUBJECT, OTHER_SUBJECT}
    statement = _normalized(_statements(mock_pg_pool)[0])
    assert "FROM activity.user_subjects AS subjects" in statement
    assert "JOIN activity.consent_grants AS grants ON grants.user_id = subjects.user_id" in statement
    assert "grants.purpose = %s" in statement
    assert "grants.revoked_at IS NULL" in statement
    assert _params(mock_pg_pool)[0] == (MODEL_TRAINING,)


@pytest.mark.asyncio
async def test_eligible_subjects_never_selects_a_user_id(mock_pg_pool: MagicMock) -> None:
    """The user id is what the join is on, never what the read returns."""
    _cursor(mock_pg_pool).fetchall.return_value = _grant_rows(SUBJECT)

    async with _connection(mock_pg_pool) as conn:
        await training_eligible_subjects(conn, MODEL_TRAINING)

    select_clause = _normalized(_statements(mock_pg_pool)[0]).split(" FROM ")[0]
    assert select_clause == "SELECT subjects.subject_id"
    assert "user_id" not in select_clause


@pytest.mark.asyncio
async def test_eligible_subjects_defaults_to_model_training(mock_pg_pool: MagicMock) -> None:
    """The default purpose is the training one, which is the strictest caller."""
    _cursor(mock_pg_pool).fetchall.return_value = []

    async with _connection(mock_pg_pool) as conn:
        assert await training_eligible_subjects(conn) == set()

    assert _params(mock_pg_pool)[0] == (MODEL_TRAINING,)


@pytest.mark.asyncio
async def test_eligible_subjects_rejects_an_unpublished_purpose(mock_pg_pool: MagicMock) -> None:
    """A typo must fail loudly, not read as an absence of consented data."""
    async with _connection(mock_pg_pool) as conn:
        with pytest.raises(ConsentPurposeError, match="unknown consent purpose"):
            await training_eligible_subjects(conn, "marketing")

    _cursor(mock_pg_pool).execute.assert_not_awaited()


# ── The four consent combinations ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_yes_grant_yes_is_included(mock_pg_pool: MagicMock) -> None:
    """Both checks pass, so the row is readable."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), [_event_row(purposes=(MODEL_TRAINING,))]]

    async with _connection(mock_pg_pool) as conn:
        events = await _drain(read_events(conn, since=SINCE, until=UNTIL))

    assert len(events) == 1
    assert events[0].subject_id == SUBJECT
    assert events[0].consent_purposes == (MODEL_TRAINING,)


@pytest.mark.asyncio
async def test_snapshot_yes_grant_revoked_is_excluded(mock_pg_pool: MagicMock) -> None:
    """A revocation is honoured going forward for data lawfully collected before it.

    The grant query returns nothing, so the reader never issues the row query at all: the
    revocation is enforced before any activity row is touched.
    """
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [[], [_event_row(purposes=(MODEL_TRAINING,))]]

    async with _connection(mock_pg_pool) as conn:
        events = await _drain(read_events(conn, since=SINCE, until=UNTIL))

    assert events == []
    assert len(_statements(mock_pg_pool)) == 1
    assert "activity.events" not in _statements(mock_pg_pool)[0]


@pytest.mark.asyncio
async def test_snapshot_no_grant_yes_is_excluded(mock_pg_pool: MagicMock) -> None:
    """A row written without the purpose stays out even once the user grants it.

    The exclusion is the database's, not the reader's: the snapshot predicate is a bound
    conjunct of the row query, so a row lacking the purpose never comes back.
    """
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), []]

    async with _connection(mock_pg_pool) as conn:
        events = await _drain(read_events(conn, since=SINCE, until=UNTIL))

    assert events == []
    row_statement = _normalized(_statements(mock_pg_pool)[1])
    assert "%s = ANY(consent_purposes)" in row_statement
    assert MODEL_TRAINING in _params(mock_pg_pool)[1]


@pytest.mark.asyncio
async def test_neither_snapshot_nor_grant_is_excluded(mock_pg_pool: MagicMock) -> None:
    """With no grant and no snapshot, nothing is read and nothing is queried."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [[], []]

    async with _connection(mock_pg_pool) as conn:
        events = await _drain(read_events(conn, since=SINCE, until=UNTIL))

    assert events == []
    assert len(_statements(mock_pg_pool)) == 1


@pytest.mark.asyncio
async def test_impressions_apply_the_same_four_way_semantics(mock_pg_pool: MagicMock) -> None:
    """The impression read carries both checks, in the same order, over its own table."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), [_impression_row()]]

    async with _connection(mock_pg_pool) as conn:
        impressions = await _drain(read_impressions(conn, since=SINCE, until=UNTIL))

    assert len(impressions) == 1
    assert impressions[0].policy_id == "policy-a"
    row_statement = _normalized(_statements(mock_pg_pool)[1])
    assert "FROM activity.impressions" in row_statement
    assert "%s = ANY(consent_purposes)" in row_statement
    assert "subject_id = ANY(%s)" in row_statement


@pytest.mark.asyncio
async def test_impressions_stop_at_a_revoked_grant(mock_pg_pool: MagicMock) -> None:
    """No eligible subject means the impression table is never touched."""
    _cursor(mock_pg_pool).fetchall.side_effect = [[], [_impression_row()]]

    async with _connection(mock_pg_pool) as conn:
        assert await _drain(read_impressions(conn, since=SINCE, until=UNTIL)) == []

    assert len(_statements(mock_pg_pool)) == 1


# ── SQL shape ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_read_applies_both_checks_bounded_by_occurred_at(mock_pg_pool: MagicMock) -> None:
    """The window is half-open and comes first, so partition pruning applies."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), []]

    async with _connection(mock_pg_pool) as conn:
        await _drain(read_events(conn, since=SINCE, until=UNTIL))

    statement = _normalized(_statements(mock_pg_pool)[1])
    assert "FROM activity.events WHERE occurred_at >= %s AND occurred_at < %s" in statement
    assert statement.index("occurred_at >= %s") < statement.index("= ANY(consent_purposes)")
    assert statement.index("= ANY(consent_purposes)") < statement.index("subject_id = ANY(%s)")
    assert "ORDER BY occurred_at, event_id" in statement
    assert statement.endswith("LIMIT %s")

    params = _params(mock_pg_pool)[1]
    assert params[0] is SINCE
    assert params[1] is UNTIL
    assert params[2] == MODEL_TRAINING
    assert params[3] == [SUBJECT]


@pytest.mark.asyncio
async def test_event_read_passes_the_eligible_set_as_the_subject_bound(mock_pg_pool: MagicMock) -> None:
    """The array bound to the re-check is exactly what the grant query returned."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(OTHER_SUBJECT, SUBJECT), []]

    async with _connection(mock_pg_pool) as conn:
        await _drain(read_events(conn, since=SINCE, until=UNTIL))

    assert _params(mock_pg_pool)[1][3] == sorted([SUBJECT, OTHER_SUBJECT])


@pytest.mark.asyncio
async def test_event_types_narrow_the_read(mock_pg_pool: MagicMock) -> None:
    """An event-type restriction is an extra bound conjunct, not a replacement."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), []]

    async with _connection(mock_pg_pool) as conn:
        await _drain(read_events(conn, since=SINCE, until=UNTIL, event_types=["recommendation.shown"]))

    statement = _normalized(_statements(mock_pg_pool)[1])
    assert "event_type = ANY(%s)" in statement
    assert "subject_id = ANY(%s)" in statement
    assert _params(mock_pg_pool)[1][4] == ["recommendation.shown"]


@pytest.mark.asyncio
async def test_policy_ids_narrow_the_impression_read(mock_pg_pool: MagicMock) -> None:
    """A policy restriction is an extra bound conjunct on the impression read."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), []]

    async with _connection(mock_pg_pool) as conn:
        await _drain(read_impressions(conn, since=SINCE, until=UNTIL, policy_ids=["policy-a"]))

    assert "policy_id = ANY(%s)" in _normalized(_statements(mock_pg_pool)[1])
    assert _params(mock_pg_pool)[1][4] == ["policy-a"]


@pytest.mark.asyncio
async def test_several_purposes_repeat_the_snapshot_check_and_intersect_the_grants(mock_pg_pool: MagicMock) -> None:
    """Asking for two purposes asks for rows both are permitted for, not either."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT, OTHER_SUBJECT), _grant_rows(SUBJECT), []]

    async with _connection(mock_pg_pool) as conn:
        await _drain(read_events(conn, since=SINCE, until=UNTIL, purposes=(MODEL_TRAINING, PRODUCT_ANALYTICS)))

    statement = _normalized(_statements(mock_pg_pool)[2])
    assert statement.count("= ANY(consent_purposes)") == 2
    params = _params(mock_pg_pool)[2]
    assert params[2] == MODEL_TRAINING
    assert params[3] == PRODUCT_ANALYTICS
    # The intersection, not the union: OTHER_SUBJECT consents to only one of the two.
    assert params[4] == [SUBJECT]


# ── Caller-supplied subjects can only narrow ────────────────────────────────


@pytest.mark.asyncio
async def test_subjects_argument_intersects_rather_than_replaces(mock_pg_pool: MagicMock) -> None:
    """A caller can restrict a read to subjects it cares about, and no further."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT, OTHER_SUBJECT), []]

    async with _connection(mock_pg_pool) as conn:
        await _drain(read_events(conn, since=SINCE, until=UNTIL, subjects=[SUBJECT]))

    assert _params(mock_pg_pool)[1][3] == [SUBJECT]


@pytest.mark.asyncio
async def test_subjects_argument_cannot_reach_an_ungranted_subject(mock_pg_pool: MagicMock) -> None:
    """Naming a subject who has not consented reads nothing, not that subject's rows."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), [_event_row(subject_id=OTHER_SUBJECT)]]

    async with _connection(mock_pg_pool) as conn:
        events = await _drain(read_events(conn, since=SINCE, until=UNTIL, subjects=[OTHER_SUBJECT]))

    assert events == []
    assert len(_statements(mock_pg_pool)) == 1


# ── Keyset paging ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_paging_resumes_after_the_last_key_of_the_previous_page(mock_pg_pool: MagicMock) -> None:
    """A full page is followed by a query resuming after its last (occurred_at, event_id)."""
    last_of_page = datetime(2026, 8, 5, tzinfo=UTC)
    last_id = UUID("33333333-3333-3333-3333-333333333333")
    page_one = [
        _event_row(occurred_at=SINCE),
        _event_row(event_id=last_id, occurred_at=last_of_page),
    ]
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), page_one, [_event_row(occurred_at=datetime(2026, 8, 6, tzinfo=UTC))]]

    async with _connection(mock_pg_pool) as conn:
        events = await _drain(read_events(conn, since=SINCE, until=UNTIL, batch_size=2))

    assert len(events) == 3
    second = _normalized(_statements(mock_pg_pool)[2])
    assert "(occurred_at, event_id) > (%s, %s)" in second
    assert _params(mock_pg_pool)[2][-3:] == (last_of_page, last_id, 2)


@pytest.mark.asyncio
async def test_a_short_page_ends_the_event_read(mock_pg_pool: MagicMock) -> None:
    """Fewer rows than the batch size means the window is exhausted; stop querying."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), [_event_row()]]

    async with _connection(mock_pg_pool) as conn:
        assert len(await _drain(read_events(conn, since=SINCE, until=UNTIL, batch_size=10))) == 1

    assert len(_statements(mock_pg_pool)) == 2


@pytest.mark.asyncio
async def test_an_exactly_full_last_page_is_confirmed_empty(mock_pg_pool: MagicMock) -> None:
    """A page that fills exactly needs one more query to learn there is no more."""
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), [_event_row()], []]

    async with _connection(mock_pg_pool) as conn:
        assert len(await _drain(read_events(conn, since=SINCE, until=UNTIL, batch_size=1))) == 1

    assert len(_statements(mock_pg_pool)) == 3


@pytest.mark.asyncio
async def test_impression_paging_resumes_on_its_own_key(mock_pg_pool: MagicMock) -> None:
    """The impression cursor is (occurred_at, impression_id), read from its own columns."""
    last_of_page = datetime(2026, 8, 9, tzinfo=UTC)
    last_id = UUID("44444444-4444-4444-4444-444444444444")
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [
        _grant_rows(SUBJECT),
        [_impression_row(impression_id=last_id, occurred_at=last_of_page)],
        [],
    ]

    async with _connection(mock_pg_pool) as conn:
        await _drain(read_impressions(conn, since=SINCE, until=UNTIL, batch_size=1))

    second = _normalized(_statements(mock_pg_pool)[2])
    assert "(occurred_at, impression_id) > (%s, %s)" in second
    assert _params(mock_pg_pool)[2][-3:] == (last_of_page, last_id, 1)


# ── Row mapping ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_rows_map_onto_the_runtime_envelope(mock_pg_pool: MagicMock) -> None:
    """Rows come back as the shared ``common.events`` model, not as tuples."""
    event_id = uuid4()
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), [_event_row(event_id=event_id, event_type="recommendation.opened")]]

    async with _connection(mock_pg_pool) as conn:
        (event,) = await _drain(read_events(conn, since=SINCE, until=UNTIL))

    assert event.event_id == event_id
    assert event.event_type == "recommendation.opened"
    assert event.producer == "catalog-api"
    assert event.payload == {"item_id": "abc"}
    assert event.occurred_at == SINCE


@pytest.mark.asyncio
async def test_impression_rows_map_onto_the_runtime_envelope(mock_pg_pool: MagicMock) -> None:
    """The four ranking-decision fields survive the round trip unchanged."""
    candidate_set_id = uuid4()
    cursor = _cursor(mock_pg_pool)
    cursor.fetchall.side_effect = [_grant_rows(SUBJECT), [_impression_row(candidate_set_id=candidate_set_id)]]

    async with _connection(mock_pg_pool) as conn:
        (impression,) = await _drain(read_impressions(conn, since=SINCE, until=UNTIL))

    assert impression.candidate_set_id == candidate_set_id
    assert impression.position == 1
    assert impression.propensity == 0.25
    assert impression.surface == "home_feed"


# ── Argument validation ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", [read_events, read_impressions])
async def test_reads_reject_an_unpublished_purpose(mock_pg_pool: MagicMock, reader: Any) -> None:
    """Neither reader accepts a purpose outside the published vocabulary."""
    async with _connection(mock_pg_pool) as conn:
        with pytest.raises(ConsentPurposeError, match="unknown consent purpose"):
            await _drain(reader(conn, since=SINCE, until=UNTIL, purposes=("everything",)))


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", [read_events, read_impressions])
async def test_reads_reject_an_empty_purpose_list(mock_pg_pool: MagicMock, reader: Any) -> None:
    """There is no unfiltered read: asking for no purpose is an error, not a wildcard."""
    async with _connection(mock_pg_pool) as conn:
        with pytest.raises(ConsentPurposeError, match="at least one consent purpose"):
            await _drain(reader(conn, since=SINCE, until=UNTIL, purposes=()))


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", [read_events, read_impressions])
async def test_reads_reject_a_naive_window(mock_pg_pool: MagicMock, reader: Any) -> None:
    """``occurred_at`` is TIMESTAMPTZ, so a naive bound would compare against the server zone."""
    async with _connection(mock_pg_pool) as conn:
        with pytest.raises(ValueError, match="timezone-aware"):
            await _drain(reader(conn, since=SINCE.replace(tzinfo=None), until=UNTIL))


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", [read_events, read_impressions])
async def test_reads_reject_an_empty_window(mock_pg_pool: MagicMock, reader: Any) -> None:
    """An inverted or empty window is a caller mistake, not an empty result."""
    async with _connection(mock_pg_pool) as conn:
        with pytest.raises(ValueError, match="since must be before until"):
            await _drain(reader(conn, since=UNTIL, until=SINCE))


# ── Module-level invariants ─────────────────────────────────────────────────


def test_the_module_issues_no_writes() -> None:
    """The read path is read-only by construction, and stays that way."""
    source = inspect.getsource(activity).upper()
    body = source[source.index('"""', source.index('"""') + 3) :]
    for verb in ("INSERT INTO", "UPDATE ", "DELETE FROM", "TRUNCATE", "COPY ", "MERGE INTO"):
        assert verb not in body, f"the activity read path must issue no {verb.strip()}"


def test_the_module_never_selects_a_user_id() -> None:
    """Only the pseudonymous subject leaves this module."""
    source = inspect.getsource(activity)
    for statement in re.findall(r"SELECT .*?FROM", source, flags=re.DOTALL):
        assert "user_id" not in statement, f"a read selected a user id: {_normalized(statement)}"
