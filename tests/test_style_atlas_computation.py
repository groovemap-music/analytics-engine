"""The style_atlas computation: once per month, pinned to the previous month, and deterministic."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import insights.computations as computations
from insights.computations import atlas_month, compute_and_store_style_atlas, endpoint_timeout
from tests.style_atlas_fixtures import next_month, synthetic_month


STYLE_YEAR_COUNTS_PATH = "/api/internal/insights/style-year-counts"
OCTOBER = datetime(2026, 10, 23, 9, 30, tzinfo=UTC)
NOVEMBER = datetime(2026, 11, 2, 0, 5, tzinfo=UTC)


class _Atlas:
    """``insights.genre_atlas_sectors`` and ``insights.style_atlas_layout``, keyed like their primary keys."""

    def __init__(self) -> None:
        self.sectors: dict[tuple[date, str], tuple[Any, ...]] = {}
        self.nodes: dict[tuple[date, str, str, str], tuple[Any, ...]] = {}
        self.statements: list[str] = []
        self.transactions = 0
        # Rows a concurrent run commits between this run's first check and its write.
        self.concurrent_sectors: list[tuple[Any, ...]] = []

    def pool(self) -> MagicMock:
        atlas = self

        class Cursor:
            def __init__(self) -> None:
                self.result: list[tuple[Any, ...]] = []

            async def __aenter__(self) -> Cursor:
                return self

            async def __aexit__(self, *exc: object) -> bool:
                return False

            async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
                sql = re.sub(r"\s+", " ", sql).strip()
                atlas.statements.append(sql)
                if "pg_advisory_xact_lock" in sql:
                    for row in atlas.concurrent_sectors:
                        atlas.sectors[(row[0], row[1])] = row
                    self.result = [(None,)]
                elif sql.startswith("SELECT 1 FROM insights.genre_atlas_sectors"):
                    self.result = [(1,)] if any(month == params[0] for month, _genre in atlas.sectors) else []
                elif sql.startswith("SELECT month, genre, ring_order"):
                    earlier = [month for month, _genre in atlas.sectors if month < params[0]]
                    latest = max(earlier, default=None)
                    self.result = sorted((row for (month, _g), row in atlas.sectors.items() if month == latest), key=lambda row: row[2])
                elif sql.startswith("SELECT style, x, y FROM insights.style_atlas_layout"):
                    self.result = [
                        (row[2], row[4], row[5])
                        for (month, kind, _s, _g), row in sorted(atlas.nodes.items())
                        if month == params[0] and kind == "style"
                    ]
                else:
                    raise AssertionError(f"unexpected SQL: {sql}")

            async def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
                sql = re.sub(r"\s+", " ", sql).strip()
                atlas.statements.append(sql)
                table = atlas.sectors if "genre_atlas_sectors" in sql else atlas.nodes
                for row in rows:
                    key = (row[0], row[1]) if table is atlas.sectors else (row[0], row[1], row[2], row[3])
                    assert key not in table, f"duplicate key {key}"
                    table[key] = row  # type: ignore[index]

            async def fetchone(self) -> tuple[Any, ...] | None:
                return self.result[0] if self.result else None

            async def fetchall(self) -> list[tuple[Any, ...]]:
                return self.result

        class Transaction:
            async def __aenter__(self) -> None:
                atlas.transactions += 1

            async def __aexit__(self, *exc: object) -> bool:
                return False

        class Connection:
            async def __aenter__(self) -> Connection:
                return self

            async def __aexit__(self, *exc: object) -> bool:
                return False

            async def set_autocommit(self, value: bool) -> None:
                assert value is False

            def transaction(self) -> Transaction:
                return Transaction()

            def cursor(self) -> Cursor:
                return Cursor()

        pool = MagicMock()
        pool.connection = MagicMock(side_effect=Connection)
        return pool

    def month_rows(self, month: date) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        return (
            sorted(row for (m, _g), row in self.sectors.items() if m == month),
            sorted(row for (m, *_rest), row in self.nodes.items() if m == month),
        )


def _client(body: dict[str, Any]) -> AsyncMock:
    response = MagicMock()
    response.json = MagicMock(return_value=body)
    response.raise_for_status = MagicMock()
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    return client


@pytest.fixture(autouse=True)
def _no_computation_log() -> Any:
    with patch.object(computations, "_log_computation", new=AsyncMock()) as log:
        yield log


class TestTheMonth:
    def test_is_the_first_of_the_utc_month(self) -> None:
        assert atlas_month(OCTOBER) == date(2026, 10, 1)
        assert atlas_month(datetime(2026, 10, 31, 23, 30, tzinfo=UTC)) == date(2026, 10, 1)


class TestTheFirstMonth:
    @pytest.mark.asyncio
    async def test_writes_every_sector_and_node_in_one_locked_transaction(self) -> None:
        month = synthetic_month()
        atlas = _Atlas()

        written = await compute_and_store_style_atlas(_client(month.producer_body()), atlas.pool(), now=OCTOBER)

        sectors, nodes = atlas.month_rows(date(2026, 10, 1))
        assert len(sectors) == 16
        assert {row[1] for row in nodes} == {"style", "ghost"}
        assert {row[2] for row in nodes if row[1] == "style"} == set(month.shares)
        assert written == len(sectors) + len(nodes)
        assert atlas.transactions == 1
        lock = atlas.statements.index("SELECT pg_advisory_xact_lock(hashtext('insights.style_atlas'))")
        assert atlas.statements[lock + 1].startswith("SELECT 1 FROM insights.genre_atlas_sectors")
        assert atlas.statements[lock + 2].startswith("INSERT INTO insights.genre_atlas_sectors")
        assert atlas.statements[lock + 3].startswith("INSERT INTO insights.style_atlas_layout")

    @pytest.mark.asyncio
    async def test_reads_the_producer_under_its_own_budget_and_logs_completed(self, _no_computation_log: AsyncMock) -> None:
        client = _client(synthetic_month().producer_body())

        await compute_and_store_style_atlas(client, _Atlas().pool(), now=OCTOBER)

        assert client.get.await_args.args == (STYLE_YEAR_COUNTS_PATH,)
        assert client.get.await_args.kwargs["timeout"] == endpoint_timeout(STYLE_YEAR_COUNTS_PATH)
        assert _no_computation_log.await_args.args[1:3] == ("style_atlas", "completed")

    @pytest.mark.asyncio
    async def test_is_deterministic_for_the_same_inputs(self) -> None:
        body = synthetic_month().producer_body()
        first, second = _Atlas(), _Atlas()

        await compute_and_store_style_atlas(_client(body), first.pool(), now=OCTOBER)
        await compute_and_store_style_atlas(_client(body), second.pool(), now=OCTOBER)

        assert first.month_rows(date(2026, 10, 1)) == second.month_rows(date(2026, 10, 1))


class TestASecondRunInTheSameMonth:
    @pytest.mark.asyncio
    async def test_is_a_no_op_that_never_reads_the_producer(self, _no_computation_log: AsyncMock) -> None:
        atlas = _Atlas()
        early = synthetic_month()
        await compute_and_store_style_atlas(_client(early.producer_body()), atlas.pool(), now=OCTOBER)
        before = atlas.month_rows(date(2026, 10, 1))

        client = _client(next_month(early).producer_body())
        written = await compute_and_store_style_atlas(client, atlas.pool(), now=datetime(2026, 10, 30, tzinfo=UTC))

        assert written == 0
        client.get.assert_not_awaited()
        assert atlas.month_rows(date(2026, 10, 1)) == before
        assert _no_computation_log.await_args.args[1:3] == ("style_atlas", "completed")

    @pytest.mark.asyncio
    async def test_a_run_that_loses_a_race_to_a_concurrent_writer_leaves_its_rows(self) -> None:
        atlas = _Atlas()
        concurrent = (date(2026, 10, 1), "Rock", 0, 0.0, 6.28, 0.65, 0.0)
        atlas.concurrent_sectors = [concurrent]

        written = await compute_and_store_style_atlas(_client(synthetic_month().producer_body()), atlas.pool(), now=OCTOBER)

        assert written == 0
        assert atlas.month_rows(date(2026, 10, 1)) == ([concurrent], [])
        assert not any(statement.startswith("INSERT") for statement in atlas.statements)


class TestTheNextMonth:
    @pytest.mark.asyncio
    async def test_pins_ring_order_and_frames_to_the_previous_month(self) -> None:
        atlas = _Atlas()
        early = synthetic_month()
        await compute_and_store_style_atlas(_client(early.producer_body()), atlas.pool(), now=OCTOBER)

        await compute_and_store_style_atlas(_client(next_month(early).producer_body()), atlas.pool(), now=NOVEMBER)

        october, _ = atlas.month_rows(date(2026, 10, 1))
        november, nodes = atlas.month_rows(date(2026, 11, 1))
        assert [row[1:5] for row in november] == [row[1:5] for row in october]
        assert nodes

    @pytest.mark.asyncio
    async def test_pins_to_the_latest_earlier_month_when_a_month_was_skipped(self) -> None:
        atlas = _Atlas()
        early = synthetic_month()
        await compute_and_store_style_atlas(_client(early.producer_body()), atlas.pool(), now=OCTOBER)
        october, _ = atlas.month_rows(date(2026, 10, 1))

        await compute_and_store_style_atlas(_client(early.producer_body()), atlas.pool(), now=datetime(2027, 1, 5, tzinfo=UTC))

        january, _ = atlas.month_rows(date(2027, 1, 1))
        assert [row[1:5] for row in january] == [row[1:5] for row in october]


class TestTheEmptyResultRule:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [{"style_years": [], "style_parents": []}, {"style_years": [{"style": "House", "year": 1990, "release_count": 30, "master_count": 3}]}],
        ids=["both-empty", "no-shares"],
    )
    async def test_an_incomplete_producer_result_lays_out_nothing(self, body: dict[str, Any]) -> None:
        atlas = _Atlas()

        written = await compute_and_store_style_atlas(_client(body), atlas.pool(), now=OCTOBER)

        assert written == 0
        assert atlas.sectors == {} and atlas.nodes == {}


class TestTheFailurePath:
    @pytest.mark.asyncio
    async def test_a_producer_error_is_logged_failed_and_reraised_without_writing(self, _no_computation_log: AsyncMock) -> None:
        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.ReadTimeout(""))
        atlas = _Atlas()

        with patch.object(computations.logger, "error") as log_error, pytest.raises(httpx.ReadTimeout):
            await compute_and_store_style_atlas(client, atlas.pool(), now=OCTOBER)

        assert log_error.call_args.kwargs["error"] == "ReadTimeout"
        assert _no_computation_log.await_args.args[1:3] == ("style_atlas", "failed")
        assert _no_computation_log.await_args.kwargs["error_message"] == "ReadTimeout"
        assert atlas.sectors == {}


class TestTheSchedulerSequence:
    @pytest.mark.asyncio
    async def test_runs_right_after_style_year_counts(self) -> None:
        order: list[str] = []

        def record(name: str) -> Any:
            async def run(*_args: object, **_kwargs: object) -> int:
                order.append(name)
                return 1

            return run

        names = [name for name in dir(computations) if name.startswith("compute_and_store_")]
        with patch.multiple(computations, **{name: record(name.removeprefix("compute_and_store_")) for name in names}):
            await computations.run_all_computations(AsyncMock(), MagicMock())

        assert order.index("style_atlas") == order.index("style_year_counts") + 1
