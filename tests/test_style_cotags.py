"""Tests for the style_cotags computation: the write, the ranks, the empty-result rule, and the failure path."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import insights.computations as computations
from insights.computations import STYLE_COTAG_MAX_RANK, _style_cotag_rows, compute_and_store_style_cotags, endpoint_timeout


STYLE_COTAGS_PATH = "/api/internal/insights/style-cotags"


def _make_pool() -> MagicMock:
    """A pool whose single cursor records every execute and executemany, in call order."""
    statements: list[str] = []

    async def record(sql: str, *_args: object) -> None:
        statements.append(re.sub(r"\s+", " ", sql).strip())

    cursor = AsyncMock()
    cursor.statements = statements
    cursor.execute = AsyncMock(side_effect=record)
    cursor.executemany = AsyncMock(side_effect=record)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)
    transaction = AsyncMock()
    transaction.__aenter__ = AsyncMock(return_value=None)
    transaction.__aexit__ = AsyncMock(return_value=None)
    conn = AsyncMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.transaction = MagicMock(return_value=transaction)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool


def _cursor(pool: MagicMock) -> AsyncMock:
    return pool.connection().cursor()


def _statements(pool: MagicMock) -> list[str]:
    """Normalised SQL of every statement on the co-tag table, in call order (the computation_log write excluded)."""
    statements: list[str] = _cursor(pool).statements
    return [statement for statement in statements if "computation_log" not in statement]


def _written(pool: MagicMock) -> list[Any]:
    """Parameter rows bulk-inserted into ``insights.style_cotags``."""
    rows: list[Any] = []
    for call in _cursor(pool).executemany.await_args_list:
        rows.extend(call.args[1])
    return rows


def _client(body: dict[str, Any]) -> AsyncMock:
    response = MagicMock()
    response.json = MagicMock(return_value=body)
    response.raise_for_status = MagicMock()
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    return client


def _cotag(style: str = "House", other: str = "Techno", release_count: int = 10) -> dict[str, Any]:
    return {"style": style, "other": other, "release_count": release_count}


class TestTheWrite:
    @pytest.mark.asyncio
    async def test_writes_each_pair_with_its_rank_and_returns_the_row_count(self) -> None:
        client = _client(
            {
                "cotags": [
                    _cotag("House", "Techno", 30),
                    _cotag("House", "Disco", 20),
                    _cotag("Techno", "House", 30),
                ]
            }
        )
        pool = _make_pool()

        written = await compute_and_store_style_cotags(client, pool)

        assert written == 3
        assert _written(pool) == [
            ("House", "Techno", 30, 1),
            ("House", "Disco", 20, 2),
            ("Techno", "House", 30, 1),
        ]

    @pytest.mark.asyncio
    async def test_replaces_the_table_inside_one_transaction_before_inserting(self) -> None:
        pool = _make_pool()

        await compute_and_store_style_cotags(_client({"cotags": [_cotag()]}), pool)

        pool.connection().transaction.assert_called_once_with()
        statements = _statements(pool)
        assert statements[0] == "DELETE FROM insights.style_cotags"
        assert statements[1].startswith("INSERT INTO insights.style_cotags (style, other, release_count, rank)")
        assert len(statements) == 2

    @pytest.mark.asyncio
    async def test_reads_the_endpoint_under_its_own_budget_and_logs_completed(self) -> None:
        client = _client({"cotags": [_cotag(), _cotag(other="Disco")]})

        with patch.object(computations, "_log_computation", new=AsyncMock()) as log:
            await compute_and_store_style_cotags(client, _make_pool())

        assert client.get.await_args.args == (STYLE_COTAGS_PATH,)
        assert client.get.await_args.kwargs["timeout"] == endpoint_timeout(STYLE_COTAGS_PATH)
        assert log.await_args.args[1:3] == ("style_cotags", "completed")
        assert log.await_args.args[4] == 2


class TestTheReadBudget:
    def test_is_three_times_the_producers_published_processing_time(self) -> None:
        assert endpoint_timeout(STYLE_COTAGS_PATH).read == 900.0

    def test_is_an_explicit_endpoint_entry(self) -> None:
        assert STYLE_COTAGS_PATH in computations.ENDPOINT_READ_TIMEOUTS


class TestTheRanks:
    def test_rank_is_the_position_within_the_style_in_the_producers_order(self) -> None:
        rows = _style_cotag_rows(
            [
                _cotag("House", "Techno", 50),
                _cotag("House", "Disco", 40),
                _cotag("House", "Garage", 40),
                _cotag("Dub", "Reggae", 7),
            ]
        )

        assert [(style, other, rank) for style, other, _count, rank in rows] == [
            ("House", "Techno", 1),
            ("House", "Disco", 2),
            ("House", "Garage", 3),
            ("Dub", "Reggae", 1),
        ]

    def test_ranks_restart_for_a_style_whose_rows_are_not_contiguous(self) -> None:
        rows = _style_cotag_rows([_cotag("House", "A"), _cotag("Dub", "B"), _cotag("House", "C"), _cotag("Dub", "D")])

        assert {(style, other): rank for style, other, _count, rank in rows} == {
            ("House", "A"): 1,
            ("House", "C"): 2,
            ("Dub", "B"): 1,
            ("Dub", "D"): 2,
        }

    def test_a_full_style_ranks_one_to_twenty_five(self) -> None:
        rows = _style_cotag_rows([_cotag("House", f"S{i}", 100 - i) for i in range(25)])

        assert [row[3] for row in rows] == list(range(1, 26))

    def test_a_style_over_the_cap_keeps_its_first_twenty_five(self) -> None:
        rows = _style_cotag_rows([_cotag("House", f"S{i}", 100 - i) for i in range(30)])

        assert STYLE_COTAG_MAX_RANK == 25
        assert [row[1] for row in rows] == [f"S{i}" for i in range(25)]
        assert max(row[3] for row in rows) == 25


class TestTheEmptyResultRule:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [{"cotags": []}, {}], ids=["empty-list", "absent"])
    async def test_an_empty_producer_result_leaves_the_previous_snapshot_intact(self, body: dict[str, Any]) -> None:
        pool = _make_pool()

        written = await compute_and_store_style_cotags(_client(body), pool)

        assert written == 0
        assert _statements(pool) == []
        _cursor(pool).executemany.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_empty_result_is_logged_completed_with_no_rows(self) -> None:
        with patch.object(computations, "_log_computation", new=AsyncMock()) as log:
            await compute_and_store_style_cotags(_client({"cotags": []}), _make_pool())

        assert log.await_args.args[1:3] == ("style_cotags", "completed")
        assert log.await_args.args[4] == 0


class TestTheFailurePath:
    @pytest.mark.asyncio
    async def test_an_api_error_is_logged_failed_and_reraised_without_touching_the_table(self) -> None:
        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.ReadTimeout(""))
        pool = _make_pool()

        with (
            patch.object(computations, "_log_computation", new=AsyncMock()) as log,
            pytest.raises(httpx.ReadTimeout),
        ):
            await compute_and_store_style_cotags(client, pool)

        assert log.await_args.args[1:3] == ("style_cotags", "failed")
        assert log.await_args.kwargs["error_message"] == "ReadTimeout"
        assert _statements(pool) == []

    @pytest.mark.asyncio
    async def test_an_http_error_status_is_a_failure(self) -> None:
        client = _client({})
        client.get.return_value.raise_for_status = MagicMock(side_effect=RuntimeError("503"))

        with (
            patch.object(computations, "_log_computation", new=AsyncMock()) as log,
            pytest.raises(RuntimeError, match="503"),
        ):
            await compute_and_store_style_cotags(client, _make_pool())

        assert log.await_args.args[2] == "failed"

    @pytest.mark.asyncio
    async def test_a_storage_error_is_logged_failed_and_reraised_through_the_transaction(self) -> None:
        """The error must propagate through ``conn.transaction()`` so PostgreSQL rolls the delete back."""
        pool = _make_pool()
        _cursor(pool).executemany = AsyncMock(side_effect=RuntimeError("DB error"))
        transaction = pool.connection().transaction()

        with (
            patch.object(computations, "_log_computation", new=AsyncMock()) as log,
            pytest.raises(RuntimeError, match="DB error"),
        ):
            await compute_and_store_style_cotags(_client({"cotags": [_cotag()]}), pool)

        assert log.await_args.args[1:3] == ("style_cotags", "failed")
        assert log.await_args.kwargs["error_message"] == "RuntimeError: DB error"
        assert transaction.__aexit__.await_args.args[0] is not None
