"""Tests for the style_year_counts computation: the write, the parent rule, and the failure path."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import insights.computations as computations
from insights.computations import (
    STYLE_PARENT_MIN_SHARE,
    _style_parent_rows,
    compute_and_store_style_year_counts,
    endpoint_timeout,
)


STYLE_YEAR_COUNTS_PATH = "/api/internal/insights/style-year-counts"


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
    """Normalised SQL of every execute and executemany, in call order."""
    statements: list[str] = _cursor(pool).statements
    return statements


def _rows(pool: MagicMock, table: str) -> list[Any]:
    """Parameter rows bulk-inserted into ``insights.<table>``."""
    rows: list[Any] = []
    for call in _cursor(pool).executemany.await_args_list:
        if f"INSERT INTO insights.{table} " in call.args[0]:
            rows.extend(call.args[1])
    return rows


def _client(body: dict[str, Any]) -> AsyncMock:
    response = MagicMock()
    response.json = MagicMock(return_value=body)
    response.raise_for_status = MagicMock()
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    return client


def _years(style: str = "House", year: int = 1990, release_count: int = 10, master_count: int = 4) -> dict[str, Any]:
    return {"style": style, "year": year, "release_count": release_count, "master_count": master_count}


def _genre_years(genre: str = "Electronic", year: int = 1990, release_count: int = 40) -> dict[str, Any]:
    return {"genre": genre, "year": year, "release_count": release_count}


def _share(style: str, genre: str, share: float) -> dict[str, Any]:
    return {"style": style, "genre": genre, "release_count": int(share * 100), "style_release_count": 100, "share": share}


class TestTheWrite:
    @pytest.mark.asyncio
    async def test_writes_counts_and_parents_and_returns_the_row_total(self) -> None:
        client = _client(
            {
                "style_years": [_years("House", 1990, 10, 4), _years("House", 1991, 12, 5)],
                "style_parents": [_share("House", "Electronic", 0.9), _share("House", "Funk / Soul", 0.3)],
                "genre_years": [_genre_years("Electronic", 1990, 40), _genre_years("Electronic", 1991, 50), _genre_years("Funk / Soul", 1990, 8)],
            }
        )
        pool = _make_pool()

        written = await compute_and_store_style_year_counts(client, pool)

        assert written == 7
        assert _rows(pool, "style_year_counts") == [("House", 1990, 10, 4), ("House", 1991, 12, 5)]
        assert _rows(pool, "style_parents") == [("House", "Electronic", 0.9, True), ("House", "Funk / Soul", 0.3, False)]
        assert _rows(pool, "genre_year_counts") == [("Electronic", 1990, 40), ("Electronic", 1991, 50), ("Funk / Soul", 1990, 8)]

    @pytest.mark.asyncio
    async def test_replaces_all_three_tables_inside_one_transaction_before_inserting(self) -> None:
        client = _client({"style_years": [_years()], "style_parents": [_share("House", "Electronic", 0.9)], "genre_years": [_genre_years()]})
        pool = _make_pool()

        await compute_and_store_style_year_counts(client, pool)

        conn = pool.connection()
        conn.transaction.assert_called_once_with()
        statements = _statements(pool)
        assert statements[:3] == [
            "DELETE FROM insights.style_year_counts",
            "DELETE FROM insights.style_parents",
            "DELETE FROM insights.genre_year_counts",
        ]
        assert statements[3].startswith("INSERT INTO insights.style_year_counts")
        assert statements[4].startswith("INSERT INTO insights.genre_year_counts")
        assert statements[5].startswith("INSERT INTO insights.style_parents")

    @pytest.mark.asyncio
    async def test_reads_the_endpoint_under_its_own_budget_and_logs_completed(self) -> None:
        client = _client({"style_years": [_years()], "style_parents": [_share("House", "Electronic", 0.9)], "genre_years": [_genre_years()]})
        pool = _make_pool()

        with patch.object(computations, "_log_computation", new=AsyncMock()) as log:
            await compute_and_store_style_year_counts(client, pool)

        assert client.get.await_args.args == (STYLE_YEAR_COUNTS_PATH,)
        assert client.get.await_args.kwargs["timeout"] == endpoint_timeout(STYLE_YEAR_COUNTS_PATH)
        assert log.await_args.args[1:3] == ("style_year_counts", "completed")
        assert log.await_args.args[4] == 3


class TestTheReadBudget:
    def test_exceeds_the_producers_published_processing_time(self) -> None:
        read = endpoint_timeout(STYLE_YEAR_COUNTS_PATH).read
        assert read is not None
        assert read > 300.0

    def test_is_an_explicit_endpoint_entry(self) -> None:
        assert STYLE_YEAR_COUNTS_PATH in computations.ENDPOINT_READ_TIMEOUTS


class TestTheParentRule:
    def test_the_threshold_is_twenty_percent(self) -> None:
        assert STYLE_PARENT_MIN_SHARE == 0.20

    def test_keeps_shares_at_or_above_the_threshold_and_drops_the_rest(self) -> None:
        rows = _style_parent_rows(
            [
                _share("House", "Electronic", 0.62),
                _share("House", "Funk / Soul", 0.2),
                _share("House", "Pop", 0.19999),
                _share("House", "Rock", 0.01),
            ]
        )

        assert [(genre, share) for _style, genre, share, _dominant in rows] == [("Electronic", 0.62), ("Funk / Soul", 0.2)]

    def test_a_style_with_no_share_at_the_threshold_has_no_parent(self) -> None:
        assert _style_parent_rows([_share("Odd", "Rock", 0.1), _share("Odd", "Pop", 0.05)]) == []

    def test_the_dominant_parent_is_the_highest_share(self) -> None:
        rows = _style_parent_rows([_share("House", "Funk / Soul", 0.4), _share("House", "Electronic", 0.55), _share("House", "Pop", 0.25)])

        assert {genre: dominant for _style, genre, _share_value, dominant in rows} == {"Electronic": True, "Funk / Soul": False, "Pop": False}

    def test_a_tie_for_the_highest_share_goes_to_the_first_genre_by_name(self) -> None:
        shares = [_share("House", "Funk / Soul", 0.4), _share("House", "Electronic", 0.4), _share("House", "Pop", 0.4)]

        for ordering in (shares, list(reversed(shares))):
            rows = _style_parent_rows(ordering)
            assert [genre for _style, genre, _share_value, dominant in rows if dominant] == ["Electronic"]

    def test_each_style_has_exactly_one_dominant_parent(self) -> None:
        rows = _style_parent_rows(
            [
                _share("House", "Electronic", 0.7),
                _share("House", "Pop", 0.3),
                _share("Disco", "Funk / Soul", 0.5),
                _share("Disco", "Electronic", 0.5),
                _share("Dub", "Reggae", 0.21),
            ]
        )

        for style in ("House", "Disco", "Dub"):
            assert sum(1 for row_style, _genre, _share_value, dominant in rows if row_style == style and dominant) == 1

    @pytest.mark.asyncio
    async def test_only_kept_parents_reach_the_table(self) -> None:
        client = _client(
            {
                "style_years": [_years("House"), _years("Odd")],
                "style_parents": [_share("House", "Electronic", 0.8), _share("House", "Pop", 0.05), _share("Odd", "Rock", 0.1)],
                "genre_years": [_genre_years()],
            }
        )
        pool = _make_pool()

        written = await compute_and_store_style_year_counts(client, pool)

        assert _rows(pool, "style_parents") == [("House", "Electronic", 0.8, True)]
        assert written == 4

    @pytest.mark.asyncio
    async def test_a_result_whose_parents_all_fall_under_the_threshold_still_replaces_the_counts(self) -> None:
        client = _client({"style_years": [_years("Odd")], "style_parents": [_share("Odd", "Rock", 0.1)], "genre_years": [_genre_years("Rock")]})
        pool = _make_pool()

        written = await compute_and_store_style_year_counts(client, pool)

        assert written == 2
        assert _rows(pool, "style_parents") == []
        assert "DELETE FROM insights.style_parents" in _statements(pool)


class TestTheEmptyResultRule:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            {"style_years": [], "style_parents": [], "genre_years": []},
            {"style_years": [_years()], "style_parents": [], "genre_years": [_genre_years()]},
            {"style_years": [], "style_parents": [_share("House", "Electronic", 0.9)], "genre_years": [_genre_years()]},
            {"style_years": [_years()], "style_parents": [_share("House", "Electronic", 0.9)], "genre_years": []},
            {"style_years": [_years()], "style_parents": [_share("House", "Electronic", 0.9)]},
        ],
        ids=["all-empty", "no-parents", "no-years", "no-genre-years", "genre-years-absent"],
    )
    async def test_an_empty_producer_result_leaves_the_previous_snapshot_intact(self, body: dict[str, Any]) -> None:
        pool = _make_pool()

        written = await compute_and_store_style_year_counts(_client(body), pool)

        assert written == 0
        assert not any(("style_" in statement or "genre_year" in statement) and "computation_log" not in statement for statement in _statements(pool))
        _cursor(pool).executemany.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_empty_result_is_logged_completed_with_no_rows(self) -> None:
        pool = _make_pool()

        with patch.object(computations, "_log_computation", new=AsyncMock()) as log:
            await compute_and_store_style_year_counts(_client({"style_years": [], "style_parents": [], "genre_years": []}), pool)

        assert log.await_args.args[1:3] == ("style_year_counts", "completed")
        assert log.await_args.args[4] == 0


class TestTheFailurePath:
    @pytest.mark.asyncio
    async def test_an_api_error_is_logged_failed_and_reraised_without_touching_the_tables(self) -> None:
        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.ReadTimeout(""))
        pool = _make_pool()

        with (
            patch.object(computations, "_log_computation", new=AsyncMock()) as log,
            pytest.raises(httpx.ReadTimeout),
        ):
            await compute_and_store_style_year_counts(client, pool)

        assert log.await_args.args[1:3] == ("style_year_counts", "failed")
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
            await compute_and_store_style_year_counts(client, _make_pool())

        assert log.await_args.args[2] == "failed"

    @pytest.mark.asyncio
    async def test_a_storage_error_is_logged_failed_and_reraised(self) -> None:
        client = _client({"style_years": [_years()], "style_parents": [_share("House", "Electronic", 0.9)], "genre_years": [_genre_years()]})
        pool = _make_pool()
        _cursor(pool).executemany = AsyncMock(side_effect=RuntimeError("DB error"))

        with (
            patch.object(computations, "_log_computation", new=AsyncMock()) as log,
            pytest.raises(RuntimeError, match="DB error"),
        ):
            await compute_and_store_style_year_counts(client, pool)

        assert log.await_args.args[1:3] == ("style_year_counts", "failed")
        assert log.await_args.kwargs["error_message"] == "RuntimeError: DB error"

    @pytest.mark.asyncio
    async def test_the_transaction_is_not_swallowed_on_error(self) -> None:
        """The error must propagate through ``conn.transaction()`` so PostgreSQL rolls back."""
        client = _client({"style_years": [_years()], "style_parents": [_share("House", "Electronic", 0.9)], "genre_years": [_genre_years()]})
        pool = _make_pool()
        _cursor(pool).executemany = AsyncMock(side_effect=RuntimeError("DB error"))
        transaction = pool.connection().transaction()

        with patch.object(computations, "_log_computation", new=AsyncMock()), pytest.raises(RuntimeError):
            await compute_and_store_style_year_counts(client, pool)

        exit_args = transaction.__aexit__.await_args.args
        assert exit_args[0] is not None


class TestTheSchedulerSequence:
    @pytest.mark.asyncio
    async def test_runs_right_after_genre_trends(self) -> None:
        order: list[str] = []

        def record(name: str) -> Any:
            async def run(*_args: object, **_kwargs: object) -> int:
                order.append(name)
                return 1

            return run

        names = [
            "artist_centrality",
            "genre_trends",
            "style_year_counts",
            "style_atlas",
            "label_longevity",
            "anniversaries",
            "data_completeness",
            "community_enrichment",
            "rarity",
            "activity_summary",
        ]
        with (
            patch.object(computations, "compute_and_store_artist_centrality", new=record("artist_centrality")),
            patch.object(computations, "compute_and_store_genre_trends", new=record("genre_trends")),
            patch.object(computations, "compute_and_store_style_year_counts", new=record("style_year_counts")),
            patch.object(computations, "compute_and_store_style_atlas", new=record("style_atlas")),
            patch.object(computations, "compute_and_store_label_longevity", new=record("label_longevity")),
            patch.object(computations, "compute_and_store_anniversaries", new=record("anniversaries")),
            patch.object(computations, "compute_and_store_data_completeness", new=record("data_completeness")),
            patch.object(computations, "compute_and_store_community_enrichment", new=record("community_enrichment")),
            patch.object(computations, "compute_and_store_rarity", new=record("rarity")),
            patch.object(computations, "compute_and_store_activity_summary", new=record("activity_summary")),
        ):
            results = await computations.run_all_computations(AsyncMock(), _make_pool())

        assert order == names
        assert results["style_year_counts"] == 1
