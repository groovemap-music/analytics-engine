"""GET /api/insights/style-atlas: the payload, the generation-keyed cache, and the degraded paths."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from insights.cache import InsightsCache
from insights.style_lifecycle import LIFECYCLE_RULE, STAGES, style_lifecycles
from tests.conftest import TEST_CACHE_GENERATION


if TYPE_CHECKING:
    from collections.abc import Callable
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient


MONTH = date(2026, 10, 1)
COMPUTED_AT = datetime(2026, 10, 1, 3, 0, tzinfo=UTC)
LAST_YEAR = 2025

SECTORS = [
    ("Electronic", 0, 0.0, 3.0, 0.5, 0.1, COMPUTED_AT),
    ("Rock", 1, 3.0, 6.5, -0.4, -0.2, COMPUTED_AT),
]
NODES = [
    ("ghost", "Ambient", "Rock", -0.5, -0.3),
    ("style", "Ambient", "Electronic", 0.4, 0.2),
    ("style", "Shoegaze", "Rock", -0.6, -0.1),
]
PARENTS = [
    ("Ambient", "Electronic", 0.8, True),
    ("Ambient", "Rock", 0.3, False),
    ("Shoegaze", "Rock", 0.9, True),
]
# Ambient emerges in 1990 (25 cumulative releases) and keeps releasing; Shoegaze never gets to 25.
YEAR_COUNTS = [("Ambient", year, 10) for year in range(1988, LAST_YEAR + 1)] + [("Shoegaze", 1991, 3), ("Shoegaze", 1992, 4)]
# Electronic emerged in 1980, ten years before Ambient. Rock reaches 25 cumulative releases in 1990, the same year.
GENRE_YEAR_COUNTS = [("Electronic", year, 30) for year in range(1980, LAST_YEAR + 1)] + [("Rock", 1989, 10), ("Rock", 1990, 15), ("Rock", 1991, 40)]


def _answer(
    *,
    month: date | None = MONTH,
    year_counts: list[tuple[str, int, int]] | None = None,
    genre_year_counts: list[tuple[str, int, int]] | None = None,
) -> Callable[[str], list[tuple[Any, ...]]]:
    """Rows for each query, told apart by the table it reads."""

    def rows(sql: str) -> list[tuple[Any, ...]]:
        if "max(month)" in sql:
            return [(month,)]
        if "genre_atlas_sectors" in sql:
            return SECTORS
        if "genre_year_counts" in sql:
            return GENRE_YEAR_COUNTS if genre_year_counts is None else genre_year_counts
        if "style_year_counts" in sql:
            return YEAR_COUNTS if year_counts is None else year_counts
        if "style_parents" in sql:
            return PARENTS
        return NODES

    return rows


def _seed(pool: MagicMock, answer: Callable[[str], list[tuple[Any, ...]]]) -> AsyncMock:
    """Point the pool's cursor at ``answer``, and return the cursor to inspect its queries."""
    cursor = pool.connection.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
    last: list[str] = []

    async def execute(sql: str, _params: Any = None) -> None:
        last[:] = [sql]

    cursor.execute = AsyncMock(side_effect=execute)
    cursor.fetchone = AsyncMock(side_effect=lambda: answer(last[0])[0])
    cursor.fetchall = AsyncMock(side_effect=lambda: answer(last[0]))
    return cursor


class TestPayload:
    @pytest.fixture
    def body(self, test_client: TestClient, mock_pg_pool: MagicMock) -> dict[str, Any]:
        _seed(mock_pg_pool, _answer())
        response = test_client.get("/api/insights/style-atlas")
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        return body

    def test_month_and_computed_at(self, body: dict[str, Any]) -> None:
        assert body["month"] == "2026-10-01"
        assert datetime.fromisoformat(body["computed_at"]) == COMPUTED_AT

    def test_genres_come_with_ring_order_and_sector_frames(self, body: dict[str, Any]) -> None:
        assert body["genres"] == [
            {"genre": "Electronic", "ring_order": 0, "sector_start": 0.0, "sector_end": 3.0, "x": 0.5, "y": 0.1, "emergence_year": 1980},
            {"genre": "Rock", "ring_order": 1, "sector_start": 3.0, "sector_end": 6.5, "x": -0.4, "y": -0.2, "emergence_year": 1990},
        ]

    def test_styles_carry_their_position_and_host_genre(self, body: dict[str, Any]) -> None:
        by_name = {style["style"]: style for style in body["styles"]}
        assert sorted(by_name) == ["Ambient", "Shoegaze"]
        assert (by_name["Ambient"]["genre"], by_name["Ambient"]["x"], by_name["Ambient"]["y"]) == ("Electronic", 0.4, 0.2)

    def test_ghosts_are_separate_copies_in_the_secondary_parents_sector(self, body: dict[str, Any]) -> None:
        assert body["ghosts"] == [{"style": "Ambient", "genre": "Rock", "x": -0.5, "y": -0.3}]

    def test_parents_have_shares_the_dominant_flag_and_the_emergence_edge(self, body: dict[str, Any]) -> None:
        ambient = next(style for style in body["styles"] if style["style"] == "Ambient")
        assert ambient["parents"] == [
            {"genre": "Electronic", "share": 0.8, "dominant": True, "emergence_edge": True},
            {"genre": "Rock", "share": 0.3, "dominant": False, "emergence_edge": False},
        ]

    def test_lifecycle_is_derived_from_the_stored_counts_with_the_same_module(self, body: dict[str, Any]) -> None:
        expected = style_lifecycles(YEAR_COUNTS, LAST_YEAR)
        by_name = {style["style"]: style for style in body["styles"]}
        for name, style in by_name.items():
            assert style["stages"] == expected.stages[expected.styles.index(name)].tolist()
            assert style["stage"] == expected.stage(name, LAST_YEAR)
            assert style["emergence_year"] == expected.emergence_years()[name]
        assert by_name["Ambient"]["emergence_year"] == 1990
        assert by_name["Ambient"]["stage"] == "established"
        assert by_name["Shoegaze"]["emergence_year"] is None
        assert by_name["Shoegaze"]["stage"] == "absent"

    def test_the_timeline_says_which_years_the_stage_list_covers_and_which_are_provisional(self, body: dict[str, Any]) -> None:
        timeline = body["timeline"]
        assert timeline == {"first_year": 1988, "last_year": LAST_YEAR, "provisional_from": LAST_YEAR - 1}
        for style in body["styles"]:
            assert len(style["stages"]) == timeline["last_year"] - timeline["first_year"] + 1
            assert "provisional" not in style

    def test_settled_stage_is_the_stage_in_the_last_non_provisional_year(self, body: dict[str, Any]) -> None:
        expected = style_lifecycles(YEAR_COUNTS, LAST_YEAR)
        settled_year = body["timeline"]["provisional_from"] - 1
        assert settled_year == LAST_YEAR - 2
        for style in body["styles"]:
            assert style["settled_stage"] == expected.stage(style["style"], settled_year)
        by_name = {style["style"]: style for style in body["styles"]}
        assert by_name["Ambient"]["settled_stage"] == "established"
        assert by_name["Shoegaze"]["settled_stage"] == "absent"

    def test_settled_stage_and_latest_stage_differ_when_the_provisional_years_decline(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        # 100 releases a year to 2023, then nothing: the settled year (2023) is still established, and the
        # provisional years drag the trailing mean under half the peak, so the latest year reads declining.
        _seed(mock_pg_pool, _answer(year_counts=[("Ambient", year, 100) for year in range(1980, 2024)]))

        body = test_client.get("/api/insights/style-atlas").json()

        ambient = next(style for style in body["styles"] if style["style"] == "Ambient")
        assert body["timeline"] == {"first_year": 1980, "last_year": LAST_YEAR, "provisional_from": LAST_YEAR - 1}
        assert ambient["settled_stage"] == "established"
        assert ambient["stage"] == "declining"
        assert ambient["stages"][-1] == STAGES.index("declining")

    def test_settled_stage_is_null_when_the_settled_year_precedes_the_timeline(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        _seed(mock_pg_pool, _answer(year_counts=[("Ambient", LAST_YEAR - 1, 30), ("Ambient", LAST_YEAR, 30)]))

        body = test_client.get("/api/insights/style-atlas").json()

        assert body["timeline"] == {"first_year": LAST_YEAR - 1, "last_year": LAST_YEAR, "provisional_from": LAST_YEAR - 1}
        assert {style["settled_stage"] for style in body["styles"]} == {None}
        assert {style["stage"] for style in body["styles"]} <= set(STAGES)

    def test_stage_indices_name_stages_through_the_rule(self, body: dict[str, Any]) -> None:
        assert body["rule"] == LIFECYCLE_RULE.describe()
        assert body["rule"]["stages"] == list(STAGES)
        assert body["rule"]["emerge_total"] == 25.0
        assert body["rule"]["provisional_years"] == 2
        assert body["rule"]["rule_version"] == "gm-design-4804/1"

    def test_a_style_with_no_year_counts_is_absent_throughout(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        _seed(mock_pg_pool, _answer(year_counts=[("Ambient", year, 10) for year in range(1988, LAST_YEAR + 1)]))

        body = test_client.get("/api/insights/style-atlas").json()

        shoegaze = next(style for style in body["styles"] if style["style"] == "Shoegaze")
        assert shoegaze["stage"] == "absent"
        assert shoegaze["emergence_year"] is None
        assert set(shoegaze["stages"]) == {0}

    def test_the_read_is_six_queries_scoped_to_the_latest_month_and_the_last_complete_year(
        self, test_client: TestClient, mock_pg_pool: MagicMock
    ) -> None:
        cursor = _seed(mock_pg_pool, _answer())

        test_client.get("/api/insights/style-atlas")

        calls = [(call.args[0], call.args[1] if len(call.args) > 1 else None) for call in cursor.execute.call_args_list]
        assert len(calls) == 6
        scoped = {
            "genre_atlas_sectors": (MONTH,),
            "style_atlas_layout": (MONTH,),
            "style_year_counts": (LAST_YEAR,),
            "genre_year_counts": (LAST_YEAR,),
            "style_parents": (MONTH,),
        }
        for table, params in scoped.items():
            matching = [args for sql, args in calls if f"insights.{table}" in sql and "max(month)" not in sql]
            assert params in matching, table


class TestEmergenceEdges:
    """A style's edge to a parent is drawn only when it emerged (25 cumulative releases) after the genre did.

    In these fixtures Ambient emerges in 1990 and Shoegaze never does.
    """

    @staticmethod
    def _read(test_client: TestClient, mock_pg_pool: MagicMock, genre_years: list[tuple[str, int, int]]) -> dict[str, Any]:
        _seed(mock_pg_pool, _answer(genre_year_counts=genre_years))
        body: dict[str, Any] = test_client.get("/api/insights/style-atlas").json()
        return body

    @staticmethod
    def _edges(body: dict[str, Any], style: str) -> dict[str, bool]:
        parents = next(entry for entry in body["styles"] if entry["style"] == style)["parents"]
        return {parent["genre"]: parent["emergence_edge"] for parent in parents}

    @pytest.mark.parametrize(
        ("genre_first_year", "drawn"),
        [(1980, True), (1989, True), (1990, False), (1991, False), (2010, False)],
        ids=["well-after", "one-year-after", "same-year", "one-year-before", "well-before"],
    )
    def test_the_edge_is_drawn_only_when_the_style_emerges_after_the_genre(
        self, test_client: TestClient, mock_pg_pool: MagicMock, genre_first_year: int, drawn: bool
    ) -> None:
        body = self._read(test_client, mock_pg_pool, [("Electronic", year, 30) for year in range(genre_first_year, LAST_YEAR + 1)])

        assert self._edges(body, "Ambient")["Electronic"] is drawn
        assert next(genre for genre in body["genres"] if genre["genre"] == "Electronic")["emergence_year"] == genre_first_year

    def test_a_genre_that_never_reaches_25_releases_has_no_emergence_year_and_no_edge(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        body = self._read(test_client, mock_pg_pool, [("Electronic", year, 1) for year in range(1990, 2000)])

        assert self._edges(body, "Ambient")["Electronic"] is False
        assert next(genre for genre in body["genres"] if genre["genre"] == "Electronic")["emergence_year"] is None

    def test_a_genre_with_no_year_counts_has_no_emergence_year_and_no_edge(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        body = self._read(test_client, mock_pg_pool, [])

        assert {genre["genre"]: genre["emergence_year"] for genre in body["genres"]} == {"Electronic": None, "Rock": None}
        assert set(self._edges(body, "Ambient").values()) == {False}

    def test_a_style_that_never_reaches_25_releases_draws_no_edge_even_to_an_old_genre(
        self, test_client: TestClient, mock_pg_pool: MagicMock
    ) -> None:
        body = self._read(test_client, mock_pg_pool, [("Rock", year, 30) for year in range(1950, LAST_YEAR + 1)])

        shoegaze = next(style for style in body["styles"] if style["style"] == "Shoegaze")
        assert shoegaze["emergence_year"] is None
        assert self._edges(body, "Shoegaze") == {"Rock": False}

    def test_every_parent_of_a_style_is_judged_on_its_own_genre(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        body = self._read(test_client, mock_pg_pool, [("Electronic", year, 30) for year in range(1980, LAST_YEAR + 1)] + [("Rock", 1995, 40)])

        assert self._edges(body, "Ambient") == {"Electronic": True, "Rock": False}


class TestEmptyDatabase:
    """Before the first monthly run there is no month, which is an empty atlas and not a 500."""

    def test_returns_an_empty_atlas(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        cursor = _seed(mock_pg_pool, _answer(month=None))

        response = test_client.get("/api/insights/style-atlas")

        assert response.status_code == 200
        assert response.json() == {
            "month": None,
            "computed_at": None,
            "rule": LIFECYCLE_RULE.describe(),
            "timeline": None,
            "genres": [],
            "styles": [],
            "ghosts": [],
        }
        assert cursor.execute.await_count == 1

    def test_the_default_empty_mock_database_is_also_empty(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/style-atlas")

        assert response.status_code == 200
        assert response.json()["styles"] == []


class TestNotReady:
    def test_503_without_a_pool(self, test_client: TestClient) -> None:
        import insights.insights as module

        module._pool = None

        response = test_client.get("/api/insights/style-atlas")

        assert response.status_code == 503


class TestCache:
    def test_miss_reads_postgres_and_stores_under_the_generation_it_read(
        self, test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock
    ) -> None:
        _seed(mock_pg_pool, _answer())

        response = test_client_with_cache.get("/api/insights/style-atlas")

        assert response.status_code == 200
        mock_cache.get.assert_called_once_with("insights:style-atlas", TEST_CACHE_GENERATION)
        mock_cache.set.assert_called_once()
        key, value, generation = mock_cache.set.call_args.args
        assert (key, generation) == ("insights:style-atlas", TEST_CACHE_GENERATION)
        assert value == response.json()

    def test_hit_skips_postgres(self, test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock) -> None:
        cached = {"month": "2026-09-01", "styles": []}
        mock_cache.get.return_value = cached

        response = test_client_with_cache.get("/api/insights/style-atlas")

        assert response.json() == cached
        mock_cache.set.assert_not_called()
        mock_pg_pool.connection.assert_not_called()

    def test_the_cached_value_is_plain_json(self, test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock) -> None:
        _seed(mock_pg_pool, _answer())

        test_client_with_cache.get("/api/insights/style-atlas")

        json.dumps(mock_cache.set.call_args.args[1])

    def test_a_recompute_retires_the_cached_atlas(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        """A real InsightsCache over a fake Valkey: a new generation makes the next read miss."""
        import insights.insights as module

        store: dict[str, str] = {}
        valkey = AsyncMock()
        valkey.get = AsyncMock(side_effect=store.get)
        valkey.set = AsyncMock(side_effect=lambda key, value, ex: store.__setitem__(key, value))
        module._cache = InsightsCache(valkey, ttl_seconds=60)
        cursor = _seed(mock_pg_pool, _answer())

        first = test_client.get("/api/insights/style-atlas")
        queries = cursor.execute.await_count
        second = test_client.get("/api/insights/style-atlas")
        assert second.json() == first.json()
        assert cursor.execute.await_count == queries
        assert "insights:g0:style-atlas" in store

        # What the scheduler's invalidate_all does first: advance the generation.
        store["insights-generation"] = "1"
        test_client.get("/api/insights/style-atlas")
        assert cursor.execute.await_count == 2 * queries
        assert "insights:g1:style-atlas" in store


class TestValkeyDown:
    @pytest.fixture
    def broken_valkey(self) -> AsyncMock:
        valkey = AsyncMock()
        valkey.get = AsyncMock(side_effect=ConnectionError("valkey is down"))
        valkey.set = AsyncMock(side_effect=ConnectionError("valkey is down"))
        return valkey

    def test_serves_from_postgres(self, test_client: TestClient, mock_pg_pool: MagicMock, broken_valkey: AsyncMock) -> None:
        import insights.insights as module

        module._cache = InsightsCache(broken_valkey, ttl_seconds=60)
        _seed(mock_pg_pool, _answer())

        response = test_client.get("/api/insights/style-atlas")

        assert response.status_code == 200
        assert response.json()["month"] == "2026-10-01"
        assert [style["style"] for style in response.json()["styles"]] == ["Ambient", "Shoegaze"]
        broken_valkey.set.assert_awaited_once()

    def test_serves_from_postgres_with_no_cache_configured(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        import insights.insights as module

        assert module._cache is None
        _seed(mock_pg_pool, _answer())

        response = test_client.get("/api/insights/style-atlas")

        assert response.status_code == 200
        assert len(response.json()["styles"]) == 2
