"""GET /api/insights/style-atlas/cotags: the payload, the generation-keyed cache, and an unknown style."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from insights.cache import InsightsCache
from tests.conftest import TEST_CACHE_GENERATION


if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient


URL = "/api/insights/style-atlas/cotags"
HOUSE = [("Techno", 300, 1), ("Disco", 250, 2), ("Garage", 250, 3)]


def _seed(pool: MagicMock, rows: list[tuple[Any, ...]]) -> AsyncMock:
    """Point the pool's cursor at ``rows`` and return the cursor to inspect its queries."""
    cursor = pool.connection.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
    cursor.execute = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=rows)
    return cursor


class TestPayload:
    def test_returns_other_release_count_and_rank_in_rank_order(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        _seed(mock_pg_pool, HOUSE)

        response = test_client.get(URL, params={"style": "House"})

        assert response.status_code == 200
        assert response.json() == [
            {"other": "Techno", "release_count": 300, "rank": 1},
            {"other": "Disco", "release_count": 250, "rank": 2},
            {"other": "Garage", "release_count": 250, "rank": 3},
        ]

    def test_the_query_is_bounded_to_the_style_asked_about_and_ordered_by_rank(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        cursor = _seed(mock_pg_pool, HOUSE)

        test_client.get(URL, params={"style": "Drum & Bass"})

        cursor.execute.assert_awaited_once()
        sql, params = cursor.execute.await_args.args
        assert "insights.style_cotags" in sql
        assert "WHERE style = %s" in sql
        assert "ORDER BY rank" in sql
        assert "LIMIT 25" in sql
        assert params == ("Drum & Bass",)

    def test_an_unknown_style_is_an_empty_list_not_an_error(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        _seed(mock_pg_pool, [])

        response = test_client.get(URL, params={"style": "No Such Style"})

        assert response.status_code == 200
        assert response.json() == []

    def test_a_missing_style_is_a_422(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        cursor = _seed(mock_pg_pool, HOUSE)

        response = test_client.get(URL)

        assert response.status_code == 422
        cursor.execute.assert_not_awaited()

    def test_the_default_empty_mock_database_is_an_empty_list(self, test_client: TestClient) -> None:
        response = test_client.get(URL, params={"style": "House"})

        assert response.status_code == 200
        assert response.json() == []


class TestNotReady:
    def test_503_without_a_pool(self, test_client: TestClient) -> None:
        import insights.insights as module

        module._pool = None

        response = test_client.get(URL, params={"style": "House"})

        assert response.status_code == 503


class TestCache:
    def test_miss_reads_postgres_and_stores_under_the_generation_it_read(
        self, test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock
    ) -> None:
        _seed(mock_pg_pool, HOUSE)

        response = test_client_with_cache.get(URL, params={"style": "House"})

        assert response.status_code == 200
        mock_cache.get.assert_called_once_with("insights:style-atlas:cotags:House", TEST_CACHE_GENERATION)
        key, value, generation = mock_cache.set.call_args.args
        assert (key, generation) == ("insights:style-atlas:cotags:House", TEST_CACHE_GENERATION)
        assert value == response.json()
        json.dumps(value)

    def test_each_style_has_its_own_key(self, test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock) -> None:
        _seed(mock_pg_pool, HOUSE)

        test_client_with_cache.get(URL, params={"style": "House"})
        test_client_with_cache.get(URL, params={"style": "Dub"})

        assert [call.args[0] for call in mock_cache.get.call_args_list] == ["insights:style-atlas:cotags:House", "insights:style-atlas:cotags:Dub"]

    def test_hit_skips_postgres(self, test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock) -> None:
        cached = [{"other": "Techno", "release_count": 1, "rank": 1}]
        mock_cache.get.return_value = cached

        response = test_client_with_cache.get(URL, params={"style": "House"})

        assert response.json() == cached
        mock_cache.set.assert_not_called()
        mock_pg_pool.connection.assert_not_called()

    def test_a_cached_empty_list_for_an_unknown_style_is_a_hit(
        self, test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock
    ) -> None:
        mock_cache.get.return_value = []

        response = test_client_with_cache.get(URL, params={"style": "No Such Style"})

        assert response.json() == []
        mock_pg_pool.connection.assert_not_called()

    def test_an_unknown_style_is_cached_too(self, test_client_with_cache: TestClient, mock_cache: AsyncMock, mock_pg_pool: MagicMock) -> None:
        _seed(mock_pg_pool, [])

        test_client_with_cache.get(URL, params={"style": "No Such Style"})

        assert mock_cache.set.call_args.args[1] == []

    def test_a_recompute_retires_the_cached_list(self, test_client: TestClient, mock_pg_pool: MagicMock) -> None:
        """A real InsightsCache over a fake Valkey: the list round-trips, and a new generation makes the next read miss."""
        import insights.insights as module

        store: dict[str, str] = {}
        valkey = AsyncMock()
        valkey.get = AsyncMock(side_effect=lambda key: store.get(key))
        valkey.set = AsyncMock(side_effect=lambda key, value, ex: store.__setitem__(key, value))
        module._cache = InsightsCache(valkey, ttl_seconds=60)
        cursor = _seed(mock_pg_pool, HOUSE)

        first = test_client.get(URL, params={"style": "House"})
        second = test_client.get(URL, params={"style": "House"})

        assert second.json() == first.json()
        assert cursor.execute.await_count == 1
        assert "insights:g0:style-atlas:cotags:House" in store

        store.clear()
        test_client.get(URL, params={"style": "House"})
        assert cursor.execute.await_count == 2
