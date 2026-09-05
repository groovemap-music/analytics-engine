"""Tests for insights FastAPI endpoints."""

import asyncio
import contextlib
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import TEST_CACHE_GENERATION


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client: TestClient) -> None:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "analytics-engine"
        assert "status" in data


class TestServiceIdentity:
    def test_runtime_identity_uses_repository_name(self) -> None:
        from insights import __version__
        from insights.insights import SERVICE_NAME, SOURCE_URL, STARTUP_BANNER, USER_AGENT, app

        assert SERVICE_NAME == "analytics-engine"
        assert "analytics-engine" in STARTUP_BANNER
        assert app.title == "GrooveMap Analytics Engine"
        assert f"analytics-engine/{__version__} (+https://github.com/groovemap-music/analytics-engine)" == USER_AGENT
        assert SOURCE_URL == "https://github.com/groovemap-music/analytics-engine"


class TestTopArtistsEndpoint:
    def test_returns_200(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/top-artists")
        assert response.status_code == 200

    def test_with_limit(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/top-artists?limit=10")
        assert response.status_code == 200


class TestGenreTrendsEndpoint:
    def test_requires_genre_param(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/genre-trends")
        assert response.status_code == 422

    def test_returns_200_with_genre(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/genre-trends?genre=Jazz")
        assert response.status_code == 200


class TestLabelLongevityEndpoint:
    def test_returns_200(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/label-longevity")
        assert response.status_code == 200


class TestThisMonthEndpoint:
    def test_returns_200(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/this-month")
        assert response.status_code == 200


class TestDataCompletenessEndpoint:
    def test_returns_200(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/data-completeness")
        assert response.status_code == 200


class TestComputationStatusEndpoint:
    def test_returns_200(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/status")
        assert response.status_code == 200

    def test_never_run_status_when_no_log_rows(self, test_client: TestClient) -> None:
        """When fetchone returns None for an insight type, status should be 'never_run'."""
        response = test_client.get("/api/insights/status")
        assert response.status_code == 200
        data = response.json()
        assert "statuses" in data
        # All 7 insight types should show 'never_run' since fetchone returns None (includes community_enrichment)
        assert len(data["statuses"]) == 7
        for status in data["statuses"]:
            assert status["status"] == "never_run"

    def test_status_with_log_rows(self, mock_http_client: AsyncMock, mock_pg_pool: AsyncMock) -> None:
        """When fetchone returns a row, status should reflect actual log data."""
        # Return a row with a real datetime for completed_at to verify serialization
        from datetime import datetime

        import insights.insights as _module

        mock_cursor = mock_pg_pool.connection.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
        mock_cursor.fetchone = AsyncMock(return_value=("artist_centrality", "completed", datetime(2026, 3, 18, 12, 0, 0, tzinfo=UTC), 1500))

        _module._http_client = mock_http_client
        _module._pool = mock_pg_pool
        _module._cache = None

        from insights.insights import app

        client = TestClient(app)
        response = client.get("/api/insights/status")
        assert response.status_code == 200
        data = response.json()
        assert "statuses" in data
        # Each status should have "completed"
        for status in data["statuses"]:
            assert status["status"] == "completed"


# ============================================================
# Cache integration tests
# ============================================================


class TestTopArtistsCacheIntegration:
    def test_cache_miss_queries_pg_and_stores(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        mock_cache.get.return_value = None
        response = test_client_with_cache.get("/api/insights/top-artists?limit=10")
        assert response.status_code == 200
        mock_cache.get.assert_called_once_with("insights:top-artists:10", TEST_CACHE_GENERATION)
        mock_cache.set.assert_called_once()
        key = mock_cache.set.call_args[0][0]
        assert key == "insights:top-artists:10"

    def test_cache_hit_returns_cached_data(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        cached = {"metric": "centrality", "items": [{"rank": 1, "artist_name": "Test"}], "count": 1}
        mock_cache.get.return_value = cached
        response = test_client_with_cache.get("/api/insights/top-artists?limit=10")
        assert response.status_code == 200
        assert response.json() == cached
        mock_cache.set.assert_not_called()


class TestGenreTrendsCacheIntegration:
    def test_cache_miss_queries_pg_and_stores(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        mock_cache.get.return_value = None
        response = test_client_with_cache.get("/api/insights/genre-trends?genre=Rock")
        assert response.status_code == 200
        mock_cache.get.assert_called_once_with("insights:genre-trends:Rock", TEST_CACHE_GENERATION)
        mock_cache.set.assert_called_once()

    def test_cache_hit_returns_cached_data(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        cached = {"genre": "Rock", "trends": [], "peak_decade": None}
        mock_cache.get.return_value = cached
        response = test_client_with_cache.get("/api/insights/genre-trends?genre=Rock")
        assert response.status_code == 200
        assert response.json() == cached
        mock_cache.set.assert_not_called()

    def test_cache_key_does_not_collide_genres_differing_only_by_colon_vs_underscore(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        """A prior `.replace(':', '_')` sanitization
        was non-injective — genre="A:B" and genre="A_B" both produced cache key
        "insights:genre-trends:A_B" and were served each other's cached response.
        The cache key must preserve the raw genre value (interior colons are
        harmless to InsightsCache.versioned_key, which only strips the fixed
        "insights:" literal prefix via removeprefix, not a colon split)."""
        mock_cache.get.return_value = None

        response = test_client_with_cache.get("/api/insights/genre-trends?genre=A:B")
        assert response.status_code == 200
        mock_cache.get.assert_called_once_with("insights:genre-trends:A:B", TEST_CACHE_GENERATION)

        mock_cache.reset_mock()
        mock_cache.get.return_value = None
        response = test_client_with_cache.get("/api/insights/genre-trends?genre=A_B")
        assert response.status_code == 200
        mock_cache.get.assert_called_once_with("insights:genre-trends:A_B", TEST_CACHE_GENERATION)


class TestLabelLongevityCacheIntegration:
    def test_cache_miss_queries_pg_and_stores(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        mock_cache.get.return_value = None
        response = test_client_with_cache.get("/api/insights/label-longevity?limit=10")
        assert response.status_code == 200
        mock_cache.get.assert_called_once_with("insights:label-longevity:10", TEST_CACHE_GENERATION)
        mock_cache.set.assert_called_once()

    def test_cache_hit_returns_cached_data(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        cached = {"items": [], "count": 0}
        mock_cache.get.return_value = cached
        response = test_client_with_cache.get("/api/insights/label-longevity?limit=10")
        assert response.status_code == 200
        assert response.json() == cached
        mock_cache.set.assert_not_called()


class TestThisMonthCacheIntegration:
    def test_cache_miss_queries_pg_empty_result_not_cached(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        mock_cache.get.return_value = None
        response = test_client_with_cache.get("/api/insights/this-month")
        assert response.status_code == 200
        # Cache key includes year-month
        call_key = mock_cache.get.call_args[0][0]
        assert call_key.startswith("insights:this-month:")
        # Empty results are NOT cached to avoid caching stale data on month boundaries
        mock_cache.set.assert_not_called()

    def test_cache_hit_returns_cached_data(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        cached = {"month": 3, "year": 2026, "items": [], "count": 0}
        mock_cache.get.return_value = cached
        response = test_client_with_cache.get("/api/insights/this-month")
        assert response.status_code == 200
        assert response.json() == cached
        mock_cache.set.assert_not_called()


class TestDataCompletenessCacheIntegration:
    def test_cache_miss_queries_pg_and_stores(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        mock_cache.get.return_value = None
        response = test_client_with_cache.get("/api/insights/data-completeness")
        assert response.status_code == 200
        mock_cache.get.assert_called_once_with("insights:data-completeness", TEST_CACHE_GENERATION)
        mock_cache.set.assert_called_once()

    def test_cache_hit_returns_cached_data(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        cached = {"items": [], "count": 0}
        mock_cache.get.return_value = cached
        response = test_client_with_cache.get("/api/insights/data-completeness")
        assert response.status_code == 200
        assert response.json() == cached
        mock_cache.set.assert_not_called()


class TestStatusEndpointNeverCached:
    def test_status_does_not_use_cache(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        response = test_client_with_cache.get("/api/insights/status")
        assert response.status_code == 200
        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()


# ============================================================
# Release rarity endpoint tests
# ============================================================


class TestReleaseRarityEndpoint:
    def test_returns_200(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/release-rarity")
        assert response.status_code == 200

    def test_with_limit(self, test_client: TestClient) -> None:
        response = test_client.get("/api/insights/release-rarity?limit=10")
        assert response.status_code == 200

    def test_not_ready(self) -> None:
        import insights.insights as _module

        original = _module._pool
        _module._pool = None
        try:
            from insights.insights import app

            client = TestClient(app)
            response = client.get("/api/insights/release-rarity")
            assert response.status_code == 503
        finally:
            _module._pool = original


class TestReleaseRarityCacheIntegration:
    def test_cache_miss_queries_pg_and_stores(
        self,
        mock_http_client: AsyncMock,
        mock_pg_pool: AsyncMock,
        mock_cache: AsyncMock,
    ) -> None:
        import insights.insights as _module

        _module._http_client = mock_http_client
        _module._pool = mock_pg_pool
        _module._cache = mock_cache

        # Return non-empty rows so caching is triggered
        mock_cursor = mock_pg_pool.connection.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                (
                    1,
                    "Title",
                    "Artist",
                    1990,
                    95.0,
                    "ultra-rare",
                    80.0,
                    90.0,
                    85.0,
                    70.0,
                    60.0,
                    50.0,
                    40.0,
                    ["vinyl"],
                    {"grooved": {"pressing_scarcity": 88.0}},
                    72.0,
                )
            ]
        )

        mock_cache.get.return_value = None

        from insights.insights import app

        client = TestClient(app)
        response = client.get("/api/insights/release-rarity?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["items"][0]["release_id"] == 1
        mock_cache.get.assert_called_once_with("insights:release-rarity:10", TEST_CACHE_GENERATION)
        mock_cache.set.assert_called_once()

    def test_grooved_release_carries_pressing_scarcity_and_family_signals(
        self,
        mock_http_client: AsyncMock,
        mock_pg_pool: AsyncMock,
        mock_cache: AsyncMock,
    ) -> None:
        """A grooved release (vinyl/shellac/grooved_other) carries a populated
        pressing_scarcity signal alongside its family breakdown (ADR 0007)."""
        import insights.insights as _module

        _module._http_client = mock_http_client
        _module._pool = mock_pg_pool
        _module._cache = mock_cache

        mock_cursor = mock_pg_pool.connection.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                (
                    101,
                    "Grooved Title",
                    "Grooved Artist",
                    1975,
                    91.0,
                    "ultra-rare",
                    77.0,
                    93.5,  # pressing_scarcity: populated for a grooved medium
                    82.0,
                    5.0,  # format_rarity: deprecated, still present at low weight
                    65.0,
                    58.0,
                    41.0,
                    ["vinyl", "shellac"],
                    {"grooved": {"pressing_scarcity": 90.0}},
                    74.0,
                )
            ]
        )
        mock_cache.get.return_value = None

        from insights.insights import app

        client = TestClient(app)
        response = client.get("/api/insights/release-rarity?limit=10")
        assert response.status_code == 200
        item = response.json()["items"][0]
        assert item["release_id"] == 101
        assert item["pressing_scarcity"] == 93.5
        assert item["media_families"] == ["vinyl", "shellac"]
        assert item["family_signals"] == {"grooved": {"pressing_scarcity": 90.0}}
        assert item["medium_rarity"] == 74.0
        assert item["format_rarity"] == 5.0

    def test_non_grooved_release_has_null_pressing_scarcity_and_empty_family_signals(
        self,
        mock_http_client: AsyncMock,
        mock_pg_pool: AsyncMock,
        mock_cache: AsyncMock,
    ) -> None:
        """A non-grooved release (e.g. digital) has no grooved extension, so
        pressing_scarcity is null and family_signals carries no entries (ADR 0007)."""
        import insights.insights as _module

        _module._http_client = mock_http_client
        _module._pool = mock_pg_pool
        _module._cache = mock_cache

        mock_cursor = mock_pg_pool.connection.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                (
                    102,
                    "Digital Title",
                    "Digital Artist",
                    2020,
                    48.0,
                    "common",
                    12.0,
                    None,  # pressing_scarcity: null, no grooved extension claims this release
                    30.0,
                    0.0,  # format_rarity: deprecated, at weight 0.0
                    20.0,
                    15.0,
                    55.0,
                    ["digital"],
                    {},
                    38.0,
                )
            ]
        )
        mock_cache.get.return_value = None

        from insights.insights import app

        client = TestClient(app)
        response = client.get("/api/insights/release-rarity?limit=10")
        assert response.status_code == 200
        item = response.json()["items"][0]
        assert item["release_id"] == 102
        assert item["pressing_scarcity"] is None
        assert item["media_families"] == ["digital"]
        assert item["family_signals"] == {}
        assert item["medium_rarity"] == 38.0

    def test_cache_hit_returns_cached_data(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
    ) -> None:
        cached = {"items": [{"release_id": 1}], "count": 1}
        mock_cache.get.return_value = cached
        response = test_client_with_cache.get("/api/insights/release-rarity")
        assert response.status_code == 200
        assert response.json() == cached
        mock_cache.set.assert_not_called()


class TestThisMonthCacheWithData:
    """Test that this-month endpoint caches when results are non-empty."""

    def test_cache_stores_non_empty_results(
        self,
        mock_http_client: AsyncMock,
        mock_pg_pool: AsyncMock,
        mock_cache: AsyncMock,
    ) -> None:
        import insights.insights as _module

        _module._http_client = mock_http_client
        _module._pool = mock_pg_pool
        _module._cache = mock_cache

        # Return non-empty rows
        mock_cursor = mock_pg_pool.connection.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
        mock_cursor.fetchall = AsyncMock(return_value=[("100", "Album", "Artist", 1990, 25)])
        mock_cache.get.return_value = None

        from insights.insights import app

        client = TestClient(app)
        response = client.get("/api/insights/this-month")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        # Non-empty results SHOULD be cached
        mock_cache.set.assert_called_once()


# ============================================================
# Lifespan tests
# ============================================================


class TestLifespan:
    def test_lifespan_annotations_resolve_at_runtime(self) -> None:
        """Keep Python 3.14's lazy lifespan annotations runtime-resolvable."""
        import inspect
        from collections.abc import AsyncGenerator
        from typing import get_type_hints

        from fastapi import FastAPI

        import insights.insights as _module

        expected = {"_app": FastAPI, "return": AsyncGenerator[None]}

        assert _module.lifespan.__annotations__ == expected
        assert inspect.get_annotations(_module.lifespan, eval_str=True) == expected
        assert get_type_hints(_module.lifespan) == expected

    @pytest.mark.asyncio
    async def test_lifespan_startup_and_shutdown(self) -> None:
        """Test the full lifespan context manager startup and shutdown paths."""
        from fastapi import FastAPI

        import insights.insights as _module

        mock_pool = AsyncMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.close = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.aclose = AsyncMock()

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        mock_health_srv = MagicMock()
        mock_health_srv.start_background = MagicMock()
        mock_health_srv.stop = MagicMock()

        mock_cache = MagicMock()

        mock_config = MagicMock()
        mock_config.postgres_host = "localhost:5432"
        mock_config.postgres_database = "test"
        mock_config.postgres_username = "user"
        mock_config.postgres_password = "pass"
        mock_config.api_base_url = "http://localhost:8004"
        mock_config.redis_host = "redis://localhost"
        mock_config.schedule_hours = 24
        mock_config.milestone_years = [25, 50]

        # Create a scheduler task that completes immediately
        async def fake_scheduler(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(100)

        fake_app = FastAPI()

        with (
            patch.object(_module, "setup_logging") as mock_setup_logging,
            patch.object(_module.InsightsConfig, "from_env", return_value=mock_config),
            patch.object(_module, "HealthServer", return_value=mock_health_srv),
            patch.object(_module, "AsyncPostgreSQLPool", return_value=mock_pool),
            patch("httpx.AsyncClient", return_value=mock_http_client),
            patch("redis.asyncio.from_url", new_callable=AsyncMock, return_value=mock_redis),
            patch.object(_module, "InsightsCache", return_value=mock_cache),
            patch.object(_module, "_scheduler_loop", side_effect=fake_scheduler),
        ):
            async with _module.lifespan(fake_app):
                # Verify startup
                mock_health_srv.start_background.assert_called_once()
                mock_pool.initialize.assert_awaited_once()
                assert _module._cache is mock_cache
                mock_setup_logging.assert_called_once_with("analytics-engine", log_file=Path("/logs/analytics-engine.log"))

            # Verify shutdown
            mock_health_srv.stop.assert_called_once()
            mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_calls_setup_telemetry_right_after_setup_logging_and_instruments_http(self) -> None:
        """setup_telemetry runs immediately after setup_logging; both HTTP surfaces are instrumented."""
        from fastapi import FastAPI

        import insights.insights as _module

        mock_pool = AsyncMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.close = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.aclose = AsyncMock()

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        mock_health_srv = MagicMock()

        mock_config = MagicMock()
        mock_config.postgres_host = "localhost:5432"
        mock_config.postgres_database = "test"
        mock_config.postgres_username = "user"
        mock_config.postgres_password = "pass"
        mock_config.api_base_url = "http://localhost:8004"
        mock_config.redis_host = "redis://localhost"
        mock_config.schedule_hours = 24
        mock_config.milestone_years = [25, 50]

        async def fake_scheduler(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(100)

        fake_app = FastAPI()
        call_order: list[str] = []

        with (
            patch.object(_module, "setup_logging", side_effect=lambda *_a, **_kw: call_order.append("setup_logging")),
            patch.object(_module, "setup_telemetry", side_effect=lambda *_a, **_kw: call_order.append("setup_telemetry")) as mock_setup_telemetry,
            patch.object(_module, "instrument_fastapi_app") as mock_instrument_fastapi,
            patch.object(_module, "instrument_httpx") as mock_instrument_httpx,
            patch.object(
                _module,
                "start_event_loop_monitor",
                side_effect=lambda *_a, **_kw: call_order.append("start_event_loop_monitor"),
            ) as mock_start_event_loop_monitor,
            patch.object(_module, "shutdown_telemetry") as mock_shutdown_telemetry,
            patch.object(_module.InsightsConfig, "from_env", return_value=mock_config),
            patch.object(_module, "HealthServer", return_value=mock_health_srv),
            patch.object(_module, "AsyncPostgreSQLPool", return_value=mock_pool),
            patch("httpx.AsyncClient", return_value=mock_http_client),
            patch("redis.asyncio.from_url", new_callable=AsyncMock, return_value=mock_redis),
            patch.object(_module, "_scheduler_loop", side_effect=fake_scheduler),
        ):
            async with _module.lifespan(fake_app):
                mock_setup_telemetry.assert_called_once_with("analytics-engine")
                mock_instrument_fastapi.assert_called_once_with(fake_app)
                mock_instrument_httpx.assert_called_once_with()
                mock_start_event_loop_monitor.assert_called_once_with()
                mock_shutdown_telemetry.assert_not_called()

            # setup_telemetry must run immediately after setup_logging, and the event-loop
            # monitor only once the providers it samples into are installed.
            assert call_order == ["setup_logging", "setup_telemetry", "start_event_loop_monitor"]
            mock_shutdown_telemetry.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_lifespan_starts_a_live_event_loop_monitor_on_its_own_loop(self) -> None:
        """With metrics exporting, the real monitor is started as a task on the lifespan's loop.

        `start_event_loop_monitor` is unpatched here: it is the library call that turns
        groovemap.runtime.event_loop.lag on, it only samples from a running loop, and it only
        samples when a metrics provider is installed — so both conditions are asserted through
        the value it hands back rather than through a mock.
        """
        from common import telemetry as common_telemetry
        from fastapi import FastAPI
        from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        import insights.insights as _module

        mock_pool = AsyncMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.close = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.aclose = AsyncMock()

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        mock_health_srv = MagicMock()

        mock_config = MagicMock()
        mock_config.postgres_host = "localhost:5432"
        mock_config.postgres_database = "test"
        mock_config.postgres_username = "user"
        mock_config.postgres_password = "pass"
        mock_config.api_base_url = "http://localhost:8004"
        mock_config.redis_host = "redis://localhost"
        mock_config.schedule_hours = 24
        mock_config.milestone_years = [25, 50]

        async def fake_scheduler(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(100)

        fake_app = FastAPI()
        provider = SdkMeterProvider(metric_readers=[InMemoryMetricReader()])
        monitors: list[object] = []

        def capture(*args: object, **kwargs: object) -> object:
            monitor = common_telemetry.start_event_loop_monitor(*args, **kwargs)
            monitors.append(monitor)
            return monitor

        with (
            patch.object(_module, "setup_logging"),
            patch.object(_module, "setup_telemetry"),
            patch.object(_module, "shutdown_telemetry"),
            patch.object(_module, "start_event_loop_monitor", side_effect=capture),
            # The library only samples into an installed metrics provider; stand one up in
            # memory so the real monitor takes its live path without a collector.
            patch.object(common_telemetry, "_provider", provider),
            patch.object(common_telemetry, "_sdk_provider", provider),
            patch.object(_module.InsightsConfig, "from_env", return_value=mock_config),
            patch.object(_module, "HealthServer", return_value=mock_health_srv),
            patch.object(_module, "AsyncPostgreSQLPool", return_value=mock_pool),
            patch("httpx.AsyncClient", return_value=mock_http_client),
            patch("redis.asyncio.from_url", new_callable=AsyncMock, return_value=mock_redis),
            patch.object(_module, "_scheduler_loop", side_effect=fake_scheduler),
        ):
            async with _module.lifespan(fake_app):
                assert len(monitors) == 1
                monitor = monitors[0]
                assert isinstance(monitor, asyncio.Task)
                assert not monitor.done()
                assert monitor.get_loop() is asyncio.get_running_loop()

            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor

    @pytest.mark.asyncio
    async def test_lifespan_telemetry_is_a_noop_without_an_otel_endpoint(self) -> None:
        """Regression: with OTEL_EXPORTER_OTLP_ENDPOINT unset, startup/shutdown behave exactly as before.

        setup_telemetry, instrument_fastapi_app, instrument_httpx, and shutdown_telemetry run for
        real here (unpatched) — conftest's isolated_otel_environment fixture guarantees no OTEL_*
        variable is set, so every one of them must degrade to a no-op and never raise.
        """
        from fastapi import FastAPI

        import insights.insights as _module

        mock_pool = AsyncMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.close = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.aclose = AsyncMock()

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        mock_health_srv = MagicMock()

        mock_config = MagicMock()
        mock_config.postgres_host = "localhost:5432"
        mock_config.postgres_database = "test"
        mock_config.postgres_username = "user"
        mock_config.postgres_password = "pass"
        mock_config.api_base_url = "http://localhost:8004"
        mock_config.redis_host = "redis://localhost"
        mock_config.schedule_hours = 24
        mock_config.milestone_years = [25, 50]

        async def fake_scheduler(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(100)

        fake_app = FastAPI()

        with (
            patch.object(_module, "setup_logging"),
            patch.object(_module.InsightsConfig, "from_env", return_value=mock_config),
            patch.object(_module, "HealthServer", return_value=mock_health_srv),
            patch.object(_module, "AsyncPostgreSQLPool", return_value=mock_pool),
            patch("httpx.AsyncClient", return_value=mock_http_client),
            patch("redis.asyncio.from_url", new_callable=AsyncMock, return_value=mock_redis),
            patch.object(_module, "_scheduler_loop", side_effect=fake_scheduler),
        ):
            async with _module.lifespan(fake_app):
                mock_pool.initialize.assert_awaited_once()
                assert _module._cache is not None

            mock_health_srv.stop.assert_called_once()
            mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_postgres_host_without_port(self) -> None:
        """When POSTGRES_HOST has no port suffix, default to port 5432."""
        from fastapi import FastAPI

        import insights.insights as _module

        mock_pool = AsyncMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.close = AsyncMock()

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        mock_health_srv = MagicMock()
        mock_health_srv.start_background = MagicMock()
        mock_health_srv.stop = MagicMock()

        mock_cache = MagicMock()

        mock_config = MagicMock()
        mock_config.postgres_host = "postgres"  # No port
        mock_config.postgres_database = "test"
        mock_config.postgres_username = "user"
        mock_config.postgres_password = "pass"
        mock_config.api_base_url = "http://localhost:8004"
        mock_config.redis_host = "redis://localhost"
        mock_config.schedule_hours = 24
        mock_config.milestone_years = [25, 50]

        async def fake_scheduler(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(100)

        fake_app = FastAPI()

        with (
            patch.object(_module, "setup_logging"),
            patch.object(_module.InsightsConfig, "from_env", return_value=mock_config),
            patch.object(_module, "HealthServer", return_value=mock_health_srv),
            patch.object(_module, "AsyncPostgreSQLPool", return_value=mock_pool) as mock_pool_cls,
            patch("httpx.AsyncClient", return_value=mock_http_client),
            patch("redis.asyncio.from_url", new_callable=AsyncMock, return_value=MagicMock()),
            patch.object(_module, "InsightsCache", return_value=mock_cache),
            patch.object(_module, "_scheduler_loop", side_effect=fake_scheduler),
        ):
            async with _module.lifespan(fake_app):
                # Verify pool was created with host="postgres" and port=5432
                call_kwargs = mock_pool_cls.call_args[1]
                assert call_kwargs["connection_params"]["host"] == "postgres"
                assert call_kwargs["connection_params"]["port"] == 5432

            mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_missing_internal_secret_warns_and_omits_header(self) -> None:
        """When INSIGHTS_INTERNAL_SECRET is unset, startup logs a warning and the HTTP
        client is built WITHOUT the X-Internal-Secret header (the API will reject calls)."""
        from fastapi import FastAPI

        import insights.insights as _module

        mock_pool = AsyncMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.close = AsyncMock()

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        mock_health_srv = MagicMock()
        mock_health_srv.start_background = MagicMock()
        mock_health_srv.stop = MagicMock()

        mock_config = MagicMock()
        mock_config.postgres_host = "localhost:5432"
        mock_config.postgres_database = "test"
        mock_config.postgres_username = "user"
        mock_config.postgres_password = "pass"
        mock_config.api_base_url = "http://localhost:8004"
        mock_config.redis_host = "redis://localhost"
        mock_config.schedule_hours = 24
        mock_config.milestone_years = [25, 50]
        mock_config.internal_secret = None  # secret unset → else branch

        async def fake_scheduler(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(100)

        fake_app = FastAPI()

        with (
            patch.object(_module, "setup_logging"),
            patch.object(_module.InsightsConfig, "from_env", return_value=mock_config),
            patch.object(_module, "HealthServer", return_value=mock_health_srv),
            patch.object(_module, "AsyncPostgreSQLPool", return_value=mock_pool),
            patch("httpx.AsyncClient", return_value=mock_http_client) as mock_client_cls,
            patch("redis.asyncio.from_url", new_callable=AsyncMock, return_value=MagicMock()),
            patch.object(_module, "InsightsCache", return_value=MagicMock()),
            patch.object(_module, "_scheduler_loop", side_effect=fake_scheduler),
            patch.object(_module.logger, "warning") as mock_warning,
        ):
            async with _module.lifespan(fake_app):
                # HTTP client built without the internal-secret header.
                headers = mock_client_cls.call_args.kwargs["headers"]
                assert "X-Internal-Secret" not in headers
                assert headers["User-Agent"] == _module.USER_AGENT

            # The missing-secret warning was emitted.
            assert any("INSIGHTS_INTERNAL_SECRET is not set" in str(c.args[0]) for c in mock_warning.call_args_list)

    @pytest.mark.asyncio
    async def test_lifespan_redis_unavailable_fallback(self) -> None:
        """When Redis is unavailable, caching should be disabled gracefully."""
        from fastapi import FastAPI

        import insights.insights as _module

        mock_pool = AsyncMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.close = AsyncMock()

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        mock_health_srv = MagicMock()
        mock_health_srv.start_background = MagicMock()
        mock_health_srv.stop = MagicMock()

        mock_config = MagicMock()
        mock_config.postgres_host = "localhost:5432"
        mock_config.postgres_database = "test"
        mock_config.postgres_username = "user"
        mock_config.postgres_password = "pass"
        mock_config.api_base_url = "http://localhost:8004"
        mock_config.redis_host = "redis://localhost"
        mock_config.schedule_hours = 24
        mock_config.milestone_years = [25, 50]

        async def fake_scheduler(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(100)

        fake_app = FastAPI()

        with (
            patch.object(_module, "setup_logging"),
            patch.object(_module.InsightsConfig, "from_env", return_value=mock_config),
            patch.object(_module, "HealthServer", return_value=mock_health_srv),
            patch.object(_module, "AsyncPostgreSQLPool", return_value=mock_pool),
            patch("httpx.AsyncClient", return_value=mock_http_client),
            patch("redis.asyncio.from_url", side_effect=ConnectionError("Redis down")),
            patch.object(_module, "_scheduler_loop", side_effect=fake_scheduler),
        ):
            async with _module.lifespan(fake_app):
                # Redis failure should result in None cache
                assert _module._redis is None
                assert _module._cache is None

            mock_health_srv.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_closes_redis_client_when_ping_fails(self) -> None:
        """from_url() is lazy — ping() is what actually
        opens the socket. If ping() raises (e.g. requirepass with a missing/
        wrong REDIS_PASSWORD), the client built by from_url() must still be
        closed before the reference is dropped — otherwise the connection
        pool and the socket ping() opened are never released."""
        from fastapi import FastAPI

        import insights.insights as _module

        mock_pool = AsyncMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.close = AsyncMock()

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        mock_health_srv = MagicMock()
        mock_health_srv.start_background = MagicMock()
        mock_health_srv.stop = MagicMock()

        mock_config = MagicMock()
        mock_config.postgres_host = "localhost:5432"
        mock_config.postgres_database = "test"
        mock_config.postgres_username = "user"
        mock_config.postgres_password = "pass"
        mock_config.api_base_url = "http://localhost:8004"
        mock_config.redis_host = "redis://localhost"
        mock_config.schedule_hours = 24
        mock_config.milestone_years = [25, 50]

        # from_url() succeeds (lazy) but the socket-opening ping() fails —
        # the exact AUTH-failure-under-requirepass scenario.
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("NOAUTH Authentication required"))
        mock_redis.aclose = AsyncMock()

        async def fake_scheduler(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(100)

        fake_app = FastAPI()

        with (
            patch.object(_module, "setup_logging"),
            patch.object(_module.InsightsConfig, "from_env", return_value=mock_config),
            patch.object(_module, "HealthServer", return_value=mock_health_srv),
            patch.object(_module, "AsyncPostgreSQLPool", return_value=mock_pool),
            patch("httpx.AsyncClient", return_value=mock_http_client),
            patch("redis.asyncio.from_url", new_callable=AsyncMock, return_value=mock_redis),
            patch.object(_module, "_scheduler_loop", side_effect=fake_scheduler),
        ):
            async with _module.lifespan(fake_app):
                # Falls back to no caching, exactly like the from_url()-raises case.
                assert _module._redis is None
                assert _module._cache is None

            mock_health_srv.stop.assert_called_once()

        # The orphaned client (from the successful from_url()) must have
        # been closed before _redis was set to None.
        mock_redis.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_redis_close_failure_on_ping_error_is_swallowed(self) -> None:
        """A failure while cleaning up the orphaned client (aclose() itself
        raising) must not prevent the graceful PostgreSQL-fallback path."""
        from fastapi import FastAPI

        import insights.insights as _module

        mock_pool = AsyncMock()
        mock_pool.initialize = AsyncMock()
        mock_pool.close = AsyncMock()

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        mock_health_srv = MagicMock()
        mock_health_srv.start_background = MagicMock()
        mock_health_srv.stop = MagicMock()

        mock_config = MagicMock()
        mock_config.postgres_host = "localhost:5432"
        mock_config.postgres_database = "test"
        mock_config.postgres_username = "user"
        mock_config.postgres_password = "pass"
        mock_config.api_base_url = "http://localhost:8004"
        mock_config.redis_host = "redis://localhost"
        mock_config.schedule_hours = 24
        mock_config.milestone_years = [25, 50]

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("Redis down"))
        mock_redis.aclose = AsyncMock(side_effect=RuntimeError("close also failed"))

        async def fake_scheduler(*_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(100)

        fake_app = FastAPI()

        with (
            patch.object(_module, "setup_logging"),
            patch.object(_module.InsightsConfig, "from_env", return_value=mock_config),
            patch.object(_module, "HealthServer", return_value=mock_health_srv),
            patch.object(_module, "AsyncPostgreSQLPool", return_value=mock_pool),
            patch("httpx.AsyncClient", return_value=mock_http_client),
            patch("redis.asyncio.from_url", new_callable=AsyncMock, return_value=mock_redis),
            patch.object(_module, "_scheduler_loop", side_effect=fake_scheduler),
        ):
            async with _module.lifespan(fake_app):
                # Must not raise — the service still falls back cleanly.
                assert _module._redis is None
                assert _module._cache is None


# ============================================================
# 503 "not ready" responses when _pool is None
# ============================================================


@pytest.fixture
def test_client_no_pool() -> TestClient:
    """Create a test client with _pool set to None (service not ready)."""
    import insights.insights as _module

    _module._pool = None
    _module._cache = None

    from insights.insights import app

    return TestClient(app)


class TestServiceNotReadyResponses:
    """All data endpoints must return 503 when the pool is not initialized."""

    def test_top_artists_503_when_no_pool(self, test_client_no_pool: TestClient) -> None:
        response = test_client_no_pool.get("/api/insights/top-artists")
        assert response.status_code == 503
        assert "error" in response.json()

    def test_genre_trends_503_when_no_pool(self, test_client_no_pool: TestClient) -> None:
        response = test_client_no_pool.get("/api/insights/genre-trends?genre=Rock")
        assert response.status_code == 503
        assert "error" in response.json()

    def test_label_longevity_503_when_no_pool(self, test_client_no_pool: TestClient) -> None:
        response = test_client_no_pool.get("/api/insights/label-longevity")
        assert response.status_code == 503
        assert "error" in response.json()

    def test_this_month_503_when_no_pool(self, test_client_no_pool: TestClient) -> None:
        response = test_client_no_pool.get("/api/insights/this-month")
        assert response.status_code == 503
        assert "error" in response.json()

    def test_data_completeness_503_when_no_pool(self, test_client_no_pool: TestClient) -> None:
        response = test_client_no_pool.get("/api/insights/data-completeness")
        assert response.status_code == 503
        assert "error" in response.json()

    def test_status_503_when_no_pool(self, test_client_no_pool: TestClient) -> None:
        response = test_client_no_pool.get("/api/insights/status")
        assert response.status_code == 503
        assert "error" in response.json()


# Every read endpoint that caches, with the cache key it uses.
_CACHED_ENDPOINTS = [
    ("/api/insights/top-artists?limit=10", "insights:top-artists:10"),
    ("/api/insights/genre-trends?genre=Rock", "insights:genre-trends:Rock"),
    ("/api/insights/label-longevity?limit=10", "insights:label-longevity:10"),
    ("/api/insights/release-rarity?limit=10", "insights:release-rarity:10"),
    ("/api/insights/data-completeness", "insights:data-completeness"),
]


class TestCacheGenerationThreading:
    """Every read endpoint must write back to the
    generation it read from, so a request straddling a recompute cannot
    re-cache pre-update data over freshly computed results.
    """

    @pytest.mark.parametrize(("url", "cache_key"), _CACHED_ENDPOINTS)
    def test_get_and_set_use_the_same_generation(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
        url: str,
        cache_key: str,
    ) -> None:
        mock_cache.get.return_value = None

        response = test_client_with_cache.get(url)

        assert response.status_code == 200
        assert mock_cache.get.call_args[0] == (cache_key, TEST_CACHE_GENERATION)
        assert mock_cache.set.call_args[0][0] == cache_key
        assert mock_cache.set.call_args[0][2] == TEST_CACHE_GENERATION, "set() must use the generation captured before the DB read, not a re-read one"

    @pytest.mark.parametrize("url", [url for url, _ in _CACHED_ENDPOINTS])
    def test_generation_is_read_before_the_cache_lookup(
        self,
        test_client_with_cache: TestClient,
        mock_cache: AsyncMock,
        url: str,
    ) -> None:
        """Ordering guard: generation() must precede get() (and therefore the DB read)."""
        calls: list[str] = []
        mock_cache.generation = AsyncMock(side_effect=lambda: calls.append("generation") or TEST_CACHE_GENERATION)
        mock_cache.get = AsyncMock(side_effect=lambda *_a, **_k: calls.append("get"))
        mock_cache.set = AsyncMock(side_effect=lambda *_a, **_k: calls.append("set"))

        response = test_client_with_cache.get(url)

        assert response.status_code == 200
        assert calls == ["generation", "get", "set"]
