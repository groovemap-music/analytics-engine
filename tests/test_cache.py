"""Tests for insights cache module."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fakeredis import FakeAsyncValkey

from insights.cache import GENERATION_KEY_NAME, InsightsCache, generation_key


@pytest.fixture
def mock_valkey() -> AsyncMock:
    """Create a mock Valkey client (for error-path tests)."""
    valkey = AsyncMock()
    valkey.get = AsyncMock(return_value=None)
    valkey.set = AsyncMock()
    valkey.incr = AsyncMock(return_value=1)
    valkey.scan = AsyncMock(return_value=(0, []))
    valkey.delete = AsyncMock()
    return valkey


@pytest.fixture
def cache(mock_valkey: AsyncMock) -> InsightsCache:
    """Create an InsightsCache with mock Valkey."""
    return InsightsCache(mock_valkey, ttl_seconds=3600)


@pytest.fixture
def real_valkey() -> Any:
    """A real (in-memory) Valkey, so generation semantics are exercised for real."""
    return FakeAsyncValkey(decode_responses=True)


@pytest.fixture
def real_cache(real_valkey: Any) -> InsightsCache:
    return InsightsCache(real_valkey, ttl_seconds=3600)


class TestVersionedKey:
    def test_namespaces_by_generation(self) -> None:
        assert InsightsCache.versioned_key("insights:top-artists:10", 3) == "groovemap:insights:g3:top-artists:10"

    def test_accepts_keys_without_the_legacy_prefix(self) -> None:
        assert InsightsCache.versioned_key("top-artists:10", 3) == "groovemap:insights:g3:top-artists:10"

    def test_generation_key_is_outside_the_scanned_namespace(self) -> None:
        """invalidate_all deletes insights:g* — the counter must not be swept with it."""
        assert not GENERATION_KEY_NAME.startswith("insights:g")
        assert generation_key() == "groovemap:insights-generation"


class TestCacheGet:
    @pytest.mark.asyncio
    async def test_returns_none_on_miss(self, cache: InsightsCache) -> None:
        assert await cache.get("insights:top-artists:10", 0) is None

    @pytest.mark.asyncio
    async def test_returns_cached_value_on_hit(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.get.return_value = '{"items": [1, 2, 3], "count": 3}'
        assert await cache.get("insights:top-artists:10", 0) == {"items": [1, 2, 3], "count": 3}

    @pytest.mark.asyncio
    async def test_reads_from_the_requested_generation(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        await cache.get("insights:top-artists:10", 7)
        mock_valkey.get.assert_awaited_once_with("groovemap:insights:g7:top-artists:10")

    @pytest.mark.asyncio
    async def test_returns_none_on_valkey_error(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.get.side_effect = ConnectionError("Valkey down")
        assert await cache.get("insights:top-artists:10", 0) is None


class TestCacheGetTelemetry:
    """`groovemap.api.cache` must record one outcome per read call, never more or fewer."""

    @pytest.mark.asyncio
    async def test_records_a_hit(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.get.return_value = '{"items": [], "count": 0}'
        with patch("insights.cache.record_cache_read") as mock_record:
            await cache.get("insights:top-artists:10", 0)
        mock_record.assert_called_once_with(hit=True)

    @pytest.mark.asyncio
    async def test_records_a_miss(self, cache: InsightsCache) -> None:
        with patch("insights.cache.record_cache_read") as mock_record:
            await cache.get("insights:top-artists:10", 0)
        mock_record.assert_called_once_with(hit=False)

    @pytest.mark.asyncio
    async def test_records_a_miss_on_valkey_error(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.get.side_effect = ConnectionError("Valkey down")
        with patch("insights.cache.record_cache_read") as mock_record:
            await cache.get("insights:top-artists:10", 0)
        mock_record.assert_called_once_with(hit=False)


class TestCacheSet:
    @pytest.mark.asyncio
    async def test_stores_value_with_ttl_in_the_generation_namespace(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        await cache.set("insights:top-artists:10", {"items": [], "count": 0}, 2)
        mock_valkey.set.assert_called_once()
        call_args = mock_valkey.set.call_args
        assert call_args[0][0] == "groovemap:insights:g2:top-artists:10"
        assert call_args[1]["ex"] == 3600

    @pytest.mark.asyncio
    async def test_silently_fails_on_valkey_error(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.set.side_effect = ConnectionError("Valkey down")
        await cache.set("insights:top-artists:10", {"items": []}, 0)  # must not raise


class TestCacheGeneration:
    @pytest.mark.asyncio
    async def test_starts_at_zero(self, real_cache: InsightsCache) -> None:
        assert await real_cache.generation() == 0

    @pytest.mark.asyncio
    async def test_advances_on_invalidation(self, real_cache: InsightsCache) -> None:
        await real_cache.invalidate_all()
        assert await real_cache.generation() == 1
        await real_cache.invalidate_all()
        assert await real_cache.generation() == 2

    @pytest.mark.asyncio
    async def test_falls_back_to_zero_on_valkey_error(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.get.side_effect = ConnectionError("Valkey down")
        assert await cache.generation() == 0


class TestCacheAsideRace:
    """A request straddling a recompute must not re-cache stale data.

    Interleaving that used to poison the cache for a full schedule interval:
      1. endpoint reads OLD rows from Postgres,
      2. scheduler commits NEW rows and invalidates,
      3. endpoint's set() lands — writing the stale result AFTER the invalidation.
    """

    @pytest.mark.asyncio
    async def test_set_after_invalidation_is_not_served(self, real_cache: InsightsCache) -> None:
        # 1. The request captures the generation before its database read.
        generation = await real_cache.generation()
        stale = {"items": ["yesterday's rankings"], "count": 1}

        # 2. The scheduler finishes its cycle and invalidates.
        await real_cache.invalidate_all()

        # 3. The in-flight request's set finally lands — into the retired generation.
        await real_cache.set("insights:top-artists:100", stale, generation)

        # A subsequent request reads the CURRENT generation and must miss,
        # falling through to the freshly computed Postgres rows.
        current = await real_cache.generation()
        assert current != generation
        assert await real_cache.get("insights:top-artists:100", current) is None

    @pytest.mark.asyncio
    async def test_normal_cache_aside_still_hits(self, real_cache: InsightsCache) -> None:
        """The fix must not break the non-racing path."""
        generation = await real_cache.generation()
        payload = {"items": [1, 2, 3], "count": 3}
        await real_cache.set("insights:top-artists:100", payload, generation)
        assert await real_cache.get("insights:top-artists:100", await real_cache.generation()) == payload

    @pytest.mark.asyncio
    async def test_entries_written_after_the_bump_are_preserved(self, real_cache: InsightsCache) -> None:
        """Housekeeping must not delete keys from the generation now being served."""
        await real_cache.invalidate_all()
        generation = await real_cache.generation()
        payload = {"items": ["fresh"], "count": 1}
        await real_cache.set("insights:top-artists:100", payload, generation)
        assert await real_cache.get("insights:top-artists:100", generation) == payload

    @pytest.mark.asyncio
    async def test_every_generation_is_distinct(self, real_cache: InsightsCache) -> None:
        seen = set()
        for _ in range(3):
            seen.add(await real_cache.generation())
            await real_cache.invalidate_all()
        assert len(seen) == 3


class TestCacheInvalidateAll:
    @pytest.mark.asyncio
    async def test_reclaims_superseded_generation_keys(self, real_cache: InsightsCache, real_valkey: Any) -> None:
        generation = await real_cache.generation()
        await real_cache.set("insights:top-artists:10", {"a": 1}, generation)
        await real_cache.set("insights:genre-trends:Rock", {"b": 2}, generation)

        await real_cache.invalidate_all()

        assert await real_valkey.get(f"groovemap:insights:g{generation}:top-artists:10") is None
        assert await real_valkey.get(f"groovemap:insights:g{generation}:genre-trends:Rock") is None

    @pytest.mark.asyncio
    async def test_does_not_touch_unversioned_insights_keys(self, real_cache: InsightsCache, real_valkey: Any) -> None:
        """The API service caches insights:data-completeness on a possibly-shared Valkey."""
        await real_valkey.set("insights:data-completeness", "api-owned")

        await real_cache.invalidate_all()

        assert await real_valkey.get("insights:data-completeness") == "api-owned"

    @pytest.mark.asyncio
    async def test_preserves_the_generation_counter(self, real_cache: InsightsCache, real_valkey: Any) -> None:
        await real_cache.invalidate_all()
        await real_cache.invalidate_all()
        assert await real_valkey.get(generation_key()) == "2"

    @pytest.mark.asyncio
    async def test_handles_no_keys(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.scan.return_value = (0, [])
        await cache.invalidate_all()
        mock_valkey.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_multiple_scan_pages(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        # incr returns 1, so "groovemap:insights:g1:" is the live generation and is skipped.
        mock_valkey.scan.side_effect = [
            ("42", ["groovemap:insights:g0:key1"]),
            ("0", ["groovemap:insights:g0:key2"]),
        ]
        await cache.invalidate_all()
        assert mock_valkey.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_bytes_keys_from_the_client(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.scan.side_effect = [(0, [b"groovemap:insights:g0:key1"])]
        await cache.invalidate_all()
        mock_valkey.delete.assert_called_once_with(b"groovemap:insights:g0:key1")

    @pytest.mark.asyncio
    async def test_scan_failure_does_not_raise(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.scan.side_effect = ConnectionError("Valkey down")
        await cache.invalidate_all()  # must not raise

    @pytest.mark.asyncio
    async def test_incr_failure_does_not_raise(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.incr.side_effect = ConnectionError("Valkey down")
        await cache.invalidate_all()  # must not raise
        mock_valkey.delete.assert_not_called()


FOREIGN_KEYS = {
    "insights:data-completeness": "api-owned",
    "insights:genre-trends:Rock": "other-app",
    "insights:generic": "other-app",
    "insights-generation-other": "other-app",
    "other-app:insights:g1:top-artists:10": "other-app",
    "groovemap:catalog:something": "catalog-owned",
}


async def _seed_legacy(valkey: Any) -> None:
    """What an unprefixed stack leaves behind: a TTL-less counter plus TTL'd generation keys."""
    await valkey.set("insights-generation", "7")
    await valkey.set("insights:g7:top-artists:10", '{"stale": true}', ex=3600)
    await valkey.set("insights:g6:data-completeness", '{"stale": true}', ex=3600)


class TestUpgradeFromUnprefixedStack:
    @pytest.mark.asyncio
    async def test_legacy_entries_are_never_served_and_generations_do_not_collide(self, real_cache: InsightsCache, real_valkey: Any) -> None:
        await _seed_legacy(real_valkey)

        generation = await real_cache.generation()
        assert generation == 0
        assert await real_cache.get("insights:top-artists:10", 7) is None
        assert await real_cache.get("insights:top-artists:10", generation) is None

        await real_cache.invalidate_all()
        assert await real_cache.generation() == 1
        # The prefixed counter is independent: the legacy one is untouched without opt-in.
        assert await real_valkey.get("insights-generation") == "7"
        assert await real_valkey.get("insights:g7:top-artists:10") == '{"stale": true}'

    @pytest.mark.asyncio
    async def test_sweep_never_touches_unprefixed_keys(self, real_cache: InsightsCache, real_valkey: Any) -> None:
        await _seed_legacy(real_valkey)
        await real_cache.invalidate_all()
        assert await real_valkey.get("insights:g6:data-completeness") is not None

    @pytest.mark.asyncio
    async def test_purge_removes_exactly_the_legacy_shapes(self, real_cache: InsightsCache, real_valkey: Any) -> None:
        await _seed_legacy(real_valkey)
        for key, value in FOREIGN_KEYS.items():
            await real_valkey.set(key, value)
        await real_cache.set("insights:top-artists:10", {"fresh": 1}, 0)

        assert await real_cache.purge_legacy_keys() == 3

        assert sorted(await real_valkey.keys("*")) == sorted([*FOREIGN_KEYS, "groovemap:insights:g0:top-artists:10"])
        for key, value in FOREIGN_KEYS.items():
            assert await real_valkey.get(key) == value
        assert await real_cache.get("insights:top-artists:10", 0) == {"fresh": 1}

    @pytest.mark.asyncio
    async def test_purge_is_idempotent(self, real_cache: InsightsCache, real_valkey: Any) -> None:
        await _seed_legacy(real_valkey)
        assert await real_cache.purge_legacy_keys() == 3
        assert await real_cache.purge_legacy_keys() == 0

    @pytest.mark.asyncio
    async def test_purge_refuses_when_prefix_is_disabled(self, real_cache: InsightsCache, real_valkey: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no prefix the "legacy" names are the live ones."""
        monkeypatch.setenv("VALKEY_KEY_PREFIX", "")
        await _seed_legacy(real_valkey)
        assert await real_cache.purge_legacy_keys() == 0
        assert await real_valkey.get("insights-generation") == "7"

    @pytest.mark.asyncio
    async def test_purge_failure_is_non_fatal(self, cache: InsightsCache, mock_valkey: AsyncMock) -> None:
        mock_valkey.delete.side_effect = ConnectionError("down")
        assert await cache.purge_legacy_keys() == 0


class TestSharedValkey:
    @pytest.mark.asyncio
    async def test_foreign_keys_survive_sweeps_and_are_not_read(self, real_cache: InsightsCache, real_valkey: Any) -> None:
        for key, value in FOREIGN_KEYS.items():
            await real_valkey.set(key, value)
        await real_cache.set("insights:top-artists:10", {"a": 1}, 0)

        await real_cache.invalidate_all()
        await real_cache.invalidate_all()

        for key, value in FOREIGN_KEYS.items():
            assert await real_valkey.get(key) == value
        assert await real_cache.get("insights:data-completeness", 0) is None

    @pytest.mark.asyncio
    async def test_sweep_ignores_foreign_keys_inside_the_prefix(self, real_cache: InsightsCache, real_valkey: Any) -> None:
        """ "insights:g*" also matches "insights:genre-...", which another service could own."""
        await real_valkey.set("groovemap:insights:genre-trends:Rock", "other-app")
        await real_cache.invalidate_all()
        assert await real_valkey.get("groovemap:insights:genre-trends:Rock") == "other-app"

    @pytest.mark.asyncio
    async def test_custom_prefix_isolates_two_stacks(self, real_valkey: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = InsightsCache(real_valkey, ttl_seconds=3600)
        monkeypatch.setenv("VALKEY_KEY_PREFIX", "stack-a")
        await cache.set("insights:top-artists:10", {"a": 1}, 0)
        monkeypatch.setenv("VALKEY_KEY_PREFIX", "stack-b")
        assert await cache.get("insights:top-artists:10", 0) is None
        await cache.invalidate_all()
        monkeypatch.setenv("VALKEY_KEY_PREFIX", "stack-a")
        assert await cache.get("insights:top-artists:10", 0) == {"a": 1}


class TestFreshStack:
    @pytest.mark.asyncio
    async def test_every_key_lives_under_the_prefix(self, real_cache: InsightsCache, real_valkey: Any) -> None:
        generation = await real_cache.generation()
        await real_cache.set("insights:top-artists:10", {"a": 1}, generation)
        await real_cache.invalidate_all()
        await real_cache.set("insights:top-artists:10", {"a": 2}, 1)

        keys = await real_valkey.keys("*")
        assert keys
        assert all(key.startswith("groovemap:") for key in keys)
        assert await real_valkey.get("groovemap:insights-generation") == "1"
        assert await real_cache.purge_legacy_keys() == 0

    @pytest.mark.asyncio
    async def test_disabled_prefix_keeps_the_original_key_names(
        self, real_cache: InsightsCache, real_valkey: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VALKEY_KEY_PREFIX", "")
        await real_cache.set("insights:top-artists:10", {"a": 1}, 0)
        await real_cache.invalidate_all()
        assert await real_valkey.get("insights-generation") == "1"
        assert await real_valkey.get("insights:g0:top-artists:10") is None


class TestNoUnprefixedKeyLiterals:
    """Guard: keys reach the Valkey client only through valkey_key/valkey_pattern (or a SCAN result)."""

    KEY_METHODS = frozenset({"get", "set", "delete", "incr", "decr", "expire", "exists", "scan", "scan_iter", "keys", "ttl", "unlink", "getdel"})

    def test_package_passes_no_literal_keys_to_the_client(self) -> None:
        import ast
        from pathlib import Path

        import insights

        offenders: list[str] = []
        for path in Path(insights.__file__).parent.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in self.KEY_METHODS):
                    continue
                if "valkey" not in ast.unparse(node.func.value).lower():
                    continue
                args = [*node.args, *(kw.value for kw in node.keywords if kw.arg in {"match", "name"})]
                offenders += [
                    f"{path.name}:{node.lineno}: {ast.unparse(arg)}"
                    for arg in args
                    if isinstance(arg, ast.Constant | ast.JoinedStr) and not isinstance(getattr(arg, "value", ""), int)
                ]
        assert offenders == []


class TestStartupLegacyCleanup:
    @staticmethod
    async def _start(real_valkey: Any, mode: str) -> None:
        from types import SimpleNamespace

        import insights.insights as module

        config = SimpleNamespace(valkey_url="valkey://x/0", schedule_hours=24, valkey_legacy_keys=mode)
        with patch("valkey.asyncio.from_url", new_callable=AsyncMock, return_value=real_valkey):
            await module._initialize_cache(config)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_read_leaves_legacy_keys(self, real_valkey: Any) -> None:
        await _seed_legacy(real_valkey)
        await self._start(real_valkey, "read")
        assert await real_valkey.get("insights-generation") == "7"

    @pytest.mark.asyncio
    async def test_migrate_removes_legacy_keys_only(self, real_valkey: Any) -> None:
        await _seed_legacy(real_valkey)
        await real_valkey.set("insights:data-completeness", "api-owned")
        await self._start(real_valkey, "migrate")
        assert sorted(await real_valkey.keys("*")) == ["insights:data-completeness"]
