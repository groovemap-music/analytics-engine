"""Valkey cache for precomputed insight results.

Uses cache-aside (lazy loading) with a **generation-versioned key namespace**.
All operations are safe — Valkey failures fall through silently so the service
can always fall back to PostgreSQL.

Why a generation, not a plain DELETE
------------------------------------
Cache-aside is a read-then-populate pair with an unbounded gap between the two
(the endpoint reads Postgres, exits the connection context — a real cooperative
scheduling yield — then serialises and writes to Valkey). The scheduler
concurrently does DELETE+INSERT in a transaction and then invalidates.

With a plain SCAN+DELETE invalidation the two can interleave as:

1. endpoint reads the OLD rows from Postgres,
2. scheduler commits the NEW rows and invalidates the cache,
3. endpoint's ``set`` lands — writing the stale result *after* the invalidation.

That stale entry then survives with TTL = ``schedule_hours`` (24h by default)
and is served to every subsequent request until the *next* cycle's
invalidation, masking freshly computed data for a full day.

Versioning the namespace removes the race instead of narrowing it. Every key is
written as ``insights:g{generation}:...``, and the scheduler bumps the
generation on invalidation. A request captures the generation *before* its
database read, so a request that straddles a recompute writes into the OLD
generation's namespace — which nobody reads any more — while readers arriving
after the bump miss cleanly and repopulate from the new data. There is no
ordering requirement between the endpoint's ``set`` and the scheduler's
invalidation at all.
"""

import json
import re
from typing import Any

import structlog
from common.config import valkey_key, valkey_key_prefix, valkey_pattern

from insights.telemetry import record_cache_read


logger = structlog.get_logger(__name__)

# A cached response body: an object, or a bare list for the endpoints that return one.
type CachedValue = dict[str, Any] | list[Any]

# Every key and scan pattern below goes through ``valkey_key`` / ``valkey_pattern``
# so it lives under the configured ``VALKEY_KEY_PREFIX`` (default ``groovemap:``).
# The names here are the unprefixed suffixes; SCAN returns keys that already carry
# the prefix, so they are never prefixed a second time.

# The monotonically increasing cache generation.
#
# Deliberately OUTSIDE the "insights:g*" sweep: invalidate_all scans and
# deletes that namespace, and deleting the generation counter would roll the
# namespace back to g0 and resurrect keys from an earlier generation.
GENERATION_KEY_NAME = "insights-generation"

# Only generation-scoped keys belong to this cache. Scanning "insights:*" would
# also sweep the API service's own "insights:data-completeness" entry (a 6h
# cache) when both services share a Valkey. The glob "insights:g*" is still wider
# than that ("insights:genre-..."), so matches are re-checked against the regex.
_VERSIONED_KEY_GLOB = "insights:g*"
_VERSIONED_KEY_RE = re.compile(r"insights:g\d+:")


def generation_key() -> str:
    """Return the prefixed key holding the cache generation counter."""
    return valkey_key(GENERATION_KEY_NAME)


class InsightsCache:
    """Valkey cache for precomputed insight results, keyed by generation."""

    def __init__(self, valkey: Any, ttl_seconds: int) -> None:
        self._valkey = valkey
        self._ttl = ttl_seconds

    @staticmethod
    def versioned_key(key: str, generation: int) -> str:
        """Return ``key`` rewritten into ``generation``'s namespace.

        Accepts keys with or without the legacy ``insights:`` prefix so call
        sites can keep using their existing cache-key strings.
        """
        suffix = key.removeprefix("insights:")
        return valkey_key("insights", f"g{generation}", suffix)

    async def generation(self) -> int:
        """Return the current cache generation.

        Callers MUST read this *before* their database read, and pass the same
        value to both :meth:`get` and :meth:`set`, so a request that straddles a
        recompute writes to the generation it actually read from.

        Returns ``0`` when the counter is unset or Valkey is unavailable.
        """
        try:
            raw = await self._valkey.get(generation_key())
            if raw is None:
                return 0
            return int(raw)
        except Exception:
            logger.debug("⚠️ Cache generation read failed, using generation 0")
            return 0

    async def get(self, key: str, generation: int) -> CachedValue | None:
        """Get a cached value from ``generation``. Returns None on miss or Valkey error.

        Records ``groovemap.api.cache`` for every read; a Valkey error counts as a miss,
        matching this method's own fall-through-to-database behavior.
        """
        try:
            raw = await self._valkey.get(self.versioned_key(key, generation))
            if raw is None:
                record_cache_read(hit=False)
                return None
            result: CachedValue = json.loads(raw)
            record_cache_read(hit=True)
            return result
        except Exception:
            logger.debug("⚠️ Cache get failed, falling through to database", key=key)
            record_cache_read(hit=False)
            return None

    async def set(self, key: str, value: CachedValue, generation: int) -> None:
        """Cache a value in ``generation`` with TTL. Silently fails if Valkey is down.

        Writing into a superseded generation is harmless by construction: those
        keys are unreachable and expire on their own.
        """
        try:
            await self._valkey.set(
                self.versioned_key(key, generation),
                json.dumps(value, default=str),
                ex=self._ttl,
            )
        except Exception:
            logger.debug("⚠️ Cache set failed", key=key)

    async def invalidate_all(self) -> None:
        """Retire the current generation and reclaim its keys.

        Bumping the generation is what actually invalidates: it happens
        atomically (INCR) and takes effect for every subsequent reader
        immediately. The SCAN+DELETE that follows is only housekeeping to free
        memory sooner than the TTL would; keys belonging to the new generation
        are deliberately skipped, since a request may already have populated
        them from freshly committed data.
        """
        try:
            generation = int(await self._valkey.incr(generation_key()))
        except Exception:
            logger.warning("⚠️ Cache generation bump failed — cache may serve stale data until TTL")
            return

        logger.info("🔄 Cache generation advanced", generation=generation)

        current_prefix = valkey_key("insights", f"g{generation}") + ":"
        try:
            cursor: str | int = "0"
            deleted = 0
            while True:
                cursor, keys = await self._valkey.scan(cursor=int(cursor), match=valkey_pattern(_VERSIONED_KEY_GLOB), count=100)
                stale = [k for k in keys if _is_versioned(_as_str(k)) and not _as_str(k).startswith(current_prefix)]
                if stale:
                    await self._valkey.delete(*stale)
                    deleted += len(stale)
                if int(cursor) == 0:
                    break
            if deleted:
                logger.info("🔄 Cache invalidated", keys_deleted=deleted, generation=generation)
        except Exception:
            # Non-fatal: the generation bump already invalidated the cache.
            logger.debug("⚠️ Cache key reclamation failed")

    async def purge_legacy_keys(self) -> int:
        """Delete keys this service wrote before key prefixes existed; return how many.

        Only for a stack that previously ran unprefixed on its own dedicated Valkey,
        enabled explicitly through ``VALKEY_LEGACY_KEYS=migrate``. The prefixed namespace
        starts empty with its own generation counter, so nothing stale is ever served;
        what the old keys would leave behind is memory: the generation counter has no
        TTL, and on a ``noeviction`` instance it would stay forever.

        Touches only the exact known legacy shapes: the ``insights-generation`` counter
        and ``insights:g<digits>:*`` entries (which carry a TTL and would age out on
        their own). Never runs without a prefix, because then the "legacy" names are
        the live ones.
        """
        if not valkey_key_prefix():
            logger.warning("⚠️ Legacy key cleanup skipped: VALKEY_KEY_PREFIX is empty, unprefixed keys are the live ones")
            return 0
        deleted = 0
        try:
            deleted += int(await self._valkey.delete(GENERATION_KEY_NAME))
            cursor: str | int = "0"
            while True:
                cursor, keys = await self._valkey.scan(cursor=int(cursor), match=_VERSIONED_KEY_GLOB, count=100)
                legacy = [k for k in keys if _VERSIONED_KEY_RE.match(_as_str(k))]
                if legacy:
                    deleted += int(await self._valkey.delete(*legacy))
                if int(cursor) == 0:
                    break
        except Exception:
            logger.warning("⚠️ Legacy key cleanup failed", deleted=deleted)
            return deleted
        logger.info("🧹 Legacy unprefixed cache keys removed", keys_deleted=deleted)
        return deleted


def _is_versioned(key: str) -> bool:
    """Whether ``key`` (as returned by SCAN, prefix included) is a generation-scoped cache key."""
    return _VERSIONED_KEY_RE.match(key, len(valkey_key_prefix())) is not None


def _as_str(key: Any) -> str:
    """Normalise a Valkey key to str (clients may return bytes)."""
    return key.decode() if isinstance(key, bytes) else str(key)
