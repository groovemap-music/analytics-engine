"""Tests for the monthly similar-artist batch: spool, memory guard, COPY, publish/retire.

Every vector and id here is synthetic. No provider data, and nothing derived from it, may
appear in this repository (ADR 0013, data rights).
"""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from insights import similar_artists as sa
from insights.embeddings.exact_top_k import exact_top_k


if TYPE_CHECKING:
    from pathlib import Path


def _vectors(n: int = 500, dim: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float16)


def _expected(vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    blocks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    exact_top_k(vectors, k, block_rows=64, threads=2, on_block=lambda lo, s, p: blocks.__setitem__(lo, (p, s)))
    return np.concatenate([blocks[lo][0] for lo in sorted(blocks)]), np.concatenate([blocks[lo][1] for lo in sorted(blocks)])


def _counting(calls: list[int]) -> Any:
    def peak() -> int:
        calls.append(1)
        return 1

    return peak


def _quiet_guard(**overrides: Any) -> sa.MemoryGuard:
    fields: dict[str, Any] = {"peak_rss": lambda: 1, "memorystatus_level": lambda: 90, "sleep": lambda _s: None}
    fields.update(overrides)
    return sa.MemoryGuard(**fields)


# ── Memory guard ──────────────────────────────────────────────────────────────────────────────


class TestMemoryGuard:
    def test_raises_once_the_getrusage_peak_passes_the_budget(self) -> None:
        guard = _quiet_guard(budget_bytes=1000, peak_rss=lambda: 1001)
        with pytest.raises(sa.MemoryBudgetExceededError, match="over the"):
            guard.check("block 3/9")

    def test_a_peak_at_the_budget_passes(self) -> None:
        _quiet_guard(budget_bytes=1000, peak_rss=lambda: 1000).check()

    def test_pauses_while_the_host_is_under_pressure_then_resumes(self) -> None:
        levels = iter([10, 20, 60])
        sleeps: list[float] = []
        guard = _quiet_guard(memorystatus_level=lambda: next(levels), sleep=sleeps.append, poll_interval_s=5.0)
        guard.check()
        assert sleeps == [5.0, 5.0]

    def test_no_memorystatus_sysctl_means_no_pause(self) -> None:
        sleeps: list[float] = []
        _quiet_guard(memorystatus_level=lambda: None, sleep=sleeps.append).check()
        assert sleeps == []

    def test_peak_rss_is_the_monotonic_getrusage_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Usage:
            ru_maxrss = 3

        monkeypatch.setattr(sa.resource, "getrusage", lambda _who: _Usage())
        monkeypatch.setattr(sa.sys, "platform", "linux")
        assert sa.peak_rss_bytes() == 3 * 1024
        monkeypatch.setattr(sa.sys, "platform", "darwin")
        assert sa.peak_rss_bytes() == 3

    def test_estimate_fits_the_full_catalog_in_the_budget(self) -> None:
        estimate = sa.estimate_peak_bytes(9_366_416, 128)
        assert 8e9 < estimate < sa.DEFAULT_MEMORY_BUDGET_BYTES


# ── Spool ─────────────────────────────────────────────────────────────────────────────────────


class TestComputeToSpool:
    def test_spool_holds_the_exact_lists(self, tmp_path: Path) -> None:
        vectors = _vectors()
        spool = sa.compute_to_spool(vectors, tmp_path, model_version="m1", k=10, block_rows=64, threads=2, guard=_quiet_guard())
        positions, scores = spool.read(0, len(vectors))
        expected_positions, expected_scores = _expected(vectors, 10)

        np.testing.assert_array_equal(positions, expected_positions)
        np.testing.assert_array_equal(scores, expected_scores)
        assert json.loads(spool.meta_path.read_text())["complete"] is True
        assert not spool.checkpoint_path.exists()

    def test_checks_the_guard_after_every_block(self, tmp_path: Path) -> None:
        calls: list[int] = []
        guard = _quiet_guard(peak_rss=_counting(calls))
        sa.compute_to_spool(_vectors(300), tmp_path, model_version="m1", k=5, block_rows=64, threads=1, guard=guard)
        assert len(calls) == 1 + 5  # once before, then once per block (300 rows / 64)

    def test_over_budget_stops_the_run(self, tmp_path: Path) -> None:
        peaks = iter([1, 1, 10**12])
        guard = _quiet_guard(budget_bytes=10**9, peak_rss=lambda: next(peaks))
        with pytest.raises(sa.MemoryBudgetExceededError):
            sa.compute_to_spool(_vectors(300), tmp_path, model_version="m1", k=5, block_rows=64, threads=1, guard=guard)

    def test_resumes_from_a_checkpoint_to_the_same_lists(self, tmp_path: Path) -> None:
        vectors = _vectors(450)
        ticks = iter(range(10**6))
        # A checkpoint after every block; the guard stops the run after block 4.
        blocks_seen = iter(range(100))
        stop_after = 4

        def peak() -> int:
            return 10**13 if next(blocks_seen) > stop_after else 1

        with pytest.raises(sa.MemoryBudgetExceededError):
            sa.compute_to_spool(
                vectors,
                tmp_path,
                model_version="m1",
                k=8,
                block_rows=64,
                threads=2,
                checkpoint_every_s=0,
                guard=_quiet_guard(peak_rss=peak),
                clock=lambda: next(ticks),
            )
        assert sa.Spool(tmp_path, 450, 8).checkpoint_path.exists()
        assert json.loads(sa.Spool(tmp_path, 450, 8).meta_path.read_text())["complete"] is False

        spool = sa.compute_to_spool(vectors, tmp_path, model_version="m1", k=8, block_rows=64, threads=2, guard=_quiet_guard())
        positions, scores = spool.read(0, 450)
        expected_positions, expected_scores = _expected(vectors, 8)
        np.testing.assert_array_equal(positions, expected_positions)
        np.testing.assert_array_equal(scores, expected_scores)

    def test_a_different_model_version_starts_over(self, tmp_path: Path) -> None:
        sa.compute_to_spool(_vectors(200, seed=1), tmp_path, model_version="m1", k=5, block_rows=64, threads=1, guard=_quiet_guard())
        vectors = _vectors(200, seed=2)
        spool = sa.compute_to_spool(vectors, tmp_path, model_version="m2", k=5, block_rows=64, threads=1, guard=_quiet_guard())
        np.testing.assert_array_equal(spool.read(0, 200)[0], _expected(vectors, 5)[0])

    def test_a_complete_spool_is_not_recomputed(self, tmp_path: Path) -> None:
        vectors = _vectors(200)
        sa.compute_to_spool(vectors, tmp_path, model_version="m1", k=5, block_rows=64, threads=1, guard=_quiet_guard())
        calls: list[int] = []
        sa.compute_to_spool(vectors, tmp_path, model_version="m1", k=5, block_rows=64, threads=1, guard=_quiet_guard(peak_rss=_counting(calls)))
        assert calls == []


# ── COPY ──────────────────────────────────────────────────────────────────────────────────────


class _FakeCopy:
    def __init__(self, sink: list[bytes]) -> None:
        self._sink = sink

    async def __aenter__(self) -> _FakeCopy:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def write(self, data: bytes) -> None:
        self._sink.append(data)


class _FakeCursor:
    def __init__(self, log: list[tuple[str, Any]], copied: list[bytes], fetchall_rows: list[tuple[Any, ...]] | None = None) -> None:
        self.log = log
        self.copied = copied
        self.fetchall_rows = fetchall_rows or []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        self.log.append((sql, params))

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self.fetchall_rows

    def copy(self, sql: str) -> _FakeCopy:
        self.log.append((sql, None))
        return _FakeCopy(self.copied)


class _FakeTransaction:
    def __init__(self, log: list[tuple[str, Any]]) -> None:
        self.log = log

    async def __aenter__(self) -> None:
        self.log.append(("BEGIN", None))

    async def __aexit__(self, exc_type: object, *_exc: object) -> bool:
        self.log.append(("ROLLBACK" if exc_type else "COMMIT", None))
        return False


class _FakeConnection:
    def __init__(self, fetchall_rows: list[tuple[Any, ...]] | None = None) -> None:
        self.log: list[tuple[str, Any]] = []
        self.copied: list[bytes] = []
        self.fetchall_rows = fetchall_rows

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self.log)

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.log, self.copied, self.fetchall_rows)


class TestWriteSimilarArtists:
    @pytest.mark.asyncio
    async def test_deletes_then_copies_every_entry_in_one_transaction(self, tmp_path: Path) -> None:
        vectors = _vectors(120)
        spool = sa.compute_to_spool(vectors, tmp_path, model_version="m1", k=4, block_rows=64, threads=1, guard=_quiet_guard())
        ids = [f"a{i}" for i in range(120)]
        conn = _FakeConnection()

        rows = await sa.write_similar_artists(conn, spool, model_version="m1", artist_ids=ids)

        assert rows == 120 * 4
        assert [entry[0] for entry in conn.log] == ["BEGIN", sa._DELETE_VERSION_SQL, sa._COPY_SQL, "COMMIT"]
        assert conn.log[1][1] == ("m1",)
        lines = b"".join(conn.copied).decode().splitlines()
        expected_positions, expected_scores = _expected(vectors, 4)
        first = lines[0].split("\t")
        assert first[:4] == ["a0", "m1", "1", f"a{expected_positions[0, 0]}"]
        assert np.float32(first[4]) == expected_scores[0, 0]  # round-trips exactly
        assert [line.split("\t")[2] for line in lines[:4]] == ["1", "2", "3", "4"]
        assert all(line.split("\t")[0] != line.split("\t")[3] for line in lines)

    @pytest.mark.asyncio
    async def test_skips_unfilled_slots(self, tmp_path: Path) -> None:
        spool = sa.compute_to_spool(_vectors(3), tmp_path, model_version="m1", k=5, block_rows=64, threads=1, guard=_quiet_guard())
        conn = _FakeConnection()
        assert await sa.write_similar_artists(conn, spool, model_version="m1", artist_ids=["x", "y", "z"]) == 3 * 2

    @pytest.mark.asyncio
    async def test_rejects_a_mismatched_id_list(self, tmp_path: Path) -> None:
        spool = sa.compute_to_spool(_vectors(10), tmp_path, model_version="m1", k=2, block_rows=64, threads=1, guard=_quiet_guard())
        with pytest.raises(ValueError, match="9 artist ids"):
            await sa.write_similar_artists(_FakeConnection(), spool, model_version="m1", artist_ids=["x"] * 9)


# ── Publish and rotate ───────────────────────────────────────────────────────────────────────


class _FakeRegistry:
    def __init__(self, *, publish_failures: int = 0, failing_retires: frozenset[str] = frozenset()) -> None:
        self.publish_failures = publish_failures
        self.failing_retires = failing_retires
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def publish(self, cursor: Any, model_version: str, **kwargs: Any) -> int:
        self.calls.append(("publish", model_version, kwargs))
        return self.publish_failures

    async def retire(self, cursor: Any, model_version: str, *, delete_release: bool = False) -> int:
        self.calls.append(("retire", model_version, {"delete_release": delete_release}))
        return 1 if model_version in self.failing_retires else 0


_PUBLISH_ARGS: dict[str, Any] = {"source_dump_id": "dump-9", "source_dump_date": date(2026, 9, 1), "k": 50, "artists": 1000}


class TestPublishAndRotate:
    @pytest.mark.asyncio
    async def test_publishes_then_retires_all_but_the_previous_release(self) -> None:
        conn = _FakeConnection(fetchall_rows=[("v9",), ("v8",), ("v7",), ("v6",)])
        registry = _FakeRegistry()

        result = await sa.publish_and_rotate(conn, registry, model_version="v9", **_PUBLISH_ARGS)

        assert registry.calls[0] == ("publish", "v9", _PUBLISH_ARGS)
        assert [(c[0], c[1]) for c in registry.calls[1:]] == [("retire", "v7"), ("retire", "v6")]
        assert all(c[2] == {"delete_release": False} for c in registry.calls[1:])  # lineage kept
        assert result == sa.RotateResult("v9", "v8", ("v7", "v6"), ())

    @pytest.mark.asyncio
    async def test_first_release_retires_nothing(self) -> None:
        conn = _FakeConnection(fetchall_rows=[("v1",)])
        registry = _FakeRegistry()
        result = await sa.publish_and_rotate(conn, registry, model_version="v1", **_PUBLISH_ARGS)
        assert result == sa.RotateResult("v1", None, (), ())
        assert [c[0] for c in registry.calls] == ["publish"]

    @pytest.mark.asyncio
    async def test_a_failed_publish_raises_and_retires_nothing(self) -> None:
        registry = _FakeRegistry(publish_failures=1)
        with pytest.raises(sa.PublishError, match="v9"):
            await sa.publish_and_rotate(_FakeConnection(fetchall_rows=[("v8",), ("v7",)]), registry, model_version="v9", **_PUBLISH_ARGS)
        assert [c[0] for c in registry.calls] == ["publish"]

    @pytest.mark.asyncio
    async def test_a_failed_retire_is_reported_not_raised(self) -> None:
        conn = _FakeConnection(fetchall_rows=[("v9",), ("v8",), ("v7",), ("v6",)])
        result = await sa.publish_and_rotate(conn, _FakeRegistry(failing_retires=frozenset({"v7"})), model_version="v9", **_PUBLISH_ARGS)
        assert result.retired == ("v6",)
        assert result.retire_failures == ("v7",)


class TestSchemaReleaseRegistry:
    @pytest.mark.asyncio
    async def test_delegates_to_the_schema_helpers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import groovemap_schema.postgres as schema

        seen: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

        async def publish(*args: Any, **kwargs: Any) -> int:
            seen.append(("publish", args, kwargs))
            return 0

        async def retire(*args: Any, **kwargs: Any) -> int:
            seen.append(("retire", args, kwargs))
            return 1

        monkeypatch.setattr(schema, "publish_artist_embedding_release", publish, raising=False)
        monkeypatch.setattr(schema, "retire_artist_similar_artists_version", retire, raising=False)
        registry = sa.SchemaReleaseRegistry()

        assert await registry.publish("cur", "v1", **_PUBLISH_ARGS) == 0
        assert await registry.retire("cur", "v0", delete_release=True) == 1
        assert seen == [("publish", ("cur", "v1"), _PUBLISH_ARGS), ("retire", ("cur", "v0"), {"delete_release": True})]
