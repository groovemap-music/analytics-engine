"""Slice mode and the read-only sizing pass, on synthetic vectors only."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from insights.embeddings.exact_top_k import PhaseProfile, TopKState, exact_top_k
from insights.similar_artists import MemoryGuard, compute_to_spool
from insights.similar_artists_sizing import (
    _assert_read_only,
    _resolve_model_version,
    extrapolate,
    load_catalog,
    run_sizing,
    sample_queries,
)
from tests.test_exact_top_k import _collect, _integer_vectors


if TYPE_CHECKING:
    from pathlib import Path


def _guard() -> MemoryGuard:
    return MemoryGuard(budget_bytes=1 << 40, memorystatus_level=lambda: None)


@pytest.mark.parametrize(("n", "queries", "block_rows", "threads"), [(700, 300, 128, 3), (700, 128, 128, 2), (513, 1, 64, 1), (900, 900, 256, 4)])
def test_slice_lists_equal_the_full_runs_first_rows(n: int, queries: int, block_rows: int, threads: int) -> None:
    # Exact-in-float32 vectors with many real ties: the slice must reproduce the full run's
    # lists position for position, not merely the same sets.
    vectors = _integer_vectors(n, 12, seed=n)
    full_scores, full_positions = _collect(vectors, 20, block_rows=block_rows, threads=threads)
    slice_scores, slice_positions = _collect(vectors, 20, block_rows=block_rows, threads=threads, query_rows=queries)
    assert slice_positions.shape == (queries, 20)
    np.testing.assert_array_equal(slice_positions, full_positions[:queries])
    np.testing.assert_array_equal(slice_scores, full_scores[:queries])


def test_slice_rejects_out_of_range_query_rows() -> None:
    vectors = np.ones((10, 4), dtype=np.float16)
    for bad in (0, 11):
        with pytest.raises(ValueError, match="query_rows"):
            exact_top_k(vectors, 3, query_rows=bad)


def test_slice_resumes_from_a_checkpoint() -> None:
    vectors = _integer_vectors(600, 12, seed=1)
    expected = _collect(vectors, 10, block_rows=128, threads=2, query_rows=400)
    saved: dict[int, TopKState] = {}

    def checkpoint(next_block: int, state: TopKState) -> None:
        if next_block == 2:
            saved[2] = TopKState(state.scores.copy(), state.positions.copy(), state.thresholds.copy())

    exact_top_k(vectors, 10, block_rows=128, threads=2, query_rows=400, after_block=checkpoint)
    resumed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    exact_top_k(
        vectors,
        10,
        block_rows=128,
        threads=2,
        query_rows=400,
        state=saved[2],
        start_block=2,
        on_block=lambda lo, s, p: resumed.__setitem__(lo, (s, p)),
    )
    assert sorted(resumed) == [256, 384]
    np.testing.assert_array_equal(np.concatenate([resumed[lo][1] for lo in sorted(resumed)]), expected[1][256:])


def test_profile_counts_blocks_and_phases() -> None:
    vectors = np.random.default_rng(3).standard_normal((300, 8)).astype(np.float16)
    sliced, full = PhaseProfile(), PhaseProfile()
    exact_top_k(vectors, 5, block_rows=64, threads=2, query_rows=128, profile=sliced)
    exact_top_k(vectors, 5, block_rows=64, threads=2, profile=full)
    assert sliced.tasks == 2 * 5  # 2 query blocks x 5 catalog blocks
    assert full.tasks == 5 * 6 // 2  # upper block triangle
    assert sliced.matmul_s > 0
    assert sliced.select_s > 0


def test_a_slice_spool_never_resumes_a_full_spool(tmp_path: Path) -> None:
    vectors = _integer_vectors(300, 12, seed=2)
    compute_to_spool(vectors, tmp_path, model_version="m", k=5, block_rows=64, threads=1, guard=_guard())
    spool = compute_to_spool(vectors, tmp_path, model_version="m", k=5, block_rows=64, threads=1, guard=_guard(), query_rows=100)
    assert spool.n_artists == 100
    assert json.loads(spool.meta_path.read_text())["identity"]["catalog_rows"] == 300
    assert spool.positions_path.stat().st_size == 100 * 5 * 4


def test_sample_is_seeded_distinct_sorted_and_inside_the_slice() -> None:
    first = sample_queries(1000, 200, seed=7)
    assert np.array_equal(first, sample_queries(1000, 200, seed=7))
    assert not np.array_equal(first, sample_queries(1000, 200, seed=8))
    assert len(set(first)) == 200
    assert (np.diff(first) > 0).all()
    assert first.min() >= 0 and first.max() < 1000
    assert len(sample_queries(50, 2000, seed=1)) == 50


def test_run_sizing_reports_phases_exactness_and_extrapolation(tmp_path: Path) -> None:
    vectors = np.random.default_rng(9).standard_normal((1500, 16)).astype(np.float16)
    report = run_sizing(vectors, tmp_path, slice_rows=500, validate=120, seed=5, k=50, block_rows=128, threads=2, guard=_guard(), read_seconds=1.5)
    assert report["writes_to_postgres"] is False
    assert report["exactness"]["passed"] is True
    assert report["exactness"]["queries"] == 120
    assert report["exactness"]["tie_tolerant_recall"] == 1.0
    assert set(report["phases"]) == {"read_embeddings_s", "compute_slice_s", "exactness_check_s"}
    assert set(report["peak_rss_bytes"]) == {"after_read", "after_compute", "after_exactness_check"}
    assert report["extrapolation"]["scale"] == pytest.approx(3.0)
    assert report["extrapolation"]["full_wall_s"] <= report["extrapolation"]["full_wall_upper_bound_s"]
    assert not list(tmp_path.iterdir()), "scratch spool must be removed"
    json.dumps(report)


def test_run_sizing_flags_a_corrupted_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import insights.similar_artists_sizing as sizing

    vectors = np.random.default_rng(4).standard_normal((600, 16)).astype(np.float16)
    real = sizing.compute_to_spool

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        spool = real(*args, **kwargs)
        positions = np.fromfile(spool.positions_path, dtype=np.int32).reshape(-1, spool.k)
        positions[:, 0] = positions[:, 5]  # the best neighbour replaced by a worse one, and duplicated
        positions.tofile(spool.positions_path)
        return spool

    monkeypatch.setattr(sizing, "compute_to_spool", corrupt)
    with pytest.raises(ValueError, match="duplicate"):
        run_sizing(vectors, tmp_path, slice_rows=200, validate=20, k=10, block_rows=64, threads=1, guard=_guard())


def test_run_sizing_without_validation_and_bad_slice(tmp_path: Path) -> None:
    vectors = np.random.default_rng(1).standard_normal((300, 8)).astype(np.float16)
    report = run_sizing(vectors, tmp_path, slice_rows=64, validate=0, k=5, block_rows=64, threads=1, guard=_guard())
    assert "exactness" not in report
    for bad in (0, 301):
        with pytest.raises(ValueError, match="slice_rows"):
            run_sizing(vectors, tmp_path, slice_rows=bad, k=5, block_rows=64, threads=1, guard=_guard())


def test_extrapolation_model() -> None:
    profile = PhaseProfile(upcast_s=10.0, matmul_s=30.0, select_s=60.0)
    gib = 1024**3
    result = extrapolate(
        slice_rows=1_000_000,
        catalog_rows=9_000_000,
        kernel_wall_s=1000.0,
        profile=profile,
        peak_rss=3 * gib,
        k=50,
        dim=128,
        threads=8,
        block_rows=4096,
    )
    assert result.scale == 9.0
    assert result.matmul_fraction == pytest.approx(0.4)
    # 1000 s x 9 x (0.4 x 0.5 + 0.6): multiply halves under symmetry, selection does not.
    assert result.full_wall_s == pytest.approx(7200.0)
    assert result.full_wall_upper_bound_s == pytest.approx(9000.0)
    assert result.full_peak_rss_bytes == 3 * gib + 8_000_000 * 408
    assert result.within_six_hours and result.within_six_hours_upper_bound
    assert result.within_memory_budget
    over = extrapolate(
        slice_rows=500_000, catalog_rows=9_000_000, kernel_wall_s=2000.0, profile=profile, peak_rss=9 * gib, k=50, dim=128, threads=8, block_rows=4096
    )
    assert not over.within_memory_budget and not over.within_six_hours


class _Cursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows

    async def __aenter__(self) -> _Cursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    async def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None


class _Conn:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.statements: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self.rows)

    async def rollback(self) -> None:
        return None


@pytest.mark.asyncio
async def test_session_must_be_read_only() -> None:
    await _assert_read_only(_Conn([("on",)]))
    with pytest.raises(RuntimeError, match="read-only"):
        await _assert_read_only(_Conn([("off",)]))


@pytest.mark.asyncio
async def test_model_version_resolution() -> None:
    one = _Conn([("v1", 10)])
    assert await _resolve_model_version(one, None, 10) == ("v1", 10)
    with pytest.raises(ValueError, match="expected 11"):
        await _resolve_model_version(one, None, 11)
    with pytest.raises(ValueError, match="exactly one"):
        await _resolve_model_version(_Conn([("v1", 1), ("v2", 2)]), None, None)
    assert await _resolve_model_version(_Conn([("v1", 1), ("v2", 2)]), "v2", None) == ("v2", 2)
    with pytest.raises(ValueError, match="not in artist_embeddings"):
        await _resolve_model_version(one, "other", None)


@pytest.mark.asyncio
async def test_load_catalog_refuses_a_writable_session() -> None:
    class Pool:
        def connection(self) -> Any:
            conn = _Conn([("off",)])

            class Ctx:
                async def __aenter__(self_inner) -> _Conn:
                    return conn

                async def __aexit__(self_inner, *exc: object) -> None:
                    return None

            return Ctx()

    with pytest.raises(RuntimeError, match="read-only"):
        await load_catalog(Pool(), None, None)
