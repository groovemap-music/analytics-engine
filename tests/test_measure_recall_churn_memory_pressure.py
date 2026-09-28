"""gm-analytics-engine-i37, 2026-09-28: unit coverage for the memory/disk-pressure fix.

Synthetic data only -- no Discogs/MusicBrainz-derived data is read or committed here. The
host swapped to 30 GB and disk fell to 198 MB while `measure_recall_churn.py` held both
months' vectors as full float32 arrays in RAM at once; this fix (1) streams a month's
vectors from its npz in bounded chunks instead of ever materializing the whole array
(`_iter_npz_vector_chunks`, `_extract_rows`, `_stream_exact_top_k`), and (2) pauses before
heavy steps while the HOST is under memory or disk pressure (`wait_for_host_pressure`,
`_free_swap_gb`). This file covers: the chunked top-k matching a full in-memory computation
on random data, the streaming primitives it's built from, the host-pressure guard with
mocked `sysctl`/`shutil.disk_usage`, and `is_result_complete`'s resume-skip check.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from scripts import measure_recall_churn as mrc


if TYPE_CHECKING:
    from pathlib import Path


def _write_synthetic_npz(path: Path, *, n: int, dim: int, seed: int) -> np.ndarray:
    """Write a tiny synthetic npz shaped like `embeddings_from_dump.py`'s real output
    (random, continuous vectors -- exact ties are ~probability zero, so the chunked and
    full-matrix top-k have a unique, directly comparable answer) and return the raw vectors
    for building a reference result."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    artist_ids = np.array([f"a{i}" for i in range(n)], dtype=object)
    np.savez_compressed(
        path,
        vectors=vectors,
        artist_ids=artist_ids,
        model_version=np.array("test-model-version"),
        dump_id=np.array("test-dump"),
        dump_date=np.array("2026-09-01"),
    )
    return vectors


class TestStreamingVectorChunks:
    def test_iter_npz_vector_chunks_reconstructs_the_full_array(self, tmp_path: Path) -> None:
        path = tmp_path / "month.npz"
        vectors = _write_synthetic_npz(path, n=737, dim=6, seed=1)

        # A tiny chunk_bytes forces many small chunks -- exercises the multi-chunk path,
        # not just a single chunk covering everything.
        chunks = list(mrc._iter_npz_vector_chunks(path, chunk_bytes=256))
        assert len(chunks) > 1
        rebuilt = np.concatenate(chunks, axis=0)
        assert np.array_equal(rebuilt, vectors)

    def test_extract_rows_matches_direct_indexing(self, tmp_path: Path) -> None:
        path = tmp_path / "month.npz"
        vectors = _write_synthetic_npz(path, n=500, dim=8, seed=2)
        rng = np.random.default_rng(3)
        positions = sorted(rng.choice(500, size=40, replace=False).tolist())

        extracted = mrc._extract_rows(path, positions, chunk_bytes=512)

        assert np.allclose(extracted, vectors[positions])

    def test_extract_rows_rejects_unsorted_positions(self, tmp_path: Path) -> None:
        path = tmp_path / "month.npz"
        _write_synthetic_npz(path, n=50, dim=4, seed=4)
        with pytest.raises(ValueError, match="sorted"):
            mrc._extract_rows(path, [5, 1, 3])

    def test_ensure_normalized_cached_is_a_no_op_when_already_cached(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "month.npz"
        _write_synthetic_npz(path, n=100, dim=4, seed=5)
        cache = {3: np.zeros(4, dtype=np.float32)}

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("_extract_rows must not be called when every position is already cached")

        monkeypatch.setattr(mrc, "_extract_rows", _boom)
        mrc._ensure_normalized_cached(path, cache, [3])  # must not raise


class TestStreamExactTopKMatchesFullMatrix:
    """The chunked implementation must match `_exact_top_k` on an already-`_normalized` full
    in-memory matrix -- the reference this repo used before gm-analytics-engine-i37's
    memory fix. Continuous random vectors make exact ties ~impossible, so both the top-k
    SET and the k-th-place score must agree exactly (tie-break order is explicitly allowed
    to differ per `_stream_exact_top_k`'s own docstring, but that never arises here)."""

    @pytest.mark.parametrize("chunk_bytes", [64, 512, 4096, 1_000_000])
    def test_matches_reference_across_chunk_sizes(self, tmp_path: Path, chunk_bytes: int) -> None:
        path = tmp_path / "month.npz"
        vectors = _write_synthetic_npz(path, n=623, dim=12, seed=42)
        rng = np.random.default_rng(99)
        query_positions = sorted(rng.choice(623, size=30, replace=False).tolist())

        reference_normalized = mrc._normalized(vectors)
        ref_results, ref_kth = mrc._exact_top_k(reference_normalized, query_positions, 10)

        chunk_results, chunk_kth = mrc._stream_exact_top_k(path, query_positions, 10, chunk_bytes=chunk_bytes)

        for ref_row, chunk_row in zip(ref_results, chunk_results, strict=True):
            assert set(ref_row) == set(chunk_row)
        for ref_score, chunk_score in zip(ref_kth, chunk_kth, strict=True):
            assert ref_score == pytest.approx(chunk_score, abs=1e-4)

    def test_excludes_the_query_s_own_position(self, tmp_path: Path) -> None:
        path = tmp_path / "month.npz"
        _write_synthetic_npz(path, n=200, dim=6, seed=7)
        results, _kth = mrc._stream_exact_top_k(path, [15, 42], 10, chunk_bytes=1024)
        assert 15 not in results[0]
        assert 42 not in results[1]

    def test_populates_and_reuses_a_given_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "month.npz"
        _write_synthetic_npz(path, n=150, dim=5, seed=8)
        cache: dict[int, np.ndarray] = {}
        mrc._stream_exact_top_k(path, [1, 2, 3], 10, chunk_bytes=512, cache=cache)
        assert {1, 2, 3}.issubset(cache.keys())
        # Every cached vector must be unit-normalized.
        for vector in cache.values():
            assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


class TestHostPressureGuard:
    """`wait_for_host_pressure` must never abort -- it blocks (via `time.sleep`, mocked away
    here) while `kern.memorystatus_level` is low, swap used is high, or free disk is low, and
    returns as soon as all three clear.

    gm-analytics-engine-i37, 2026-09-28: this guard's FIRST version checked free swap, which
    is the wrong metric on macOS -- swap files grow dynamically in ~1 GB increments and are
    rarely shrunk back, so free swap sits under 2 GB almost permanently even on a healthy
    host (caught live: 0.79 GB free with memorystatus_level=71%, a perfectly healthy
    machine -- that guard would have waited forever). `kern.memorystatus_level` (0-100, "%
    free") is the primary signal now; swap USED (not free) is kept only as a backstop against
    the specific ~30 GB runaway this bead hit on 2026-09-27.
    """

    def test_memorystatus_level_parses_sysctl_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Result:
            returncode = 0
            stdout = "71\n"

        monkeypatch.setattr(mrc.subprocess, "run", lambda *a, **k: _Result())
        assert mrc._memorystatus_level_pct() == 71

    def test_memorystatus_level_returns_none_when_sysctl_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Result:
            returncode = 1
            stdout = ""

        monkeypatch.setattr(mrc.subprocess, "run", lambda *a, **k: _Result())
        assert mrc._memorystatus_level_pct() is None

    def test_memorystatus_level_returns_none_on_unparseable_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Result:
            returncode = 0
            stdout = "not-a-number\n"

        monkeypatch.setattr(mrc.subprocess, "run", lambda *a, **k: _Result())
        assert mrc._memorystatus_level_pct() is None

    def test_swap_used_gb_parses_sysctl_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Result:
            returncode = 0
            stdout = "vm.swapusage: total = 15360.00M  used = 14225.81M  free = 1134.19M  (encrypted)\n"

        monkeypatch.setattr(mrc.subprocess, "run", lambda *a, **k: _Result())
        assert mrc._swap_used_gb() == pytest.approx(14225.81 / 1024.0, abs=1e-6)

    def test_swap_used_gb_returns_none_when_sysctl_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Result:
            returncode = 1
            stdout = ""

        monkeypatch.setattr(mrc.subprocess, "run", lambda *a, **k: _Result())
        assert mrc._swap_used_gb() is None

    def test_returns_immediately_when_pressure_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc, "_memorystatus_level_pct", lambda: 71)
        monkeypatch.setattr(mrc, "_swap_used_gb", lambda: 5.0)
        monkeypatch.setattr(mrc.shutil, "disk_usage", lambda _path: type("U", (), {"free": 50 * 1024**3})())

        def _boom(_seconds: float) -> None:
            raise AssertionError("must not sleep when memory, swap, and disk are all fine")

        monkeypatch.setattr(mrc.time, "sleep", _boom)
        mrc.wait_for_host_pressure()  # must return without sleeping.

    def test_a_healthy_low_free_swap_host_does_not_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The exact incident this guard's rewrite fixes: 0.79 GB free swap (which the OLD
        free-swap check would have blocked on forever) but memorystatus_level=71% (healthy)
        and swap used well under the 26 GB backstop -- must return immediately."""
        monkeypatch.setattr(mrc, "_memorystatus_level_pct", lambda: 71)
        monkeypatch.setattr(mrc, "_swap_used_gb", lambda: 14.5)
        monkeypatch.setattr(mrc.shutil, "disk_usage", lambda _path: type("U", (), {"free": 20 * 1024**3})())

        def _boom(_seconds: float) -> None:
            raise AssertionError("a healthy memorystatus_level must not wait on low free swap")

        monkeypatch.setattr(mrc.time, "sleep", _boom)
        mrc.wait_for_host_pressure()

    def test_waits_while_memorystatus_level_is_low_then_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        level_values = iter([10, 10, 80])  # under threshold twice, then clears.
        sleep_calls = []

        monkeypatch.setattr(mrc, "_memorystatus_level_pct", lambda: next(level_values))
        monkeypatch.setattr(mrc, "_swap_used_gb", lambda: 5.0)
        monkeypatch.setattr(mrc.shutil, "disk_usage", lambda _path: type("U", (), {"free": 50 * 1024**3})())
        monkeypatch.setattr(mrc.time, "sleep", sleep_calls.append)

        mrc.wait_for_host_pressure(poll_interval_s=0.01, log_interval_s=0.0)

        assert len(sleep_calls) == 2

    def test_waits_while_swap_used_backstop_is_tripped_then_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        swap_values = iter([30.0, 30.0, 10.0])  # over the 26 GB backstop twice, then clears.
        sleep_calls = []

        monkeypatch.setattr(mrc, "_memorystatus_level_pct", lambda: 80)
        monkeypatch.setattr(mrc, "_swap_used_gb", lambda: next(swap_values))
        monkeypatch.setattr(mrc.shutil, "disk_usage", lambda _path: type("U", (), {"free": 50 * 1024**3})())
        monkeypatch.setattr(mrc.time, "sleep", sleep_calls.append)

        mrc.wait_for_host_pressure(poll_interval_s=0.01, log_interval_s=0.0)

        assert len(sleep_calls) == 2

    def test_waits_while_disk_is_low_then_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        disk_frees = iter([1 * 1024**3, 1 * 1024**3, 20 * 1024**3])
        sleep_calls = []

        monkeypatch.setattr(mrc, "_memorystatus_level_pct", lambda: 80)
        monkeypatch.setattr(mrc, "_swap_used_gb", lambda: 5.0)
        monkeypatch.setattr(mrc.shutil, "disk_usage", lambda _path: type("U", (), {"free": next(disk_frees)})())
        monkeypatch.setattr(mrc.time, "sleep", sleep_calls.append)

        mrc.wait_for_host_pressure(poll_interval_s=0.01, log_interval_s=0.0)

        assert len(sleep_calls) == 2

    def test_none_signals_degrade_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`sysctl` unavailable (e.g. non-macOS) for BOTH memory signals must not make this
        guard wait forever -- it should fall back to the disk check alone."""
        monkeypatch.setattr(mrc, "_memorystatus_level_pct", lambda: None)
        monkeypatch.setattr(mrc, "_swap_used_gb", lambda: None)
        monkeypatch.setattr(mrc.shutil, "disk_usage", lambda _path: type("U", (), {"free": 50 * 1024**3})())

        def _boom(_seconds: float) -> None:
            raise AssertionError("must not sleep when both memory signals are unknown but disk is fine")

        monkeypatch.setattr(mrc.time, "sleep", _boom)
        mrc.wait_for_host_pressure()


class TestIsResultComplete:
    """The resume-skip check gm-analytics-engine-i37's driver uses to recognize a weight
    whose result reached disk before a crash, without redoing a possibly multi-hour build."""

    def test_missing_file_is_not_complete(self, tmp_path: Path) -> None:
        assert mrc.is_result_complete(tmp_path / "does-not-exist.json") is False

    def test_non_json_content_is_not_complete(self, tmp_path: Path) -> None:
        path = tmp_path / "result.json"
        path.write_text("not json{{{")
        assert mrc.is_result_complete(path) is False

    def test_partial_result_is_not_complete(self, tmp_path: Path) -> None:
        path = tmp_path / "result.json"
        path.write_text(json.dumps({"partial": True, "september": {}}))
        assert mrc.is_result_complete(path) is False

    def test_missing_partial_key_is_not_complete(self, tmp_path: Path) -> None:
        path = tmp_path / "result.json"
        path.write_text(json.dumps({"september": {}}))
        assert mrc.is_result_complete(path) is False

    def test_non_partial_result_is_complete(self, tmp_path: Path) -> None:
        path = tmp_path / "result.json"
        path.write_text(json.dumps({"partial": False, "september": {"model_version": "m"}}))
        assert mrc.is_result_complete(path) is True

    def test_non_dict_json_is_not_complete(self, tmp_path: Path) -> None:
        path = tmp_path / "result.json"
        path.write_text(json.dumps([1, 2, 3]))
        assert mrc.is_result_complete(path) is False
