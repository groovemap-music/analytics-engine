"""Measure real-embedding ANN recall@10 and month-over-month churn, for
gm-analytics-engine-ieu.3 -- the two open ADR 0013 preconditions.

Takes the compact ``(artist_id, vector)`` ``.npz`` files `scripts/embeddings_from_dump.py`
produced for two consecutive months and, against a throwaway PostgreSQL 19 + pgvector
container (one month's rows in an otherwise-empty `public.artist_embeddings` table at a
time -- the real DDL from `database-schema`, not a stand-in):

- writes the month's rows through the real `insights.embedding_pipeline._write_embeddings`
  (falling back to `COPY` only if a timed trial extrapolates past ~30 minutes, per the
  maintainer's condition -- see `_write_month`),
- builds the real HNSW index, named by `insights.embedding_pipeline._index_name`, under a
  session-only `maintenance_work_mem` bump (full scale first; a subset fallback is used
  only after a real failure). August uses database-schema's documented 2 GB value.
  September, per a dispatcher decision made after August's build measurably slowed around
  the point its HNSW graph likely outgrew that 2 GB, uses a higher one instead (raising it
  needs the container's `--shm-size` to cover it too, so this script restarts the
  throwaway container between months when the two values differ, rather than a plain
  `TRUNCATE` -- `--shm-size` can only be set at container creation). Both builds' settings
  and wall times are recorded side by side; see docs/recall_and_churn.md.
- sweeps recall@10 against exact cosine (computed in NumPy directly from the in-memory
  vectors, no Postgres needed for the exact side) over `ef_search` in `EF_SEARCH_SWEEP`,
  naming the smallest value reaching recall@10 >= 0.95 (or stating that none does), and
- computes each month's top-10 (both exact and, at the named production `ef_search`, via
  the live ANN index) for a deterministic sample of artists common to both months, so
  churn can be reported both on exact vectors and on what the served index would actually
  return.

Sampling is deterministic and repeatable: `_deterministic_sample` ranks candidates by
`splitmix64(node_key("a", artist_id) XOR seed)` -- the same hash the pinned FastRP
projection itself uses -- and takes the smallest `n`. Two fixed seeds (`QUERY_SAMPLE_SEED`,
`CHURN_SAMPLE_SEED`) are recorded here so a re-run picks the identical sample.

No provider-derived data is committed: this script reads the local `.npz` scratch files
(themselves gitignored/uncommitted) and writes only aggregate JSON to stdout.

    uv run python scripts/measure_recall_churn.py \\
        ~/.cache/groovemap-spikes/embeddings-scratch/aug.npz \\
        ~/.cache/groovemap-spikes/embeddings-scratch/sept.npz \\
        --host 127.0.0.1 --port 5432 --database groovemap \\
        --username groovemap --password integration-test-password \\
        --out ~/.cache/groovemap-spikes/embeddings-scratch/recall_churn.json
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import re
import resource
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import numpy as np
import numpy.lib.format as npy_format


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
from common import AsyncPostgreSQLPool

from insights.embedding_pipeline import (
    ARTIST_EMBEDDINGS_TABLE,
    FastRPConfig,
    _already_loaded,
    _index_name,
    _sql_string_literal,
    _write_embeddings,
    stored_model_version,
)
from insights.embeddings.graph import node_key
from insights.embeddings.projection import splitmix64


QUERY_SAMPLE_SEED: int = 0x51EC_A11A  # "recall" sample seed, arbitrary but fixed.
CHURN_SAMPLE_SEED: int = 0xC8027A11  # "churn" sample seed, arbitrary but fixed.
QUERY_SAMPLE_SIZE: int = 2_000
CHURN_SAMPLE_SIZE: int = 10_000
EF_SEARCH_SWEEP: tuple[int, ...] = (40, 100, 200, 400, 800, 1000)  # pgvector caps hnsw.ef_search at 1000.
RECALL_TARGET: float = 0.95
HNSW_M: int = 16
HNSW_EF_CONSTRUCTION: int = 64
# The larger-index variant the maintainer approved alongside i37's w0 sweep (kn3's edges-v2
# Sept measurement, now merged as docs/recall_and_churn.md's "Recall and tie structure":
# tie-tolerant recall tops out at 0.8778 at ef_search=1000, over-fetch+re-rank gains nothing
# over the index at the same ef_search -- so index size and w0 are what's left to try).
# Same `halfvec_cosine_ops`, same `maintenance_work_mem`/parallel-worker settings as the
# standard variant; only `m`/`ef_construction` differ.
HNSW_LARGER_M: int = 32
HNSW_LARGER_EF_CONSTRUCTION: int = 128
# `--skip-larger-variant`'s real finding, real enough to record even though the build never
# finished (maintainer decision, 2026-09-27): September's m=32 build for w0=0 overflowed
# maintenance_work_mem=8GB at ~6.1M of 9,366,416 tuples and fell to the on-disk build path
# (workers on IO/DataFileRead, ~80 tuples/s), abandoned after ~4h50m -- finishing would take
# ~8h per build, repeated per w0 and again for the winner run. That overflow, not a completed
# measurement, IS the recorded outcome for the larger-index variant at this catalog scale.
LARGER_VARIANT_SKIP_REASON: str = (
    "September's m=32 build for w0=0 overflowed maintenance_work_mem=8GB at ~6.1M of 9,366,416 tuples and fell to the "
    "on-disk build path (~80 tuples/s); finishing would take ~8h per build, repeated per w0 and again for the winner "
    "run. Skipped per the maintainer's decision (2026-09-27); the overflow itself is the finding -- see "
    "larger_variant_finding in the top-level results."
)
DEFAULT_MAINTENANCE_WORK_MEM: str = "2GB"  # database-schema's documented build-time value.
TRIAL_BATCH_ROWS: int = 100_000
TRIAL_TIME_BUDGET_S: float = 30 * 60  # 30 minutes, per the maintainer's condition.

# The maintainer's tie-tolerant recall definition (docs/recall_and_churn.md, "Recall and tie
# structure"): count an ANN top-10 candidate as a hit even when it's outside the EXACT top-10,
# as long as its own exact cosine similarity is within this much of the exact 10th-place
# score -- a large share of this graph's top-10 boundaries are ties or near-ties (FastRP's
# zero-weight-on-self-projection artifact), so part of strict recall's "misses" are really
# the ANN index and the brute-force exact computation each validly picking a different member
# of a tied group, not a genuinely worse neighbour.
TIE_TOLERANCE: Final = 1e-4

# Power-of-two degree-bucket edges for gm-analytics-engine-i37's recall-by-degree breakdown
# (kn3's measurement has no graph, so it can't compute this itself). FastRP's propagation is
# degree-sensitive by construction -- a hub's embedding averages over far more neighbours
# than a leaf's -- so recall is a-priori expected to vary by degree; these boundaries are a
# reasonable default pending maintainer feedback, not a tuned choice.
DEGREE_BUCKET_EDGES: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Resolved once to a full path, matching this repo's own `GIT = shutil.which("git")`
# convention (tests/test_repository_compliance.py) -- S607 wants a full executable path,
# not a bare name resolved via $PATH at call time.
DOCKER: Final = shutil.which("docker") or "docker"

DOCKER_POLL_INTERVAL_S: Final = 60.0

# gm-analytics-engine-i37, 2026-09-28: the host (not the throwaway container) swapped to
# 30 GB and its own free disk fell to 198 MB while this script held both months' vectors as
# full float32 arrays (~10 GB combined) plus Postgres/Colima's own footprint. Two separate
# fixes: (1) never materialize a whole month's vector array at once -- stream it from the
# npz in bounded chunks instead (`_iter_npz_vector_chunks`, `_extract_rows`,
# `_stream_exact_top_k`); (2) pause before every heavy step, and between ef_search sweep
# iterations, while the HOST is under memory or disk pressure (`wait_for_host_pressure`).
_DEFAULT_CHUNK_BYTES: Final = 256 * 1024 * 1024  # ~256 MB per streamed chunk of a month's vectors.
_WRITE_CHUNK_BYTES: Final = 256 * 1024 * 1024  # ditto, for streaming a month's rows into Postgres.

HOST_PRESSURE_POLL_INTERVAL_S: Final = 60.0
HOST_PRESSURE_LOG_INTERVAL_S: Final = 600.0  # log at most once per 10 minutes while waiting -- never abort.
# `kern.memorystatus_level` (0-100, "% free" in Activity Monitor's own "Memory Pressure"
# sense) is the primary signal, not free swap: macOS grows its swap files dynamically in
# ~1 GB increments and rarely shrinks them back, so `vm.swapusage`'s "free" figure sits under
# 2 GB almost permanently even on a healthy machine (gm-analytics-engine-i37, 2026-09-28 --
# this guard's first version used free swap and would have waited forever: 0.79 GB free with
# memorystatus_level=71%, a perfectly healthy host). Swap USED, not free, is kept as a
# backstop against the specific runaway this bead hit (~30 GB used on 2026-09-27).
MIN_MEMORYSTATUS_LEVEL_PCT: Final = 25
MAX_SWAP_USED_GB: Final = 26.0
MIN_FREE_DISK_GB: Final = 8.0


def _peak_rss_mb() -> float:
    """This process's peak resident set size so far, in MB -- `resource.getrusage`'s
    `ru_maxrss` is bytes on macOS/BSD and KiB on Linux, so this normalizes for whichever
    platform is running. Logged after every heavy step (gm-analytics-engine-i37's
    memory-pressure fix, 2026-09-28) so the measured peak, not a guess, is what gets reported.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def _memorystatus_level_pct() -> int | None:
    """`kern.memorystatus_level` (0-100): macOS's own "% of memory free" figure -- the same
    number Activity Monitor's "Memory Pressure" gauge is built from. `None` if that sysctl
    isn't available (e.g. this ever runs on Linux CI), so `wait_for_host_pressure` degrades
    to its swap-used and disk checks rather than failing outright."""
    result = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, check=False)  # noqa: S607
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _swap_used_gb() -> float | None:
    """Swap USED, in GB, parsed from `sysctl vm.swapusage` -- the backstop against the
    specific runaway gm-analytics-engine-i37 hit on 2026-09-27 (~30 GB used), not the primary
    signal (see `MIN_MEMORYSTATUS_LEVEL_PCT`'s comment for why free swap alone is the wrong
    metric on macOS). `None` if that sysctl isn't available."""
    result = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=False)  # noqa: S607 -- fixed argv, read-only.
    if result.returncode != 0:
        return None
    match = re.search(r"used\s*=\s*([\d.]+)M", result.stdout)
    return float(match.group(1)) / 1024.0 if match else None


def wait_for_host_pressure(
    *,
    min_memorystatus_level_pct: float = MIN_MEMORYSTATUS_LEVEL_PCT,
    max_swap_used_gb: float = MAX_SWAP_USED_GB,
    min_free_disk_gb: float = MIN_FREE_DISK_GB,
    poll_interval_s: float = HOST_PRESSURE_POLL_INTERVAL_S,
    log_interval_s: float = HOST_PRESSURE_LOG_INTERVAL_S,
    label: str = "",
) -> None:
    """Block (polling, never aborting) while the HOST -- not the throwaway container -- is
    under memory or disk pressure: `kern.memorystatus_level` below MIN_MEMORYSTATUS_LEVEL_PCT
    (primary signal), swap used above MAX_SWAP_USED_GB (backstop against the specific
    2026-09-27 runaway), or free disk below MIN_FREE_DISK_GB. Called before every heavy step
    (load, write, index build, sweep, churn) and between `EF_SEARCH_SWEEP` iterations
    (gm-analytics-engine-i37, 2026-09-28 -- see the module-level comment above
    `_DEFAULT_CHUNK_BYTES`, and `MIN_MEMORYSTATUS_LEVEL_PCT`'s own comment for why this isn't
    a free-swap check). Logs at most once per LOG_INTERVAL_S while waiting, not on every
    poll -- this can legitimately wait a long time without anything having gone wrong.
    """
    last_logged = 0.0
    while True:
        level_pct = _memorystatus_level_pct()
        swap_used_gb = _swap_used_gb()
        free_disk_gb = shutil.disk_usage("/").free / 1e9
        memory_ok = level_pct is None or level_pct >= min_memorystatus_level_pct
        swap_ok = swap_used_gb is None or swap_used_gb <= max_swap_used_gb
        disk_ok = free_disk_gb >= min_free_disk_gb
        if memory_ok and swap_ok and disk_ok:
            return
        now = time.monotonic()
        if now - last_logged >= log_interval_s:
            level_text = "n/a" if level_pct is None else f"{level_pct}%"
            swap_text = "n/a" if swap_used_gb is None else f"{swap_used_gb:.2f} GB"
            print(
                f"⏳ {label}host under pressure (memorystatus_level={level_text}, swap used={swap_text}, "
                f"free disk={free_disk_gb:.2f} GB) -- waiting for level>={min_memorystatus_level_pct:.0f}%, "
                f"swap used<={max_swap_used_gb:.1f} GB, disk>={min_free_disk_gb:.1f} GB",
                file=sys.stderr,
                flush=True,
            )
            last_logged = now
        time.sleep(poll_interval_s)


def _iter_npz_vector_chunks(path: Path, *, member: str = "vectors", chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> Iterator[np.ndarray]:
    """Stream MEMBER's rows from an `.npz` written by `np.savez_compressed`, roughly
    CHUNK_BYTES worth at a time, decompressing incrementally via `zipfile` rather than
    materializing the whole array in RAM the way `np.load` does -- the ~5 GB-per-month RSS
    spike gm-analytics-engine-i37 hit on 2026-09-27/28 (host swapped to 30 GB with both
    months' vectors live in Python RAM at once). Requires a 2D, C-contiguous member (true of
    every `vectors` array `embeddings_from_dump.py` has ever written).
    """
    with zipfile.ZipFile(path) as zf, zf.open(f"{member}.npy") as fh:
        major, _minor = npy_format.read_magic(fh)
        if major == 1:
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(fh)
        elif major == 2:
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(fh)
        else:
            raise ValueError(f"{path}:{member}: unsupported .npy format version {major}")
        if fortran_order:
            raise ValueError(f"{path}:{member}: fortran-order arrays are not supported by chunked reading")
        if len(shape) != 2:
            raise ValueError(f"{path}:{member}: expected a 2D array, got shape {shape}")
        n_rows, n_cols = shape
        row_bytes = n_cols * dtype.itemsize
        chunk_rows = max(1, chunk_bytes // row_bytes)
        remaining = n_rows
        while remaining > 0:
            take = min(chunk_rows, remaining)
            buf = fh.read(take * row_bytes)
            if len(buf) != take * row_bytes:
                raise ValueError(f"{path}:{member}: truncated read ({len(buf)} of {take * row_bytes} bytes) -- npz may be corrupt")
            yield np.frombuffer(buf, dtype=dtype).reshape(take, n_cols)
            remaining -= take


def _extract_rows(path: Path, positions: list[int], *, member: str = "vectors", chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> np.ndarray:
    """Pull out exactly the rows at POSITIONS (must be sorted, ascending, and unique) from
    MEMBER, in ONE streaming pass over PATH's npz (`_iter_npz_vector_chunks`) -- the
    small-random-access counterpart to that function's bulk streaming, for the handful of
    query/candidate positions the recall and churn computations need at any one time, never
    the whole month's matrix."""
    if positions != sorted(set(positions)):
        raise ValueError("_extract_rows requires positions to be sorted, ascending, and unique")
    if not positions:
        return np.empty((0, 0), dtype=np.float32)
    result: list[np.ndarray | None] = [None] * len(positions)
    want_index = 0
    global_position = 0
    for chunk in _iter_npz_vector_chunks(path, member=member, chunk_bytes=chunk_bytes):
        chunk_end = global_position + chunk.shape[0]
        while want_index < len(positions) and positions[want_index] < chunk_end:
            result[want_index] = chunk[positions[want_index] - global_position].copy()
            want_index += 1
        global_position = chunk_end
        if want_index >= len(positions):
            break
    if want_index < len(positions):
        raise ValueError(f"{path}:{member}: position {positions[want_index]} out of range (only {global_position} rows total)")
    return np.stack(result)  # type: ignore[arg-type]


def _ensure_normalized_cached(path: Path, cache: dict[int, np.ndarray], positions: Iterable[int], *, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> None:
    """Make sure CACHE has a normalized vector for every position in POSITIONS, streaming
    PATH's `vectors` member once (bounded memory, `_extract_rows`) to fetch and normalize
    whatever isn't already there. A no-op -- no I/O at all -- once everything's already
    cached, which is the common case: a query's own position is cached once (by
    `compute_exact_ground_truth` / `_stream_exact_top_k`) and reused for the rest of that
    month's `EF_SEARCH_SWEEP`."""
    missing = sorted({p for p in positions if p not in cache})
    if not missing:
        return
    normalized = _normalized(_extract_rows(path, missing, chunk_bytes=chunk_bytes))
    for position, vector in zip(missing, normalized, strict=True):
        cache[position] = vector


def _stream_exact_top_k(
    path: Path,
    query_positions: list[int],
    k: int,
    *,
    member: str = "vectors",
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    cache: dict[int, np.ndarray] | None = None,
) -> tuple[list[list[int]], list[float]]:
    """`_exact_top_k`'s streaming counterpart: the same exact top-`k` cosine neighbours (by
    position, excluding self) for each of QUERY_POSITIONS and each query's exact k-th-place
    score, computed WITHOUT ever materializing PATH's full `vectors` array -- two streaming
    passes over it instead (`_iter_npz_vector_chunks`): pass 1 (`_extract_rows`, via CACHE if
    given) pulls out and normalizes just the query rows; pass 2 streams every row, scores it
    against every query, and keeps a running top-`k` per query, merging each chunk's scores
    into the running best-so-far the same way a single in-memory `argpartition` would.
    Numerically equivalent to `_exact_top_k(_normalized(full_matrix), query_positions, k)` on
    the same data for non-tied scores (verified directly against it on synthetic data with
    continuous, effectively-never-tied random vectors); tie-break ORDER among exactly equal
    scores may differ from the single-array version, which is fine here -- every caller
    either treats the top-k as a SET (`_recall_at_k`, `_jaccard`) or explicitly tolerates
    ties by score (`_tie_tolerant_recall_at_k`, `TIE_TOLERANCE`), never by position.
    """
    n_queries = len(query_positions)
    if cache is not None:
        _ensure_normalized_cached(path, cache, query_positions, chunk_bytes=chunk_bytes)
        queries = np.stack([cache[p] for p in query_positions]).astype(np.float32)
    else:
        unique_sorted = sorted(set(query_positions))
        normalized = _normalized(_extract_rows(path, unique_sorted, member=member, chunk_bytes=chunk_bytes))
        by_position = dict(zip(unique_sorted, normalized, strict=True))
        queries = np.stack([by_position[p] for p in query_positions]).astype(np.float32)

    dim = queries.shape[1]
    # Bound this pass's per-chunk memory by the DOMINANT cost -- the (chunk_rows, n_queries)
    # score matrix, not the (chunk_rows, dim) vector chunk itself (dim=128 is tiny next to
    # n_queries up to CHURN_SAMPLE_SIZE=10,000).
    bytes_per_row = dim * 4 + n_queries * 4 + n_queries * 4  # chunk row (f32) + scores (f32) + positions (i32).
    chunk_rows = max(1, chunk_bytes // bytes_per_row)

    top_scores = np.full((n_queries, k), -np.inf, dtype=np.float32)
    top_positions = np.full((n_queries, k), -1, dtype=np.int32)

    position = 0
    for chunk in _iter_npz_vector_chunks(path, member=member, chunk_bytes=chunk_rows * dim * 4):
        chunk_normalized = _normalized(chunk)
        scores = chunk_normalized @ queries.T  # (chunk_rows, n_queries)
        for qi, qp in enumerate(query_positions):
            if position <= qp < position + chunk.shape[0]:
                scores[qp - position, qi] = -np.inf  # exclude self.
        new_positions = np.broadcast_to((position + np.arange(chunk.shape[0], dtype=np.int32))[:, None], scores.shape)
        combined_scores = np.concatenate([top_scores.T, scores], axis=0)
        combined_positions = np.concatenate([top_positions.T, new_positions], axis=0)
        # `argpartition` (O(n)), not `argsort` (O(n log n)): the running top-k doesn't need
        # to stay SORTED between merges, only the final result does (sorted once, below, on
        # just k elements). A full sort of (k + chunk_rows) elements every chunk was this
        # function's actual dominant cost, not the matmul above -- gm-analytics-engine-i37,
        # 2026-09-28, found while validating this fix's own peak-RSS claim: a 2,000-query
        # ground-truth pass over 2M synthetic rows took 358s with a full sort per chunk.
        keep = min(k, combined_scores.shape[0])
        order = np.argpartition(-combined_scores, keep - 1, axis=0)[:keep] if keep > 0 else np.empty((0, n_queries), dtype=np.intp)
        top_scores = np.full((n_queries, k), -np.inf, dtype=np.float32)
        top_positions = np.full((n_queries, k), -1, dtype=np.int32)
        top_scores[:, :keep] = np.take_along_axis(combined_scores, order, axis=0).T
        top_positions[:, :keep] = np.take_along_axis(combined_positions, order, axis=0).T
        position += chunk.shape[0]

    # One final sort, on just k (small) elements per query, for the "highest score first"
    # ordering `_exact_top_k` callers expect and to put the k-th (minimum, since sorted
    # descending) score last.
    final_order = np.argsort(-top_scores, axis=1)
    top_scores = np.take_along_axis(top_scores, final_order, axis=1)
    top_positions = np.take_along_axis(top_positions, final_order, axis=1)

    results = [top_positions[i].tolist() for i in range(n_queries)]
    kth_scores = [float(top_scores[i, -1]) if k > 0 else float("-inf") for i in range(n_queries)]
    return results, kth_scores


def wait_for_docker(*, poll_interval_s: float = DOCKER_POLL_INTERVAL_S) -> None:
    """Block until `docker info` succeeds, polling and logging while waiting -- so the
    recall/churn phase never fails outright just because Docker (or Colima's VM under it)
    isn't up yet or is between containers when this script starts. Called once, at the very
    top of `main_async`, before this script assumes it can talk to a container at all --
    every other Docker interaction here (`_restart_container`, the throwaway container
    `main()`'s caller starts) happens after this point.
    """
    attempt = 0
    while True:
        attempt += 1
        result = subprocess.run([DOCKER, "info"], capture_output=True)  # noqa: S603 -- DOCKER is a resolved full path (S607), no arguments.
        if result.returncode == 0:
            print(f"✅ Docker is reachable (attempt {attempt})", file=sys.stderr, flush=True)
            return
        stderr_tail = result.stderr.decode(errors="replace").strip().splitlines()[-1:] or [""]
        print(f"⏳ Docker not reachable yet ({stderr_tail[0]}) -- waiting {poll_interval_s:.0f}s (attempt {attempt})", file=sys.stderr, flush=True)
        time.sleep(poll_interval_s)


_VECTOR_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

# Copied verbatim from database-schema's `_ARTIST_EMBEDDINGS_STATEMENT` /
# `_ARTIST_EMBEDDINGS_MODEL_VERSION_INDEX` (src/groovemap_schema/postgres.py) -- the real
# landed DDL (gm-database-schema-lhp2), not the integration tier's inline stand-in (that
# stand-in predates lhp2 landing on origin/main and is now stale).
_ARTIST_EMBEDDINGS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS public.artist_embeddings (
        artist_id        TEXT NOT NULL,
        model_version    TEXT NOT NULL,
        embedding        halfvec(128) NOT NULL,
        source_dump_id   TEXT NOT NULL,
        source_dump_date DATE NOT NULL,
        computed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (artist_id, model_version)
    )
"""
_ARTIST_EMBEDDINGS_MODEL_VERSION_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_artist_embeddings_model_version ON public.artist_embeddings (model_version)"
)


def _deterministic_sample(candidates: list[str], n: int, seed: int) -> list[str]:
    """The smallest `n` of CANDIDATES by `splitmix64(node_key("a", id) XOR seed)`.

    Deterministic and repeatable for a fixed seed and candidate set -- the AC's "repeatable
    measurement" requirement. Uses the same hash family the pinned FastRP projection itself
    is built from (`insights.embeddings.projection.splitmix64`), not a fresh RNG.
    """
    scored = sorted(candidates, key=lambda aid: splitmix64(node_key("a", aid) ^ seed))
    return scored[:n]


def _load_month(path: Path) -> dict[str, Any]:
    # allow_pickle=True: the object-dtype artist_ids array needs it to deserialize. Safe
    # here -- this file is written by scripts/embeddings_from_dump.py on this same machine
    # in this same measurement workflow, never fetched from an untrusted source.
    with np.load(path, allow_pickle=True) as data:
        method_version = str(data["model_version"])
        dump_id = str(data["dump_id"])
        # `weights` (gm-analytics-engine-i37's w0 sweep, weights=w0,1,1,1,1): the actual tuple
        # `embeddings_from_dump.py`'s `FastRPConfig` was built from for THIS file, not always
        # `FastRPConfig()`'s w0=0 default -- a file from the sweep's w0=0.1 or w0=0.25 run has
        # a genuinely different `model_version`, and reconstructing `config` from the saved
        # weights (rather than assuming the default and rejecting anything else, this script's
        # pre-sweep behaviour) is what lets `stored_model_version` below compose the SAME
        # string for this file's own weights, whichever they are. An older npz without a
        # `weights` array predates the sweep and used the w0=0 default throughout.
        weights = tuple(data["weights"].tolist()) if "weights" in data else FastRPConfig().weights
        config = FastRPConfig(weights=weights)
        if method_version != config.model_version:
            raise ValueError(
                f"{path}: saved method_version {method_version!r} does not match "
                f"FastRPConfig(weights={weights}).model_version {config.model_version!r} reconstructed from "
                "this same file's saved weights -- the npz is inconsistent with itself."
            )
        # `degrees` (i37's degree-bucketed recall breakdown): each artist's undirected degree
        # in this month's graph, position-aligned with `artist_ids`/`vectors` -- this script
        # has no graph of its own, so a file saved before this bead (no `degrees` array) just
        # skips the degree breakdown rather than failing outright. Small enough (one int per
        # artist) to load eagerly, unlike `vectors` below.
        degrees = data.get("degrees", None)
        if degrees is None:
            print(
                f"⚠️  {path}: no saved `degrees` array (predates gm-analytics-engine-i37) -- degree-bucketed recall will be skipped", file=sys.stderr
            )
        return {
            "artist_ids": [str(aid) for aid in data["artist_ids"]],
            # NOT `data["vectors"]`: gm-analytics-engine-i37, 2026-09-28 -- that materializes
            # this month's whole (~5 GB at real catalog scale) vector array in RAM, and doing
            # this for both months at once is most of what swapped the host to 30 GB.
            # `path` is kept instead; every consumer that needs vectors streams them from it
            # in bounded chunks (`_iter_npz_vector_chunks`, `_extract_rows`,
            # `_stream_exact_top_k`) rather than holding the full array.
            "path": path,
            "degrees": degrees,
            # `embeddings_from_dump.py` saves the bare `FastRPConfig.model_version`
            # (`method_version` here), not production's *stored* value -- it never imports
            # `stored_model_version` at all. Composing it here, the same way production
            # does (`f"{config.model_version}:{_EDGE_SET_VERSION}@{dump_id}"`), matters for
            # more than labelling: without the dump id folded in, Aug and Sept would
            # resolve to the IDENTICAL model_version string (same `FastRPConfig()` for
            # both), which would collide on `_index_name`'s composed index name too --
            # Sept's `CREATE INDEX IF NOT EXISTS` would then silently no-op against
            # Aug's (by-then-truncated, so empty) index instead of building a real one.
            # Caught before this ran end to end against real data. `config` (not always the
            # w0=0 default any more) is what keeps two DIFFERENT w0 sweep runs of the SAME
            # month from colliding on that index name too.
            "method_version": method_version,
            "model_version": stored_model_version(config, dump_id),
            "dump_id": dump_id,
            "dump_date": str(data["dump_date"]),
        }


def _normalized(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize every row to unit length, float32. A zero row (no propagated signal
    reaches an isolated vertex) is left as all-zeros rather than divided by zero."""
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    norms[norms == 0] = 1.0
    return vectors.astype(np.float32) / norms[:, None]


def _exact_top_k(normalized: np.ndarray, query_positions: list[int], k: int) -> tuple[list[list[int]], list[float]]:
    """Exact top-`k` cosine neighbours (by position, excluding self) for each query
    position, brute-force in NumPy against every row of NORMALIZED, plus each query's exact
    k-th-place (last) score -- `_tie_tolerant_recall_at_k` needs that threshold, and it costs
    nothing extra to return alongside the top-k this function already computes.

    Takes an already-`_normalized` matrix (not raw vectors) so a caller sweeping several
    `ef_search` values, or also scoring arbitrary ANN candidates for tie-tolerance, normalizes
    the month's vectors ONCE rather than once per use.
    """
    results: list[list[int]] = []
    kth_scores: list[float] = []
    for position in query_positions:
        scores = normalized @ normalized[position]
        scores[position] = -np.inf  # exclude self
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        results.append(top.tolist())
        kth_scores.append(float(scores[top[-1]]) if len(top) else float("-inf"))
    return results, kth_scores


async def _apply_schema(conn: Any) -> None:
    async with conn.cursor() as cursor:
        await cursor.execute(_VECTOR_EXTENSION_SQL)
        await cursor.execute(_ARTIST_EMBEDDINGS_TABLE_SQL)
        await cursor.execute(_ARTIST_EMBEDDINGS_MODEL_VERSION_INDEX_SQL)


def _restart_container(*, name: str, image: str, shm_size: str, username: str, password: str, database: str) -> tuple[str, int]:
    """Stop the throwaway container (started with `--rm`, so stopping removes it) and start
    a fresh one of the same name/image/credentials but a new `--shm-size`, for September's
    higher `maintenance_work_mem` (the container's shared memory must be able to hold it --
    the dispatcher's condition). Returns the new (host, port) to connect to.

    A full container restart, not an in-place `TRUNCATE`, is deliberate: `--shm-size` is set
    at container creation and cannot be changed on a running container. A fresh container
    also starts genuinely empty (no leftover per-`model_version` HNSW index from a previous
    month at all, not just an emptied one) -- `_drop_index` is still called for the
    same-container (`TRUNCATE`-only) path in `main_async`, since that path keeps the
    container across months.

    `DOCKER`/`name`/`image` are all this script's own constants or caller-supplied
    identifiers, never attacker-controlled input, so a fixed-argument-list `subprocess.run`
    (S603) with a resolved full executable path (S607) is the deliberate shape here, not an
    oversight.
    """
    subprocess.run([DOCKER, "stop", name], check=True, capture_output=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [
            DOCKER,
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--publish",
            "127.0.0.1::5432",
            "--shm-size",
            shm_size,
            "--env",
            f"POSTGRES_USER={username}",
            "--env",
            f"POSTGRES_PASSWORD={password}",
            "--env",
            f"POSTGRES_DB={database}",
            image,
        ],
        check=True,
        capture_output=True,
    )
    for _attempt in range(60):
        ready = subprocess.run(  # noqa: S603
            [DOCKER, "exec", name, "pg_isready", "--username", username, "--dbname", database],
            capture_output=True,
        )
        if ready.returncode == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"container {name!r} did not become ready within 120s of restart")
    published = subprocess.run([DOCKER, "port", name, "5432/tcp"], check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603
    host, _, port = published.rpartition(":")
    return host or "127.0.0.1", int(port)


async def _write_chunk_via_copy(conn: Any, month: dict[str, Any], ids_slice: list[str], vectors_chunk: np.ndarray) -> None:
    """One COPY of IDS_SLICE/VECTORS_CHUNK -- pulled out of `_write_month` so its trial-vs-
    COPY branch can call this once per streamed chunk instead of once for "the remainder"
    of an in-memory array. `computed_at` is omitted from the column list, not passed as an
    explicit NULL: the real column is `TIMESTAMPTZ NOT NULL DEFAULT NOW()`, and a DEFAULT
    only fires when a COPY row's column list leaves it out entirely -- an explicit NULL (what
    an earlier version of this fallback passed) violates NOT NULL outright.
    """
    async with (
        conn.cursor() as cursor,
        cursor.copy(f"COPY {ARTIST_EMBEDDINGS_TABLE} (artist_id, model_version, embedding, source_dump_id, source_dump_date) FROM STDIN") as copy,
    ):
        for index in range(len(ids_slice)):
            vector_literal = "[" + ",".join(f"{value:g}" for value in vectors_chunk[index].tolist()) + "]"
            await copy.write_row((ids_slice[index], month["model_version"], vector_literal, month["dump_id"], month["dump_date"]))


async def _write_month(conn: Any, month: dict[str, Any]) -> dict[str, Any]:
    """Write one month's rows, timing a `TRIAL_BATCH_ROWS` trial through the real
    `_write_embeddings` first; falls back to `COPY` only if that trial extrapolates past
    `TRIAL_TIME_BUDGET_S` for the full month, per the maintainer's condition.

    Skips the write entirely if this stored `model_version` already has rows -- the same
    `_already_loaded` idempotency check `insights.embedding_pipeline.load_embeddings` itself
    runs -- so a script restart against a container this month's rows already reached
    (a crash after write but before the index build, say) doesn't repeat an ~18-minute write.

    Streams `month["path"]`'s vectors in `_WRITE_CHUNK_BYTES`-sized chunks
    (`_iter_npz_vector_chunks`) rather than holding the whole month's array at once
    (gm-analytics-engine-i37, 2026-09-28) -- the trial is measured on (up to)
    `TRIAL_BATCH_ROWS` rows from the FIRST chunk; every row after that, in that chunk and
    every subsequent one, goes through whichever path (`_write_embeddings` or COPY) the
    trial's extrapolation picked.
    """
    if await _already_loaded(conn, month["model_version"]):
        print("  already loaded (skipping write)", file=sys.stderr)
        return {"rows_written": len(month["artist_ids"]), "trial_elapsed_s": 0.0, "extrapolated_full_s": 0.0, "used_copy": False, "skipped": True}

    artist_ids = month["artist_ids"]
    total = len(artist_ids)

    rows_written = 0
    trial_elapsed = 0.0
    extrapolated_s = 0.0
    used_copy = False
    trial_measured = False
    position = 0

    for chunk in _iter_npz_vector_chunks(month["path"], chunk_bytes=_WRITE_CHUNK_BYTES):
        ids_slice = artist_ids[position : position + chunk.shape[0]]

        if not trial_measured:
            trial_n = min(TRIAL_BATCH_ROWS, chunk.shape[0])
            trial_started = time.perf_counter()
            trial_rows = await _write_embeddings(
                conn,
                model_version=month["model_version"],
                dump_id=month["dump_id"],
                dump_date=month["dump_date"],
                artist_ids=ids_slice[:trial_n],
                vectors=chunk[:trial_n],
            )
            trial_elapsed = time.perf_counter() - trial_started
            extrapolated_s = trial_elapsed * (total / trial_n) if trial_n else 0.0
            print(
                f"  trial: {trial_rows:,} rows in {trial_elapsed:.1f}s -> extrapolated full month {extrapolated_s:.0f}s "
                f"({extrapolated_s / 60:.1f} min)",
                file=sys.stderr,
            )
            rows_written += trial_rows
            trial_measured = True
            if extrapolated_s > TRIAL_TIME_BUDGET_S and trial_n < total:
                print(
                    f"  extrapolated time exceeds {TRIAL_TIME_BUDGET_S / 60:.0f} min budget -- falling back to COPY for the remainder",
                    file=sys.stderr,
                )
                used_copy = True
            rest_ids, rest_vectors = ids_slice[trial_n:], chunk[trial_n:]
        else:
            rest_ids, rest_vectors = ids_slice, chunk

        if rest_ids:
            if used_copy:
                await _write_chunk_via_copy(conn, month, rest_ids, rest_vectors)
                rows_written += len(rest_ids)
            else:
                remaining_rows = await _write_embeddings(
                    conn,
                    model_version=month["model_version"],
                    dump_id=month["dump_id"],
                    dump_date=month["dump_date"],
                    artist_ids=rest_ids,
                    vectors=rest_vectors,
                )
                rows_written += remaining_rows

        position += chunk.shape[0]

    print(f"  peak RSS after write: {_peak_rss_mb():.0f} MB", file=sys.stderr)
    return {"rows_written": rows_written, "trial_elapsed_s": trial_elapsed, "extrapolated_full_s": extrapolated_s, "used_copy": used_copy}


async def _build_index(
    conn: Any, model_version: str, maintenance_work_mem: str, *, m: int = HNSW_M, ef_construction: int = HNSW_EF_CONSTRUCTION
) -> dict[str, Any]:
    """Build the full-scale HNSW index for MODEL_VERSION's rows, named by `_index_name`.

    A per-`model_version` PARTIAL index (`WHERE model_version = ...`), exactly the shape
    `insights.embedding_pipeline._log_operator_step` logs as the real operator statement
    and `gm-database-schema-19g5` builds -- not a whole-table index. This matters even
    though this script only ever keeps one month's rows in the table at a time: a
    whole-table index, once built, is a live HNSW graph that keeps accepting inserts for
    *any* `model_version` written afterward (there is no `WHERE` filtering what the index
    accepts), so a later month's rows get added to the same graph one at a time via
    per-row HNSW maintenance instead of getting their own bulk-built graph -- silently
    correct but catastrophically slow (a real incident this bead hit: September's `COPY`
    crawled at ~20k rows/min, on pace for over 5 hours, because it was appending to
    August's already-built whole-table index instead of building its own).

    Full scale first, per the maintainer's condition: a subset fallback is used only after
    a real failure here, with that failure's wall time and peak memory recorded -- this
    function does not pre-emptively choose a smaller scale. `maintenance_work_mem` is a
    session-only `SET`, reverted after, matching `database-schema.build_artist_embeddings_
    index`'s own pattern -- the caller decides the value (August's first attempt used the
    documented 2GB; both months' final runs use more -- see docs/recall_and_churn.md).

    `m`/`ef_construction` default to the standard variant (`HNSW_M`/`HNSW_EF_CONSTRUCTION`);
    the caller passes `HNSW_LARGER_M`/`HNSW_LARGER_EF_CONSTRUCTION` for the larger-index
    variant (gm-analytics-engine-i37's maintainer-approved addition) instead. Since
    `_index_name` keys only on `model_version`, not on `m`/`ef_construction`, the two variants
    can never coexist under the same name -- exactly the point: they're built, measured, and
    dropped one at a time (`measure_index_variant`), never together, to keep Colima's VM
    within its memory budget.
    """
    index_name = _index_name(model_version)
    async with conn.cursor() as cursor:
        await cursor.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
        started = time.perf_counter()
        # A bind parameter in CREATE INDEX's WHERE clause hits psycopg's
        # `IndeterminateDatatype` (PostgreSQL can't infer the parameter's type in this DDL
        # context, confirmed against the live container before this ran for real) -- the
        # same class of limitation `_create_pipeline_login`'s own docstring notes for a `DO`
        # block's body. `_sql_string_literal` (the same helper `_log_operator_step` uses for
        # this exact WHERE clause) escapes it as a literal instead.
        await cursor.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} ON {ARTIST_EMBEDDINGS_TABLE} "
            f"USING hnsw (embedding halfvec_cosine_ops) WITH (m = {m}, ef_construction = {ef_construction}) "
            f"WHERE model_version = {_sql_string_literal(model_version)}"
        )
        elapsed = time.perf_counter() - started
        await cursor.execute("RESET maintenance_work_mem")
    return {
        "index_name": index_name,
        "build_elapsed_s": elapsed,
        "maintenance_work_mem": maintenance_work_mem,
        "m": m,
        "ef_construction": ef_construction,
    }


async def _index_size_bytes(conn: Any, model_version: str) -> int:
    """The live index's on-disk size in bytes (`pg_relation_size`) -- part of the
    maintainer-approved standard-vs-larger-variant comparison (index size is a real cost, not
    just a build-time one)."""
    index_name = _index_name(model_version)
    async with conn.cursor() as cursor:
        await cursor.execute(f"SELECT pg_relation_size({_sql_string_literal(index_name)}::regclass)")
        (size,) = await cursor.fetchone()
    return int(size)


async def _drop_index(conn: Any, model_version: str) -> None:
    """Drop the previous month's per-`model_version` HNSW index, before that month's rows
    are truncated -- so a later `CREATE INDEX IF NOT EXISTS` under a *different* name
    (per-`model_version`, so it never collides) always builds a fresh graph, and the old,
    now-empty-table index is never left around taking up (admittedly small) catalog space.
    """
    index_name = _index_name(model_version)
    async with conn.cursor() as cursor:
        await cursor.execute(f"DROP INDEX IF EXISTS {index_name}")


async def _ann_top_k(
    conn: Any,
    model_version: str,
    path: Path,
    normalized_cache: dict[int, np.ndarray],
    artist_ids: list[str],
    positions: list[int],
    ef_search: int,
    k: int,
    *,
    latencies_ms: list[float] | None = None,
) -> list[list[str] | None]:
    """The live index's top-`k` artist_ids for each query position's vector, at EF_SEARCH.

    Takes PATH and NORMALIZED_CACHE rather than a full in-memory `vectors` array
    (gm-analytics-engine-i37, 2026-09-28): `_ensure_normalized_cached` fetches (streaming,
    bounded memory) whatever query positions aren't already cached, then this queries with
    the cached NORMALIZED vector -- fine for `<=>` (cosine distance), which is invariant to
    scaling either side by a positive constant, so a normalized query vector orders results
    identically to the raw one the table itself stores.

    Excludes the query's own artist_id: its vector is stored verbatim in the table, so an
    un-excluded query always ranks itself first (distance 0), the same one-row exclusion
    `_exact_top_k` applies by setting its own score to `-inf`. Without this, ANN and exact
    are compared on a different footing (ANN off by exactly one true positive every time --
    caught by this script's own synthetic smoke test before it ran against real data).

    Deliberately NOT `WHERE artist_id != %s` alongside the `ORDER BY ... LIMIT` -- a second
    equality filter on top of the ANN ordering risks the planner choosing a different plan
    than a plain filtered-by-model_version index scan (relevant for the real, ~7-10M-row
    table this queries; not distinguishable on the synthetic smoke-test table this bug was
    caught on). Requesting `k + 1` rows with only the `model_version` filter and dropping
    the self row client-side gets the same result without touching the query plan at all.
    Same shape catalog-api's own kNN endpoint (gm-catalog-api-2zsq) will need for its
    self-exclusion, there behind a `model_version`-filtered *partial* index instead of this
    script's whole-table one -- see docs/recall_and_churn.md.

    `latencies_ms`, if given, gets one entry appended per query -- wall time for that single
    round trip (execute + fetch), in milliseconds -- for `measure_index_variant`'s mean/p95
    per-query latency report (gm-analytics-engine-i37's maintainer-approved index-variant
    comparison). `None` (the default, and what every pre-existing caller still gets) skips
    the timing calls entirely rather than paying for a throwaway list.
    """
    _ensure_normalized_cached(path, normalized_cache, positions)
    results: list[list[str] | None] = []
    async with conn.cursor() as cursor:
        await cursor.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        for position in positions:
            literal = "[" + ",".join(f"{value:g}" for value in normalized_cache[position].tolist()) + "]"
            query_started = time.perf_counter() if latencies_ms is not None else None
            await cursor.execute(
                f"SELECT artist_id FROM {ARTIST_EMBEDDINGS_TABLE} "  # noqa: S608
                f"WHERE model_version = %s ORDER BY embedding <=> %s::halfvec LIMIT %s",
                (model_version, literal, k + 1),
            )
            rows = await cursor.fetchall()
            if query_started is not None:
                latencies_ms.append((time.perf_counter() - query_started) * 1000.0)  # type: ignore[union-attr]
            own_id = artist_ids[position]
            results.append([artist_id for (artist_id,) in rows if artist_id != own_id][:k])
    return results


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching `numpy.percentile`'s default -- pulled out to
    a plain function so a caller with a `list[float]` (latency samples) doesn't need to build
    a NumPy array just to ask for one number."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct)) if values else 0.0


def _recall_at_k(ann: list[list[str]], exact_ids: list[list[str]], k: int) -> float:
    scores = []
    for ann_ids, exact in zip(ann, exact_ids, strict=True):
        if not exact:
            continue
        overlap = len(set(ann_ids[:k]) & set(exact[:k]))
        scores.append(overlap / min(k, len(exact)))
    return sum(scores) / len(scores) if scores else 0.0


def _tie_tolerant_recall_at_k(
    ann: list[list[str]],
    exact_ids: list[list[str]],
    kth_scores: list[float],
    query_positions: list[int],
    normalized_cache: dict[int, np.ndarray],
    id_to_position: dict[str, int],
    k: int,
) -> float:
    """`_recall_at_k`'s tie-tolerant counterpart (see `TIE_TOLERANCE`'s docstring for the
    definition): an ANN candidate outside the exact top-`k` still counts as a hit when its
    own exact cosine similarity to the query is within `TIE_TOLERANCE` of `kth_scores`, the
    exact k-th-place score `_exact_top_k`/`_stream_exact_top_k` already computed for that
    same query.

    Takes NORMALIZED_CACHE (position -> normalized vector), not a full in-memory matrix
    (gm-analytics-engine-i37, 2026-09-28) -- the caller (`_sweep_recall`) ensures every
    candidate position this loop might look up is already cached before calling in, via
    `_ensure_normalized_cached`, so this stays a plain dict lookup rather than doing I/O
    itself. A candidate this month's `id_to_position` doesn't recognise (shouldn't happen --
    both come from the same month's `artist_ids` -- but the ANN index is a live, separately-
    queried system), or one somehow still missing from the cache, is treated as a miss
    rather than raising, the same permissive stance `_recall_at_k`'s plain set-intersection
    already takes toward an ANN id absent from the exact top-`k`.
    """
    scores = []
    for ann_ids, exact, kth_score, query_position in zip(ann, exact_ids, kth_scores, query_positions, strict=True):
        if not exact:
            continue
        exact_set = set(exact[:k])
        hits = 0
        for candidate in ann_ids[:k]:
            if candidate in exact_set:
                hits += 1
                continue
            candidate_position = id_to_position.get(candidate)
            if candidate_position is None:
                continue
            candidate_vector = normalized_cache.get(candidate_position)
            if candidate_vector is None:
                continue
            candidate_score = float(candidate_vector @ normalized_cache[query_position])
            if candidate_score >= kth_score - TIE_TOLERANCE:
                hits += 1
        scores.append(hits / min(k, len(exact)))
    return sum(scores) / len(scores) if scores else 0.0


def _degree_bucket_label(degree: int) -> str:
    """A human-readable power-of-two bucket label for one artist's undirected graph degree,
    e.g. ``"4-7"``, ``"256+"`` -- see `DEGREE_BUCKET_EDGES`."""
    if degree < DEGREE_BUCKET_EDGES[0]:
        return "0"
    for lo, hi in itertools.pairwise(DEGREE_BUCKET_EDGES):
        if lo <= degree < hi:
            return f"{lo}-{hi - 1}"
    return f"{DEGREE_BUCKET_EDGES[-1]}+"


def _degree_bucket_sort_key(label: str) -> int:
    """Numeric sort key for a `_degree_bucket_label` string: its own lower bound (``"0"`` ->
    0, ``"4-7"`` -> 4, ``"256+"`` -> 256) -- a plain string sort would put ``"128-255"``
    before ``"16-31"`` (lexical, not numeric, order)."""
    return int(label.rstrip("+").split("-")[0])


def _recall_by_degree_bucket(
    ann_ids_by_query: list[list[str]],
    exact_ids_by_query: list[list[str]],
    kth_scores: list[float],
    query_positions: list[int],
    degrees: np.ndarray | None,
    normalized_cache: dict[int, np.ndarray],
    id_to_position: dict[str, int],
    k: int,
) -> dict[str, dict[str, Any]] | None:
    """Strict and tie-tolerant recall@`k`, grouped by the QUERY artist's degree bucket in
    THIS month's graph (gm-analytics-engine-i37: kn3's own measurement has no graph, hence no
    degree, to compute this from). Free of any extra ANN queries -- it re-groups the SAME
    per-query `ann_ids_by_query`/`exact_ids_by_query` rows the caller's `ef_search` sweep
    already fetched for the full query sample, it never re-queries the index.

    Returns `None` (rather than an empty dict) when `degrees` itself is `None` -- an npz
    saved before this bead -- so a caller can tell "not computed" apart from "computed, but
    every bucket happened to be empty".
    """
    if degrees is None:
        return None
    buckets: dict[str, list[int]] = {}
    for index, position in enumerate(query_positions):
        buckets.setdefault(_degree_bucket_label(int(degrees[position])), []).append(index)
    result: dict[str, dict[str, Any]] = {}
    for label, indices in sorted(buckets.items(), key=lambda item: _degree_bucket_sort_key(item[0])):
        ann_subset = [ann_ids_by_query[i] for i in indices]
        exact_subset = [exact_ids_by_query[i] for i in indices]
        kth_subset = [kth_scores[i] for i in indices]
        positions_subset = [query_positions[i] for i in indices]
        result[label] = {
            "n": len(indices),
            "recall_strict": _recall_at_k(ann_subset, exact_subset, k),
            "recall_tie_tolerant": _tie_tolerant_recall_at_k(
                ann_subset, exact_subset, kth_subset, positions_subset, normalized_cache, id_to_position, k
            ),
        }
    return result


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


async def _sweep_recall(
    conn: Any,
    month: dict[str, Any],
    *,
    label: str,
    query_sample_positions: list[int],
    exact_ids_by_query: list[list[str]],
    kth_scores: list[float],
    normalized_cache: dict[int, np.ndarray],
    id_to_position: dict[str, int],
) -> dict[str, Any]:
    """Strict + tie-tolerant + degree-bucketed recall@10, AND per-query ANN latency, across
    `EF_SEARCH_SWEEP`, against whatever HNSW index for `month["model_version"]` is currently
    live. Shared by `measure_index_variant` for both the standard and larger-index variants
    (gm-analytics-engine-i37) -- everything here is variant-agnostic; `measure_index_variant`
    is what builds/drops the index around this call.

    NORMALIZED_CACHE (position -> normalized vector) replaces a full in-memory matrix
    (gm-analytics-engine-i37, 2026-09-28): each `ef_search` iteration fetches whichever new
    candidate positions this sweep point's ANN results introduced (`_ensure_normalized_cached`,
    streaming, bounded memory) before scoring tie-tolerance against them, and the host-
    pressure guard (`wait_for_host_pressure`) runs between iterations too, not just before
    this function is entered.
    """
    print(f"=== {label}: recall@10 sweep (strict + tie-tolerant + degree-bucketed + latency) ===", file=sys.stderr)
    recall_by_ef: dict[int, float] = {}
    recall_tie_tolerant_by_ef: dict[int, float] = {}
    recall_by_ef_by_degree_bucket: dict[int, dict[str, dict[str, Any]] | None] = {}
    latency_ms_by_ef: dict[int, dict[str, float]] = {}
    production_ef_search: int | None = None
    for ef_search in EF_SEARCH_SWEEP:
        wait_for_host_pressure(label=f"{label} (ef_search={ef_search}): ")
        ann_started = time.perf_counter()
        latencies_ms: list[float] = []
        ann_ids_by_query = await _ann_top_k(
            conn,
            month["model_version"],
            month["path"],
            normalized_cache,
            month["artist_ids"],
            query_sample_positions,
            ef_search,
            10,
            latencies_ms=latencies_ms,
        )
        ann_elapsed = time.perf_counter() - ann_started
        recall = _recall_at_k(ann_ids_by_query, exact_ids_by_query, 10)
        # Every candidate the ANN index just returned needs its own normalized vector for
        # tie-tolerance scoring below -- fetch (streaming) whichever of them isn't already
        # cached, in ONE pass, before either recall breakdown looks any of them up.
        candidate_positions = {id_to_position[candidate] for ann_ids in ann_ids_by_query for candidate in ann_ids[:10] if candidate in id_to_position}
        _ensure_normalized_cached(month["path"], normalized_cache, candidate_positions)
        tie_recall = _tie_tolerant_recall_at_k(
            ann_ids_by_query, exact_ids_by_query, kth_scores, query_sample_positions, normalized_cache, id_to_position, 10
        )
        recall_by_ef[ef_search] = recall
        recall_tie_tolerant_by_ef[ef_search] = tie_recall
        # gm-analytics-engine-i37: re-groups this SAME sweep point's per-query results by the
        # query artist's degree in this month's graph -- no extra ANN queries, kn3's own
        # measurement has no graph to compute this breakdown from at all.
        recall_by_ef_by_degree_bucket[ef_search] = _recall_by_degree_bucket(
            ann_ids_by_query, exact_ids_by_query, kth_scores, query_sample_positions, month.get("degrees"), normalized_cache, id_to_position, 10
        )
        latency_ms_by_ef[ef_search] = {
            "mean_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
            "p95_ms": _percentile(latencies_ms, 95),
        }
        print(
            f"  ef_search={ef_search}: recall@10={recall:.4f} (tie-tolerant {tie_recall:.4f}) "
            f"latency mean={latency_ms_by_ef[ef_search]['mean_ms']:.2f}ms p95={latency_ms_by_ef[ef_search]['p95_ms']:.2f}ms "
            f"({ann_elapsed:.1f}s for {len(query_sample_positions):,} queries)",
            file=sys.stderr,
        )
        if production_ef_search is None and recall >= RECALL_TARGET:
            production_ef_search = ef_search

    print(f"  peak RSS after sweep: {_peak_rss_mb():.0f} MB", file=sys.stderr)
    return {
        "recall_by_ef_search": recall_by_ef,
        "recall_tie_tolerant_by_ef_search": recall_tie_tolerant_by_ef,
        "recall_by_ef_search_by_degree_bucket": recall_by_ef_by_degree_bucket,
        "latency_ms_by_ef_search": latency_ms_by_ef,
        "production_ef_search": production_ef_search,
    }


async def measure_index_variant(
    conn: Any,
    month: dict[str, Any],
    *,
    label: str,
    m: int,
    ef_construction: int,
    maintenance_work_mem: str,
    query_sample_positions: list[int],
    exact_ids_by_query: list[list[str]],
    kth_scores: list[float],
    normalized_cache: dict[int, np.ndarray],
    id_to_position: dict[str, int],
    drop_after: bool,
) -> dict[str, Any]:
    """Build ONE HNSW index variant (`m`/`ef_construction`), measure it (build time, on-disk
    size, then the full `_sweep_recall`), and -- unless `drop_after` is False -- drop it
    before returning. `drop_after=False` is for the standard variant inside `measure_month`,
    whose index `churn_top_k` still needs to query after this returns; every other caller
    (the larger-index variant, gm-analytics-engine-i37's maintainer-approved addition) drops
    it, so at most one index for this `model_version` exists in Postgres/Colima at a time --
    "build the variants one at a time... drop each index after measuring" is a memory
    constraint on the host running Colima's VM, not a suggestion.
    """
    wait_for_host_pressure(label=f"{label} variant, before index build: ")
    print(
        f"=== {label} variant (m={m}, ef_construction={ef_construction}): building (maintenance_work_mem={maintenance_work_mem}) ===", file=sys.stderr
    )
    index_result = await _build_index(conn, month["model_version"], maintenance_work_mem, m=m, ef_construction=ef_construction)
    index_size_bytes = await _index_size_bytes(conn, month["model_version"])
    print(f"  {index_result['index_name']}: {index_result['build_elapsed_s']:.1f}s, size {index_size_bytes / 1e6:.1f} MB", file=sys.stderr)

    sweep = await _sweep_recall(
        conn,
        month,
        label=f"{label} ({m=}, {ef_construction=})",
        query_sample_positions=query_sample_positions,
        exact_ids_by_query=exact_ids_by_query,
        kth_scores=kth_scores,
        normalized_cache=normalized_cache,
        id_to_position=id_to_position,
    )

    if drop_after:
        print(f"=== {label} variant: dropping index ({index_result['index_name']}) ===", file=sys.stderr)
        await _drop_index(conn, month["model_version"])

    return {"index": index_result, "index_size_bytes": index_size_bytes, **sweep}


class ExactGroundTruth(NamedTuple):
    """The brute-force ground truth for one month's query sample -- computed ONCE per month
    (`compute_exact_ground_truth`) and reused for both the standard and larger-index
    variants, since neither depends on which HNSW variant happens to be live: recomputing it
    a second time for the larger variant would cost another ~exact_elapsed_s (measured at
    ~4 minutes on edges-v2's real catalog scale) for byte-identical output.

    `cache` (position -> normalized vector) replaces what used to be a full in-memory
    `normalized` matrix (gm-analytics-engine-i37, 2026-09-28) -- it starts out seeded with
    just the query positions (`_stream_exact_top_k` populates it as a side effect) and grows
    on demand as `_sweep_recall` looks up new ANN candidate positions, never holding more
    than the union of positions actually looked up.
    """

    cache: dict[int, np.ndarray]
    id_to_position: dict[str, int]
    exact_ids_by_query: list[list[str]]
    kth_scores: list[float]
    exact_elapsed_s: float


def compute_exact_ground_truth(month: dict[str, Any], query_sample_positions: list[int], *, label: str) -> ExactGroundTruth:
    print(f"=== {label}: exact ground truth ({len(query_sample_positions):,} queries) ===", file=sys.stderr)
    exact_started = time.perf_counter()
    cache: dict[int, np.ndarray] = {}
    id_to_position = {aid: index for index, aid in enumerate(month["artist_ids"])}
    exact_positions_by_query, kth_scores = _stream_exact_top_k(month["path"], query_sample_positions, 10, cache=cache)
    exact_ids_by_query = [[month["artist_ids"][position] for position in row] for row in exact_positions_by_query]
    exact_elapsed = time.perf_counter() - exact_started
    print(f"  exact: {exact_elapsed:.1f}s (peak RSS {_peak_rss_mb():.0f} MB)", file=sys.stderr)
    return ExactGroundTruth(cache, id_to_position, exact_ids_by_query, kth_scores, exact_elapsed)


async def measure_month(
    conn: Any,
    month: dict[str, Any],
    *,
    label: str,
    query_sample_positions: list[int],
    query_sample_ids: list[str],
    maintenance_work_mem: str,
    ground_truth: ExactGroundTruth,
) -> dict[str, Any]:
    wait_for_host_pressure(label=f"{label}, before write: ")
    print(f"\n=== {label}: writing embeddings ===", file=sys.stderr)
    write_result = await _write_month(conn, month)
    print(f"  wrote {write_result['rows_written']:,} rows", file=sys.stderr)

    # The standard variant's index is kept alive (drop_after=False): churn_top_k, called by
    # the caller right after this returns, still needs to query it at churn_ef_search_used.
    variant = await measure_index_variant(
        conn,
        month,
        label="standard",
        m=HNSW_M,
        ef_construction=HNSW_EF_CONSTRUCTION,
        maintenance_work_mem=maintenance_work_mem,
        query_sample_positions=query_sample_positions,
        exact_ids_by_query=ground_truth.exact_ids_by_query,
        kth_scores=ground_truth.kth_scores,
        normalized_cache=ground_truth.cache,
        id_to_position=ground_truth.id_to_position,
        drop_after=False,
    )

    churn_ef_search = variant["production_ef_search"] or EF_SEARCH_SWEEP[-1]
    print(
        f"=== {label}: churn top-10 at ef_search={churn_ef_search} ({len(query_sample_ids)} query positions reused for churn sample separately) ===",
        file=sys.stderr,
    )

    return {
        "write": write_result,
        "exact_elapsed_s": ground_truth.exact_elapsed_s,
        **variant,
        "churn_ef_search_used": churn_ef_search,
    }


def exact_churn_top_k(month: dict[str, Any], churn_sample_ids: list[str]) -> dict[str, list[str]]:
    """EXACT top-10 (brute-force, no Postgres/index touched at all) for CHURN_SAMPLE_IDS,
    keyed by artist_id -- the index-free half of `churn_top_k`, split out so a month whose
    Postgres/index work is skipped entirely can still get its churn side measured
    (gm-analytics-engine-i37's maintainer-approved recall-phase trim: exact churn runs for
    every w0, ANN churn only for the single best one -- see `main_async`). Streams
    `month["path"]`'s vectors in bounded chunks (`_stream_exact_top_k`) rather than
    materializing the whole month's array (gm-analytics-engine-i37, 2026-09-28)."""
    id_to_position = {aid: index for index, aid in enumerate(month["artist_ids"])}
    positions = [id_to_position[aid] for aid in churn_sample_ids if aid in id_to_position]
    exact, _kth_scores = _stream_exact_top_k(month["path"], positions, 10)
    print(f"  peak RSS after exact churn: {_peak_rss_mb():.0f} MB", file=sys.stderr)
    return {month["artist_ids"][position]: [month["artist_ids"][n] for n in row] for position, row in zip(positions, exact, strict=True)}


async def churn_top_k(conn: Any, month: dict[str, Any], churn_sample_ids: list[str], ef_search: int) -> dict[str, list[str]]:
    """Both exact and ANN top-10 (at EF_SEARCH) for CHURN_SAMPLE_IDS, keyed by artist_id.

    Exact: `exact_churn_top_k`, brute-force, streamed straight from PATH -- see its own
    docstring. ANN: the live index, at the production `ef_search` -- what the served list
    would be. Uses a throwaway cache dict (gm-analytics-engine-i37, 2026-09-28), not the
    recall sweep's -- churn only ever queries ONE ef_search value, once, so there is no reuse
    to share a longer-lived cache for.
    """
    wait_for_host_pressure(label="churn: ")
    exact_by_id = exact_churn_top_k(month, churn_sample_ids)
    id_to_position = {aid: index for index, aid in enumerate(month["artist_ids"])}
    positions = [id_to_position[aid] for aid in churn_sample_ids if aid in id_to_position]
    ann = await _ann_top_k(conn, month["model_version"], month["path"], {}, month["artist_ids"], positions, ef_search, 10)
    ann_by_id = {month["artist_ids"][position]: (row or []) for position, row in zip(positions, ann, strict=True)}
    print(f"  peak RSS after churn: {_peak_rss_mb():.0f} MB", file=sys.stderr)
    return {"exact": exact_by_id, "ann": ann_by_id}


def compute_churn(aug_top10: dict[str, list[str]], sept_top10: dict[str, list[str]], sample_ids: list[str]) -> dict[str, float]:
    jaccards = [_jaccard(set(aug_top10.get(aid, [])), set(sept_top10.get(aid, []))) for aid in sample_ids if aid in aug_top10 and aid in sept_top10]
    return {"mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0, "n": len(jaccards)}


def _save_checkpoint(
    path: Path, aug_result: dict[str, Any], aug_churn: dict[str, Any], aug_larger_variant: dict[str, Any] | None, model_version: str
) -> None:
    """Persist August's full result (recall sweep + churn top-10 + the larger-index variant,
    if measured) to disk immediately after it's computed, before September starts -- so a
    restart between months (a container swap for a higher maintenance_work_mem, say) can skip
    re-doing August's multi-hour build entirely, per the dispatcher's ask about resumability.
    Keyed on August's own stored model_version so a checkpoint from a different config/dump is
    never reused."""
    path.write_text(
        json.dumps(
            {"model_version": model_version, "aug_result": aug_result, "aug_churn": aug_churn, "aug_larger_variant": aug_larger_variant}, default=str
        )
    )


def _load_checkpoint(path: Path, model_version: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("model_version") != model_version:
        print(f"  checkpoint at {path} is for a different model_version ({data.get('model_version')!r}); ignoring", file=sys.stderr)
        return None
    # `aug_larger_variant` predates a checkpoint saved before this bead's addition; `.get`
    # rather than a plain key lookup lets an older checkpoint still resume (just without
    # that data, same as `_load_month`'s own `degrees`-missing fallback).
    return data["aug_result"], data["aug_churn"], data.get("aug_larger_variant")


def is_result_complete(path: Path) -> bool:
    """True when PATH is a `--out` result this script itself wrote, and it is non-partial --
    i.e. a full, finished run (`_main_async_trimmed`/`main_async`'s final return), not a
    partial checkpoint written before a since-interrupted later step. Any read/parse failure
    (missing file, truncated JSON, wrong shape) is treated as "not complete" rather than
    raised -- an orchestrator asking "can I skip re-running this?" should get a plain no, the
    same way a never-run weight would.

    gm-analytics-engine-i37, 2026-09-28: the recall+quality driver was killed by a host
    memory/disk crisis mid-loop, after one weight's result had already been written
    non-partial but before the driver's own state file recorded that -- this is the check
    that lets a restart recognize "this weight is actually done" from the result file alone,
    without redoing a possibly multi-hour build.
    """
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError, OSError:
        return False
    return isinstance(data, dict) and data.get("partial") is False


async def _main_async_trimmed(
    args: argparse.Namespace,
    aug: dict[str, Any],
    sept: dict[str, Any],
    common_ids: list[str],
    churn_sample_ids: list[str],
    sept_query_ids: list[str],
    sept_query_positions: list[int],
) -> dict[str, Any]:
    """gm-analytics-engine-i37's maintainer-approved recall-phase trim (`--skip-ann-churn`):
    process ONLY September's Postgres/index work (standard-variant recall sweep, and the
    larger-index variant unless `--skip-larger-variant` is also set -- see that flag's own
    help text for why it usually is: the m=32 build overflowed maintenance_work_mem and fell
    to an ~8h on-disk path per build). August contributes only its raw vectors, for EXACT
    churn -- no Postgres write, no index, no recall sweep, and no ANN churn for August at all.

    Writes a PARTIAL result to `args.out` right after September's standard-variant recall
    sweep and the exact churn are both in hand, BEFORE attempting the larger variant (if not
    skipped) -- so a cancel during that step (the m=32 overflow that prompted `--skip-larger-
    variant` in the first place) never loses the numbers that already exist.

    Run this for every w0 except the single best one (selected afterward by tie-tolerant
    recall + quality, from September's results across the sweep -- see docs/
    embedding_weight_sweep.md for the exact rule and the winner), which instead runs the full
    (non-trimmed) flow in `main_async` below, the only place ANN churn is computed.
    """
    print(
        "\n=== --skip-ann-churn: September only (write+index+recall+larger variant); August contributes raw vectors for exact churn only ===",
        file=sys.stderr,
    )
    pool = AsyncPostgreSQLPool(
        connection_params={"host": args.host, "port": args.port, "dbname": args.database, "user": args.username, "password": args.password},
        min_connections=1,
        max_connections=1,
    )
    await pool.initialize()
    try:
        async with pool.connection() as conn:
            await _apply_schema(conn)
            wait_for_host_pressure(label="September ground truth: ")
            sept_ground_truth = compute_exact_ground_truth(sept, sept_query_positions, label="September")
            sept_result = await measure_month(
                conn,
                sept,
                label="September",
                query_sample_positions=sept_query_positions,
                query_sample_ids=sept_query_ids,
                maintenance_work_mem=args.sept_maintenance_work_mem,
                ground_truth=sept_ground_truth,
            )

            wait_for_host_pressure(label="exact churn: ")
            print("\n=== September: exact-only churn against August's raw vectors (no ANN -- trimmed mode) ===", file=sys.stderr)
            aug_churn_exact = exact_churn_top_k(aug, churn_sample_ids)
            sept_churn_exact = exact_churn_top_k(sept, churn_sample_ids)
            churn_exact = compute_churn(aug_churn_exact, sept_churn_exact, churn_sample_ids)
            print(f"  churn (exact cosine): mean_jaccard={churn_exact['mean_jaccard']:.4f} (n={churn_exact['n']})", file=sys.stderr)

            partial = {
                "mode": "skip_ann_churn",
                "partial": True,
                "sampling": {
                    "churn_sample_seed": CHURN_SAMPLE_SEED,
                    "churn_sample_size_requested": CHURN_SAMPLE_SIZE,
                    "churn_sample_size_actual": len(churn_sample_ids),
                    "common_artists": len(common_ids),
                    "rule": "smallest N by splitmix64(node_key('a', artist_id) XOR seed)",
                },
                "august": {
                    "dump_id": aug["dump_id"],
                    "dump_date": aug["dump_date"],
                    "model_version": aug["model_version"],
                    "n_vectors": len(aug["artist_ids"]),
                    "note": "no Postgres/index work in this mode -- raw vectors only, for exact churn",
                },
                "september": {
                    "dump_id": sept["dump_id"],
                    "dump_date": sept["dump_date"],
                    "model_version": sept["model_version"],
                    "n_vectors": len(sept["artist_ids"]),
                    **sept_result,
                },
                "churn_exact_cosine": churn_exact,
                "churn_ann_at_production_ef_search": None,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(partial, indent=2, default=str))
            print(f"  partial result (standard variant + exact churn, before any larger-variant attempt) written to {args.out}", file=sys.stderr)

            if args.skip_larger_variant:
                print("=== September: --skip-larger-variant set -- skipping the larger-index (m=32) variant ===", file=sys.stderr)
                sept_larger_variant = {"skipped": True, "reason": LARGER_VARIANT_SKIP_REASON}
            else:
                print(
                    f"\n=== September: dropping the standard index ({sept_result['index']['index_name']}) before the larger-index variant ===",
                    file=sys.stderr,
                )
                await _drop_index(conn, sept["model_version"])

                sept_larger_variant = await measure_index_variant(
                    conn,
                    sept,
                    label="larger",
                    m=HNSW_LARGER_M,
                    ef_construction=HNSW_LARGER_EF_CONSTRUCTION,
                    maintenance_work_mem=args.sept_maintenance_work_mem,
                    query_sample_positions=sept_query_positions,
                    exact_ids_by_query=sept_ground_truth.exact_ids_by_query,
                    kth_scores=sept_ground_truth.kth_scores,
                    normalized_cache=sept_ground_truth.cache,
                    id_to_position=sept_ground_truth.id_to_position,
                    drop_after=True,
                )
    finally:
        await pool.close()

    result = dict(partial)
    result["partial"] = False
    result["september"] = {**result["september"], "larger_index_variant": sept_larger_variant}
    return result


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    print("=== waiting for Docker to be reachable ===", file=sys.stderr)
    wait_for_docker()
    wait_for_host_pressure(label="before loading months: ")

    print(f"loading {args.aug}", file=sys.stderr)
    aug = _load_month(args.aug)
    print(f"loading {args.sept}", file=sys.stderr)
    sept = _load_month(args.sept)

    common_ids = sorted(set(aug["artist_ids"]) & set(sept["artist_ids"]))
    print(f"common artists (both months): {len(common_ids):,}", file=sys.stderr)
    churn_sample_ids = _deterministic_sample(common_ids, CHURN_SAMPLE_SIZE, CHURN_SAMPLE_SEED)

    sept_query_ids = _deterministic_sample(sept["artist_ids"], QUERY_SAMPLE_SIZE, QUERY_SAMPLE_SEED)
    sept_id_to_position = {aid: index for index, aid in enumerate(sept["artist_ids"])}
    sept_query_positions = [sept_id_to_position[aid] for aid in sept_query_ids]

    if args.skip_ann_churn:
        return await _main_async_trimmed(args, aug, sept, common_ids, churn_sample_ids, sept_query_ids, sept_query_positions)

    # ---- Full flow below: both months' Postgres/index work, ANN churn between them. Used
    # only for the single best w0 (see _main_async_trimmed's docstring) -- every other w0
    # takes the trimmed branch above and returns before reaching here. ----
    aug_query_ids = _deterministic_sample(aug["artist_ids"], QUERY_SAMPLE_SIZE, QUERY_SAMPLE_SEED)
    aug_id_to_position = {aid: index for index, aid in enumerate(aug["artist_ids"])}
    aug_query_positions = [aug_id_to_position[aid] for aid in aug_query_ids]

    checkpoint_path = args.out.with_suffix(".august_checkpoint.json")
    checkpoint = _load_checkpoint(checkpoint_path, aug["model_version"])
    if checkpoint is not None:
        print(f"\n=== August: resuming from checkpoint {checkpoint_path} (skipping write/build/sweep) ===", file=sys.stderr)
        aug_result, aug_churn, aug_larger_variant = checkpoint
        if aug_larger_variant is None:
            print(
                "  ⚠️  checkpoint has no larger-index-variant result (predates gm-analytics-engine-i37, or that "
                "step hadn't run yet) -- skipping it for August rather than backfilling: the checkpoint-resume "
                "path assumes no further container work is needed for August, and backfilling would need one.",
                file=sys.stderr,
            )
    else:
        pool = AsyncPostgreSQLPool(
            connection_params={"host": args.host, "port": args.port, "dbname": args.database, "user": args.username, "password": args.password},
            min_connections=1,
            max_connections=1,
        )
        await pool.initialize()
        async with pool.connection() as conn:
            await _apply_schema(conn)
            wait_for_host_pressure(label="August ground truth: ")
            aug_ground_truth = compute_exact_ground_truth(aug, aug_query_positions, label="August")
            aug_result = await measure_month(
                conn,
                aug,
                label="August",
                query_sample_positions=aug_query_positions,
                query_sample_ids=aug_query_ids,
                maintenance_work_mem=args.aug_maintenance_work_mem,
                ground_truth=aug_ground_truth,
            )
            print("\n=== August: churn top-10 (exact + ANN) for the common-artist sample, BEFORE dropping the table ===", file=sys.stderr)
            aug_churn = await churn_top_k(conn, aug, churn_sample_ids, aug_result["churn_ef_search_used"])

            print(
                f"\n=== August: dropping the standard index ({aug_result['index']['index_name']}) before the larger-index variant ===",
                file=sys.stderr,
            )
            await _drop_index(conn, aug["model_version"])

            if args.skip_larger_variant:
                print("=== August: --skip-larger-variant set -- skipping the larger-index (m=32) variant ===", file=sys.stderr)
                aug_larger_variant = {"skipped": True, "reason": LARGER_VARIANT_SKIP_REASON}
            elif args.skip_larger_variant_aug:
                print("=== August: --skip-larger-variant-aug set -- measuring the larger variant on September only ===", file=sys.stderr)
                aug_larger_variant = {"skipped": True, "reason": "--skip-larger-variant-aug: measured on September only"}
            else:
                aug_larger_variant = await measure_index_variant(
                    conn,
                    aug,
                    label="larger",
                    m=HNSW_LARGER_M,
                    ef_construction=HNSW_LARGER_EF_CONSTRUCTION,
                    maintenance_work_mem=args.aug_maintenance_work_mem,
                    query_sample_positions=aug_query_positions,
                    exact_ids_by_query=aug_ground_truth.exact_ids_by_query,
                    kth_scores=aug_ground_truth.kth_scores,
                    normalized_cache=aug_ground_truth.cache,
                    id_to_position=aug_ground_truth.id_to_position,
                    drop_after=True,
                )
        await pool.close()
        _save_checkpoint(checkpoint_path, aug_result, aug_churn, aug_larger_variant, aug["model_version"])
        print(f"  checkpoint saved: {checkpoint_path}", file=sys.stderr)

    if checkpoint is not None:
        # August's phase never touched a container in THIS invocation (it was skipped
        # entirely), so there is nothing here to restart -- the caller is responsible for
        # having --host/--port already point at a container provisioned for September's
        # settings before resuming from a checkpoint.
        print(f"\n=== using the given container ({args.host}:{args.port}) for September (resumed from checkpoint) ===", file=sys.stderr)
    elif args.sept_maintenance_work_mem != args.aug_maintenance_work_mem:
        print(
            f"\n=== restarting {args.container_name} with --shm-size {args.sept_shm_size} for September's "
            f"maintenance_work_mem={args.sept_maintenance_work_mem} (dispatcher-approved side-by-side comparison) ===",
            file=sys.stderr,
        )
        new_host, new_port = _restart_container(
            name=args.container_name,
            image=args.image,
            shm_size=args.sept_shm_size,
            username=args.username,
            password=args.password,
            database=args.database,
        )
        print(f"  restarted: {new_host}:{new_port}", file=sys.stderr)
        args.host, args.port = new_host, new_port
    else:
        print("\n=== dropping August's index and table before September (same container/mwm) ===", file=sys.stderr)

    pool = AsyncPostgreSQLPool(
        connection_params={"host": args.host, "port": args.port, "dbname": args.database, "user": args.username, "password": args.password},
        min_connections=1,
        max_connections=1,
    )
    await pool.initialize()
    try:
        async with pool.connection() as conn:
            await _apply_schema(conn)
            if checkpoint is None and args.sept_maintenance_work_mem == args.aug_maintenance_work_mem:
                # Same container carried over from August (and August actually ran in THIS
                # invocation, not resumed from a checkpoint against a fresh one): drop its
                # per-model_version index explicitly, then truncate its rows. Both are
                # per-`model_version` (index name and the index's own `WHERE` predicate), so
                # this was never a hard requirement for correctness -- a stale, now-empty
                # partial index for August's model_version doesn't accept September's rows
                # either way -- but it's cheap cleanup and keeps the catalog tidy.
                await _drop_index(conn, aug["model_version"])
                async with conn.cursor() as cursor:
                    await cursor.execute(f"TRUNCATE {ARTIST_EMBEDDINGS_TABLE}")

            wait_for_host_pressure(label="September ground truth: ")
            sept_ground_truth = compute_exact_ground_truth(sept, sept_query_positions, label="September")
            sept_result = await measure_month(
                conn,
                sept,
                label="September",
                query_sample_positions=sept_query_positions,
                query_sample_ids=sept_query_ids,
                maintenance_work_mem=args.sept_maintenance_work_mem,
                ground_truth=sept_ground_truth,
            )
            print("\n=== September: churn top-10 (exact + ANN) for the common-artist sample ===", file=sys.stderr)
            sept_churn = await churn_top_k(conn, sept, churn_sample_ids, sept_result["churn_ef_search_used"])

            churn_exact = compute_churn(aug_churn["exact"], sept_churn["exact"], churn_sample_ids)
            churn_ann = compute_churn(aug_churn["ann"], sept_churn["ann"], churn_sample_ids)
            print(
                f"  churn (exact cosine): mean_jaccard={churn_exact['mean_jaccard']:.4f} (n={churn_exact['n']}); "
                f"churn (ANN @ production ef_search): mean_jaccard={churn_ann['mean_jaccard']:.4f} (n={churn_ann['n']})",
                file=sys.stderr,
            )

            partial = {
                "sampling": {
                    "query_sample_seed": QUERY_SAMPLE_SEED,
                    "query_sample_size": QUERY_SAMPLE_SIZE,
                    "churn_sample_seed": CHURN_SAMPLE_SEED,
                    "churn_sample_size_requested": CHURN_SAMPLE_SIZE,
                    "churn_sample_size_actual": len(churn_sample_ids),
                    "common_artists": len(common_ids),
                    "rule": "smallest N by splitmix64(node_key('a', artist_id) XOR seed)",
                },
                "august": {
                    "dump_id": aug["dump_id"],
                    "dump_date": aug["dump_date"],
                    "model_version": aug["model_version"],
                    "n_vectors": len(aug["artist_ids"]),
                    **aug_result,
                    "larger_index_variant": aug_larger_variant,
                },
                "september": {
                    "dump_id": sept["dump_id"],
                    "dump_date": sept["dump_date"],
                    "model_version": sept["model_version"],
                    "n_vectors": len(sept["artist_ids"]),
                    **sept_result,
                },
                "churn_exact_cosine": churn_exact,
                "churn_ann_at_production_ef_search": churn_ann,
                "partial": True,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(partial, indent=2, default=str))
            print(
                f"  partial result (both months' standard variant + exact/ANN churn, before September's larger-variant attempt) written to {args.out}",
                file=sys.stderr,
            )

            if args.skip_larger_variant:
                print("=== September: --skip-larger-variant set -- skipping the larger-index (m=32) variant ===", file=sys.stderr)
                sept_larger_variant = {"skipped": True, "reason": LARGER_VARIANT_SKIP_REASON}
            else:
                print(
                    f"\n=== September: dropping the standard index ({sept_result['index']['index_name']}) before the larger-index variant ===",
                    file=sys.stderr,
                )
                await _drop_index(conn, sept["model_version"])
                sept_larger_variant = await measure_index_variant(
                    conn,
                    sept,
                    label="larger",
                    m=HNSW_LARGER_M,
                    ef_construction=HNSW_LARGER_EF_CONSTRUCTION,
                    maintenance_work_mem=args.sept_maintenance_work_mem,
                    query_sample_positions=sept_query_positions,
                    exact_ids_by_query=sept_ground_truth.exact_ids_by_query,
                    kth_scores=sept_ground_truth.kth_scores,
                    normalized_cache=sept_ground_truth.cache,
                    id_to_position=sept_ground_truth.id_to_position,
                    drop_after=True,
                )
    finally:
        await pool.close()

    return {
        **{k: v for k, v in partial.items() if k != "partial"},
        "september": {**partial["september"], "larger_index_variant": sept_larger_variant},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aug", type=Path)
    parser.add_argument("sept", type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--container-name", default="gm-ieu3-measure-pg", help="for restarting between August and September if the mwm values differ")
    parser.add_argument("--image", default="database-schema-postgres19-pgvector:local")
    parser.add_argument("--aug-maintenance-work-mem", default=DEFAULT_MAINTENANCE_WORK_MEM)
    parser.add_argument("--sept-maintenance-work-mem", default=DEFAULT_MAINTENANCE_WORK_MEM)
    parser.add_argument("--sept-shm-size", default="2g", help="only used if --sept-maintenance-work-mem differs from --aug-maintenance-work-mem")
    parser.add_argument(
        "--skip-larger-variant-aug",
        action="store_true",
        help="skip the larger-index variant (m=32, ef_construction=128) for August, measuring it on September only -- "
        "the dispatcher's escape hatch for when time per w0 becomes excessive (say so in the report if used). Has no "
        "effect together with --skip-ann-churn, which already never measures a larger variant for August. Superseded "
        "by --skip-larger-variant, which skips it everywhere.",
    )
    parser.add_argument(
        "--skip-larger-variant",
        action="store_true",
        help="skip the larger-index variant (m=32, ef_construction=128) in BOTH months and in both --skip-ann-churn "
        "and full mode. Maintainer decision, 2026-09-27: September's m=32 build for w0=0 overflowed "
        "maintenance_work_mem=8GB at ~6.1M of 9,366,416 tuples and fell to an on-disk build path at ~80 tuples/s -- "
        "an ~8h build per weight, repeated per w0 and again for the winner run. That overflow IS the recorded "
        "finding (LARGER_VARIANT_SKIP_REASON, also in the results JSON's larger_variant_finding); this flag stops "
        "actually attempting the build. The standard-variant recall sweep, degree buckets, latency, and churn are "
        "all unaffected and still run.",
    )
    parser.add_argument(
        "--skip-ann-churn",
        action="store_true",
        help="maintainer-approved recall-phase trim: process ONLY September (write, standard-index recall sweep, "
        "always the larger-index variant too), and compute churn on EXACT top-10 alone -- no Postgres write, no "
        "index, and no ANN churn for August at all. Use for every w0 except the single best one (by tie-tolerant "
        "recall + quality, chosen from the September results), which should instead run WITHOUT this flag so both "
        "months' standard indexes exist and ANN churn between them can be computed.",
    )
    args = parser.parse_args()

    result = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
