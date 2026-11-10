"""Read-only sizing and exactness pass for the monthly exact similar-artist job (gm-analytics-engine-3am).

Run it before any full-catalog run. It reads one ``model_version`` of ``public.artist_embeddings``
over a read-only session, computes exact top-K lists for a *slice* (the first ``--slice-rows``
artists in ``artist_id`` order) against the **whole** catalog with the production kernel, and
reports aggregates only: per-phase wall-clock, peak RSS after each phase, and an extrapolation to
the full catalog. It then checks the slice's stored-format top-10 lists against an independent
brute force on a seeded random sample (``insights.embeddings.independent_exact_top_k``, which does not
import the production kernel).

It writes nothing to Postgres: no release is created, no list is stored or published, and the
"already published" skip in ``run_similar_artists`` does not apply, so it is safe to run against
production while a release is current. The only writes are the local scratch spool (removed on
exit) and the optional ``--output`` JSON. No artist id, name or vector is logged or reported.

``python -m insights.similar_artists_sizing --help``; the exact run command, expected output and the
extrapolation rules are in ``docs/similar_artists.md`` ("Sizing run").
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import structlog

from insights.embeddings.exact_top_k import DEFAULT_BLOCK_ROWS, DEFAULT_K, PhaseProfile
from insights.embeddings.independent_exact_top_k import compare_stored_top10, independent_top_k
from insights.similar_artists import (
    DEFAULT_MEMORY_BUDGET_BYTES,
    DEFAULT_THREADS,
    MemoryGuard,
    compute_to_spool,
    estimate_peak_bytes,
    peak_rss_bytes,
    read_embeddings,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


logger = structlog.get_logger(__name__)

DEFAULT_SAMPLE: Final = 2000
DEFAULT_SEED: Final = 20261110
SIX_HOURS_S: Final = 6 * 3600.0
_EXPECTED_SESSION: Final = "on"
_SELECT_VERSIONS_SQL: Final = "SELECT model_version, count(*) FROM public.artist_embeddings GROUP BY model_version"


@dataclass(frozen=True)
class Extrapolation:
    """Full-catalog projection from one slice. See `extrapolate`."""

    scale: float
    matmul_fraction: float
    full_wall_s: float
    full_wall_upper_bound_s: float
    full_peak_rss_bytes: int
    estimate_peak_bytes: int
    within_six_hours: bool
    within_six_hours_upper_bound: bool
    within_memory_budget: bool


def extrapolate(
    *,
    slice_rows: int,
    catalog_rows: int,
    kernel_wall_s: float,
    profile: PhaseProfile,
    peak_rss: int,
    k: int,
    dim: int,
    threads: int,
    block_rows: int,
    budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> Extrapolation:
    """Project a slice's kernel wall-clock and peak RSS to the whole catalog.

    **Wall-clock.** The slice multiplies ``slice_rows x catalog_rows`` scores in full, so its
    time is linear in ``slice_rows``; the full run multiplies only the upper block triangle
    (the production kernel's symmetry), about half of the ``catalog_rows x catalog_rows`` matrix.
    The block upcast and the multiply scale with that halving; candidate selection does not,
    because every block of the triangle is selected from on both its row and its column side
    (the column side is measured cheaper, so ignoring that is the conservative choice):

        full = wall x (N / Q) x (f_matmul+upcast x 0.5 + f_select)

    with the fractions taken from the kernel's own thread-seconds. ``full_wall_upper_bound_s`` is
    ``wall x N / Q``, i.e. no benefit from symmetry at all.

    **Peak RSS.** The slice already holds the whole catalog's vectors, ids and norms; what grows
    is the running lists, ``(N - Q) x (k x 8 + 8)`` bytes. Spools are plain file writes and do not
    count toward RSS.
    """
    scale = catalog_rows / slice_rows
    total = profile.upcast_s + profile.matmul_s + profile.select_s
    multiply_fraction = (profile.upcast_s + profile.matmul_s) / total if total else 0.5
    full_wall = kernel_wall_s * scale * (multiply_fraction * 0.5 + (1.0 - multiply_fraction))
    upper_bound = kernel_wall_s * scale
    full_peak = peak_rss + (catalog_rows - slice_rows) * (k * 8 + 8)
    return Extrapolation(
        scale=scale,
        matmul_fraction=multiply_fraction,
        full_wall_s=full_wall,
        full_wall_upper_bound_s=upper_bound,
        full_peak_rss_bytes=full_peak,
        estimate_peak_bytes=estimate_peak_bytes(catalog_rows, dim, k=k, threads=threads, block_rows=block_rows),
        within_six_hours=full_wall <= SIX_HOURS_S,
        within_six_hours_upper_bound=upper_bound <= SIX_HOURS_S,
        within_memory_budget=full_peak <= budget_bytes,
    )


def sample_queries(slice_rows: int, sample: int, seed: int) -> NDArray[np.int64]:
    """``sample`` distinct slice positions, seeded and sorted (all of them if the slice is smaller)."""
    count = min(sample, slice_rows)
    return np.sort(np.random.default_rng(seed).choice(slice_rows, size=count, replace=False)).astype(np.int64)


def validate_sample(
    vectors: NDArray[np.floating],
    spool_read: Callable[[int, int], tuple[NDArray[np.int32], NDArray[np.float32]]],
    queries: NDArray[np.int64],
    *,
    tolerance: float = 1e-4,
    query_rows: int = 256,
) -> dict[str, Any]:
    """Compare the job's top-10 for ``queries`` with an independent exact computation over
    the whole catalog: recall 1.0 with ties within ``tolerance`` is a pass."""
    stored_positions = np.empty((len(queries), 10), dtype=np.int32)
    stored_scores = np.empty((len(queries), 10), dtype=np.float32)
    for row, query in enumerate(queries):
        positions, scores = spool_read(int(query), int(query) + 1)
        stored_positions[row], stored_scores[row] = positions[0, :10], scores[0, :10]
    exact_positions, exact_scores = independent_top_k(vectors, queries, k=10, query_rows=query_rows)
    return compare_stored_top10(vectors, queries, stored_positions, stored_scores, exact_positions, exact_scores, tolerance=tolerance)


def run_sizing(
    vectors: NDArray[np.floating],
    scratch_root: Path,
    *,
    slice_rows: int,
    validate: int = DEFAULT_SAMPLE,
    seed: int = DEFAULT_SEED,
    k: int = DEFAULT_K,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    threads: int = DEFAULT_THREADS,
    guard: MemoryGuard | None = None,
    read_seconds: float = 0.0,
    peak_after_read: int | None = None,
) -> dict[str, Any]:
    """Slice compute, optional exactness check and extrapolation over already-loaded
    ``vectors``. Pure of the database; returns the aggregate report."""
    catalog_rows, dim = int(vectors.shape[0]), int(vectors.shape[1])
    if not 0 < slice_rows <= catalog_rows:
        raise ValueError(f"slice_rows must be in 1..{catalog_rows}, got {slice_rows}")
    guard = guard or MemoryGuard()
    profile = PhaseProfile()
    scratch_root.mkdir(parents=True, exist_ok=True)
    # Spool plus the old and replacement checkpoint, as in `scripts/measure-exact-top-k-sizing.py`.
    required_disk = slice_rows * (k * 8 * 3 + 16) + 256 * 1024**2
    free_disk = shutil.disk_usage(scratch_root).free
    if free_disk < required_disk:
        raise RuntimeError(f"sizing needs {required_disk} free bytes in {scratch_root}; available {free_disk}")

    report: dict[str, Any] = {
        "mode": "dry_run_read_only",
        "writes_to_postgres": False,
        "slice_rows": slice_rows,
        "catalog_rows": catalog_rows,
        "dimensions": dim,
        "k": k,
        "block_rows": block_rows,
        "threads": threads,
        "seed": seed,
    }
    phases: dict[str, float] = {"read_embeddings_s": read_seconds}
    peaks: dict[str, int] = {"after_read": peak_after_read if peak_after_read is not None else peak_rss_bytes()}

    with tempfile.TemporaryDirectory(prefix="gm-similar-sizing-", dir=scratch_root) as scratch:
        started = time.monotonic()
        spool = compute_to_spool(
            vectors,
            Path(scratch),
            model_version="sizing-slice",
            k=k,
            block_rows=block_rows,
            threads=threads,
            guard=guard,
            query_rows=slice_rows,
            profile=profile,
        )
        phases["compute_slice_s"] = time.monotonic() - started
        peaks["after_compute"] = peak_rss_bytes()
        report["spool_bytes"] = spool.positions_path.stat().st_size + spool.scores_path.stat().st_size

        if validate > 0:
            queries = sample_queries(slice_rows, validate, seed)
            started = time.monotonic()
            report["exactness"] = validate_sample(vectors, spool.read, queries)
            phases["exactness_check_s"] = time.monotonic() - started
            peaks["after_exactness_check"] = peak_rss_bytes()

    kernel_wall = phases["compute_slice_s"]
    report["phases"] = {name: round(value, 3) for name, value in phases.items()}
    report["peak_rss_bytes"] = peaks
    report["throughput"] = {
        "artists_per_second": slice_rows / kernel_wall,
        "seconds_per_block_row": kernel_wall / -(-slice_rows // block_rows),
    }
    report["kernel_thread_seconds"] = {
        "upcast": profile.upcast_s,
        "matmul": profile.matmul_s,
        "select": profile.select_s,
        "score_blocks": profile.tasks,
    }
    report["extrapolation"] = asdict(
        extrapolate(
            slice_rows=slice_rows,
            catalog_rows=catalog_rows,
            kernel_wall_s=kernel_wall,
            profile=profile,
            # Peak at the end of compute: the exactness check must not inflate the projection.
            peak_rss=peaks["after_compute"],
            k=k,
            dim=dim,
            threads=threads,
            block_rows=block_rows,
            budget_bytes=guard.budget_bytes,
        )
    )
    return report


# ── Database access (read-only) ──────────────────────────────────────────────────────────────


async def _resolve_model_version(conn: Any, requested: str | None, expect_rows: int | None) -> tuple[str, int]:
    async with conn.cursor() as cursor:
        await cursor.execute(_SELECT_VERSIONS_SQL)
        versions = {row[0]: int(row[1]) for row in await cursor.fetchall()}
    await conn.rollback()
    if requested is None:
        if len(versions) != 1:
            raise ValueError(f"expected exactly one model_version in artist_embeddings, found {len(versions)}; pass --model-version")
        requested = next(iter(versions))
    if requested not in versions:
        raise ValueError("requested model_version is not in artist_embeddings")
    if expect_rows is not None and versions[requested] != expect_rows:
        raise ValueError(f"model_version has {versions[requested]} rows, expected {expect_rows}")
    return requested, versions[requested]


async def _assert_read_only(conn: Any) -> None:
    async with conn.cursor() as cursor:
        await cursor.execute("SHOW default_transaction_read_only")
        row = await cursor.fetchone()
    await conn.rollback()
    if row is None or row[0] != _EXPECTED_SESSION:
        raise RuntimeError("database session is not read-only; refusing to run the sizing pass")


async def load_catalog(pool: Any, model_version: str | None, expect_rows: int | None) -> tuple[str, NDArray[np.float16]]:
    """The catalog's vectors in ``artist_id`` order over a read-only session. The artist ids
    are discarded unread: sizing needs positions only."""
    async with pool.connection() as conn:
        await _assert_read_only(conn)
        version, _count = await _resolve_model_version(conn, model_version, expect_rows)
        _ids, vectors = await read_embeddings(conn, version)
    del _ids
    return version, vectors


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m insights.similar_artists_sizing", description=__doc__.split("\n\n")[0])
    parser.add_argument("--slice-rows", type=int, required=True, help="first N artists in artist_id order to compute (e.g. 500000, 1000000)")
    parser.add_argument("--validate-sample", type=int, default=DEFAULT_SAMPLE, help="seeded random sample checked against brute force; 0 skips")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-version", help="default: the single model_version in artist_embeddings")
    parser.add_argument("--expect-rows", type=int, help="fail unless the model_version has exactly this many rows")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    parser.add_argument("--scratch-dir", type=Path, default=Path(getenv("SIMILAR_ARTISTS_SPOOL_DIR") or "/scratch"))
    parser.add_argument("--budget-gb", type=float, default=DEFAULT_MEMORY_BUDGET_BYTES / 1024**3, help="abort if peak RSS passes this (GiB)")
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    return parser.parse_args(argv)


async def _load_from_env(args: argparse.Namespace) -> tuple[str, NDArray[np.float16], float, int]:  # pragma: no cover -- needs a live database.
    from common import AsyncPostgreSQLPool, parse_postgres_host_port  # noqa: PLC0415
    from common.config import _build_postgres_connstr, get_secret  # noqa: PLC0415

    username = get_secret("EMBEDDING_PIPELINE_POSTGRES_USERNAME")
    password = get_secret("EMBEDDING_PIPELINE_POSTGRES_PASSWORD")
    database = getenv("POSTGRES_DATABASE")
    if not (username and password and database):
        raise ValueError("EMBEDDING_PIPELINE_POSTGRES_USERNAME, EMBEDDING_PIPELINE_POSTGRES_PASSWORD and POSTGRES_DATABASE are required")
    host, port = parse_postgres_host_port(_build_postgres_connstr())
    pool = AsyncPostgreSQLPool(
        connection_params={
            "host": host,
            "port": port,
            "dbname": database,
            "user": username,
            "password": password,
            # Server-enforced: any write this session attempts fails, whatever the code does.
            "options": "-c default_transaction_read_only=on",
        },
        min_connections=1,
        max_connections=1,
    )
    await pool.initialize()
    try:
        started = time.monotonic()
        version, vectors = await load_catalog(pool, args.model_version, args.expect_rows)
        return version, vectors, time.monotonic() - started, peak_rss_bytes()
    finally:
        await pool.close()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover -- exercised through run_sizing in tests.
    args = _parse_args(argv)
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    version, vectors, read_seconds, peak_after_read = asyncio.run(_load_from_env(args))
    logger.info("📖 Catalog read", artists=int(vectors.shape[0]), read_s=round(read_seconds, 1), peak_rss_gb=round(peak_after_read / 1e9, 2))
    report = run_sizing(
        vectors,
        args.scratch_dir,
        slice_rows=args.slice_rows,
        validate=args.validate_sample,
        seed=args.seed,
        block_rows=args.block_rows,
        threads=args.threads,
        guard=MemoryGuard(budget_bytes=int(args.budget_gb * 1024**3)),
        read_seconds=read_seconds,
        peak_after_read=peak_after_read,
    )
    report["model_version_hash"] = hashlib.blake2b(version.encode(), digest_size=6).hexdigest()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)  # noqa: T201
    exactness = report.get("exactness")
    if exactness is not None and not exactness["passed"]:
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    main()
