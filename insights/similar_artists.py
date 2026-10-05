"""Monthly exact top-K similar-artist lists (ADR 0013, D serving mode).

gm-analytics-engine-8ts found that serving similar artists straight off the HNSW index fails
all three of the maintainer's thresholds, so the 2026-09-29 decision is to precompute each
artist's exact top-K cosine neighbours once a month and serve those lists instead
(`docs/embedding_tie_break.md`). This module is that batch job, run after
`insights.embedding_pipeline` has written a month's vectors. Method, sizing, and operations
are in `docs/similar_artists.md`. In order:

1. **Compute to a local spool** (`compute_to_spool`). `insights.embeddings.exact_top_k` runs
   over the whole catalog and each finalized row block's lists are written to two raw files
   (`positions.i32`, `scores.f32`, one ``k``-wide row per artist) at that block's offset.
   Plain file writes, not a memory map, so the spool never counts toward this process's RSS.
   Every ``checkpoint_every_s`` the running state and the next block index are saved,
   atomically (write to a temporary name, then `Path.replace`), so a crashed or stopped run
   resumes from its last checkpoint rather than from zero. `MemoryGuard` runs between blocks.
2. **Write atomically** (`write_similar_artists`). One transaction deletes any rows a
   previous, failed attempt left under this `model_version` and streams every list into
   `public.artist_similar_artists` with `COPY`. Nothing is visible to `catalog-api` until
   step 3, and a failure rolls the whole version back.
3. **Publish and rotate** (`publish_and_rotate`). `publish_artist_embedding_release` makes
   this `model_version` current, and every older version except the one it replaced is
   retired: the current and previous lists stay, so a bad month can be rolled back by
   republishing the previous one.

The create, publish and retire helpers are an immutable promotion of database-schema
(`insights.schema_release_contract`), with producer provenance and an AST compatibility
gate. `ReleaseRegistry` keeps the flow testable without a database or dev schema package.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
import structlog

from insights import schema_release_contract as release_contract
from insights.embeddings.exact_top_k import DEFAULT_BLOCK_ROWS, DEFAULT_K, EMPTY_POSITION, TopKState, exact_top_k, reciprocal_norms


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from datetime import date
    from pathlib import Path

    from numpy.typing import NDArray


logger = structlog.get_logger(__name__)

SIMILAR_ARTISTS_TABLE: Final = "public.artist_similar_artists"
RELEASES_TABLE: Final = "public.artist_embedding_releases"

# The 12 GB pipeline budget docs/embeddings.md measures FastRP against; the top-K stage
# runs after FastRP has released its arrays, under the same budget.
DEFAULT_MEMORY_BUDGET_BYTES: Final = 12 * 1024**3
DEFAULT_THREADS: Final = 8
DEFAULT_CHECKPOINT_EVERY_S: Final = 1800.0
# Measured per-thread working set of one exact_top_k task at block_rows=4096 (a 64 MB score
# block, its segment maxima, first-sight partition copies, and candidate gathers), from the
# 1M-artist sizing run in docs/similar_artists.md: 2.1 GB of transients across 8 threads.
_TASK_BYTES_PER_BLOCK_CELL: Final = 16

_DELETE_VERSION_SQL: Final = f"DELETE FROM {SIMILAR_ARTISTS_TABLE} WHERE release_id = %s"  # noqa: S608
_COPY_SQL: Final = f"COPY {SIMILAR_ARTISTS_TABLE} (release_id, artist_id, similar_artist_ids, scores) FROM STDIN"
_RELEASE_LOCK_SQL: Final = release_contract._ARTIST_EMBEDDING_RELEASES_LOCK_SQL
_RELEASE_WRITE_GUARD_SQL: Final = f"SELECT artists FROM {RELEASES_TABLE} WHERE release_id = %s FOR UPDATE"  # noqa: S608
_CURRENT_RELEASE_SQL: Final = f"SELECT model_version FROM {RELEASES_TABLE} WHERE is_current"  # noqa: S608
_RELEASES_BY_RECENCY_SQL: Final = f"SELECT model_version FROM {RELEASES_TABLE} WHERE artists > 0 ORDER BY published_at DESC, model_version DESC"  # noqa: S608
_COPY_CHUNK_ARTISTS: Final = 20_000
_RELEASE_EXISTS_SQL: Final = f"SELECT 1 FROM {RELEASES_TABLE} WHERE model_version = %s AND artists > 0"  # noqa: S608
# Artist order fixes catalog positions, and so the tie order: equal scores rank by artist_id.
_COUNT_EMBEDDINGS_SQL: Final = "SELECT count(*) FROM public.artist_embeddings WHERE model_version = %s"
_READ_EMBEDDINGS_SQL: Final = "SELECT artist_id, embedding::text FROM public.artist_embeddings WHERE model_version = %s ORDER BY artist_id"
_FETCH_SIZE: Final = 50_000


# ── Memory guard ──────────────────────────────────────────────────────────────────────────────


def peak_rss_bytes() -> int:
    """This process's peak resident set size. `ru_maxrss` only ever grows, so an idle wait
    that lets the OS page this process out cannot make it look smaller than it is (the
    failure 66608ed fixed in `scripts/embeddings_from_dump.py`)."""
    scale = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def memorystatus_level_pct() -> int | None:
    """macOS's `kern.memorystatus_level` (0-100, Activity Monitor's "memory free" gauge), or
    `None` where that sysctl does not exist (Linux containers)."""
    try:
        result = subprocess.run(["/usr/sbin/sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    try:
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except ValueError:
        return None


class MemoryBudgetExceededError(RuntimeError):
    """This process's peak RSS passed its budget."""


def estimate_peak_bytes(n_artists: int, dim: int, *, k: int = DEFAULT_K, threads: int = DEFAULT_THREADS, block_rows: int = DEFAULT_BLOCK_ROWS) -> int:
    """Expected peak RSS of `compute_to_spool`: the ``float16`` vectors, the running lists
    (``float32`` score and ``int32`` position per slot), the per-row norms and thresholds,
    and each thread's task working set."""
    vectors = n_artists * dim * 2
    lists = n_artists * k * 8 + n_artists * 8
    tasks = threads * block_rows * block_rows * _TASK_BYTES_PER_BLOCK_CELL
    return vectors + lists + tasks


@dataclass
class MemoryGuard:
    """Raise once peak RSS passes ``budget_bytes``; pause (never abort) while the host is
    under memory pressure, polling ``kern.memorystatus_level`` where it exists."""

    budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES
    min_memorystatus_level_pct: int = 25
    poll_interval_s: float = 60.0
    peak_rss: Callable[[], int] = peak_rss_bytes
    memorystatus_level: Callable[[], int | None] = memorystatus_level_pct
    sleep: Callable[[float], None] = time.sleep

    def check(self, label: str = "") -> None:
        peak = self.peak_rss()
        if peak > self.budget_bytes:
            raise MemoryBudgetExceededError(f"{label}: peak RSS {peak / 1e9:.2f} GB is over the {self.budget_bytes / 1e9:.2f} GB budget")
        waited = 0.0
        while (level := self.memorystatus_level()) is not None and level < self.min_memorystatus_level_pct:
            if waited == 0.0:
                logger.warning("⏳ Host under memory pressure, pausing", label=label, memorystatus_level=level)
            self.sleep(self.poll_interval_s)
            waited += self.poll_interval_s
        if waited:
            logger.info("▶️ Host memory pressure cleared, resuming", label=label, waited_s=waited)


# ── Compute to a resumable local spool ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Spool:
    """A finished (or in-progress) run's lists on local disk, row ``r`` being artist ``r``."""

    directory: Path
    n_artists: int
    k: int

    @property
    def positions_path(self) -> Path:
        return self.directory / "positions.i32"

    @property
    def scores_path(self) -> Path:
        return self.directory / "scores.f32"

    @property
    def checkpoint_path(self) -> Path:
        return self.directory / "checkpoint.npz"

    @property
    def meta_path(self) -> Path:
        return self.directory / "meta.json"

    def read(self, start: int, stop: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Rows ``start:stop`` of the lists, sorted by (score desc, position asc)."""
        count = (stop - start) * self.k
        positions = np.fromfile(self.positions_path, dtype=np.int32, count=count, offset=start * self.k * 4)
        scores = np.fromfile(self.scores_path, dtype=np.float32, count=count, offset=start * self.k * 4)
        return positions.reshape(-1, self.k), scores.reshape(-1, self.k)


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    tmp.replace(path)


def _write_rows(path: Path, start_row: int, row_bytes: int, data: NDArray[Any]) -> None:
    with path.open("r+b") as fh:
        fh.seek(start_row * row_bytes)
        fh.write(np.ascontiguousarray(data).tobytes())
        fh.flush()
        os.fsync(fh.fileno())


def compute_to_spool(
    vectors: NDArray[np.floating],
    directory: Path,
    *,
    model_version: str,
    k: int = DEFAULT_K,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    threads: int = DEFAULT_THREADS,
    checkpoint_every_s: float = DEFAULT_CHECKPOINT_EVERY_S,
    guard: MemoryGuard | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Spool:
    """Compute every artist's exact top-``k`` into a spool under ``directory``, resuming from
    its checkpoint when one exists for the same ``model_version`` and shape. Returns the
    finished spool (``meta.json`` then records ``complete: true``)."""
    guard = guard or MemoryGuard()
    n_artists = vectors.shape[0]
    spool = Spool(directory, n_artists, k)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"model_version": model_version, "n_artists": n_artists, "dim": int(vectors.shape[1]), "k": k, "block_rows": block_rows}

    meta: dict[str, Any] = json.loads(spool.meta_path.read_text()) if spool.meta_path.exists() else {}
    state: TopKState | None = None
    start_block = 0
    if meta.get("identity") == identity and meta.get("complete"):
        logger.info("⏭️ Similar-artist spool already complete", model_version=model_version, directory=str(directory))
        return spool
    if meta.get("identity") == identity and spool.checkpoint_path.exists():
        with np.load(spool.checkpoint_path) as saved:
            state = TopKState(saved["scores"], saved["positions"], saved["thresholds"])
            start_block = int(saved["next_block"])
        logger.info("🔁 Resuming similar-artist computation from checkpoint", model_version=model_version, start_block=start_block)
    else:
        for path in (spool.positions_path, spool.scores_path, spool.checkpoint_path):
            path.unlink(missing_ok=True)
        with spool.positions_path.open("wb") as fh:
            fh.truncate(n_artists * k * 4)
        with spool.scores_path.open("wb") as fh:
            fh.truncate(n_artists * k * 4)
        meta = {"identity": identity, "complete": False}
        _atomic_write(spool.meta_path, lambda p: p.write_text(json.dumps(meta)))

    guard.check("before exact top-K")
    last_checkpoint = clock()
    n_blocks = -(-n_artists // block_rows)

    def on_block(row_start: int, scores: NDArray[np.float32], positions: NDArray[np.int32]) -> None:
        _write_rows(spool.positions_path, row_start, k * 4, positions)
        _write_rows(spool.scores_path, row_start, k * 4, scores)

    def after_block(next_block: int, running: TopKState) -> None:
        nonlocal last_checkpoint
        guard.check(f"block {next_block}/{n_blocks}")
        if next_block < n_blocks and clock() - last_checkpoint >= checkpoint_every_s:
            _save_checkpoint(spool, running, next_block)
            last_checkpoint = clock()
            logger.info("💾 Similar-artist checkpoint", next_block=next_block, blocks=n_blocks, peak_rss_gb=round(guard.peak_rss() / 1e9, 2))

    exact_top_k(
        vectors,
        k,
        block_rows=block_rows,
        threads=threads,
        state=state,
        start_block=start_block,
        on_block=on_block,
        after_block=after_block,
        inv_norms=reciprocal_norms(vectors),
    )
    meta["complete"] = True
    _atomic_write(spool.meta_path, lambda p: p.write_text(json.dumps(meta)))
    spool.checkpoint_path.unlink(missing_ok=True)
    return spool


def _save_checkpoint(spool: Spool, state: TopKState, next_block: int) -> None:
    def write(path: Path) -> None:
        with path.open("wb") as fh:
            np.savez(fh, scores=state.scores, positions=state.positions, thresholds=state.thresholds, next_block=next_block)

    _atomic_write(spool.checkpoint_path, write)


# ── Write, publish, retire ───────────────────────────────────────────────────────────────────


def _copy_rows(spool: Spool, artist_ids: Sequence[str], release_id: int) -> Iterator[tuple[int, str, list[str], list[float]]]:
    """One compact COPY row per artist, with ordered, paired neighbour arrays.

    `write_row` delegates quoting/escaping of arbitrary text ids to psycopg, including
    commas, braces, backslashes, tabs, and newlines. Unfilled slots are omitted together.
    """
    for start in range(0, spool.n_artists, _COPY_CHUNK_ARTISTS):
        stop = min(start + _COPY_CHUNK_ARTISTS, spool.n_artists)
        positions, scores = spool.read(start, stop)
        for row in range(stop - start):
            valid = positions[row] != EMPTY_POSITION
            yield (
                release_id,
                artist_ids[start + row],
                [artist_ids[int(position)] for position in positions[row, valid]],
                scores[row, valid].tolist(),
            )


async def write_similar_artists(
    conn: Any,
    spool: Spool,
    *,
    model_version: str,
    artist_ids: Sequence[str],
    source_dump_id: str,
    source_dump_date: date,
    registry: ReleaseRegistry | None = None,
) -> int:
    """Create/reuse an unpublished release and replace its artist arrays atomically.

    Creation, lineage/K validation, deletion and COPY share one transaction. A failed
    COPY rolls all of them back. Previously published lists are never overwritten.
    """
    if len(artist_ids) != spool.n_artists:
        raise ValueError(f"{len(artist_ids)} artist ids for a spool of {spool.n_artists} artists")
    registry = registry or SchemaReleaseRegistry()
    rows = 0
    async with conn.transaction(), conn.cursor() as cursor:
        await cursor.execute(_RELEASE_LOCK_SQL)
        release_id = await registry.create(cursor, model_version, source_dump_id=source_dump_id, source_dump_date=source_dump_date, k=spool.k)
        if release_id is None:
            raise PublishError(f"creating {model_version!r} failed; see the schema helper's log")
        await cursor.execute(_RELEASE_WRITE_GUARD_SQL, (release_id,))
        existing = await cursor.fetchone()
        if existing is None or existing[0] > 0:
            raise PublishError(f"refusing to overwrite published or missing release {model_version!r}")
        await cursor.execute(_DELETE_VERSION_SQL, (release_id,))
        async with cursor.copy(_COPY_SQL) as copy:
            for row in _copy_rows(spool, artist_ids, release_id):
                await copy.write_row(row)
                rows += 1
    logger.info("💾 Similar-artist lists written", model_version=model_version, rows=rows)
    return rows


class ReleaseRegistry(Protocol):
    """The promoted producer helpers; creation returns an id, others failure counts."""

    async def create(self, cursor: Any, model_version: str, *, source_dump_id: str, source_dump_date: date, k: int) -> int | None: ...

    async def publish(self, cursor: Any, model_version: str, *, artists: int) -> int: ...

    async def retire(self, cursor: Any, model_version: str, *, delete_release: bool = False) -> int: ...


class SchemaReleaseRegistry:
    """Runtime-safe adapter over the immutable promoted producer binding."""

    async def create(self, cursor: Any, model_version: str, *, source_dump_id: str, source_dump_date: date, k: int) -> int | None:
        return await release_contract.create_artist_embedding_release(
            cursor, model_version, source_dump_id=source_dump_id, source_dump_date=source_dump_date, k=k
        )

    async def publish(self, cursor: Any, model_version: str, *, artists: int) -> int:
        return await release_contract.publish_artist_embedding_release(cursor, model_version, artists=artists)

    async def retire(self, cursor: Any, model_version: str, *, delete_release: bool = False) -> int:
        return await release_contract.retire_artist_similar_artists_version(cursor, model_version, delete_release=delete_release)


class PublishError(RuntimeError):
    """`publish_artist_embedding_release` reported a failure."""


@dataclass(frozen=True)
class RotateResult:
    published: str
    kept_previous: str | None
    retired: tuple[str, ...]
    retire_failures: tuple[str, ...]


async def publish_and_rotate(
    conn: Any,
    registry: ReleaseRegistry,
    *,
    model_version: str,
    artists: int,
) -> RotateResult:
    """Publish ``model_version`` as current, then retire the rows of every release older
    than the one it replaced. Release rows are kept as lineage. A failed retire is logged
    and reported, not raised: the new release is already live and correct."""
    async with conn.transaction(), conn.cursor() as cursor:
        await cursor.execute(_RELEASE_LOCK_SQL)
        await cursor.execute(_CURRENT_RELEASE_SQL)
        current = await cursor.fetchone()
        previous = current[0] if current and current[0] != model_version else None
        if await registry.publish(cursor, model_version, artists=artists):
            raise PublishError(f"publishing {model_version!r} failed; see the schema helper's log")
        await cursor.execute(_RELEASES_BY_RECENCY_SQL)
        older = [row[0] for row in await cursor.fetchall() if row[0] != model_version]
        if current and current[0] == model_version and older:
            previous = older[0]
        # Retain the actual displaced current release, not a newer pending target.
        to_retire = [version for version in older if version != previous]
        retired: list[str] = []
        failures: list[str] = []
        for version in to_retire:
            (failures if await registry.retire(cursor, version) else retired).append(version)
    if failures:
        logger.error("❌ Some superseded similar-artist versions were not retired", failed=failures)
    logger.info("✅ Similar-artist release published", model_version=model_version, kept_previous=previous, retired=retired)
    return RotateResult(model_version, previous, tuple(retired), tuple(failures))


# ── The monthly stage ────────────────────────────────────────────────────────────────────────


async def read_embeddings(conn: Any, model_version: str) -> tuple[list[str], NDArray[np.float16]]:
    """Every artist's stored vector for ``model_version``, in ``artist_id`` order, as
    ``float16`` (the ``halfvec`` column's own precision), streamed through a named cursor
    in one read-only transaction into an array sized by a count taken in that same
    transaction."""
    ids: list[str] = []
    async with conn.transaction():
        async with conn.cursor() as cursor:
            await cursor.execute(_COUNT_EMBEDDINGS_SQL, (model_version,))
            (count,) = await cursor.fetchone()
        vectors: NDArray[np.float16] | None = None
        async with conn.cursor(name="similar_artists_embeddings") as cursor:
            await cursor.execute(_READ_EMBEDDINGS_SQL, (model_version,))
            while batch := await cursor.fetchmany(_FETCH_SIZE):
                parsed = np.array([text[1:-1].split(",") for _artist_id, text in batch], dtype=np.float32)
                if vectors is None:
                    vectors = np.empty((count, parsed.shape[1]), dtype=np.float16)
                vectors[len(ids) : len(ids) + len(batch)] = parsed
                ids.extend(artist_id for artist_id, _text in batch)
    if vectors is None:
        return [], np.empty((0, 0), dtype=np.float16)
    return ids, vectors[: len(ids)]


@dataclass(frozen=True)
class SimilarArtistsResult:
    model_version: str
    artists: int
    rows_written: int
    skipped: bool
    rotation: RotateResult | None = None


async def run_similar_artists(
    pool: Any,
    *,
    model_version: str,
    source_dump_id: str,
    source_dump_date: date,
    spool_root: Path,
    registry: ReleaseRegistry | None = None,
    k: int = DEFAULT_K,
    threads: int = DEFAULT_THREADS,
    guard: MemoryGuard | None = None,
) -> SimilarArtistsResult:
    """Compute, write, and publish ``model_version``'s similar-artist lists, or no-op if that
    release has already been published. Pending targets remain retryable. Each database step takes its own connection, so none is held
    open across the hours-long compute."""
    registry = registry or SchemaReleaseRegistry()
    async with pool.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(_RELEASE_EXISTS_SQL, (model_version,))
            if await cursor.fetchone() is not None:
                logger.info("⏭️ Similar-artist release already published", model_version=model_version)
                return SimilarArtistsResult(model_version, 0, 0, skipped=True)
        artist_ids, vectors = await read_embeddings(conn, model_version)
    if not artist_ids:
        raise ValueError(f"no artist_embeddings rows for {model_version!r}")

    guard = guard or MemoryGuard()
    estimate = estimate_peak_bytes(len(artist_ids), vectors.shape[1], k=k, threads=threads)
    logger.info(
        "🔢 Exact top-K starting",
        model_version=model_version,
        artists=len(artist_ids),
        k=k,
        threads=threads,
        estimated_peak_gb=round(estimate / 1e9, 2),
    )
    spool_dir = spool_root / hashlib.blake2b(model_version.encode(), digest_size=8).hexdigest()
    spool = await asyncio.to_thread(compute_to_spool, vectors, spool_dir, model_version=model_version, k=k, threads=threads, guard=guard)
    del vectors

    async with pool.connection() as conn:
        rows = await write_similar_artists(
            conn,
            spool,
            model_version=model_version,
            artist_ids=artist_ids,
            source_dump_id=source_dump_id,
            source_dump_date=source_dump_date,
            registry=registry,
        )
    async with pool.connection() as conn:
        rotation = await publish_and_rotate(
            conn,
            registry,
            model_version=model_version,
            artists=len(artist_ids),
        )
    shutil.rmtree(spool_dir, ignore_errors=True)
    return SimilarArtistsResult(model_version, len(artist_ids), rows, skipped=False, rotation=rotation)
