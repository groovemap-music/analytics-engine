"""Standalone, deadline-bounded exact similar-artist run (gm-analytics-engine-dhit).

``analytics-engine-similar-artists`` computes, or resumes, the exact top-K lists for one
``model_version`` already in ``public.artist_embeddings`` (it never recomputes embeddings),
and publishes them to ``public.artist_similar_artists`` once every row block is done. The
full catalog takes about 34 h at 8 threads (``docs/similar_artists.md``, "Sizing run"), so
the operator's 2026-11-11 decision is to run it exactly, over several nightly windows: each
night's run is given a ``--deadline`` or ``--max-runtime``, stops at the first row-block
boundary past it with a checkpoint, and exits ``paused``; the next night resumes.

Everything below the CLI is the existing stage, `insights.similar_artists.run_similar_artists`
(kernel, spool, checkpoint, write, publish, rotate); this module adds only the target lookup,
the persistent-spool rule, a lock, and the status contract. The CLI, exit codes and status JSON
are documented in ``docs/similar_artists.md`` ("Nightly runs"); the deployment job and the
homelab timer (separate beads) depend on them.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn

import structlog
from common import setup_logging

from insights.similar_artists import (
    DEFAULT_CHECKPOINT_EVERY_S,
    DEFAULT_MEMORY_BUDGET_BYTES,
    DEFAULT_THREADS,
    MemoryGuard,
    ReleaseRegistry,
    SimilarArtistsResult,
    run_similar_artists,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


logger = structlog.get_logger(__name__)

SERVICE_NAME: Final = "analytics-engine-similar-artists"
SPOOL_DIR_ENV: Final = "SIMILAR_ARTISTS_SPOOL_DIR"
STATUS_KIND: Final = "similar_artists_status"

EXIT_DONE: Final = 0
EXIT_FAILED: Final = 1
EXIT_USAGE: Final = 2
# sysexits.h EX_TEMPFAIL: "temporary failure, the user is invited to retry".
EXIT_PAUSED: Final = 75

# Directories that do not survive a reboot or a container: a spool there cannot resume.
NON_PERSISTENT_ROOTS: Final = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))  # noqa: S108 -- refused, never written.

_EMBEDDING_VERSIONS_SQL: Final = "SELECT DISTINCT model_version FROM public.artist_embeddings"
_PUBLISHED_VERSIONS_SQL: Final = "SELECT model_version FROM public.artist_embedding_releases WHERE artists > 0"
_EMBEDDING_LINEAGE_SQL: Final = "SELECT DISTINCT source_dump_id, source_dump_date FROM public.artist_embeddings WHERE model_version = %s"
_RELEASE_LINEAGE_SQL: Final = "SELECT source_dump_id, source_dump_date FROM public.artist_embedding_releases WHERE model_version = %s"

Status = Literal["done", "paused", "failed"]


class UsageError(ValueError):
    """The run was asked for something it refuses to do (exit 2, status ``failed``)."""


class SpoolBusyError(RuntimeError):
    """Another run holds this spool root's lock."""


# ── Status contract ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JobStatus:
    """The final status line. ``blocks_done``/``blocks_total``/``eta_s`` are ``None`` when no
    compute ran (already published, nothing pending, or a failure before it started)."""

    status: Status
    exit_code: int
    model_version: str | None = None
    published: bool = False
    already_published: bool = False
    blocks_done: int | None = None
    blocks_total: int | None = None
    eta_s: float | None = None
    artists: int = 0
    rows_written: int = 0
    elapsed_s: float = 0.0
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps({"kind": STATUS_KIND, **asdict(self)}, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_result(cls, result: SimilarArtistsResult, elapsed_s: float) -> JobStatus:
        progress = result.progress
        return cls(
            status="paused" if result.paused else "done",
            exit_code=EXIT_PAUSED if result.paused else EXIT_DONE,
            model_version=result.model_version,
            published=not (result.paused or result.skipped),
            already_published=result.skipped,
            blocks_done=None if progress is None else progress.blocks_done,
            blocks_total=None if progress is None else progress.blocks_total,
            eta_s=None if progress is None else progress.eta_s,
            artists=result.artists,
            rows_written=result.rows_written,
            elapsed_s=round(elapsed_s, 1),
        )


# ── Arguments ────────────────────────────────────────────────────────────────────────────────


def resolve_spool_root(arg: Path | None, env: str | None, *, non_persistent_roots: tuple[Path, ...] | None = None) -> Path:
    """The explicit, persistent spool root a resumable run needs.

    Raises:
        UsageError: If none was given, it is relative, or it lies under a directory that does
            not survive a restart (``/tmp`` -- the pipeline's container default -- ``/var/tmp``,
            ``/dev/shm``), where a paused run's checkpoint would be lost.
    """
    raw = arg if arg is not None else (Path(env) if env else None)
    if raw is None:
        raise UsageError(f"a resumable run needs a persistent spool directory: pass --spool-dir or set {SPOOL_DIR_ENV}")
    if not raw.is_absolute():
        raise UsageError(f"--spool-dir must be an absolute path, got {str(raw)!r}")
    resolved = raw.resolve()
    for root in NON_PERSISTENT_ROOTS if non_persistent_roots is None else non_persistent_roots:
        for candidate in {root, root.resolve()}:
            if resolved == candidate or candidate in resolved.parents:
                raise UsageError(f"spool directory {str(raw)!r} is under {root}, which does not persist; a paused run could not resume from it")
    return raw


def resolve_deadline(deadline: datetime | None, max_runtime_s: float | None, *, wall_now: datetime, mono_now: float) -> float | None:
    """The earlier of ``deadline`` (absolute, timezone-aware) and ``mono_now + max_runtime_s``,
    as a ``time.monotonic`` reading, or ``None`` for an unbounded run."""
    remaining: list[float] = []
    if deadline is not None:
        if deadline.tzinfo is None:
            raise UsageError("--deadline needs a UTC offset, e.g. 2026-11-12T06:00:00Z")
        remaining.append((deadline - wall_now).total_seconds())
    if max_runtime_s is not None:
        if max_runtime_s <= 0:
            raise UsageError(f"--max-runtime must be positive, got {max_runtime_s}")
        remaining.append(max_runtime_s)
    return mono_now + min(remaining) if remaining else None


class _ArgumentParser(argparse.ArgumentParser):
    """Argument errors become a `UsageError`, so they still end in a status line."""

    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def _parse_deadline(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 time: {value!r}") from exc


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = _ArgumentParser(prog=SERVICE_NAME, description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-version", help="default: the one model_version in artist_embeddings without a published release")
    parser.add_argument("--spool-dir", type=Path, help=f"persistent spool root (or {SPOOL_DIR_ENV}); required, never under /tmp")
    parser.add_argument("--deadline", type=_parse_deadline, help="absolute stop time with a UTC offset, e.g. 2026-11-12T06:00:00Z")
    parser.add_argument("--max-runtime", type=float, metavar="SECONDS", help="stop after this many seconds (the earlier of this and --deadline wins)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--budget-gb", type=float, default=DEFAULT_MEMORY_BUDGET_BYTES / 1024**3, help="fail if peak RSS passes this (GiB)")
    parser.add_argument(
        "--checkpoint-every", type=float, default=DEFAULT_CHECKPOINT_EVERY_S, metavar="SECONDS", help="crash-safety checkpoint interval"
    )
    parser.add_argument("--status-file", type=Path, help="also write the final status JSON here (atomically)")
    return parser.parse_args(argv)


# ── Target and lock ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Target:
    model_version: str
    source_dump_id: str
    source_dump_date: date


async def _fetch_all(conn: Any, sql: str, params: tuple[Any, ...] | None = None) -> list[tuple[Any, ...]]:
    async with conn.cursor() as cursor:
        await cursor.execute(sql, params)
        rows: list[tuple[Any, ...]] = await cursor.fetchall()
    return rows


async def resolve_target(conn: Any, requested: str | None) -> Target | None:
    """The ``model_version`` to run and its dump lineage, or ``None`` when no version was
    requested and every one in ``artist_embeddings`` is already published.

    Raises:
        UsageError: If several versions are unpublished and none was requested, or the
            requested one is neither in ``artist_embeddings`` nor a published release.
        ValueError: If a version's embeddings carry more than one dump lineage.
    """
    versions = {row[0] for row in await _fetch_all(conn, _EMBEDDING_VERSIONS_SQL)}
    published = {row[0] for row in await _fetch_all(conn, _PUBLISHED_VERSIONS_SQL)}
    if requested is None:
        pending = sorted(versions - published)
        if not pending:
            return None
        if len(pending) > 1:
            raise UsageError(f"{len(pending)} model_versions in artist_embeddings have no published release; pass --model-version")
        requested = pending[0]
    if requested in versions:
        lineage = await _fetch_all(conn, _EMBEDDING_LINEAGE_SQL, (requested,))
        if len(lineage) != 1:
            raise ValueError(f"{requested!r} has {len(lineage)} dump lineages in artist_embeddings, expected 1")
    elif requested in published:
        lineage = await _fetch_all(conn, _RELEASE_LINEAGE_SQL, (requested,))
    else:
        raise UsageError(f"model_version {requested!r} is not in artist_embeddings")
    (source_dump_id, source_dump_date) = lineage[0]
    return Target(requested, source_dump_id, source_dump_date)


@contextlib.contextmanager
def spool_lock(spool_root: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``spool_root`` so two runs never write one spool."""
    spool_root.mkdir(parents=True, exist_ok=True)
    with (spool_root / ".lock").open("a") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpoolBusyError(f"another similar-artist run holds {spool_root / '.lock'}") from exc
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# ── The run ──────────────────────────────────────────────────────────────────────────────────


async def run_job(
    pool: Any,
    *,
    model_version: str | None,
    spool_root: Path,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    registry: ReleaseRegistry | None = None,
    threads: int = DEFAULT_THREADS,
    guard: MemoryGuard | None = None,
    checkpoint_every_s: float = DEFAULT_CHECKPOINT_EVERY_S,
    **stage_options: Any,
) -> JobStatus:
    """Resolve the target, then run (or resume) the existing stage under the spool lock.
    ``deadline`` is a ``clock()`` reading. Exceptions propagate; `main` turns them into a
    ``failed`` status."""
    started = clock()
    with spool_lock(spool_root):
        async with pool.connection() as conn:
            target = await resolve_target(conn, model_version)
        if target is None:
            logger.info("⏭️ Every model_version in artist_embeddings is already published; nothing to do")
            return JobStatus(status="done", exit_code=EXIT_DONE, elapsed_s=round(clock() - started, 1))
        result = await run_similar_artists(
            pool,
            model_version=target.model_version,
            source_dump_id=target.source_dump_id,
            source_dump_date=target.source_dump_date,
            spool_root=spool_root,
            registry=registry,
            threads=threads,
            guard=guard,
            checkpoint_every_s=checkpoint_every_s,
            deadline=deadline,
            clock=clock,
            **stage_options,
        )
    return JobStatus.from_result(result, clock() - started)


def failed_status(error: BaseException, elapsed_s: float) -> JobStatus:
    code = EXIT_USAGE if isinstance(error, UsageError) else EXIT_FAILED
    detail = str(error)
    return JobStatus(
        status="failed", exit_code=code, elapsed_s=round(elapsed_s, 1), error=f"{type(error).__name__}: {detail}" if detail else type(error).__name__
    )


def emit_status(status: JobStatus, status_file: Path | None) -> None:
    """Print the status as the last stdout line, and write it atomically to ``status_file``."""
    line = status.to_json()
    if status_file is not None:
        tmp = status_file.with_name(f".{status_file.name}.tmp")
        tmp.write_text(line + "\n")
        tmp.replace(status_file)
    sys.stdout.flush()
    print(line, flush=True)  # noqa: T201 -- the status contract, not a log line.


async def _connect_and_run(
    args: argparse.Namespace, spool_root: Path, deadline: float | None
) -> JobStatus:  # pragma: no cover -- needs a live database.
    from common import AsyncPostgreSQLPool, parse_postgres_host_port  # noqa: PLC0415
    from common.config import _build_postgres_connstr, get_secret  # noqa: PLC0415

    username = get_secret("EMBEDDING_PIPELINE_POSTGRES_USERNAME")
    password = get_secret("EMBEDDING_PIPELINE_POSTGRES_PASSWORD")
    database = getenv("POSTGRES_DATABASE")
    if not (username and password and database):
        raise UsageError("EMBEDDING_PIPELINE_POSTGRES_USERNAME, EMBEDDING_PIPELINE_POSTGRES_PASSWORD and POSTGRES_DATABASE are required")
    host, port = parse_postgres_host_port(_build_postgres_connstr())
    pool = AsyncPostgreSQLPool(
        connection_params={"host": host, "port": port, "dbname": database, "user": username, "password": password},
        min_connections=1,
        max_connections=1,
    )
    await pool.initialize()
    try:
        return await run_job(
            pool,
            model_version=args.model_version,
            spool_root=spool_root,
            deadline=deadline,
            threads=args.threads,
            guard=MemoryGuard(budget_bytes=int(args.budget_gb * 1024**3)),
            checkpoint_every_s=args.checkpoint_every,
        )
    finally:
        await pool.close()


def main(argv: list[str] | None = None, *, run: Callable[[argparse.Namespace, Path, float | None], Any] | None = None) -> NoReturn:
    """Entry point: run, print the status line, and exit with its code. ``run`` replaces the
    database-backed run in tests."""
    started = time.monotonic()
    wall_started = datetime.now(UTC)
    status_file: Path | None = None
    try:
        args = parse_args(argv)
        status_file = args.status_file
        setup_logging(SERVICE_NAME, log_file=Path(f"/logs/{SERVICE_NAME}.log"))
        spool_root = resolve_spool_root(args.spool_dir, os.environ.get(SPOOL_DIR_ENV))
        deadline = resolve_deadline(args.deadline, args.max_runtime, wall_now=wall_started, mono_now=started)
        status = asyncio.run((run or _connect_and_run)(args, spool_root, deadline))
        status = replace(status, elapsed_s=round(time.monotonic() - started, 1))
    except Exception as error:
        logger.exception("❌ Similar-artist run failed")
        status = failed_status(error, time.monotonic() - started)
    logger.info(
        "🏁 Similar-artist run finished",
        status=status.status,
        model_version=status.model_version,
        blocks_done=status.blocks_done,
        blocks=status.blocks_total,
    )
    emit_status(status, status_file)
    sys.exit(status.exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
