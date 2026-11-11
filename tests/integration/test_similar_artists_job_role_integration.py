"""The nightly similar-artist job, run as the restricted `embedding_pipeline` LOGIN role (gm-analytics-engine-js2v).

gm-analytics-engine-dhit and -d6vm tested every query the job added (target resolution, the
loader latch read, `pg_locks`, `pg_stat_activity`, the write and publish) against fakes only, so
nothing proved the production role may run them. On Lux the role had no SELECT on
`public.loader_extraction_latch` and the load check would have raised every night.

What this module does, and why it cannot drift:

- **Provisioning.** `tests/integration/conftest.py` applies the pinned `groovemap-database-schema`'s
  own `create_postgres_schema`, which creates the NOLOGIN `embedding_pipeline` group role and every
  grant it holds in production, then makes `embedding_pipeline_login`, a LOGIN member of it. No
  grant is written in this repository. The pin is the one `groovemap-database-schema` revision in
  `pyproject.toml` (`[tool.uv.sources]`, mirrored in `uv.lock`): bump it, run `uv lock`, and this
  module follows.
- **The job runs as the role.** Each case starts `python -m insights.similar_artists_job` (the
  `analytics-engine-similar-artists` entry point) in a child process whose environment holds only
  the restricted login's credentials, never the owner's, so no job query can run as the owner or a
  superuser. The owner pool is used only by fixtures and assertions, to seed and to read back.
- **The latch grant.** The pinned revision is checked against the live database
  (`has_table_privilege`), not against a version number. While it lacks SELECT on the latch
  (before gm-database-schema-37cl), `test_load_check_fails_on_the_latch_read_without_the_grant`
  records the failure the nightly run would hit, and the end-to-end cases report XFAIL naming it.
  Once the pin carries the grant, that test skips and the end-to-end cases must pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psycopg
import pytest
import pytest_asyncio

from insights.embedding_pipeline import EMBEDDING_PIPELINE_LOCK_KEY
from insights.similar_artists_job import EXIT_DONE, EXIT_FAILED, EXIT_PAUSED, LATCH_LOADERS, LATCH_STALE_AFTER
from tests.integration.conftest import _PIPELINE_LOGIN_ROLE


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from common import AsyncPostgreSQLPool

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session"), pytest.mark.timeout(180)]

MODEL_VERSION = "role-test-v1"
DUMP_ID = "role-test-dump-2026-11"
DUMP_DATE = date(2026, 11, 1)
K = 50  # The kernel's default list length.
ARTISTS = 80  # More than K, so every list is full; one row block, so a run is quick and light.
DIMENSIONS = 128  # The width of `public.artist_embeddings.embedding` (halfvec(128)).
LATCH_TABLE = "public.loader_extraction_latch"

_MISSING_GRANT = (
    f"the pinned database-schema does not grant {_PIPELINE_LOGIN_ROLE}'s group role SELECT on {LATCH_TABLE} "
    "(gm-database-schema-37cl); bump the groovemap-database-schema pin to the release that carries it"
)


# ── Environment and process helpers ──────────────────────────────────────────────────────────────


def _job_env() -> dict[str, str]:
    """The child's whole environment: the restricted login's credentials and nothing of the owner's."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "POSTGRES_HOST": os.environ["POSTGRES_HOST"],
        "POSTGRES_DATABASE": os.environ["POSTGRES_DATABASE"],
        "EMBEDDING_PIPELINE_POSTGRES_USERNAME": os.environ["EMBEDDING_PIPELINE_POSTGRES_USERNAME"],
        "EMBEDDING_PIPELINE_POSTGRES_PASSWORD": os.environ["EMBEDDING_PIPELINE_POSTGRES_PASSWORD"],
    }
    assert env["EMBEDDING_PIPELINE_POSTGRES_USERNAME"] == _PIPELINE_LOGIN_ROLE
    assert env["EMBEDDING_PIPELINE_POSTGRES_USERNAME"] != os.environ["POSTGRES_USERNAME"]
    return env


async def _run_job(spool: Path, *args: str) -> tuple[int, dict[str, Any], str]:
    """Run the job as a process; return its exit code, its final status line, and its stdout."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "insights.similar_artists_job",
        "--spool-dir",
        str(spool),
        "--threads",
        "2",
        *args,
        env=_job_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    async with asyncio.timeout(120):
        raw, _ = await process.communicate()
    output = raw.decode(errors="replace")
    status_lines = [line for line in output.splitlines() if line.startswith("{") and '"kind":"similar_artists_status"' in line]
    assert status_lines, f"the job printed no status line (exit {process.returncode}):\n{output}"
    assert process.returncode is not None
    return process.returncode, json.loads(status_lines[-1]), output


def _past_deadline() -> str:
    return (datetime.now(UTC) - timedelta(minutes=1)).isoformat()


# ── Fixtures ─────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def spool() -> Iterator[Path]:
    # pytest's tmp_path is under the system temp dir, and the job refuses /tmp, /var/tmp and
    # /dev/shm (a spool there would not survive a restart), so make one beside the pytest cache.
    base = Path(".pytest_cache").resolve() / "similar-artists-role-spool"
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(dir=base))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest_asyncio.fixture(loop_scope="session")
async def catalog(schema_owner_pool: AsyncPostgreSQLPool) -> AsyncIterator[None]:
    """A small synthetic catalog in `artist_embeddings`, seeded as the owner, and no latch rows."""
    rng = np.random.default_rng(20261111)
    vectors = rng.standard_normal((ARTISTS, DIMENSIONS)).astype(np.float16)
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(f"DELETE FROM {LATCH_TABLE}")  # noqa: S608 -- constant table; TRUNCATE is blocked by a foreign key.
        for number, vector in enumerate(vectors):
            literal = "[" + ",".join(f"{float(x):.4f}" for x in vector) + "]"
            await cursor.execute(
                "INSERT INTO public.artist_embeddings (artist_id, model_version, embedding, source_dump_id, source_dump_date) "
                "VALUES (%s, %s, %s::halfvec, %s, %s)",
                (f"artist-{number:03d}", MODEL_VERSION, literal, DUMP_ID, DUMP_DATE),
            )
        await conn.commit()
    yield
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(f"DELETE FROM {LATCH_TABLE}")  # noqa: S608 -- constant table.
        await conn.commit()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def latch_select_granted(schema_owner_pool: AsyncPostgreSQLPool) -> bool:
    """Whether the pinned schema lets the restricted login SELECT the loader latch."""
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("SELECT has_table_privilege(%s, %s, 'SELECT')", (_PIPELINE_LOGIN_ROLE, LATCH_TABLE))
        (granted,) = await cursor.fetchone()
    return bool(granted)


@pytest.fixture
def needs_latch_grant(latch_select_granted: bool) -> None:
    """Report XFAIL (naming gm-database-schema-37cl) until the pinned schema grants the latch read."""
    if not latch_select_granted:
        pytest.xfail(_MISSING_GRANT)


async def _insert_latch(pool: AsyncPostgreSQLPool, *, loader: str, version: str, age: timedelta, refreshed: bool = False) -> None:
    created = datetime.now(UTC) - age
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            f"INSERT INTO {LATCH_TABLE} (loader, version, signals, created_at, refreshed_at) VALUES (%s, %s, %s, %s, %s)",  # noqa: S608 -- constant table.
            (loader, version, ["releases"], created, created if refreshed else None),
        )
        await conn.commit()


@pytest.fixture
def advisory_lock_holder() -> Iterator[psycopg.Connection[Any]]:
    """Another session that holds the embeddings pipeline's advisory lock, released on exit."""
    conn = psycopg.connect(
        host=os.environ["POSTGRES_HOST"].rsplit(":", 1)[0],
        port=int(os.environ["POSTGRES_HOST"].rsplit(":", 1)[1]),
        dbname=os.environ["POSTGRES_DATABASE"],
        user=os.environ["POSTGRES_USERNAME"],
        password=os.environ["POSTGRES_PASSWORD"],
        autocommit=True,
    )
    conn.execute("SELECT pg_advisory_lock(%s)", (EMBEDDING_PIPELINE_LOCK_KEY,))
    try:
        yield conn
    finally:
        conn.close()


class _RollBackWrite(Exception):
    """Raised inside a transaction to roll it back."""


async def _published(pool: AsyncPostgreSQLPool) -> tuple[int, int]:
    """(published release artists, stored full top-K lists) for MODEL_VERSION, read as the owner."""
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("SELECT COALESCE(max(artists), 0) FROM public.artist_embedding_releases WHERE model_version = %s", (MODEL_VERSION,))
        (artists,) = await cursor.fetchone()
        await cursor.execute(
            "SELECT count(*) FROM public.artist_similar_artists AS s JOIN public.artist_embedding_releases AS r USING (release_id) "
            "WHERE r.model_version = %s AND cardinality(s.similar_artist_ids) = %s",
            (MODEL_VERSION, K),
        )
        (rows,) = await cursor.fetchone()
    return int(artists), int(rows)


# ── The documented failure on a pinned schema without the latch grant ────────────────────────────


async def test_load_check_fails_on_the_latch_read_without_the_grant(
    catalog: None, spool: Path, latch_select_granted: bool, schema_owner_pool: AsyncPostgreSQLPool
) -> None:
    """Without the grant the check raises, the job exits `failed`, and nothing is computed or written.

    This is the nightly failure found on Lux 2026-11-11. It skips once the pinned database-schema
    carries gm-database-schema-37cl, which is the cue that the end-to-end cases below now run.
    """
    if latch_select_granted:
        pytest.skip("the pinned database-schema grants SELECT on the loader latch (gm-database-schema-37cl)")

    code, status, output = await _run_job(spool, "--skip-check-only")

    assert code == EXIT_FAILED, output
    assert status["status"] == "failed"
    assert status["decision"] is None
    assert "permission denied for table loader_extraction_latch" in status["error"], status
    assert await _published(schema_owner_pool) == (0, 0)


# ── The full path as the restricted role (needs the latch grant) ─────────────────────────────────


@pytest.mark.usefixtures("needs_latch_grant")
async def test_check_only_decides_run_with_no_loads_in_flight(catalog: None, spool: Path) -> None:
    code, status, output = await _run_job(spool, "--skip-check-only")

    assert code == EXIT_DONE, output
    assert (status["status"], status["decision"], status["reason"]) == ("checked", "run", None)
    assert status["model_version"] == MODEL_VERSION
    assert not list(spool.glob("*/")), "--skip-check-only must not touch the spool"


@pytest.mark.usefixtures("needs_latch_grant")
async def test_check_only_skips_while_a_loader_latch_is_in_progress(catalog: None, spool: Path, schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _insert_latch(schema_owner_pool, loader=LATCH_LOADERS[0], version="2026-11", age=timedelta(hours=1))

    code, status, output = await _run_job(spool, "--skip-check-only")

    assert code == EXIT_DONE, output
    assert (status["status"], status["decision"]) == ("checked", "skip")
    assert "discogs extraction 2026-11 latch open" in status["reason"]
    assert "extraction in progress" in status["reason"]


@pytest.mark.usefixtures("needs_latch_grant")
async def test_check_only_runs_past_a_finished_or_stale_latch(catalog: None, spool: Path, schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _insert_latch(schema_owner_pool, loader="discogs", version="2026-10", age=timedelta(hours=3), refreshed=True)
    await _insert_latch(schema_owner_pool, loader="musicbrainz", version="2026-09", age=LATCH_STALE_AFTER + timedelta(hours=1))

    code, status, output = await _run_job(spool, "--skip-check-only")

    assert code == EXIT_DONE, output
    assert (status["status"], status["decision"]) == ("checked", "run")
    assert "past the 48 h bound: ignored" in status["reason"]


@pytest.mark.usefixtures("needs_latch_grant")
async def test_check_only_skips_while_the_embeddings_advisory_lock_is_held(
    catalog: None, spool: Path, advisory_lock_holder: psycopg.Connection[Any]
) -> None:
    del advisory_lock_holder  # held for the duration of the run, by another session.

    code, status, output = await _run_job(spool, "--skip-check-only")

    assert code == EXIT_DONE, output
    assert (status["status"], status["decision"]) == ("checked", "skip")
    assert "the embeddings pipeline holds its run lock" in status["reason"]


@pytest.mark.usefixtures("needs_latch_grant")
async def test_check_only_skips_while_the_pipeline_role_has_an_uncommitted_embeddings_write(
    catalog: None, spool: Path, pipeline_pool: AsyncPostgreSQLPool
) -> None:
    """The pipeline's own DELETE + COPY transaction, held open by a second session of the same role."""
    async with pipeline_pool.connection() as conn:
        with pytest.raises(_RollBackWrite):
            async with conn.transaction():
                await conn.execute("DELETE FROM public.artist_embeddings WHERE model_version = %s AND artist_id = 'artist-000'", (MODEL_VERSION,))
                code, status, output = await _run_job(spool, "--skip-check-only")
                raise _RollBackWrite

    assert code == EXIT_DONE, output
    assert (status["status"], status["decision"]) == ("checked", "skip")
    assert "uncommitted writes to public.artist_embeddings" in status["reason"]


@pytest.mark.usefixtures("needs_latch_grant")
async def test_a_run_skips_while_a_latch_is_in_progress_and_writes_nothing(
    catalog: None, spool: Path, schema_owner_pool: AsyncPostgreSQLPool
) -> None:
    await _insert_latch(schema_owner_pool, loader="musicbrainz", version="2026-11", age=timedelta(minutes=30))

    code, status, output = await _run_job(spool)

    assert code == EXIT_DONE, output
    assert (status["status"], status["decision"], status["published"]) == ("skipped", "skip", False)
    assert await _published(schema_owner_pool) == (0, 0)


@pytest.mark.usefixtures("needs_latch_grant")
async def test_pause_resume_and_publish_as_the_restricted_role(catalog: None, spool: Path, schema_owner_pool: AsyncPostgreSQLPool) -> None:
    # A deadline that has already passed: the load check, target resolution and the embeddings
    # read all run, no block is computed, and the run pauses without writing.
    code, paused, output = await _run_job(spool, "--deadline", _past_deadline())
    assert code == EXIT_PAUSED, output
    assert (paused["status"], paused["decision"], paused["published"]) == ("paused", "run", False)
    assert paused["model_version"] == MODEL_VERSION
    assert paused["blocks_done"] == 0
    assert paused["blocks_total"] == 1
    assert await _published(schema_owner_pool) == (0, 0)

    # The next night resumes the same spool, then writes and publishes in one run.
    code, done, output = await _run_job(spool)
    assert code == EXIT_DONE, output
    assert (done["status"], done["decision"], done["published"], done["already_published"]) == ("done", "run", True, False)
    assert (done["artists"], done["blocks_done"], done["blocks_total"]) == (ARTISTS, 1, 1)
    assert done["rows_written"] == ARTISTS
    assert await _published(schema_owner_pool) == (ARTISTS, ARTISTS)

    # Once published, a rerun resolves nothing pending and does nothing.
    code, again, output = await _run_job(spool)
    assert code == EXIT_DONE, output
    assert (again["status"], again["model_version"], again["published"]) == ("done", None, False)
    assert await _published(schema_owner_pool) == (ARTISTS, ARTISTS)
