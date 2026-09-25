"""The monthly FastRP embedding load, under the `embedding_pipeline` role (ADR 0013).

`insights/embeddings/` holds the algorithm and its interfaces; this module is "the pipeline
that calls it" docs/embeddings.md defers to — reading the graph from PostgreSQL, running
`fastrp`, and writing `public.artist_embeddings`. It is a separate entry point
(`analytics-engine-embeddings`), not part of the always-on FastAPI service in
`insights.insights`, because it connects under a different, deliberately narrower role.

ADR 0013's 2026-09-24 amendment grants `embedding_pipeline` `SELECT` on every relation in the
`graph` schema and `SELECT, INSERT, UPDATE, DELETE` on `public.artist_embeddings` alone — no
DDL, no ownership, nothing on any other schema. Three consequences shape this module:

- **No index DDL, ever.** Building or rebuilding the ANN index over a `model_version` needs
  table ownership this role does not have and never will; `_log_operator_step` logs the
  statement an operator with a different, more privileged credential runs after this job's
  transaction commits (`build_artist_embeddings_index`-equivalent — see
  database-schema's "Building the artist HNSW index"). Per-`model_version` partial indexes
  (`gm-database-schema-19g5`) are not landed as of this writing, so the statement logged is the
  forward-looking, `model_version`-filtered shape rather than something database-schema
  documents today. Retiring a superseded `model_version`'s rows and index is that follow-on's
  business, never this job's.
- **No writes outside `artist_embeddings`.** Every read below is a `SELECT` against `graph`;
  the one write statement targets `public.artist_embeddings` and nothing else. A connection
  authenticated as this role gets a permission error on anything else — see
  `tests/integration/test_embedding_pipeline_integration.py`.
- **No log table.** Other scheduled computations in `insights/computations.py` write their
  outcome to `insights.computation_log`; this role holds nothing on the `insights` schema, so
  this job cannot do that. `public.artist_embeddings` itself — `source_dump_id`,
  `source_dump_date`, `computed_at` — is the only durable lineage record it can leave, and
  doubles as the idempotency check below.

## The stored `model_version` is per dump, not per algorithm

`FastRPConfig.model_version` (`insights/embeddings/fastrp.py`) names only the method, its
parameters, and the projection seed rule — the same string every month an operator does not
change the algorithm. Writing that string directly into `artist_embeddings.model_version`
would mean a second month's load lands on the *same* primary key, `(artist_id, model_version)`,
as the first: it would upsert the first month's rows in place under a version `catalog-api` may
still be serving, drive row-by-row HNSW maintenance on that live index instead of a clean
`CREATE INDEX CONCURRENTLY`, discard the first dump's rows before a purge-by-dump could ever
reach them, and leave ieu.3's month-over-month churn measurement with only one month on disk to
compare.

`stored_model_version(config, dump_id)` — `f"{config.model_version}@{dump_id}"` — is what this
module actually reads and writes as `model_version`. Composing the dump id in is what lets two
months coexist: each dump gets its own primary-key value, so a second month's load is a
brand-new set of rows, never a write to the first month's. `config.model_version` (the pure
method string) is recorded separately, in every log line here, as `method_version` — see
"Bit-identity and lineage" below.

## Idempotency

The job is idempotent per stored `model_version` (which already encodes `(dump_id,
method_version)`): re-running it for a dump already recorded under this algorithm is a no-op.
The read-only idempotency check (`_already_loaded`) and the write (`_write_embeddings`) are not
one transaction — the read runs before the (multi-minute, CPU-bound) graph read and `fastrp`
compute, and only the write itself is wrapped in a transaction. That is deliberate rather than a
race: if the write transaction never commits (a crash, a killed process), no row carries the new
stored `model_version`, so a retry's idempotency check correctly reports "not loaded" and redoes
the full load; holding one long-lived transaction across the whole compute would only add lock
and connection-lifetime risk for no additional safety.

`_write_embeddings`' `ON CONFLICT (artist_id, model_version) DO UPDATE` is retry safety for
exactly that crash case — a partial previous attempt's rows under *this same* stored
`model_version` — not a cross-dump upsert: since the stored value already differs per dump, the
conflict target can never match a row from a different month. Every query and write in this
module is scoped by the stored `model_version` throughout, so a version `catalog-api` is
currently serving is untouched by a load of a new one.

## Bit-identity and lineage

docs/embeddings.md's "Determinism" section notes that bit-identity holds for one build of NumPy
and SciPy, so cross-month comparisons need to know which build produced which vectors. Every
run logs `numpy_version`/`scipy_version` alongside `method_version`/`model_version`/`dump_id`
at the start of `load_embeddings`, since `public.artist_embeddings` itself has no column for
them — the pipeline role's lineage columns (`source_dump_id`, `source_dump_date`, `computed_at`)
are fixed by database-schema's DDL, so logs are the durable record for this.

## Scheduling

There is no in-process scheduler loop here, unlike `insights.insights`'s `_scheduler_loop`.
This is a one-shot script (`analytics-engine-embeddings`, `main()` below), meant to be invoked
by the deployment layer once a month, after that month's dump has loaded and `SOURCE_DUMP_ID`/
`SOURCE_DUMP_DATE` are known to the invoker. Wiring the exact monthly trigger (cron, a
`CronJob`, an operator running it by hand) is a deployment-repo concern outside this bead.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import scipy
import structlog
from common import (
    AsyncPostgreSQLPool,
    describe_exception,
    parse_postgres_host_port,
    setup_logging,
    setup_telemetry,
    shutdown_telemetry,
)
from common.config import _build_postgres_connstr, get_secret

from insights.embeddings import AdjacencyBuilder, FastRPConfig, NodeIndex, fastrp, node_keys
from insights.telemetry import computation_span, record_computation, record_embedding_pipeline_failure, record_embedding_rows_written


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


logger = structlog.get_logger(__name__)

SERVICE_NAME: Final = "analytics-engine-embeddings"
COMPUTATION_NAME: Final = "embedding_pipeline"

ARTIST_EMBEDDINGS_TABLE: Final = "public.artist_embeddings"

# The six vertex kinds insights/embeddings/graph.py's node identity covers (its module
# docstring: "a one-character kind (a artist, r release, l label, m master, g genre, s
# style)"). `graph.vertex_degree` carries one row per vertex of the *whole* property-graph
# path-traversal surface (person, company, medium, ...); filtering to these six is what makes
# this the FastRP subgraph rather than that wider graph.
_VERTEX_KINDS: Final = ("a", "r", "l", "m", "g", "s")

# (relation, source column, source kind, target column, target kind) for every edge relation
# that connects two of the six kinds above -- release<->artist/label/master/genre/style and
# master<->artist/genre/style, exactly the bipartite shape the six kinds admit. Declared in
# database-schema's `graph.catalog` property graph (docs/architecture.md, "Property graph").
_EDGE_RELATIONS: Final[tuple[tuple[str, str, str, str, str], ...]] = (
    ("graph.by_artist", "release_id", "r", "artist_id", "a"),
    ("graph.on_label", "release_id", "r", "label_id", "l"),
    ("graph.derived_from", "release_id", "r", "master_id", "m"),
    ("graph.in_genre", "release_id", "r", "genre_name", "g"),
    ("graph.in_style", "release_id", "r", "style_name", "s"),
    ("graph.master_by_artist", "master_id", "m", "artist_id", "a"),
    ("graph.master_in_genre", "master_id", "m", "genre_name", "g"),
    ("graph.master_in_style", "master_id", "m", "style_name", "s"),
)

# Rows fetched per round trip from a server-side (named) cursor. Bounds how much of one block
# a Python list holds at a time; the full result set — up to 32.8M vertices, 222M edges at
# catalog scale (docs/embeddings.md) — is never materialized in one piece.
_CURSOR_FETCH_SIZE: Final = 50_000

# Rows per multi-row upsert statement.
_UPSERT_BATCH_SIZE: Final = 5_000

_UPSERT_SQL: Final = f"""
    INSERT INTO {ARTIST_EMBEDDINGS_TABLE}
        (artist_id, model_version, embedding, source_dump_id, source_dump_date, computed_at)
    VALUES (%s, %s, %s::halfvec, %s, %s, %s)
    ON CONFLICT (artist_id, model_version) DO UPDATE SET
        embedding = EXCLUDED.embedding,
        source_dump_id = EXCLUDED.source_dump_id,
        source_dump_date = EXCLUDED.source_dump_date,
        computed_at = EXCLUDED.computed_at
"""  # noqa: S608 -- ARTIST_EMBEDDINGS_TABLE is a module constant, not caller input.

_ALREADY_LOADED_SQL: Final = f"SELECT 1 FROM {ARTIST_EMBEDDINGS_TABLE} WHERE model_version = %s LIMIT 1"  # noqa: S608

_NAME_SLUG_PATTERN: Final = re.compile(r"[^a-z0-9]+")

# The separator composing the stored `model_version` from the algorithm's own version and the
# dump id — see "The stored model_version is per dump, not per algorithm" above. Rejected
# inside a dump id so the composed string is always unambiguous to a human reading it back.
_STORED_VERSION_SEPARATOR: Final = "@"


@dataclass(frozen=True)
class EmbeddingPipelineConfig:
    """Configuration for one monthly embedding load, under the `embedding_pipeline` role.

    Deliberately separate from `insights.config.InsightsConfig`: this job authenticates as a
    different, narrower-privileged role than the always-on service, so it reads its own
    username/password secrets rather than the service's `POSTGRES_USERNAME`/`POSTGRES_PASSWORD`.
    It shares the same host, port, and database, because `graph` and `public.artist_embeddings`
    live in the one catalog database every service already points at.
    """

    postgres_host: str
    postgres_username: str
    postgres_password: str
    postgres_database: str
    source_dump_id: str
    source_dump_date: date

    @classmethod
    def from_env(cls) -> EmbeddingPipelineConfig:
        """Create configuration from environment variables.

        Raises:
            ValueError: If a required variable is missing, or `SOURCE_DUMP_DATE` is not an
                ISO date.
        """
        postgres_username = get_secret("EMBEDDING_PIPELINE_POSTGRES_USERNAME")
        postgres_password = get_secret("EMBEDDING_PIPELINE_POSTGRES_PASSWORD")
        postgres_database = getenv("POSTGRES_DATABASE")
        source_dump_id = getenv("SOURCE_DUMP_ID")
        source_dump_date_raw = getenv("SOURCE_DUMP_DATE")
        missing_vars = [
            name
            for name, value in (
                ("EMBEDDING_PIPELINE_POSTGRES_USERNAME", postgres_username),
                ("EMBEDDING_PIPELINE_POSTGRES_PASSWORD", postgres_password),
                ("POSTGRES_DATABASE", postgres_database),
                ("SOURCE_DUMP_ID", source_dump_id),
                ("SOURCE_DUMP_DATE", source_dump_date_raw),
            )
            if not value
        ]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        try:
            source_dump_date = date.fromisoformat(cast("str", source_dump_date_raw))
        except ValueError as exc:
            raise ValueError(f"SOURCE_DUMP_DATE must be an ISO date (YYYY-MM-DD), got {source_dump_date_raw!r}") from exc
        return cls(
            postgres_host=_build_postgres_connstr(),
            postgres_username=cast("str", postgres_username),
            postgres_password=cast("str", postgres_password),
            postgres_database=cast("str", postgres_database),
            source_dump_id=cast("str", source_dump_id),
            source_dump_date=source_dump_date,
        )


@dataclass(frozen=True)
class LoadResult:
    """The outcome of one embedding load attempt.

    `method_version` is the pure `FastRPConfig.model_version` (algorithm, parameters, seed);
    `model_version` is `stored_model_version(config, dump_id)`, the value actually written to
    and read from `artist_embeddings.model_version`. See the module docstring.
    """

    method_version: str
    model_version: str
    rows_written: int
    skipped: bool


def stored_model_version(config: FastRPConfig, dump_id: str) -> str:
    """The `artist_embeddings.model_version` value one dump's load reads and writes.

    Composes the algorithm's own version with the dump id so two dumps under an unchanged
    algorithm land on different primary-key values instead of one upserting the other's rows
    in place — see the module docstring. `model_version` is `TEXT`, so there is no length
    bound to enforce here beyond what a reasonable `dump_id` already is.

    Args:
        config: The FastRP method configuration.
        dump_id: The current dump's identifier.

    Raises:
        ValueError: If `dump_id` contains the separator (`@`), which would make the composed
            string ambiguous to read back.
    """
    if _STORED_VERSION_SEPARATOR in dump_id:
        raise ValueError(f"dump_id must not contain {_STORED_VERSION_SEPARATOR!r}, got {dump_id!r}")
    return f"{config.model_version}{_STORED_VERSION_SEPARATOR}{dump_id}"


def _sql_string_literal(value: str) -> str:
    """A single-quoted SQL string literal, escaped for the operator step's logged statement.

    Logged only, never executed by this job — but an operator may copy it verbatim, and a
    `dump_id` is free text that could otherwise carry a quote that breaks the pasted statement.
    """
    return "'" + value.replace("'", "''") + "'"


def _halfvec_literal(vector: NDArray[np.floating]) -> str:
    """Render one embedding row as pgvector's `halfvec` text input format.

    No `pgvector` Python dependency is added for this: the text format
    (``"[v1,v2,...]"``, cast with ``::halfvec`` in `_UPSERT_SQL`) is part of pgvector's SQL
    input syntax and needs no client-side adapter.
    """
    return "[" + ",".join(f"{value:g}" for value in vector.tolist()) + "]"


def _index_name_slug(model_version: str) -> str:
    """Turn a `model_version` string into a safe SQL-identifier fragment for the operator step.

    Logged only — this job never executes the statement it names.
    """
    slug = _NAME_SLUG_PATTERN.sub("_", model_version.lower()).strip("_")
    return slug[:48]


def _log_operator_step(model_version: str) -> None:
    """Log the ANN-index build this job never runs itself. See the module docstring.

    `model_version` here is the *stored* value (`stored_model_version(...)`, including the
    dump id) — the WHERE clause must match what is actually in the table. The index name uses
    `_index_name_slug`, since a dump id can carry characters (`@`, `-`, `/`, ...) that are not
    valid in an unquoted SQL identifier; the WHERE clause's value uses `_sql_string_literal`.
    """
    index_name = f"idx_artist_embeddings_{_index_name_slug(model_version)}_hnsw"
    statement = (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} "
        f"ON {ARTIST_EMBEDDINGS_TABLE} USING hnsw (embedding halfvec_cosine_ops) "
        f"WHERE model_version = {_sql_string_literal(model_version)}"
    )
    logger.info(
        "🛠️ Operator step required — build the ANN index for this model_version",
        model_version=model_version,
        statement=statement,
    )


async def _already_loaded(conn: Any, model_version: str) -> bool:
    """Return whether this stored `model_version` already has rows in `artist_embeddings`."""
    async with conn.cursor() as cursor:
        await cursor.execute(_ALREADY_LOADED_SQL, (model_version,))
        return await cursor.fetchone() is not None


async def _read_vertices(conn: Any) -> tuple[NodeIndex, list[str]]:
    """Stream every `(kind, key)` vertex of the six FastRP kinds; return it and the artist ids.

    A named (server-side) cursor, so the full vertex set is never materialized as a Python
    list in one piece. PostgreSQL only allows `DECLARE CURSOR` inside a transaction block, so
    this opens one read-only transaction for the duration of the scan — `AsyncPostgreSQLPool`
    connections default to autocommit, unlike a plain `psycopg.AsyncConnection`.
    """
    key_chunks: list[NDArray[np.uint64]] = []
    artist_ids: list[str] = []
    async with conn.transaction(), conn.cursor(name="embedding_pipeline_vertices") as cursor:
        await cursor.execute("SELECT kind, key FROM graph.vertex_degree WHERE kind = ANY(%s)", (list(_VERTEX_KINDS),))
        while True:
            batch = await cursor.fetchmany(_CURSOR_FETCH_SIZE)
            if not batch:
                break
            key_chunks.append(node_keys(batch))
            artist_ids.extend(key for kind, key in batch if kind == "a")
    keys = np.concatenate(key_chunks) if key_chunks else np.zeros(0, dtype=np.uint64)
    return NodeIndex(keys), artist_ids


async def _stream_edge_blocks(conn: Any, builder: AdjacencyBuilder) -> None:
    """Add every edge of the eight relations that connect the six FastRP kinds, in blocks.

    One read-only transaction for the whole scan — see `_read_vertices` on why a named
    cursor needs one.
    """
    async with conn.transaction():
        for table, source_column, source_kind, target_column, target_kind in _EDGE_RELATIONS:
            cursor_name = f"embedding_pipeline_{table.replace('.', '_')}"
            async with conn.cursor(name=cursor_name) as cursor:
                await cursor.execute(f"SELECT {source_column}, {target_column} FROM {table}")  # noqa: S608 -- table/columns are from the fixed _EDGE_RELATIONS tuple.
                while True:
                    batch = await cursor.fetchmany(_CURSOR_FETCH_SIZE)
                    if not batch:
                        break
                    sources = node_keys((source_kind, str(source_key)) for source_key, _target_key in batch)
                    targets = node_keys((target_kind, str(target_key)) for _source_key, target_key in batch)
                    builder.add_edges(sources, targets)


async def _write_embeddings(
    conn: Any,
    *,
    model_version: str,
    dump_id: str,
    dump_date: date,
    artist_ids: Sequence[str],
    vectors: NDArray[np.floating],
) -> int:
    """Upsert every artist's embedding row for the stored `model_version`, in one transaction.

    `model_version` here is the *stored* value (`stored_model_version(...)`), already unique
    per dump — so `ON CONFLICT (artist_id, model_version) DO UPDATE` can only ever match a row
    this same dump's own, possibly-partial, previous attempt wrote, never another dump's. It
    exists purely for that crash-retry case: `COPY` has no `ON CONFLICT` clause, and a plain
    re-`INSERT` would fail outright on a retry that reaches those previously-committed rows
    (which cannot happen after this function returns, by the idempotency check above, but a
    retry that resumes mid-run before that check would still see them).
    """
    computed_at = datetime.now(UTC)
    rows_written = 0
    async with conn.transaction(), conn.cursor() as cursor:
        for start in range(0, len(artist_ids), _UPSERT_BATCH_SIZE):
            stop = min(start + _UPSERT_BATCH_SIZE, len(artist_ids))
            batch = [
                (
                    artist_ids[index],
                    model_version,
                    _halfvec_literal(vectors[index]),
                    dump_id,
                    dump_date,
                    computed_at,
                )
                for index in range(start, stop)
            ]
            await cursor.executemany(_UPSERT_SQL, batch)
            rows_written += len(batch)
    return rows_written


async def load_embeddings(pool: AsyncPostgreSQLPool, config: FastRPConfig, dump_id: str, dump_date: date) -> LoadResult:
    """Load one month's FastRP embeddings for `dump_id`, or no-op if already loaded.

    Args:
        pool: A pool connected as the `embedding_pipeline` role (or a role holding it).
        config: The FastRP method configuration; `config.model_version` is recorded as
            `method_version` in every log line here.
        dump_id: The current dump's identifier, recorded as `source_dump_id` lineage and
            composed into the stored `model_version` (see the module docstring).
        dump_date: The current dump's date, recorded as `source_dump_date` lineage.

    Returns:
        The load outcome — `skipped=True` when this stored `model_version` was already loaded.

    Raises:
        ValueError: If `dump_id` contains `stored_model_version`'s separator (`@`).
    """
    version = stored_model_version(config, dump_id)
    logger.info(
        "🔢 Embedding pipeline run starting",
        method_version=config.model_version,
        model_version=version,
        dump_id=dump_id,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
    )
    async with pool.connection() as conn:
        if await _already_loaded(conn, version):
            logger.info("⏭️ Embedding load skipped — already loaded", model_version=version, dump_id=dump_id)
            return LoadResult(method_version=config.model_version, model_version=version, rows_written=0, skipped=True)

        nodes, artist_ids = await _read_vertices(conn)
        if not artist_ids:
            logger.warning("⚠️ No artist vertices found in the graph — nothing to embed", model_version=version, dump_id=dump_id)
            return LoadResult(method_version=config.model_version, model_version=version, rows_written=0, skipped=False)

        builder = AdjacencyBuilder(nodes)
        await _stream_edge_blocks(conn, builder)
        adjacency = builder.build()

        artist_positions = nodes.positions(node_keys(("a", artist_id) for artist_id in artist_ids))
        # float16 output: the stored column is `halfvec`, so a wider dtype buys nothing and
        # doubles the resident array (docs/embeddings.md, "Memory and time at catalog scale").
        # block_columns defaults to 4 and threads=6 for the same reason: the pair the 12 GB
        # full-catalog budget was measured against.
        vectors = fastrp(adjacency, config, rows=artist_positions, out_dtype=np.float16, threads=6)

        rows_written = await _write_embeddings(
            conn,
            model_version=version,
            dump_id=dump_id,
            dump_date=dump_date,
            artist_ids=artist_ids,
            vectors=vectors,
        )

    logger.info("💾 Embedding load complete", model_version=version, dump_id=dump_id, rows_written=rows_written)
    _log_operator_step(version)
    return LoadResult(method_version=config.model_version, model_version=version, rows_written=rows_written, skipped=False)


async def run_embedding_pipeline(
    pool: AsyncPostgreSQLPool,
    dump_id: str,
    dump_date: date,
    config: FastRPConfig | None = None,
) -> LoadResult:
    """Run one embedding load, recording duration, rows written, and failure metrics.

    Mirrors `insights.computations.run_all_computations`'s span-and-metric shape, without the
    `insights.computation_log` write that function's `_record_lifecycle` also does — this role
    cannot make it (see the module docstring).
    """
    config = config or FastRPConfig()
    started = time.perf_counter()
    try:
        with computation_span(COMPUTATION_NAME):
            result = await load_embeddings(pool, config, dump_id, dump_date)
    except Exception as error:
        record_computation(COMPUTATION_NAME, time.perf_counter() - started, success=False)
        record_embedding_pipeline_failure()
        logger.error(
            "❌ Embedding pipeline failed",
            error=describe_exception(error),
            method_version=config.model_version,
            dump_id=dump_id,
        )
        raise
    record_computation(COMPUTATION_NAME, time.perf_counter() - started, success=True)
    record_embedding_rows_written(result.rows_written)
    return result


async def _initialize_pool(config: EmbeddingPipelineConfig) -> AsyncPostgreSQLPool:
    """Connect the one-shot pool, as the `embedding_pipeline`-scoped login."""
    host, port = parse_postgres_host_port(config.postgres_host)
    pool = AsyncPostgreSQLPool(
        connection_params={
            "host": host,
            "port": port,
            "dbname": config.postgres_database,
            "user": config.postgres_username,
            "password": config.postgres_password,
        },
        min_connections=1,
        max_connections=1,
    )
    await pool.initialize()
    return pool


async def _run(config: EmbeddingPipelineConfig) -> LoadResult:
    pool = await _initialize_pool(config)
    try:
        return await run_embedding_pipeline(pool, config.source_dump_id, config.source_dump_date)
    finally:
        await pool.close()


def main() -> None:  # pragma: no cover -- exercised through _run/run_embedding_pipeline in tests.
    """Run one monthly embedding load; exit non-zero on failure.

    Invoked as `analytics-engine-embeddings` (see `pyproject.toml`), by the deployment layer's
    own monthly schedule, after `SOURCE_DUMP_ID`/`SOURCE_DUMP_DATE` are known — see the module
    docstring's "Scheduling" section.
    """
    setup_logging(SERVICE_NAME, log_file=Path(f"/logs/{SERVICE_NAME}.log"))
    setup_telemetry(SERVICE_NAME)
    try:
        config = EmbeddingPipelineConfig.from_env()
        result = asyncio.run(_run(config))
    except Exception:
        logger.exception("❌ Embedding pipeline run failed")
        shutdown_telemetry()
        sys.exit(1)
    logger.info("✅ Embedding pipeline run complete", model_version=result.model_version, rows_written=result.rows_written, skipped=result.skipped)
    shutdown_telemetry()


if __name__ == "__main__":  # pragma: no cover
    main()
