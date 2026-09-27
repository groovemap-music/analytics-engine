"""Shared fixtures for the embedding-pipeline PG19+pgvector integration tier.

Run via `just test-integration-pg19` / `scripts/test-integration-pg19.sh`, never directly:
that script starts the disposable PostgreSQL 19 + pgvector container and supplies every
environment variable `_required_env` below reads. See that script and `docs/embeddings.md`.

The schema applied here is the real one: `groovemap_schema.postgres.create_postgres_schema`,
from a pinned `groovemap-database-schema` dev dependency (rev
`26d03e66c0b815d15870768a92b7d49521658838`, version 0.4.1), the same way `catalog-api` applies
it in its own `postgres_pool` fixture. That revision carries `gm-database-schema-lhp2` (the
vector extension, `public.artist_embeddings`, the `embedding_pipeline` role and its grants),
`gm-database-schema-19g5` (the per-`model_version` partial HNSW index procedure this repository
does not exercise here — see `docs/embeddings.md`), and `gm-database-schema-ug3v`
(`graph.track_credited_on` / `graph.track_by_artist`, the two relations this bead,
gm-analytics-engine-x3d, adds to the pipeline's edge set). Applying the producer's own DDL —
rather than a hand-rolled subset — is what keeps this fixture from drifting behind the objects
`insights.embedding_pipeline` actually reads; see `test_real_databases.py` in `catalog-api` for
the same rationale.

`create_postgres_schema` also declares `graph.credit_role_category`, the SQL rendering of
`common.credit_roles.ROLE_CATEGORIES` (the same taxonomy
`insights.embedding_pipeline._KEPT_CREDIT_CATEGORIES` filters against) that
`graph.credited_on.role_category` and `graph.track_credited_on.role_category` are GENERATED
from — this fixture no longer renders its own copy of that function.

The eight plain-scan `graph` edge relations plus `graph.vertex_degree` are base tables in the
real schema (loader-written, not views) — `create_postgres_schema` creates them empty, and this
fixture seeds them directly, representative of what a loader's `extraction_complete` pass
leaves behind. `public.catalog_document_sentinel` is the one object here with no real-schema
counterpart: a stand-in for a catalog document table (e.g. `public.releases`) the pipeline role
is never granted anything on, kept minimal rather than switched to a real table so this fixture
does not have to seed one just to prove a permission boundary.

`graph.track_credited_on` and `graph.track_by_artist` are likewise real, loader-written base
tables, not views (the same shape `graph.credited_on`/`graph.same_as` already are) — seeded per
test in `test_embedding_pipeline_integration.py`, not here, since their interesting cases (a
track with an empty `track_position`, two tracks sharing one — a gap the loader bead's own
parity fixture left, since `track_ordinal`/`sub_track_ordinal`, not `track_position`, are the
real primary-key columns) need their own small fixtures rather than this session-scoped base
graph.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from common import AsyncPostgreSQLPool, parse_postgres_host_port
from groovemap_schema.postgres import EMBEDDING_PIPELINE_ROLE, create_postgres_schema
from psycopg import sql


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_PIPELINE_LOGIN_ROLE = "embedding_pipeline_login"

# A small, fully-connected synthetic graph exercising all six FastRP vertex kinds and all
# eight pre-ieu.6 edge relations `insights.embedding_pipeline` reads: artists 1 and 2 share
# release 101, 2 and 3 share release 102; release 101 also carries a label, a master, a genre,
# and a style, and the master repeats the artist/genre/style tags the way a Discogs master
# does. The ninth relation, the release-level credited-artist edge, is exercised separately
# (`test_credited_artist_edges_are_filtered_resolved_and_embedded`) with its own seed rows,
# inserted and cleaned up per test rather than added to this session-scoped base graph.
ARTIST_IDS: tuple[str, ...] = ("1", "2", "3")

_VERTEX_ROWS: tuple[tuple[str, str, int], ...] = (
    ("a", "1", 2),
    ("a", "2", 2),
    ("a", "3", 1),
    ("r", "101", 4),
    ("r", "102", 2),
    ("l", "501", 1),
    ("m", "601", 3),
    ("g", "Fixture Genre", 2),
    ("s", "Fixture Style", 2),
)

_EDGE_ROWS: dict[str, tuple[tuple[str, str], ...]] = {
    "graph.by_artist": (("101", "1"), ("101", "2"), ("102", "2"), ("102", "3")),
    "graph.on_label": (("101", "501"),),
    "graph.derived_from": (("101", "601"),),
    "graph.in_genre": (("101", "Fixture Genre"),),
    "graph.in_style": (("101", "Fixture Style"),),
    "graph.master_by_artist": (("601", "1"),),
    "graph.master_in_genre": (("601", "Fixture Genre"),),
    "graph.master_in_style": (("601", "Fixture Style"),),
}

# (table, column pair) — the eight plain-scan edge relations, in the same shape
# `_EDGE_RELATIONS` in `insights/embedding_pipeline.py` reads. The ninth, `graph.credited_on`
# joined to `graph.same_as`, is not a plain two-column table and is declared separately below.
_EDGE_COLUMNS: dict[str, tuple[str, str]] = {
    "graph.by_artist": ("release_id", "artist_id"),
    "graph.on_label": ("release_id", "label_id"),
    "graph.derived_from": ("release_id", "master_id"),
    "graph.in_genre": ("release_id", "genre_name"),
    "graph.in_style": ("release_id", "style_name"),
    "graph.master_by_artist": ("master_id", "artist_id"),
    "graph.master_in_genre": ("master_id", "genre_name"),
    "graph.master_in_style": ("master_id", "style_name"),
}


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} must be supplied by `just test-integration-pg19`")
    return value


async def _open_pool(*, username: str, password: str) -> AsyncPostgreSQLPool:
    host, port = parse_postgres_host_port(_required_env("POSTGRES_HOST"))
    pool = AsyncPostgreSQLPool(
        connection_params={
            "host": host,
            "port": port,
            "dbname": _required_env("POSTGRES_DATABASE"),
            "user": username,
            "password": password,
        },
        min_connections=1,
        max_connections=2,
        max_retries=1,
        health_check_interval=3600,
    )
    await pool.initialize()
    return pool


async def _apply_schema_and_seed(pool: AsyncPostgreSQLPool) -> None:
    """Apply the real schema, then seed the graph tables and the sentinel it does not declare.

    Every statement `create_postgres_schema` runs is `IF NOT EXISTS`/idempotent, so a non-zero
    failure count means this integration image and the pinned producer revision disagree — a
    fixture bug, not something a test should be left to discover as a missing relation (the
    same assertion `catalog-api`'s `postgres_pool` fixture makes).
    """
    failures = await create_postgres_schema(pool)
    assert failures == 0, f"{failures} schema statements failed against the integration container"
    async with pool.connection() as conn:
        await conn.set_autocommit(True)
        async with conn.cursor() as cursor:
            # A stand-in for a catalog document table (e.g. `public.releases`) the pipeline
            # role is never granted anything on — see the module docstring.
            await cursor.execute("CREATE TABLE IF NOT EXISTS public.catalog_document_sentinel (id TEXT PRIMARY KEY)")
            for kind, key, degree in _VERTEX_ROWS:
                await cursor.execute(
                    "INSERT INTO graph.vertex_degree (kind, key, degree) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING", (kind, key, degree)
                )
            for table, rows in _EDGE_ROWS.items():
                source_column, target_column = _EDGE_COLUMNS[table]
                for source_value, target_value in rows:
                    await cursor.execute(
                        f"INSERT INTO {table} ({source_column}, {target_column}) VALUES (%s, %s) ON CONFLICT DO NOTHING",  # noqa: S608 -- table/columns are from the fixed _EDGE_COLUMNS mapping.
                        (source_value, target_value),
                    )


async def _create_pipeline_login(pool: AsyncPostgreSQLPool, *, password: str) -> None:
    """Create a LOGIN role that is a member of `embedding_pipeline`, for this tier only.

    `embedding_pipeline` itself is NOLOGIN (ADR 0013): production and CI provision a login
    that holds membership in it, never a password on the group role itself. This mirrors
    that shape rather than connecting as the group role directly.

    `CREATE ROLE`'s `PASSWORD` clause takes a string literal, not a bind parameter — inside a
    `DO` block's body PostgreSQL has no parameter slot to type at all (`could not determine
    data type of parameter $1`), so the password is composed client-side with
    `psycopg.sql.Literal` instead of a `%s` placeholder.
    """
    statement = sql.SQL(
        """
        DO $create_pipeline_login$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = {login_role_literal}) THEN
                CREATE ROLE {login_role} LOGIN PASSWORD {password} IN ROLE {group_role};
            END IF;
        END
        $create_pipeline_login$
        """
    ).format(
        login_role_literal=sql.Literal(_PIPELINE_LOGIN_ROLE),
        login_role=sql.Identifier(_PIPELINE_LOGIN_ROLE),
        password=sql.Literal(password),
        group_role=sql.Identifier(EMBEDDING_PIPELINE_ROLE),
    )
    async with pool.connection() as conn:
        await conn.set_autocommit(True)
        async with conn.cursor() as cursor:
            await cursor.execute(statement)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def schema_owner_pool() -> AsyncIterator[AsyncPostgreSQLPool]:
    """The integration container's superuser pool: applies schema, seeds, and provisions.

    Session-scoped: the container the test-runner script starts is one instance for the
    whole run, and seeding is one-shot — seeding a second time would hit each edge table's
    primary key (guarded by `ON CONFLICT DO NOTHING` regardless, but there is no reason to
    redo it).
    """
    pool = await _open_pool(username=_required_env("POSTGRES_USERNAME"), password=_required_env("POSTGRES_PASSWORD"))
    await _apply_schema_and_seed(pool)
    await _create_pipeline_login(pool, password=_required_env("EMBEDDING_PIPELINE_POSTGRES_PASSWORD"))
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pipeline_pool(schema_owner_pool: AsyncPostgreSQLPool) -> AsyncIterator[AsyncPostgreSQLPool]:
    """A pool connected as the `embedding_pipeline`-scoped login the pipeline runs under."""
    del schema_owner_pool  # ordering only: schema, seed, and role must exist first.
    pool = await _open_pool(
        username=_required_env("EMBEDDING_PIPELINE_POSTGRES_USERNAME"),
        password=_required_env("EMBEDDING_PIPELINE_POSTGRES_PASSWORD"),
    )
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _reset_artist_embeddings(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    """Empty `artist_embeddings` before every test — the one state the tests mutate.

    The pipeline role itself cannot `TRUNCATE` (see
    `test_pipeline_role_cannot_write_outside_artist_embeddings`), so this runs as the schema
    owner, keeping each test's `load_embeddings` call independent of the others' writes.
    """
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("TRUNCATE public.artist_embeddings")
