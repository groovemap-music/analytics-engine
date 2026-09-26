"""Shared fixtures for the embedding-pipeline PG19+pgvector integration tier.

Run via `just test-integration-pg19` / `scripts/test-integration-pg19.sh`, never directly:
that script starts the disposable PostgreSQL 19 + pgvector container and supplies every
environment variable `_required_env` below reads. See that script and `docs/embeddings.md`.

The schema applied here is a minimal, self-contained stand-in for the objects ADR 0013's
2026-09-24 amendment adds in `database-schema` — `public.artist_embeddings`, the
`embedding_pipeline` role and its grants, and the nine `graph` schema edge relations plus
`graph.vertex_degree` this pipeline reads — declared inline rather than taken as a dependency
on that repository's package. `database-schema`'s own molecule that adds these objects
(`gm-database-schema-lhp2`) has not been pushed to `origin/main` as of this bead (see the
submission notes), so a pinned `groovemap-database-schema` git dependency cannot resolve; this
suite tests this repository's own `insights.embedding_pipeline` code against the *documented*
shapes (docs/embeddings.md, and database-schema's docs/architecture.md, "Vector embeddings and
the embedding pipeline role") instead of the producer's authoritative DDL. Once that molecule
lands, `groovemap-database-schema` should become a pinned dev dependency the way
`catalog-api` already does it, and this fixture should apply its real
`create_postgres_schema` instead.

`graph.credited_on` and `graph.same_as` (ieu.6) are the two exceptions to "plain two-column
edge table": `credited_on.role_category` is a GENERATED column bound to
`graph.credit_role_category`, a SQL rendering of `common.credit_roles.ROLE_CATEGORIES` (the
same taxonomy `insights.embedding_pipeline._KEPT_CREDIT_CATEGORIES` filters against), built here
by `_credit_role_category_function_sql()` from that shared, already-vendored dependency rather
than hand-copied — so a taxonomy change in `groovemap-runtime` changes this fixture's function
body the same way it would change database-schema's real one, and this suite cannot silently
drift from what `role_category` actually resolves to.

The graph tables here are created directly, not projected from catalog documents: they are
base tables in the real schema too (loader-written, not views), so seeding them directly is
representative of what a loader's `extraction_complete` pass leaves behind.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from common import AsyncPostgreSQLPool, parse_postgres_host_port
from common.credit_roles import ROLE_CATEGORIES
from psycopg import sql


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


EMBEDDING_PIPELINE_ROLE = "embedding_pipeline"
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


def _sql_literal(value: str) -> str:
    """A single-quoted SQL string literal for embedding directly into DDL text."""
    return "'" + value.replace("'", "''") + "'"


def _credit_role_category_function_sql() -> str:
    """Return `graph.credit_role_category`, rendered from the real, vendored taxonomy.

    Verbatim shape of database-schema's `_credit_role_category_function`/`_role_category_branches`
    (docs/architecture.md, "Vector embeddings and the embedding pipeline role"), but built from
    `common.credit_roles.ROLE_CATEGORIES` here rather than hand-copied, so this fixture's
    function body tracks the shared taxonomy `insights.embedding_pipeline._KEPT_CREDIT_CATEGORIES`
    filters against exactly, including its longest-fragment-first, cross-category specificity
    rule for a compound credit like "Recorded By, Mastered By".
    """
    fragments = {role: category for category, roles in ROLE_CATEGORIES.items() for role in roles}
    branches = sorted(fragments.items(), key=lambda pair: (-len(pair[0]), pair[0]))
    exact = "\n".join(f"        WHEN normalized.role = {_sql_literal(fragment)} THEN {_sql_literal(category)}" for fragment, category in branches)
    contained = "\n".join(
        f"        WHEN strpos(normalized.role, {_sql_literal(fragment)}) > 0 THEN {_sql_literal(category)}" for fragment, category in branches
    )
    return f"""
    CREATE OR REPLACE FUNCTION graph.credit_role_category(raw_role text)
    RETURNS text
    LANGUAGE sql
    IMMUTABLE
    PARALLEL SAFE
    RETURNS NULL ON NULL INPUT
    AS $credit_role_category$
    SELECT CASE
{exact}
{contained}
        ELSE 'other'
    END
    FROM (SELECT btrim(lower(raw_role)) AS role) AS normalized
    $credit_role_category$
    """  # noqa: S608 -- built from the fixed, vendored ROLE_CATEGORIES taxonomy, never caller input.


_SCHEMA_STATEMENTS: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE SCHEMA IF NOT EXISTS graph",
    # `kind` is TEXT here rather than production's one-byte `"char"` — a fixture
    # simplification; nothing this pipeline reads depends on the storage width.
    """
    CREATE TABLE IF NOT EXISTS graph.vertex_degree (
        kind TEXT NOT NULL,
        key TEXT NOT NULL,
        degree BIGINT NOT NULL,
        PRIMARY KEY (kind, key)
    )
    """,
    *(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            {columns[0]} TEXT NOT NULL,
            {columns[1]} TEXT NOT NULL,
            PRIMARY KEY ({columns[0]}, {columns[1]})
        )
        """
        for table, columns in _EDGE_COLUMNS.items()
    ),
    _credit_role_category_function_sql(),
    # `role_category` is GENERATED, exactly like database-schema's real `credited_on` — see
    # that table's DDL comment on why: nine downstream credits functions read it there, and a
    # loader that forgot to set it would produce a silent null rather than a failure. Naming it
    # in an INSERT is therefore an error, never a value this fixture's seed rows supply.
    """
    CREATE TABLE IF NOT EXISTS graph.credited_on (
        person_name   TEXT NOT NULL,
        release_id    TEXT NOT NULL,
        role          TEXT NOT NULL,
        role_category TEXT GENERATED ALWAYS AS (graph.credit_role_category(role)) STORED,
        PRIMARY KEY (person_name, release_id, role)
    )
    """,
    # No release column: the same person credited by id on any release asserts the same row
    # (database-schema's `derive_release` docstring on `same_as`) — additive, never pruned.
    """
    CREATE TABLE IF NOT EXISTS graph.same_as (
        person_name TEXT NOT NULL,
        artist_id   TEXT NOT NULL,
        PRIMARY KEY (person_name, artist_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.artist_embeddings (
        artist_id        TEXT NOT NULL,
        model_version    TEXT NOT NULL,
        embedding        halfvec(128) NOT NULL,
        source_dump_id   TEXT NOT NULL,
        source_dump_date DATE NOT NULL,
        computed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (artist_id, model_version)
    )
    """,
    # A stand-in for a catalog document table (e.g. `public.artists`) the pipeline role is
    # never granted anything on — real database-schema tables aren't declared here (see the
    # module docstring), but the permission boundary this exercises is the same one.
    "CREATE TABLE IF NOT EXISTS public.catalog_document_sentinel (id TEXT PRIMARY KEY)",
    f"""
    DO $create_embedding_pipeline_role$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{EMBEDDING_PIPELINE_ROLE}') THEN
            CREATE ROLE {EMBEDDING_PIPELINE_ROLE} NOLOGIN;
        END IF;
    END
    $create_embedding_pipeline_role$
    """,  # noqa: S608 -- EMBEDDING_PIPELINE_ROLE is a module-level constant, not caller input.
    f"GRANT USAGE ON SCHEMA graph TO {EMBEDDING_PIPELINE_ROLE}",
    f"GRANT SELECT ON ALL TABLES IN SCHEMA graph TO {EMBEDDING_PIPELINE_ROLE}",
    f"GRANT SELECT, INSERT, UPDATE, DELETE ON public.artist_embeddings TO {EMBEDDING_PIPELINE_ROLE}",
)


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
    async with pool.connection() as conn:
        await conn.set_autocommit(True)
        async with conn.cursor() as cursor:
            for statement in _SCHEMA_STATEMENTS:
                await cursor.execute(statement)
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
