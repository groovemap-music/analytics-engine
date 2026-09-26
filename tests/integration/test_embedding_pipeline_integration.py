"""Real-engine regressions for `insights.embedding_pipeline` (gm-analytics-engine-ieu.2, ieu.6).

Run via `just test-integration-pg19`. Asserts what a unit test cannot: writes under the real
`embedding_pipeline` role's grants land, the same stored `model_version` is a true no-op on a
second run, a second dump's load never touches an earlier dump's rows (the review fix — see
`insights.embedding_pipeline`'s module docstring, "The stored model_version is per dump, not
per algorithm"), a write outside `public.artist_embeddings` is refused by PostgreSQL itself
rather than merely by this repository's own code, and — for ieu.6 — that the real
`graph.credit_role_category`/`graph.credited_on`/`graph.same_as` join and filter behave the way
`insights.embedding_pipeline`'s unit tests can only fake: a dropped-category credit really is
excluded, an unresolvable name really joins to nothing, and an ambiguous one really fans out.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from typing import TYPE_CHECKING

import psycopg
import pytest

from insights.embedding_pipeline import ARTIST_EMBEDDINGS_TABLE, load_embeddings, stored_model_version
from insights.embeddings import FastRPConfig
from tests.integration.conftest import ARTIST_IDS


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from common import AsyncPostgreSQLPool


pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]

DUMP_ID = "fixture-dump-2026-09"
DUMP_DATE = date(2026, 9, 1)


async def _embedding_rows(pool: AsyncPostgreSQLPool, model_version: str) -> list[tuple[str, str, str, str, str]]:
    """Every column of a stored `model_version`'s rows, for byte-for-byte before/after checks."""
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "SELECT artist_id, source_dump_id, source_dump_date::text, computed_at::text, embedding::text "  # noqa: S608
            f"FROM {ARTIST_EMBEDDINGS_TABLE} WHERE model_version = %s ORDER BY artist_id",
            (model_version,),
        )
        return await cursor.fetchall()


async def test_load_writes_an_embedding_per_artist(pipeline_pool: AsyncPostgreSQLPool) -> None:
    config = FastRPConfig()

    result = await load_embeddings(pipeline_pool, config, DUMP_ID, DUMP_DATE)

    assert result.skipped is False
    assert result.method_version == config.model_version
    assert result.model_version == stored_model_version(config, DUMP_ID)
    assert result.rows_written == len(ARTIST_IDS)
    rows = await _embedding_rows(pipeline_pool, result.model_version)
    assert [row[0] for row in rows] == sorted(ARTIST_IDS)
    assert all(row[1] == DUMP_ID for row in rows)


async def test_rerun_for_the_same_dump_is_a_no_op(pipeline_pool: AsyncPostgreSQLPool) -> None:
    config = FastRPConfig()

    first = await load_embeddings(pipeline_pool, config, DUMP_ID, DUMP_DATE)
    assert first.skipped is False
    before = await _embedding_rows(pipeline_pool, first.model_version)

    second = await load_embeddings(pipeline_pool, config, DUMP_ID, DUMP_DATE)

    assert second.skipped is True
    assert second.rows_written == 0
    assert second.model_version == first.model_version
    after = await _embedding_rows(pipeline_pool, first.model_version)
    # Same rows, untouched — in particular computed_at did not advance on the no-op run.
    assert after == before


async def test_two_dumps_coexist_and_the_earlier_ones_rows_are_untouched(pipeline_pool: AsyncPostgreSQLPool) -> None:
    """The review fix: a second dump's load must land on its own stored model_version, not
    upsert the first dump's rows in place — both months' rows coexist afterward, byte for
    byte unchanged for the first, and the first dump is still independently idempotent."""
    config = FastRPConfig()

    first = await load_embeddings(pipeline_pool, config, DUMP_ID, DUMP_DATE)
    before = await _embedding_rows(pipeline_pool, first.model_version)

    next_dump_id = "fixture-dump-2026-10"
    next_dump_date = date(2026, 10, 1)
    second = await load_embeddings(pipeline_pool, config, next_dump_id, next_dump_date)

    assert second.skipped is False
    assert second.rows_written == len(ARTIST_IDS)
    assert second.model_version != first.model_version
    assert second.model_version == stored_model_version(config, next_dump_id)

    # The first dump's rows, under its own stored version, are exactly as they were.
    after_first = await _embedding_rows(pipeline_pool, first.model_version)
    assert after_first == before

    # The second dump's rows exist separately, under its own stored version.
    after_second = await _embedding_rows(pipeline_pool, second.model_version)
    assert [row[0] for row in after_second] == sorted(ARTIST_IDS)
    assert all(row[1] == next_dump_id for row in after_second)

    # Both months are on disk at once — required for ieu.3's churn measurement.
    async with pipeline_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(f"SELECT count(*) FROM {ARTIST_EMBEDDINGS_TABLE}")  # noqa: S608
        assert (await cursor.fetchone())[0] == 2 * len(ARTIST_IDS)

    # The first dump is still independently idempotent after the second dump's load.
    rerun_first = await load_embeddings(pipeline_pool, config, DUMP_ID, DUMP_DATE)
    assert rerun_first.skipped is True
    assert rerun_first.model_version == first.model_version


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param("INSERT INTO public.catalog_document_sentinel (id) VALUES ('999')", id="catalog-document-table"),
        pytest.param("UPDATE graph.vertex_degree SET degree = degree + 1 WHERE kind = 'a'", id="graph-schema-write"),
        pytest.param(f"TRUNCATE {ARTIST_EMBEDDINGS_TABLE}", id="truncate-the-one-writable-table"),
        pytest.param(f"CREATE INDEX ON {ARTIST_EMBEDDINGS_TABLE} (artist_id)", id="index-ddl-on-the-one-writable-table"),
        # PostgreSQL 15+ revokes CREATE on schema `public` from PUBLIC by default, and the
        # role is granted nothing on `public` (ADR 0013: "It holds no other privilege.") —
        # this never depends on any other schema's table list existing.
        pytest.param("CREATE TABLE public.rogue_table (id int)", id="no-ddl-anywhere"),
    ],
)
async def test_pipeline_role_cannot_write_outside_artist_embeddings(pipeline_pool: AsyncPostgreSQLPool, statement: str) -> None:
    async with pipeline_pool.connection() as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            async with conn.transaction(), conn.cursor() as cursor:
                await cursor.execute(statement)

        # The role's one grant still works on the same, now-clean, connection.
        async with conn.transaction(), conn.cursor() as cursor:
            await cursor.execute(f"SELECT count(*) FROM {ARTIST_EMBEDDINGS_TABLE}")  # noqa: S608
            assert (await cursor.fetchone())[0] == 0


# ── Release-level credited-artist edges (ieu.6) ─────────────────────────────────────────────────
#
# `graph.credited_on` and `graph.same_as` are session-scoped tables the autouse
# `_reset_artist_embeddings` fixture does not touch (it only truncates `artist_embeddings`), and
# they are read in full on every `load_embeddings` call regardless of `dump_id` — there is no
# per-dump scoping on the graph itself. A test that inserts rows into them must therefore clean
# up after itself, or it leaks state into every other test that runs `load_embeddings` in the
# same session; `_credited_rows` below is that cleanup, as an async context manager rather than
# a fixture so each test controls exactly which rows it adds.

CREDITED_DUMP_ID = "fixture-dump-2026-11-credits"
CREDITED_DUMP_DATE = date(2026, 11, 1)


@asynccontextmanager
async def _credited_rows(
    schema_owner_pool: AsyncPostgreSQLPool,
    *,
    credited_on: Iterable[tuple[str, str, str]],
    same_as: Iterable[tuple[str, str]],
) -> AsyncIterator[None]:
    """Insert `graph.credited_on`/`graph.same_as` rows for the duration of one test."""
    credited_on = list(credited_on)
    same_as = list(same_as)
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        for person_name, release_id, role in credited_on:
            await cursor.execute("INSERT INTO graph.credited_on (person_name, release_id, role) VALUES (%s, %s, %s)", (person_name, release_id, role))
        for person_name, artist_id in same_as:
            await cursor.execute("INSERT INTO graph.same_as (person_name, artist_id) VALUES (%s, %s)", (person_name, artist_id))
    try:
        yield
    finally:
        async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
            for person_name, release_id, role in credited_on:
                await cursor.execute(
                    "DELETE FROM graph.credited_on WHERE person_name = %s AND release_id = %s AND role = %s", (person_name, release_id, role)
                )
            for person_name, artist_id in same_as:
                await cursor.execute("DELETE FROM graph.same_as WHERE person_name = %s AND artist_id = %s", (person_name, artist_id))


async def _embedded_artist_ids(pool: AsyncPostgreSQLPool, model_version: str) -> set[str]:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(f"SELECT artist_id FROM {ARTIST_EMBEDDINGS_TABLE} WHERE model_version = %s", (model_version,))  # noqa: S608
        return {row[0] for row in await cursor.fetchall()}


async def test_credited_artist_edges_are_filtered_resolved_and_embedded(
    schema_owner_pool: AsyncPostgreSQLPool, pipeline_pool: AsyncPostgreSQLPool
) -> None:
    """One scenario, five assertions: the kept-category credit resolves and gets embedded; the
    dropped-category credit does not, even though its name *is* resolvable; the unresolvable
    name contributes nothing; and the ambiguous name fans out to both of its resolved artists.

    All four new artists (7, 9, 10, 11) have no `graph.vertex_degree` row — they are reachable
    only through the credited-artist edge, the case `_read_vertices`'s discovery queries exist
    for (see its docstring).
    """
    config = FastRPConfig()
    async with _credited_rows(
        schema_owner_pool,
        credited_on=[
            ("Session Player", "101", "Bass"),  # session -> kept
            ("Dropped Person", "101", "Design"),  # design -> dropped, despite resolving
            ("Unresolvable Ghost", "101", "Engineer"),  # engineering -> kept, but never resolved
            ("Ambiguous Name", "102", "Producer"),  # production -> kept, resolves to two artists
            ("Other Role Person", "101", "Liaison"),  # matches no fragment -> "other" -> kept
        ],
        same_as=[
            ("Session Player", "7"),
            ("Dropped Person", "8"),
            ("Ambiguous Name", "9"),
            ("Ambiguous Name", "10"),
            ("Other Role Person", "11"),
        ],
    ):
        result = await load_embeddings(pipeline_pool, config, CREDITED_DUMP_ID, CREDITED_DUMP_DATE)
        embedded = await _embedded_artist_ids(pipeline_pool, result.model_version)

    assert result.skipped is False
    # The three base artists, plus every kept-category credit's resolved artist(s) -- 7 (session,
    # one match), 9 and 10 (production, ambiguous -- both), 11 (other, one match). 8 is excluded
    # by the role-category filter alone: Dropped Person *does* resolve, to a real artist id, and
    # that id is never embedded because "design" is not in `_KEPT_CREDIT_CATEGORIES`.
    assert embedded == set(ARTIST_IDS) | {"7", "9", "10", "11"}
    assert "8" not in embedded  # dropped by role_category, not by an unresolvable name
    assert result.rows_written == len(ARTIST_IDS) + 4
