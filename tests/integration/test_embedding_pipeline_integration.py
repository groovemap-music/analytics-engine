"""Real-engine regressions for `insights.embedding_pipeline` (gm-analytics-engine-ieu.2).

Run via `just test-integration-pg19`. Asserts what a unit test cannot: writes under the real
`embedding_pipeline` role's grants land, the same stored `model_version` is a true no-op on a
second run, a second dump's load never touches an earlier dump's rows (the review fix — see
`insights.embedding_pipeline`'s module docstring, "The stored model_version is per dump, not
per algorithm"), and a write outside `public.artist_embeddings` is refused by PostgreSQL itself
rather than merely by this repository's own code.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import psycopg
import pytest

from insights.embedding_pipeline import ARTIST_EMBEDDINGS_TABLE, load_embeddings, stored_model_version
from insights.embeddings import FastRPConfig
from tests.integration.conftest import ARTIST_IDS


if TYPE_CHECKING:
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
