"""Real-engine regressions for `insights.embedding_pipeline` (gm-analytics-engine-ieu.2).

Run via `just test-integration-pg19`. Asserts the three things a unit test cannot: writes
under the real `embedding_pipeline` role's grants land, the same (dump, model_version) pair
is a true no-op on a second run, and a write outside `public.artist_embeddings` is refused by
PostgreSQL itself rather than merely by this repository's own code.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import psycopg
import pytest

from insights.embedding_pipeline import ARTIST_EMBEDDINGS_TABLE, load_embeddings
from insights.embeddings import FastRPConfig
from tests.integration.conftest import ARTIST_IDS


if TYPE_CHECKING:
    from common import AsyncPostgreSQLPool


pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]

DUMP_ID = "fixture-dump-2026-09"
DUMP_DATE = date(2026, 9, 1)


async def _embedding_rows(pool: AsyncPostgreSQLPool, model_version: str) -> list[tuple[str, str, str]]:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            f"SELECT artist_id, source_dump_id, computed_at::text FROM {ARTIST_EMBEDDINGS_TABLE} WHERE model_version = %s ORDER BY artist_id",  # noqa: S608
            (model_version,),
        )
        return await cursor.fetchall()


async def test_load_writes_an_embedding_per_artist(pipeline_pool: AsyncPostgreSQLPool) -> None:
    config = FastRPConfig()

    result = await load_embeddings(pipeline_pool, config, DUMP_ID, DUMP_DATE)

    assert result.skipped is False
    assert result.rows_written == len(ARTIST_IDS)
    rows = await _embedding_rows(pipeline_pool, config.model_version)
    assert [artist_id for artist_id, _dump, _computed in rows] == sorted(ARTIST_IDS)
    assert all(dump_id == DUMP_ID for _artist_id, dump_id, _computed in rows)


async def test_rerun_for_the_same_dump_is_a_no_op(pipeline_pool: AsyncPostgreSQLPool) -> None:
    config = FastRPConfig()

    first = await load_embeddings(pipeline_pool, config, DUMP_ID, DUMP_DATE)
    assert first.skipped is False
    before = await _embedding_rows(pipeline_pool, config.model_version)

    second = await load_embeddings(pipeline_pool, config, DUMP_ID, DUMP_DATE)

    assert second.skipped is True
    assert second.rows_written == 0
    after = await _embedding_rows(pipeline_pool, config.model_version)
    # Same rows, untouched — in particular computed_at did not advance on the no-op run.
    assert after == before


async def test_a_new_dump_recomputes_the_same_model_version(pipeline_pool: AsyncPostgreSQLPool) -> None:
    """A different dump under the same model_version is not idempotent against it — the
    pipeline recomputes and updates the existing rows' lineage in place."""
    config = FastRPConfig()
    await load_embeddings(pipeline_pool, config, DUMP_ID, DUMP_DATE)

    next_dump_id = "fixture-dump-2026-10"
    next_dump_date = date(2026, 10, 1)
    result = await load_embeddings(pipeline_pool, config, next_dump_id, next_dump_date)

    assert result.skipped is False
    assert result.rows_written == len(ARTIST_IDS)
    rows = await _embedding_rows(pipeline_pool, config.model_version)
    assert all(dump_id == next_dump_id for _artist_id, dump_id, _computed in rows)


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
