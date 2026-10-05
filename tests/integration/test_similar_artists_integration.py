"""Synthetic compact-COPY and publication regressions under the real PG19 role."""

from datetime import date
from typing import Any

import numpy as np
import psycopg
import pytest

from insights import similar_artists as sa


pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]


async def _rows(pool: Any, sql: str, params: Any = None) -> list[tuple[Any, ...]]:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(sql, params)
        return await cursor.fetchall()


async def _write(pool: Any, spool: sa.Spool, ids: list[str], version: str) -> int:
    async with pool.connection() as conn:
        return await sa.write_similar_artists(
            conn, spool, model_version=version, artist_ids=ids, source_dump_id=version, source_dump_date=date(2026, 9, 1)
        )


async def _publish(pool: Any, version: str, artists: int = 4) -> sa.RotateResult:
    async with pool.connection() as conn:
        return await sa.publish_and_rotate(conn, sa.SchemaReleaseRegistry(), model_version=version, artists=artists)


async def test_compact_copy_preserves_order_quoting_and_current_previous(pipeline_pool: Any, tmp_path: Any) -> None:
    vectors = np.eye(4, dtype=np.float16)
    ids = ["comma,brace{", 'quote"slash\\', "tab\tline\n", "plain"]
    spool = sa.compute_to_spool(vectors, tmp_path, model_version="synthetic", k=3, block_rows=64, threads=1)
    for version in ("fixture-release-1", "fixture-release-2", "fixture-release-3"):
        assert await _write(pipeline_pool, spool, ids, version) == 4
        rotation = await _publish(pipeline_pool, version)
        if version == "fixture-release-2":
            await _write(pipeline_pool, spool, ids, "fixture-future-pending")
    assert rotation.kept_previous == "fixture-release-2"
    assert rotation.retired == ("fixture-release-1",)
    rows = await _rows(
        pipeline_pool,
        "SELECT r.model_version, a.artist_id, a.similar_artist_ids, a.scores FROM public.artist_similar_artists a JOIN public.artist_embedding_releases r USING (release_id) WHERE r.artists > 0",
    )
    assert len(rows) == 8
    for version, artist_id, neighbours, scores in rows:
        assert version in {"fixture-release-2", "fixture-release-3"}
        assert neighbours == [other for other in ids if other != artist_id]
        assert scores == [0.0] * 3
    assert await _rows(pipeline_pool, "SELECT model_version FROM public.artist_embedding_releases WHERE is_current") == [("fixture-release-3",)]
    # Republish the retained release, then retain the one it actually replaces.
    assert (await _publish(pipeline_pool, "fixture-release-2")).kept_previous == "fixture-release-3"
    assert (
        len(
            await _rows(
                pipeline_pool,
                "SELECT a.artist_id FROM public.artist_similar_artists a JOIN public.artist_embedding_releases r USING (release_id) WHERE r.artists > 0",
            )
        )
        == 8
    )


async def test_failed_copy_rolls_back_deleted_rows_and_failed_publish_preserves_current(pipeline_pool: Any, tmp_path: Any) -> None:
    spool = sa.compute_to_spool(np.eye(4, dtype=np.float16), tmp_path, model_version="synthetic", k=3, block_rows=64, threads=1)
    ids = ["1", "2", "3", "4"]
    version = "fixture-pending"
    await _write(pipeline_pool, spool, ids, version)
    before = await _rows(pipeline_pool, "SELECT artist_id, similar_artist_ids, scores FROM public.artist_similar_artists ORDER BY artist_id")
    # A duplicate artist id fails on the second COPY row after DELETE and one insert.
    with pytest.raises(psycopg.errors.UniqueViolation):
        await _write(pipeline_pool, spool, ["1", "1", "3", "4"], version)
    assert await _rows(pipeline_pool, "SELECT artist_id, similar_artist_ids, scores FROM public.artist_similar_artists ORDER BY artist_id") == before
    await _publish(pipeline_pool, version)
    with pytest.raises(sa.PublishError, match="overwrite"):
        await _write(pipeline_pool, spool, ids, version)
    assert await _rows(pipeline_pool, "SELECT artist_id, similar_artist_ids, scores FROM public.artist_similar_artists ORDER BY artist_id") == before
    with pytest.raises(sa.PublishError):
        await _publish(pipeline_pool, version, artists=-1)
    assert await _rows(pipeline_pool, "SELECT model_version FROM public.artist_embedding_releases WHERE is_current") == [(version,)]
    # Pending targets must not cause the monthly stage to skip computation/publication.
    async with pipeline_pool.connection() as conn, conn.cursor() as cursor:
        assert (
            await sa.SchemaReleaseRegistry().create(cursor, "fixture-pending-next", source_dump_id="next", source_dump_date=date(2026, 10, 1), k=3)
            is not None
        )
        await cursor.execute(sa._RELEASE_EXISTS_SQL, ("fixture-pending-next",))
        assert await cursor.fetchone() is None
