"""Promoted database-schema release helpers; edit only through producer promotion.

Producer commit 4e9720d838c7da8a6bde139c64a69d781c0f67f0. Provenance and
compatibility are checked by scripts/check-contracts.py. No schema initializer is included.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from datetime import date


logger = logging.getLogger(__name__)


def _valid_model_version(value: str) -> bool:
    """Return whether `value` is a usable `model_version` to filter or delete by.

    Unlike `index_name` and `maintenance_work_mem`, `model_version` is never interpolated
    bare -- it is always escaped as a quoted SQL string literal (`_sql_string_literal`) for
    the `WHERE` clause, or bound as an ordinary parameter for the retiring `DELETE` -- so
    this does not need a restrictive grammar. It only has to catch the caller mistake an
    empty or blank string would be: `model_version` is `NOT NULL` and never blank in
    practice (see `stored_model_version` in `analytics-engine`), so one reaching here is a
    sign of a wiring bug upstream, not a version this repository has ever written.
    """
    return value.strip() != ""


_ARTIST_EMBEDDING_RELEASES_LOCK_SQL = "LOCK TABLE public.artist_embedding_releases IN SHARE ROW EXCLUSIVE MODE"


_ARTIST_EMBEDDING_RELEASES_CLEAR_CURRENT_SQL = "UPDATE public.artist_embedding_releases SET is_current = FALSE WHERE is_current"


_ARTIST_EMBEDDING_RELEASES_CREATE_SQL = """
    INSERT INTO public.artist_embedding_releases (model_version, source_dump_id, source_dump_date, k)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (model_version) DO UPDATE SET model_version = EXCLUDED.model_version
    RETURNING release_id, source_dump_id, source_dump_date, k
"""


async def create_artist_embedding_release(cursor: Any, model_version: str, *, source_dump_id: str, source_dump_date: date, k: int) -> int | None:
    """Create a non-current batch target, returning its generated id (None on failure).

    Retries return the original id only when lineage and K match. They never alter a live
    release or reuse the same model_version for a different batch definition. Publish and
    retire retain the failure-count convention; creation returns an id, not a failure count.
    """
    if not _valid_model_version(model_version):
        logger.error("❌ Refusing to create an artist embedding release: invalid model_version %r", model_version)
        return None
    try:
        async with cursor.connection.transaction():
            await cursor.execute(_ARTIST_EMBEDDING_RELEASES_CREATE_SQL, (model_version, source_dump_id, source_dump_date, k))
            row = await cursor.fetchone()
            if row is None or tuple(row[1:]) != (source_dump_id, source_dump_date, k):
                raise ValueError("existing model_version has different lineage or k")
            release_id = int(row[0])
    except Exception as error:
        logger.error("❌ Failed to create artist embedding release %r: %s", model_version, error)
        return None
    return release_id


def _valid_artist_release_reference(reference: str | int) -> bool:
    return _valid_model_version(reference) if isinstance(reference, str) else type(reference) is int and reference > 0


async def publish_artist_embedding_release(cursor: Any, model_version: str | int, *, artists: int) -> int:
    """Atomically publish a previously created release by model_version or release_id.

    Returns a failure count. A missing target or failed update rolls the entire flip back.
    Re-publishing preserves published_at. The batch must finish writing before calling this.
    """
    if not _valid_artist_release_reference(model_version):
        logger.error("❌ Refusing to publish an artist embedding release: invalid reference %r", model_version)
        return 1
    lookup_sql = (
        "SELECT release_id, is_current FROM public.artist_embedding_releases WHERE model_version = %s FOR UPDATE"
        if isinstance(model_version, str)
        else "SELECT release_id, is_current FROM public.artist_embedding_releases WHERE release_id = %s FOR UPDATE"
    )
    try:
        async with cursor.connection.transaction():
            await cursor.execute(_ARTIST_EMBEDDING_RELEASES_LOCK_SQL)
            await cursor.execute(
                lookup_sql,
                (model_version,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise ValueError("release must be created before publication")
            release_id, was_current = row
            await cursor.execute(_ARTIST_EMBEDDING_RELEASES_CLEAR_CURRENT_SQL)
            await cursor.execute(
                "UPDATE public.artist_embedding_releases SET artists = %s, is_current = TRUE, "
                "published_at = CASE WHEN %s THEN published_at ELSE NOW() END WHERE release_id = %s",
                (artists, was_current, release_id),
            )
    except Exception as error:
        logger.error("❌ Failed to publish artist embedding release %r: %s", model_version, error)
        return 1
    return 0


_ARTIST_EMBEDDING_RELEASES_IS_CURRENT_QUERY = "SELECT is_current FROM public.artist_embedding_releases WHERE model_version = %s FOR UPDATE"


_ARTIST_SIMILAR_ARTISTS_VERSION_DELETE_SQL = (
    "DELETE FROM public.artist_similar_artists WHERE release_id = (SELECT release_id FROM public.artist_embedding_releases WHERE model_version = %s)"
)


_ARTIST_EMBEDDING_RELEASES_VERSION_DELETE_SQL = "DELETE FROM public.artist_embedding_releases WHERE model_version = %s"


async def retire_artist_similar_artists_version(cursor: Any, model_version: str, *, delete_release: bool = False) -> int:
    """Delete a superseded release's lists, optionally cascading through its release row.

    Returns a failure count; refuses the current release. Check and deletion share the same
    transaction and lock as publish so a concurrent publish cannot invalidate the guard.
    Missing releases are successful no-ops. Retained release rows preserve lineage.
    """
    if not _valid_model_version(model_version):
        logger.error("❌ Refusing to retire an artist_similar_artists version: invalid model_version %r", model_version)
        return 1
    try:
        async with cursor.connection.transaction():
            await cursor.execute(_ARTIST_EMBEDDING_RELEASES_LOCK_SQL)
            await cursor.execute(_ARTIST_EMBEDDING_RELEASES_IS_CURRENT_QUERY, (model_version,))
            row = await cursor.fetchone()
            if row is not None and row[0]:
                logger.error("❌ Refusing to retire %r: it is the current artist embedding release", model_version)
                return 1
            delete_sql = _ARTIST_EMBEDDING_RELEASES_VERSION_DELETE_SQL if delete_release else _ARTIST_SIMILAR_ARTISTS_VERSION_DELETE_SQL
            await cursor.execute(delete_sql, (model_version,))
    except Exception as error:
        logger.error("❌ Failed to retire artist_similar_artists version %r: %s", model_version, error)
        return 1
    return 0
