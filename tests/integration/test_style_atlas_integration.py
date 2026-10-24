"""The style_atlas computation against database-schema's real atlas tables.

The unit suite fakes the two tables. This one checks the writes against the real DDL: the
first-of-month CHECK, the primary keys, the ``node_kind`` CHECK, and NOT NULL on every column.
It also re-reads the rows through PostgreSQL to check the once-a-month no-op and the pinning
to the previous month, and reads the stored month back through the endpoint's query path.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
import pytest_asyncio

from insights.computations import compute_and_store_style_atlas, compute_and_store_style_landmarks, compute_and_store_style_year_counts
from insights.style_atlas_read import read_style_atlas
from tests.style_atlas_fixtures import next_month, synthetic_month


if TYPE_CHECKING:
    from common import AsyncPostgreSQLPool


pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]

OCTOBER = datetime(2026, 10, 23, tzinfo=UTC)
NOVEMBER = datetime(2026, 11, 2, tzinfo=UTC)


def _client(body: dict[str, Any]) -> AsyncMock:
    response = MagicMock()
    response.json = MagicMock(return_value=body)
    response.raise_for_status = MagicMock()
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    return client


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _empty_atlas(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "TRUNCATE insights.genre_atlas_sectors, insights.style_atlas_layout, insights.style_year_counts, insights.style_parents, insights.genre_year_counts, insights.style_landmark_masters"
        )


async def _rows(pool: AsyncPostgreSQLPool, month: date) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "SELECT genre, ring_order, sector_start, sector_end, x, y FROM insights.genre_atlas_sectors WHERE month = %s ORDER BY ring_order",
            (month,),
        )
        sectors = await cursor.fetchall()
        await cursor.execute(
            "SELECT node_kind, style, genre, x, y FROM insights.style_atlas_layout WHERE month = %s ORDER BY node_kind, style, genre",
            (month,),
        )
        nodes = await cursor.fetchall()
    return sectors, nodes


async def test_a_month_is_written_once_and_the_next_month_is_pinned_to_it(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    early = synthetic_month()
    late = next_month(early)

    written = await compute_and_store_style_atlas(_client(early.producer_body()), schema_owner_pool, now=OCTOBER)
    october = await _rows(schema_owner_pool, date(2026, 10, 1))
    again = await compute_and_store_style_atlas(_client(late.producer_body()), schema_owner_pool, now=OCTOBER)
    november_written = await compute_and_store_style_atlas(_client(late.producer_body()), schema_owner_pool, now=NOVEMBER)
    november = await _rows(schema_owner_pool, date(2026, 11, 1))

    assert written == len(october[0]) + len(october[1]) > 16
    assert again == 0
    assert await _rows(schema_owner_pool, date(2026, 10, 1)) == october
    assert november_written == len(november[0]) + len(november[1])
    assert [row[:4] for row in november[0]] == [row[:4] for row in october[0]]
    assert {row[0] for row in november[1]} == {"style", "ghost"}


async def test_the_stored_month_reads_back_as_an_atlas_with_derived_lifecycles(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        assert (await read_style_atlas(cursor)).month is None

    month = synthetic_month()
    client = _client(month.producer_body())
    await compute_and_store_style_year_counts(client, schema_owner_pool)
    await compute_and_store_style_atlas(client, schema_owner_pool, now=OCTOBER)

    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        atlas = await read_style_atlas(cursor)

    assert atlas.month == date(2026, 10, 1)
    assert atlas.timeline is not None
    assert (atlas.timeline.first_year, atlas.timeline.last_year) == (1950, 2025)
    assert [genre.ring_order for genre in atlas.genres] == list(range(len(atlas.genres)))
    assert len(atlas.styles) == len({style.style for style in atlas.styles}) > 100
    assert atlas.ghosts
    years = atlas.timeline.last_year - atlas.timeline.first_year + 1
    for style in atlas.styles:
        assert len(style.stages) == years
        assert [parent.genre for parent in style.parents if parent.dominant] == [style.genre]
    genre_emergence = {genre.genre: genre.emergence_year for genre in atlas.genres}
    assert any(year is not None for year in genre_emergence.values())
    for style in atlas.styles:
        for parent in style.parents:
            genre_year = genre_emergence[parent.genre]
            assert parent.emergence_edge == (style.emergence_year is not None and genre_year is not None and style.emergence_year > genre_year)
    assert any(parent.emergence_edge for style in atlas.styles for parent in style.parents)
    hosted = {(ghost.style, ghost.genre) for ghost in atlas.ghosts}
    by_style = {style.style: style for style in atlas.styles}
    assert all(any(parent.genre == genre and not parent.dominant for parent in by_style[name].parents) for name, genre in hosted)


def _landmark(style: str, master_id: str, title: str | None, year: int | None, release_count: int) -> dict[str, Any]:
    return {"style": style, "master_id": master_id, "title": title, "year": year, "release_count": release_count}


async def _landmark_rows(pool: AsyncPostgreSQLPool) -> list[tuple[Any, ...]]:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("SELECT style, master_id, title, year, release_count, rank FROM insights.style_landmark_masters ORDER BY style, rank")
        return list(await cursor.fetchall())


async def test_landmarks_are_ranked_replaced_and_kept_on_an_empty_result(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    first = {
        "landmarks": [
            _landmark("House", "m1", "Strictly Rhythm", 1991, 30),
            _landmark("House", "m2", None, None, 20),
            _landmark("Dub", "m9", "King Tubby", 1976, 12),
        ]
    }

    written = await compute_and_store_style_landmarks(_client(first), schema_owner_pool)

    assert written == 3
    assert await _landmark_rows(schema_owner_pool) == [
        ("Dub", "m9", "King Tubby", 1976, 12, 1),
        ("House", "m1", "Strictly Rhythm", 1991, 30, 1),
        ("House", "m2", "", None, 20, 2),
    ]

    second = {"landmarks": [_landmark("House", "m7", "Acid Tracks", 1987, 99)]}
    assert await compute_and_store_style_landmarks(_client(second), schema_owner_pool) == 1
    assert await _landmark_rows(schema_owner_pool) == [("House", "m7", "Acid Tracks", 1987, 99, 1)]

    assert await compute_and_store_style_landmarks(_client({"landmarks": []}), schema_owner_pool) == 0
    assert await _landmark_rows(schema_owner_pool) == [("House", "m7", "Acid Tracks", 1987, 99, 1)]


async def test_a_full_style_of_twenty_five_is_accepted_and_a_longer_one_is_cut_to_the_rank_check(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    body = {"landmarks": [_landmark("House", f"m{i:02d}", f"Master {i}", 1990, 100 - i) for i in range(27)]}

    written = await compute_and_store_style_landmarks(_client(body), schema_owner_pool)

    rows = await _landmark_rows(schema_owner_pool)
    assert written == 25
    assert [row[5] for row in rows] == list(range(1, 26))


async def test_a_failed_write_rolls_back_to_the_previous_snapshot(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await compute_and_store_style_landmarks(_client({"landmarks": [_landmark("House", "m1", "Keep Me", 1991, 30)]}), schema_owner_pool)
    duplicate_master = {"landmarks": [_landmark("Dub", "m9", "Twice", 1976, 12), _landmark("Dub", "m9", "Twice", 1976, 12)]}

    with pytest.raises(psycopg.errors.UniqueViolation):
        await compute_and_store_style_landmarks(_client(duplicate_master), schema_owner_pool)

    assert await _landmark_rows(schema_owner_pool) == [("House", "m1", "Keep Me", 1991, 30, 1)]
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "SELECT status FROM insights.computation_log WHERE insight_type = 'style_landmarks' ORDER BY started_at DESC, id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "failed"


async def test_the_atlas_serves_masters_for_the_styles_in_the_layout_only(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    month = synthetic_month()
    client = _client(month.producer_body())
    await compute_and_store_style_year_counts(client, schema_owner_pool)
    await compute_and_store_style_atlas(client, schema_owner_pool, now=OCTOBER)
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        placed = sorted((await read_style_atlas(cursor)).styles, key=lambda style: style.style)
    first, second = placed[0].style, placed[1].style
    landmarks = {
        "landmarks": [
            _landmark(second, "m3", "Third Title", 2001, 5),
            _landmark(first, "m2", None, None, 8),
            _landmark(first, "m1", "First Title", 1999, 9),
            _landmark("Not In The Layout", "m4", "Elsewhere", 1980, 70),
        ]
    }
    await compute_and_store_style_landmarks(_client(landmarks), schema_owner_pool)

    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        atlas = await read_style_atlas(cursor)

    assert [(master.style, master.rank, master.master_id, master.title, master.year, master.release_count) for master in atlas.masters] == [
        (first, 1, "m2", "", None, 8),
        (first, 2, "m1", "First Title", 1999, 9),
        (second, 1, "m3", "Third Title", 2001, 5),
    ]


async def test_an_atlas_with_a_layout_and_no_landmarks_has_no_masters(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    client = _client(synthetic_month().producer_body())
    await compute_and_store_style_year_counts(client, schema_owner_pool)
    await compute_and_store_style_atlas(client, schema_owner_pool, now=OCTOBER)

    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        atlas = await read_style_atlas(cursor)

    assert atlas.styles
    assert atlas.masters == []
