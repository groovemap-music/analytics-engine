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

import pytest
import pytest_asyncio

from insights.computations import compute_and_store_style_atlas, compute_and_store_style_year_counts
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
        await cursor.execute("TRUNCATE insights.genre_atlas_sectors, insights.style_atlas_layout, insights.style_year_counts, insights.style_parents")


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
    hosted = {(ghost.style, ghost.genre) for ghost in atlas.ghosts}
    by_style = {style.style: style for style in atlas.styles}
    assert all(any(parent.genre == genre and not parent.dominant for parent in by_style[name].parents) for name, genre in hosted)
