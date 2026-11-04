"""Real-PostgreSQL regressions for the batched `insights.release_rarity` replacement (gm-analytics-engine-v9ak).

The mock-pool unit tests cannot show what the server does with the COPY stream: text that needs
COPY escaping, NULLs, JSONB columns, and an all-or-nothing replacement when a row fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import psycopg
import pytest
import pytest_asyncio

from insights import computations


if TYPE_CHECKING:
    from common import AsyncPostgreSQLPool


pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _rarity_table(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    """Make `release_rarity` the shape `compute_and_store_rarity` writes, and empty it.

    The pinned database-schema revision predates the three family columns the code writes, so
    they are added here when missing; on a newer revision this is a no-op.
    """
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        for column, column_type in (("media_families", "JSONB"), ("family_signals", "JSONB"), ("medium_rarity", "REAL")):
            await cursor.execute(f"ALTER TABLE insights.release_rarity ADD COLUMN IF NOT EXISTS {column} {column_type}")
        await cursor.execute("TRUNCATE insights.release_rarity")


def _item(release_id: str, **overrides: Any) -> dict[str, Any]:
    return {
        "release_id": release_id,
        "title": "Plain",
        "artist_name": "Artist",
        "year": 1970,
        "rarity_score": 12.5,
        "tier": "rare",
        "hidden_gem_score": 3.0,
        "media_families": ["vinyl"],
        "family_signals": {"grooved": {"pressing_scarcity": 92.0}},
        "medium_rarity": 7.0,
        **overrides,
    }


async def _store(pool: AsyncPostgreSQLPool, items: list[dict[str, Any]]) -> int:
    async def fake_fetch(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return items

    with patch.object(computations, "_fetch_from_api", fake_fetch):
        return await computations.compute_and_store_rarity(None, pool)  # type: ignore[arg-type]


async def _rows(pool: AsyncPostgreSQLPool) -> list[tuple[Any, ...]]:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "SELECT release_id, title, artist_name, year, rarity_score, tier, hidden_gem_score, pressing_scarcity, "
            "media_families, family_signals, medium_rarity FROM insights.release_rarity ORDER BY release_id"
        )
        return await cursor.fetchall()


async def test_copy_round_trips_escaping_nulls_and_jsonb(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    awkward = 'tab\t newline\n back\\slash "quote" {brace} \\N 日本語'
    items = [
        _item("1", title=awkward, artist_name=""),
        _item("2", year=None, hidden_gem_score=None, media_families=[], family_signals={}, medium_rarity=None),
        _item("3", title=None),
    ]
    items[2].pop("artist_name")

    assert await _store(schema_owner_pool, items) == 3

    rows = await _rows(schema_owner_pool)
    assert rows[0] == (1, awkward, "", 1970, 12.5, "rare", 3.0, None, ["vinyl"], {"grooved": {"pressing_scarcity": 92.0}}, 7.0)
    assert rows[1] == (2, "Plain", "Artist", None, 12.5, "rare", None, None, [], {}, None)
    assert rows[2][1] is None  # an explicit None title stays NULL, not the empty string


async def test_a_second_run_replaces_the_first_runs_rows(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool, [_item("1"), _item("2")])

    await _store(schema_owner_pool, [_item("2", title="Newer"), _item("3")])

    assert [(row[0], row[1]) for row in await _rows(schema_owner_pool)] == [(2, "Newer"), (3, "Plain")]


async def test_a_failing_row_rolls_back_the_replacement_and_keeps_the_previous_table(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool, [_item("1"), _item("2")])
    before = await _rows(schema_owner_pool)

    with pytest.raises(psycopg.errors.UniqueViolation):
        await _store(schema_owner_pool, [_item("5"), _item("6"), _item("5")])

    assert await _rows(schema_owner_pool) == before


async def test_stores_many_rows_in_one_batch(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    items = [_item(str(index), rarity_score=float(index % 100)) for index in range(20_000)]

    assert await _store(schema_owner_pool, items) == 20_000

    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("SELECT count(*), max(release_id) FROM insights.release_rarity")
        assert await cursor.fetchone() == (20_000, 19_999)
