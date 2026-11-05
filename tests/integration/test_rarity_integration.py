"""Real-PostgreSQL regressions for the paged `insights.release_rarity` replacement.

The mock-pool unit tests cannot show what the server does with the COPY stream: text that needs
COPY escaping, NULLs, JSONB columns, and an all-or-nothing replacement when a row fails, the writer's
connection drops mid-run, or the walk ends short. The HTTP side is a fake; the database is real.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import psycopg
import pytest
import pytest_asyncio
from common import parse_postgres_host_port

from insights import computations
from insights.catalog_api_contract import RARITY_PAGE_PATH


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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
        "label_ids": [10, 2**40],
        "artist_ids": [5],
        **overrides,
    }


def _api(
    pages: list[list[dict[str, Any]]],
    *,
    release_total: int | None = None,
    before_page: Callable[[int], Awaitable[None]] | None = None,
) -> httpx.AsyncClient:
    """A client whose rarity routes serve `pages`, calling `before_page` before each scored page."""

    async def handle(request: httpx.Request) -> httpx.Response:
        index = int(request.url.params["after"] or 0)
        following = str(index + 1) if index + 1 < len(pages) else None
        if request.url.path != RARITY_PAGE_PATH:
            total = sum(len(page) for page in pages) if release_total is None else release_total
            return httpx.Response(200, json={"artist": [], "label": [], "genre": [], "releases": 0, "release_total": total, "next_after": following})
        if before_page is not None:
            await before_page(index)
        return httpx.Response(200, json={"items": pages[index], "next_after": following})

    return httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://api")


async def _store(pool: AsyncPostgreSQLPool, items: list[dict[str, Any]]) -> int:
    async with _api([items]) as client:
        return await computations.compute_and_store_rarity(client, pool)


async def _admin() -> psycopg.AsyncConnection[Any]:
    """A connection of the test's own: the pool's two are the writer's and the log's."""
    host, port = parse_postgres_host_port(os.environ["POSTGRES_HOST"])
    return await psycopg.AsyncConnection.connect(
        host=host,
        port=port,
        dbname=os.environ["POSTGRES_DATABASE"],
        user=os.environ["POSTGRES_USERNAME"],
        password=os.environ["POSTGRES_PASSWORD"],
        autocommit=True,
    )


async def _no_wait(_seconds: float) -> None:
    return None


async def _last_log(pool: AsyncPostgreSQLPool) -> tuple[Any, ...]:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "SELECT status, rows_affected, error_message FROM insights.computation_log "
            "WHERE insight_type = 'release_rarity' ORDER BY started_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
    assert row is not None
    return tuple(row)


async def _rows(pool: AsyncPostgreSQLPool) -> list[tuple[Any, ...]]:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "SELECT release_id, title, artist_name, year, rarity_score, tier, hidden_gem_score, pressing_scarcity, "
            "media_families, family_signals, medium_rarity, label_ids, artist_ids FROM insights.release_rarity ORDER BY release_id"
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
    assert rows[0] == (1, awkward, "", 1970, 12.5, "rare", 3.0, None, ["vinyl"], {"grooved": {"pressing_scarcity": 92.0}}, 7.0, [10, 2**40], [5])
    assert rows[1] == (2, "Plain", "Artist", None, 12.5, "rare", None, None, [], {}, None, [10, 2**40], [5])
    assert rows[2][1] is None  # an explicit None title stays NULL, not the empty string


async def test_label_and_artist_ids_are_stored_and_an_older_stream_still_loads(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    older = _item("3")
    del older["label_ids"], older["artist_ids"]
    items = [_item("1", label_ids=[3, 7, 9], artist_ids=[11, 12]), _item("2", label_ids=[], artist_ids=[]), older]

    assert await _store(schema_owner_pool, items) == 3

    rows = {row[0]: row[-2:] for row in await _rows(schema_owner_pool)}
    assert rows == {1: ([3, 7, 9], [11, 12]), 2: ([], []), 3: (None, None)}
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "SELECT release_id FROM insights.release_rarity WHERE label_ids @> ARRAY[7]::bigint[] AND artist_ids @> ARRAY[12]::bigint[]"
        )
        assert await cursor.fetchall() == [(1,)]


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


async def test_pages_land_in_one_replacement_and_readers_keep_the_old_table_until_the_commit(
    schema_owner_pool: AsyncPostgreSQLPool,
) -> None:
    await _store(schema_owner_pool, [_item("1"), _item("2")])
    seen: list[int] = []
    reader = await _admin()

    async def count_rows(_page: int) -> None:
        cursor = await reader.execute("SELECT count(*) FROM insights.release_rarity")
        row = await cursor.fetchone()
        assert row is not None
        seen.append(row[0])

    try:
        async with _api([[_item("10"), _item("11")], [_item("12")], [_item("13")]], before_page=count_rows) as client:
            assert await computations.compute_and_store_rarity(client, schema_owner_pool) == 4
    finally:
        await reader.close()

    assert seen == [2, 2, 2]
    assert [row[0] for row in await _rows(schema_owner_pool)] == [10, 11, 12, 13]


async def test_a_dropped_connection_mid_run_fails_loudly_and_keeps_the_previous_table(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool, [_item("1"), _item("2")])
    before = await _rows(schema_owner_pool)
    admin = await _admin()

    async def drop_the_writer(page: int) -> None:
        if page == 2:
            cursor = await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE state = 'idle in transaction' AND datname = current_database() AND pid <> pg_backend_pid()"
            )
            assert [row[0] for row in await cursor.fetchall()] == [True]

    try:
        async with _api([[_item("10")], [_item("11")], [_item("12")], [_item("13")]], before_page=drop_the_writer) as client:
            with pytest.raises(psycopg.OperationalError):
                await computations.compute_and_store_rarity(client, schema_owner_pool)
    finally:
        await admin.close()

    assert await _rows(schema_owner_pool) == before
    status, rows_affected, error = await _last_log(schema_owner_pool)
    assert (status, rows_affected) == ("failed", 0)
    assert error

    # The pool recovers, so the next cycle's run replaces the table.
    assert await _store(schema_owner_pool, [_item("20")]) == 1
    assert [row[0] for row in await _rows(schema_owner_pool)] == [20]


async def test_a_page_that_never_comes_back_keeps_the_previous_table(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool, [_item("1"), _item("2")])
    before = await _rows(schema_owner_pool)

    async def api_goes_away(page: int) -> None:
        if page == 1:
            raise httpx.ConnectError("refused")

    with patch.object(computations, "_RARITY_REQUEST_ATTEMPTS", 2), patch.object(computations.asyncio, "sleep", side_effect=_no_wait):
        async with _api([[_item("10")], [_item("11")]], before_page=api_goes_away) as client:
            with pytest.raises(httpx.ConnectError):
                await computations.compute_and_store_rarity(client, schema_owner_pool)

    assert await _rows(schema_owner_pool) == before
    assert (await _last_log(schema_owner_pool))[0] == "failed"


async def test_a_walk_short_of_the_first_release_total_keeps_the_previous_table(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool, [_item("1"), _item("2")])
    before = await _rows(schema_owner_pool)

    async with _api([[_item("10")]], release_total=2) as client:
        with pytest.raises(computations.RarityWalkShortError):
            await computations.compute_and_store_rarity(client, schema_owner_pool)

    assert await _rows(schema_owner_pool) == before
    assert (await _last_log(schema_owner_pool))[0] == "failed"


async def test_a_run_gives_up_on_a_table_lock_instead_of_waiting_behind_it(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool, [_item("1"), _item("2")])
    before = await _rows(schema_owner_pool)
    holder = await _admin()
    await holder.set_autocommit(False)
    await holder.execute("LOCK TABLE insights.release_rarity IN SHARE MODE")
    started = time.monotonic()

    try:
        with patch.object(computations, "RARITY_LOCK_TIMEOUT", "500ms"):
            async with _api([[_item("10")]]) as client:
                with pytest.raises(psycopg.errors.LockNotAvailable):
                    await computations.compute_and_store_rarity(client, schema_owner_pool)
    finally:
        await holder.rollback()
        await holder.close()

    assert time.monotonic() - started < 10.0
    assert await _rows(schema_owner_pool) == before
