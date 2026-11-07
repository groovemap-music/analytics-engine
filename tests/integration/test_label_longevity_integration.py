"""Real-PostgreSQL regressions for label longevity's placeholder labels.

The mock-pool unit tests cannot show what the real `insights.label_longevity` table does with
a placeholder row: its `rank` primary key, the `is_placeholder` column the pinned database-schema
revision adds, or what the read endpoint's SQL returns. The HTTP side is a fake catalog-api that
answers like contract 1.10.0; the database, the writer and the read endpoint are real.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio

from insights import computations, insights
from insights.catalog_api_contract import LABEL_LONGEVITY_PATH


if TYPE_CHECKING:
    from common import AsyncPostgreSQLPool


pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _empty_label_longevity(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("TRUNCATE insights.label_longevity")


def _label(label_id: str, name: str, years_active: int, releases: int, *, placeholder: bool = False) -> dict[str, Any]:
    return {
        "label_id": label_id,
        "label_name": name,
        "first_year": 2025 - years_active,
        "last_year": 2025,
        "years_active": years_active,
        "total_releases": releases,
        "peak_decade": 1970,
        "is_placeholder": placeholder,
    }


# What catalog-api returns for limit=3&include_placeholders=true: the top three real labels
# with the placeholders that rank among them, in years-active order.
_API_ROWS = [
    _label("1", "Blue Note", 90, 4500),
    _label("2", "Not On Label", 80, 900_000, placeholder=True),
    _label("3", "Impulse!", 70, 800),
    _label("4", "Verve", 60, 700),
]


def _api(calls: list[dict[str, str]]) -> httpx.AsyncClient:
    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == LABEL_LONGEVITY_PATH
        calls.append(dict(request.url.params))
        return httpx.Response(200, json={"items": _API_ROWS, "count": len(_API_ROWS)})

    return httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://api")


class _DictCache:
    """The `InsightsCache` surface the endpoint uses, backed by a dict."""

    def __init__(self) -> None:
        self.entries: dict[str, Any] = {}

    async def generation(self) -> int:
        return 1

    async def get(self, key: str, _generation: int) -> Any:
        return self.entries.get(key)

    async def set(self, key: str, value: Any, _generation: int) -> None:
        self.entries[key] = value


async def _store(pool: AsyncPostgreSQLPool) -> tuple[int, list[dict[str, str]]]:
    calls: list[dict[str, str]] = []
    async with _api(calls) as client:
        return await computations.compute_and_store_label_longevity(client, pool, limit=3), calls


async def _read(pool: AsyncPostgreSQLPool, query: str = "", cache: _DictCache | None = None, *, limit: int = 3) -> list[dict[str, Any]]:
    include = "include_placeholders=true" in query
    with patch.object(insights, "_pool", pool), patch.object(insights, "_cache", cache):
        response = await insights.label_longevity(limit=limit, include_placeholders=include)
    assert response.status_code == 200
    body = json.loads(bytes(response.body))
    assert body["count"] == len(body["items"])
    return list(body["items"])


async def test_run_stores_placeholders_after_the_real_labels_and_logs_completed(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    stored, calls = await _store(schema_owner_pool)

    assert stored == 4
    assert calls == [{"limit": "3", "include_placeholders": "true"}]
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("SELECT rank, label_id, is_placeholder FROM insights.label_longevity ORDER BY rank")
        assert [tuple(r) for r in await cursor.fetchall()] == [
            (1, "1", False),
            (2, "3", False),
            (3, "4", False),
            (4, "2", True),
        ]
        await cursor.execute("SELECT status FROM insights.computation_log WHERE insight_type = 'label_longevity' ORDER BY started_at DESC LIMIT 1")
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "completed"


async def test_default_read_has_no_placeholder_and_real_ranks_are_consecutive(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool)

    items = await _read(schema_owner_pool)

    assert [(i["rank"], i["label_id"], i["is_placeholder"]) for i in items] == [(1, "1", False), (2, "3", False), (3, "4", False)]


async def test_include_placeholders_lists_not_on_label_where_it_ranks(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool)

    items = await _read(schema_owner_pool, "include_placeholders=true")

    # Not On Label sits second, between Blue Note and Impulse!, yet keeps its stored rank (4) and takes
    # no place from a real label: the real labels' ranks are the same as in the default read.
    assert [(i["label_id"], i["rank"], i["is_placeholder"]) for i in items] == [("1", 1, False), ("2", 4, True), ("3", 2, False), ("4", 3, False)]
    assert items[1]["label_name"] == "Not On Label"


async def test_a_smaller_limit_keeps_only_the_placeholders_that_rank_among_the_real_labels(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool)

    two = await _read(schema_owner_pool, "include_placeholders=true", limit=2)
    one = await _read(schema_owner_pool, "include_placeholders=true", limit=1)

    assert [i["label_id"] for i in two] == ["1", "2", "3"]
    assert [i["label_id"] for i in one] == ["1"]


async def test_the_two_cached_responses_do_not_mix(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool)
    cache = _DictCache()

    with_placeholders = await _read(schema_owner_pool, "include_placeholders=true", cache)
    default = await _read(schema_owner_pool, "", cache)
    # Served from the cache now; each key still returns its own response.
    assert await _read(schema_owner_pool, "include_placeholders=true", cache) == with_placeholders
    assert await _read(schema_owner_pool, "", cache) == default

    assert len(cache.entries) == 2
    assert any(i["is_placeholder"] for i in with_placeholders)
    assert not any(i["is_placeholder"] for i in default)


async def test_a_rerun_replaces_the_placeholder_rows(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    await _store(schema_owner_pool)
    await _store(schema_owner_pool)

    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("SELECT count(*) FILTER (WHERE is_placeholder), count(*) FROM insights.label_longevity")
        row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == (1, 4)
