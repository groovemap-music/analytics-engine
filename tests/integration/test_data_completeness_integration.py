"""Real-PostgreSQL regressions for `insights.data_completeness.completeness_pct` being nullable.

Artists and labels report NULL once images leave the score (the database-schema change that made the
column nullable, gm-database-schema-5e8f). The job must store that NULL, and the read endpoint must
serve it as null instead of raising on `float(None)` (which 500s every entity type). The catalog-api
side is a fake HTTP payload; the database and the endpoint's SQL are real.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio


if TYPE_CHECKING:
    from common import AsyncPostgreSQLPool


pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _empty_data_completeness(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("TRUNCATE insights.data_completeness")


def _row(entity_type: str, completeness_pct: float | None) -> dict[str, object]:
    return {
        "entity_type": entity_type,
        "total_count": 1000,
        "with_image": 0,
        "with_year": 900,
        "with_country": 800,
        "with_genre": 700,
        "completeness_pct": completeness_pct,
    }


async def _stored_pct(pool: AsyncPostgreSQLPool) -> dict[str, float | None]:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("SELECT entity_type, completeness_pct FROM insights.data_completeness")
        return {entity: (float(pct) if pct is not None else None) for entity, pct in await cursor.fetchall()}


async def test_job_stores_null_and_numeric_completeness(schema_owner_pool: AsyncPostgreSQLPool) -> None:
    from insights.computations import compute_and_store_data_completeness

    payload = [_row("artists", None), _row("labels", None), _row("releases", 89.67)]
    with patch("insights.computations._fetch_from_api", AsyncMock(return_value=payload)):
        stored = await compute_and_store_data_completeness(AsyncMock(), schema_owner_pool)

    assert stored == 3
    assert await _stored_pct(schema_owner_pool) == {"artists": None, "labels": None, "releases": 89.67}


async def test_endpoint_serves_null_completeness_as_null(schema_owner_pool: AsyncPostgreSQLPool, monkeypatch: pytest.MonkeyPatch) -> None:
    import insights.insights as insights_module
    from insights.computations import compute_and_store_data_completeness

    payload = [_row("artists", None), _row("releases", 89.67)]
    with patch("insights.computations._fetch_from_api", AsyncMock(return_value=payload)):
        await compute_and_store_data_completeness(AsyncMock(), schema_owner_pool)

    monkeypatch.setattr(insights_module, "_pool", schema_owner_pool)
    monkeypatch.setattr(insights_module, "_cache", None)
    response = await insights_module.data_completeness()

    assert response.status_code == 200
    body = json.loads(response.body)
    by_type = {item["entity_type"]: item["completeness_pct"] for item in body["items"]}
    assert by_type == {"artists": None, "releases": 89.67}
    assert body["count"] == 2
