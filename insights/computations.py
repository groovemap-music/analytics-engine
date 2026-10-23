"""Fetch, transform, and persist scheduled insight computations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, Final, ParamSpec, cast

import httpx
import structlog
from common import describe_exception
from psycopg.types.json import Jsonb

from insights.activity import PRODUCT_ANALYTICS, read_events, read_impressions
from insights.catalog_api_contract import (
    ANNIVERSARIES_PATH,
    ARTIST_CENTRALITY_PATH,
    COMMUNITY_ENRICHMENT_MAX_PROCESSING_SECONDS,
    COMMUNITY_ENRICHMENT_PATH,
    DATA_COMPLETENESS_PATH,
    GENRE_TRENDS_PATH,
    LABEL_LONGEVITY_PATH,
    RARITY_SCORES_PATH,
    STYLE_YEAR_COUNTS_MAX_PROCESSING_SECONDS,
    STYLE_YEAR_COUNTS_PATH,
)
from insights.style_atlas import Sector, build_layout, hull_check, movement, style_positions
from insights.style_lifecycle import last_complete_year, style_lifecycles
from insights.telemetry import computation_span, record_computation


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterable
    from datetime import date
    from uuid import UUID


logger = structlog.get_logger(__name__)
P = ParamSpec("P")


# ── Per-endpoint HTTP timeouts ──────────────────────────────────────────────
#
# Every /api/internal/insights/* endpoint runs an uncached full-scan computation
# on a cold cache, so a single scalar timeout for the whole client is wrong in
# both directions: passing ``timeout=300.0`` also gives *connect* 300 seconds
# (a dead API should fail in seconds), while 300s of *read* is far under the
# documented worst-case latency of the heavy endpoints. Production saw
# data_completeness fail once with ReadTimeout and community_enrichment fail
# four times for the same reason.
#
# Split the budget: a short connect/write/pool timeout, and a per-endpoint read
# timeout with generous headroom over the worst observed cold-cache latency.
#
#   endpoint              worst observed cold-cache cost          read budget
#   --------------------  --------------------------------------  -----------
#   data-completeness     ~400s releases seq scan (>600s on bad         1800s
#                         days); API caches the result for 6h
#   rarity-scores         chunked full-graph Neo4j scans                1800s
#   community-enrichment  API-published bounded processing time plus    1800s
#                         20% transport/storage headroom
#   style-year-counts     API-published bounded processing time x 3      900s
#                         (a scan of every release-style tag)
#   everything else       full-graph Neo4j aggregations                  900s
_CONNECT_TIMEOUT_SECONDS = 10.0
_WRITE_TIMEOUT_SECONDS = 30.0
_POOL_TIMEOUT_SECONDS = 60.0
DEFAULT_READ_TIMEOUT_SECONDS = 900.0

# A genre is a style's parent when it covers at least this share of the style's releases.
STYLE_PARENT_MIN_SHARE: Final = 0.20

ENDPOINT_READ_TIMEOUTS: dict[str, float] = {
    DATA_COMPLETENESS_PATH: 1800.0,
    RARITY_SCORES_PATH: 1800.0,
    COMMUNITY_ENRICHMENT_PATH: COMMUNITY_ENRICHMENT_MAX_PROCESSING_SECONDS * 1.2,
    STYLE_YEAR_COUNTS_PATH: STYLE_YEAR_COUNTS_MAX_PROCESSING_SECONDS * 3.0,
}


def endpoint_timeout(path: str | None = None) -> httpx.Timeout:
    """Return the ``httpx.Timeout`` to use for an internal computation endpoint.

    Args:
        path: The API path being requested. ``None`` yields the default budget,
            which is what the shared client is constructed with.

    Returns:
        A timeout with a short connect/write/pool budget and a read budget sized
        for that endpoint's worst-case cold-cache latency.
    """
    read = ENDPOINT_READ_TIMEOUTS.get(path or "", DEFAULT_READ_TIMEOUT_SECONDS)
    return httpx.Timeout(
        connect=_CONNECT_TIMEOUT_SECONDS,
        read=read,
        write=_WRITE_TIMEOUT_SECONDS,
        pool=_POOL_TIMEOUT_SECONDS,
    )


async def _log_computation(
    pool: Any,
    insight_type: str,
    status: str,
    started_at: datetime,
    rows_affected: int = 0,
    error_message: str | None = None,
) -> None:
    """Write a computation log entry."""
    completed_at = datetime.now(UTC)
    duration_ms = int((completed_at - started_at).total_seconds() * 1000)
    async with pool.connection() as conn, conn.cursor() as cursor:
        cursor = cast("Any", cursor)
        await cursor.execute(
            """
            INSERT INTO insights.computation_log
                (insight_type, status, started_at, completed_at, rows_affected, duration_ms, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (insight_type, status, started_at, completed_at, rows_affected, duration_ms, error_message),
        )


def _record_lifecycle(
    insight_type: str,
    failure_message: str,
) -> Callable[
    [Callable[Concatenate[httpx.AsyncClient, Any, P], Coroutine[Any, Any, int]]],
    Callable[Concatenate[httpx.AsyncClient, Any, P], Coroutine[Any, Any, int]],
]:
    """Keep persistence and error reporting consistent across computations."""

    def decorate(
        operation: Callable[Concatenate[httpx.AsyncClient, Any, P], Coroutine[Any, Any, int]],
    ) -> Callable[Concatenate[httpx.AsyncClient, Any, P], Coroutine[Any, Any, int]]:
        @wraps(operation)
        async def wrapped(client: httpx.AsyncClient, pool: Any, *args: P.args, **kwargs: P.kwargs) -> int:
            started_at = datetime.now(UTC)
            try:
                rows_affected = await operation(client, pool, *args, **kwargs)
                await _log_computation(pool, insight_type, "completed", started_at, rows_affected)
                return rows_affected
            except Exception as error:
                description = describe_exception(error)
                logger.error(failure_message, error=description)
                try:
                    await _log_computation(pool, insight_type, "failed", started_at, error_message=description)
                except Exception as log_error:
                    logger.warning("⚠️ Failed to log computation error", error=describe_exception(log_error))
                raise

        return wrapped

    return decorate


async def _get_json(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, Any] | None = None,
    timeout: httpx.Timeout | float | None = None,
) -> dict[str, Any]:
    """GET a computation endpoint and return its JSON body.

    Unless ``timeout`` is given explicitly, the per-endpoint budget from
    :func:`endpoint_timeout` is applied — never the client's scalar default.
    """
    kwargs: dict[str, Any] = {}
    if params:
        kwargs["params"] = params
    kwargs["timeout"] = endpoint_timeout(path) if timeout is None else timeout
    response = await client.get(path, **kwargs)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


async def _fetch_from_api(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, Any] | None = None,
    timeout: httpx.Timeout | float | None = None,
) -> list[dict[str, Any]]:
    """Fetch the ``items`` list of a computation endpoint from the API service."""
    data = await _get_json(client, path, params, timeout)
    items: list[dict[str, Any]] = data.get("items", [])
    return items


@_record_lifecycle("artist_centrality", "❌ Artist centrality computation failed")
async def compute_and_store_artist_centrality(client: httpx.AsyncClient, pool: Any, limit: int = 100) -> int:
    """Compute artist centrality and store results."""
    results = await _fetch_from_api(client, ARTIST_CENTRALITY_PATH, {"limit": limit})
    if not results:
        logger.info("📊 No artist centrality results to store")
        return 0

    results = [row for row in results if row.get("artist_name")]
    if not results:
        logger.info("📊 No artist centrality results with valid names")
        return 0

    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            cursor = cast("Any", cursor)
            await cursor.execute("DELETE FROM insights.artist_centrality")
            for rank, row in enumerate(results, 1):
                await cursor.execute(
                    """
                        INSERT INTO insights.artist_centrality (rank, artist_id, artist_name, edge_count)
                        VALUES (%s, %s, %s, %s)
                        """,
                    (rank, row["artist_id"], row["artist_name"], row["edge_count"]),
                )
    logger.info("💾 Artist centrality stored", count=len(results))
    return len(results)


@_record_lifecycle("genre_trends", "❌ Genre trends computation failed")
async def compute_and_store_genre_trends(client: httpx.AsyncClient, pool: Any) -> int:
    """Compute genre trends and store results."""
    results = await _fetch_from_api(client, GENRE_TRENDS_PATH)
    if not results:
        return 0

    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            cursor = cast("Any", cursor)
            await cursor.execute("DELETE FROM insights.genre_trends")
            for row in results:
                await cursor.execute(
                    """
                        INSERT INTO insights.genre_trends (genre, decade, release_count)
                        VALUES (%s, %s, %s)
                        """,
                    (row["genre"], row["decade"], row["release_count"]),
                )
    logger.info("💾 Genre trends stored", count=len(results))
    return len(results)


def _style_parent_rows(shares: Iterable[dict[str, Any]]) -> list[tuple[str, str, float, bool]]:
    """Reduce raw producer shares to the parents worth keeping, one of them dominant.

    The producer publishes every genre that co-occurs with a style and applies no floor; the
    parent threshold is this consumer's. A style keeps each genre at ``share >=
    STYLE_PARENT_MIN_SHARE`` and marks exactly one of those dominant: the highest share, with
    ties broken by genre name so a re-run over the same data picks the same parent. A style
    whose every share is under the threshold has no parent and writes no row.
    """
    by_style: dict[str, list[tuple[str, float]]] = {}
    for row in shares:
        share = float(row["share"])
        if share >= STYLE_PARENT_MIN_SHARE:
            by_style.setdefault(row["style"], []).append((row["genre"], share))

    rows: list[tuple[str, str, float, bool]] = []
    for style, parents in sorted(by_style.items()):
        dominant_genre = min(parents, key=lambda parent: (-parent[1], parent[0]))[0]
        rows.extend((style, genre, share, genre == dominant_genre) for genre, share in sorted(parents))
    return rows


@_record_lifecycle("style_year_counts", "❌ Style year counts computation failed")
async def compute_and_store_style_year_counts(client: httpx.AsyncClient, pool: Any) -> int:
    """Compute per-style release counts by year and each style's parent genres, and store them.

    One producer read feeds both ``insights.style_year_counts`` and ``insights.style_parents``,
    replaced together in a single transaction so a reader never sees counts from one snapshot
    beside parents from another.

    A producer result missing either half is treated as empty and leaves the previous snapshot
    intact: with real catalog data every style that has dated releases also has genre
    co-occurrences, so a lone half is an incomplete read, and writing it would wipe the other
    table's good rows.

    Returns:
        The number of rows written across both tables.
    """
    data = await _get_json(client, STYLE_YEAR_COUNTS_PATH)
    style_years: list[dict[str, Any]] = data.get("style_years", [])
    style_parents: list[dict[str, Any]] = data.get("style_parents", [])
    if not style_years or not style_parents:
        if style_years or style_parents:
            logger.warning(
                "⚠️ Style year counts result is missing one half — keeping the previous snapshot",
                style_years=len(style_years),
                style_parents=len(style_parents),
            )
        return 0

    year_rows = [(row["style"], row["year"], row["release_count"], row["master_count"]) for row in style_years]
    parent_rows = _style_parent_rows(style_parents)

    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            cursor = cast("Any", cursor)
            await cursor.execute("DELETE FROM insights.style_year_counts")
            await cursor.execute("DELETE FROM insights.style_parents")
            await cursor.executemany(
                """
                INSERT INTO insights.style_year_counts (style, year, release_count, master_count)
                VALUES (%s, %s, %s, %s)
                """,
                year_rows,
            )
            if parent_rows:
                await cursor.executemany(
                    """
                    INSERT INTO insights.style_parents (style, genre, share, is_dominant)
                    VALUES (%s, %s, %s, %s)
                    """,
                    parent_rows,
                )
    logger.info("💾 Style year counts stored", style_years=len(year_rows), style_parents=len(parent_rows))
    return len(year_rows) + len(parent_rows)


# One writer per month: the lock serialises concurrent runs, and the in-transaction re-check
# turns the loser into a no-op instead of a second, interleaved layout.
_STYLE_ATLAS_LOCK_SQL: Final = "SELECT pg_advisory_xact_lock(hashtext('insights.style_atlas'))"
_STYLE_ATLAS_MONTH_EXISTS_SQL: Final = "SELECT 1 FROM insights.genre_atlas_sectors WHERE month = %s LIMIT 1"
_STYLE_ATLAS_PINNED_SECTORS_SQL: Final = """
    SELECT month, genre, ring_order, sector_start, sector_end, x, y
    FROM insights.genre_atlas_sectors
    WHERE month = (SELECT max(month) FROM insights.genre_atlas_sectors WHERE month < %s)
    ORDER BY ring_order
"""
_STYLE_ATLAS_PREVIOUS_STYLES_SQL: Final = "SELECT style, x, y FROM insights.style_atlas_layout WHERE month = %s AND node_kind = 'style'"


def atlas_month(now: datetime | None = None) -> date:
    """The first day of the UTC month a run at ``now`` lays out."""
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return moment.date().replace(day=1)


@_record_lifecycle("style_atlas", "❌ Style atlas computation failed")
async def compute_and_store_style_atlas(client: httpx.AsyncClient, pool: Any, now: datetime | None = None) -> int:
    """Lay out the month's Genre Atlas and store its sectors and nodes, once per month.

    A month that already has ``insights.genre_atlas_sectors`` rows is a no-op, before any
    producer read: the layout is the month's frame, so a daily run never moves it. Otherwise
    the ring order and sector frames are pinned to the latest earlier month's rows (see
    :mod:`insights.style_atlas`), and both tables are written in one transaction.

    The layout reads the producer's style-year-counts response itself, not the stored
    ``insights.style_parents``. The lean between ring neighbours and the ring's genre overlap
    need the raw shares under the 0.20 parent threshold, which that table does not keep. The
    parents are reduced with the same rule as the stored table, so hulls match it. Emergence
    is the 25-release year from the same response's release-year counts
    (:mod:`insights.style_lifecycle`).

    Returns:
        The number of rows written across both tables, 0 for a no-op.
    """
    month = atlas_month(now)
    async with pool.connection() as conn, conn.cursor() as cursor:
        cursor = cast("Any", cursor)
        await cursor.execute(_STYLE_ATLAS_MONTH_EXISTS_SQL, (month,))
        if await cursor.fetchone() is not None:
            logger.info("📊 Style atlas already laid out for this month", month=month.isoformat())
            return 0
        await cursor.execute(_STYLE_ATLAS_PINNED_SECTORS_SQL, (month,))
        pinned_rows = await cursor.fetchall()
        previous_positions: dict[str, tuple[float, float]] = {}
        if pinned_rows:
            await cursor.execute(_STYLE_ATLAS_PREVIOUS_STYLES_SQL, (pinned_rows[0][0],))
            previous_positions = {style: (x, y) for style, x, y in await cursor.fetchall()}

    data = await _get_json(client, STYLE_YEAR_COUNTS_PATH)
    style_years: list[dict[str, Any]] = data.get("style_years", [])
    style_shares: list[dict[str, Any]] = data.get("style_parents", [])
    if not style_years or not style_shares:
        logger.warning(
            "⚠️ Style atlas input is missing a half — not laying out this month", style_years=len(style_years), style_parents=len(style_shares)
        )
        return 0

    pinned = [Sector(genre, ring_order, start, end, x, y) for _month, genre, ring_order, start, end, x, y in pinned_rows]
    shares: dict[str, dict[str, float]] = {}
    for row in style_shares:
        shares.setdefault(row["style"], {})[row["genre"]] = float(row["share"])
    parents = _style_parent_rows(style_shares)
    lifecycles = style_lifecycles(((row["style"], row["year"], row["release_count"]) for row in style_years), last_complete_year(month))
    layout = build_layout(shares, parents, lifecycles.emergence_years(), pinned)

    hulls = hull_check(layout.nodes, parents)
    moved = movement(previous_positions, style_positions(layout.nodes))
    log = logger.warning if hulls.intruding_nodes or hulls.omitted_pairs else logger.info
    log(
        "📐 Style atlas laid out",
        month=month.isoformat(),
        ring_pinned=layout.ring_pinned,
        frames_pinned=layout.frames_pinned,
        intruding_nodes=hulls.intruding_nodes,
        omitted_parent_pairs=hulls.omitted_pairs,
        p95_movement_nn=None if moved is None else round(moved.p95_nn, 3),
        share_moving_over_1nn=None if moved is None else round(moved.share_over_1nn, 4),
    )

    sector_rows = [(month, s.genre, s.ring_order, s.start, s.end, s.x, s.y) for s in layout.sectors]
    node_rows = [(month, n.kind, n.style, n.genre, n.x, n.y) for n in layout.nodes]
    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            cursor = cast("Any", cursor)
            await cursor.execute(_STYLE_ATLAS_LOCK_SQL)
            await cursor.execute(_STYLE_ATLAS_MONTH_EXISTS_SQL, (month,))
            if await cursor.fetchone() is not None:
                logger.info("📊 Style atlas written by a concurrent run — leaving it", month=month.isoformat())
                return 0
            await cursor.executemany(
                """
                INSERT INTO insights.genre_atlas_sectors (month, genre, ring_order, sector_start, sector_end, x, y)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                sector_rows,
            )
            if node_rows:
                await cursor.executemany(
                    """
                    INSERT INTO insights.style_atlas_layout (month, node_kind, style, genre, x, y)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    node_rows,
                )
    logger.info("💾 Style atlas stored", month=month.isoformat(), sectors=len(sector_rows), nodes=len(node_rows))
    return len(sector_rows) + len(node_rows)


@_record_lifecycle("label_longevity", "❌ Label longevity computation failed")
async def compute_and_store_label_longevity(client: httpx.AsyncClient, pool: Any, limit: int = 50) -> int:
    """Compute label longevity and store results."""
    results = await _fetch_from_api(client, LABEL_LONGEVITY_PATH, {"limit": limit})
    if not results:
        return 0

    current_year = datetime.now(UTC).year
    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            cursor = cast("Any", cursor)
            await cursor.execute("DELETE FROM insights.label_longevity")
            for rank, row in enumerate(results, 1):
                still_active = row["last_year"] is not None and row["last_year"] >= current_year - 2
                await cursor.execute(
                    """
                        INSERT INTO insights.label_longevity
                            (rank, label_id, label_name, first_year, last_year,
                             years_active, total_releases, peak_decade, still_active)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                    (
                        rank,
                        row["label_id"],
                        row["label_name"],
                        row["first_year"],
                        row["last_year"],
                        row["years_active"],
                        row["total_releases"],
                        row.get("peak_decade"),
                        still_active,
                    ),
                )
    logger.info("💾 Label longevity stored", count=len(results))
    return len(results)


@_record_lifecycle("anniversaries", "❌ Anniversaries computation failed")
async def compute_and_store_anniversaries(
    client: httpx.AsyncClient,
    pool: Any,
    current_year: int | None = None,
    current_month: int | None = None,
    milestone_years: list[int] | None = None,
) -> int:
    """Compute monthly anniversaries and store results."""
    now = datetime.now(UTC)
    year = current_year or now.year
    month = current_month or now.month

    if milestone_years is None:
        milestone_years = [25, 30, 40, 50, 75, 100]

    milestones_str = ",".join(str(m) for m in milestone_years)
    results = await _fetch_from_api(
        client,
        ANNIVERSARIES_PATH,
        {"year": year, "month": month, "milestones": milestones_str},
    )
    if not results:
        return 0

    rows_written = 0
    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            cursor = cast("Any", cursor)
            await cursor.execute(
                "DELETE FROM insights.monthly_anniversaries WHERE computed_year = %s AND computed_month = %s",
                (year, month),
            )
            for row in results:
                anniversary = year - int(row["release_year"])
                if anniversary not in milestone_years:
                    logger.warning(
                        "⚠️ Skipping anniversary row — computed anniversary not in milestones",
                        master_id=row.get("master_id"),
                        anniversary=anniversary,
                    )
                    continue
                await cursor.execute(
                    """
                        INSERT INTO insights.monthly_anniversaries
                            (master_id, title, artist_name, release_year, anniversary,
                             computed_month, computed_year)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (master_id, computed_year, computed_month) DO UPDATE
                        SET title = EXCLUDED.title, artist_name = EXCLUDED.artist_name,
                            anniversary = EXCLUDED.anniversary, computed_at = NOW()
                        """,
                    (row["master_id"], row["title"], row.get("artist_name"), int(row["release_year"]), anniversary, month, year),
                )
                rows_written += 1
    logger.info("💾 Monthly anniversaries stored", count=rows_written, year=year, month=month)
    return rows_written


@_record_lifecycle("data_completeness", "❌ Data completeness computation failed")
async def compute_and_store_data_completeness(client: httpx.AsyncClient, pool: Any) -> int:
    """Compute data completeness and store results."""
    # This endpoint's cold full scans need the extended read budget above.
    results = await _fetch_from_api(client, DATA_COMPLETENESS_PATH)
    if not results:
        return 0

    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            cursor = cast("Any", cursor)
            await cursor.execute("DELETE FROM insights.data_completeness")
            for row in results:
                await cursor.execute(
                    """
                        INSERT INTO insights.data_completeness
                            (entity_type, total_count, with_image, with_year,
                             with_country, with_genre, completeness_pct)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                    (
                        row["entity_type"],
                        row["total_count"],
                        row["with_image"],
                        row["with_year"],
                        row["with_country"],
                        row["with_genre"],
                        row["completeness_pct"],
                    ),
                )
    logger.info("💾 Data completeness stored", count=len(results))
    return len(results)


@_record_lifecycle("community_enrichment", "❌ Community enrichment failed")
async def compute_and_store_community_enrichment(client: httpx.AsyncClient, pool: Any) -> int:
    """Trigger community enrichment via the API internal endpoint."""
    del pool
    path = COMMUNITY_ENRICHMENT_PATH
    response = await client.get(path, timeout=endpoint_timeout(path))
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    enriched = int(data.get("enriched", 0))
    logger.info("📊 Community enrichment complete", enriched=enriched)
    return enriched


@_record_lifecycle("release_rarity", "❌ Release rarity computation failed")
async def compute_and_store_rarity(client: httpx.AsyncClient, pool: Any) -> int:
    """Compute release rarity scores and store results."""
    # Chunked, sequential rarity scans need the extended read budget above.
    results = await _fetch_from_api(client, RARITY_SCORES_PATH)
    if not results:
        logger.info("📊 No rarity score results to store")
        return 0

    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            cursor = cast("Any", cursor)
            await cursor.execute("DELETE FROM insights.release_rarity")
            for row in results:
                await cursor.execute(
                    """
                        INSERT INTO insights.release_rarity
                            (release_id, title, artist_name, year, rarity_score, tier,
                             hidden_gem_score, pressing_scarcity, label_catalog,
                             format_rarity, temporal_scarcity, graph_isolation,
                             collection_prevalence, media_families, family_signals,
                             medium_rarity)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                    (
                        row["release_id"],
                        row.get("title", ""),
                        row.get("artist_name", ""),
                        row.get("year"),
                        row["rarity_score"],
                        row["tier"],
                        row.get("hidden_gem_score"),
                        # Grooved-only per ADR 0007; null when no family extension claims the release.
                        row.get("pressing_scarcity"),
                        row.get("label_catalog"),
                        row.get("format_rarity"),
                        row.get("temporal_scarcity"),
                        row.get("graph_isolation"),
                        row.get("collection_prevalence"),
                        Jsonb(row.get("media_families") or []),
                        Jsonb(row.get("family_signals") or {}),
                        row.get("medium_rarity"),
                    ),
                )
    logger.info("💾 Release rarity scores stored", count=len(results))
    return len(results)


# ── Activity summary ────────────────────────────────────────────────────────
#
# The one computation that reads PostgreSQL rather than catalog-api. Its inputs are
# `activity.events` and `activity.impressions`, reached only through insights/activity.py
# so ADR 0010's two consent checks are both applied, and filtered on `product_analytics`:
# counting what happened is product analytics, and a subject who consented to model
# training has not thereby consented to being counted here.

# The table this computation writes and `/api/insights/activity-summary` reads. It is
# DECLARED IN `database-schema`, not here: this repository contains no DDL and every other
# `insights.*` table is created by that repository's `_INSIGHTS_STATEMENTS`. Adding the
# table there is the filed follow-on; `tests/test_activity_summary.py` documents it and
# keeps the SQL below agreeing with this name.
ACTIVITY_SUMMARY_TABLE: Final = "insights.activity_summary"

# Whole UTC days of activity each run summarises. Seven covers a week of daily runs with
# enough overlap that one skipped cycle leaves no gap.
ACTIVITY_SUMMARY_DEFAULT_DAYS: Final = 7

# The two groupings the summary carries, distinguished by the `dimension` column: events
# grouped by type, impressions grouped by ranking policy.
EVENT_TYPE_DIMENSION: Final = "event_type"
POLICY_ID_DIMENSION: Final = "policy_id"


@dataclass
class _ActivityBucket:
    """The running counts for one ``(day, dimension, key)`` group."""

    records: int = 0
    subjects: set[UUID] = field(default_factory=set)
    candidate_sets: set[UUID] = field(default_factory=set)


def activity_summary_window(days: int, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Return the half-open ``occurred_at`` bound covering the last ``days`` whole UTC days.

    Whole days, because the summary is keyed by date and a window that started mid-day would
    make the oldest row mean something different from the rest. The upper bound is the start
    of tomorrow, so today is included in full and is simply incomplete until the day ends —
    each run rewrites it.

    Args:
        days: Number of whole UTC days to cover, including today.
        now: The moment to anchor on. Defaults to the current time.

    Returns:
        ``(since, until)``, inclusive lower and exclusive upper.

    Raises:
        ValueError: If ``days`` is not positive.
    """
    if days < 1:
        raise ValueError(f"days must be positive, got {days}")
    moment = now or datetime.now(UTC)
    midnight = moment.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    until = midnight + timedelta(days=1)
    return until - timedelta(days=days), until


def _activity_summary_rows(buckets: dict[tuple[date, str, str], _ActivityBucket]) -> list[tuple[Any, ...]]:
    """Flatten the accumulated buckets into insertable rows, in a stable order.

    ``candidate_set_count`` is null for the event rows: a candidate set is a property of a
    ranking decision, so counting them for an event would be counting nothing rather than
    counting zero.
    """
    return [
        (
            summary_date,
            dimension,
            dimension_key,
            bucket.records,
            len(bucket.subjects),
            len(bucket.candidate_sets) if dimension == POLICY_ID_DIMENSION else None,
        )
        for (summary_date, dimension, dimension_key), bucket in sorted(buckets.items())
    ]


async def _store_activity_summary(pool: Any, since: date, until: date, rows: Iterable[tuple[Any, ...]]) -> None:
    """Replace the summary rows covering ``[since, until)`` with ``rows``.

    The window is rewritten even when it is empty, which is where this differs from the
    catalog-derived computations above: those leave the previous snapshot intact on an empty
    producer result, because an empty result there means the producer had nothing to say. An
    empty result here can mean a subject revoked consent, and the previous run's counts for
    that subject must not survive it.
    """
    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            cursor = cast("Any", cursor)
            await cursor.execute(
                "DELETE FROM insights.activity_summary WHERE summary_date >= %s AND summary_date < %s",
                (since, until),
            )
            for row in rows:
                await cursor.execute(
                    """
                        INSERT INTO insights.activity_summary
                            (summary_date, dimension, dimension_key, record_count, subject_count, candidate_set_count)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                    row,
                )


@_record_lifecycle("activity_summary", "❌ Activity summary computation failed")
async def compute_and_store_activity_summary(client: httpx.AsyncClient, pool: Any, days: int = ACTIVITY_SUMMARY_DEFAULT_DAYS) -> int:
    """Summarise consented first-party activity, so operators can see it flowing.

    Per day and event type, the number of events and the number of distinct subjects. Per
    day and ranking policy, the number of impressions, the distinct subjects, and the
    distinct candidate sets. Nothing here reaches a raw activity row: the summary is what
    an operator reads, and the consent filter is what produced it.

    Args:
        client: Unused. Every other computation fetches from catalog-api; this one reads the
            shared database, and keeps the signature so ``_record_lifecycle`` and
            ``run_all_computations`` treat all ten computations identically.
        pool: The shared PostgreSQL pool. The ``activity`` and ``insights`` schemas live in
            the same database.
        days: Whole UTC days to summarise, including today.

    Returns:
        The number of summary rows written.
    """
    del client
    since, until = activity_summary_window(days)
    buckets: dict[tuple[date, str, str], _ActivityBucket] = {}

    async with pool.connection() as conn:
        async for event in read_events(conn, since=since, until=until, purposes=(PRODUCT_ANALYTICS,)):
            bucket = buckets.setdefault((event.occurred_at.astimezone(UTC).date(), EVENT_TYPE_DIMENSION, event.event_type), _ActivityBucket())
            bucket.records += 1
            bucket.subjects.add(event.subject_id)
        async for impression in read_impressions(conn, since=since, until=until, purposes=(PRODUCT_ANALYTICS,)):
            bucket = buckets.setdefault((impression.occurred_at.astimezone(UTC).date(), POLICY_ID_DIMENSION, impression.policy_id), _ActivityBucket())
            bucket.records += 1
            bucket.subjects.add(impression.subject_id)
            bucket.candidate_sets.add(impression.candidate_set_id)

    rows = _activity_summary_rows(buckets)
    await _store_activity_summary(pool, since.date(), until.date(), rows)
    logger.info("💾 Activity summary stored", count=len(rows), since=since.date().isoformat(), until=until.date().isoformat())
    return len(rows)


async def run_all_computations(
    client: httpx.AsyncClient,
    pool: Any,
    *,
    milestone_years: list[int] | None = None,
) -> dict[str, int]:
    """Run all insight computations and return row counts per type."""
    logger.info("🔄 Starting all insight computations...")
    results: dict[str, int] = {}
    errors: dict[str, str] = {}

    computations: list[tuple[str, Callable[[], Coroutine[Any, Any, int]]]] = [
        ("artist_centrality", lambda: compute_and_store_artist_centrality(client, pool)),
        ("genre_trends", lambda: compute_and_store_genre_trends(client, pool)),
        ("style_year_counts", lambda: compute_and_store_style_year_counts(client, pool)),
        ("style_atlas", lambda: compute_and_store_style_atlas(client, pool)),
        ("label_longevity", lambda: compute_and_store_label_longevity(client, pool)),
        (
            "anniversaries",
            lambda: compute_and_store_anniversaries(client, pool, milestone_years=milestone_years),
        ),
        ("data_completeness", lambda: compute_and_store_data_completeness(client, pool)),
        ("community_enrichment", lambda: compute_and_store_community_enrichment(client, pool)),
        ("release_rarity", lambda: compute_and_store_rarity(client, pool)),
        ("activity_summary", lambda: compute_and_store_activity_summary(client, pool)),
    ]

    for name, factory in computations:
        started = time.perf_counter()
        try:
            # The `insights {computation}` root span wraps exactly what the duration histogram
            # measures, so a slow computation is attributable to the calls it made. The
            # exception is re-raised through the span (which marks it ERROR) and caught here,
            # because one failed computation must not stop the remaining ones.
            with computation_span(name):
                results[name] = await factory()
        except Exception as e:
            record_computation(name, time.perf_counter() - started, success=False)
            logger.error("❌ Computation failed — continuing with remaining computations", computation=name, error=describe_exception(e))
            errors[name] = describe_exception(e)
        else:
            record_computation(name, time.perf_counter() - started, success=True)

    total = sum(results.values())
    logger.info("✅ All insight computations complete", total_rows=total, breakdown=results, failed=list(errors.keys()) or None)
    return results
