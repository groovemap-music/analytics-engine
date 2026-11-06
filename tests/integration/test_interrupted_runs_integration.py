"""An interrupted computation is visible in /api/insights/status, proved on a real PostgreSQL.

Each case runs a real lifecycle-wrapped computation in a child process, inside a rarity-style
transaction, and ends it the way production ends one. The start row is written and committed before
the work begins and outside that transaction, so a rollback or a kill cannot erase it.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import pytest_asyncio

import insights.insights as engine
from insights.computations import _record_lifecycle, mark_interrupted_runs


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from common import AsyncPostgreSQLPool

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session"), pytest.mark.timeout(120)]

CHILD = Path(__file__).with_name("killable_run.py")
INSIGHT = "release_rarity"


async def _status(pool: AsyncPostgreSQLPool) -> dict[str, Any]:
    with patch.object(engine, "_pool", pool):
        response = await engine.computation_status()
    statuses = json.loads(response.body)["statuses"]
    return next(row for row in statuses if row["insight_type"] == INSIGHT)


async def _rows(pool: AsyncPostgreSQLPool) -> list[tuple[Any, ...]]:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "SELECT status, completed_at IS NOT NULL, error_message FROM insights.computation_log WHERE insight_type = %s ORDER BY id", (INSIGHT,)
        )
        return [tuple(row) for row in await cursor.fetchall()]


async def _until_printed(child: asyncio.subprocess.Process, marker: bytes) -> None:
    """Read the child's output, log lines included, until it prints ``marker``."""
    assert child.stdout is not None
    async with asyncio.timeout(60):
        while True:
            line = await child.stdout.readline()
            assert line, f"the child exited before printing {marker!r}"
            if line.strip() == marker:
                return


async def _seed_completed_run(pool: AsyncPostgreSQLPool) -> str:
    """A finished run two days ago; returns its completed_at as the status endpoint renders it."""
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "INSERT INTO insights.computation_log (insight_type, status, started_at, completed_at, duration_ms) "
            "VALUES (%s, 'completed', NOW() - interval '2 days', NOW() - interval '2 days', 1000) RETURNING completed_at",
            (INSIGHT,),
        )
        row = await cursor.fetchone()
    assert row is not None
    return str(row[0].isoformat().replace("+00:00", ""))[:19]


async def _spawn(mode: str) -> asyncio.subprocess.Process:
    child = await asyncio.create_subprocess_exec(sys.executable, str(CHILD), mode, stdout=asyncio.subprocess.PIPE, env=os.environ.copy())
    await _until_printed(child, b"READY")
    return child


@pytest_asyncio.fixture(loop_scope="session")
async def pool(schema_owner_pool: AsyncPostgreSQLPool) -> AsyncIterator[AsyncPostgreSQLPool]:
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("DELETE FROM insights.computation_log")
    yield schema_owner_pool
    async with schema_owner_pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute("DELETE FROM insights.computation_log")


async def test_a_completed_run_reports_as_it_always_did(pool: AsyncPostgreSQLPool) -> None:
    @_record_lifecycle(INSIGHT, "failed")
    async def compute(_client: Any, _pool: Any) -> int:
        return 7

    assert await compute(None, pool) == 7

    assert await _rows(pool) == [("completed", True, None)]
    status = await _status(pool)
    assert status["status"] == "completed"
    assert status["last_computed"] is not None
    assert status["duration_ms"] is not None


async def test_sigkill_mid_transaction_leaves_a_run_that_status_reports_interrupted_after_restart(pool: AsyncPostgreSQLPool) -> None:
    seeded = await _seed_completed_run(pool)
    child = await _spawn("hang")
    # Written, committed and visible from another session while the run's own transaction is still open.
    assert [row[:2] for row in await _rows(pool)] == [("completed", True), ("running", False)]
    live = await _status(pool)
    assert live["status"] == "running"
    assert live["last_computed"].startswith(seeded), "a live run must not stand in for the last finished one"

    child.send_signal(signal.SIGKILL)
    await child.wait()
    killed_at = time.monotonic()

    assert [row[:2] for row in await _rows(pool)][-1] == ("running", False), "SIGKILL runs no code, so the row is still open until a restart"
    assert await mark_interrupted_runs(pool) == 1  # what lifespan() does before the scheduler starts
    status = await _status(pool)
    print(f"\nSIGKILL: status {status['status']!r} {time.monotonic() - killed_at:.3f}s after the kill, once the engine restarts")
    assert status["status"] == "interrupted"
    assert status["last_computed"].startswith(seeded), "an interrupted run did not compute anything: last_computed is the last finished run"
    assert [row[:2] for row in await _rows(pool)][-1] == ("interrupted", True)


async def test_sigterm_is_recorded_by_the_run_itself_before_it_exits(pool: AsyncPostgreSQLPool) -> None:
    child = await _spawn("hang")
    sent_at = time.monotonic()
    child.send_signal(signal.SIGTERM)
    await _until_printed(child, b"CANCELLED")
    await child.wait()
    elapsed = time.monotonic() - sent_at

    print(f"\nSIGTERM: row closed {elapsed:.2f}s after the signal, no restart needed")
    assert (await _rows(pool))[0][:2] == ("interrupted", True)
    assert (await _status(pool))["status"] == "interrupted"
    assert elapsed < 10


async def test_a_failed_transaction_rolls_back_but_the_record_stays(pool: AsyncPostgreSQLPool) -> None:
    child = await asyncio.create_subprocess_exec(sys.executable, str(CHILD), "fail", stdout=asyncio.subprocess.PIPE, env=os.environ.copy())
    await _until_printed(child, b"FAILED")
    await child.wait()

    rows = await _rows(pool)
    assert [row[:2] for row in rows] == [("failed", True)]
    assert "mid-transaction" in (rows[0][2] or "")


async def test_a_run_nobody_swept_is_reported_stale_after_the_bound(pool: AsyncPostgreSQLPool) -> None:
    async with pool.connection() as conn, conn.cursor() as cursor:
        await cursor.execute(
            "INSERT INTO insights.computation_log (insight_type, status, started_at) VALUES (%s, 'running', NOW() - make_interval(secs => %s))",
            (INSIGHT, engine.RUNNING_STALE_AFTER_S + 60),
        )
        await cursor.execute(
            "INSERT INTO insights.computation_log (insight_type, status, started_at, completed_at, duration_ms) "
            "VALUES (%s, 'completed', NOW() - interval '2 days', NOW() - interval '2 days', 1000)",
            (INSIGHT,),
        )

    status = await _status(pool)
    assert status["status"] == "stale"
    assert status["last_computed"] is not None, "last_computed stays on the last run that finished"


async def test_a_run_whose_start_row_is_gone_still_logs_its_outcome(pool: AsyncPostgreSQLPool) -> None:
    @_record_lifecycle(INSIGHT, "failed")
    async def compute(_client: Any, _pool: Any) -> int:
        async with _pool.connection() as conn, conn.cursor() as cursor:
            await cursor.execute("DELETE FROM insights.computation_log WHERE insight_type = %s", (INSIGHT,))
        return 3

    assert await compute(None, pool) == 3

    assert [row[:2] for row in await _rows(pool)] == [("completed", True)]


async def test_a_failed_latest_run_reports_the_last_completed_time(pool: AsyncPostgreSQLPool) -> None:
    seeded = await _seed_completed_run(pool)

    @_record_lifecycle(INSIGHT, "failed")
    async def compute(_client: Any, _pool: Any) -> int:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await compute(None, pool)

    status = await _status(pool)
    assert status["status"] == "failed"
    assert status["last_computed"].startswith(seeded)
