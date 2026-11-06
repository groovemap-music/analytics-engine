"""A child process that runs one real, lifecycle-wrapped computation until it is killed.

Used by ``test_interrupted_runs_integration.py``. It opens a rarity-style transaction on the real
PostgreSQL (a DELETE that is never committed), prints ``READY`` once the computation is inside it,
and then sleeps. ``--on-sigterm cancel`` stands in for the service shutdown path: the signal
cancels the scheduler's task, which is how the cancellation reaches a running computation.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from typing import Any

from common import AsyncPostgreSQLPool, parse_postgres_host_port

from insights.computations import _record_lifecycle


@_record_lifecycle("release_rarity", "❌ Release rarity computation failed")
async def hang_inside_a_transaction(_client: Any, pool: Any) -> int:
    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            await cursor.execute("DELETE FROM insights.release_rarity")
            print("READY", flush=True)
            await asyncio.sleep(3600)
    return 0


@_record_lifecycle("release_rarity", "❌ Release rarity computation failed")
async def fail_inside_a_transaction(_client: Any, pool: Any) -> int:
    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        async with conn.transaction(), conn.cursor() as cursor:
            await cursor.execute("DELETE FROM insights.release_rarity")
            raise RuntimeError("the computation failed mid-transaction")


async def main(mode: str) -> None:
    host, port = parse_postgres_host_port(os.environ["POSTGRES_HOST"])
    pool = AsyncPostgreSQLPool(
        connection_params={
            "host": host,
            "port": port,
            "dbname": os.environ["POSTGRES_DATABASE"],
            "user": os.environ["POSTGRES_USERNAME"],
            "password": os.environ["POSTGRES_PASSWORD"],
        },
        min_connections=1,
        max_connections=2,
        max_retries=1,
        health_check_interval=3600,
    )
    await pool.initialize()
    operation = hang_inside_a_transaction if mode == "hang" else fail_inside_a_transaction
    task = asyncio.create_task(operation(None, pool))
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        print("CANCELLED", flush=True)
    except RuntimeError:
        print("FAILED", flush=True)
    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hang", "fail"])
    asyncio.run(main(parser.parse_args().mode))
    sys.exit(0)
