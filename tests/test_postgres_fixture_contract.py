"""Regression tests for the shared PostgreSQL fixture protocol."""

import inspect
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_pool_and_connection_are_strict_async_context_boundaries(mock_pg_pool: Any) -> None:
    """Pool checkout and cursor creation return contexts without being awaited."""
    unknown_pool_method = "invented_pool_method"
    with pytest.raises(AttributeError):
        getattr(mock_pg_pool, unknown_pool_method)

    connection_context = mock_pg_pool.connection()
    assert not inspect.isawaitable(connection_context)
    async with connection_context as connection:
        unknown_connection_method = "invented_connection_method"
        with pytest.raises(AttributeError):
            getattr(connection, unknown_connection_method)

        cursor_context = connection.cursor()
        assert not inspect.isawaitable(cursor_context)
        async with cursor_context as cursor:
            await cursor.execute("SELECT 1")
            assert await cursor.fetchone() is None
            assert await cursor.fetchall() == []


@pytest.mark.asyncio
async def test_transaction_is_a_strict_async_context(mock_pg_pool: Any) -> None:
    """The transaction boundary matches ``async with conn.transaction()``."""
    async with mock_pg_pool.connection() as connection:
        transaction_context = connection.transaction()
        assert not inspect.isawaitable(transaction_context)
        async with transaction_context as transaction:
            assert transaction is transaction_context

    connection.transaction.assert_called_once_with()
