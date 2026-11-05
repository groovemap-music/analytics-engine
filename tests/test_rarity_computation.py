"""Tests for rarity score computation pipeline."""

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from insights.catalog_api_contract import RARITY_PAGE_PATH, RARITY_QUALITY_COUNTS_PATH
from insights.computations import (
    RARITY_LOCK_TIMEOUT,
    RarityWalkShortError,
    compute_and_store_community_enrichment,
    compute_and_store_rarity,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


def _make_mock_pool() -> MagicMock:
    """Create mock pool with cursor for storing results."""
    mock_cursor = AsyncMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)
    mock_copy = MagicMock()
    mock_copy.write_row = AsyncMock()
    mock_copy.__aenter__ = AsyncMock(return_value=mock_copy)
    mock_copy.__aexit__ = AsyncMock(return_value=False)
    mock_cursor.copy = MagicMock(return_value=mock_copy)

    mock_conn = AsyncMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_tx_cm = AsyncMock()
    mock_tx_cm.__aenter__ = AsyncMock(return_value=None)
    mock_tx_cm.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=mock_tx_cm)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)
    return mock_pool


def _copy_of(mock_pool: MagicMock) -> MagicMock:
    """The mock `COPY` object the pool's cursor handed out."""
    mock_cursor = mock_pool.connection.return_value.__aenter__.return_value.cursor.return_value
    return mock_cursor.copy.return_value


def _copied_rows(mock_pool: MagicMock) -> list[tuple[Any, ...]]:
    return [call.args[0] for call in _copy_of(mock_pool).write_row.call_args_list]


_MOCK_RARITY_ITEMS = [
    {
        "release_id": "1",
        "title": "Test Release",
        "artist_name": "Test Artist",
        "year": 1970,
        "rarity_score": 85.0,
        "tier": "ultra-rare",
        "hidden_gem_score": 60.0,
        "pressing_scarcity": 100.0,
        "label_catalog": 75.0,
        "format_rarity": 95.0,
        "temporal_scarcity": 80.0,
        "graph_isolation": 70.0,
        "collection_prevalence": 85.0,
        "media_families": ["vinyl"],
        "family_signals": {"grooved": {"pressing_scarcity": 92.0}},
        "medium_rarity": 78.0,
    }
]

# A grooved release: the grooved family extension claims it, so pressing_scarcity
# and family_signals are populated (ADR 0007).
_MOCK_GROOVED_RARITY_ITEM = {
    "release_id": "grooved-1",
    "title": "Grooved Release",
    "artist_name": "Grooved Artist",
    "year": 1975,
    "rarity_score": 91.0,
    "tier": "ultra-rare",
    "hidden_gem_score": 77.0,
    "pressing_scarcity": 93.5,
    "label_catalog": 82.0,
    "format_rarity": 5.0,
    "temporal_scarcity": 65.0,
    "graph_isolation": 58.0,
    "collection_prevalence": 41.0,
    "media_families": ["vinyl", "shellac"],
    "family_signals": {"grooved": {"pressing_scarcity": 90.0}},
    "medium_rarity": 74.0,
}

# A non-grooved release: no family extension claims it, so pressing_scarcity is
# null and family_signals is empty (ADR 0007).
_MOCK_NON_GROOVED_RARITY_ITEM = {
    "release_id": "digital-1",
    "title": "Digital Release",
    "artist_name": "Digital Artist",
    "year": 2020,
    "rarity_score": 48.0,
    "tier": "common",
    "hidden_gem_score": 12.0,
    "pressing_scarcity": None,
    "label_catalog": 30.0,
    "format_rarity": 0.0,
    "temporal_scarcity": 20.0,
    "graph_isolation": 15.0,
    "collection_prevalence": 55.0,
    "media_families": ["digital"],
    "family_signals": {},
    "medium_rarity": 38.0,
}


class _FakeApi:
    """The two paged routes behind a real `httpx` client, serving `pages` and recording every call."""

    def __init__(self, pages: list[list[dict[str, Any]]], *, release_total: int | None = None) -> None:
        self.pages = pages
        self.release_total = sum(len(page) for page in pages) if release_total is None else release_total
        self.calls: list[tuple[str, str, Any]] = []
        self.failures: dict[tuple[str, str], list[int | Exception]] = {}

    def fail(self, path: str, after: str, *outcomes: int | Exception) -> None:
        """Answer the next requests for this page with these statuses or transport errors."""
        self.failures[(path, after)] = list(outcomes)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        after = request.url.params["after"]
        body = json.loads(request.content) if request.content else None
        self.calls.append((path, after, body))
        queued = self.failures.get((path, after))
        if queued:
            outcome = queued.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return httpx.Response(outcome)
        index = int(after or 0)
        following = str(index + 1) if index + 1 < len(self.pages) else None
        if path == RARITY_QUALITY_COUNTS_PATH:
            return httpx.Response(
                200,
                json={
                    "artist": [[0.0, 1], [index + 1.0, 2]],
                    "label": [[1.0, 1]],
                    "genre": [],
                    "releases": len(self.pages[index]) if self.pages else 0,
                    "release_total": self.release_total - index,
                    "next_after": following,
                },
            )
        return httpx.Response(200, json={"items": self.pages[index], "next_after": following})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle), base_url="http://api")

    def after_values(self, path: str) -> list[str]:
        return [after for called, after, _body in self.calls if called == path]


@pytest.fixture
def no_backoff() -> Iterator[AsyncMock]:
    with patch("insights.computations.asyncio.sleep", new_callable=AsyncMock) as sleep:
        yield sleep


def _rows(*release_ids: str) -> list[dict[str, Any]]:
    return [{**_MOCK_RARITY_ITEMS[0], "release_id": release_id} for release_id in release_ids]


def _statements(mock_pool: MagicMock) -> list[str]:
    mock_cursor = mock_pool.connection.return_value.__aenter__.return_value.cursor.return_value
    return [call.args[0] for call in mock_cursor.execute.call_args_list if "computation_log" not in call.args[0]]


class TestComputeAndStoreRarity:
    @pytest.mark.asyncio
    async def test_walks_every_page_into_one_transaction(self) -> None:
        api = _FakeApi([_rows("1", "2"), _rows("3")])
        mock_pool = _make_mock_pool()

        async with api.client() as client:
            rows = await compute_and_store_rarity(client, mock_pool)

        assert rows == 3
        mock_conn = mock_pool.connection.return_value.__aenter__.return_value
        mock_conn.transaction.assert_called_once()
        assert [row[0] for row in _copied_rows(mock_pool)] == ["1", "2", "3"]
        assert api.after_values(RARITY_PAGE_PATH) == ["", "1"]

    @pytest.mark.asyncio
    async def test_scores_every_page_against_the_counts_summed_across_pages(self) -> None:
        api = _FakeApi([_rows("1"), _rows("2")])

        async with api.client() as client:
            await compute_and_store_rarity(client, _make_mock_pool())

        bodies = [body for path, _after, body in api.calls if path == RARITY_PAGE_PATH]
        assert bodies[0] == {"artist": [[0.0, 2], [1.0, 2], [2.0, 2]], "label": [[1.0, 2]], "genre": []}
        assert bodies[1] == bodies[0]

    @pytest.mark.asyncio
    async def test_sets_the_lock_timeout_before_it_touches_the_table(self) -> None:
        api = _FakeApi([_rows("1")])
        mock_pool = _make_mock_pool()

        async with api.client() as client:
            await compute_and_store_rarity(client, mock_pool)

        assert _statements(mock_pool) == [f"SET LOCAL lock_timeout = '{RARITY_LOCK_TIMEOUT}'", "DELETE FROM insights.release_rarity"]

    @pytest.mark.asyncio
    async def test_the_table_is_not_touched_until_the_counts_are_complete(self, no_backoff: AsyncMock) -> None:
        api = _FakeApi([_rows("1"), _rows("2")])
        api.fail(RARITY_QUALITY_COUNTS_PATH, "1", *[503] * 10)
        mock_pool = _make_mock_pool()

        async with api.client() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await compute_and_store_rarity(client, mock_pool)

        assert _statements(mock_pool) == []
        assert no_backoff.await_count == 9

    @pytest.mark.asyncio
    async def test_an_empty_catalog_keeps_the_previous_table(self) -> None:
        api = _FakeApi([])
        api.pages = [[]]
        mock_pool = _make_mock_pool()

        async with api.client() as client:
            rows = await compute_and_store_rarity(client, mock_pool)

        assert rows == 0
        assert "DELETE FROM insights.release_rarity" not in _statements(mock_pool)
        _copy_of(mock_pool).write_row.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [503, 500, 429, httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("dropped"), httpx.ConnectError("refused")],
        ids=["503", "500", "429", "read-timeout", "dropped", "refused"],
    )
    async def test_repeats_the_same_page_and_writes_it_once(self, failure: int | Exception, no_backoff: AsyncMock) -> None:
        api = _FakeApi([_rows("1"), _rows("2"), _rows("3")])
        api.fail(RARITY_PAGE_PATH, "1", failure, failure)
        mock_pool = _make_mock_pool()

        async with api.client() as client:
            rows = await compute_and_store_rarity(client, mock_pool)

        assert rows == 3
        assert api.after_values(RARITY_PAGE_PATH) == ["", "1", "1", "1", "2"]
        assert [row[0] for row in _copied_rows(mock_pool)] == ["1", "2", "3"]
        assert [call.args[0] for call in no_backoff.await_args_list] == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_backoff_doubles_up_to_its_cap(self, no_backoff: AsyncMock) -> None:
        api = _FakeApi([_rows("1")])
        api.fail(RARITY_PAGE_PATH, "", *[503] * 9)

        async with api.client() as client:
            await compute_and_store_rarity(client, _make_mock_pool())

        assert [call.args[0] for call in no_backoff.await_args_list] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]

    @pytest.mark.asyncio
    async def test_a_page_that_never_comes_back_fails_the_run(self, no_backoff: AsyncMock) -> None:
        api = _FakeApi([_rows("1"), _rows("2")])
        api.fail(RARITY_PAGE_PATH, "1", *[httpx.ConnectError("refused")] * 10)
        mock_pool = _make_mock_pool()

        async with api.client() as client:
            with pytest.raises(httpx.ConnectError):
                await compute_and_store_rarity(client, mock_pool)

        # The exception leaves the transaction block, which is what rolls it back.
        mock_conn = mock_pool.connection.return_value.__aenter__.return_value
        assert isinstance(mock_conn.transaction.return_value.__aexit__.await_args.args[1], httpx.ConnectError)
        assert no_backoff.await_count == 9

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 404, 422])
    async def test_a_request_the_api_refuses_is_not_repeated(self, status: int, no_backoff: AsyncMock) -> None:
        api = _FakeApi([_rows("1")])
        api.fail(RARITY_PAGE_PATH, "", status)

        async with api.client() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await compute_and_store_rarity(client, _make_mock_pool())

        assert api.after_values(RARITY_PAGE_PATH) == [""]
        no_backoff.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_walk_short_of_the_first_release_total_rolls_back(self) -> None:
        api = _FakeApi([_rows("1"), _rows("2")], release_total=3)
        mock_pool = _make_mock_pool()

        with patch("insights.computations._log_computation") as mock_log:
            async with api.client() as client:
                with pytest.raises(RarityWalkShortError, match="wrote 2 rows of 3 releases"):
                    await compute_and_store_rarity(client, mock_pool)

        mock_conn = mock_pool.connection.return_value.__aenter__.return_value
        assert isinstance(mock_conn.transaction.return_value.__aexit__.await_args.args[1], RarityWalkShortError)
        assert mock_log.call_args[0][2] == "failed"

    @pytest.mark.asyncio
    async def test_a_catalog_that_grew_during_the_walk_is_stored(self) -> None:
        api = _FakeApi([_rows("1"), _rows("2")], release_total=1)

        async with api.client() as client:
            rows = await compute_and_store_rarity(client, _make_mock_pool())

        assert rows == 2

    @pytest.mark.asyncio
    async def test_logs_progress_on_a_long_walk(self) -> None:
        api = _FakeApi([_rows(str(index)) for index in range(3)])

        with (
            patch("insights.computations._RARITY_PROGRESS_PAGES", 2),
            patch("insights.computations.logger") as mock_logger,
        ):
            async with api.client() as client:
                await compute_and_store_rarity(client, _make_mock_pool())

        mock_logger.info.assert_any_call("📊 Storing release rarity scores", rows=2)

    @pytest.mark.asyncio
    async def test_logs_computation(self) -> None:
        api = _FakeApi([_rows("1")])

        with patch("insights.computations._log_computation") as mock_log:
            async with api.client() as client:
                await compute_and_store_rarity(client, _make_mock_pool())

        mock_log.assert_called_once()
        args = mock_log.call_args
        assert args[0][1] == "release_rarity"
        assert args[0][2] == "completed"

    @pytest.mark.asyncio
    async def test_grooved_release_writes_pressing_scarcity_and_family_signals(self) -> None:
        """A grooved release's COPY row carries a populated pressing_scarcity and
        the grooved family's signal breakdown (ADR 0007)."""
        mock_pool = _make_mock_pool()

        async with _FakeApi([[_MOCK_GROOVED_RARITY_ITEM]]).client() as client:
            rows = await compute_and_store_rarity(client, mock_pool)

        assert rows == 1
        (params,) = _copied_rows(mock_pool)
        assert params[0] == "grooved-1"
        assert params[7] == 93.5  # pressing_scarcity
        assert params[13].obj == ["vinyl", "shellac"]  # media_families (Jsonb)
        assert params[14].obj == {"grooved": {"pressing_scarcity": 90.0}}  # family_signals (Jsonb)
        assert params[15] == 74.0  # medium_rarity
        assert params[16:] == (None, None)  # label_ids, artist_ids: an older stream omits them

    @pytest.mark.asyncio
    async def test_non_grooved_release_writes_null_pressing_scarcity_and_empty_family_signals(self) -> None:
        """A non-grooved release's COPY row carries a null pressing_scarcity and
        an empty family_signals object, since no family extension claims it (ADR 0007)."""
        mock_pool = _make_mock_pool()

        async with _FakeApi([[_MOCK_NON_GROOVED_RARITY_ITEM]]).client() as client:
            rows = await compute_and_store_rarity(client, mock_pool)

        assert rows == 1
        (params,) = _copied_rows(mock_pool)
        assert params[0] == "digital-1"
        assert params[7] is None  # pressing_scarcity
        assert params[13].obj == ["digital"]  # media_families (Jsonb)
        assert params[14].obj == {}  # family_signals (Jsonb)
        assert params[15] == 38.0  # medium_rarity

    @pytest.mark.asyncio
    async def test_label_and_artist_ids_from_the_stream_are_copied_as_given(self) -> None:
        mock_pool = _make_mock_pool()
        item = {**_MOCK_GROOVED_RARITY_ITEM, "label_ids": [4, 9], "artist_ids": []}

        async with _FakeApi([[item]]).client() as client:
            await compute_and_store_rarity(client, mock_pool)

        (params,) = _copied_rows(mock_pool)
        assert params[16:] == ([4, 9], [])  # an empty list is "none", distinct from a stream without the field

    @pytest.mark.asyncio
    async def test_logs_failure(self) -> None:
        with (
            patch("insights.computations._rarity_request", side_effect=RuntimeError("fail")),
            patch("insights.computations._log_computation") as mock_log,
            pytest.raises(RuntimeError),
        ):
            await compute_and_store_rarity(AsyncMock(), _make_mock_pool())

        mock_log.assert_called_once()
        args = mock_log.call_args
        assert args[0][2] == "failed"


class TestComputeAndStoreCommunityEnrichment:
    @pytest.mark.asyncio
    async def test_community_enrichment_success(self) -> None:
        mock_client = AsyncMock()
        mock_pool = _make_mock_pool()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"enriched": 5}
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("insights.computations._log_computation") as mock_log:
            result = await compute_and_store_community_enrichment(mock_client, mock_pool)

        assert result == 5
        from insights.computations import endpoint_timeout

        path = "/api/internal/insights/community-enrichment"
        mock_client.get.assert_called_once_with(path, timeout=endpoint_timeout(path))
        mock_log.assert_called_once()
        args = mock_log.call_args
        assert args[0][1] == "community_enrichment"
        assert args[0][2] == "completed"

    @pytest.mark.asyncio
    async def test_community_enrichment_failure(self) -> None:
        mock_client = AsyncMock()
        mock_pool = _make_mock_pool()

        mock_client.get = AsyncMock(side_effect=RuntimeError("connection error"))

        with (
            patch("insights.computations._log_computation") as mock_log,
            pytest.raises(RuntimeError, match="connection error"),
        ):
            await compute_and_store_community_enrichment(mock_client, mock_pool)

        mock_log.assert_called_once()
        args = mock_log.call_args
        assert args[0][1] == "community_enrichment"
        assert args[0][2] == "failed"
