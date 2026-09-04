"""Tests for rarity score computation pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from insights.computations import compute_and_store_community_enrichment, compute_and_store_rarity


def _make_mock_pool() -> MagicMock:
    """Create mock pool with cursor for storing results."""
    mock_cursor = AsyncMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

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


class TestComputeAndStoreRarity:
    @pytest.mark.asyncio
    async def test_fetches_and_stores(self) -> None:
        mock_client = AsyncMock()
        mock_pool = _make_mock_pool()

        with patch("insights.computations._fetch_from_api") as mock_fetch:
            mock_fetch.return_value = _MOCK_RARITY_ITEMS
            rows = await compute_and_store_rarity(mock_client, mock_pool)

        assert rows == 1
        # The per-endpoint budget is applied inside _fetch_from_api, so the
        # call site no longer passes a magic scalar.
        mock_fetch.assert_called_once_with(mock_client, "/api/internal/insights/rarity-scores")

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        mock_client = AsyncMock()
        mock_pool = _make_mock_pool()

        with patch("insights.computations._fetch_from_api") as mock_fetch:
            mock_fetch.return_value = []
            rows = await compute_and_store_rarity(mock_client, mock_pool)

        assert rows == 0

    @pytest.mark.asyncio
    async def test_logs_computation(self) -> None:
        mock_client = AsyncMock()
        mock_pool = _make_mock_pool()

        with (
            patch("insights.computations._fetch_from_api") as mock_fetch,
            patch("insights.computations._log_computation") as mock_log,
        ):
            mock_fetch.return_value = _MOCK_RARITY_ITEMS
            await compute_and_store_rarity(mock_client, mock_pool)

        mock_log.assert_called_once()
        args = mock_log.call_args
        assert args[0][1] == "release_rarity"
        assert args[0][2] == "completed"

    @pytest.mark.asyncio
    async def test_grooved_release_writes_pressing_scarcity_and_family_signals(self) -> None:
        """A grooved release's INSERT carries a populated pressing_scarcity and
        the grooved family's signal breakdown (ADR 0007)."""
        mock_client = AsyncMock()
        mock_pool = _make_mock_pool()

        with patch("insights.computations._fetch_from_api") as mock_fetch:
            mock_fetch.return_value = [_MOCK_GROOVED_RARITY_ITEM]
            rows = await compute_and_store_rarity(mock_client, mock_pool)

        assert rows == 1
        mock_cursor = mock_pool.connection.return_value.__aenter__.return_value.cursor.return_value
        insert_call = next(c for c in mock_cursor.execute.call_args_list if "INSERT INTO insights.release_rarity" in c.args[0])
        params = insert_call.args[1]
        assert params[0] == "grooved-1"
        assert params[7] == 93.5  # pressing_scarcity
        assert params[13].obj == ["vinyl", "shellac"]  # media_families (Jsonb)
        assert params[14].obj == {"grooved": {"pressing_scarcity": 90.0}}  # family_signals (Jsonb)
        assert params[15] == 74.0  # medium_rarity

    @pytest.mark.asyncio
    async def test_non_grooved_release_writes_null_pressing_scarcity_and_empty_family_signals(self) -> None:
        """A non-grooved release's INSERT carries a null pressing_scarcity and
        an empty family_signals object, since no family extension claims it (ADR 0007)."""
        mock_client = AsyncMock()
        mock_pool = _make_mock_pool()

        with patch("insights.computations._fetch_from_api") as mock_fetch:
            mock_fetch.return_value = [_MOCK_NON_GROOVED_RARITY_ITEM]
            rows = await compute_and_store_rarity(mock_client, mock_pool)

        assert rows == 1
        mock_cursor = mock_pool.connection.return_value.__aenter__.return_value.cursor.return_value
        insert_call = next(c for c in mock_cursor.execute.call_args_list if "INSERT INTO insights.release_rarity" in c.args[0])
        params = insert_call.args[1]
        assert params[0] == "digital-1"
        assert params[7] is None  # pressing_scarcity
        assert params[13].obj == ["digital"]  # media_families (Jsonb)
        assert params[14].obj == {}  # family_signals (Jsonb)
        assert params[15] == 38.0  # medium_rarity

    @pytest.mark.asyncio
    async def test_logs_failure(self) -> None:
        mock_client = AsyncMock()
        mock_pool = _make_mock_pool()

        with (
            patch("insights.computations._fetch_from_api", side_effect=RuntimeError("fail")),
            patch("insights.computations._log_computation") as mock_log,
            pytest.raises(RuntimeError),
        ):
            await compute_and_store_rarity(mock_client, mock_pool)

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
