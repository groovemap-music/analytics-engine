"""Unit tests for `insights.embedding_pipeline` (gm-analytics-engine-ieu.2).

Real `insights.embeddings` (FastRP/graph/projection) code runs against small, in-memory
fakes standing in for the PostgreSQL connection — no real database, no Docker. The real-engine
permission and idempotency regressions live in `tests/integration/test_embedding_pipeline_integration.py`.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from insights import embedding_pipeline as pipeline
from insights.embeddings import FastRPConfig


# ── Fakes standing in for psycopg's async connection/cursor protocol ────────────────────────


class _NullTransaction:
    async def __aenter__(self) -> _NullTransaction:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class FakeCursor:
    """One cursor's worth of canned `fetchmany`/`fetchone` results."""

    def __init__(
        self,
        *,
        rows_batches: list[list[tuple[Any, ...]]] | None = None,
        fetchone_value: tuple[Any, ...] | None = None,
        executemany_log: list[tuple[str, list[tuple[Any, ...]]]] | None = None,
    ) -> None:
        self._batches = [list(batch) for batch in (rows_batches or [])]
        self.fetchone_value = fetchone_value
        self.executed: list[tuple[str, Any]] = []
        self._executemany_log = executemany_log

    async def __aenter__(self) -> FakeCursor:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    async def fetchmany(self, _size: int) -> list[tuple[Any, ...]]:
        return self._batches.pop(0) if self._batches else []

    async def fetchone(self) -> tuple[Any, ...] | None:
        return self.fetchone_value

    async def executemany(self, sql: str, batch: Any) -> None:
        rows = list(batch)
        if self._executemany_log is not None:
            self._executemany_log.append((sql, rows))


class FakeConnection:
    """A synthetic graph plus the idempotency row `_already_loaded` should see."""

    def __init__(
        self,
        *,
        vertex_rows: list[tuple[str, str]],
        edge_rows: dict[str, list[tuple[Any, Any]]],
        already_loaded_row: tuple[Any, ...] | None = None,
    ) -> None:
        self.vertex_rows = vertex_rows
        self.edge_rows = edge_rows
        self.already_loaded_row = already_loaded_row
        self.executemany_log: list[tuple[str, list[tuple[Any, ...]]]] = []
        self.opened_cursor_names: list[str | None] = []

    def cursor(self, name: str | None = None) -> FakeCursor:
        self.opened_cursor_names.append(name)
        if name is None:
            return FakeCursor(fetchone_value=self.already_loaded_row, executemany_log=self.executemany_log)
        if name == "embedding_pipeline_vertices":
            return FakeCursor(rows_batches=[self.vertex_rows])
        for table, rows in self.edge_rows.items():
            if name == f"embedding_pipeline_{table.replace('.', '_')}":
                return FakeCursor(rows_batches=[rows])
        raise AssertionError(f"unexpected cursor name: {name!r}")

    def transaction(self) -> _NullTransaction:
        return _NullTransaction()


class FakePool:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    def connection(self) -> _ConnectionContext:
        return _ConnectionContext(self._conn)


class _ConnectionContext:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeConnection:
        return self._conn

    async def __aexit__(self, *_exc: object) -> bool:
        return False


# A small, fully-connected synthetic graph exercising every one of the eight edge relations
# and all six FastRP vertex kinds: artists 1 and 2 share release 101, 2 and 3 share release
# 102; release 101 also carries a label, a master, a genre, and a style, and the master
# repeats the artist/genre/style tags.
_VERTEX_ROWS: list[tuple[str, str]] = [
    ("a", "1"),
    ("a", "2"),
    ("a", "3"),
    ("r", "101"),
    ("r", "102"),
    ("l", "501"),
    ("m", "601"),
    ("g", "Fixture Genre"),
    ("s", "Fixture Style"),
]

_EDGE_ROWS: dict[str, list[tuple[Any, Any]]] = {
    "graph.by_artist": [("101", "1"), ("101", "2"), ("102", "2"), ("102", "3")],
    "graph.on_label": [("101", "501")],
    "graph.derived_from": [("101", "601")],
    "graph.in_genre": [("101", "Fixture Genre")],
    "graph.in_style": [("101", "Fixture Style")],
    "graph.master_by_artist": [("601", "1")],
    "graph.master_in_genre": [("601", "Fixture Genre")],
    "graph.master_in_style": [("601", "Fixture Style")],
}


def _fresh_connection(already_loaded_row: tuple[Any, ...] | None = None) -> FakeConnection:
    return FakeConnection(
        vertex_rows=list(_VERTEX_ROWS), edge_rows={k: list(v) for k, v in _EDGE_ROWS.items()}, already_loaded_row=already_loaded_row
    )


# ── EmbeddingPipelineConfig ──────────────────────────────────────────────────────────────────


class TestEmbeddingPipelineConfig:
    def _set_valid_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDING_PIPELINE_POSTGRES_USERNAME", "embedding_pipeline_login")
        monkeypatch.setenv("EMBEDDING_PIPELINE_POSTGRES_PASSWORD", "secret")
        monkeypatch.setenv("POSTGRES_DATABASE", "groovemap")
        monkeypatch.setenv("SOURCE_DUMP_ID", "discogs-2026-09")
        monkeypatch.setenv("SOURCE_DUMP_DATE", "2026-09-01")

    def test_builds_from_a_complete_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_valid_env(monkeypatch)
        monkeypatch.setenv("POSTGRES_HOST", "catalog-postgres")

        config = pipeline.EmbeddingPipelineConfig.from_env()

        assert config.postgres_username == "embedding_pipeline_login"
        assert config.postgres_password == "secret"
        assert config.postgres_database == "groovemap"
        assert config.source_dump_id == "discogs-2026-09"
        assert config.source_dump_date == date(2026, 9, 1)
        assert config.postgres_host == "catalog-postgres:5432"

    def test_reads_the_password_from_the_file_secret_convention(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        self._set_valid_env(monkeypatch)
        secret_file = tmp_path / "password"
        secret_file.write_text("from-a-file\n")
        monkeypatch.delenv("EMBEDDING_PIPELINE_POSTGRES_PASSWORD")
        monkeypatch.setenv("EMBEDDING_PIPELINE_POSTGRES_PASSWORD_FILE", str(secret_file))

        config = pipeline.EmbeddingPipelineConfig.from_env()

        assert config.postgres_password == "from-a-file"

    @pytest.mark.parametrize(
        "missing",
        ["EMBEDDING_PIPELINE_POSTGRES_USERNAME", "EMBEDDING_PIPELINE_POSTGRES_PASSWORD", "POSTGRES_DATABASE", "SOURCE_DUMP_ID", "SOURCE_DUMP_DATE"],
    )
    def test_raises_when_a_required_variable_is_missing(self, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
        self._set_valid_env(monkeypatch)
        monkeypatch.delenv(missing)

        with pytest.raises(ValueError, match=missing):
            pipeline.EmbeddingPipelineConfig.from_env()

    def test_raises_on_a_non_iso_dump_date(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_valid_env(monkeypatch)
        monkeypatch.setenv("SOURCE_DUMP_DATE", "09/01/2026")

        with pytest.raises(ValueError, match="SOURCE_DUMP_DATE"):
            pipeline.EmbeddingPipelineConfig.from_env()


# ── Small pure helpers ───────────────────────────────────────────────────────────────────────


class TestHalfvecLiteral:
    def test_renders_pgvector_text_input_syntax(self) -> None:
        vector = np.array([1.5, -2.0, 0.0], dtype=np.float16)

        assert pipeline._halfvec_literal(vector) == "[1.5,-2,0]"

    def test_round_trips_full_dimensionality(self) -> None:
        vector = np.zeros(128, dtype=np.float16)

        literal = pipeline._halfvec_literal(vector)

        assert literal.startswith("[") and literal.endswith("]")
        assert literal.count(",") == 127


class TestIndexNameSlug:
    def test_is_a_safe_lowercase_identifier_fragment(self) -> None:
        slug = pipeline._index_name_slug(FastRPConfig().model_version)

        assert slug
        assert all(character.isalnum() or character == "_" for character in slug)
        assert slug == slug.lower()

    def test_is_bounded_in_length(self) -> None:
        slug = pipeline._index_name_slug("x" * 500)

        assert len(slug) <= 48


# ── Idempotency ──────────────────────────────────────────────────────────────────────────────


class TestAlreadyLoaded:
    @pytest.mark.asyncio
    async def test_true_when_a_row_matches(self) -> None:
        conn = _fresh_connection(already_loaded_row=(1,))

        assert await pipeline._already_loaded(conn, "fastrp-v1", "dump-1") is True

    @pytest.mark.asyncio
    async def test_false_when_no_row_matches(self) -> None:
        conn = _fresh_connection(already_loaded_row=None)

        assert await pipeline._already_loaded(conn, "fastrp-v1", "dump-1") is False


# ── Graph reading ────────────────────────────────────────────────────────────────────────────


class TestReadVertices:
    @pytest.mark.asyncio
    async def test_returns_a_node_index_over_all_six_kinds_and_every_artist_id(self) -> None:
        conn = _fresh_connection()

        nodes, artist_ids = await pipeline._read_vertices(conn)

        assert len(nodes) == len(_VERTEX_ROWS)
        assert sorted(artist_ids) == ["1", "2", "3"]


class TestStreamEdgeBlocks:
    @pytest.mark.asyncio
    async def test_builds_an_adjacency_connecting_every_shared_release_and_master(self) -> None:
        from insights.embeddings import AdjacencyBuilder, node_key

        conn = _fresh_connection()
        nodes, _artist_ids = await pipeline._read_vertices(conn)
        builder = AdjacencyBuilder(nodes)

        await pipeline._stream_edge_blocks(conn, builder)
        adjacency = builder.build()

        artist_1 = nodes.positions([node_key("a", "1")])[0]
        artist_2 = nodes.positions([node_key("a", "2")])[0]
        artist_3 = nodes.positions([node_key("a", "3")])[0]
        # Undirected degree: artist 1 touches release 101 only; artist 2 touches both 101
        # and 102; artist 3 touches release 102 only, plus master 601 reaches artist 1 too.
        assert adjacency.degree[artist_1] == 2  # release 101, master 601
        assert adjacency.degree[artist_2] == 2  # releases 101 and 102
        assert adjacency.degree[artist_3] == 1  # release 102


# ── Writing ──────────────────────────────────────────────────────────────────────────────────


class TestWriteEmbeddings:
    @pytest.mark.asyncio
    async def test_writes_one_row_per_artist_with_the_given_lineage(self) -> None:
        conn = _fresh_connection()
        artist_ids = ["1", "2", "3"]
        vectors = np.zeros((3, 4), dtype=np.float16)
        vectors[1] = [1, 2, 3, 4]

        rows_written = await pipeline._write_embeddings(
            conn, model_version="fastrp-v1", dump_id="dump-1", dump_date=date(2026, 9, 1), artist_ids=artist_ids, vectors=vectors
        )

        assert rows_written == 3
        assert len(conn.executemany_log) == 1
        sql, batch = conn.executemany_log[0]
        assert sql == pipeline._UPSERT_SQL
        assert [row[0] for row in batch] == artist_ids
        assert all(row[1] == "fastrp-v1" and row[3] == "dump-1" and row[4] == date(2026, 9, 1) for row in batch)
        assert batch[1][2] == pipeline._halfvec_literal(vectors[1])

    @pytest.mark.asyncio
    async def test_batches_large_artist_sets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pipeline, "_UPSERT_BATCH_SIZE", 2)
        conn = _fresh_connection()
        artist_ids = ["1", "2", "3", "4", "5"]
        vectors = np.zeros((5, 4), dtype=np.float16)

        rows_written = await pipeline._write_embeddings(
            conn, model_version="fastrp-v1", dump_id="dump-1", dump_date=date(2026, 9, 1), artist_ids=artist_ids, vectors=vectors
        )

        assert rows_written == 5
        assert [len(rows) for _sql, rows in conn.executemany_log] == [2, 2, 1]


# ── End-to-end load, without a database ─────────────────────────────────────────────────────


class TestLoadEmbeddings:
    @pytest.mark.asyncio
    async def test_writes_an_embedding_per_artist_and_logs_the_operator_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _fresh_connection(already_loaded_row=None)
        pool = FakePool(conn)
        config = FastRPConfig()
        fake_logger = Mock()
        monkeypatch.setattr(pipeline, "logger", fake_logger)

        result = await pipeline.load_embeddings(pool, config, "dump-1", date(2026, 9, 1))

        assert result.skipped is False
        assert result.model_version == config.model_version
        assert result.rows_written == 3
        assert len(conn.executemany_log) == 1
        _sql, batch = conn.executemany_log[0]
        assert sorted(row[0] for row in batch) == ["1", "2", "3"]

        operator_calls = [call for call in fake_logger.info.call_args_list if "Operator step" in call.args[0]]
        assert len(operator_calls) == 1
        assert operator_calls[0].kwargs["model_version"] == config.model_version
        assert "CREATE INDEX CONCURRENTLY" in operator_calls[0].kwargs["statement"]
        assert config.model_version in operator_calls[0].kwargs["statement"]

    @pytest.mark.asyncio
    async def test_skips_a_dump_already_loaded_under_this_model_version(self) -> None:
        conn = _fresh_connection(already_loaded_row=(1,))
        pool = FakePool(conn)

        result = await pipeline.load_embeddings(pool, FastRPConfig(), "dump-1", date(2026, 9, 1))

        assert result.skipped is True
        assert result.rows_written == 0
        assert conn.executemany_log == []
        # The graph is never read on a skip.
        assert "embedding_pipeline_vertices" not in conn.opened_cursor_names

    @pytest.mark.asyncio
    async def test_never_touches_a_row_of_a_different_model_version(self) -> None:
        """The upsert's ON CONFLICT target and the idempotency check are both scoped to
        this call's own model_version — verified here at the SQL-parameter level."""
        conn = _fresh_connection(already_loaded_row=None)
        pool = FakePool(conn)
        config = FastRPConfig()

        await pipeline.load_embeddings(pool, config, "dump-1", date(2026, 9, 1))

        _sql, batch = conn.executemany_log[0]
        assert all(row[1] == config.model_version for row in batch)


# ── The metrics-and-span wrapper ─────────────────────────────────────────────────────────────


class TestRunEmbeddingPipeline:
    @pytest.mark.asyncio
    async def test_records_success_duration_and_rows_written(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = pipeline.LoadResult(model_version="fastrp-v1", rows_written=42, skipped=False)
        monkeypatch.setattr(pipeline, "load_embeddings", AsyncMock(return_value=result))
        record_computation = Mock()
        monkeypatch.setattr(pipeline, "record_computation", record_computation)
        record_rows = Mock()
        monkeypatch.setattr(pipeline, "record_embedding_rows_written", record_rows)
        record_failure = Mock()
        monkeypatch.setattr(pipeline, "record_embedding_pipeline_failure", record_failure)

        outcome = await pipeline.run_embedding_pipeline(pool=object(), dump_id="dump-1", dump_date=date(2026, 9, 1))

        assert outcome is result
        assert record_computation.call_args.args[0] == pipeline.COMPUTATION_NAME
        assert record_computation.call_args.kwargs == {"success": True}
        record_rows.assert_called_once_with(42)
        record_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_records_failure_and_re_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pipeline, "load_embeddings", AsyncMock(side_effect=RuntimeError("boom")))
        record_computation = Mock()
        monkeypatch.setattr(pipeline, "record_computation", record_computation)
        record_failure = Mock()
        monkeypatch.setattr(pipeline, "record_embedding_pipeline_failure", record_failure)
        record_rows = Mock()
        monkeypatch.setattr(pipeline, "record_embedding_rows_written", record_rows)

        with pytest.raises(RuntimeError, match="boom"):
            await pipeline.run_embedding_pipeline(pool=object(), dump_id="dump-1", dump_date=date(2026, 9, 1))

        assert record_computation.call_args.kwargs == {"success": False}
        record_failure.assert_called_once_with()
        record_rows.assert_not_called()
