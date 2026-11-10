"""Attribute the embedding pipeline's peak memory by phase on a synthetic, catalog-shaped graph.

Runs the real ``_read_vertices``, ``_stream_edge_blocks``, ``_assemble``, ``fastrp`` and
``_write_embeddings`` against an in-memory stand-in for the PostgreSQL connection, sampling
process RSS every 50 ms and recording the high-water mark of each phase. Synthetic and
read-only: no database or provider data is touched.

    uv run python scripts/profile-embedding-pipeline.py 0.1     # 0.1 x the 2026-11 lux catalog
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from typing import Any

import numpy as np
import psutil

from insights import embedding_pipeline as ep
from insights.embeddings import AdjacencyBuilder, FastRPConfig, NodeIndex, estimate_peak_bytes, fastrp, node_keys


# 2026-11 lux load: 9,152,168 artist rows (docs/embeddings.md); nodes/edges per the x3d estimate.
LUX_NODES = 31_271_244
LUX_EDGES = 243_870_971
LUX_ARTISTS = 9_152_168


class Sampler:
    def __init__(self) -> None:
        self._proc = psutil.Process()
        self.phase = "start"
        self.peaks: dict[str, int] = {}
        self.ends: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            self.peaks[self.phase] = max(self.peaks.get(self.phase, 0), rss)
            time.sleep(0.05)

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def enter(self, phase: str) -> None:
        self.ends[self.phase] = self._proc.memory_info().rss
        self.phase = phase


class FakeCursor:
    def __init__(self, tables: dict[str, Any]) -> None:
        self._tables = tables
        self._rows: Any = iter(())

    async def __aenter__(self) -> FakeCursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, query: str, params: Any = None) -> None:  # noqa: ARG002
        self._rows = self._tables[query]()

    async def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        return [row for _, row in zip(range(size), self._rows, strict=False)]


class FakeConn:
    def __init__(self, tables: dict[str, Any]) -> None:
        self._tables = tables

    def transaction(self) -> _NullAsync:
        return _NullAsync()

    def cursor(self, name: str | None = None) -> FakeCursor:  # noqa: ARG002
        return FakeCursor(self._tables)


class _NullAsync:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> None:
        return None


async def main(scale: float) -> None:
    rng = np.random.default_rng(7)
    nodes_n = round(LUX_NODES * scale)
    edges_n = round(LUX_EDGES * scale)
    artists_n = round(LUX_ARTISTS * scale)
    releases_n = round(nodes_n * 0.6)
    others_n = max(nodes_n - releases_n - artists_n, 1)
    relations = ep._EDGE_RELATIONS
    per_relation = edges_n // len(relations)

    def vertices() -> Any:
        for i in range(artists_n // 2):
            yield ("a", str(100_000_000 + i))
        for i in range(releases_n):
            yield ("r", str(i))
        for kind in "lmgs":
            for i in range(others_n // 4):
                yield (kind, str(i))

    def edges_for(source_kind: str, target_kind: str) -> Any:
        def make() -> Any:
            for start in range(0, per_relation, 1 << 17):
                size = min(1 << 17, per_relation - start)
                src = rng.integers(0, releases_n if source_kind == "r" else others_n // 4, size)
                if target_kind == "a":
                    dst = 100_000_000 + (artists_n * rng.random(size) ** 1.2).astype(np.int64)
                else:
                    dst = ((others_n // 4) * rng.random(size) ** 2).astype(np.int64)
                yield from zip(src.tolist(), dst.tolist(), strict=True)

        return make

    tables: dict[str, Any] = {"SELECT kind, key FROM graph.vertex_degree WHERE kind = ANY(%s)": vertices}
    patched_relations = []
    for name, _query, params, source_kind, target_kind in relations:
        tables[name] = edges_for(source_kind, target_kind)
        patched_relations.append((name, name, params, source_kind, target_kind))
    ep._EDGE_RELATIONS = tuple(patched_relations)  # type: ignore[misc]
    ep._DISCOVERY_QUERIES = ()  # type: ignore[misc]
    conn: Any = FakeConn(tables)

    with Sampler() as sampler:
        base = psutil.Process().memory_info().rss
        sampler.enter("read_vertices")
        vertices_set = await ep._read_vertices(conn)
        sampler.enter("stream_edges")
        blocks = await ep._stream_edge_blocks(conn, vertices_set)
        sampler.enter("assemble")
        nodes = NodeIndex(np.unique(np.concatenate(vertices_set.key_chunks)))
        sampler.enter("assemble_positions")
        builder = AdjacencyBuilder(nodes)
        while blocks:
            sources, targets = blocks.pop()
            builder.add_edges(sources, targets)
        del sources, targets
        sampler.enter("assemble_build")
        artist_ids = vertices_set.artist_ids
        adjacency = builder.build()
        del vertices_set, blocks
        sampler.enter("fastrp")
        positions = nodes.positions(node_keys(("a", a) for a in artist_ids))
        vectors = fastrp(adjacency, FastRPConfig(self_weight=0.05), rows=positions, out_dtype=np.float16, threads=6)
        sampler.enter("write")
        words = ep._halfvec_binary_rows
        for start in range(0, len(artist_ids), ep._COPY_BATCH_SIZE):
            words(vectors[start : start + ep._COPY_BATCH_SIZE])
        sampler.enter("done")
    n, e, a = len(nodes), adjacency.undirected_edges, len(artist_ids)
    estimate = estimate_peak_bytes(n, e, a, FastRPConfig(self_weight=0.05), out_itemsize=2)
    print(
        json.dumps(
            {
                "scale": scale,
                "nodes": n,
                "undirected_edges": e,
                "streamed_edges": per_relation * len(relations),
                "artist_rows": a,
                "baseline_rss": base,
                "phase_peak_rss": sampler.peaks,
                "phase_end_rss": sampler.ends,
                "estimate": estimate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.02))
