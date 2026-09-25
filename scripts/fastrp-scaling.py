"""Measure FastRP time and peak memory on synthetic catalog-shaped graphs.

Each size runs in its own subprocess so peak RSS is per run. The graphs are random
and synthetic: release nodes linked to artist, label, genre, style, and master nodes
with a skewed popularity, at the full catalog's ratios (32.8M nodes, at most 222M
undirected edges, 10.2M artists). No provider data is read.

    uv run python scripts/fastrp-scaling.py 500000 1000000 2000000 --block-columns 8 --half
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time


THREADS = os.environ.get("FASTRP_THREADS", "6")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, THREADS)

import numpy as np  # noqa: E402

from insights.embeddings import AdjacencyBuilder, NodeIndex, estimate_peak_bytes, fastrp  # noqa: E402


CATALOG_NODES = 32_800_000
CATALOG_EDGES = 222_000_000
CATALOG_ARTISTS = 10_200_000
RELEASE_SHARE = 19_417_067 / CATALOG_NODES
EDGE_BLOCK = 1 << 20


def peak_rss() -> int:
    # macOS reports ru_maxrss in bytes, Linux in KiB.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def run_one(n_nodes: int, block_columns: int, half: bool) -> dict[str, float]:
    rng = np.random.default_rng(n_nodes)
    n_edges = round(n_nodes * CATALOG_EDGES / CATALOG_NODES)
    n_releases = round(n_nodes * RELEASE_SHARE)
    n_artists = round(n_nodes * CATALOG_ARTISTS / CATALOG_NODES)
    started = time.perf_counter()
    keys = rng.integers(0, 2**64 - 1, size=n_nodes, dtype=np.uint64, endpoint=True)
    nodes = NodeIndex(keys)
    builder = AdjacencyBuilder(nodes)
    others = n_nodes - n_releases
    for start in range(0, n_edges, EDGE_BLOCK):
        size = min(EDGE_BLOCK, n_edges - start)
        releases = rng.integers(0, n_releases, size=size)
        targets = n_releases + (others * rng.random(size) ** 2).astype(np.int64)
        builder.add_edge_positions(releases, targets)
    adjacency = builder.build()
    built = time.perf_counter()
    build_rss = peak_rss()
    rows = np.arange(n_releases, n_releases + n_artists)
    embedding = fastrp(adjacency, rows=rows, block_columns=block_columns, out_dtype=np.float16 if half else np.float32, threads=int(THREADS))
    finished = time.perf_counter()
    estimate = estimate_peak_bytes(n_nodes, adjacency.undirected_edges, rows.size, block_columns=block_columns, out_itemsize=embedding.itemsize)
    return {
        "nodes": n_nodes,
        "undirected_edges": adjacency.undirected_edges,
        "rows": int(rows.size),
        "block_columns": block_columns,
        "threads": int(THREADS),
        "out_dtype": str(embedding.dtype),
        "build_s": round(built - started, 1),
        "fastrp_s": round(finished - built, 1),
        "build_peak_rss_gb": round(build_rss / 1e9, 3),
        "peak_rss_gb": round(peak_rss() / 1e9, 3),
        "estimated_peak_gb": round(estimate["peak"] / 1e9, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, nargs="+")
    parser.add_argument("--block-columns", type=int, default=8)
    parser.add_argument("--half", action="store_true", help="return float16, as stored in halfvec")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(run_one(args.nodes[0], args.block_columns, args.half)))
        return
    results = []
    for n in args.nodes:
        command = [sys.executable, __file__, str(n), "--block-columns", str(args.block_columns), "--child"]
        if args.half:
            command.append("--half")
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout  # noqa: S603 - fixed interpreter and arguments
        results.append(json.loads(output))
        print(output.strip(), flush=True)
    if len(results) > 1:
        small, large = results[0], results[-1]
        ratio = CATALOG_NODES / large["nodes"]
        per_node_rss = (large["peak_rss_gb"] - small["peak_rss_gb"]) / (large["nodes"] - small["nodes"])
        extrapolated = {
            "catalog_peak_rss_gb": round(large["peak_rss_gb"] + per_node_rss * (CATALOG_NODES - large["nodes"]), 1),
            "catalog_fastrp_s": round(large["fastrp_s"] * ratio),
            "catalog_build_s": round(large["build_s"] * ratio),
            "catalog_estimated_peak_gb": round(
                estimate_peak_bytes(
                    CATALOG_NODES, CATALOG_EDGES, CATALOG_ARTISTS, block_columns=args.block_columns, out_itemsize=2 if args.half else 4
                )["peak"]
                / 1e9,
                1,
            ),
        }
        print(json.dumps(extrapolated))


if __name__ == "__main__":
    main()
