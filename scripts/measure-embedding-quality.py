#!/usr/bin/env python3
"""Support script for docs/embedding_quality.md.

Two small, self-contained pieces used to re-run the design repo's gm-design-chw.2 proxy
benchmark against a shipped FastRP embeddings artifact (an `.npz` with `artist_ids` and
`vectors`, as produced by gm-analytics-engine-ieu.3) instead of the benchmark's own
from-scratch training run:

* `map-embeddings`: looks up each artist in the benchmark's rebuilt subset by Discogs id in
  the shipped `.npz`, and writes a local-index-aligned `.npy` for the benchmark's own
  `evaluate.py --emb`, plus a coverage/duplicate-vector-group side JSON (local indices only
  -- no ids, names, vectors, or edges).
* `dup-effect`: splits `evaluate.py`'s per-query test recall@10 (see its `--dump-per-query`)
  by whether the query artist's shipped vector sits in a byte-duplicate group -- a known
  FastRP artifact of iteration weights that give zero weight to a node's own projection.

Run these from a copy of the design repo's docs/spikes/gm-design-chw.2/ harness (outside any
repository, per that spike's own README) -- `map-embeddings` imports that harness's
`catalog.py`. Nothing here talks to a database, a registry, or the network; it only reads
local files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def dup_groups(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group sizes and unique-group ids for exact byte-duplicate rows."""
    v = np.ascontiguousarray(vectors)
    flat = v.view(np.dtype((np.void, v.dtype.itemsize * v.shape[1]))).reshape(-1)
    _uniq, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    return counts[inv], inv


def cmd_map_embeddings(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.harness_dir))
    from catalog import load_subset  # noqa: PLC0415 -- only importable once harness_dir is on sys.path

    full, _pre, _seeds = load_subset(args.subset)
    local_ids = full.artist_ids.astype(np.int64)  # ascending; local index == position here

    # allow_pickle: artist_ids is dtype=object (numeric-id strings) in this npz, a trusted
    # local artifact of our own pipeline, not untrusted input.
    npz = np.load(args.npz, allow_pickle=True)
    shipped_ids = npz["artist_ids"].astype(np.int64)
    shipped_vecs = npz["vectors"].astype(np.float32)
    group_size, _group_id = dup_groups(npz["vectors"])

    order = np.argsort(shipped_ids)
    shipped_ids_sorted = shipped_ids[order]
    pos_in_shipped = np.clip(np.searchsorted(shipped_ids_sorted, local_ids), 0, len(shipped_ids_sorted) - 1)
    found = shipped_ids_sorted[pos_in_shipped] == local_ids
    orig_row = order[pos_in_shipped]

    dim = shipped_vecs.shape[1]
    out = np.zeros((len(local_ids), dim), dtype=np.float32)
    out[found] = shipped_vecs[orig_row[found]]

    dup_size_per_local = np.zeros(len(local_ids), dtype=np.int32)
    dup_size_per_local[found] = group_size[orig_row[found]]

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, out)
    meta = {
        "model_version": str(npz["model_version"]),
        "dump_id": str(npz["dump_id"]),
        "subset_artists": len(local_ids),
        "subset_artists_with_shipped_vector": int(found.sum()),
        "coverage": float(found.mean()),
        "shipped_file_total_vectors": len(shipped_ids),
        "shipped_file_dup_group_pct": float((group_size > 1).mean()),
        "dup_size_per_local_artist": dup_size_per_local.tolist(),
        "has_vector_per_local_artist": found.tolist(),
    }
    args.out_meta.write_text(json.dumps(meta))
    print(json.dumps({k: v for k, v in meta.items() if not isinstance(v, list)}, indent=2))


def summarize(recalls: np.ndarray, rng: np.random.Generator) -> dict:
    if len(recalls) == 0:
        return {"queries": 0}
    boots = rng.integers(0, len(recalls), size=(2000, len(recalls)))
    means = recalls[boots].mean(axis=1)
    return {
        "queries": len(recalls),
        "recall_at_10": float(recalls.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def cmd_dup_effect(args: argparse.Namespace) -> None:
    per_query = json.loads(args.per_query_json.read_text())[args.name]
    meta = json.loads(args.meta_json.read_text())
    has_vector = np.array(meta["has_vector_per_local_artist"], dtype=bool)
    dup_size = np.array(meta["dup_size_per_local_artist"], dtype=np.int32)

    idx = np.array([r["artist_idx"] for r in per_query], dtype=np.int64)
    recall = np.array([r["recall_at_10"] for r in per_query], dtype=np.float64)
    q_has_vector = has_vector[idx]
    q_dup_size = dup_size[idx]

    rng = np.random.default_rng(0)
    print(
        json.dumps(
            {
                "all_queries": summarize(recall, rng),
                "query_has_shipped_vector": summarize(recall[q_has_vector], rng),
                "query_missing_shipped_vector": summarize(recall[~q_has_vector], rng),
                "query_vector_in_dup_group": summarize(recall[q_has_vector & (q_dup_size > 1)], rng),
                "query_vector_unique": summarize(recall[q_has_vector & (q_dup_size == 1)], rng),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("map-embeddings", help="map a shipped artist_ids+vectors npz onto the benchmark's local index")
    p1.add_argument("harness_dir", type=Path, help="path to a checkout of the gm-design-chw.2 harness (has catalog.py)")
    p1.add_argument("subset", type=Path, help="the harness's build_subset.py output directory")
    p1.add_argument("npz", type=Path, help="shipped embeddings npz (artist_ids, vectors)")
    p1.add_argument("out_npy", type=Path)
    p1.add_argument("out_meta", type=Path)
    p1.set_defaults(func=cmd_map_embeddings)

    p2 = sub.add_parser("dup-effect", help="split per-query recall@10 by duplicate-vector-group membership")
    p2.add_argument("per_query_json", type=Path, help="evaluate.py --dump-per-query-out output")
    p2.add_argument("meta_json", type=Path, help="map-embeddings' out_meta output")
    p2.add_argument("--name", default="shipped")
    p2.set_defaults(func=cmd_dup_effect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
