"""Measure tie-tolerant ANN recall@10 on the current (September) real embeddings, for
gm-analytics-engine-kn3 -- maintainer decision (D option A, step 1) following up on
gm-analytics-engine-ieu.3's finding that 38.7% of vectors are exact byte-duplicates and
17.5% of the 2,000-query recall sample has an exact tie at/adjacent to the 10th-place
exact-cosine score, so strict recall@10 (0.8405 at ef_search=1000, September) undercounts
equally-correct neighbours.

Reuses ieu.3's own harness (`scripts/measure_recall_churn.py`, imported as a sibling
module, not copied) for everything that harness already gets right: the real
`artist_embeddings` DDL, `_write_embeddings`, the per-`model_version` **partial** HNSW
index via `_index_name`/`_sql_string_literal`, the deterministic 2,000-query sample
(`QUERY_SAMPLE_SEED = 0x51ECA11A = 1_374_462_234`, the same seed named in the bead), and
`ef_search` self-exclusion (`k + 1` rows, own `artist_id` dropped client-side).

This script is September-only (`~/.cache/groovemap-spikes/embeddings-scratch/sept.npz`,
read-only -- never modified or deleted, another bead also reads that directory) -- it does
not repeat August or the churn measurement, both already recorded in
`docs/recall_and_churn.md`. It adds three things ieu.3's own run started but did not
finish or did not attempt:

1. **Tie-tolerant recall@10.** For every ef_search in {40,100,200,400,800,1000}: strict
   recall@10 (should reproduce ieu.3's 0.8405) alongside tie-tolerant recall@10, where an
   ANN-returned candidate counts as a hit if its own EXACT cosine similarity to the query
   is >= the query's exact 10th-place cosine score - 1e-4 -- not merely "is one of the
   exact top 10 ids", which is exactly what strict recall requires and what a tied
   11th/12th/... place candidate fails even though it is equally correct.
2. **Breakdown by duplicate-group size.** Each query's own exact-duplicate-group size
   (computed once, in NumPy, over all of September's vectors -- a byte-identical vector
   compare, the same notion ieu.3's write-up already found 38.7%/753,588 groups on
   August), bucketed into {1 (unique), 2-4, 5-20, 21+}.
3. **Over-fetch + exact re-rank.** For k' in {50,100,200} at ef_search in {100,200,400}:
   fetch k' ANN candidates, re-rank them by EXACT cosine (both months' vectors already
   live in memory -- no second Postgres round trip needed for the exact side), take the
   top 10 of that re-ranked list, and report strict/tie-tolerant recall@10 on it plus
   per-query latency (the ANN fetch + re-rank wall time).

**Not attempted: breakdown by the query artist's graph degree.** True vertex degree
(`insights.embeddings.graph.Adjacency.degree`) is only available as a byproduct of
`scripts/embeddings_from_dump.py`'s full graph build, which needs September's
`releases.xml.gz` dump. Only `artists`/`masters`/`labels` (~1.1 GB combined) are cached
locally; `releases.xml.gz` (19.4M records, vs. masters' 597 MB for 2.6M) is not, and this
host's disk floor was already breached by other concurrent work before this bead started
(see the dispatcher's own note) -- downloading a multi-GB dump to compute it would make
that worse, not better. Flagged to the dispatcher rather than silently faking or
approximating a degree metric; see "Degree-bucket gap" in `docs/recall_and_churn.md`.

    uv run python scripts/measure_tie_tolerant_recall.py \\
        ~/.cache/groovemap-spikes/embeddings-scratch/sept.npz \\
        --out ~/.cache/groovemap-spikes/embeddings-scratch/tie_tolerant_recall.json

This script starts and removes its own throwaway container (unlike
`measure_recall_churn.py`, which connects to one the caller already started) -- no
``--host``/``--port`` to point at one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

import measure_recall_churn as base  # sibling script, same directory -- see module docstring.
import numpy as np


# sept.npz was computed under ieu.6's edge set (`_EDGE_SET_VERSION = "edges-v2"` at the
# time) -- but `base._load_month` composes its `model_version` via
# `insights.embedding_pipeline.stored_model_version`, which reads that module's *current*
# `_EDGE_SET_VERSION`, a shared constant that gm-analytics-engine-x3d has since bumped to
# "edges-v3" (it adds track-credited-artist/track-performer relations neither sept.npz nor
# this measurement's graph ever had). Left alone, this script would silently mislabel every
# row it writes -- the same npz vectors, stamped with a `model_version` claiming a graph
# they were never computed against. Pinned explicitly here instead of trusted from the
# live (still-evolving) module constant; must match the value already recorded for
# September in `recall_churn.json`/`docs/recall_and_churn.md`.
_PINNED_EDGE_SET_VERSION: Final = "edges-v2"

TIE_TOLERANCE: Final = 1e-4
OVERFETCH_KPRIME_SWEEP: tuple[int, ...] = (50, 100, 200)
OVERFETCH_EF_SEARCH_SWEEP: tuple[int, ...] = (100, 200, 400)
# (lower, upper_inclusive_or_None, label) -- matches the "21+" example already called out
# in docs/recall_and_churn.md's tie-structure finding.
DUP_GROUP_BUCKETS: tuple[tuple[int, int | None, str], ...] = (
    (1, 1, "1 (unique)"),
    (2, 4, "2-4"),
    (5, 20, "5-20"),
    (21, None, "21+"),
)


def _bucket_dup_group_size(size: int) -> str:
    for lower, upper, label in DUP_GROUP_BUCKETS:
        if size >= lower and (upper is None or size <= upper):
            return label
    raise AssertionError(f"unbucketed duplicate-group size {size}")  # pragma: no cover -- buckets are exhaustive from 1 up.


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-normalize VECTORS to unit cosine length, upcast to float32 -- the same upcast
    `measure_recall_churn._exact_top_k` applies before normalizing (pgvector's own HNSW
    distance stays in `halfvec`/float16 throughout; the exact side deliberately does not,
    per ieu.3's own tie-structure finding about why the two paths can disagree on ties)."""
    v32 = vectors.astype(np.float32)
    norms = np.linalg.norm(v32, axis=1)
    norms[norms == 0] = 1.0
    return v32 / norms[:, None]


def _duplicate_group_sizes(vectors: np.ndarray) -> np.ndarray:
    """Per-row exact-duplicate-group size (including the row itself), byte-identical
    compare -- the same notion ieu.3's write-up already used to find 38.7% duplicates on
    August (753,588 groups, largest 818 members)."""
    contiguous = np.ascontiguousarray(vectors)
    structured = contiguous.view([("", contiguous.dtype)] * contiguous.shape[1]).reshape(-1)
    _, inverse, counts = np.unique(structured, return_inverse=True, return_counts=True)
    return counts[inverse]


def _exact_top10_with_threshold(normalized: np.ndarray, query_positions: list[int]) -> tuple[list[list[int]], list[float]]:
    """Per query position: the exact top-10 positions (excluding self) and the 10th-place
    (smallest of the top 10) exact cosine score -- the tie-tolerance threshold."""
    top_positions: list[list[int]] = []
    thresholds: list[float] = []
    for position in query_positions:
        scores = normalized @ normalized[position]
        scores[position] = -np.inf  # exclude self, same convention as base._exact_top_k.
        top = np.argpartition(-scores, 10)[:10]
        top = top[np.argsort(-scores[top])]
        top_positions.append(top.tolist())
        thresholds.append(float(scores[top[-1]]))
    return top_positions, thresholds


def _tie_tolerant_hits(
    candidate_ids: list[str], id_to_position: dict[str, int], normalized: np.ndarray, query_position: int, threshold: float
) -> int:
    """How many of CANDIDATE_IDS have an exact cosine to the query >= THRESHOLD - TIE_TOLERANCE."""
    if not candidate_ids:
        return 0
    positions = np.fromiter((id_to_position[cid] for cid in candidate_ids), dtype=np.int64, count=len(candidate_ids))
    scores = normalized[positions] @ normalized[query_position]
    return int(np.count_nonzero(scores >= threshold - TIE_TOLERANCE))


async def _ann_top_k_timed(
    conn: Any, model_version: str, vectors: np.ndarray, artist_ids: list[str], positions: list[int], ef_search: int, k: int
) -> tuple[list[list[str]], list[float]]:
    """Like `base._ann_top_k`, but also returns each query's own wall-clock latency (the
    single `SELECT ... ORDER BY ... LIMIT` round trip) -- needed for the over-fetch +
    re-rank section's per-query latency numbers. Same self-exclusion convention: request
    `k + 1` rows under only the `model_version` filter, drop the query's own artist_id
    client-side (see `base._ann_top_k`'s own docstring for why not `WHERE artist_id != %s`)."""
    results: list[list[str]] = []
    latencies: list[float] = []
    async with conn.cursor() as cursor:
        await cursor.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        for position in positions:
            literal = "[" + ",".join(f"{value:g}" for value in vectors[position].tolist()) + "]"
            started = time.perf_counter()
            await cursor.execute(
                f"SELECT artist_id FROM {base.ARTIST_EMBEDDINGS_TABLE} "  # noqa: S608
                f"WHERE model_version = %s ORDER BY embedding <=> %s::halfvec LIMIT %s",
                (model_version, literal, k + 1),
            )
            rows = await cursor.fetchall()
            latencies.append(time.perf_counter() - started)
            own_id = artist_ids[position]
            results.append([artist_id for (artist_id,) in rows if artist_id != own_id][:k])
    return results, latencies


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _start_container(*, name: str, image: str, shm_size: str, username: str, password: str, database: str) -> tuple[str, int]:
    """Start this bead's own throwaway container (`--rm`, so a later `docker stop` also
    removes it) -- deliberately not `base._restart_container` (that helper assumes an
    existing container of the same name to stop first; this script only ever starts one).
    `-c max_parallel_maintenance_workers=4` on the postgres command line, per the
    dispatcher's ask (parallel HNSW builds), alongside the session-only
    `maintenance_work_mem` `base._build_index` sets for the CREATE INDEX call itself."""
    subprocess.run(  # noqa: S603 -- DOCKER/name/image are this script's own constants/args, never attacker-controlled.
        [
            base.DOCKER,
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--publish",
            "127.0.0.1::5432",
            "--shm-size",
            shm_size,
            "--env",
            f"POSTGRES_USER={username}",
            "--env",
            f"POSTGRES_PASSWORD={password}",
            "--env",
            f"POSTGRES_DB={database}",
            image,
            "postgres",
            "-c",
            "max_parallel_maintenance_workers=4",
        ],
        check=True,
        capture_output=True,
    )
    for _attempt in range(60):
        ready = subprocess.run(  # noqa: S603
            [base.DOCKER, "exec", name, "pg_isready", "--username", username, "--dbname", database],
            capture_output=True,
        )
        if ready.returncode == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"container {name!r} did not become ready within 120s")
    published = subprocess.run([base.DOCKER, "port", name, "5432/tcp"], check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603
    host, _, port = published.rpartition(":")
    return host or "127.0.0.1", int(port)


def _stop_container(name: str) -> None:
    subprocess.run([base.DOCKER, "stop", name], check=False, capture_output=True)  # noqa: S603 -- best-effort cleanup.


def _month_with_pinned_edge_set(month: dict[str, Any]) -> dict[str, Any]:
    """Override `_load_month`'s `model_version` to use `_PINNED_EDGE_SET_VERSION` instead
    of whatever `insights.embedding_pipeline._EDGE_SET_VERSION` currently is -- see the
    module-level pitfall note above `_PINNED_EDGE_SET_VERSION`."""
    corrected = f"{month['method_version']}:{_PINNED_EDGE_SET_VERSION}@{month['dump_id']}"
    return {**month, "model_version": corrected}


async def measure(args: argparse.Namespace) -> dict[str, Any]:
    print(f"loading {args.sept}", file=sys.stderr)
    month = _month_with_pinned_edge_set(base._load_month(args.sept))
    print(f"  model_version (edge set pinned to {_PINNED_EDGE_SET_VERSION!r}): {month['model_version']}", file=sys.stderr)
    artist_ids = month["artist_ids"]
    vectors = month["vectors"]
    id_to_position = {aid: index for index, aid in enumerate(artist_ids)}

    query_ids = base._deterministic_sample(artist_ids, base.QUERY_SAMPLE_SIZE, base.QUERY_SAMPLE_SEED)
    query_positions = [id_to_position[aid] for aid in query_ids]
    print(f"query sample: {len(query_positions):,} artists (seed {base.QUERY_SAMPLE_SEED:#x} = {base.QUERY_SAMPLE_SEED})", file=sys.stderr)

    print("computing exact top-10 + tie thresholds (NumPy, no Postgres)...", file=sys.stderr)
    normalized = _normalize(vectors)
    exact_top10_positions, thresholds = _exact_top10_with_threshold(normalized, query_positions)
    exact_top10_ids = [[artist_ids[p] for p in row] for row in exact_top10_positions]

    print("computing duplicate-group sizes (NumPy, no Postgres)...", file=sys.stderr)
    dup_sizes = _duplicate_group_sizes(vectors)
    n_duplicates = int(np.count_nonzero(dup_sizes > 1))
    query_dup_bucket = [_bucket_dup_group_size(int(dup_sizes[p])) for p in query_positions]
    print(
        f"  {n_duplicates:,} of {len(artist_ids):,} vectors ({n_duplicates / len(artist_ids):.1%}) are exact duplicates of at least one other",
        file=sys.stderr,
    )

    print(f"\n=== starting container {args.container_name!r} ({args.image}, --shm-size {args.shm_size}) ===", file=sys.stderr)
    host, port = _start_container(
        name=args.container_name, image=args.image, shm_size=args.shm_size, username=args.username, password=args.password, database=args.database
    )
    print(f"  ready at {host}:{port}", file=sys.stderr)

    try:
        pool = base.AsyncPostgreSQLPool(
            connection_params={"host": host, "port": port, "dbname": args.database, "user": args.username, "password": args.password},
            min_connections=1,
            max_connections=1,
        )
        await pool.initialize()
        try:
            async with pool.connection() as conn:
                await base._apply_schema(conn)

                print(f"\n=== writing {len(artist_ids):,} September rows ===", file=sys.stderr)
                write_result = await base._write_month(conn, month)
                print(f"  wrote {write_result['rows_written']:,} rows", file=sys.stderr)

                print(
                    f"\n=== building HNSW index (m={base.HNSW_M}, ef_construction={base.HNSW_EF_CONSTRUCTION}, maintenance_work_mem={args.maintenance_work_mem}) ===",
                    file=sys.stderr,
                )
                index_result = await base._build_index(conn, month["model_version"], args.maintenance_work_mem)
                print(f"  {index_result['index_name']}: {index_result['build_elapsed_s']:.1f}s", file=sys.stderr)

                print("\n=== recall@10 sweep (strict + tie-tolerant, by dup-group bucket) ===", file=sys.stderr)
                sweep: dict[int, dict[str, Any]] = {}
                for ef_search in base.EF_SEARCH_SWEEP:
                    ann_ids_by_query, _latencies = await _ann_top_k_timed(
                        conn, month["model_version"], vectors, artist_ids, query_positions, ef_search, 10
                    )
                    strict_scores: list[float] = []
                    tie_scores: list[float] = []
                    by_dup_bucket: dict[str, list[tuple[float, float]]] = {label: [] for _, _, label in DUP_GROUP_BUCKETS}
                    for i, ann_ids in enumerate(ann_ids_by_query):
                        strict_hit = len(set(ann_ids) & set(exact_top10_ids[i])) / 10.0
                        tie_hit = _tie_tolerant_hits(ann_ids, id_to_position, normalized, query_positions[i], thresholds[i]) / 10.0
                        strict_scores.append(strict_hit)
                        tie_scores.append(tie_hit)
                        by_dup_bucket[query_dup_bucket[i]].append((strict_hit, tie_hit))
                    sweep[ef_search] = {
                        "strict_recall_at_10": sum(strict_scores) / len(strict_scores),
                        "tie_tolerant_recall_at_10": sum(tie_scores) / len(tie_scores),
                        "by_dup_group_bucket": {
                            label: {
                                "n_queries": len(pairs),
                                "strict_recall_at_10": sum(p[0] for p in pairs) / len(pairs) if pairs else None,
                                "tie_tolerant_recall_at_10": sum(p[1] for p in pairs) / len(pairs) if pairs else None,
                            }
                            for label, pairs in by_dup_bucket.items()
                        },
                    }
                    print(
                        f"  ef_search={ef_search}: strict={sweep[ef_search]['strict_recall_at_10']:.4f} "
                        f"tie_tolerant={sweep[ef_search]['tie_tolerant_recall_at_10']:.4f}",
                        file=sys.stderr,
                    )

                print("\n=== over-fetch + exact re-rank ===", file=sys.stderr)
                overfetch: dict[str, dict[str, Any]] = {}
                for ef_search in OVERFETCH_EF_SEARCH_SWEEP:
                    for kprime in OVERFETCH_KPRIME_SWEEP:
                        key = f"ef_search={ef_search},k_prime={kprime}"
                        candidates_by_query, latencies = await _ann_top_k_timed(
                            conn, month["model_version"], vectors, artist_ids, query_positions, ef_search, kprime
                        )
                        strict_scores = []
                        tie_scores = []
                        rerank_latencies: list[float] = []
                        for i, candidate_ids in enumerate(candidates_by_query):
                            rerank_started = time.perf_counter()
                            if candidate_ids:
                                candidate_positions = np.fromiter(
                                    (id_to_position[cid] for cid in candidate_ids), dtype=np.int64, count=len(candidate_ids)
                                )
                                candidate_scores = normalized[candidate_positions] @ normalized[query_positions[i]]
                                order = np.argsort(-candidate_scores)[:10]
                                reranked_ids = [candidate_ids[j] for j in order]
                                reranked_scores = candidate_scores[order]
                            else:
                                reranked_ids, reranked_scores = [], np.zeros(0)
                            rerank_latencies.append(latencies[i] + (time.perf_counter() - rerank_started))
                            strict_scores.append(len(set(reranked_ids) & set(exact_top10_ids[i])) / 10.0)
                            tie_hit_count = int(np.count_nonzero(reranked_scores >= thresholds[i] - TIE_TOLERANCE))
                            tie_scores.append(tie_hit_count / 10.0)
                        overfetch[key] = {
                            "ef_search": ef_search,
                            "k_prime": kprime,
                            "strict_recall_at_10": sum(strict_scores) / len(strict_scores),
                            "tie_tolerant_recall_at_10": sum(tie_scores) / len(tie_scores),
                            "per_query_latency_s": _percentiles(rerank_latencies),
                        }
                        print(
                            f"  ef_search={ef_search} k'={kprime}: strict={overfetch[key]['strict_recall_at_10']:.4f} "
                            f"tie_tolerant={overfetch[key]['tie_tolerant_recall_at_10']:.4f} "
                            f"mean_latency_ms={overfetch[key]['per_query_latency_s']['mean'] * 1000:.2f}",
                            file=sys.stderr,
                        )
        finally:
            await pool.close()
    finally:
        print(f"\n=== removing container {args.container_name!r} ===", file=sys.stderr)
        _stop_container(args.container_name)

    return {
        "dump_id": month["dump_id"],
        "dump_date": month["dump_date"],
        "model_version": month["model_version"],
        "n_vectors": len(artist_ids),
        "query_sample_seed": base.QUERY_SAMPLE_SEED,
        "query_sample_size": len(query_positions),
        "tie_tolerance": TIE_TOLERANCE,
        "n_exact_duplicate_vectors": n_duplicates,
        "write": write_result,
        "index": index_result,
        "recall_sweep": {str(k): v for k, v in sweep.items()},
        "overfetch_rerank": overfetch,
        "degree_bucket_breakdown": None,
        "degree_bucket_gap_reason": (
            "true graph degree needs September's releases.xml.gz (not cached locally; only "
            "artists/masters/labels are) and this host's disk floor was already breached by "
            "other concurrent work -- see docs/recall_and_churn.md 'Degree-bucket gap'"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sept", type=Path)
    parser.add_argument("--database", default="groovemap")
    parser.add_argument("--username", default="groovemap")
    parser.add_argument("--password", default="integration-test-password")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--container-name", default="gm-analytics-engine-kn3-pg")
    parser.add_argument("--image", default="database-schema-postgres19-pgvector:local")
    parser.add_argument("--shm-size", default="9g")
    parser.add_argument("--maintenance-work-mem", default="8GB")
    args = parser.parse_args()

    result = asyncio.run(measure(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
