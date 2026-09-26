"""Measure real-embedding ANN recall@10 and month-over-month churn, for
gm-analytics-engine-ieu.3 -- the two open ADR 0013 preconditions.

Takes the compact ``(artist_id, vector)`` ``.npz`` files `scripts/embeddings_from_dump.py`
produced for two consecutive months and, against a throwaway PostgreSQL 19 + pgvector
container (one month's rows in an otherwise-empty `public.artist_embeddings` table at a
time -- the real DDL from `database-schema`, not a stand-in):

- writes the month's rows through the real `insights.embedding_pipeline._write_embeddings`
  (falling back to `COPY` only if a timed trial extrapolates past ~30 minutes, per the
  maintainer's condition -- see `_write_month`),
- builds the real HNSW index, named by `insights.embedding_pipeline._index_name`, under a
  session-only 2 GB `maintenance_work_mem` bump (full scale first; a subset fallback is
  used only after a real failure),
- sweeps recall@10 against exact cosine (computed in NumPy directly from the in-memory
  vectors, no Postgres needed for the exact side) over `ef_search` in `EF_SEARCH_SWEEP`,
  naming the smallest value reaching recall@10 >= 0.95 (or stating that none does), and
- computes each month's top-10 (both exact and, at the named production `ef_search`, via
  the live ANN index) for a deterministic sample of artists common to both months, so
  churn can be reported both on exact vectors and on what the served index would actually
  return.

Sampling is deterministic and repeatable: `_deterministic_sample` ranks candidates by
`splitmix64(node_key("a", artist_id) XOR seed)` -- the same hash the pinned FastRP
projection itself uses -- and takes the smallest `n`. Two fixed seeds (`QUERY_SAMPLE_SEED`,
`CHURN_SAMPLE_SEED`) are recorded here so a re-run picks the identical sample.

No provider-derived data is committed: this script reads the local `.npz` scratch files
(themselves gitignored/uncommitted) and writes only aggregate JSON to stdout.

    uv run python scripts/measure_recall_churn.py \\
        ~/.cache/groovemap-spikes/embeddings-scratch/aug.npz \\
        ~/.cache/groovemap-spikes/embeddings-scratch/sept.npz \\
        --host 127.0.0.1 --port 5432 --database groovemap \\
        --username groovemap --password integration-test-password \\
        --out ~/.cache/groovemap-spikes/embeddings-scratch/recall_churn.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from common import AsyncPostgreSQLPool
from insights.embedding_pipeline import ARTIST_EMBEDDINGS_TABLE, _index_name, _write_embeddings
from insights.embeddings.graph import node_key
from insights.embeddings.projection import splitmix64


QUERY_SAMPLE_SEED: int = 0x51EC_A11A  # "recall" sample seed, arbitrary but fixed.
CHURN_SAMPLE_SEED: int = 0xC8027A11  # "churn" sample seed, arbitrary but fixed.
QUERY_SAMPLE_SIZE: int = 2_000
CHURN_SAMPLE_SIZE: int = 10_000
EF_SEARCH_SWEEP: tuple[int, ...] = (40, 100, 200, 400, 800, 1000)  # pgvector caps hnsw.ef_search at 1000.
RECALL_TARGET: float = 0.95
HNSW_M: int = 16
HNSW_EF_CONSTRUCTION: int = 64
MAINTENANCE_WORK_MEM: str = "2GB"
TRIAL_BATCH_ROWS: int = 100_000
TRIAL_TIME_BUDGET_S: float = 30 * 60  # 30 minutes, per the maintainer's condition.

_VECTOR_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

# Copied verbatim from database-schema's `_ARTIST_EMBEDDINGS_STATEMENT` /
# `_ARTIST_EMBEDDINGS_MODEL_VERSION_INDEX` (src/groovemap_schema/postgres.py) -- the real
# landed DDL (gm-database-schema-lhp2), not the integration tier's inline stand-in (that
# stand-in predates lhp2 landing on origin/main and is now stale).
_ARTIST_EMBEDDINGS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS public.artist_embeddings (
        artist_id        TEXT NOT NULL,
        model_version    TEXT NOT NULL,
        embedding        halfvec(128) NOT NULL,
        source_dump_id   TEXT NOT NULL,
        source_dump_date DATE NOT NULL,
        computed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (artist_id, model_version)
    )
"""
_ARTIST_EMBEDDINGS_MODEL_VERSION_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_artist_embeddings_model_version ON public.artist_embeddings (model_version)"
)


def _deterministic_sample(candidates: list[str], n: int, seed: int) -> list[str]:
    """The smallest `n` of CANDIDATES by `splitmix64(node_key("a", id) XOR seed)`.

    Deterministic and repeatable for a fixed seed and candidate set -- the AC's "repeatable
    measurement" requirement. Uses the same hash family the pinned FastRP projection itself
    is built from (`insights.embeddings.projection.splitmix64`), not a fresh RNG.
    """
    scored = sorted(candidates, key=lambda aid: splitmix64(node_key("a", aid) ^ seed))
    return scored[:n]


def _load_month(path: Path) -> dict[str, Any]:
    # allow_pickle=True: the object-dtype artist_ids array needs it to deserialize. Safe
    # here -- this file is written by scripts/embeddings_from_dump.py on this same machine
    # in this same measurement workflow, never fetched from an untrusted source.
    with np.load(path, allow_pickle=True) as data:
        return {
            "artist_ids": [str(aid) for aid in data["artist_ids"]],
            "vectors": data["vectors"],
            "model_version": str(data["model_version"]),
            "dump_id": str(data["dump_id"]),
            "dump_date": str(data["dump_date"]),
        }


def _exact_top_k(vectors: np.ndarray, query_positions: list[int], k: int) -> list[list[int]]:
    """Exact top-`k` cosine neighbours (by position, excluding self) for each query
    position, brute-force in NumPy against every row of VECTORS."""
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    norms[norms == 0] = 1.0
    normalized = vectors.astype(np.float32) / norms[:, None]
    results: list[list[int]] = []
    for position in query_positions:
        scores = normalized @ normalized[position]
        scores[position] = -np.inf  # exclude self
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        results.append(top.tolist())
    return results


async def _apply_schema(conn: Any) -> None:
    async with conn.cursor() as cursor:
        await cursor.execute(_VECTOR_EXTENSION_SQL)
        await cursor.execute(_ARTIST_EMBEDDINGS_TABLE_SQL)
        await cursor.execute(_ARTIST_EMBEDDINGS_MODEL_VERSION_INDEX_SQL)


async def _truncate_table(conn: Any) -> None:
    async with conn.cursor() as cursor:
        await cursor.execute(f"TRUNCATE {ARTIST_EMBEDDINGS_TABLE}")  # noqa: S608 -- constant, no caller input.
        await cursor.execute("DROP INDEX IF EXISTS idx_artist_embeddings_embedding_hnsw")


async def _write_month(conn: Any, month: dict[str, Any]) -> dict[str, Any]:
    """Write one month's rows, timing a `TRIAL_BATCH_ROWS` trial through the real
    `_write_embeddings` first; falls back to `COPY` only if that trial extrapolates past
    `TRIAL_TIME_BUDGET_S` for the full month, per the maintainer's condition."""
    artist_ids = month["artist_ids"]
    vectors = month["vectors"]
    total = len(artist_ids)
    trial_n = min(TRIAL_BATCH_ROWS, total)

    trial_started = time.perf_counter()
    trial_rows = await _write_embeddings(
        conn,
        model_version=month["model_version"],
        dump_id=month["dump_id"],
        dump_date=month["dump_date"],
        artist_ids=artist_ids[:trial_n],
        vectors=vectors[:trial_n],
    )
    trial_elapsed = time.perf_counter() - trial_started
    extrapolated_s = trial_elapsed * (total / trial_n) if trial_n else 0.0
    print(f"  trial: {trial_rows:,} rows in {trial_elapsed:.1f}s -> extrapolated full month {extrapolated_s:.0f}s ({extrapolated_s / 60:.1f} min)", file=sys.stderr)

    used_copy = False
    if extrapolated_s > TRIAL_TIME_BUDGET_S and trial_n < total:
        print(f"  extrapolated time exceeds {TRIAL_TIME_BUDGET_S / 60:.0f} min budget -- falling back to COPY for the remainder", file=sys.stderr)
        used_copy = True
        remaining_ids = artist_ids[trial_n:]
        remaining_vectors = vectors[trial_n:]
        computed_at_sql = "NOW()"
        async with conn.cursor() as cursor, cursor.copy(
            f"COPY {ARTIST_EMBEDDINGS_TABLE} (artist_id, model_version, embedding, source_dump_id, source_dump_date, computed_at) "  # noqa: S608
            f"FROM STDIN"
        ) as copy:
            for index in range(len(remaining_ids)):
                vector_literal = "[" + ",".join(f"{value:g}" for value in remaining_vectors[index].tolist()) + "]"
                await copy.write_row(
                    (remaining_ids[index], month["model_version"], vector_literal, month["dump_id"], month["dump_date"], None)
                )
        rows_written = trial_rows + len(remaining_ids)
    elif trial_n < total:
        write_started = time.perf_counter()
        remaining_rows = await _write_embeddings(
            conn,
            model_version=month["model_version"],
            dump_id=month["dump_id"],
            dump_date=month["dump_date"],
            artist_ids=artist_ids[trial_n:],
            vectors=vectors[trial_n:],
        )
        write_elapsed = time.perf_counter() - write_started
        print(f"  remainder: {remaining_rows:,} rows in {write_elapsed:.1f}s", file=sys.stderr)
        rows_written = trial_rows + remaining_rows
    else:
        rows_written = trial_rows

    return {"rows_written": rows_written, "trial_elapsed_s": trial_elapsed, "extrapolated_full_s": extrapolated_s, "used_copy": used_copy}


async def _build_index(conn: Any, model_version: str) -> dict[str, Any]:
    """Build the full-scale HNSW index for MODEL_VERSION's rows, named by `_index_name`.

    Full scale first, per the maintainer's condition: a subset fallback is used only after
    a real failure here, with that failure's wall time and peak memory recorded -- this
    function does not pre-emptively choose a smaller scale.
    """
    index_name = _index_name(model_version)
    async with conn.cursor() as cursor:
        await cursor.execute(f"SET maintenance_work_mem = '{MAINTENANCE_WORK_MEM}'")
        started = time.perf_counter()
        await cursor.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} ON {ARTIST_EMBEDDINGS_TABLE} "  # noqa: S608
            f"USING hnsw (embedding halfvec_cosine_ops) WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        )
        elapsed = time.perf_counter() - started
        await cursor.execute("RESET maintenance_work_mem")
    return {"index_name": index_name, "build_elapsed_s": elapsed}


async def _ann_top_k(conn: Any, model_version: str, vectors: np.ndarray, artist_ids: list[str], positions: list[int], ef_search: int, k: int) -> list[list[str] | None]:
    """The live index's top-`k` artist_ids for each query position's vector, at EF_SEARCH.

    Excludes the query's own artist_id: its vector is stored verbatim in the table, so an
    un-excluded query always ranks itself first (distance 0), the same one-row exclusion
    `_exact_top_k` applies by setting its own score to `-inf`. Without this, ANN and exact
    are compared on a different footing (ANN off by exactly one true positive every time --
    caught by this script's own synthetic smoke test before it ran against real data).
    """
    results: list[list[str] | None] = []
    async with conn.cursor() as cursor:
        await cursor.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        for position in positions:
            literal = "[" + ",".join(f"{value:g}" for value in vectors[position].tolist()) + "]"
            await cursor.execute(
                f"SELECT artist_id FROM {ARTIST_EMBEDDINGS_TABLE} "  # noqa: S608
                f"WHERE model_version = %s AND artist_id != %s ORDER BY embedding <=> %s::halfvec LIMIT %s",
                (model_version, artist_ids[position], literal, k),
            )
            rows = await cursor.fetchall()
            results.append([artist_id for (artist_id,) in rows][:k])
    return results


def _recall_at_k(ann: list[list[str]], exact_ids: list[list[str]], k: int) -> float:
    scores = []
    for ann_ids, exact in zip(ann, exact_ids, strict=True):
        if not exact:
            continue
        overlap = len(set(ann_ids[:k]) & set(exact[:k]))
        scores.append(overlap / min(k, len(exact)))
    return sum(scores) / len(scores) if scores else 0.0


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


async def measure_month(
    conn: Any,
    month: dict[str, Any],
    *,
    label: str,
    query_sample_positions: list[int],
    query_sample_ids: list[str],
) -> dict[str, Any]:
    print(f"\n=== {label}: writing embeddings ===", file=sys.stderr)
    write_result = await _write_month(conn, month)
    print(f"  wrote {write_result['rows_written']:,} rows", file=sys.stderr)

    print(f"=== {label}: building HNSW index (full scale) ===", file=sys.stderr)
    index_result = await _build_index(conn, month["model_version"])
    print(f"  {index_result['index_name']}: {index_result['build_elapsed_s']:.1f}s", file=sys.stderr)

    print(f"=== {label}: exact ground truth ({len(query_sample_positions):,} queries) ===", file=sys.stderr)
    exact_started = time.perf_counter()
    exact_ids_by_query = [
        [month["artist_ids"][position] for position in row]
        for row in _exact_top_k(month["vectors"], query_sample_positions, 10)
    ]
    exact_elapsed = time.perf_counter() - exact_started
    print(f"  exact: {exact_elapsed:.1f}s", file=sys.stderr)

    print(f"=== {label}: recall@10 sweep ===", file=sys.stderr)
    recall_by_ef: dict[int, float] = {}
    production_ef_search: int | None = None
    for ef_search in EF_SEARCH_SWEEP:
        ann_started = time.perf_counter()
        ann_ids_by_query = await _ann_top_k(conn, month["model_version"], month["vectors"], month["artist_ids"], query_sample_positions, ef_search, 10)
        ann_elapsed = time.perf_counter() - ann_started
        recall = _recall_at_k(ann_ids_by_query, exact_ids_by_query, 10)
        recall_by_ef[ef_search] = recall
        print(f"  ef_search={ef_search}: recall@10={recall:.4f} ({ann_elapsed:.1f}s for {len(query_sample_positions):,} queries)", file=sys.stderr)
        if production_ef_search is None and recall >= RECALL_TARGET:
            production_ef_search = ef_search

    churn_ef_search = production_ef_search or EF_SEARCH_SWEEP[-1]
    print(f"=== {label}: churn top-10 at ef_search={churn_ef_search} ({len(query_sample_ids)} query positions reused for churn sample separately) ===", file=sys.stderr)

    return {
        "write": write_result,
        "index": index_result,
        "exact_elapsed_s": exact_elapsed,
        "recall_by_ef_search": recall_by_ef,
        "production_ef_search": production_ef_search,
        "churn_ef_search_used": churn_ef_search,
    }


async def churn_top_k(conn: Any, month: dict[str, Any], churn_sample_ids: list[str], ef_search: int) -> dict[str, list[str]]:
    """Both exact and ANN top-10 (at EF_SEARCH) for CHURN_SAMPLE_IDS, keyed by artist_id.

    Exact: brute-force in NumPy against this month's full in-memory vector array.
    ANN: the live index, at the production `ef_search` -- what the served list would be.
    """
    id_to_position = {aid: index for index, aid in enumerate(month["artist_ids"])}
    positions = [id_to_position[aid] for aid in churn_sample_ids if aid in id_to_position]
    exact = _exact_top_k(month["vectors"], positions, 10)
    exact_by_id = {month["artist_ids"][position]: [month["artist_ids"][n] for n in row] for position, row in zip(positions, exact, strict=True)}
    ann = await _ann_top_k(conn, month["model_version"], month["vectors"], month["artist_ids"], positions, ef_search, 10)
    ann_by_id = {month["artist_ids"][position]: (row or []) for position, row in zip(positions, ann, strict=True)}
    return {"exact": exact_by_id, "ann": ann_by_id}


def compute_churn(aug_top10: dict[str, list[str]], sept_top10: dict[str, list[str]], sample_ids: list[str]) -> dict[str, float]:
    jaccards = [_jaccard(set(aug_top10.get(aid, [])), set(sept_top10.get(aid, []))) for aid in sample_ids if aid in aug_top10 and aid in sept_top10]
    return {"mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0, "n": len(jaccards)}


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    print(f"loading {args.aug}", file=sys.stderr)
    aug = _load_month(args.aug)
    print(f"loading {args.sept}", file=sys.stderr)
    sept = _load_month(args.sept)

    common_ids = sorted(set(aug["artist_ids"]) & set(sept["artist_ids"]))
    print(f"common artists (both months): {len(common_ids):,}", file=sys.stderr)
    churn_sample_ids = _deterministic_sample(common_ids, CHURN_SAMPLE_SIZE, CHURN_SAMPLE_SEED)

    aug_query_ids = _deterministic_sample(aug["artist_ids"], QUERY_SAMPLE_SIZE, QUERY_SAMPLE_SEED)
    aug_id_to_position = {aid: index for index, aid in enumerate(aug["artist_ids"])}
    aug_query_positions = [aug_id_to_position[aid] for aid in aug_query_ids]

    sept_query_ids = _deterministic_sample(sept["artist_ids"], QUERY_SAMPLE_SIZE, QUERY_SAMPLE_SEED)
    sept_id_to_position = {aid: index for index, aid in enumerate(sept["artist_ids"])}
    sept_query_positions = [sept_id_to_position[aid] for aid in sept_query_ids]

    pool = AsyncPostgreSQLPool(
        connection_params={"host": args.host, "port": args.port, "dbname": args.database, "user": args.username, "password": args.password},
        min_connections=1,
        max_connections=1,
    )
    await pool.initialize()
    try:
        async with pool.connection() as conn:
            await _apply_schema(conn)

            aug_result = await measure_month(conn, aug, label="August", query_sample_positions=aug_query_positions, query_sample_ids=aug_query_ids)
            print("\n=== August: churn top-10 (exact + ANN) for the common-artist sample, BEFORE dropping the table ===", file=sys.stderr)
            aug_churn = await churn_top_k(conn, aug, churn_sample_ids, aug_result["churn_ef_search_used"])

            print("\n=== truncating table before September ===", file=sys.stderr)
            await _truncate_table(conn)

            sept_result = await measure_month(conn, sept, label="September", query_sample_positions=sept_query_positions, query_sample_ids=sept_query_ids)
            print("\n=== September: churn top-10 (exact + ANN) for the common-artist sample ===", file=sys.stderr)
            sept_churn = await churn_top_k(conn, sept, churn_sample_ids, sept_result["churn_ef_search_used"])
    finally:
        await pool.close()

    churn_exact = compute_churn(aug_churn["exact"], sept_churn["exact"], churn_sample_ids)
    churn_ann = compute_churn(aug_churn["ann"], sept_churn["ann"], churn_sample_ids)

    return {
        "sampling": {
            "query_sample_seed": QUERY_SAMPLE_SEED,
            "query_sample_size": QUERY_SAMPLE_SIZE,
            "churn_sample_seed": CHURN_SAMPLE_SEED,
            "churn_sample_size_requested": CHURN_SAMPLE_SIZE,
            "churn_sample_size_actual": len(churn_sample_ids),
            "common_artists": len(common_ids),
            "rule": "smallest N by splitmix64(node_key('a', artist_id) XOR seed)",
        },
        "august": {"dump_id": aug["dump_id"], "dump_date": aug["dump_date"], "model_version": aug["model_version"], "n_vectors": len(aug["artist_ids"]), **aug_result},
        "september": {"dump_id": sept["dump_id"], "dump_date": sept["dump_date"], "model_version": sept["model_version"], "n_vectors": len(sept["artist_ids"]), **sept_result},
        "churn_exact_cosine": churn_exact,
        "churn_ann_at_production_ef_search": churn_ann,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aug", type=Path)
    parser.add_argument("sept", type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
