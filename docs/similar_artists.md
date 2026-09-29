# Exact similar-artist lists

The maintainer's 2026-09-29 serving-mode decision ("D", `docs/embedding_tie_break.md`) is to serve precomputed monthly exact top-K similar-artist lists, not live HNSW search. gm-analytics-engine-8ts measured the index against the three serving thresholds and it failed all three. Exact top-10 lists were stable month over month (churn 0.9519). This document covers the batch job that computes those lists: `insights/embeddings/exact_top_k.py` (the kernel) and `insights/similar_artists.py` (spool, write, publish). The job runs as the last stage of `analytics-engine-embeddings`.

No provider-derived data (ids, names, vectors, lists) is committed anywhere in this repository. The tests use synthetic vectors, and the sizing below reports only aggregates.

## What is computed

For every artist in a `model_version` of `public.artist_embeddings`, the job finds the `K = 50` other artists with the highest cosine similarity over the whole catalog. It is exact: no index and no approximation. Self is excluded. A duplicate vector still appears, at score 1.

The job writes these lists to `public.artist_similar_artists` under the same `model_version`, one row per `(artist_id, rank)`, with ranks starting at 1. Then `publish_artist_embedding_release` makes that version current.

Rank order is score descending, then catalog position ascending. The stage reads vectors in `artist_id` order, so position order is `artist_id` text order, and exactly equal scores always rank the lower `artist_id` first. The order does not depend on block size, thread count, or the order in which blocks reach a row. `tests/test_exact_top_k.py` checks this against brute force on vectors whose dot products are exact in `float32`, so the ties are real rather than rounding accidents.

## Method

The 9.37M × 9.37M score matrix is never materialized, so the job computes it in blocks.

- **Half-precision storage, `float32` arithmetic.** The vectors stay `float16`, as stored in the `halfvec` column: 2.4 GB for the catalog. Each `4096`-row block is upcast to `float32` and scaled by its rows' reciprocal norms just before it is multiplied. Every score is therefore the `float32` dot product of `float32`-normalized vectors, the same number an independent brute force computes. Scores agree with brute force to within 3e-7.
- **Symmetry.** Only the upper block triangle `(i, j), j ≥ i` is multiplied, with Accelerate's multithreaded `sgemm` on macOS and the platform BLAS elsewhere. Each block of scores updates block `i`'s lists from its rows and block `j`'s lists from its columns. This halves the multiply, which is the part no selection trick can shrink.
- **Running top-K with a cheap prefilter.** Each artist keeps its best `K` so far and its current `K`-th best score as a threshold. Most of each score block is rejected in one pass:
  - the job takes the maximum of every run of 64 scores along a row, and of every run of 64 along a column;
  - it examines individually only the runs whose maximum clears that list's own threshold;
  - candidates are merged under a per-block lock using a single `int64` rank key (score, position), which is several times faster than a two-key `lexsort`.
  
  Rows seeing a block for the first time, when their threshold is still at the floor, are cut to the block-local top `K` with `np.partition` instead. Taller tiles, for example 8 × 64, were tried and rejected: they must test against the lowest threshold among their rows, and thresholds vary too much between artists. On real data, 94% of 8 × 64 tiles still passed.
- **Row blocks finalize in order.** Once every task `(i', i)` with `i' ≤ i` has run, block `i` has seen every column. Its lists are final and are written to the spool. Checkpoints are taken at this boundary.
- **Threads.** A thread pool runs the tasks for one `i`. NumPy releases the GIL inside BLAS and inside the elementwise passes, so eight threads overlap the multiply (on the AMX units) with selection (on the cores).

## Pipeline stage

`run_embedding_pipeline` calls `insights.similar_artists.run_similar_artists` after each load, whether the load wrote vectors or found them already loaded:

1. **Skip if published.** If `artist_embedding_releases` already has this `model_version`, the stage does nothing.
2. **Read back.** The stage reads the stored vectors back from `artist_embeddings` in `artist_id` order, through a named cursor, into a preallocated `float16` array. It does not reuse the load's in-memory array, so FastRP's peak (about 10 GB, `docs/embeddings.md`) and this stage's peak never overlap.
3. **Compute to a spool** (`compute_to_spool`). Finished lists go to two raw files under `SIMILAR_ARTISTS_SPOOL_DIR`, `positions.i32` and `scores.f32` (3.75 GB for the catalog at K=50). They are written with plain file writes rather than a memory map, so they never count toward this process's RSS.
   - Every 30 minutes, the running state and the next block index are saved to `checkpoint.npz`. Every checkpoint and metadata write goes to a temporary name first and is then renamed into place.
   - A run with the same `model_version`, catalog size, and `K` resumes from its last checkpoint. A different one starts over.
4. **Write** (`write_similar_artists`). One transaction deletes any rows a failed attempt left under this `model_version` and streams every list in with `COPY`. `catalog-api` sees nothing until the next step, and a failure rolls the whole version back.
5. **Publish and rotate** (`publish_and_rotate`).
   - `publish_artist_embedding_release` makes this version current.
   - Every release older than the one it replaced then has its rows retired with `retire_artist_similar_artists_version`. The previous release stays, so a bad month can be rolled back by republishing it.
   - Release rows are kept as lineage.
   - A failed retire is logged and reported but does not fail the run, because the new release is already live and correct.

Each database step takes its own pool connection, so none is held open across the multi-hour compute. The publish and retire helpers belong to database-schema. They are reached through the `ReleaseRegistry` protocol, so the flow is unit-tested with a fake. `SchemaReleaseRegistry` resolves the helpers by name at call time, because the pinned `groovemap-database-schema` revision predates them.

## Memory guard

`MemoryGuard` runs before the compute and after every row block. It does two things:

- **Budget.** It raises `MemoryBudgetExceededError` once this process's peak RSS passes the 12 GB pipeline budget. Peak RSS is `resource.getrusage`'s `ru_maxrss`. This is the monotonic peak: an idle wait that lets the OS page the process out cannot make it read smaller than it is, which was the failure 66608ed fixed in `scripts/embeddings_from_dump.py`.
- **Host pressure.** While `kern.memorystatus_level` is below 25%, it pauses without aborting and polls every 60 s. Where that sysctl does not exist, as in a Linux container, it never pauses.

`estimate_peak_bytes` is logged at the start of the compute. It is the sum of:

- the `float16` vectors;
- the running lists: `K` × (4-byte score + 4-byte position) per artist, plus norms and thresholds;
- 16 bytes per block cell per thread, measured.

## Sizing

All runs used the first 1,000,000 artists of the real 2026-09 `edges-v3, w0=0, self=0.05` vectors (local only), `K = 50`, `block_rows = 4096`, and 8 threads. The host was the shared 10-core (8P + 2E) M1 Pro with 32 GB, under whatever load the other work on it was producing. One pair is one cell of the upper triangle, and it updates two lists.

| Run | Host load average | Wall | Steady pairs/s | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Final kernel | 5 → 19 | 258.3 s | 2.42e9 | 2.92 GB |
| Before the single-key merge | 8 → 11 | 227.0 s | 3.0–3.3e9 | 3.48 GB |
| Final prefilter, heavily loaded host | 30 → 55 | 922.9 s | 5.4e8 | 3.50 GB |

A 1M-row slice under-represents the steady state, because thresholds tighten as the catalog grows: a full-catalog row sees far fewer candidates per block. Naive `(N/n)²` scaling is therefore pessimistic.

Extrapolated to the full catalog (9,366,416 artists, 4.386e13 pairs):

- **`sgemm` floor.** Eight concurrent 4096 × 4096 × 128 `sgemm` calls measure 1.5–1.7 TFLOPS in total. The triangle's 1.12e16 FLOP then takes about **2.0 h** even if selection cost nothing.
- **At the final kernel's steady rate:** 4.386e13 / 2.42e9 = 18,100 s, about **5.0 h**, plus about a minute of first-sight partitioning in block 0. Naive `(N/n)²` scaling of the same run gives 6.3 h.
- **Under heavy host load** (other seats saturating the machine), the same kernel extrapolates to about 22 h. Runtime on this host is dominated by what else is running.

Peak RSS for the full catalog is about **9.0 GB**, within the 12 GB budget with about 3 GB of headroom:

| Component | Size |
| --- | ---: |
| Vectors (`float16`) | 2.40 GB |
| Running lists | 3.75 GB |
| Task working set (measured at 1M) | 2.1 GB |
| Artist ids, held for the write | about 0.6 GB |

The spool (3.75 GB) and a checkpoint (3.8 GB) are on disk, not in RSS.

## Storage

Measured on a throwaway PG19 container with database-schema's `fecb0a4` DDL: 2,000,000 synthetic rows with a real-length stored `model_version` (148 characters).

| Relation | Bytes per row |
| --- | ---: |
| Heap | 215.6 |
| Primary key `(artist_id, model_version, rank)` | 327.1 |
| `(model_version, artist_id)` index | 16.2 |
| **Total** | **559** |

At K=50 the catalog is 468.3M rows, about **262 GB per release**, or about 520 GB with the previous release retained. At K=10 it is about 52 GB per release. Most of the cost is the 148-byte `model_version`, repeated on every row in both the heap and the primary key. The same lists stored one row per artist (id, a small release id, a `text[]` of ids, a `real[]` of scores) measured 943 bytes per artist, about 8.8 GB per release at K=50. This is open with the dispatcher and database-schema. The write path above follows the landed DDL and will follow it if the DDL changes.

## Operations

- **Disk.** The spool directory needs about 7.5 GB free for the full catalog at K=50: the spool plus one checkpoint, written to a temporary name before the old one is replaced. It must survive a process restart for a stopped run to resume. The directory for a published `model_version` is removed after publishing.
- **Stopping and resuming.** Killing the process loses at most one checkpoint interval (30 minutes). Rerunning the pipeline for the same dump skips the already-loaded embeddings, reads them back, and resumes the spool from its checkpoint.
- **Rolling back.** Republishing the previous `model_version` with `publish_artist_embedding_release` switches `catalog-api` back. Its rows are still present, because only releases older than the previous one are retired.
- **Validation.** For the real-dump check, independently recompute exact top-10 for a 2,000-artist sample (tie-tolerant recall 1.0, ties within 1e-4) and measure Aug → Sept top-10 Jaccard. These runs are local-only. Results are recorded here once the full-catalog run is approved.
