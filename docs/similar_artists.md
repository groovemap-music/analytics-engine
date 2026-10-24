# Exact similar-artist lists

The maintainer's 2026-09-29 serving-mode decision ("D", `docs/embedding_tie_break.md`) is to serve precomputed monthly exact top-K similar-artist lists, not live HNSW search. gm-analytics-engine-8ts measured the index against the three serving thresholds and it failed all three. Exact top-10 lists were stable month over month (churn 0.9519). This document covers the batch job that computes those lists: `insights/embeddings/exact_top_k.py` (the kernel) and `insights/similar_artists.py` (spool, write, publish). The job runs as the last stage of `analytics-engine-embeddings` only with explicit `ENABLE_SIMILAR_ARTISTS=true`. The default is `false`, and any value other than exactly `true` or `false` fails closed. Configuring `SIMILAR_ARTISTS_SPOOL_DIR` alone does not enable it. The flag stays off in production until [gm-analytics-engine-3am](#sizing-and-validation-pending-gm-analytics-engine-3am) has sized the full-catalog run and validated the stored lists on a real dump.

No provider-derived data (ids, names, vectors, lists) is committed anywhere in this repository. The tests use synthetic vectors, and the sizing below reports only aggregates.

## What is computed

For every artist in a `model_version` of `public.artist_embeddings`, the job finds the `K = 50` other artists with the highest cosine similarity over the whole catalog. It is exact: no index and no approximation. Self is excluded. A duplicate vector still appears, at score 1.

The job writes these lists to `public.artist_similar_artists` under the same `model_version`, one row per `(release_id, artist_id)`, with ordered `similar_artist_ids TEXT[]` and
`scores REAL[]`. The generated integer `release_id` refers to the unique `model_version`
in `artist_embedding_releases`; the version label is not repeated per neighbour. Then `publish_artist_embedding_release` makes that version current.

Rank order is score descending, then catalog position ascending. The stage reads vectors in `artist_id` order, so position order is `artist_id` text order, and exactly equal scores always rank the lower `artist_id` first. The order does not depend on block size, thread count, or the order in which blocks reach a row. `tests/test_exact_top_k.py` checks this against brute force on vectors whose dot products are exact in `float32`, so the ties are real rather than rounding accidents.

## Method

The 9.37M × 9.37M score matrix is never materialized, so the job computes it in blocks.

- **Half-precision storage, `float32` arithmetic.** The vectors stay `float16`, as stored in the `halfvec` column: 2.4 GB for the catalog. Each `4096`-row block is upcast to `float32` and scaled by its rows' reciprocal norms just before it is multiplied. Every score is therefore the `float32` dot product of `float32`-normalized vectors, the same number an independent brute force computes. Scores agree with brute force to within 3e-7.
- **Symmetry.** Only the upper block triangle `(i, j), j ≥ i` is multiplied, with Accelerate's multithreaded `sgemm` on macOS and the platform BLAS elsewhere. Each block of scores updates block `i`'s lists from its rows and block `j`'s lists from its columns. This halves the multiply, which is the part no selection trick can shrink.
- **Running top-K with a cheap prefilter.** Each artist keeps its best `K` so far and its current `K`-th best score as a threshold. Most of each score block is rejected in one pass:
  - the job takes the maximum of every run of 64 scores along a row, and of every run of 16 along a column (the narrower column run was measured about 29% cheaper to select from, with identical candidates);
  - it examines individually only the runs whose maximum clears that list's own threshold;
  - candidates are merged under a per-block lock using a single `int64` rank key (score, position), which is several times faster than a two-key `lexsort`.
  
  Rows seeing a block for the first time, when their threshold is still at the floor, are cut to the block-local top `K` with `np.partition` instead. Taller tiles, for example 8 × 64, were tried and rejected: they must test against the lowest threshold among their rows, and thresholds vary too much between artists. On real data, 94% of 8 × 64 tiles still passed.
- **Row blocks finalize in order.** Once every task `(i', i)` with `i' ≤ i` has run, block `i` has seen every column. Its lists are final and are written to the spool. Checkpoints are taken at this boundary.
- **Threads.** A thread pool runs the tasks for one `i`. NumPy releases the GIL inside BLAS and inside the elementwise passes, so eight threads overlap the multiply (on the AMX units) with selection (on the cores).

## Pipeline stage

`run_embedding_pipeline` calls `insights.similar_artists.run_similar_artists` only when it is given a spool directory, and the monthly entry point gives it one only with `ENABLE_SIMILAR_ARTISTS=true`. When enabled, the stage follows the load whether that load wrote the vectors or found them already loaded:

1. **Skip if published.** If `artist_embedding_releases` has this `model_version` with `artists > 0` (a published
   release), the stage does nothing. A pending target (`artists = 0`) is retried.
2. **Read back.** The stage reads the stored vectors back from `artist_embeddings` in `artist_id` order, through a named cursor, into a preallocated `float16` array. It does not reuse the load's in-memory array, so FastRP's peak (about 10 GB, `docs/embeddings.md`) and this stage's peak never overlap.
3. **Compute to a spool** (`compute_to_spool`). Finished lists go to two raw files under `SIMILAR_ARTISTS_SPOOL_DIR`, `positions.i32` and `scores.f32` (3.75 GB for the catalog at K=50). They are written with plain file writes rather than a memory map, so they never count toward this process's RSS.
   - Every 30 minutes, the running state and the next block index are saved to `checkpoint.npz`. Every checkpoint and metadata write goes to a temporary name first and is then renamed into place.
   - A run with the same `model_version`, catalog size, and `K` resumes from its last checkpoint. A different one starts over.
4. **Write** (`write_similar_artists`). One transaction takes the producer release lock, creates or reuses the non-current
   target with matching lineage and K, deletes pending rows by its generated id, and
   streams one paired-array row per artist with `COPY.write_row`. Psycopg handles text
   quoting. Previously published lists cannot be overwritten. `catalog-api` sees nothing until the next step, and a failure rolls the whole version back.
5. **Publish and rotate** (`publish_and_rotate`).
   - `publish_artist_embedding_release` makes this version current.
   - Every other previously published release then has its rows retired with `retire_artist_similar_artists_version`. The release that was actually current before the flip stays, so a bad month can be rolled back by republishing it.
   - Release rows are kept as lineage.
   - A failed retire is logged and reported but does not fail the run, because the new release is already live and correct.

Each database step takes its own pool connection, so none is held open across the multi-hour compute.

The create, publish and retire helpers, and the release-table lock they take, are database-schema's own (`groovemap_schema.postgres`), imported at runtime. `groovemap-database-schema` is a first-party runtime dependency pinned to database-schema `dc8c2c6e2e006ee7a9be7501d572073a558d23c0` (v0.4.1), which carries the compact per-artist shape from gm-database-schema-hqjp. No copy of the producer's SQL lives in this repository, so it cannot drift. `scripts/prepare-runtime-wheel.sh` stages that wheel next to the `python-libraries` runtime wheel for the image, and `scripts/check-repository-compliance.py` fails if the two places that name the pin disagree.

## Memory guard

`MemoryGuard` runs before the compute and after every row block. It does two things:

- **Budget.** It raises `MemoryBudgetExceededError` once this process's peak RSS passes the 12 GB pipeline budget. Peak RSS is `resource.getrusage`'s `ru_maxrss`. This is the monotonic peak: an idle wait that lets the OS page the process out cannot make it read smaller than it is, which was the failure 66608ed fixed in `scripts/embeddings_from_dump.py`.
- **Host pressure.** While `kern.memorystatus_level` is below 25%, it pauses without aborting and polls every 60 s. Where that sysctl does not exist, as in a Linux container, it never pauses.

`estimate_peak_bytes` is logged at the start of the compute. It is the sum of:

- the `float16` vectors;
- the running lists: `K` × (4-byte score + 4-byte position) per artist, plus norms and thresholds;
- 16 bytes per block cell per thread, measured.

## Storage

`public.artist_similar_artists` holds one row per artist per release: the integer `release_id`, the `artist_id`, and the ordered `similar_artist_ids TEXT[]` / `scores REAL[]` pair. A `CHECK` keeps the two arrays the same length, and a trigger rejects a list longer than the release's `K`. The producer measured 942.08 bytes per artist (1M synthetic artists, K=50), about 8.8 GB per release for the catalog. That excludes WAL, embeddings, replicas and dead tuples. An earlier one-row-per-neighbour layout measured 559 bytes per neighbour, about 262 GB per release, mostly the 148-byte `model_version` repeated on every row; that is why the storage was reshaped (maintainer decision, 2026-09-29).

## Operations

- **Disk.** The spool directory needs about 11.3 GB free for the full catalog at K=50: the 3.75 GB spool plus both the
  previous and temporary replacement checkpoints (about 3.78 GB each). Atomic replacement
  temporarily retains both checkpoint files. Add database capacity separately: about
  17.6 GB for current/previous compact releases, before WAL/temp-file headroom. On an
  ongoing monthly rotation, the next target coexists with both retained releases until
  publication/retirement, so peak lists alone can be about 26.4 GB (three releases). It must survive a process restart for a stopped run to resume. The directory for a published `model_version` is removed after publishing.
- **Stopping and resuming.** Killing the process loses at most one checkpoint interval (30 minutes). Rerunning the pipeline for the same dump skips the already-loaded embeddings, reads them back, and resumes the spool from its checkpoint.
- **Rolling back.** Republishing the previous `model_version` with `publish_artist_embedding_release` switches `catalog-api` back. Its rows are still present, because only releases older than the previous one are retired.
- **Run window.** Run the full catalog only when the host is quiet. A run that does not finish in one window may resume from its block checkpoint on a later night. There is no approximate first pass and no degree cutoff.

## Testing

All tests use synthetic vectors and ids.

- `tests/test_exact_top_k.py` checks the kernel against an independent brute force over block sizes, thread counts and ragged final blocks. It also covers self exclusion when an exact duplicate exists, deterministic tie order on vectors whose dot products are exact in `float32`, zero vectors, and resuming from a checkpoint.
- `tests/test_similar_artists.py` covers the spool, checkpoint resume, the COPY rows, publish and rotate, and the monthly stage against in-memory fakes. Its memory-guard tests include one that runs the compute in a fresh process with the real `getrusage` peak and a budget the running lists must exceed, and checks that the guard stops the run and leaves the spool incomplete.
- `just test-integration-pg18` starts a disposable official PostgreSQL 18 container (the digest database-schema's required tier pins), applies database-schema's own `create_postgres_schema`, and runs `tests/integration/test_similar_artists_integration.py` as the real `embedding_pipeline` role. That covers COPY quoting and order, rollback of a failed COPY, refusing to overwrite a published release, publication, keeping the current and previous releases, and retiring older ones. The similarity tables need no pgvector, so the bare image is enough. There is no PostgreSQL 19 GA image yet, so this PG18 tier is the one to run. `just test-integration-pg19` runs the same suite, plus the embedding pipeline's, on the local PG19 beta + pgvector image when one is built. It is advisory.

## Sizing and validation: pending gm-analytics-engine-3am

This repository's tests do not size the full-catalog run or validate stored lists on real data. Both moved to gm-analytics-engine-3am (scope split, 2026-10-20):

- **Sizing.** Measure throughput on a 500k–1M-artist slice, then extrapolate wall-clock time and peak RSS for the full catalog against the 12 GB budget and the roughly six-hour window before any full run. `scripts/measure-exact-top-k-sizing.py` does this from a local snapshot. It reads only the requested prefix of the vectors and never the artist ids.
- **Validation.** On the real 2026-09 dump, compare the stored top-10 lists for a 2,000-artist sample against an independent exact computation (`scripts/independent_exact_top_k.py`, which does not import the production kernel): recall 1.0, ties within 1e-4. Then check that Aug → Sept top-10 Jaccard is about 0.95, consistent with 8ts.

Both run locally only, on real data that is never committed. Until 3am reports, `ENABLE_SIMILAR_ARTISTS` stays off in production.

Earlier bounded measurements give 3am a starting point. They are not acceptance results. All used the 2026-09 `edges-v3, w0=0, self=0.05` vectors, K=50, 4096-row blocks and eight kernel threads:

| Slice and host | Kernel wall | Peak RSS | Full-catalog projection |
| --- | ---: | ---: | ---: |
| 1M artists, shared 10-core M1 Pro, light load | 258.3 s | 2.92 GB | about 5.0 h at the steady rate |
| 500k artists, same host, admitted window | 58.3 s | 2.52 GB | 5.69 h (quadratic) |
| 500k artists, isolated x86_64 container (Core Ultra 5 235HX, OpenBLAS), after the column-segment change | 104.6 s | 1.85 GB | 10.20 h (quadratic) |

The projected full-pipeline peak is about 9.0 GB, inside the 12 GB budget. The time projections disagree by host, and the most recent one is over the six-hour window. Settling that is 3am's job. The run conditions, hashes and rejected tuning trials are in this file's git history (commit `798228d`).
