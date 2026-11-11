# Exact similar-artist lists

The maintainer's 2026-09-29 serving-mode decision ("D", `docs/embedding_tie_break.md`) is to serve precomputed monthly exact top-K similar-artist lists, not live HNSW search. gm-analytics-engine-8ts measured the index against the three serving thresholds and it failed all three. Exact top-10 lists were stable month over month (churn 0.9519). This document covers the batch job that computes those lists: `insights/embeddings/exact_top_k.py` (the kernel) and `insights/similar_artists.py` (spool, write, publish). The job runs as the last stage of `analytics-engine-embeddings` only with explicit `ENABLE_SIMILAR_ARTISTS=true`. The default is `false`, and any value other than exactly `true` or `false` fails closed. Configuring `SIMILAR_ARTISTS_SPOOL_DIR` alone does not enable it. The flag stays off in production until [gm-analytics-engine-3am](#sizing-run) has sized the full-catalog run and validated the stored lists on a real dump.

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

The create, publish and retire helpers, and the release-table lock they take, are database-schema's own (`groovemap_schema.postgres`), imported at runtime. `groovemap-database-schema` is a first-party runtime dependency pinned to database-schema `08106040f3c4d7e6494dfbbc495c43d91a0dfb38` (v0.9.0 line, main after the tag, adds insights.label_longevity.is_placeholder), which carries the compact per-artist shape from gm-database-schema-hqjp. No copy of the producer's SQL lives in this repository, so it cannot drift. `scripts/prepare-runtime-wheel.sh` stages that wheel next to the `python-libraries` runtime wheel for the image, and `scripts/check-repository-compliance.py` fails if the two places that name the pin disagree.

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
- `tests/test_similar_artists_sizing.py` checks that a slice's lists equal the full run's first rows position for position (with real ties, ragged blocks and a resume), the extrapolation arithmetic, that a corrupted list fails the exactness check, and that a writable session or an ambiguous `model_version` stops the sizing pass.
- `tests/test_similar_artists.py` covers the spool, checkpoint resume, the COPY rows, publish and rotate, and the monthly stage against in-memory fakes. Its memory-guard tests include one that runs the compute in a fresh process with the real `getrusage` peak and a budget the running lists must exceed, and checks that the guard stops the run and leaves the spool incomplete.
- `just test-integration-pg18` starts a disposable official PostgreSQL 18 container (the digest database-schema's required tier pins), applies database-schema's own `create_postgres_schema`, and runs `tests/integration/test_similar_artists_integration.py` as the real `embedding_pipeline` role. That covers COPY quoting and order, rollback of a failed COPY, refusing to overwrite a published release, publication, keeping the current and previous releases, and retiring older ones. The similarity tables need no pgvector, so the bare image is enough. There is no PostgreSQL 19 GA image yet, so this PG18 tier is the one to run. `just test-integration-pg19` runs the same suite, plus the embedding pipeline's, on the local PG19 beta + pgvector image when one is built. It is advisory.

## Sizing run

gm-analytics-engine-3am sizes the full-catalog run and checks exactness before `ENABLE_SIMILAR_ARTISTS` is ever turned on in production. The operator's decision (2026-11-10) is to run it **on Lux, read-only, against the production `public.artist_embeddings`** (9,152,168 rows, one `model_version`, discogs-2026-11), not on the Mac. It publishes nothing. The full-catalog run needs its own operator approval after these numbers are reported.

The tool is `insights/similar_artists_sizing.py`, shipped in the image as `python -m insights.similar_artists_sizing`. It reuses the production kernel and the production halfvec reader, so the numbers are the job's own.

### What it does

1. **Read-only session.** It connects with `-c default_transaction_read_only=on`, so the server rejects any write, and refuses to start if `SHOW default_transaction_read_only` is not `on`. It never imports the write, publish or retire path, and it does not consult `artist_embedding_releases`, so a published current release does not make it skip.
2. **Resolve the version.** With no `--model-version` it requires exactly one `model_version` in `artist_embeddings`; `--expect-rows 9152168` also fails if the row count differs.
3. **Slice against the full catalog.** It reads every vector (`artist_id` order, so position order and tie order match the job) and computes exact top-50 for the **first N artists** only. d4d's algorithm exploits symmetry (only the upper block triangle is multiplied, each block updating both its row and column lists), and a prefix of artists has no symmetric shortcut: a slice list is exact only if it sees every candidate. So a slice of N artists multiplies the full `N × 9.15M` block and selects from its row side only (`exact_top_k(query_rows=N)`). That is not the same as running d4d on a catalog truncated to N artists, which would produce wrong lists and a quadratic cost. The slice's lists are identical, position for position, to the first N rows of a full run (`tests/test_similar_artists_sizing.py`).
4. **Exactness check.** For a seeded random sample of 2,000 slice artists (`--seed`, default 20261110) it takes the job's stored-order top-10 and compares them with `insights/embeddings/independent_exact_top_k.py`, a brute force that does not import the kernel and streams the whole catalog in bounded chunks. It re-scores every neighbour independently and requires tie-tolerant recall 1.0 with scores within 1e-4 (`exactness.passed`). The process exits 2 if it fails.
5. **Report.** One JSON document of aggregates (no ids, names or vectors): wall-clock per phase, peak RSS after each phase, kernel thread-seconds split into upcast, multiply and selection, and the extrapolation below.

### Run command

The sizing run needs a writable scratch directory owned by uid 1000 with about 1.6 GB free for a 1M slice (the spool plus checkpoints; the tool checks before it reads), on disk rather than tmpfs so it does not count against the container. The image must be built from a revision that contains `insights/similar_artists_sizing.py`.

```sh
mkdir -p /srv/scratch/similar-sizing && chown 1000:1000 /srv/scratch/similar-sizing

for SLICE in 500000 1000000; do
  docker run --rm \
    --network <network that reaches the production postgres> \
    --memory 16g --memory-swap 16g --cpus <cores free on Lux> \
    -v /srv/scratch/similar-sizing:/scratch \
    -e POSTGRES_HOST=<host or host:port> -e POSTGRES_DATABASE=<catalog database> \
    -e EMBEDDING_PIPELINE_POSTGRES_USERNAME_FILE=/run/secrets/ae_user \
    -e EMBEDDING_PIPELINE_POSTGRES_PASSWORD_FILE=/run/secrets/ae_password \
    -v <secret file>:/run/secrets/ae_user:ro -v <secret file>:/run/secrets/ae_password:ro \
    --entrypoint /app/.venv/bin/python \
    <analytics-engine image, pinned by digest> \
    -m insights.similar_artists_sizing \
      --slice-rows "$SLICE" --expect-rows 9152168 \
      --validate-sample 2000 --seed 20261110 \
      --threads 8 --scratch-dir /scratch --budget-gb 12 \
      --output /scratch/sizing-"$SLICE".json
done
```

Notes:

- Use the existing `embedding_pipeline` login (`EMBEDDING_PIPELINE_POSTGRES_*`, `_FILE` form preferred), the login the monthly pipeline already uses to read `artist_embeddings`. If the dispatcher has a narrower read-only role on Lux, use it in its place; the read-only session option makes the role's write privileges irrelevant either way. No other variable is needed: `ENABLE_SIMILAR_ARTISTS` and `SOURCE_DUMP_*` are not read.
- `--memory 16g` is a safety cap on a host that has ~45 GB free; the tool's own guard aborts at `--budget-gb 12`, so a run that would not fit the budget stops and reports instead of being OOM-killed. Do not raise the guard to see how far it goes: the budget is the question being answered.
- `--threads` and `--cpus` should match; the earlier measurements used 8 threads. Run when Lux is quiet, and say what else was running in the report.
- Run the two slices one after the other, as separate processes, so each has its own peak RSS (`ru_maxrss` is a process-lifetime maximum). Each re-reads the catalog.
- Expected output: JSON on stdout (logs go to stderr) and in `/scratch/sizing-<N>.json`, shaped as below, ending with `"exactness": {"passed": true, ...}`. Delete `/scratch/sizing-*.json` when done; the spool is removed automatically.

```text
{
  "mode": "dry_run_read_only", "writes_to_postgres": false,
  "slice_rows": <N>, "catalog_rows": 9152168, "dimensions": 128, "k": 50,
  "phases": {"read_embeddings_s": <R>, "compute_slice_s": <T>, "exactness_check_s": <V>},
  "peak_rss_bytes": {"after_read": <P0>, "after_compute": <P1>, "after_exactness_check": <P2>},
  "throughput": {"artists_per_second": <A>, "seconds_per_block_row": <S>},
  "kernel_thread_seconds": {"upcast": <U>, "matmul": <M>, "select": <L>, "score_blocks": <B>},
  "exactness": {"queries": 2000, "exact_set_recall": <r>, "tie_tolerant_recall": 1.0,
                "score_max_abs_error": <e>, "score_violation_count": 0,
                "canonical_order_violations": 0, "passed": true, "tolerance": 0.0001},
  "extrapolation": {"scale": <9152168/N>, "matmul_fraction": <f>, "full_wall_s": <W>,
                    "full_wall_upper_bound_s": <Wmax>, "full_peak_rss_bytes": <Pfull>,
                    "estimate_peak_bytes": <E>, "within_six_hours": <bool>,
                    "within_six_hours_upper_bound": <bool>, "within_memory_budget": <bool>},
  "model_version_hash": "<12 hex>"
}
```

### Extrapolating to the 9.15M catalog

Let `N = 9,152,168`, `Q` the slice, `T = compute_slice_s`, `scale = N / Q`.

- **Wall-clock.** The slice's work is linear in `Q` (every row block sees all `N` columns), so `T × scale` is what the same method costs for all `N` rows **without** symmetry: `full_wall_upper_bound_s`. The production kernel multiplies only the upper triangle, which halves the upcast and the multiply but not the selection (every block is selected from on both its row and column side). With `f` the multiply-plus-upcast share of kernel thread-seconds, `(upcast + matmul) / (upcast + matmul + select)`:

  `full_wall_s = T × scale × (0.5 f + (1 − f))`

  Report `full_wall_s` as the estimate and `full_wall_upper_bound_s` as the ceiling, and compare both with the six-hour quiet-host target. Two slices (500k and 1M) should agree on seconds per query block row; if they differ by more than ~10%, say so and trust the larger slice. The real symmetric run has a slightly cheaper column-side selection than this model assumes, so the model leans high. Real-data selection cost also depends on how fast thresholds rise, which a prefix of `artist_id` order may not represent. The slice is a first-N prefix, not a random sample, so name that bias in the report. A full run's first checkpoints (every 30 minutes, block index in the log) give the first real check of the projection.
- **Peak RSS.** The slice process already holds the whole catalog's vectors, norms, ids and read buffers, so the part that grows is the running lists, `K × 8 + 8 = 408` bytes per artist:

  `full_peak = P1 + (N − Q) × 408`

  where `P1 = peak_rss_bytes.after_compute` (the exactness check is not in it). Compare `full_peak` with the 12 GB budget, and against `estimate_peak_bytes` (the closed-form estimate the guard logs) as a cross-check; the two should agree to within a few hundred MB. The spool and checkpoints are file writes and do not count toward RSS, but the first full-run checkpoint transient has not been measured at this size, so leave at least 1 GB of margin below 12 GB before recommending a full run.

Measured on Lux on 2026-11-11, read-only (`writes_to_postgres: false`), publishing nothing. Per-run result JSON is not committed; only these aggregates are recorded.

Run facts: analytics-engine v0.8.0 image; `insights.similar_artists_sizing`; catalog 9,152,168 artists, 128 dimensions, one `model_version` (hash `e7dd9162b4ef`); K = 50; 4,096-row blocks; 8 threads; `--memory 16g`; `--budget-gb 12`; validation seed 20261110 with 2,000 queries. The two slices ran as separate containers, one after the other.

- 500k slice: 04:32Z to 07:58Z (3 h 26 min).
- 1M slice: 07:58Z to 14:03Z (6 h 05 min).
- Lux load during each run is not captured in the result JSON, so it is not recorded here.

Per-slice measurements (`R` read, `T` compute, `V` exactness check):

| Slice | `R` read | `T` compute | `V` exactness | Artists/s | s per block row | Peak RSS after read | Peak RSS after compute `P1` | matmul fraction `f` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500k (5.5% of catalog) | 45.6 s | 12,147 s (3.37 h) | 157.5 s | 41.16 | 98.76 | 3.12 GB | 4.87 GB | 0.809 |
| 1M (10.9% of catalog) | 42.0 s | 21,754 s (6.04 h) | 156.7 s | 45.97 | 88.79 | 3.12 GB | 4.96 GB | 0.784 |

Peak RSS did not change during the exactness check in either run. Kernel thread-seconds (matmul / select / upcast / score blocks): 500k 77,261 / 18,405 / 723 / 274,905; 1M 133,847 / 37,180 / 1,481 / 547,575.

Exactness (both slices, 2,000-artist seeded sample, checked against an independent brute force over the whole catalog): exact set recall 1.0 and tie-tolerant recall 1.0 in both. Max absolute score error 6.56e-7 (500k) and 5.96e-7 (1M), against a tolerance of 1e-4. Score violations 0 and canonical-order violations 0 in both. Both runs passed.

Full-catalog extrapolation (N = 9,152,168; the 500k scale is 18.30 and the 1M scale is 9.15):

| Slice | Full wall, estimate | Full wall, upper bound | Full peak RSS | vs 12 GB budget | vs 6 h window |
| --- | ---: | ---: | ---: | --- | --- |
| 500k | 132,400 s = 36.8 h | 222,345 s = 61.8 h | 8.40 GB | fits, 3.6 GB below budget | not within, estimate or upper bound |
| 1M | 121,004 s = 33.6 h | 199,098 s = 55.3 h | 8.28 GB | fits, 3.7 GB below budget | not within, estimate or upper bound |

- The closed-form `estimate_peak_bytes` is 8.22 GB for both slices, within 0.2 GB of the extrapolated peak.
- The two slices give 98.76 and 88.79 s per block row, 11% apart. That is over the doc's 10% threshold, so per the rule above the 1M slice is the one to trust: 33.6 h estimate, 55.3 h upper bound. The 500k figures are shown for the spread.

Conclusion (facts only):

- **Memory fits.** The extrapolated full-catalog peak is 8.3–8.4 GB against the 12 GB budget, with more than 1 GB of margin.
- **Time does not fit the 6 h window.** The exact full-catalog run measures about 34–37 h at 8 threads (33.6 h from the 1M slice, the one to trust). The upper bound is 55–62 h. Neither the estimate nor the upper bound is within 6 h.
- **Caveats.** Both slices are first-N prefixes by `artist_id`, not random samples, so the projection carries that bias. The selection-cost model leans high. The first real check of the projection will be the first full-run checkpoint. The full-run checkpoint transient at this size is still unmeasured.
- **Still deferred.** The Aug→Sept Jaccard check waits for a second month of embeddings (see above).

### Month-over-month Jaccard

The Aug → Sept top-10 Jaccard check (about 0.95, consistent with 8ts) **waits for a second month of embeddings**. Lux holds only discogs-2026-11 and the 2026-09 dump the bead named is no longer the data there, so the sizing run cannot measure it. It moves to the next monthly run, where the previous release's stored lists are still present for comparison (only releases older than the previous one are retired).

### Local development

`scripts/measure-exact-top-k-sizing.py` still measures a local snapshot slice (500k–1M) on a workstation, as a truncated catalog. It is not a substitute for the Lux run: it sizes a different, smaller problem.

Earlier bounded measurements give 3am a starting point. They are not acceptance results. All used the 2026-09 `edges-v3, w0=0, self=0.05` vectors, K=50, 4096-row blocks and eight kernel threads:

| Slice and host | Kernel wall | Peak RSS | Full-catalog projection |
| --- | ---: | ---: | ---: |
| 1M artists, shared 10-core M1 Pro, light load | 258.3 s | 2.92 GB | about 5.0 h at the steady rate |
| 500k artists, same host, admitted window | 58.3 s | 2.52 GB | 5.69 h (quadratic) |
| 500k artists, isolated x86_64 container (Core Ultra 5 235HX, OpenBLAS), after the column-segment change | 104.6 s | 1.85 GB | 10.20 h (quadratic) |

The projected full-pipeline peak is about 9.0 GB, inside the 12 GB budget. The time projections disagree by host, and the most recent one is over the six-hour window. Settling that is 3am's job. The run conditions, hashes and rejected tuning trials are in this file's git history (commit `798228d`).
