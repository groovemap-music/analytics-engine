# Exact similar-artist lists

The maintainer's 2026-09-29 serving-mode decision ("D", `docs/embedding_tie_break.md`) is to serve precomputed monthly exact top-K similar-artist lists, not live HNSW search. gm-analytics-engine-8ts measured the index against the three serving thresholds and it failed all three. Exact top-10 lists were stable month over month (churn 0.9519). This document covers the batch job that computes those lists: `insights/embeddings/exact_top_k.py` (the kernel) and `insights/similar_artists.py` (spool, write, publish). The job runs as the last stage of `analytics-engine-embeddings`.

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
  - the job takes the maximum of every run of 64 scores along a row, and of every run of 64 along a column;
  - it examines individually only the runs whose maximum clears that list's own threshold;
  - candidates are merged under a per-block lock using a single `int64` rank key (score, position), which is several times faster than a two-key `lexsort`.
  
  Rows seeing a block for the first time, when their threshold is still at the floor, are cut to the block-local top `K` with `np.partition` instead. Taller tiles, for example 8 × 64, were tried and rejected: they must test against the lowest threshold among their rows, and thresholds vary too much between artists. On real data, 94% of 8 × 64 tiles still passed.
- **Row blocks finalize in order.** Once every task `(i', i)` with `i' ≤ i` has run, block `i` has seen every column. Its lists are final and are written to the spool. Checkpoints are taken at this boundary.
- **Threads.** A thread pool runs the tasks for one `i`. NumPy releases the GIL inside BLAS and inside the elementwise passes, so eight threads overlap the multiply (on the AMX units) with selection (on the cores).

## Pipeline stage

`run_embedding_pipeline` calls `insights.similar_artists.run_similar_artists` after each load, whether the load wrote vectors or found them already loaded:

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

Each database step takes its own pool connection, so none is held open across the multi-hour compute. The create, publish and retire helpers are promoted verbatim from approved
database-schema commit `4e9720d838c7da8a6bde139c64a69d781c0f67f0` into
`insights/schema_release_contract.py`. `contracts/database-schema/artist-similarity/v1/source.json`
records source/binding hashes and the exact producer pin. `scripts/check-contracts.py`
compares every promoted helper/constant AST against the dev-only producer package;
future drift fails the gate. Production imports the binding without installing the schema
initializer or its clients. The restricted-role PG19 tier verifies real COPY, rollback,
publication, current/previous retention, and retirement.

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

Historical rank-row measurement, superseded by the compact contract above. Measured on a
throwaway PG19 container with database-schema's `fecb0a4` DDL: 2,000,000 synthetic rows with a real-length stored `model_version` (148 characters).

| Relation | Bytes per row |
| --- | ---: |
| Heap | 215.6 |
| Primary key `(artist_id, model_version, rank)` | 327.1 |
| `(model_version, artist_id)` index | 16.2 |
| **Total** | **559** |

At K=50 the catalog is 468.3M rows, about **262 GB per release**, or about 520 GB with the previous release retained. At K=10 it is about 52 GB per release. Most of the cost is the 148-byte `model_version`, repeated on every row in both the heap and the primary key. The same lists stored one row per artist (id, a small release id, a `text[]` of ids, a `real[]` of scores) measured 943 bytes per artist, about 8.8 GB per release at K=50. The approved compact producer measurement is 942.08 bytes per artist (1M synthetic artists,
K=50, PG19), approximately 8.8 GB per release. The adapter now follows that compact
contract. This projection excludes WAL, embeddings, replicas, spare disk and dead tuples;
actual identifiers/compression can change it.

## Operations

- **Disk.** The spool directory needs about 11.3 GB free for the full catalog at K=50: the 3.75 GB spool plus both the
  previous and temporary replacement checkpoints (about 3.78 GB each). Atomic replacement
  temporarily retains both checkpoint files. Add database capacity separately: about
  17.6 GB for current/previous compact releases, before WAL/temp-file headroom. On an
  ongoing monthly rotation, the next target coexists with both retained releases until
  publication/retirement, so peak lists alone can be about 26.4 GB (three releases). It must survive a process restart for a stopped run to resume. The directory for a published `model_version` is removed after publishing.
- **Stopping and resuming.** Killing the process loses at most one checkpoint interval (30 minutes). Rerunning the pipeline for the same dump skips the already-loaded embeddings, reads them back, and resumes the spool from its checkpoint.
- **Rolling back.** Republishing the previous `model_version` with `publish_artist_embedding_release` switches `catalog-api` back. Its rows are still present, because only releases older than the previous one are retired.
- **Validation.** For the real-dump check, independently recompute exact top-10 for a 2,000-artist sample (tie-tolerant recall 1.0, ties within 1e-4) and measure Aug → Sept top-10 Jaccard. These runs are local-only. The earlier 8ts churn result is context; it does not validate this batch
  writer. Full-catalog stored-list recall/churn results are still required before acceptance.

## Current continuation status (2026-10-02)

The same developer actor preserved all five existing implementation commits. Lightweight
archive-header inspection confirmed local August and September `w0=0, self=0.05` snapshots:
9,330,617 and 9,366,416 artists respectively, 128-dimensional `float16` vectors. Scalar
metadata confirms algorithm v2, weights `0,1,1,1,1`, and `self_weight=0.05`. The NPZ
method-version metadata omits the edge-set suffix; `edges-v3` provenance is the prior
8ts documented build and the local filenames, not an independently verified NPZ field. No artist
ids or vectors were exported or committed. The previous sizing measurements above remain historical evidence. The fresh bounded
measurement is reported below.

Only 12–13 GiB was free on the shared APFS volume during continuation. A full compute
with atomic checkpoint replacement plus two compact releases needs approximately 29 GB
before WAL, embedding storage and operational slack for a fresh August/September pair.
An ongoing monthly rotation can temporarily need about 37.7 GB with three releases
plus spool/checkpoints. The full run and real September
stored-list recall / August→September churn validation were therefore **not run**.
This bead is not ready for acceptance until those checks can be completed on adequately
provisioned storage with an isolated host resource window. No Docker settings or existing
data were changed, and no prune was performed.

`scripts/measure-exact-top-k-sizing.py` provides a bounded 500k–1M sizing pass. For a local
NPZ, it reads only the requested prefix of `vectors.npy` (128 MB for 500k artists), never
artist ids, and removes only its own temporary spool. It reports wall time, getrusage peak
RSS, quadratic full-catalog extrapolation and the 12 GB / six-hour comparisons. A 500k
run at the production 4096-row blocks and eight threads estimates about 2.48 GB kernel
RSS and at most about 0.61 GB spool/checkpoint disk, plus a conservative free-space margin.
It must be serialized with other host work.

### Fresh bounded September sizing

The 2026-10-02 continuation ran only the first 500,000 September vectors. A temporary
wrapper used the installed Beadhive admission API with actual configuration (`capacity=2`),
held both distinct permits `[0, 1]` through `ExitStack`, and waited 17.09 seconds for existing
validation to finish. It changed no admission configuration or environment overrides. Only
one child ran while permits were held; the child timeout was 900 seconds. Both permits and
the wrapper-owned temporary spool directory were released normally.

Command inside that admitted window (local snapshot, never committed):

```sh
.venv/bin/python scripts/measure-exact-top-k-sizing.py \
  --snapshot ~/.cache/groovemap-spikes/embeddings-scratch/sept_v3.w0-0.self-0.05.npz \
  --rows 500000 --threads 8 --block-rows 4096 --scratch-root /private/tmp/<owned-directory>
```

| Measurement | Result |
| --- | ---: |
| Artists / dimensions / K | 500,000 / 128 / 50 |
| Kernel wall time | 58.3203 s |
| Child elapsed, including input read | 59.1190 s |
| Total elapsed, including admission wait | 76.2099 s |
| `getrusage` peak RSS | 2,517,483,520 bytes (2.52 GB) |
| Temporary spool | 200,000,000 bytes; no checkpoint was due |
| Free disk before / after | 13,394,345,984 / 13,369,331,712 bytes |
| Conservative quadratic full-catalog wall time | 20,465.69 s (5.685 h) |
| Estimated full kernel / pipeline peak (including held ids) | 8.367 / 8.966 GB |

The elapsed projection is `58.3203 × (9,366,416 / 500,000)²`; it includes startup selection
cost and remains an estimate, not a full-catalog measurement. Runtime and memory fit the
six-hour / 12 GiB envelope on this isolated host window. Disk does not: the full compute,
COPY/publication, 2,000-sample stored-list recall and monthly Jaccard checks remain **not run**.
No full job was started after the sizing result. The benchmark was an aggregate sizing
experiment, not a production release publication or real acceptance substitute.

If the fresh extrapolation exceeds about six hours or the RSS budget, stop before the
full job. Options for an explicit maintainer decision are an approximate candidate pass
with exact reranking, sharding on suitable hosts, or a documented minimum-degree subset.
These change the current whole-catalog exact acceptance and must not be substituted silently.
Adequate scratch/database storage and an isolated run preserve the current acceptance.

### Independent acceptance-comparator preparation (2026-10-03)

`scripts/independent_exact_top_k.py` provides a comparator that does not import the
production exact kernel. It streams the whole candidate catalog separately for each
bounded query batch: defaults are 32 queries × 65,536 candidates (an 8 MiB float32
score matrix), with independently normalized float32 vectors. It never materializes
2,000 × 9.37M scores or holds both months' vectors. Input positions must correspond
to lexically sorted artist IDs; snapshot replay must establish that order before
both production computation and independent comparison. Exact score ties sort by
position, including ties across candidate chunks.

`compare_stored_top10` accepts actual persisted scores separately from independently
computed top-10 reference lists. It independently rescores each stored neighbour
from the same vectors/query positions. It requires ten unique, in-range neighbours
per query, rejects self and nonfinite data, checks canonical persisted ordering,
and reports persisted-score maximum absolute error and violation counts even when
every stored ID belongs to the reference set. Boundary-tolerant recall uses actual
recomputed cosine scores, never the database score. Exact set recall, per-query
minimum tie-tolerant recall, `1e-4` tolerance, score errors and order violations remain
separate aggregates; passing requires full tie-tolerant recall and zero score/order
violations. Eight synthetic regressions cover the independent full-matrix oracle,
last candidate chunk, zeros/ties, forged scores, self/duplicate/out-of-range IDs,
nonfinite scores, cardinality and lexical ordering.

This comparator is preparation, not stored-list acceptance evidence. The initial
automatic approval rejection of the snapshot transfer was resolved with concrete
operator authorization, destination ownership and public-catalog provenance.
The coordinator transferred both exact snapshots, and destination checksums were
verified before the fresh bounded remote calibration below. No full batch or
stored-list recall / August→September Jaccard result has been produced.

### Fresh remote calibration: time admission failed

The isolated x86_64 validation image was built from clean source
`c0625f217f9e637420a389afda08f7a36ccb99a7`, with the immutable runtime/schema pins
above. Its image ID was
`sha256:82ba788fae6cafac06ce54a4aefcddf3fc556a9e85a4425b584e5a26c41ef5f1`.
The source archive SHA-256 was
`0eeeb88fa95aac223ce811af79dd32818da57181d6f9d846465d076d0c59982d`;
the source/wheel bundle hashes were verified at the destination. This was a
validation-only local image, not a production release or published artifact.

The actual inputs were `aug_v3.w0-0.self-0.05.npz` (SHA-256
`47c85fed6d6a1dc5f93cfccac5d254aedb24180bc8184e2846dd2a41e5747cb6`)
and `sept_v3.w0-0.self-0.05.npz` (SHA-256
`de5290fcf2dab3e701a9e16b86ebc4befc2793a57cbf28446c8e975967b033a4`).
The NPZ scalar metadata independently verifies algorithm v2, 128 dimensions,
weights `(0,1,1,1,1)`, self weight `0.05`, and the respective dump IDs. The declared
edges-v3 lineage comes from prior 8ts provenance; it is not independently verified
by the NPZ metadata or inferred from filenames/current `_EDGE_SET_VERSION`.
These are distinct inputs from the old `sept.npz` whose edges-v2 labelling pitfall
is documented in [recall and churn](recall_and_churn.md#model-version-labelling-pitfall).
Algorithm v2 and edge-set version are separate concepts.

On 2026-10-03 the committed sizing script read only the first 500,000 September
vectors and ran exact K50 with 4,096-row blocks. The Intel Core Ultra 5 235HX host
had sufficient free disk and RAM. Python 3.14.7 / NumPy 2.5.3 used OpenBLAS
0.3.34.106.0, eight kernel workers and one BLAS thread per worker. An exclusive
validation lock and coordinated resource window serialized the trial. Container
limits were eight CPUs, 10.5 GiB memory with no additional swap, 256 PIDs,
read-only inputs/source, no network, and a 1,800-second deadline. The isolated PG
allocation was reserved at 1.5 GiB but no database was started; combined planned
limits remained 12 GiB.

| Measurement | Actual / projection |
| --- | --- |
| Kernel / whole process wall time | 112.5901 / 113.8003 s |
| `getrusage` peak RSS | 1,938,362,368 bytes (1.94 GB) |
| Cgroup memory peak / hard limit | 1,936,039,936 / 11,274,289,152 bytes |
| Conservative full September time | 39,509.9920 s (10.975 h) |
| Estimated full kernel / pipeline peak | 8.367 / 8.966 GB |
| Free disk after trial | approximately 1.114 TB |

The process exited normally, its temporary sizing scratch/container were removed,
and release of the exclusive lock was verified. No matching Docker OOM event was
observed in the retained trial interval. Cgroup `memory.events` and pressure
counters were not captured before container removal; no zero-counter claim is
made. The aggregate log SHA-256 is
`90755bd8d1f1c5789b33f69f2933011544b0d7c65c2586206d9de746824b4665`.

**Do not start the full batch on this result:** memory and disk fit, but the
projected monthly compute exceeds the approximately six-hour limit. COPY and
publication are not included in that time projection. Stored September top-10
recall, August→September Jaccard and real publication/retention remain unmeasured.
Bounded CPU/BLAS tuning or a faster host/exact sharding may preserve whole-catalog
exactness, but require fresh measured admission. Approximate candidate reranking
or a minimum-degree subset changes the acceptance and requires an explicit
maintainer decision. No approximation, subset, time-budget relaxation or full-run
result has been substituted.
