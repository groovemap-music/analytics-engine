# FastRP self term: tie-break re-embedding and D serving-mode verdict

Status: **complete.** This document records gm-analytics-engine-8ts's re-embedding of the
2026-08 and 2026-09 Discogs dumps with edges-v3 and a deterministic FastRP self term
(`self_weight = 0.05` on top of `w0 = 0`), following up gm-analytics-engine-i37's finding
(docs/embedding_weight_sweep.md) that 42.22% of edges-v3 artist vectors are exact
byte-duplicates at every `w0` swept, because nothing in FastRP's sum carries a node's own
identity — only its neighbourhood's. Measured against i37's own w0=0 numbers throughout.

No provider-derived data (ids, names, vectors, edges) is committed anywhere in this
repository; only this document, the measurement scripts, and aggregate counts are.

## Headline numbers

- **The self term eliminates the duplicate-vector problem outright: 42.22% → 0.0%.**
  `shipped_file_dup_group_pct` in the chw.2 mapping metadata for September's `self=0.05`
  file is exactly `0.0`. `self_weight * normalize(R[v])` — `R[v]` a pure function of node
  `v`'s stable key — is distinct for every node with overwhelming probability, so no two
  artists share a vector regardless of how identical their one-hop neighbourhood is.
- **Recall improves, but not enough to clear the serving bar.** Strict recall@10 at
  `ef_search = 1000` is 0.8335 (August) / 0.8290 (September) — 3.4–3.8 points above i37's
  0.7953 (August) / 0.7952 (September) — but still below the maintainer's 0.85 threshold at
  every swept `ef_search`, on both months.
- **Exact churn improved sharply (0.8886 → 0.9519), but ANN churn only partly followed
  (0.6888 → 0.7243), so the index/exact gap widened, not narrowed: 0.20 → 0.23.** Breaking
  ties makes the underlying embeddings noticeably more stable month over month, but the
  served HNSW index still disagrees with exact brute force on which member of a
  now-*near*-tied (rather than exactly-tied) group lands in the top 10 often enough that the
  gap is *larger* than before, not smaller. This is the single most important number in this
  document for the D serving-mode decision below.
- **chw.2 quality improved and clears i37's own number**: fused recall@10 (test split, "all"
  view) is 27.23% at the dev-selected α=0.8, against i37's 27.19% for w0=0 — a small, real
  gain, not a regression. Gain over the all-artist heuristic baseline (17.99%) is +51.4%
  (bootstrap 95% CI [+45.0%, +57.9%]).
- **Quiet-machine latency, finally measured off-load**: mean/p95 24.8/32.8 ms at
  `ef_search=200`, 38.7/46.3 ms at 400, 69.8/84.1 ms at 800, 84.7/101.2 ms at 1000 — well
  below i37's own "upper bounds only" numbers (91–296 ms mean at `ef_search=1000`, measured
  under host load). These are the first real (not load-inflated) latency numbers for this
  edge set.
- **With ties broken, the standard (`m=16`) index build itself got measurably heavier**:
  August's build overflowed `maintenance_work_mem=8GB` near the end (approximately 9.15M of
  9,330,617 rows already inserted, HNSW graph-construction candidate lists no longer fitting
  in the configured work memory) and fell into IO-bound behaviour for its last stretch (about
  14–27 rows/s observed), taking 82.4 minutes overall against September's 42.3 minutes for a
  near-identical row count built at the same settings without hitting the same wall. This is
  a direct, structural consequence of eliminating duplicates: a build that could previously
  skip or short-circuit graph-construction work for byte-identical candidate vectors now has
  to do that work for every one of them. See "Index-build memory finding" below.
- **D serving-mode verdict: do not serve from the index.** All three of the maintainer's
  thresholds fail (ANN churn ≥ 0.85, ANN churn within 0.05 of exact, strict recall@10 ≥ 0.85
  at a named `ef_search`) — see "D serving-mode verdict" below for the numbers against each.
  Recommend precomputed monthly exact top-10 lists instead. The self term is still worth
  keeping regardless of that verdict: it is what makes exact top-10 (what would actually ship
  under that recommendation) trustworthy — 0 duplicate-vector ties, 3.4–3.8 points more
  strict recall, and exact churn up to 0.9519.

## Method

Reused gm-analytics-engine-i37's own harness unchanged (`scripts/embeddings_from_dump.py`,
`scripts/measure_recall_churn.py`, `scripts/measure-embedding-quality.py`, the chw.2 harness
checkout and its already-built 10% artist-seeded subset), plus this bead's own additions:

- **A configurable FastRP self term** (`insights/embeddings/fastrp.py`, gm-analytics-engine-8ts):
  `FastRPConfig.self_weight`, a weight on `normalize(R[v])` — the node's own hashed
  projection row, obtained from the same per-block projection call `fastrp()` already makes
  before propagating anything, so no extra materialisation of `R` is needed. Named in
  `model_version` (`:self=0.05:` after `:beta=...:`), `FASTRP_ALGORITHM_VERSION` bumped to 2.
  The default (`0.0`) is bit-identical to the pre-8ts sum, proven with a pinned hash of
  i37's own merge-commit output as a parity test.
- **One config, both months, no sweep**: `w0=0`, `self_weight=0.05`, edges-v3, both dumps
  re-embedded from scratch (dumps were re-fetched — resumable, sha256-verified against each
  dump's own published `discogs_<id>_CHECKSUM.txt` via the `data.discogs.com` proxy, since
  the real S3 bucket 403s). `0.05` was not swept against `0.02`/`0.1` — the bead's time
  budget went to getting one config fully measured (including the quiet-machine latency
  pass) rather than a second sweep; see "Follow-up" below.
- **The full recall/churn flow, not the trimmed one**: `measure_recall_churn.py` WITHOUT
  `--skip-ann-churn` — both months' standard HNSW indexes built and measured, and the real
  ANN churn between them — since there is only one config here, not a sweep to trim.
  `--skip-larger-variant` still applies (`m=32` remains unmeasured; i37's overflow finding
  for it stands unchanged, this bead didn't re-attempt it).
- **A quiet-machine latency pass** (new for this bead, `8ts_latency_pass.py`): waits (up to
  12h) for `kern.memorystatus_level >= 60` and 1-minute load average `< 4`, then re-measures
  mean/p95 latency at `ef_search ∈ {200, 400, 800, 1000}` against the still-live September
  index, reusing `measure_recall_churn.py`'s own tested query-sampling and per-query-timing
  helpers rather than reimplementing them. The host was quiet at the start
  (`memorystatus_level=65`, 1-minute load average 3.11) and had drifted to load 5.08 by the
  end of the four-point sweep — still well under the 4.0 gate for any individual measurement
  already taken, each `ef_search` point was gated independently.

Every other measurement detail — deterministic query/churn sampling, tie-tolerant recall's
1e-4 threshold, degree-bucketed recall, the real DDL and `_write_embeddings` path — is
unchanged from docs/recall_and_churn.md and docs/embedding_weight_sweep.md; this document
does not repeat that background.

## Comparison against i37 (w0=0, both self_weight=0 and self_weight=0.05)

### Duplicate-vector share

| | i37 (`self_weight=0`) | This bead (`self_weight=0.05`) |
| --- | ---: | ---: |
| Shipped-vector exact-duplicate share (September) | 42.22% | **0.0%** |

### Strict and tie-tolerant recall@10 vs. `ef_search`

Real per-`model_version` partial HNSW index, `m=16, ef_construction=64`,
`maintenance_work_mem=8GB` both months, both runs:

| `ef_search` | Aug strict (i37) | Aug strict (8ts) | Aug tie-tol. (i37) | Aug tie-tol. (8ts) | Sept strict (i37) | Sept strict (8ts) | Sept tie-tol. (i37) | Sept tie-tol. (8ts) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 0.4926 | 0.5638 | 0.5196 | 0.5820 | 0.5124 | 0.5640 | 0.5449 | 0.5840 |
| 100 | 0.5988 | 0.6618 | 0.6346 | 0.6829 | 0.6128 | 0.6523 | 0.6511 | 0.6748 |
| 200 | 0.6728 | 0.7236 | 0.7157 | 0.7459 | 0.6776 | 0.7158 | 0.7225 | 0.7390 |
| 400 | 0.7323 | 0.7734 | 0.7792 | 0.7958 | 0.7330 | 0.7686 | 0.7812 | 0.7912 |
| 800 | 0.7837 | 0.8197 | 0.8316 | 0.8420 | 0.7828 | 0.8169 | 0.8314 | 0.8393 |
| 1000 | **0.7953** | **0.8335** | **0.8454** | **0.8554** | **0.7952** | **0.8290** | **0.8449** | **0.8511** |

Strict recall@10 is up 3.4–3.8 points at `ef_search=1000` on both months, and the gain holds
(or grows) at every smaller `ef_search` too — breaking exact ties gives the ANN index
genuinely more to work with, not just a headline-number improvement at one setting. No
`ef_search` reaches the maintainer's 0.85 strict-recall threshold on either month; 1000 is
also the pgvector maximum for `hnsw.ef_search`, so there is no larger value left to try.

### Churn (10,000-artist deterministic common-artist sample, August → September)

| | i37 (`self_weight=0`) | This bead (`self_weight=0.05`) | Change |
| --- | ---: | ---: | ---: |
| Exact cosine (raw vectors, no index) | 0.8886 | **0.9519** | +0.0633 |
| Served ANN index, `ef_search=1000` | 0.6888 | **0.7243** | +0.0355 |
| Index/exact gap | 0.1998 | **0.2276** | +0.0278 (wider) |

The self term makes the embeddings themselves noticeably more stable month over month
(exact churn +6.3 points) — expected, since a deterministic per-node term now anchors part
of each vector regardless of neighbourhood churn. The served index followed only about half
as far (+3.6 points), so the fraction of month-over-month movement attributable to the index
disagreeing with exact brute force (rather than the data itself moving) *grew*, from about
0.20 to about 0.23. Ties being broken changes *which* near-neighbours the ANN index and
exact top-10 disagree about (a near-tie rather than an exact tie), it does not close that
gap; see docs/recall_and_churn.md's tie-structure discussion for the underlying mechanism,
which this bead's numbers say is not eliminated by a self term, only reshaped.

### chw.2 proxy-benchmark quality (test split, recall@10, "all" view)

| | i37 (`self_weight=0`) | This bead (`self_weight=0.05`) |
| --- | ---: | ---: |
| Embedding alone | 26.26% | 26.13% |
| Fused @ dev-selected α | 27.19% (α=0.8) | **27.23%** (α=0.8) |
| Gain over all-artist heuristic (17.99%), 95% CI | +51.17% [+44.65%, +57.75%] | **+51.4%** [+45.0%, +57.9%] |

The embedding-alone score is essentially flat (a fractional decrease, within noise of the
benchmark's own bootstrap CI width); the fused score — the production-relevant number, since
fusing with the heuristic is what would actually ship — improves slightly and clears i37's
own number. chw.2's queries are active seed artists who almost always have a distinguishing
neighbourhood already (i37's own finding: only 0.1% of test queries had a shipped vector in
a duplicate group even at 42.2% overall duplication), so this benchmark was never going to
show the self term's effect as strongly as the duplicate-share or churn numbers above — it
mostly confirms the self term does not *hurt* the signal chw.2 measures, while directly
fixing the artifact chw.2 structurally could not see.

### Quiet-machine latency (September, `self_weight=0.05`, off-load)

| `ef_search` | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| 200 | 24.8 | 32.8 |
| 400 | 38.7 | 46.3 |
| 800 | 69.8 | 84.1 |
| 1000 | 84.7 | 101.2 |

Host state: quiet at the start of this pass (`memorystatus_level=65`, 1-minute load average
3.11), drifted to load 5.08 by the time the fourth (`ef_search=1000`) point finished — still
comfortably under load for the pass overall, and each point's own host state was checked
before that point's measurement began. These numbers supersede i37's own "upper bounds
only" figures (91–296 ms mean at `ef_search=1000`, measured under sustained host load) as
the first real off-load latency characterization of this edge set and index configuration.

## FastRP time and peak RSS

Peak RSS during `scripts/embeddings_from_dump.py`'s stream-parse + FastRP pipeline
(`resource.getrusage`, footprint-aware `wait_for_memory` guard active throughout — see
"Memory-guard incident and fix" below for why that guard's own correctness mattered this
time):

| Phase | August | September |
| --- | ---: | ---: |
| `same_as` map (pass 1) | 5.31 GB | 5.33 GB |
| Full parse (pass 2, both dumps) | 7.34 GB | 7.06 GB |
| Parser structures freed, pre-build | 9.94 GB | 8.37 GB |
| `AdjacencyBuilder.build()` (graph, parser still resident) | **12.01 GB** | **10.70 GB** |

| Phase timing | August | September |
| --- | ---: | ---: |
| Parse (both passes) | 7230.0 s (120.5 min) | 6977.4 s (116.3 min) |
| Graph build | 338.3 s (5.6 min) | 352.4 s (5.9 min) |
| FastRP alone | 651.2 s (10.9 min) | 979.3 s (16.3 min) |

Both months' peaks (12.01 GB, 10.70 GB) land comfortably under docs/embeddings.md's ~12 GB
budget and under i37's own edges-v3 range for graph construction (17.27–19.26 GB was i37's
figure for the same phase, self_weight=0 — this bead's own numbers are markedly lower,
consistent with i37's own note that graph-construction peak varies with host conditions at
the time of the run, not with FastRP configuration).

### Memory-guard incident and fix

September's initial run of this bead deadlocked in `wait_for_memory`'s own footprint-aware
guard: `process_footprint_bytes()` preferred `psutil`'s live RSS reading, and idling in that
function's own poll loop is exactly when macOS is most likely to page out or compress this
process's inactive resident pages — the underlying allocations (the parsed `same_as` map,
the graph being built) were still held and would have been paged straight back in the moment
the next phase touched them, but the live reading collapsed from 6.22 GB to 0.02 GB across
five consecutive 60-second polls anyway. Since `required_free = expected_peak - footprint +
margin`, a footprint reading that shrinks while waiting makes `required_free` *rise* each
poll instead of converging — it reached 19.98 GB, a bound this host cannot satisfy while
Colima holds 16 GiB of it, an unconditional deadlock, not a slow wait. The `expected_peak_gb`
value in use (18 GB, carried over from an older edges-v2-era measurement) was also stale
against this run's own real measured peak (12.01 GB, table above).

Fixed in commit `66608ed`: `process_footprint_bytes()` now prefers `resource.getrusage`'s
`ru_maxrss` (monotonic, never decreasing for the life of the process) over the live psutil
reading, with psutil demoted to a fallback. Covered by
`tests/test_embeddings_from_dump_memory_guard.py` (preference order, the psutil fallback
path, a real allocate-then-free monotonicity check, and three `wait_for_memory` convergence
cases). September's parse had no on-disk checkpoint at the time of the deadlock (omitted
deliberately for this bead's tight disk budget — see "Disk budget" below), so it was
restarted from scratch under the fixed code and the corrected `--expected-peak-gb 12.5`;
it converged immediately on that restart and completed normally (see the timing table
above, which reflects the successful restarted run).

### Disk budget

Both months' dumps (releases ~11.2–11.3 GB, masters ~0.6 GB each) were downloaded, parsed,
and deleted (`--delete-dumps-after-parse`) one month at a time rather than kept
simultaneously — this host had only about 20 GB free for the whole bead, and a graph or
same_as checkpoint would not reliably have fit alongside an 11.6 GB local dump under the
maintainer's 8 GB floor, so neither was used; a crash after this bead's own memory-guard fix
would have cost a redo of that month's parse, not silent corruption or a violated disk
floor (the write path's own `wait_for_disk` guard never writes without the required
margin).

## Index-build memory finding

With duplicate vectors eliminated, the standard (`m=16, ef_construction=64`) HNSW build
itself became measurably more expensive for August: `maintenance_work_mem=8GB` was
exhausted near the end of that build — at approximately 9.15M of August's 9,330,617 total
rows — after which the build fell into IO-bound behaviour (observed throughput dropping to
roughly 14–27 rows/s for its final stretch) before completing. August's build took 4944.0 s
(82.4 min) against September's 2540.2 s (42.3 min) for an almost identical row count
(9,366,416) built at the same settings, on the same host, back to back — the ~2× slowdown is
consistent with that near-overflow, not noise.

This is a direct, structural consequence of the self term working as intended: when many
candidate vectors in a neighbourhood were byte-identical, HNSW's graph-construction
candidate-list bookkeeping could short-circuit or dedupe comparisons among them cheaply;
with every vector now distinct, that shortcut is gone and the build's working-set footprint
inside `maintenance_work_mem` grows accordingly. **Recommendation: budget approximately
10 GB of `maintenance_work_mem` for production HNSW builds on this catalog once ties are
broken** (up from the 8 GB used throughout this bead and i37's), rather than assuming the
pre-self-term memory profile still holds.

## D serving-mode verdict

Against the maintainer's three thresholds (serve from the index only if **all three** hold;
otherwise recommend precomputed monthly exact top-10 lists):

| Threshold | Required | Measured | Result |
| --- | --- | ---: | --- |
| ANN churn at a named `ef_search` | ≥ 0.85 | 0.7243 (`ef_search=1000`) | **FAIL** |
| ANN churn within 0.05 of exact churn | gap ≤ 0.05 | 0.2276 (0.9519 − 0.7243) | **FAIL** |
| Strict recall@10 at a named `ef_search` | ≥ 0.85 | 0.8335 (Aug), 0.8290 (Sept), both at `ef_search=1000` (the pgvector maximum) | **FAIL** |

**Verdict: do not serve similar-artist results from the live ANN index. Recommend
precomputed monthly exact top-10 lists instead**, computed once per monthly dump from the
raw embeddings (as this bead's own `exact_elapsed_s` figures show: 361.4 s for August,
283.8 s for September, well within a monthly batch job's budget) and served as a static
lookup, sidestepping the HNSW index's recall and churn gap entirely.

**The self term is still worth keeping regardless of this verdict.** It is precisely what
makes a precomputed exact top-10 list trustworthy month to month: zero duplicate-vector ties
(down from 42.22%), 3.4–3.8 points more strict recall at every `ef_search` measured, and
exact churn up to 0.9519 (up from 0.8886) — all properties of the *exact* computation this
recommendation would actually serve, independent of the ANN index's own recall/churn gap
that fails the three thresholds above. Reverting to `self_weight=0` to avoid the heavier
index build (see "Index-build memory finding") would reintroduce the duplicate-vector
problem into that same precomputed list.

## Follow-up

- **Self-weight sweep.** Only `0.05` was measured end to end here; `0.02` and `0.1` (allowed
  as optional by this bead's brief) were not, for time-budget reasons — the full flow (both
  months, full recall/churn/ANN-churn, chw.2 quality, and a quiet-machine latency pass) for
  one config already spans the timings in this document. If a future bead revisits the ANN
  churn/recall gap, sweeping `self_weight` the same way i37 swept `w0` (trimmed
  `--skip-ann-churn` per candidate, full flow for the winner only) would reuse this bead's
  `--self-weight` CLI flag (`scripts/embeddings_from_dump.py`) directly.
- **Why ANN churn didn't close proportionally to exact churn** is not isolated by this
  bead's measurements — plausible mechanisms (near-tie neighbourhoods still landing
  differently in the ANN graph vs. exact brute force; HNSW's approximate nature simply having
  more near-equally-good candidates to choose between now that the self term perturbs
  formerly-identical vectors) are consistent with the data but not distinguished by it.

## Environment and versions

- numpy 2.5.3, scipy 1.18.1 (both months, same build).
- PostgreSQL 19 + pgvector (`database-schema-postgres19-pgvector:local`), HNSW
  `m=16, ef_construction=64` for every measured index (`m=32` not attempted; i37's overflow
  finding for it stands, unchanged by this bead).
- Dump ids: `discogs_20260801` (August), `discogs_20260901` (September).
- Stored `model_version`s: `fastrp-v2:dim=128:weights=0,1,1,1,1:beta=0:self=0.05:
  proj=achlioptas-s3:rows=splitmix64(blake2b64(kind,key)):seed=20260924:edges-v3@discogs_2026{08,09}01`.
- chw.2 harness: i37's already-built checkout and 10% artist-seeded subset, reused unchanged
  (not rebuilt) — see docs/embedding_weight_sweep.md for how that subset was originally
  produced.

## Cleanup

The throwaway `gm-8ts-recall-pg` container and its rows/index were removed at the end of the
measurement run. No provider-derived data is committed to this repository; only this
document, the code changes (self term, memory-guard fix, `--self-weight` CLI flag), and the
aggregate numbers above are.
