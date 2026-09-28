# Edges-v3 re-embedding: step-0 weight sweep, recall, churn, and quality

Status: **complete.** This document records gm-analytics-engine-i37's re-embedding of the
2026-08 and 2026-09 Discogs dumps with the landed edges-v3 edge set (gm-analytics-engine-x3d:
adds `graph.track_credited_on` — track and sub-track credits, resolved through `same_as`
like release-level credits — and `graph.track_by_artist` — track performers, on top of
edges-v2's release-level `graph.credited_on`), for a step-0 FastRP weight sweep
`w0 ∈ {0, 0.1, 0.25}` (weights
`w0,1,1,1,1`, everything else unchanged; each `w0` is its own `model_version`). For each
weight: September recall@10 (strict and tie-tolerant) vs. `ef_search`, month-over-month
exact churn, and chw.2 proxy-benchmark quality. The single best weight by the stated rule
then got the full flow: both months' standard HNSW indexes, and ANN churn between them.

No provider-derived data (ids, names, vectors, edges) is committed anywhere in this
repository; only this document, the measurement scripts, and aggregate counts are.

## Headline numbers

- **The sweep's premise did not hold: the duplicate-vector share is 42.2% at every weight,
  identical to eleven decimal places (`shipped_file_dup_group_pct = 0.42218656527747644` in
  all three `i37_quality.w0-{0,0.1,0.25}.json`).** FastRP's embedding is `sum_k w_k *
  normalize((P^(k+1) R)[v])` (`insights/embeddings/fastrp.py`'s own docstring) — the `k=0`
  term `w0` weights is `P¹R`, the **one-hop neighbour mean**, not the node's own raw
  projection `R`. Two artists with an identical one-hop neighbourhood get an identical
  `P¹R` (and identical higher-order terms) regardless of what `w0` is, including `w0=0` —
  there was never a "self" term in this formula for `w0` to reintroduce. Reweighting that
  shared term (0 → 0.1 → 0.25) is exactly why recall and chw.2 quality moved a little across
  the sweep while the duplicate share did not move at all. **A real fix needs a genuine self
  term** — the node's own untransformed identity, distinct from `P¹R` (e.g. a `k=-1`/`"self"`
  weight) — **or an explicit deterministic tie-break**, neither of which this bead
  implements; re-measure duplicate share, recall, and ANN churn once either lands. See
  "Winner selection" below for what this means for the ranking below.
- **w0=0 still wins on chw.2 quality and is the recommended weight of the three actually
  swept**, by the rule below. The margin over the runner-up (w0=0.25) is real, but since none
  of the three weights touches the duplicate-vector mechanism above, this ranking says which
  one-hop reweighting chw.2 prefers, not which weight "fixes" the duplicates — none of them
  do.
- **The served ANN index adds about 0.20 of churn on top of the data's own move.** The
  winner's month-over-month churn is 0.8886 on exact cosine but only **0.6888 on the served
  index at `ef_search = 1000`** — a materially larger index/exact gap than edges-v2 saw on
  the same comparison (0.8026 index vs. 0.9083 exact, a gap of about 0.11). Don't read past
  this number without noting it: whatever ships behind the served kNN endpoint will look
  noticeably less stable month over month than the underlying embeddings do.
- **Recall is lower than edges-v2's.** Best observed strict recall@10 at `ef_search = 1000`
  is 0.7745–0.7972 across the three weights (edges-v3), against edges-v2's 0.8429 (August).
  Edges-v3's own duplicate-vector share is also higher: 42.2%, against edges-v2's 38.7%.
  Widening the credit-edge scope (x3d) did not improve — and may have worsened — the tie
  structure that already limited edges-v2's recall (see docs/recall_and_churn.md's "Recall
  and tie structure").
- **The larger-index variant (`m = 32`, `ef_construction = 128`) was not measured.**
  September's build for w0=0 overflowed `maintenance_work_mem = 8GB` at about 6.1M of
  9,366,416 tuples and fell to an on-disk build path (about 80 tuples/s), abandoned after
  about 4h50m having reached roughly 7.01M tuples. The overflow itself is the finding at
  this catalog scale — see "Larger-index variant: not measured" below.
- **FastRP's peak RSS (9.87–14.33 GB) brackets docs/embeddings.md's own ~10.1 GB edges-v3
  estimate rather than confirming it**, and September's `w0=0.1`/`w0=0.25` (14.33 GB each)
  land 4.2 GB above that estimate — eating the estimate's already-thin margin and leaving the
  overall 12 GB budget with none to spare at those two points. Graph construction with the
  parser still resident peaked meaningfully higher still (17.27–19.26 GB, both months, every
  weight). See "Memory findings" below.
- **All latency numbers here were measured on a heavily loaded host and are upper bounds
  only.** Mean per-query latency at `ef_search = 1000` ranged 91–296 ms across every run in
  this bead; a quiet-machine re-measure is still owed before these numbers inform any
  serving-latency decision.

## Method

Same measurement harness as gm-analytics-engine-ieu.3/kn3 (`scripts/measure_recall_churn.py`
against a throwaway PostgreSQL 19 + pgvector container, real DDL, real
`insights.embedding_pipeline._write_embeddings`), extended this bead with:

- **Per-w0 ANN churn was trimmed to the winner only, by maintainer decision.** Every weight
  except the winner ran `measure_recall_churn.py --skip-ann-churn`: September's write, index,
  and recall sweep, plus EXACT-only churn against August's raw vectors (no Postgres write, no
  index, no ANN churn for August at all). Only the winning weight (w0=0, chosen from these
  trimmed results — see "Winner selection") re-ran the full, untrimmed flow: both months'
  standard indexes built and measured, and the real ANN churn between them. This is why every
  per-weight table in this document reports exact churn for all three weights but ANN churn
  ("Winner's full run" below) for w0=0 alone.
- **`--skip-larger-variant`**: drops the `m=32` build entirely, in both months and both the
  trimmed and full flows, once the overflow above made it clear finishing even one such
  build would cost hours, repeated per weight and again for the winner run.
- **Streaming exact top-k** (`_iter_npz_vector_chunks`, `_extract_rows`,
  `_stream_exact_top_k`): a month's vectors are read from its `.npz` in bounded chunks
  rather than held as one full in-memory array, after holding both months' full float32
  arrays at once (~10 GB combined) was a major contributor to a host memory/disk crisis
  partway through this bead (host swapped to 30 GB, free disk fell to 198 MB). Measured
  peak RSS on the real 9.3M-artist catalog: **4.0–4.7 GB**, against a 3 GB target validated
  separately on a synthetic 2M-row/128-dim array (2.4 GB peak there) — the gap is expected:
  the synthetic validation's per-chunk cost is independent of catalog size, but
  `artist_ids`/`degrees`/`id_to_position`/the normalized-vector cache all scale with it, and
  real scale is ~4.65× the synthetic test's.
- **A host-pressure guard** (`wait_for_host_pressure`) that pauses — polling, never
  aborting — before every heavy step and between `ef_search` sweep iterations, while the
  HOST (not the throwaway container) is short on memory or disk. Gates on
  `kern.memorystatus_level < 25%` (primary) and swap used `> 26 GB` (backstop) plus free
  disk `< 8 GB`; an earlier version gated on free swap instead, which is the wrong metric on
  macOS (swap files grow in ~1 GB increments and are rarely shrunk back, so free swap sits
  under 2 GB almost permanently even on a healthy host) and had to be fixed mid-bead.
- **Per-weight Postgres reclaim**: once a weight's recall+quality results are durably on
  disk, its HNSW index is dropped and its rows deleted from the shared
  `public.artist_embeddings` table, then `VACUUM`ed (not `FULL`) and `fstrim`'d — three
  weights' rows and indexes accumulating in the same table at once (17 GB heap + 4.4 GB pkey
  for two weights, +3.8 GB HNSW for a third) is what caused the memory/disk crisis above.

Every other measurement detail — deterministic query/churn sampling
(`_deterministic_sample`, seeded `splitmix64` ranking), tie-tolerant recall's 1e-4 threshold,
degree-bucketed recall, the real DDL and `_write_embeddings` path — is unchanged from
docs/recall_and_churn.md; this document does not repeat that background.

## Per-weight results (September, standard variant, `m=16, ef_construction=64`)

Strict and tie-tolerant recall@10 vs. `ef_search`, real per-`model_version` partial HNSW
index, `maintenance_work_mem = 8GB`:

| `ef_search` | w0=0 strict | w0=0 tie-tol. | w0=0.1 strict | w0=0.1 tie-tol. | w0=0.25 strict | w0=0.25 tie-tol. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 0.4913 | 0.5285 | 0.4953 | 0.5281 | 0.4749 | 0.5038 |
| 100 | 0.6004 | 0.6434 | 0.5974 | 0.6379 | 0.5751 | 0.6138 |
| 200 | 0.6732 | 0.7208 | 0.6652 | 0.7093 | 0.6513 | 0.6950 |
| 400 | 0.7363 | 0.7850 | 0.7209 | 0.7674 | 0.7055 | 0.7503 |
| 800 | 0.7835 | 0.8346 | 0.7734 | 0.8214 | 0.7581 | 0.8077 |
| 1000 | **0.7972** | **0.8486** | **0.7876** | **0.8371** | **0.7745** | **0.8243** |

No `ef_search` reaches recall@10 ≥ 0.95 for any weight; no production `ef_search` is named,
consistent with edges-v2's own finding. Recall degrades monotonically with `w0` — putting
weight on a node's own projection (`w0 > 0`) costs recall here, though see "Winner
selection" for why that's not decisive by itself.

Month-over-month exact-cosine churn (10,000-artist deterministic common-artist sample,
September vs. August's raw vectors, no ANN/index involved):

| `w0` | Mean top-10 Jaccard (exact) |
| --- | ---: |
| 0 | 0.8882 |
| 0.1 | 0.8857 |
| 0.25 | 0.8862 |

All three weights land within 0.003 of each other on exact churn — the embeddings' own
month-over-month stability barely depends on `w0` in this range.

chw.2 proxy-benchmark quality (test split, recall@10, "all" view — known + novel
collaborators; all-artist heuristic baseline 17.988%, reproduced exactly — see
docs/embedding_quality.md's own reproduction check):

| `w0` | Embedding alone | Fused @ dev-selected α | Gain over all-artist heuristic (95% CI) |
| --- | ---: | ---: | --- |
| 0 | 26.264% | **27.192%** (α=0.8) | **+51.17%** [+44.65%, +57.75%] |
| 0.1 | 26.066% | 26.740% (α=0.9) | +48.65% [+41.76%, +55.09%] |
| 0.25 | 25.673% | 26.868% (α=0.8) | +49.36% [+42.75%, +55.59%] |

**Gap to the design spike's own headline number:** the spike's fused FastRP
(`frp128_w01111` @ its own α=0.9) scored 31.73% on this same benchmark
(docs/embedding_quality.md's "Headline results"). The winner's fused result here (w0=0,
27.192% @ α=0.8) is **4.54 points below** the spike's 31.73% — a smaller gap than
docs/embedding_quality.md's own edges-v2 comparison found for the identical `w0=0`
configuration (6.40 points, 25.329% vs. 31.727%), consistent with edges-v3's wider credit
scope helping chw.2 quality somewhat even as it hurt ANN recall and churn (see "Edges-v3
against edges-v2" below).

Standard-variant index size and build time (September, this trim run):

| `w0` | Index size | Build time |
| --- | ---: | ---: |
| 0 | 4.020 GB | 84.3 min |
| 0.1 | 4.029 GB | 96.5 min |
| 0.25 | 4.042 GB | 49.6 min |

## Winner selection

**Winner: w0=0.** Rule (from the driver, verbatim): *"Ranked by chw.2 quality: FUSED
test-split recall@10 (`evaluate.py`'s own dev-selected fusion alpha per w0) on September.
Ties within 0.001 broken by September's standard-variant tie-tolerant recall@10 at
`ef_search=1000`."* w0=0's fused recall@10 (27.192%) beats the runner-up (w0=0.25, 26.868%)
by 0.32 points — outside the 0.001 tie band, so the tie-break rule was never invoked; w0=0
wins outright on quality.

Two things temper that result:

- **The rule favours quality over recall.** w0=0 also has the *best* recall@10 of the three
  weights at every swept `ef_search` (the table above), so quality and recall agree here —
  but the rule would have picked w0=0 on quality alone even had recall disagreed.
- **chw.2 barely exercises the duplicate-vector artifact all three weights share.**
  docs/embedding_quality.md's own "Duplicate-vector-group effect" found only 5 of 4,721 test
  queries (0.1%) have a shipped vector in a duplicate group, because chw.2's queries are
  active seed artists who almost always have a distinguishing neighbourhood — the same
  structural gap applies here across all three `w0` values in this sweep, at both 38.7%
  (edges-v2) and edges-v3's higher 42.2% duplicate share. chw.2's recall@10 numbers above are
  a real, reproduced measurement of the weights it *can* distinguish on, but they say little
  about how the choice of `w0` affects the long tail of duplicate-vector artists this
  benchmark structurally cannot query.
- **The sweep's premise — that `w0 > 0` breaks the duplicates — did not hold, for any of the
  three weights.** All three land on the identical 42.2% duplicate share ("Headline numbers"
  above): `w0` re-weights `P¹R` (the one-hop neighbour mean), never the node's own identity,
  so recall and quality moving slightly across the sweep while duplicates stayed frozen is
  exactly what the formula predicts, not a surprising result this measurement failed to
  explain. **Recommended follow-up, before drawing further conclusions from any `w0` choice
  here:** add a genuine self term (the node's own untransformed projection `R`, distinct from
  `P¹R` — e.g. a `k=-1`/`"self"` weight in `FastRPConfig`) or an explicit deterministic
  tie-break for exactly-tied top-10 candidates, then re-measure duplicate-vector share,
  recall, and ANN churn against that changed configuration.

## Winner's full run (w0=0, both months, ANN churn)

Real per-`model_version` partial HNSW index, `m=16, ef_construction=64`,
`maintenance_work_mem = 8GB` both months:

| `ef_search` | August strict | August tie-tol. | September strict | September tie-tol. |
| --- | ---: | ---: | ---: | ---: |
| 40 | 0.4926 | 0.5196 | 0.5124 | 0.5449 |
| 100 | 0.5988 | 0.6346 | 0.6128 | 0.6511 |
| 200 | 0.6728 | 0.7157 | 0.6776 | 0.7225 |
| 400 | 0.7323 | 0.7792 | 0.7330 | 0.7812 |
| 800 | 0.7837 | 0.8316 | 0.7828 | 0.8314 |
| 1000 | **0.7953** | **0.8454** | **0.7952** | **0.8449** |

Both months land in the same 0.7952–0.7972 range as the trim run's own September
measurement of the identical `w0=0` config (0.7972) — the small spread across three
independent builds of the same model_version is expected HNSW build-to-build variance, not
a real difference.

Month-over-month churn, 10,000-artist common-artist sample:

| | Mean top-10 Jaccard |
| --- | ---: |
| Exact cosine (both months' raw vectors) | **0.8886** |
| Served ANN index, `ef_search = 1000` | **0.6888** |

**Call this out explicitly:** edges-v3's exact churn (0.8886) is close to edges-v2's
(0.9083) — the underlying embeddings are about as stable as before — but the *served*
number is materially worse: 0.6888 against edges-v2's 0.8026. The index adds roughly 0.20
of churn on top of the data's own move here, against roughly 0.11 for edges-v2. Given
edges-v3 also has a higher duplicate-vector share (42.2% vs. 38.7%) and lower recall
(0.7952–0.7972 vs. 0.8429), the same tie-structure mechanism docs/recall_and_churn.md
describes (the ANN index and exact brute force validly disagreeing on which member of a
tied or near-tied group lands in the top 10) is the leading explanation, more pronounced
here than it was for edges-v2.

Index build (winner's full run — a **separate** build from the trim run's own September
index above, at a different point in the bead):

| Month | Index size | Build time | Write path |
| --- | ---: | ---: | --- |
| August | 4.008 GB | 57.1 min | `COPY` fallback (trial extrapolated past the 30-min budget) |
| September | 4.016 GB | **213.9 min** | real `_write_embeddings` (already loaded from a prior write) |

September's winner-run index build took 213.9 minutes against the trim run's 84.3 minutes
for the *same* `w0=0` config — consistent with "all latency numbers here were measured on a
heavily loaded host": this build ran later in the bead, well after the host was under
sustained memory/disk pressure from concurrent work, not because anything about the index
itself changed.

## Edges-v3 against edges-v2

| | Edges-v2 (ieu.3) | Edges-v3, this sweep |
| --- | ---: | ---: |
| Strict recall@10, `ef_search=1000` | 0.8429 (Aug), 0.8405 (Sept) | 0.7952–0.7972 (all three w0) |
| Exact churn (Aug→Sept) | 0.9083 | 0.8882–0.8886 |
| ANN churn @ `ef_search=1000` | 0.8026 | 0.6888 |
| Duplicate-vector share | 38.7% | 42.2% |
| chw.2 fused quality, w0=0 @ α=0.8 | 25.329% (docs/embedding_quality.md) | 27.192% |

Edges-v3 (x3d's `graph.track_credited_on` and `graph.track_by_artist` edges, on top of
edges-v2's release-level `graph.credited_on`) was expected to enrich the graph, not to change the
zero-self-weight tie mechanic docs/recall_and_churn.md documents. It did neither cleanly:
recall and served-index churn are both worse, and the duplicate-vector share grew rather
than shrank. Whether the wider credit scope itself contributes to more, not fewer, tied
neighbourhoods (more artists sharing an otherwise-identical small credit-only neighbourhood)
is a plausible mechanism but not isolated by this bead's measurements — the graph was built
once per weight, not ablated between edges-v2 and edges-v3 scope at fixed `w0`.

Quality moved in the *opposite* direction from recall and churn, though: the identical
`w0=0` configuration scores 1.86 points higher on chw.2 under edges-v3 (27.192%) than
edges-v2 (25.329%, docs/embedding_quality.md). The wider credit scope plausibly adds
genuine similar-artist signal chw.2's active-seed-artist queries can use, at the same time
as it plausibly worsens the tie structure the ANN index and exact recall are more sensitive
to (few-hundred-artist duplicate groups, not chw.2's typical query) — both effects are
consistent with the same underlying change, not a contradiction, but neither is isolated by
this bead's measurements.

## Larger-index variant: not measured

The maintainer-approved larger-index variant (`m=32, ef_construction=128`, same
`maintenance_work_mem`/parallel-worker settings as the standard variant) was dropped
entirely after a real failure, not skipped speculatively:

September's `m=32` build for `w0=0` overflowed `maintenance_work_mem = 8GB` at approximately
6.1M of 9,366,416 tuples and fell back to PostgreSQL's on-disk HNSW build path (workers on
`IO/DataFileRead`, roughly 80 tuples/s). It was abandoned after about 4h50m, having reached
roughly 7.01M of 9,366,416 tuples (about 75%) — finishing would have taken on the order of
8 hours for that one build, repeated per weight and again for the winner's full run. **The
overflow itself is the finding**: at this catalog scale (~9.4M rows), `m=32` needs either a
materially higher `maintenance_work_mem` or a subset/two-phase build strategy to complete in
a reasonable window, neither of which was in scope here. Every measurement in this document
used `measure_recall_churn.py --skip-larger-variant`.

## Memory findings

Peak RSS during `scripts/embeddings_from_dump.py`'s stream-parse + FastRP pipeline
(`resource.getrusage`, footprint-aware `wait_for_memory` guard active throughout):

| Phase | August | September |
| --- | ---: | ---: |
| `same_as` map (pass 1) | 5.82–5.89 GB | — |
| Full parse (pass 2, both dumps) | 7.62–7.63 GB | — |
| Parser structures freed, pre-build | 9.92–10.03 GB | — |
| `AdjacencyBuilder.build()` (graph, parser still resident) | **19.26 GB** | **17.27 GB** |
| FastRP alone (parser structures already freed), by `w0` | 9.87 / 10.65 / 10.65 GB | 11.66 / 14.33 / 14.33 GB |

FastRP compute time alone (post-parse, by `w0`; from `i37_sweep_run.log`'s own `fastrp: …s`
lines):

| Month | w0=0 | w0=0.1 | w0=0.25 |
| --- | ---: | ---: | ---: |
| August | 623.8 s | 638.5 s | 637.2 s |
| September | 600.8 s | 614.6 s | 643.0 s |

**Reconciling against the estimate.** docs/embeddings.md's own edges-v3 estimate (added
alongside x3d, never previously checked against a real-dump run) is: extending its ieu.6
figures to about 31.3M nodes and 243.9M edges gives `estimate_peak_bytes(...)` ≈ 9.1 GB of
arrays; adding its standard ~1 GB interpreter/allocator overhead puts the estimated peak RSS
at **about 10.1 GB — "still within the 12 GB budget, but with markedly less headroom"**
(docs/embeddings.md, verbatim: about 1.9 GB of margin left, against about 3.5 GB before
edges-v3). This bead is that real-dump measurement docs/embeddings.md said should confirm
the budget before the next monthly load runs under `"edges-v3"`, and the result is mixed:

- FastRP-alone peaks (9.87–14.33 GB) bracket the 10.1 GB estimate rather than confirming it —
  August's three weights (9.87–10.65 GB) land close to and slightly below it, but
  September's `w0=0.1`/`w0=0.25` (14.33 GB each) land **4.2 GB above it**, eating well past
  the estimate's already-thin 1.9 GB of margin and leaving under 0 GB of headroom against the
  overall 12 GB budget at those two points.
- Graph construction with the parser still resident (17.27–19.26 GB) is not the phase the
  10.1 GB estimate describes (that estimate is FastRP's own array footprint, computed after
  the parser's structures are freed — see "FastRP alone" above), so it is not a like-for-like
  comparison, but it is the pipeline's real peak, and it is well above the 12 GB budget
  either way.

**The 12 GB budget did not hold at real-dump scale for September's `w0=0.1`/`w0=0.25`, and
the pipeline's true peak (graph construction) is 17.27–19.26 GB regardless of weight.** The
footprint-aware memory guard (not a flat threshold) is what let this run at all on a host
with far less than 19 GB free at times; a fixed 12 GB ceiling, enforced rather than
estimated, would need to be revisited before the next edges-v3 monthly load, not assumed
from the pre-measurement estimate alone.

Standard-variant (`m=16`) HNSW index size was consistent across every build in this bead:
**about 4.0 GB per month**, regardless of weight (4.007–4.042 GB across all five builds —
three trim-run September builds plus the winner's own August and September builds).

## Latency: measured under load, upper bounds only

Every per-query ANN latency number in this document (and its `latency_ms_by_ef_search`
counterpart in the underlying JSON) was measured on a host that was, for much of this bead,
under sustained memory and disk pressure from concurrent work (the crisis this bead's own
streaming/reclaim/guard fixes exist to address, plus other hives sharing the same host's
validation and Docker capacity). Mean per-query latency at `ef_search = 1000` ranged from 91
to 296 ms across the runs in this bead — a roughly 3× spread for supposedly comparable
measurements, which is itself evidence of load, not a real latency difference between runs.
**These numbers are upper bounds on what a quiet machine would show, not a latency
characterization of the served index.** A re-measure on an otherwise-idle host is still
owed before any of these latency figures should inform a serving-latency decision.

## Environment and versions

- numpy 2.5.3, scipy 1.18.1 (both months, same build).
- PostgreSQL 19 + pgvector (`database-schema-postgres19-pgvector:local`), HNSW
  `m = 16, ef_construction = 64` for every measured index (the `m=32` variant was never
  successfully built — see above).
- Dump ids: `discogs_20260801` (August), `discogs_20260901` (September).
- Stored `model_version`s: `fastrp-v1:dim=128:weights={w0},1,1,1,1:beta=0:proj=achlioptas-s3:
  rows=splitmix64(blake2b64(kind,key)):seed=20260924:edges-v3@discogs_2026{08,09}01`, for
  `w0 ∈ {0, 0.1, 0.25}`.
- chw.2 harness: the design repo's `gm-design-chw.2` spike, reproduced outside any repo
  from a freshly-streamed `discogs_20260901_releases.xml.gz` (10% artist-seeded subset,
  `build_subset.py --rate 0.10`, no `--expand`); reproduction against the spike's own
  committed baseline numbers is docs/embedding_quality.md's own check, not repeated here.

## Cleanup

The throwaway `gm-i37-recall-pg` container and its per-weight rows/indexes were reclaimed
incrementally during the run (see "Per-weight Postgres reclaim" in "Method") and the
container itself has since been removed. No provider-derived data is committed to this
repository; only this document, the two measurement scripts, and the aggregate numbers above
are.
