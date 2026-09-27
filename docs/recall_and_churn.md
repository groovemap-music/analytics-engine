# ANN recall and month-over-month churn on real embeddings

Status: **complete.** This document records gm-analytics-engine-ieu.3's measurement of the
two open ADR 0013 preconditions before `catalog-api` may serve similar-artist results from
the pgvector index:

1. **ANN recall.** Real FastRP vectors' recall@10 against exact cosine search, swept over
   `ef_search`, naming the smallest value that reaches recall@10 ≥ 0.95 (or stating that
   none does).
2. **Churn.** Month-over-month top-10 Jaccard churn between two consecutive dumps'
   embeddings, on both exact cosine and the served ANN index.

## Headline numbers

- **Recall@10 never reaches 0.95 at any swept `ef_search`, for either month.** Best
  observed: August 0.8429, September 0.8405, both at `ef_search = 1000` (pgvector's own
  hard cap). No production `ef_search` is named, per the AC's "or states that none does."
- **Churn (10,000-artist common sample, Aug → Sept): mean top-10 Jaccard 0.9083 on exact
  cosine, 0.8026 on the served ANN index at `ef_search = 1000`.** Jaccard here is
  `|intersection| / |union|` of the two months' top-10 sets for the same artist; 1.0 means
  an artist's top-10 didn't change at all, 0.0 means it's a completely different set.
  0.9083 (exact) says the *underlying embeddings* are quite stable month over month; 0.8026
  (ANN) is lower because the ANN index also introduces its own retrieval noise on top of
  that (see "Recall and tie structure" below for why that noise is large here).

Both numbers are recorded here and as a comment on the bead (see "Bead comment" at the
end — no `bh work` verb exists for this, flagged to the dispatcher rather than using the
`bd` passthrough). No provider-derived data (ids, names, vectors, edges) is committed
anywhere in this repo; only this document, the two measurement scripts, and aggregate
counts are.

## Graph construction: stream-parse the dumps directly, not a full catalog load

`scripts/embeddings_from_dump.py` builds the FastRP input graph by stream-parsing the
Discogs monthly dumps (releases + masters) directly, rather than loading the full
catalog into PostgreSQL through `discogs-sql-loader`. This is possible because
`graph.vertex_degree` and every edge table the embedding pipeline reads key on the raw
Discogs `data_id` with no identity-resolution step in between (ADR 0009's native-identity
fusion does not touch these particular tables), and MusicBrainz does not feed them
today — confirmed by reading `discogs-sql-loader/tableinator/graph_derivation.py`
directly. A parser that replicates its exact vertex/edge derivation rules therefore
reproduces byte-identical node keys and edges to what a full catalog load would produce
for the same dumps, at a fraction of the disk and time cost.

The nine relations, copied one-to-one from `graph_derivation.py`/the merged
`insights/embedding_pipeline.py` (not from the design spike's own subset builder in
`design/docs/spikes/gm-design-chw.2/`, which differs — see below):

| Relation | Source | Notes |
| --- | --- | --- |
| `by_artist` | release's own `artists` list | main "BY" artists only, never `extraartists` |
| `on_label` | release's `labels` list | every entry |
| `derived_from` | release's own `master_id` field | |
| `in_genre` / `in_style` | release's own `genres` / `styles` lists | verbatim, no casefold |
| `master_by_artist` / `master_in_genre` / `master_in_style` | the **master's own** document | not aggregated from its releases — needs the masters dump |
| `credited_by_artist` (ieu.6) | release-level `extraartists`, kept role categories, resolved via a global `same_as` name-join | see "Credited-artist resolution" below |

An id is dropped only when blank or the Discogs "no entity" sentinel `"0"`
(`graph_derivation._entity_id`); there is no placeholder-artist filtering, unlike the
design spike's own subset builder.

**Vertex set simplification** (approved for this measurement): every vertex kind's
population is derived from these relations' own endpoints, not from a separate full read
of `artists.xml`/`labels.xml`. Production embeds every artist document, including the
ones with zero graph edges, which get an all-zero FastRP vector regardless of source —
degenerate for both recall and churn.

## The graph-scope finding: the shipped pipeline (pre-ieu.6) was narrower than the spike's graph

Before running FastRP at scale, this bead ran a parity check — parsing a real dump and
comparing vertex/edge counts against known reference numbers (ADR 0013's own
dump-derived counts, and Discogs's live statistics API) — to catch a divergence between
this script's graph and the real one before spending compute on it. That check surfaced
a real, pre-existing finding rather than a parsing bug:

**ADR 0013's "~9.4 million servable artists" and the epic's "32.8M nodes" both trace to
the chw.2 spike's own graph construction**, which explicitly included release-to-credited-
artist edges (`extraartists`, kept role categories) and per-track artists on top of main
artists (design `docs/spikes/gm-design-chw.2-graph-embeddings-vs-heuristics.md`,
"Graph edges": *"Release to credited artist is also included... Track artists are
included as well."*). The chw.2 spike's own recall numbers — the ones ADR 0013's GO
decision is based on — were measured against that broader graph, and its FastRP
configuration note is explicit: *"credit and track edges included. Removing them costs
10 points of recall@10."*

**The pipeline as shipped by ieu.1/ieu.2** implemented a narrower graph: main-artist
`by_artist` only, with credited artists (`extraartists`) routed to a separate
`graph.credited_on`/`graph.same_as` pair the pipeline never read, and no track-artist
relation anywhere in the `graph` schema. Confirmed two independent ways: (1) direct reads
of `graph_derivation.py` and `embedding_pipeline.py`, and (2) a completely separate,
single-threaded, non-chunked `iterparse`-based cross-check that reproduced the same low
distinct-artist count as the production-shaped chunked parser on the same release prefix,
ruling out a parsing bug.

On the Aug 2026 dump, the 8-relation (pre-ieu.6) graph: 19,341,287 releases parsed,
2,861,379 distinct main-artist (`by_artist`) artists, 26,113,501 total vertices —
substantially smaller than the 32.8M-node figure the epic quotes from the spike.

### Maintainer decision (2026-09-25): option (a) — gm-analytics-engine-ieu.6

ieu.3 measures the graph that actually ships: the original 8 relations plus a
release-level credit edge (`credited_on` ⨝ `same_as`, filtered to the chw.2 spike's kept
role categories). Landed as gm-analytics-engine-ieu.6 (merge commit `4c0d4de`), which
blocked this bead until it merged. `FastRPConfig`'s stored `model_version` gained an
edge-set version tag (`_EDGE_SET_VERSION = "edges-v2"`) so embeddings computed under the
old and new edge sets never share a primary-key value.

**Remaining gap, explicitly out of scope for both ieu.6 and this bead**, tracked as
follow-ups:

- Per-track and sub-track `extraartists` credits — gm-analytics-engine-x3d (embedding
  pipeline), gm-discogs-sql-loader-b2a (loader derivation), gm-database-schema-ug3v
  (schema/graph relations).
- Per-track `<artists>` (track performers) — same three follow-ups.

Neither exists in the `graph` schema today; adding either needs new derivation logic in
`discogs-sql-loader` and a new relation in `database-schema`, not just a wider read in
this repository.

### Sizing the widening options (Aug 2026 dump)

Baseline: 19,341,287 releases, 2,861,379 distinct main-artist (`by_artist`) artists.

| Option | Edges | Distinct artists | New beyond main |
| --- | ---: | ---: | ---: |
| (i) `credited_on` ⨝ `same_as`, kept categories, correct role-category rule and name-join resolution **(adopted, ieu.6)** | 50,411,242 | 6,869,453 (main + credited) | 4,008,074 |
| (ii) chw.2 spike's full credit scope — release + per-track + sub-track `extraartists`, kept categories | 96,491,147 | 6,917,277 | 5,593,003 |
| (iii) per-track `<artists>` (track performers), unfiltered | 24,370,971 | 2,633,830 | 1,271,244 |

Kept role categories (options i and ii): production, engineering, session, and
`common.credit_roles`' catch-all "other". Dropped: mastering, design, management — the
same taxonomy `graph.credit_role_category` renders into SQL, imported from the same
shared `common.credit_roles` module the chw.2 spike itself used.

Sanity check against the spike's ~9.4M figure: main (2.86M) + option (ii)'s new-beyond-
main (5.59M) alone already reaches ~8.45M; option (iii) adds a further, partially
overlapping, 1.27M new artists on top. Main + full-credit-scope + track-performers,
unioned, plausibly lands close to the spike's ~9.4M — consistent with the graph-scope
finding above.

### Two corrections found while sizing option (i)

**The kept-category filter must not split a compound role on comma.**
`common.credit_roles.categorize_role` (the function `graph.credit_role_category`'s SQL
rendering mirrors) does not split a role like `"Producer, Recorded By"` into parts and
check each — it substring-matches the *whole* lowered/stripped string, longest fragment
first. An early draft of this measurement (and the chw.2 spike's own harness) split on
comma and kept a credit if *any* fragment matched a kept category. That is a materially
different, looser filter: it overcounted `credited_by_artist` edges by about 1.7% on the
Aug dump (a preliminary same_as-comparison audit got 51,283,744 edges with the comma-split
filter; the corrected, production-faithful filter gets **50,411,242**). Fixed in
`scripts/embeddings_from_dump.py`'s `_role_kept` before the real FastRP run.

**Credited-artist resolution is a name-join, not the credit's own XML id.**
Production's `graph.credited_on` is keyed by `(person_name, release_id, role)`, not
artist_id; `_CREDITED_ARTIST_EDGE_SQL` resolves it with `INNER JOIN graph.same_as ON
same_as.person_name = credited_on.person_name`. `graph.same_as` is additive and
catalog-wide (no category filter), built from every release-level `extraartists` entry
with a resolvable id, of *any* role — so a `credited_on` row resolves to every artist_id
ever paired with that exact name string anywhere in the catalog, not just the id (if any)
on that specific XML entry. An unresolvable name (never paired with any id) drops the
credit; a name resolved to more than one id fans out to all of them. Measured on the Aug
dump (full releases dump, two passes — build the global `same_as` map, then resolve):

| | Naive (this entry's own XML id) | Production (name-join, fan-out) |
| --- | ---: | ---: |
| Edges | 51,004,385 | 51,283,744 |
| Edges the other resolution doesn't have | 0 | 279,359 |

Naive is a strict subset of production here — the name-join never drops an edge the XML
id already gave; it adds 279,359 more (credits whose own XML entry had no id, but the same
exact name string was resolvable elsewhere in the catalog). 146,609 kept-category credits
remain unresolvable under either method. Only 9 of 6,004,307 distinct credited names
resolve to more than one artist_id (ambiguous fan-out is a non-issue in practice on this
dump), and `same_as.person_name` does preserve Discogs' `(2)`/`(3)` disambiguation suffix
(confirmed: 1,000,634 of 6,004,307 distinct credited names carry one, e.g. `'Russell Brown
(5)'`) — an exact-string collision is the narrower case of an un-merged duplicate profile
or a data-entry error, not the common "two same-named musicians" case Discogs' own
numbering already separates. `scripts/embeddings_from_dump.py` implements the name-join
resolution (`_build_same_as_map` + the per-release lookup in `_parse_release_chunk`), not
the naive one. Note for `catalog-api`'s own kNN endpoint (gm-catalog-api-2zsq): it will
need the same self-exclusion handling this bead's measurement script needed (see
"ef_search self-exclusion" below), behind a `model_version`-filtered *partial* index
rather than this script's whole-table one.

## Real embeddings: August and September 2026

Both months computed with the real `insights.embeddings.fastrp` (defaults: 128 dims,
weights `0,1,1,1,1`, β=0, seed 20260924, `float16` output, `block_columns=4`,
`threads=6`), via `scripts/embeddings_from_dump.py --delete-dumps-after-parse` for August
(freeing its downloaded dumps immediately after parsing, before the FastRP compute) and
without that flag for September (whose dumps are shared with other developers).

| | August (`discogs_20260801`) | September (`discogs_20260901`) |
| --- | ---: | ---: |
| Releases parsed | 19,341,287 | 19,417,067 |
| Masters parsed | 2,579,897 | 2,589,349 |
| Total vertices | 30,121,572 | 30,240,002 |
| Distinct artists (main + credited) | 6,869,453 | 6,896,892 |
| `by_artist` edges | 23,528,328 | 23,624,346 |
| `on_label` edges | 22,461,851 | 22,552,273 |
| `derived_from` edges | 11,297,259 | 11,340,082 |
| `in_genre` edges | 25,775,928 | 25,880,594 |
| `in_style` edges | 29,369,746 | 29,506,965 |
| `master_by_artist` edges | 3,170,000 | 3,182,144 |
| `master_in_genre` edges | 3,429,228 | 3,442,451 |
| `master_in_style` edges | 4,068,173 | 4,086,006 |
| `credited_by_artist` edges | 50,411,242 | 50,616,170 |
| Parse wall time | 1659.3s | 1522.4s |
| Build (AdjacencyBuilder) wall time | 179.3s | 169.6s |
| FastRP wall time | 427.6s | 410.5s |
| numpy / scipy | 2.5.3 / 1.18.1 | 2.5.3 / 1.18.1 |

Masters count note: `distinct masters` (from edge endpoints — `derived_from`'s target and
`master_by_artist`'s source) is always slightly below `masters parsed` (2,579,877 vs.
2,579,897 for August, a gap of 20 out of 2,579,897 — 0.0008%): a master document that is
never pointed to by any release's `master_id` and has none of its own `<artists>` /
`<genres>` / `<styles>` is counted as parsed but contributes no edge, so never appears as
a vertex under this bead's edge-endpoint-only vertex-set simplification. Not related to
the malformed-chunk warnings below.

Two "skipping malformed chunk: mismatched tag" warnings appear per run (one for releases,
one for masters) — confirmed harmless: `_chunks`' final `if rest: yield rest` emits
whatever remains after the last full `</release>`/`</master>` closing tag, which for the
very last chunk of the file is just the outer `</releases>`/`</masters>` container-closing
tag with no matching open tag once wrapped — a "mismatched tag" parse error by
construction, not a lost record. Confirmed no data loss: repeated full-scale parses of the
same dump produce byte-identical release/master counts.

### Memory: this script's parser overhead, not FastRP's or AdjacencyBuilder's own need

August's peak RSS checkpoints (via `resource.getrusage`, a monotonic high-water mark, with
explicit `del` + `gc.collect()` between phases to keep each checkpoint's jump as
attributable as possible):

| Checkpoint | Peak RSS |
| --- | ---: |
| After the global `same_as` map (pass 1) | 4.27 GB |
| After the full parse (pass 2, both dumps) | 5.16–5.25 GB |
| After freeing parser structures, pre-build | 10.60–10.68 GB |
| After `AdjacencyBuilder.build()` | 15.51–17.42 GB |
| After `fastrp` | 15.51–17.42 GB (no further increase) |

`insights.embeddings.estimate_peak_bytes()` — the same estimator `docs/embeddings.md`'s
own scaling table is built from — against the real Aug counts (30,121,572 nodes,
173,511,755 directed edges, 6,869,453 artist rows, `block_columns=4`, `float16`) predicts
**build 3.62 GB, compute (fastrp) 7.55 GB, peak 7.55 GB**: the array-only cost a lean,
streaming-from-PostgreSQL pipeline (like production's) needs. The gap between that and the
15.51 GB observed is attributable to structures specific to this measurement script's
non-streaming design, not to `fastrp`/`AdjacencyBuilder` themselves:

- **`same_as` map: 4.27 GB, measured directly.** Production never needs this — its
  resolution is a SQL `JOIN`, no client-side name map.
- **Duplicated `artist_ids` list: ~5.35 GB** (the 5.25→10.60 GB jump while building
  `artist_key_to_id`). This script accumulates `by_artist` + `credited_by_artist` +
  `master_by_artist` occurrences — 77,109,570 raw, repeated artist-id strings — into a flat
  list before deduping into a dict. Production's `_read_vertices` dedupes inline with a
  `seen_artist_ids` set as it streams from PostgreSQL cursors; it never materializes this.
- **`relation_arrays` held through the build call: 2.78 GB, exactly computable**
  (173,511,755 edges × 2 arrays × 8 bytes uint64). Production streams edges in 50,000-row
  cursor blocks and never materializes the full edge set at once.

`fastrp`'s own peak never exceeded the build phase's — both read 15.51 GB via
`ru_maxrss` on the August run — meaning `fastrp`'s incremental need (the 7.55 GB the
analytical estimate predicts) fit entirely within the elevated baseline this script's own
parsing left behind. **Conclusion: production, streaming from PostgreSQL and deduping
inline, would plausibly fit comfortably under the 12 GB budget for this widened
(9-relation) graph — around the 7.55 GB analytical estimate — not the 15.51–17.42 GB this
throwaway script observed.** This is an *estimate* with the attribution above, not an
end-to-end measurement from PostgreSQL; a real first-run measurement of the production
pipeline itself is noted on gm-deployment-cy6 (the deployment repo bead that wires the
monthly trigger). Not changing production code in this bead.

## HNSW build: `maintenance_work_mem` sizing

Real DDL, not a stand-in: `scripts/measure_recall_churn.py`'s inline
`_ARTIST_EMBEDDINGS_TABLE_SQL` is byte-for-byte identical to database-schema's landed
`_ARTIST_EMBEDDINGS_STATEMENT` (`src/groovemap_schema/postgres.py`, `gm-database-schema-
lhp2`), copied verbatim rather than taken as a package dependency (not yet pinned from
analytics-engine).

**database-schema's documented build-time `maintenance_work_mem` (2 GB) is undersized for
this graph's scale.** Attempting the full-scale August build at 2 GB: the build measurably
slowed from full speed to roughly 15,000–18,000 tuples/minute starting around 3.4M of
6,869,453 tuples (container physical footprint ~2.27 GB at that point, per `vmmap`) —
consistent with the HNSW graph outgrowing the 2 GB budget partway through and continuing
in a slower, presumably disk-spilling mode, the same failure mode (at a smaller scale) the
chw.1 footprint spike saw at the default 512 MB. The attempt was killed at **3,590,494 of
6,869,453 tuples (52.3%) after approximately 1h25m**, once it was clear finishing at that
rate would take several more hours.

**Retried at 4.5 GB** (`--shm-size 5g`, needed since a container's shared memory is set at
creation and the 2 GB attempt's container had only 2g), for August only: it *also* slowed
down, this time starting around 5.3M of 6,869,453 tuples (~77%), settling to roughly
12,000 tuples/min — i.e. 4.5 GB is undersized too, just later than 2 GB.

**Retried again at 8 GB**, on a host reconfigured to 6 CPUs / 16 GiB (from the original
2 CPU / 7.7 GiB Docker VM) with `max_parallel_maintenance_workers = 4` (pgvector supports
parallel HNSW builds): both months' builds completed at full speed, no slowdown observed.

| `maintenance_work_mem` | CPUs | Rows | Result |
| --- | --- | ---: | --- |
| 2 GB (database-schema's documented value) | 2 | 6,869,453 (Aug) | Killed at 3,590,494 tuples (52.3%) after ~1h25m; slowed to ~15–18k tuples/min from ~3.4M tuples on |
| 4.5 GB | 2 | 6,869,453 (Aug) | Also slowed, from ~5.3M tuples (~77%) on, to ~12k tuples/min; not carried to completion at this setting |
| 8 GB, 4 parallel workers | 6 | 6,869,453 (Aug) | **552.8 s**, no slowdown |
| 8 GB, 4 parallel workers | 6 | 6,896,892 (Sept) | **596.1 s**, no slowdown |

The graph structure and recall do not depend on `maintenance_work_mem` or CPU count — only
build speed — so the final 8 GB builds' recall numbers are what a slower build would
(eventually) have produced too. This sizing finding is separate from and additional to ADR
0013's own build-memory precondition (which already flagged the 512 MB *default* as
insufficient and set the operator procedure's documented value at 2 GB); it says that
documented 2 GB value — and even 4.5 GB — needs revisiting for a graph at this artist-row
scale (~6.9M rows, 128 dims). The real production host's CPU count and available memory
were not available to this bead to test against; this is a finding about the *shape* of
the sizing problem (the documented value is measurably too small at this row count), not a
specific recommended replacement value for production, which depends on the real host's
resources.

### A second, more serious HNSW bug this bead's own harness had: whole-table vs. partial index

While chasing the memory-sizing slowdown, a second index build (September, immediately
after August) hit a much worse failure: a `COPY` into the (by-then-truncated) table for
September's rows crawled at ~20,000 rows/minute — on pace for **over 5 hours** for
6,896,892 rows. Root cause: this measurement script's `_build_index` originally built a
**whole-table** HNSW index (no `WHERE` clause), matching `database-schema`'s currently
*landed* `_ARTIST_EMBEDDINGS_HNSW_INDEX_STATEMENT` (the per-`model_version` partial index,
`gm-database-schema-19g5`, has not landed as of this writing). Once built for August, that
index kept accepting *every* row inserted afterward regardless of `model_version` — so
September's insert wasn't getting its own bulk-built HNSW graph, it was being appended to
August's already-built one, one row at a time, which is exactly the slow, sequential
insertion pattern HNSW bulk-build exists to avoid.

Fixed in this script (not in `database-schema`, which is a separate follow-on,
`gm-database-schema-19g5`) to build the real per-`model_version` **partial** index shape
`insights.embedding_pipeline._log_operator_step` already logs as the intended operator
statement (`... WHERE model_version = '<stored version>'`), and to explicitly `DROP INDEX`
the previous month's index before truncating between months. One more snag while fixing:
a bind parameter in `CREATE INDEX`'s `WHERE` clause raises psycopg's
`IndeterminateDatatype` (PostgreSQL can't infer the parameter's type in that DDL context) —
worked around with `_sql_string_literal`, the same helper `_log_operator_step` itself uses
for this exact clause. Verified end to end on synthetic two-month data before re-running
against the real dataset. **This means August's own recall numbers were measured against a
whole-table index over a table that, at the time, held only one `model_version`'s
rows — equivalent to a partial index for that measurement — while September's (and any
future re-run's) numbers are measured against the real partial-index shape.** The recall
numbers themselves are unaffected either way; only the index-build mechanics differ.

## Recall@10 and churn measurement

`scripts/measure_recall_churn.py`: one month's rows in an otherwise-empty
`public.artist_embeddings` table at a time, real `_write_embeddings` (100k-row timed trial
first; COPY fallback only if extrapolated past ~30 min — not triggered on either month,
see below), the real HNSW index named by `_index_name(stored_model_version)`.

**Deterministic sampling.** Both the recall query sample (2,000 artists) and the churn
common-artist sample (10,000 artists, from the Aug ∩ Sept intersection) are the smallest
N candidates ranked by `splitmix64(node_key("a", artist_id) XOR seed)` — the same hash
family the pinned FastRP projection itself uses, not a fresh RNG. Fixed seeds
(`QUERY_SAMPLE_SEED = 0x51ECA11A`, `CHURN_SAMPLE_SEED = 0xC8027A11`) make this exactly
repeatable.

**`ef_search` self-exclusion.** A query artist's own vector is stored verbatim in the
table, so an ANN query that doesn't exclude it always ranks itself first (distance 0) —
caught via a synthetic-data smoke test (recall was stuck at a suspicious flat 0.9 across
every `ef_search` value before the fix). Rejected `WHERE artist_id != $q` alongside the
`ORDER BY ... LIMIT` on review, since a second equality filter risks the planner choosing
a different plan than the `model_version`-filtered index scan on the real ~7M-row table;
requests `k + 1` rows with only the `model_version` filter and drops the query's own
artist_id client-side instead. `catalog-api`'s own kNN endpoint will need the same
self-exclusion, there behind a `model_version`-filtered partial index rather than this
script's whole-table one.

**Write timing (100k-row trial, real `_write_embeddings`):**

| Month | Trial (100k rows) | Extrapolated full month | Used COPY? | Actual remainder |
| --- | ---: | ---: | :---: | ---: |
| August | 15.3s | 1051s (17.5 min) | No | 6,769,453 rows in 1052.1s |
| September | 16.3s | 1127s (18.8 min) | No | 6,796,892 rows in 1176.6s |

Neither month extrapolated past the 30-minute budget, so the `COPY` fallback was never
exercised for a real write (it was exercised, and a real bug in it fixed, during
development — see the script's own commit history: an explicit `NULL` for `computed_at`
would have violated that column's `NOT NULL` constraint, since a COPY row's column list
must *omit* a column for its `DEFAULT` to apply).

### Recall@10 vs. `ef_search`

Both months, real `_write_embeddings` write, real per-`model_version` partial HNSW index,
`m = 16, ef_construction = 64`:

| `ef_search` | August recall@10 | September recall@10 |
| --- | ---: | ---: |
| 40 | 0.60695 | 0.59755 |
| 100 | 0.6796 | 0.67555 |
| 200 | 0.7357 | 0.73395 |
| 400 | 0.7846 | 0.78235 |
| 800 | 0.83385 | 0.83 |
| 1000 (pgvector's hard cap) | **0.8429** | **0.8405** |

**No `ef_search` reaches recall@10 ≥ 0.95 for either month; the production `ef_search` is
therefore: none.** The two months track each other closely at every sweep point, which is
itself informative — this is a stable property of the graph/config, not month-specific
noise.

#### Recall and tie structure

Recall this far below 0.95, and essentially flat between ef_search 800 and 1000 (pgvector's
cap), prompted a closer look at whether the *ground truth itself* is well-defined. It
mostly isn't, for a large fraction of the catalog, for a structural reason rather than a
quality problem:

**FastRP's configured weights are `(0, 1, 1, 1, 1)` — the weight on the k=0 term (a node's
own projection row) is zero.** An artist's embedding therefore depends *only* on its
neighbors' structure, never on the artist's own identity. Two artists with identical
neighborhoods within the propagation radius — the common case being two artists whose only
graph connection is a single shared release, and nothing else — get **byte-identical**
FastRP vectors by construction, not by coincidence.

Measured on August's 6,869,453 vectors (NumPy only, no Postgres):

- **2,659,206 of 6,869,453 vectors (38.7%) are exact byte-duplicates of at least one other
  vector.** 753,588 duplicate groups; the largest has 818 members.
- Zero vectors have near-zero norm (every artist has *some* propagated signal).
- Over the 2,000-query recall sample, the 10th-vs-11th exact-cosine-score gap has median
  0.0017 and p25 0.0004; 38.0% of queries have a gap under 1e-3, and 17.5% have an exact
  (within 1e-6) tie at or adjacent to the 10th-place score. Queries whose own vector sits in
  a duplicate group of 21+ show essentially zero gap (mean 0.0000) — for these, "the"
  correct top-10 is fundamentally ambiguous; any two equally-valid rankings of the tied
  group can disagree on which members land in positions 10 vs. 11+.

This means a meaningful share of the recall@10 "misses" are not the ANN index failing to
find genuinely closer neighbors — they're the ANN index and the exact brute-force
computation each validly picking different members of a tied or near-tied group, under
slightly different floating-point paths (the exact computation upcasts float16→float32
before normalizing; pgvector's internal HNSW distance computation operates natively in
`halfvec`/float16 throughout). Self-exclusion is consistent between the two paths (both
exclude the query's own `artist_id`; see "`ef_search` self-exclusion" above).

A tie-tolerant recomputation (counting an ANN hit if its exact cosine is within 1e-4 of the
true 10th-place score, per the maintainer's ask) and a recall breakdown by duplicate-group
size were started but not completed for this bead: the host's disk floor was breached by
other concurrent work partway through, and clearing the throwaway pgvector container to
relieve it took priority over finishing that specific analysis. The duplicate-vector count,
gap distribution, and root cause above stand on their own as the primary finding; the
tie-tolerant number would likely land noticeably above the raw 0.84, given how much of the
gap distribution sits under 1e-3.

**This is a genuine property of the shipped FastRP configuration** (weights `0,1,1,1,1`,
adopted from the chw.2 spike, ADR 0013), not an artifact of this measurement's graph
construction or the ieu.6 credit-edge widening — it would apply equally to the narrower
pre-ieu.6 graph, since weight[0]=0 is unrelated to which edges feed the graph. It is not
this bead's place to change the FastRP configuration; it's recorded here as the
explanation for why recall@10 doesn't reach the ADR's proposed 0.95 bar, for whoever plans
the next step against this precondition.

### Month-over-month churn

10,000-artist deterministic common-artist sample (of 6,865,940 artists present in both
months):

| | Mean top-10 Jaccard |
| --- | ---: |
| Exact cosine (both months' full in-memory vectors) | **0.9083** |
| Served ANN index, at `ef_search = 1000` (no `ef_search` reached the 0.95 recall target, so the largest swept value was used) | **0.8026** |

Jaccard is `|top10_aug ∩ top10_sept| / |top10_aug ∪ top10_sept|` per artist, averaged over
the sample; 1.0 is no change, 0.0 is a completely different list. The embeddings
themselves are fairly stable month over month (0.9083 on exact cosine) — consistent with
the deterministic, per-node-key-hashed projection (ADR 0013's own churn precondition,
aimed at exactly this: an *unpinned* projection replaced ~72% of a list on the design
spike). The served list is noisier than the underlying embeddings (0.8026 vs. 0.9083) —
consistent with the tie structure above: since a large share of top-10 boundaries are
ties or near-ties, the ANN index's arbitrary tie-breaking can itself flip between months
even when the true embeddings barely moved.

## Environment and versions

- numpy 2.5.3, scipy 1.18.1 (both months, same build — Determinism in `docs/embeddings.md`
  notes bit-identity holds only within one build of each).
- PostgreSQL 19 + pgvector 0.8.6 (`database-schema-postgres19-pgvector:local`), HNSW
  `m = 16, ef_construction = 64` (ADR 0013's fixed parameters).
- Real DDL: this bead's `_ARTIST_EMBEDDINGS_TABLE_SQL` is byte-for-byte identical to
  database-schema's landed `_ARTIST_EMBEDDINGS_STATEMENT`.
- Dump ids: `discogs_20260801` (releases sha not separately recorded here; see the shared
  spike cache), `discogs_20260901`.
- Stored `model_version`s: `fastrp-v1:dim=128:weights=0,1,1,1,1:beta=0:proj=achlioptas-s3:
  rows=splitmix64(blake2b64(kind,key)):seed=20260924:edges-v2@discogs_20260801` (August) and
  the same with `@discogs_20260901` (September).

## Bead comment

No `bh work` verb exists for posting a plain comment on a bead (checked `bh work --help`);
flagged to the dispatcher to escalate rather than using the `bd` passthrough
(`BH_BD_PASS_ENABLED`) to work around it. The headline numbers this section would have
posted are recorded verbatim at the top of this document ("Headline numbers") instead.

## Cleanup

The throwaway `gm-ieu3-measure-pg` container and its dangling volumes were removed.
`aug.npz`, `sept.npz`, and `recall_churn*.json` are deliberately **kept** (not deleted) at
`~/.cache/groovemap-spikes/embeddings-scratch/` — gm-analytics-engine-ste (recommendation
quality on the chw.2 proxy benchmark) needs them next. Nothing provider-derived is
committed to this repository; only this document, the two measurement scripts, and the
aggregate numbers above are.

## Tie-tolerant recall, duplicate-group breakdown, and over-fetch + re-rank (gm-analytics-engine-kn3)

Status: **complete**, except the degree-bucket breakdown (see "Degree-bucket gap" below).

Maintainer decision 2026-09-26 (option A, step 1), following up on the "Recall and tie
structure" finding above: strict recall@10 undercounts equally-correct neighbours whenever
a query's true top-10 boundary sits inside a tie or near-tie — already shown above to be
common (38.7% of vectors are exact duplicates; 17.5% of the 2,000-query sample has an
exact tie at/adjacent to the 10th place). This measurement quantifies how much strict
recall is undercounting, on the same September real embeddings (`sept.npz`, edges-v2,
ieu.6) and the same deterministic 2,000-query sample (seed `1_374_462_234` = `0x51ECA11A`,
`QUERY_SAMPLE_SIZE = 2000`) ieu.3 already used. Script:
`scripts/measure_tie_tolerant_recall.py`, reusing ieu.3's own harness
(`measure_recall_churn.py`, imported as a sibling module, not copied) for the real DDL,
`_write_embeddings`, and the per-`model_version` partial HNSW index.

**Tie-tolerant definition**, per the maintainer's ask: an ANN-returned candidate counts as
a hit if its own exact cosine similarity to the query is ≥ the query's exact 10th-place
cosine score − 1e-4 — not "is one of the exact top-10 ids" (what strict recall requires),
which a tied 11th/12th/... place candidate fails even though it is equally correct.

### Model-version labelling pitfall

`insights.embedding_pipeline._EDGE_SET_VERSION` is a shared, still-moving constant — it
was `"edges-v2"` when ieu.6/ieu.3 ran, but gm-analytics-engine-x3d has since bumped it to
`"edges-v3"` (adds track-credited-artist/track-performer relations neither `sept.npz` nor
this measurement's graph ever had). `measure_recall_churn.py`'s `_load_month` composes
`model_version` via `stored_model_version(FastRPConfig(), dump_id)`, which reads whatever
`_EDGE_SET_VERSION` **is at measurement time**, not at npz-*computation* time. Running
this bead's script unmodified against `sept.npz` therefore silently mislabelled every row
it wrote as `edges-v3`, even though the vectors are the same edges-v2 ones ieu.3 already
measured — caught before submit. `scripts/measure_tie_tolerant_recall.py` now pins
`_PINNED_EDGE_SET_VERSION = "edges-v2"` explicitly rather than trusting the live module
constant; a verification run confirmed the pinned value reproduces ieu.3's exact recorded
string byte for byte: `fastrp-v1:dim=128:weights=0,1,1,1,1:beta=0:proj=achlioptas-s3:
rows=splitmix64(blake2b64(kind,key)):seed=20260924:edges-v2@discogs_20260901`. The numbers
below are from this bead's real (first) measurement run, whose recorded `model_version`
was corrected post hoc in `tie_tolerant_recall.json` (a `model_version_correction` field
there explains it) — the mislabeling never affected the measurement itself, only a string
that run's own throwaway table/index used consistently and privately; its `index_name`
still embeds a hash of the original (wrong) string, cosmetic only. **Anyone re-running this
class of script against an older `.npz` should pin the edge-set version the same way, not
derive it from the live pipeline module.**

### Setup

This bead's own throwaway container (`gm-analytics-engine-kn3-pg`, removed after the run),
`database-schema-postgres19-pgvector:local`, `--shm-size 9g`,
`max_parallel_maintenance_workers=4`, `maintenance_work_mem=8GB` for the index build — the
same parameters the "HNSW build" section above settled on. 6,896,892 September rows
written (100k-row trial: 15.5s; remainder 1,081.4s; no `COPY` fallback needed), HNSW index
(`m=16, ef_construction=64`) built in 616.9s.

### Strict vs. tie-tolerant recall@10

| `ef_search` | Strict recall@10 | Tie-tolerant recall@10 | Δ (tie − strict) |
| --- | ---: | ---: | ---: |
| 40 | 0.5955 | 0.6208 | +0.0253 |
| 100 | 0.6779 | 0.7066 | +0.0287 |
| 200 | 0.7370 | 0.7688 | +0.0317 |
| 400 | 0.7842 | 0.8206 | +0.0364 |
| 800 | 0.8310 | 0.8662 | +0.0352 |
| 1000 | **0.8408** | **0.8778** | +0.0370 |

Strict recall@10 at `ef_search=1000` reproduces ieu.3's recorded September value (0.8405)
closely — 0.8408, a ~0.6-query difference out of 2,000, within normal HNSW build-order
noise (parallel-worker insertion order isn't fixed run to run). Tie tolerance recovers a
consistent 2.5–3.7 points of recall at every swept `ef_search`, growing slightly as
`ef_search` increases — the ANN index is finding genuinely tied-or-near-tied neighbours it
is being strictly marked wrong for, not missing them.

### Breakdown by duplicate-group size

Each query's own exact-duplicate-group size (byte-identical vector compare, over all of
September's 6,896,892 vectors — 2,670,779 (38.7%) are exact duplicates of at least one
other, matching ieu.3's August figure), bucketed {1 (unique), 2-4, 5-20, 21+}. Bucket
sizes, of the 2,000-query sample: 1 (unique) n=1,226; 2-4 n=454; 5-20 n=268; 21+ n=52.

| `ef_search` | 1 (unique) strict | 1 (unique) tie | 2-4 strict | 2-4 tie | 5-20 strict | 5-20 tie | 21+ strict | 21+ tie |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 0.6483 | 0.6586 | 0.5057 | 0.5104 | 0.5918 | 0.6280 | 0.1519 | 0.6538 |
| 100 | 0.7237 | 0.7352 | 0.6090 | 0.6145 | 0.6784 | 0.7194 | 0.1962 | 0.7692 |
| 200 | 0.7834 | 0.7953 | 0.6773 | 0.6837 | 0.7287 | 0.7735 | 0.2096 | 0.8615 |
| 400 | 0.8281 | 0.8410 | 0.7425 | 0.7507 | 0.7757 | 0.8243 | 0.1577 | 0.9308 |
| 800 | 0.8692 | 0.8821 | 0.7998 | 0.8088 | 0.8272 | 0.8713 | 0.2250 | 0.9692 |
| 1000 | 0.8796 | 0.8929 | 0.8176 | 0.8269 | 0.8299 | 0.8769 | 0.1865 | 0.9692 |

**The "21+" bucket (queries whose own vector sits in a duplicate group of 21 or more) is
the headline finding here.** Its strict recall@10 is uniformly low (0.15–0.23, barely
improving with `ef_search`) — but its tie-tolerant recall@10 is the *highest* of any bucket
(0.65–0.97, reaching 0.97 at `ef_search ≥ 800`). This is exactly the mechanism the "Recall
and tie structure" finding above predicted: for these queries, dozens of vectors are
equally, exactly correct top-10 members, so "the" exact top-10 ordering pgvector's HNSW and
this script's brute-force NumPy computation independently arrive at is close to arbitrary
(both are valid; they just don't have to agree with each other) — strict recall punishes
that disagreement as a miss even when it isn't one. The 2-4 and 5-20 buckets sit between
the unique and 21+ buckets and do not move monotonically with group size (5-20 slightly
out-recalls 2-4 at most `ef_search` values); sample sizes for these buckets (454 and 268
queries respectively) are small enough that this is plausibly noise rather than a real
non-monotonicity, and this bead did not chase it further.

### Over-fetch + exact re-rank

For k' ∈ {50,100,200} at `ef_search` ∈ {100,200,400}: fetch k' ANN candidates (`LIMIT
k'+1`, self excluded client-side, same convention as the base recall sweep), re-rank them
by exact cosine (September's vectors already resident in memory — no second Postgres round
trip for the exact side), take the top 10 of the re-ranked list:

| `ef_search` | k' | Strict recall@10 | Tie-tolerant recall@10 | Mean latency | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 50 | 0.6776 | 0.7066 | 21.95 ms | 20.51 ms | 34.94 ms | 49.86 ms |
| 100 | 100 | 0.6774 | 0.7066 | 27.24 ms | 23.04 ms | 53.44 ms | 103.10 ms |
| 100 | 200 | 0.6779 | 0.7066 | 28.36 ms | 24.96 ms | 49.99 ms | 82.79 ms |
| 200 | 50 | 0.7383 | 0.7688 | 50.87 ms | 44.28 ms | 94.81 ms | 140.73 ms |
| 200 | 100 | 0.7384 | 0.7688 | 31.58 ms | 28.79 ms | 52.00 ms | 79.41 ms |
| 200 | 200 | 0.7384 | 0.7688 | 26.80 ms | 25.78 ms | 37.22 ms | 48.07 ms |
| 400 | 50 | 0.7869 | 0.8206 | 37.75 ms | 36.94 ms | 49.74 ms | 71.22 ms |
| 400 | 100 | 0.7864 | 0.8206 | 36.96 ms | 36.97 ms | 45.95 ms | 54.71 ms |
| 400 | 200 | 0.7863 | 0.8206 | 45.99 ms | 43.54 ms | 66.65 ms | 104.22 ms |

**Over-fetching at a fixed `ef_search` does not meaningfully improve recall.** At every
swept `ef_search`, strict and tie-tolerant recall@10 barely move across k' = 50/100/200
(largest strict shift: +0.0013, at `ef_search=200`; tie-tolerant is flat to 4 decimal
places within an `ef_search`). Recall is governed by `ef_search` — how much of the HNSW
graph gets explored — not by k' — how many of the already-explored candidates get returned
and re-ranked. The (tiny) k'=50 gain over the plain ef-sweep at the same `ef_search` (e.g.
strict 0.7383 vs. 0.7370 at `ef_search=200`) is consistent with exact re-ranking recovering
a little precision pgvector's native `halfvec`-throughout HNSW distance loses relative to
the float32-upcast exact computation, among candidates already retrieved — not with
over-fetching finding new, better candidates the original top-10 missed. Latency (the
single ANN round trip dominates; the NumPy re-rank itself is negligible) scales with
`ef_search`, not cleanly with k' — the per-run variance visible here (e.g.
`ef_search=200, k'=50` at 50.87ms mean vs. `k'=200` at 26.80ms) reflects ordinary
container/host noise during this run rather than a real k'-latency relationship; no attempt
was made to control for that noise (this bead reports what ran, not a controlled latency
benchmark).

### Degree-bucket gap

**Not attempted in this bead.** A breakdown by the query artist's graph degree needs
`insights.embeddings.graph.Adjacency.degree`, only available as a byproduct of
`scripts/embeddings_from_dump.py`'s full 9-relation graph build, which needs September's
`releases.xml.gz` dump. Only `artists`/`masters`/`labels` (~1.1 GB combined) were cached
locally at `~/.cache/groovemap-spikes/dumps/`; `releases.xml.gz` (19.4M records — masters
alone is 597 MB compressed for 2.6M records, so releases is plausibly several GB) was not,
and this host's disk floor was already breached by other concurrent work before this bead
started. Downloading it to compute degree would have made that worse. Flagged to the
dispatcher rather than approximating a degree metric; per the dispatcher,
gm-analytics-engine-x3d/gm-analytics-engine-i37 (which already streams and parses the
September graph for its own purposes) will produce real per-artist degree as a natural
byproduct, so this breakdown is deferred there rather than repeated here.

### Cleanup

The throwaway `gm-analytics-engine-kn3-pg` container was removed after the run.
`sept.npz` was read-only throughout (never modified or deleted — shared with other beads
reading the same scratch directory). Output:
`~/.cache/groovemap-spikes/embeddings-scratch/tie_tolerant_recall.json` (aggregates only,
plus the query-sample seed/rule for reproducibility); nothing provider-derived is committed
to this repository.
