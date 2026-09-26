# ANN recall and month-over-month churn on real embeddings

Status: **in progress — real embeddings computed for both months, Docker/pgvector
measurement running**. This document records gm-analytics-engine-ieu.3's measurement of
the two open ADR 0013 preconditions before `catalog-api` may serve similar-artist results
from the pgvector index:

1. **ANN recall.** Real FastRP vectors' recall@10 against exact cosine search, swept over
   `ef_search`, naming the smallest value that reaches recall@10 ≥ 0.95 (or stating that
   none does).
2. **Churn.** Month-over-month top-10 Jaccard churn between two consecutive dumps'
   embeddings, on both exact cosine and the served ANN index.

Both numbers, once measured, are recorded here and as a comment on the bead. No
provider-derived data (ids, names, vectors, edges) is committed anywhere in this repo;
only this document, the two measurement scripts, and aggregate counts are.

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
| (i) `credited_on` ⨝ `same_as`, kept categories, correct role-category rule and name-join resolution **(adopted, ieu.6)** | 50,411,242 | see final parity report below | see final parity report below |
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
creation and the 2 GB attempt's container had only 2g), for both months, to keep the graph
in memory throughout the build:

| `maintenance_work_mem` | Rows | Result |
| --- | ---: | --- |
| 2 GB (database-schema's documented value) | 6,869,453 (Aug) | Killed at 3,590,494 tuples (52.3%) after ~1h25m; slowed to ~15–18k tuples/min from ~3.4M tuples on |
| 4.5 GB | 6,869,453 (Aug) | *(filled in once complete)* |
| 4.5 GB | 6,896,892 (Sept) | *(filled in once complete)* |

The graph structure and recall do not depend on `maintenance_work_mem` — only build
speed — so the 4.5 GB build's recall numbers are directly comparable to what a 2 GB build
would (eventually) have produced. This sizing finding is separate from and additional to
ADR 0013's own build-memory precondition (which already flagged the 512 MB *default* as
insufficient and set the operator procedure's documented value at 2 GB); it says that
documented 2 GB value itself needs revisiting for a graph at this artist-row scale.

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

| Month | Trial (100k rows) | Extrapolated full month | Used COPY? |
| --- | ---: | ---: | :---: |
| August | 15.7s | 1082s (18.0 min) | No — real write took 1048.8s for the remaining 6,769,453 rows |
| September | *(filled in once complete)* | | |

### Recall@10 vs. `ef_search`

*(Filled in once the measurement completes — see "Status" below for what remains.)*

### Month-over-month churn

*(Filled in once the measurement completes.)*

## Status

Real embeddings for both months are computed and saved locally (not committed) at
`~/.cache/groovemap-spikes/embeddings-scratch/{aug,sept}.npz`. The Docker/pgvector
measurement (`scripts/measure_recall_churn.py`) is running against a throwaway
`database-schema-postgres19-pgvector:local` container. Remaining steps:

1. Finish the August and September full-scale HNSW builds (4.5 GB `maintenance_work_mem`)
   and the recall@10 sweep for each.
2. Compute exact + ANN top-10 churn for the common-artist sample.
3. Fill in the recall/churn tables above with exact figures (no rounding into verdicts).
4. Record both numbers as a comment on gm-analytics-engine-ieu.3.
5. Tear down the container/volumes, delete the local scratch `.npz` files and any other
   provider-derived scratch artifacts, confirm nothing provider-derived is staged for
   commit.
6. `bh work check` / `bh work submit`.
