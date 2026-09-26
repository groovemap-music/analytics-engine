# ANN recall and month-over-month churn on real embeddings

Status: **in progress, blocked on gm-analytics-engine-ieu.6**. This document records
gm-analytics-engine-ieu.3's measurement of the two open ADR 0013 preconditions before
`catalog-api` may serve similar-artist results from the pgvector index:

1. **ANN recall.** Real FastRP vectors' recall@10 against exact cosine search, swept over
   `ef_search`, naming the smallest value that reaches recall@10 ≥ 0.95 (or stating that
   none does).
2. **Churn.** Month-over-month top-10 Jaccard churn between two consecutive dumps'
   embeddings.

Both numbers, once measured, are recorded here and as a comment on the bead. No
provider-derived data (ids, names, vectors, edges) is committed anywhere in this repo;
only this document, the measurement script, and aggregate counts are.

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

The rules, copied one-to-one from `graph_derivation.py` (not from the design spike's own
subset builder in `design/docs/spikes/gm-design-chw.2/`, which differs — see below):

| Relation | Source | Notes |
| --- | --- | --- |
| `by_artist` | release's own `artists` list | main "BY" artists only, never `extraartists` |
| `on_label` | release's `labels` list | every entry |
| `derived_from` | release's own `master_id` field | |
| `in_genre` / `in_style` | release's own `genres` / `styles` lists | verbatim, no casefold |
| `master_by_artist` / `master_in_genre` / `master_in_style` | the **master's own** document | not aggregated from its releases — needs the masters dump |
| `credited_by_artist`\* | release's own **release-level** `extraartists` | resolved to an artist id, kept role categories only — see below |

\* Added by gm-analytics-engine-ieu.6 (in progress as of this writing), which now blocks
this bead. The name, join shape, and category list here are this script's
best-informed placeholder pending ieu.6's merged implementation — see "Status" below.

An id is dropped only when blank or the Discogs "no entity" sentinel `"0"`
(`graph_derivation._entity_id`); there is no placeholder-artist filtering, unlike the
design spike's own subset builder.

**Vertex set simplification** (approved for this measurement): every vertex kind's
population is derived from these relations' own endpoints, not from a separate full read
of `artists.xml`/`labels.xml`. Production embeds every artist document, including the
ones with zero graph edges, which get an all-zero FastRP vector regardless of source —
degenerate for both recall and churn.

## The graph-scope finding: the shipped pipeline is narrower than the spike's graph

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

**The shipped pipeline** (`insights/embedding_pipeline.py`'s `_EDGE_RELATIONS`,
landed by gm-analytics-engine-ieu.1/ieu.2) implements a narrower graph: main-artist
`by_artist` only. Credited artists (`extraartists`) go to a separate `graph.credited_on` /
`graph.same_as` pair that the pipeline never reads at all, and there is no track-artist
relation anywhere in the `graph` schema. This was confirmed two independent ways: (1)
direct reads of `graph_derivation.py` and `embedding_pipeline.py`, and (2) a completely
separate, single-threaded, non-chunked `iterparse`-based cross-check that reproduced the
same low distinct-artist count as the production-shaped chunked parser on the same
release prefix, ruling out a parsing bug.

On the Aug 2026 dump: 19,341,287 releases parsed, 2,861,379 distinct main-artist
(`by_artist`) artists — the whole 8-relation graph (26,113,501 total vertices) is
substantially smaller than the 32.8M-node figure the epic quotes from the spike.

### Maintainer decision (2026-09-25): option (a)

ieu.3 measures the graph that will actually ship: the current 8 relations plus
release-level credit edges (`credited_on` ⨝ `same_as`, filtered to the chw.2 spike's kept
role categories). That widening is gm-analytics-engine-ieu.6, filed and now blocking this
bead. Sized on the Aug dump (see "Sizing the widening options" below), it is the
smallest of the three options considered and reaches close to, but not all the way to,
the spike's own ~9.4M-artist population.

**Remaining gap, explicitly out of scope for both ieu.6 and this bead**, tracked as
follow-ups:

- Per-track and sub-track `extraartists` credits — gm-analytics-engine-x3d (embedding
  pipeline), gm-discogs-sql-loader-b2a (loader derivation), gm-database-schema-ug3v
  (schema/graph relations).
- Per-track `<artists>` (track performers) — same three follow-ups.

Neither exists in the `graph` schema today; adding either needs new derivation logic in
`discogs-sql-loader` and a new relation in `database-schema`, not just a wider read in
this repository.

### Sizing the widening options (Aug 2026 dump, `scripts/embeddings_from_dump.py`'s parser)

Baseline: 19,341,287 releases, 2,861,379 distinct main-artist (`by_artist`) artists.

| Option | Edges | Distinct artists | New beyond main |
| --- | ---: | ---: | ---: |
| (i) `credited_on` as it exists today — release-level `extraartists` only, kept categories, resolvable id **(adopted, ieu.6)** | 51,004,385 | 5,124,569 | 4,072,191 |
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

## Status

Blocked on gm-analytics-engine-ieu.6 landing the `credited_by_artist` relation in the
shipped pipeline. Once it merges:

1. Reconcile `scripts/embeddings_from_dump.py`'s placeholder `credited_by_artist`
   relation (name, join shape, category list) against ieu.6's actual merged
   implementation.
2. Re-run the parity check on both months' dumps against the updated graph.
3. Run the real `insights.embeddings.fastrp` for August 2026 and September 2026.
4. Bring up a throwaway PostgreSQL 19 + pgvector container, load one month's embeddings
   at a time into the real `public.artist_embeddings` table, and build that month's HNSW
   index with `_index_name(stored_model_version)`.
5. Measure recall@10 vs. `ef_search` (exact ground truth computed directly from the
   in-memory embedding array, no Postgres needed for that side) and month-over-month
   top-10 Jaccard churn — both on exact cosine and on the served ANN index at the
   production `ef_search`, per the maintainer's request, over a deterministic sample
   (a fixed hash of the Discogs artist id, documented here once chosen).
6. Record both numbers here and as a bead comment, with exact figures (no rounding into
   verdicts), the dump ids/dates, wall times, peak memory, and the numpy/scipy/pgvector
   versions used.

<!-- Filled in once ieu.6 lands and the measurement runs; see "Status" above. -->
