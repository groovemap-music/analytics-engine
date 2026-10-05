# FastRP artist embeddings

[ADR 0013](https://github.com/groovemap-music/design/blob/main/docs/adr/0013-pgvector-catalog-embeddings.md) adopts FastRP graph embeddings for similar-artist retrieval. This repository owns the computation. `insights/embeddings/` holds the algorithm and its interfaces. Reading the graph from PostgreSQL, the monthly recompute, and writing `public.artist_embeddings` belong to the pipeline that calls it.

## Configuration

| Parameter | Value |
| --- | --- |
| Dimensions | 128 |
| Iteration weights | `0,1,1,1,1`: five propagation steps, with the first one unweighted |
| Degree normalization β | 0 |
| Self term | 0.05 × the vertex's own normalized projection row (monthly pipeline; `FastRPConfig()` defaults to 0) |
| Projection | Very sparse (Achlioptas, s = 3): ±√3 with probability 1/6 each, otherwise 0 |
| Projection row of a vertex | SplitMix64 over `blake2b64(kind, key)` XOR a per-column salt, derived from the pinned seed `20260924` |

`FastRPConfig.model_version` names all of this. The monthly pipeline runs `insights.embedding_pipeline.PRODUCTION_FASTRP_CONFIG`, which is `FastRPConfig(self_weight=0.05)`. Its value is `fastrp-v2:dim=128:weights=0,1,1,1,1:beta=0:self=0.05:proj=achlioptas-s3:rows=splitmix64(blake2b64(kind,key)):seed=20260924`, and the stored `model_version` appends `:edges-v3@<dump id>`. The self term (gm-analytics-engine-8ts, `docs/embedding_tie_break.md`) removed the byte-duplicate vectors that tied 42.22% of edges-v3 artists and raised exact top-10 month-over-month churn from 0.8886 to 0.9519. `FastRPConfig()` itself keeps `self_weight=0`, which reproduces the pre-8ts sum bit for bit. `FASTRP_ALGORITHM_VERSION` is bumped whenever a code change alters any output bit, so vectors from two versions of the code never share a `model_version`.

Each month's exact top-K similar-artist lists are computed from these vectors; see `docs/similar_artists.md`.

## Interfaces

```python
from insights.embeddings import AdjacencyBuilder, FastRPConfig, NodeIndex, fastrp, node_keys

nodes = NodeIndex(node_keys(vertices))  # (kind, key) pairs, e.g. ("a", "123")
builder = AdjacencyBuilder(nodes)
for sources, targets in edge_blocks:  # uint64 node keys, one cursor block at a time
    builder.add_edges(sources, targets)
adjacency = builder.build()
artists = nodes.positions(node_keys(("a", key) for key in artist_ids))
vectors = fastrp(adjacency, FastRPConfig(), rows=artists, out_dtype=np.float16, threads=6)
```

- A vertex's identity is the `(kind, key)` pair that `graph.vertex_degree` uses. `node_key` hashes the pair to 64 bits. A duplicate key, whether from a repeated vertex or a hash collision, is rejected.
- `AdjacencyBuilder` takes edges in blocks, as key pairs or as positions. It buffers them as `int32` pairs and builds the CSR matrix with a counting sort. Direction is ignored, parallel edges collapse, and self-loops are dropped.
- `fastrp` processes the embedding columns `block_columns` at a time (four by default), using two passes. The first pass sums each power's squared row norms across all blocks. The second pass recomputes each block and adds its normalized contribution. It returns only `rows`. With `out_dtype=np.float16` the result is the `float32` result rounded once, which is exactly what a `halfvec` column stores.

## Determinism

- **Same graph, same bytes.** Node positions are ranks of node keys. `build` sorts every row's neighbours, so each output row is summed over its neighbours in node-key order, whatever order the edges arrived in. Squared norms are accumulated one column at a time in a fixed order. The output is therefore byte-identical across runs, edge orders, edge block sizes, `block_columns`, and thread counts. The tests assert each of these.
- **Unchanged regions keep their vectors.** A node's vector reads rows of `D^-1 A` within four hops of it, and projection rows within five hops. A projection row is a function of the vertex's own key, not of its position or of a random stream. So when no edge is added or removed at any vertex within four hops of a node, its vector stays byte-identical, even as other vertices are added and every position shifts. The tests add an unrelated component whose keys interleave the originals and compare every original vector byte for byte. They also add one edge at the end of a path and check that exactly the five vertices within four hops of the edit change.
- **Scope.** Bit identity holds for one build of NumPy and SciPy. Floating-point results can differ in the last bit across CPU architectures or library releases, so comparisons across months are made between runs in the same pipeline image.
- **Pinned values.** `node_key` and the signs of the projection for fixed vertices are pinned in tests. A change to either fails the tests before it can reach a stored vector.

## Parity with the spike

Given the same projection matrix, `fastrp` reproduces the spike's `embed.fastrp` (design `docs/spikes/gm-design-chw.2/embed.py`). On a synthetic catalog-shaped graph of 28,772 nodes and 109,759 edges, the largest absolute difference is 8.8e-7 and the smallest row cosine is 0.99999976. The only numerical difference is that squared norms accumulate in `float64` rather than in `float32`. `tests/test_fastrp.py` keeps a copy of the spike function and asserts agreement to within 2e-6.

The hashed projection is statistically another draw of the same distribution. On that graph, artist top-10 lists under the hashed projection overlap the spike's seed-0 lists with a Jaccard index of 0.131. The spike's own seed 0 and seed 1 overlap each other at 0.131.

## Memory and time at catalog scale

`scripts/fastrp-scaling.py` runs synthetic graphs at the full catalog's proportions: 6.77 undirected edges per node, 59% release nodes, and 31% artist rows returned. Each size runs in its own process. All runs used six threads and `float16` output, on the shared 10-core, 32 GB development host with a load average of about 10, so the times are pessimistic.

| Nodes | Edges | `block_columns` | Build | FastRP | Peak RSS | Array estimate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0M | 6.8M | 8 | 2.4 s | 14.6 s | 0.60 GB | 0.32 GB |
| 2.0M | 13.5M | 8 | 5.5 s | 32.9 s | 1.02 GB | 0.64 GB |
| 4.0M | 27.1M | 8 | 11.0 s | 70.4 s | 1.93 GB | 1.28 GB |
| 8.0M | 54.1M | 8 | 23.1 s | 151.3 s | 3.25 GB | 2.56 GB |
| 1.0M | 6.8M | 4 | 2.7 s | 14.1 s | 0.54 GB | 0.29 GB |
| 8.0M | 54.1M | 4 | 24.2 s | 179.2 s | 2.71 GB | 2.30 GB |

Before ieu.6, this document estimated the full catalog at 32.8M nodes, at most 222M edges, and 10.2M artist rows — the chw.2 spike's own coarse extrapolation ("Full-catalog extrapolation (estimated, not run)"), not a count against a real dump. Extrapolating the measurements above linearly from 1M to 8M nodes against that estimate:

| `block_columns` | Peak RSS | Array estimate | FastRP | Build |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 12.6 GB | 10.5 GB | about 10 min | about 1.5 min |
| **4 (default)** | **10.4 GB** | **9.4 GB** | **about 12 min** | **about 1.5 min** |
| 128, one pass | — | 54 GB | — | — |

The pipeline therefore runs with the defaults (four columns per block and two passes), `float16` output, and six threads, which stays within the 12 GB budget. At that (pre-ieu.6) peak, the resident set is as follows:

- the CSR transition matrix: 3.7 GB of `int32` indices and `float32` values
- the returned `float16` artist rows: 2.6 GB
- two `n × 4` `float32` blocks, one of them the product with `P`: 1.0 GB
- the pass-one squared norms: 1.0 GB, which pass two replaces with 0.16 GB of scales for the returned rows only
- node keys and degrees: 0.5 GB
- interpreter and allocator overhead: about 1 GB

`estimate_peak_bytes` computes the array share for any graph size, block width, and output precision.

Memory can be reduced further, at a cost:

- A `float32` output adds 2.6 GB. The stored column is `halfvec`, so `float32` output buys nothing.
- One column per block saves about 0.8 GB and costs time.
- Returning only served artists (about 9.4M) saves about 0.2 GB.

Threads add little memory, because each task works on at most 262,144 rows.

### Updated for the release-level credited-artist edges (ieu.6)

The 2026-08 dump gives a real count in place of the spike's extrapolation: about 30M nodes and 174M edges once the credited-artist relation's 51,004,385 edges and 4,072,191 credited-only artists are included (see "Release-level credited-artist edges" above) — both somewhat below the earlier 32.8M/222M guess, since that guess was never checked against a real dump. Artist rows written grow from the 2,861,379 main artists to about 6.93M (2,861,379 main plus 4,072,191 credited-only), since `_read_vertices` returns every artist vertex, not only main artists, and `_write_embeddings` writes one row per vertex it returns.

`estimate_peak_bytes(30_000_000, 174_000_000, 6_933_570, block_columns=4, out_itemsize=2)` gives an array estimate of about 7.6 GB (build 3.6 GB, compute 7.6 GB) — lower than the pre-ieu.6 9.4 GB estimate above despite the added edges and artist rows, because the real node/edge counts are themselves lower than the spike's guess. Adding the same roughly 1 GB of interpreter and allocator overhead this document already carries puts the estimated peak RSS at about 8.5 GB, still within the 12 GB budget with headroom to spare. The scaling table's proportions (6.77 undirected edges per node) no longer describe the graph exactly — the credited-artist relation shifts the edge-per-node ratio somewhat — but `estimate_peak_bytes` takes node and edge counts directly, so the estimate above does not depend on that ratio holding.

### Updated for track-level credits and track performers (x3d)

gm-analytics-engine-x3d adds the two track-level relations `gm-database-schema-ug3v` declared: `graph.track_credited_on` (per-track and sub-track `extraartists`, resolved through `graph.same_as` exactly like the release-level relation above) and `graph.track_by_artist` (per-track and sub-track `<artists>` performers, direct `artist_id`, no name resolution). Neither count below is a real-dump measurement of the resolved, catalog-scale graph this pipeline actually builds — that re-measurement is explicitly out of this bead's scope, left to a follow-up maintainer decision alongside the recall/churn re-run `_EDGE_SET_VERSION`'s bump to `"edges-v3"` calls for — but the chw.2 spike's own dump-scale sizing and the exact `gm-database-schema-ug3v` counts give a defensible estimate:

- **Track and sub-track credits.** The chw.2 spike's combined credit scope (release, track, and sub-track `extraartists`, deduplicated into one per-release credit set — see "Track-level credits and track performers (x3d)" below for why this pipeline dedupes the same way) was about 96.5M kept-category credit edges on the 2026-08 dump, against 51.0M for the release-level relation alone. The difference, about 45.5M edges, is the estimated *net-new* contribution of track- and sub-track-level credits once deduplicated against the release-level relation this pipeline already reads — an estimate carried over from the spike's own scope figures, not a `same_as`-resolved measurement against the real catalog.
- **Track performers.** `graph.track_by_artist` is sized on the real 2026-08 dump at 24,370,971 edges, naming 1,271,244 artists beyond the main-artist set — an exact loader-derived count, not an extrapolation.

Extending the ieu.6 estimate: about 30M + 1.27M ≈ 31.3M nodes, 174M + 45.5M + 24.4M ≈ 243.9M edges, and 6.93M + 1.27M ≈ 8.2M artist rows written (an upper bound: it treats every track-performer artist beyond main as also beyond the 4.07M already-credited-only pool ieu.6 added, which the source data does not yet confirm one way or the other — the true figure is somewhere between 6.93M and 8.2M). `estimate_peak_bytes(31_271_244, 243_870_971, 8_204_814, block_columns=4, out_itemsize=2)` gives an array estimate of about 9.1 GB (build 4.8 GB, compute 9.1 GB). Adding the same roughly 1 GB of interpreter and allocator overhead puts the estimated peak RSS at about 10.1 GB — still within the 12 GB budget, but with markedly less headroom than the 8.5 GB the ieu.6 estimate carried (about 1.9 GB of margin left, against about 3.5 GB before). A real-dump measurement, not this estimate, should confirm the budget still holds before the next monthly load runs under `"edges-v3"`.

## The monthly load pipeline

`insights/embedding_pipeline.py` is the pipeline the module docstring above defers to: it
reads the `graph` schema, calls `fastrp`, and writes `public.artist_embeddings`. It is a
separate entry point, `analytics-engine-embeddings` (`main()` in that module), not part of the
always-on FastAPI service — it authenticates under the `embedding_pipeline` role's own
credentials (`EMBEDDING_PIPELINE_POSTGRES_USERNAME`/`_PASSWORD`, the same `_FILE` secret
convention as everything else), a different, deliberately narrower login than the service's
`POSTGRES_USERNAME`. See database-schema's "Vector embeddings and the embedding pipeline role"
for the grant: `SELECT` on every relation in `graph`, `SELECT, INSERT, UPDATE, DELETE` on
`public.artist_embeddings` alone, nothing else.

**Reading the graph.** The pipeline reads the six vertex kinds this module's node identity
covers (artist, release, label, master, genre, style) from `graph.vertex_degree`, and the eleven
edge relations that connect them (`graph.by_artist`, `graph.on_label`, `graph.derived_from`,
`graph.in_genre`, `graph.in_style`, `graph.master_by_artist`, `graph.master_in_genre`,
`graph.master_in_style`, the release-level credited-artist relation ieu.6 added, and — since
this bead — the track-credited-artist and track-performer relations below) — each via a named
(server-side) PostgreSQL cursor, fetched in 50,000-row blocks, so the full vertex and edge sets
are never materialized as Python lists in one piece. `fastrp` is then called with
`PRODUCTION_FASTRP_CONFIG` (the self term at 0.05, above) and the defaults documented above: `out_dtype=np.float16`, `block_columns=4` (the default), and `threads=6` —
the configuration the scaling table above was measured against, which stays within the 12 GB
full-catalog budget.

**Release-level credited-artist edges (ieu.6).** The chw.2 spike's adopted FastRP
configuration includes credit and track edges — "removing credit and track edges costs 10
points of recall@10" — but the eight relations above are all main-artist (`by_artist`); no
credit reaches the graph. Production already stores release-level credits:
`graph.credited_on` is `(person_name, release_id, role)`, with a GENERATED `role_category`
column over the same `common.credit_roles` taxonomy the spike's harness used
(`graph.credit_role_category`), and `graph.same_as` is the separate `(person_name, artist_id)`
table resolving a credited name to a catalog artist id. The pipeline reads
`graph.credited_on JOIN graph.same_as ON person_name`, filtered to the spike's kept categories
— `production`, `engineering`, `session`, `other` — and drops `mastering`, `design`, and
`management`, exactly `KEPT_CREDIT_CATEGORIES` in the spike harness
(`design/docs/spikes/gm-design-chw.2/parse_dump.py`): a cutting engineer or sleeve photographer
links releases by vendor, not by sound.

`graph.same_as` has no release column, so a credited name resolves to zero, one, or more artist
ids independent of which release asked. This pipeline's rule, which the spike's harness never
had to make (its harness read Discogs artist ids directly out of the dump, bypassing this
name-based split entirely): a name with no resolved id joins to nothing and the credit is
silently dropped, and a name resolved to more than one id joins to every one of them, fanning
the credited-artist edge out rather than picking one. Both are accepted rather than treated as
errors — see `_CREDITED_ARTIST_EDGE_SQL`'s comment in `insights/embedding_pipeline.py` for the
full reasoning.

`same_as.person_name` is written verbatim from the Discogs `extraartists` name text
(discogs-sql-loader's `_credits`: "reads the name verbatim ... folding or trimming it here
would key the vertex differently from the node"), and it is catalog-wide and role-unfiltered —
it accumulates from *every* extraartists credit ever loaded, regardless of that credit's role
category, not only the kept ones this pipeline reads. That verbatim name already carries
Discogs' own `(2)`/`(3)` disambiguation suffix, which Discogs mints specifically so that two
different real people never share a plain display name: "John Smith" and "John Smith (2)" are
different `person_name` strings and never join together. An exact-string collision under this
scheme is therefore not the common "two musicians named John Smith" case — Discogs' own
numbering already separates those — but the narrower case of an un-merged duplicate artist
profile (two ids for what is, in the underlying catalog, the same real person, before a
moderator merges them) or a genuine data-entry error (a contributor crediting an existing name
without checking whether it was already taken). How common that narrower case actually is on
the real catalog is being measured separately (Aug-dump collision count); if it turns out to be
material, the fan-out rule above should become a drop-ambiguous-names rule instead, filed as a
follow-up rather than guessed at here.

Because a session player or producer credited only this way is never a main artist, an alias,
or a group member, `graph.vertex_degree` — scoped to the ten path-traversal relations
database-schema sums it over — has no row for them. `_read_vertices` runs two extra discovery
queries (over the same kept-category filter) to find exactly those artist and, defensively,
release ids before building the node index, so the credited-artist edge never names a vertex
the pipeline has not already seen; see that function's docstring.

`FASTRP_ALGORITHM_VERSION`/`config.model_version` cover the *algorithm*; they do not change
when the graph fed into it does. So that a dump reprocessed under a different edge set can
never land on, be skipped as, or silently overwrite an earlier edge set's rows, the *stored*
`model_version` composes in a separate `_EDGE_SET_VERSION` tag (`"edges-v1"` before this bead,
`"edges-v2"` after — bumped again, to `"edges-v3"`, by x3d's track-level relations below) — see
"The stored `model_version` is per dump, not per algorithm" below.

On the 2026-08 dump this relation adds 51,004,385 edges and 5,124,569 distinct credited
artists, of which 4,072,191 are never a main artist — beyond the 2,861,379 main-artist nodes
the eight relations above already carry. The full graph is therefore about 30M nodes and 174M
edges, up from the pre-ieu.6 estimate this document carried before real dump-based counts were
available (see "Memory and time at catalog scale" below for what that does to the peak-memory
estimate).

**Track-level credits and track performers (x3d).** `gm-database-schema-ug3v` declared two
further relations once discogs-sql-loader and database-schema landed the derivation this bead
was blocked on: `graph.track_credited_on` — `(person_name, release_id, track_ordinal,
sub_track_ordinal, track_position, role, role_category)`, one row per `tracklist[].extraartists`
or `tracklist[].sub_tracks[].extraartists` credit — and `graph.track_by_artist` —
`(release_id, track_ordinal, sub_track_ordinal, track_position, artist_id)`, one row per
`tracklist[].artists`/`tracklist[].sub_tracks[].artists` performer. Both key a track by
`(track_ordinal, sub_track_ordinal)`, never by the dump's own `track_position` string: a heading
entry's position is empty and two entries can share one, so a key built from it would drop the
first case and collapse the second — the fixtures in
`tests/integration/test_embedding_pipeline_integration.py` seed exactly that empty-position and
shared-position shape, a gap left over from the loader bead's own parity fixture.

`graph.track_credited_on` resolves through `graph.same_as` exactly like the release-level
relation — same kept/dropped category split, same fan-out-on-ambiguity and drop-on-unresolved
rules (see `_TRACK_CREDITED_ARTIST_EDGE_SQL`'s comment in `insights/embedding_pipeline.py`).
**This pipeline treats a track-level credit and a release-level credit to the same
(release, artist) pair as the same edge, not two.** Both assert the same fact — this artist is
credited on this release — discovered one nesting level apart, and `AdjacencyBuilder` already
collapses a parallel edge between the same two positions regardless of which `_EDGE_RELATIONS`
entry contributed it, so registering the track-level query as its own entry reproduces the
chw.2 spike's own treatment (`design/docs/spikes/gm-design-chw.2/parse_dump.py`: release, track,
and sub-track extraartists are unioned into one `credits` set per release before it ever becomes
an edge) without a hand-written `UNION` in SQL. `graph.track_by_artist`, by contrast, is a
genuinely different signal — a various-artists compilation's track performer is very often a
different artist than whichever name the release itself is credited to, the same reason
`by_artist` and `credited_on` are already two separate relations at release level — so it gets
its own edge, mirroring the spike's own separate `trackartists` list.

`graph.vertex_degree`'s ten path-traversal relations cover neither new relation, so an artist or
release reachable only through a track-level credit or performer needs the same discovery
treatment ieu.6 introduced for release-level credits: `_read_vertices` runs one artist- and one
release-discovery query per new relation (`_TRACK_CREDITED_ARTIST_IDS_SQL`/
`_TRACK_CREDITED_RELEASE_IDS_SQL`, `_TRACK_PERFORMER_ARTIST_IDS_SQL`/
`_TRACK_PERFORMER_RELEASE_IDS_SQL`), over the same kept-category filter where one applies, before
building the node index.

`_EDGE_SET_VERSION` bumps to `"edges-v3"` for these two relations — see "The stored
`model_version` is per dump and per edge set, not per algorithm" below — and see "Updated for
track-level credits and track performers (x3d)" above for the resulting node/edge/memory
estimate. No real-dump re-embedding or recall/churn re-measurement is part of this bead; that is
a separate, follow-up maintainer decision, the same way ieu.6's real-catalog measurement of the
release-level relation's fan-out-vs-drop choice was left pending in the section above.

**Lineage.** `SOURCE_DUMP_ID` and `SOURCE_DUMP_DATE` (required, no default — the invoker
supplies them, since it is the one that knows which dump just landed) become
`artist_embeddings.source_dump_id`/`source_dump_date` on every row the run writes.
`numpy_version`/`scipy_version` (bit-identity holds only for one build of each — see
"Determinism" above) are recorded in the run's own logs at the start of every call, since
`artist_embeddings` has no column for them.

**The stored `model_version` is per dump and per edge set, not per algorithm.**
`FastRPConfig.model_version` names only the method, its parameters, and the projection seed
rule — the same string every month an operator does not change the algorithm.
`stored_model_version(config, dump_id)`
(`f"{config.model_version}:{_EDGE_SET_VERSION}@{dump_id}"`) is the value this job actually reads
and writes as `artist_embeddings.model_version`; `config.model_version` itself is recorded
separately, as `method_version`, in every log line. Composing the dump id in is what lets two
months coexist under the table's `(artist_id, model_version)` primary key — a second month's
load is a brand-new set of rows under its own key, never an upsert of the first month's. An
earlier revision of this job stored the bare `config.model_version`, which meant a second dump
would silently overwrite the first month's rows in place; see the module docstring's "The
stored model_version is per dump, not per algorithm" for the full rationale this review caught.
`_EDGE_SET_VERSION` (`"edges-v1"` before ieu.6, `"edges-v2"` after it added the release-level
credited-artist relation, `"edges-v3"` after x3d added the track-credited-artist and
track-performer relations) applies the identical idea to the graph: bumping it whenever
`_EDGE_RELATIONS` changes means a dump reprocessed under a different edge set also lands on its
own rows rather than colliding with, or being skipped as, the previous edge set's.

**Idempotency.** The job is idempotent per stored `model_version` (which already encodes
`(dump_id, method_version)`): re-running for a dump already recorded under this algorithm is a
no-op — it does not re-read the graph or re-run `fastrp`. Every read and write this job makes is
scoped to the stored `model_version` it is computing — it never touches a row of any other
stored `model_version`, so both an earlier month's rows and a version `catalog-api` is currently
serving are untouched by a load of a new one. `ON CONFLICT (artist_id, model_version) DO UPDATE`
in the write is retry safety for a crash within the *same* dump's load, never a cross-dump
upsert, since the stored value already differs per dump.

**No index DDL, ever.** `embedding_pipeline` holds no DDL privilege and no ownership of
`public.artist_embeddings` — building or rebuilding the ANN index over a stored `model_version`
is always a separate, more privileged, human- or automation-driven operator step, run after this
job's transaction commits. The job logs the statement that step should run: a stored-
`model_version`-filtered `CREATE INDEX CONCURRENTLY ... USING hnsw (embedding
halfvec_cosine_ops) WITH (m = 16, ef_construction = 64) WHERE model_version = '...'` — the
`WITH` clause states ADR 0013's fixed HNSW parameters explicitly rather than leaving them to
pgvector's own defaults. The `WHERE` value is safely quoted (`_sql_string_literal`); the index
name comes from `_index_name`, not a plain truncated slug of the stored value —
`FastRPConfig.model_version` alone is already well past PostgreSQL's 63-byte identifier limit
(NAMEDATALEN - 1), so a naive slug would silently truncate before ever reaching the `@dump_id`
suffix that makes two months distinct, giving every month the *same*, colliding index name
(the round-2 review regression). `_index_name` instead composes a short, human-legible dump-id
fragment with a hash of the *entire* stored value, so two different stored versions can never
collide on one name, and the same stored version is always named the same thing. Per-
`model_version` partial indexes (`gm-database-schema-19g5`) are not landed as of this writing,
so the statement logged is the forward-looking shape the maintainer specified rather than
something database-schema documents today. Retiring a superseded `model_version`'s rows and
index, once `catalog-api` has switched to the new one, is
that follow-on's business, never this job's.

**Metrics.** `groovemap.insights.computation.duration` (histogram, `computation=
embedding_pipeline`, reusing the same instrument every other scheduled computation uses),
`groovemap.insights.embedding_pipeline.rows_written` (counter), and
`groovemap.insights.embedding_pipeline.failures` (counter) — see docs/operations.md.
`insights.computation_log`, the other computations' outcome log, is not written here: the
pipeline role holds nothing on the `insights` schema, so `public.artist_embeddings`'s own
lineage columns are the only durable record this job can leave.

**Scheduling.** There is no in-process scheduler loop, unlike `insights.insights`'s
`_scheduler_loop`. `analytics-engine-embeddings` is a one-shot script meant to be invoked by
the deployment layer once a month, after that month's dump has loaded and `SOURCE_DUMP_ID`/
`SOURCE_DUMP_DATE` are known. The exact monthly trigger (cron, a `CronJob`, an operator running
it by hand — the same shape as database-schema's own `build_artist_embeddings_index` operator
procedure) is a deployment-repo concern this bead does not fix; filed as gm-deployment-cy6.

**Testing.** `tests/test_embedding_pipeline.py` exercises the real FastRP/graph code against
small in-memory fakes of the PostgreSQL connection — no database, no Docker, including that two
dumps under one config get different stored versions rather than one upserting the other, and,
for x3d, that a track-level credit resolving to the same `(release, artist)` pair a release-level
credit already named adds no extra degree to either endpoint — the dedup decision this document
and `_TRACK_CREDITED_ARTIST_EDGE_SQL`'s comment describe, asserted directly against the real
`AdjacencyBuilder`.
`tests/integration/test_embedding_pipeline_integration.py` (`just test-integration-pg19`) runs
against a real PostgreSQL 19 + pgvector container with a real `embedding_pipeline`-scoped
login, asserting the idempotency, coexistence, and permission-boundary behavior above against
the engine itself — including that an earlier dump's rows are byte-for-byte unchanged after a
later dump loads — on a small synthetic graph, plus, for x3d, the real `graph.track_credited_on`/
`graph.track_by_artist` shape: a `track_position` of `NULL` (an empty string in the dump) and two
entries sharing one `track_position` across distinct `track_ordinal`s — a gap left by the loader
bead's own parity fixture, since `track_ordinal`/`sub_track_ordinal`, not `track_position`, are
the real primary-key columns — plus artist and release discovery for both new relations against
the real schema. That tier's `conftest.py` applies the real ADR 0013 schema objects via a pinned
`groovemap-database-schema` dev dependency's own `create_postgres_schema` (gm-analytics-engine-qzl),
the same way `catalog-api` applies it; see that `conftest.py`'s module docstring for the pinned
revision. Neither test tier commits provider-derived data or real embeddings, per ADR 0013's
data-rights section.
