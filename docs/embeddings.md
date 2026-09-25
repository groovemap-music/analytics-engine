# FastRP artist embeddings

[ADR 0013](https://github.com/groovemap-music/design/blob/main/docs/adr/0013-pgvector-catalog-embeddings.md) adopts FastRP graph embeddings for similar-artist retrieval. This repository owns the computation. `insights/embeddings/` holds the algorithm and its interfaces. Reading the graph from PostgreSQL, the monthly recompute, and writing `public.artist_embeddings` belong to the pipeline that calls it.

## Configuration

| Parameter | Value |
| --- | --- |
| Dimensions | 128 |
| Iteration weights | `0,1,1,1,1`: five propagation steps, with the first one unweighted |
| Degree normalization β | 0 |
| Projection | Very sparse (Achlioptas, s = 3): ±√3 with probability 1/6 each, otherwise 0 |
| Projection row of a vertex | SplitMix64 over `blake2b64(kind, key)` XOR a per-column salt, derived from the pinned seed `20260924` |

`FastRPConfig().model_version` names all of this. The current value is `fastrp-v1:dim=128:weights=0,1,1,1,1:beta=0:proj=achlioptas-s3:rows=splitmix64(blake2b64(kind,key)):seed=20260924`. `FASTRP_ALGORITHM_VERSION` is bumped whenever a code change alters any output bit, so vectors from two versions of the code never share a `model_version`.

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

The full catalog is 32.8M nodes, at most 222M edges, and 10.2M artist rows. Extrapolating the measurements linearly from 1M to 8M nodes:

| `block_columns` | Peak RSS | Array estimate | FastRP | Build |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 12.6 GB | 10.5 GB | about 10 min | about 1.5 min |
| **4 (default)** | **10.4 GB** | **9.4 GB** | **about 12 min** | **about 1.5 min** |
| 128, one pass | — | 54 GB | — | — |

The pipeline therefore runs with the defaults (four columns per block and two passes), `float16` output, and six threads, which stays within the 12 GB budget. At the peak, the resident set is as follows:

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
covers (artist, release, label, master, genre, style) from `graph.vertex_degree`, and the eight
edge relations that connect them (`graph.by_artist`, `graph.on_label`, `graph.derived_from`,
`graph.in_genre`, `graph.in_style`, `graph.master_by_artist`, `graph.master_in_genre`,
`graph.master_in_style`) — each via a named (server-side) PostgreSQL cursor, fetched in
50,000-row blocks, so the full vertex and edge sets are never materialized as Python lists in
one piece. `fastrp` is then called with the defaults documented above: `out_dtype=np.float16`,
`block_columns=4` (the default), and `threads=6` — the configuration the scaling table above
was measured against, which stays within the 12 GB full-catalog budget.

**Lineage.** `SOURCE_DUMP_ID` and `SOURCE_DUMP_DATE` (required, no default — the invoker
supplies them, since it is the one that knows which dump just landed) become
`artist_embeddings.source_dump_id`/`source_dump_date` on every row the run writes.

**Idempotency.** The job is idempotent per `(source_dump_id, model_version)`: re-running for a
dump already recorded under a `model_version` is a no-op — it does not re-read the graph or
re-run `fastrp`. A *different* dump under the *same* `model_version` is not idempotent against
it: the pipeline recomputes and upserts, updating the existing rows' vectors and lineage in
place, because `model_version` names the method and its parameters, not a point in time. Every
read and write this job makes is scoped to the `model_version` it is computing — it never
touches a row of any other `model_version`, so a version `catalog-api` is currently serving is
untouched by a load of a new one.

**No index DDL, ever.** `embedding_pipeline` holds no DDL privilege and no ownership of
`public.artist_embeddings` — building or rebuilding the ANN index over a `model_version` is
always a separate, more privileged, human- or automation-driven operator step, run after this
job's transaction commits. The job logs the statement that step should run (a
`model_version`-filtered `CREATE INDEX CONCURRENTLY ... USING hnsw (embedding
halfvec_cosine_ops) WHERE model_version = '...'`); per-`model_version` partial indexes
(`gm-database-schema-19g5`) are not landed as of this writing, so the statement logged is the
forward-looking shape the maintainer specified rather than something database-schema documents
today. Retiring a superseded `model_version`'s rows and index, once `catalog-api` has switched
to the new one, is that follow-on's business, never this job's.

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
procedure) is a deployment-repo concern this bead does not fix.

**Testing.** `tests/test_embedding_pipeline.py` exercises the real FastRP/graph code against
small in-memory fakes of the PostgreSQL connection — no database, no Docker.
`tests/integration/test_embedding_pipeline_integration.py` (`just test-integration-pg19`) runs
against a real PostgreSQL 19 + pgvector container with a real `embedding_pipeline`-scoped
login, asserting the idempotency and permission-boundary behavior above against the engine
itself, on a small synthetic graph. That tier's `conftest.py` applies a minimal, inline stand-in
for the ADR 0013 schema objects rather than taking `database-schema` as a dependency —
`database-schema`'s own molecule that adds them (`gm-database-schema-lhp2`) had not reached
`origin/main` as of this bead; see that `conftest.py`'s module docstring for the follow-up.
Neither test tier commits provider-derived data or real embeddings, per ADR 0013's data-rights
section.
