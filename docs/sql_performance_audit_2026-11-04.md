# SQL performance audit, 2026-11-04

Bead `gm-analytics-engine-afek`. Read-only: no production code changed, nothing was run against lux.

## Scope and evidence level

This is a **static review plus local measurements on synthetic data**. It is not a measurement on the real catalog.

- On 2026-11-04 lux Postgres held only part of the load: 3.97M of 19.6M releases and 7.9M of 10.3M artists. Labels and masters were complete. The operator directed that the review should not wait for a complete load, and that Postgres should not be queried while the load's maintenance window changes indexes. **No query was sent to lux.**
- Local numbers come from throw-away containers (`postgres:18-alpine`, and the pgvector image for the embedding tables), loaded with generated rows shaped like the production DDL (`database-schema` v0.6.1 + 3 commits: SHA-256 unique keys, hash indexes on `person_name`). Datasets stayed under 1.7 GB, and the containers and their volumes are removed.
- Every local timing is a **warm-cache, single-session, small-table lower bound**. The "full catalog" column is a linear extrapolation from row counts in `docs/embeddings.md` (51.0M release-level credited-artist edges, about 96.5M kept track-credit edges, 24.4M track-performer edges, about 8.2M artist rows). Where scaling is not linear, the text says so.
- `EXPLAIN` shapes come from synthetic statistics. The planner's choice on the real tables may differ, which is itself the first finding below.

## Where the SQL is

About 95 statement lines sit behind three surfaces. Almost none of the expensive aggregation is in this repository.

| Surface | Statements | Tables | Class | Cost |
| --- | --- | --- | --- | --- |
| Embedding load, `insights/embedding_pipeline.py` | 11 edge scans, 6 vertex-discovery scans, 1 `graph.vertex_degree` scan, 1 upsert, 1 idempotency probe | `graph.*` (read), `public.artist_embeddings` (write) | monthly batch | the hot spots below |
| Similar-artist stage, `insights/similar_artists.py` | read-back, `COPY`, release registry helpers, lock and guard | `public.artist_embeddings`, `public.artist_similar_artists`, `public.artist_embedding_releases` | monthly batch | minutes to tens of minutes; see H4, H5 |
| Scheduled computations, `insights/computations.py` | `DELETE` + row-by-row `INSERT` per insight, plus the style-atlas reads | `insights.*` | daily batch | small; see H6 |
| Activity summary, `insights/activity.py` | consent join, keyset pages over `activity.events` and `activity.impressions` | `activity.*` | daily batch | scales with event volume; see H3 |
| API reads, `insights/insights.py`, `style_atlas_read.py` | one indexed `SELECT ... LIMIT` per endpoint, 8 `computation_log` probes for `/status` | `insights.*` | user-facing | not a hot spot |

The heavy graph aggregations that the timeouts in `computations.py` mention (data completeness at about 400 s, rarity, community enrichment, style year counts, landmarks, co-tags) are **computed by `catalog-api` and fetched over HTTP**. Their SQL or Cypher is not in this repository, so it cannot be audited here. File that audit against `catalog-api` (proposal F7).

### User-facing latency

Every API read selects from a small precomputed `insights.*` table by a column that has an index, with a `LIMIT`, behind a cache-aside layer. Examples are `release_rarity (rarity_score DESC)`, `monthly_anniversaries (computed_year, computed_month)`, `computation_log (insight_type, started_at DESC)` and `activity_summary (summary_date DESC)`. The one possible exception is `ORDER BY rank LIMIT n` on `artist_centrality` and `label_longevity`, which hold at most 100 and 50 rows by construction, so no index is needed. **No user-facing hot spot was found**; the work is in the batch jobs.

## Ranked hot spots

### H1. The credit joins repeat their work, and the planner may pick the slow join (embedding load) , highest

`_CREDITED_ARTIST_EDGE_SQL` and `_CREDITED_ARTIST_IDS_SQL` run the **identical** `credited_on JOIN same_as ON person_name` with a `DISTINCT`. The second returns only a column the first already returns. `_TRACK_CREDITED_ARTIST_EDGE_SQL` and `_TRACK_CREDITED_ARTIST_IDS_SQL` repeat the pair on `track_credited_on`. The release-id discovery and the track-performer scans (`track_by_artist` read three times) repeat in the same way. So each large join and its `DISTINCT` run twice per load, and the second pass carries no new information.

Local evidence (3.0M `credited_on` rows, of which 1.71M kept-category; 1.5M `same_as` rows; warm):

| Query | Execution | Plan |
| --- | --- | --- |
| edge query (`DISTINCT release_id, artist_id`) | 8.07 s | Nested Loop, 1.02M probes of `same_as_person_name_hash` |
| artist-id discovery (`DISTINCT artist_id`) | 6.29 s | the same Nested Loop |
| release-id discovery | 0.74 s | Seq Scan + HashAggregate |
| edge query with `enable_nestloop=off` | **4.02 s** | Hash Join, build side `same_as`, 101 MB hash |
| the join without `DISTINCT` (rows streamed) | 6.06 s | Nested Loop |

- The artist-id discovery is 6.29 of 15.1 s, **about 42% of the credit stage**, for rows the edge scan already streams.
- With the hash index in place the planner **chose the Nested Loop**. The estimate is `cost=0.00..0.06` per probe, and the hash join was 2x faster even warm. Warm, one probe costs about 4.7 µs. At full size, `same_as` and its hash index will not fit in `shared_buffers`, so each probe becomes a random read. The loop then does one probe per kept credit row: at least 51M for release credits, and about 96.5M for track credits.
- `DISTINCT` forces a `HashAggregate` that holds every output row before the first one is returned. The local run used 127 MB for 1.92M groups, about 66 B per group. That gives **about 3.4 GB for 51M edge groups and about 6.4 GB for 96.5M**, or the same volume again as temp files if `work_mem` is lower, on a Postgres shared with other homelab databases. The named cursor streams the result, but only after that aggregate is complete.
- Full-catalog extrapolation (linear in kept rows, warm): hash join about 2.3 µs per row, so about 2 min for release credits and about 4 min for track credits. Nested Loop about 4.7 µs per row, so about 4 min and 8 min warm, and plausibly 10x that if probes go to disk. Treat this as a lower bound.
- The earlier role-category index (`credited_on_role_category`) was retired by v0.6.1 with the old key. The `role_category = ANY(...)` filter now has no index and is a sequential scan of `credited_on`. That is the right access path, because the kept categories are most of the rows (1.71M of 3.0M here), but it means the scan reads every credit row.

**Cause.** Discovery was written as separate queries so the node set exists before edges stream. The artist set is a pure projection of the edge result. Resolving names through `same_as` is a join over a table far larger than the set of names it needs, and nothing steers the planner to the hash join.

**Proposed fix (F1).** (a) Collect artist ids from the edge stream and delete `_CREDITED_ARTIST_IDS_SQL`, `_TRACK_CREDITED_ARTIST_IDS_SQL` and `_TRACK_PERFORMER_ARTIST_IDS_SQL`. This needs a node set known before the edges, so build the `NodeIndex` after the stream rather than before it (or stream edges once into a growing key buffer). (b) Keep the release-id queries; they are cheap (0.74 s) because they skip the join, and they cover releases whose credits never resolve through `same_as`. (c) Make the join shape explicit: `SET LOCAL enable_nestloop = off` for the load transaction, or a one-off `CREATE TEMP TABLE` of distinct credited names. Choose after measuring on the full data.

**Gain.** About 40% of the credit stage in the local run, and more at scale because the second pass is the one that pays the random I/O. Estimate: 30 to 50% of the edge-read phase. **Risk.** Medium: `_EDGE_SET_VERSION` and bit-identity. The node set must stay equal, and the artist output order must be shown not to change any stored vector (not verified in this audit whether anything depends on that order). Compare node sets and a vector checksum on a fixed fixture before and after.

### H2. `SELECT DISTINCT` pushed into the database for edges that are deduplicated again in Python

`Adjacency.build()` already collapses parallel edges (see the comment above `_TRACK_CREDITED_ARTIST_EDGE_SQL`), yet three of the eleven edge queries apply `DISTINCT` server-side. That is where the 3 to 6 GB aggregate state in H1 comes from, and `track_by_artist` spilled to temp files even at 2.0M rows with `work_mem=64MB` (`Batches: 5`, 1.6 MB of temp). Without `DISTINCT` the join streams: 6.06 s against 8.07 s for the Nested Loop plan.

**Proposed fix (F2).** Drop `DISTINCT` from the edge queries and let the CSR build dedupe, or dedupe in blocks. **Gain.** Removes the multi-GB aggregate and the temp spill on a shared server, and about 25% of the query time locally. **Risk.** The client sees up to about 2x more rows per relation (a person credited under several roles). That moves cost into Python's `node_keys` path, which is already the CPU-bound step (`str(...)` and hashing per row for 244M edges), so measure the full run before and after. Do F1 and F2 together; they share the same function.

### H3. Activity keyset paging rescans the whole window on every page

`_page_query` orders by `(occurred_at, event_id)` and the only indexes are `(subject_id, occurred_at DESC)` and `(event_type, occurred_at DESC)`. No index serves the ordering, so each page is a scan of the window followed by a top-N sort. Local test: 400k events in one partition, 15,000 eligible subjects, **102 ms per 1,000-row page** (Parallel Seq Scan, `Rows Removed by Filter: 111,107` per worker, 6,190 buffers).

- Total work is pages x window, so quadratic: N events read in pages of 1,000 cost about N/1000 x (N x 0.25 µs). 4M events would take about 4,000 pages x about 1 s, about 70 min, against a straight scan of a few seconds. The batch size is `DEFAULT_BATCH_SIZE`; the constant was not changed.
- The full eligible-subject array is sent as a bind parameter on every page, so request size is pages x subjects.
- Activity volumes are small today and the tables are empty on lux, so this has not bitten yet. It is the one statement that gets **worse than linearly** as data grows.

**Proposed fix (F3).** Add a per-partition index on `(occurred_at, event_id)` (database-schema change), or page by partition and join consent through `activity.user_subjects` in SQL instead of shipping the array. **Gain.** From quadratic to linear. **Risk.** Low for the index; cross-repo for the schema owner.

### H4. Read-back of the embeddings through `embedding::text`

`_READ_EMBEDDINGS_SQL` selects `embedding::text ... ORDER BY artist_id`. Local (pgvector image, 300k rows, 128 dimensions): **26k rows/s including the NumPy parse**, so about 5 min for 8.2M rows. That is smaller than H1 but runs before every top-K compute. Two things are unmeasured: (i) `ORDER BY artist_id` walks the primary key `(artist_id, model_version)` and filters by version, so with several dumps stored it reads every version's rows (about 2x with the last two months kept). (ii) The text form is parsed in Python.

**Proposed fix (F4).** Use binary `COPY ... TO STDOUT` or `halfvec` binary transfer, and read in `artist_id` order from the per-version index. **Gain.** Parse time off the critical path, plus a bounded read when versions accumulate. **Risk.** Low.

### H5. Embedding upsert and similar-artist `COPY` are Python-bound

| Step (local, 300k rows) | Rate | Full catalog (8.2M) |
| --- | --- | --- |
| `executemany` upsert, 5,000-row batches | 5.7k rows/s (11.1 s of 52.7 s is building the vector literal) | about 24 min |
| `COPY` of the same rows | 11.5k rows/s | about 12 min |
| similar-artist `COPY`, K=50 arrays (200k artists) | 8.0k artists/s, row build 16% | about 17 min |

These are one-shot, off the request path, and the upsert is a single transaction. The `ON CONFLICT` upsert is only needed for crash-retry (the docstring says so); a `COPY` into the version's rows, after a delete of any partial rows, halves the time. The `artist_similar_artists` K-length trigger per row was **not** included in the local table, so the 17 min is a floor. Memory and disk for the stage are already sized in `docs/similar_artists.md` (peak about 9 GB, 11.3 GB spool); the compute duration belongs to `gm-analytics-engine-3am`, so this audit does not duplicate it.

**Proposed fix (F5).** Switch the write to `COPY`, and measure the trigger cost on the real table. **Gain.** About 12 min per month. **Risk.** Low to medium (crash-retry semantics).

### H6. `DELETE` and row-by-row `INSERT` per insight

`compute_and_store_*` replace each table with `DELETE FROM` and one `execute` per row (artist centrality, genre trends, label longevity, data completeness, monthly anniversaries and, importantly, release rarity). Rarity's row count is set by `catalog-api` and **not bounded here**; it is written one round trip per row and has three indexes. If it is in the millions, that is the longest daily write. The DELETE inside the transaction also leaves dead tuples for autovacuum on every run.

**Proposed fix (F6).** `executemany` (pipeline mode) or `COPY` for rarity, and `TRUNCATE` if readers can tolerate its lock, or a swap table. First ask `catalog-api` for the row count. **Gain.** Unknown until the rarity size is measured. **Risk.** Low.

## Batch jobs: duration and peak resource

| Job | Duration | Peak resource | Basis |
| --- | --- | --- | --- |
| Edge and vertex reads (H1, H2) | at least 6 to 12 min warm for the credit joins, plus the 8 plain scans; unknown cold | Postgres: 3.4 to 6.4 GB hash aggregate or the same in temp files; client: the 244M-edge CSR build (10.1 GB estimate in `docs/embeddings.md`) | linear extrapolation from the local run |
| FastRP compute | about 12 min (docs) | 10.1 GB RSS estimate | `docs/embeddings.md`; not re-measured |
| Embedding write (H5) | about 24 min | one transaction, the PK index maintained | extrapolated |
| Read-back (H4) | about 5 min | halfvec array 2.4 GB | extrapolated |
| Similar-artist compute | 5 to 10 h | about 9.0 GB | `docs/similar_artists.md`; sizing is bead 3am |
| Similar-artist `COPY` (H5) | about 17 min | WAL, plus 17.6 GB of lists | extrapolated |
| Daily insight writes (H6) | seconds, except rarity (unmeasured) | small | static |

**Impact on concurrent reads**, not measurable here. The hash aggregate, the sequential scan of the three credit tables, and the 8.8 GB `artist_similar_artists` release write all compete for I/O and WAL with the other databases on the shared Postgres. No read was run against lux. When the load completes, a live read of `pg_stat_statements` and `pg_stat_io` during a monthly run is the way to settle it.

## What is not a problem

- The eight plain `graph.*` edge scans are one sequential pass each through a server-side cursor with a 50,000-row fetch. They are as cheap as a scan can be.
- `graph.vertex_degree` is read once per load.
- The hash index on `person_name` is correct for the join (it serves `=`), and is why the query text did not need to change. The risk is only the plan choice in H1.
- The idempotency probe and registry calls are single-row.

## Live reads wanted from the operator (after the load's maintenance window)

All read-only, to run once the load finishes and the indexes settle:

1. Done on 2026-11-04 for the two edge queries and the artist-id discovery (see the live-reads section). Still wanted: the same after v0.6.1 is applied on lux and the load completes, with and without `enable_nestloop`, which `gm-lux-sql` cannot set.
2. Done for partial data (see above). Repeat after the load: `SELECT relname, reltuples, relpages FROM pg_class` for `graph.credited_on`, `same_as`, `track_credited_on`, `track_by_artist` and `insights.release_rarity`, to replace the extrapolated row counts.
3. Done (see above).
4. `activity.events` is empty today; repeat when it holds data, to place H3.

## Live reads on lux (read-only, 2026-11-04 ~20:00 UTC)

Taken through the operator's read-only `gm-lux-sql` wrapper, one statement at a time, with Postgres idle (the load is stalled). **Scale: partial data.** `credited_on` held 18.0M rows (14.6M kept-category), `track_credited_on` 21.7M, `same_as` 2.85M, `track_by_artist` 7.9M. The full 2026-08 dump is about 147M credit rows, so scale these timings by 3x to 5x. `gm-lux-sql` cannot `SET enable_nestloop`, so the hash-join comparison from the local run could not be repeated live.

**Settings.** `work_mem` 64 MB, `hash_mem_multiplier` 2 (so 128 MB per hash node), `shared_buffers` 6 GB, `effective_cache_size` 18 GB, `random_page_cost` 1.1, `max_parallel_workers_per_gather` 4, `statement_timeout` 120 s, `temp_file_limit` unlimited.

**Important: lux still has the pre-v0.6.1 indexes.** `same_as_pkey` is `btree (person_name, artist_id)`, `credited_on_role_category` still exists, and there is no `person_name` hash index. So the live plans below are the *old* shape. The local hash-index finding in H1 is a prediction for after the migration, not something lux shows yet.

| Query (partial data) | Execution | Plan | Notes |
| --- | --- | --- | --- |
| `_CREDITED_ARTIST_EDGE_SQL` | **25.9 s** | Seq Scan `credited_on` (14.6M kept of 18.1M), Nested Loop + Memoize over `same_as_pkey`, HashAggregate | 12.5M groups; **Batches 9, 425 MB temp spilled**; Memoize 86% hit (2.04M misses, 1.03M evictions) |
| `_CREDITED_ARTIST_IDS_SQL` | **13.4 s** | Index Only Scan `credited_on_role_category`, same Nested Loop + Memoize, HashAggregate | 1.79M groups, in memory; **repeats the join the edge query already did** |
| `_TRACK_CREDITED_ARTIST_EDGE_SQL` | **26.2 s** | Seq Scan `track_credited_on` (21.8M), Nested Loop + Memoize, HashAggregate | 12.5M groups; **Batches 17, 637 MB temp spilled** |

What this changes in the report:

- **H1 is confirmed live.** The artist-id discovery cost 13.4 s next to a 25.9 s edge query, so 34% of the release-credit stage (31% with the track-level queries included). It is the same join, and its output is a projection of the edge query.
- **The cost is per kept credit row, about 1.8 µs (release) and 1.2 µs (track) here.** At 3x to 5x the rows that is roughly 1.5 to 2.5 min per credit edge query on a warm cache. Memoize keeps the old btree plan healthy while the distinct-name working set fits (86% to 93% hits). Whether the plan stays good after the hash-index migration is open: Memoize applies to nested loops generally, so the migration may keep the benefit, but this needs a check on lux once v0.6.1 is applied.
- **H2 is confirmed, and the memory estimate was too high.** Under the live `work_mem` the aggregate spills rather than holding gigabytes: **425 MB and 637 MB of temp for 12.5M groups each**. Linearly, that is about 1.5 to 2 GB of temp files per edge query at full size, on the shared server, not the 3.4 to 6.4 GB of RAM estimated earlier. Treat the earlier RAM figures as the upper bound if `work_mem` is raised. The temp writes are real shared-disk I/O either way.
- The Seq Scan of `credited_on` is 1.7 s warm and about 100k pages read; not a concern at this scale.
- Not read live: `activity.events` is empty (partitions `events_y2026m11`, `m12` and default are 40 kB), so H3 stays an extrapolation. `insights.release_rarity` is empty (40 kB, `reltuples` -1), so H6 still has no row count.

## Proposed follow-up beads

Not filed here; the bead asks for proposals. Each carries a measured target to confirm on the live data.

| ID | Title | Target |
| --- | --- | --- |
| F1 | Embedding pipeline: derive artist ids from the edge stream, remove three discovery joins, pin the hash join | credit-read phase down at least 30% on the same data; node set and vector checksum unchanged |
| F2 | Embedding pipeline: drop server-side `DISTINCT` from the edge queries | zero temp files and no multi-GB aggregate on live Postgres; end-to-end read time not worse |
| F3 | Activity reads: linear keyset paging (index or per-partition paging) | time per page flat as the window grows; a 4M-event window under 1 min |
| F4 | Similar-artist read-back: binary transfer, per-version ordered read | 8.2M rows read in under 2 min |
| F5 | Embedding write and similar-artist publish via `COPY` | write at least 2x faster; crash-retry test still passes |
| F6 | Daily insight writes: batch the rarity insert (after counting its rows) | rarity write time per row count, target at least 10x fewer round trips |
| F7 | catalog-api: audit the uncached full-scan insight queries (data completeness, rarity, community enrichment, style year counts, landmarks, co-tags) | cold-cache time under the current 900 to 1800 s budgets with margin |

F1 and F2 touch the same function and should be one molecule. F3 needs a database-schema index. Coordinate F4 and F5 with `gm-analytics-engine-3am`.

## Method notes

- Local schema: `graph.credited_on`, `same_as`, `track_credited_on` and `track_by_artist` per `database-schema` `postgres.py` (generated `role_category`, unique key over `graph.text_key(...)`, hash index on `person_name`). Synthetic names are Zipf-skewed; the real distribution of credits per name may make the join output larger or smaller.
- Containers: `postgres:18-alpine` (`shared_buffers=512MB`, `work_mem=64MB`) and the local pgvector image, each with a memory limit. Both were started with `--rm`, removed with `docker rm -f -v`, and left no volumes.
- The scripts were kept in the session scratchpad and are not part of this change.
