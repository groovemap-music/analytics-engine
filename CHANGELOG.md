# Changelog

All notable changes to this project will be documented here by Commitizen.

## v0.9.0 (2026-11-11)

### Release notes

Adds `analytics-engine-similar-artists` (`insights/similar_artists_job.py`, gm-analytics-engine-dhit), the exact similar-artist job run on its own for an existing embeddings `model_version`, without recomputing embeddings. It is bounded by `--deadline` or `--max-runtime`: at the first block boundary past the deadline it checkpoints atomically and exits 75 (paused), and the next run resumes from the spool. It needs a persistent `--spool-dir` (a `/tmp` spool is refused for resumable runs), publishes once, idempotently, when complete, and ends with a JSON status line. Exit 0 means done or skipped, 75 paused, 1 or 2 failed; see docs/similar_artists.md "Nightly runs". Before each run it checks for load activity (gm-analytics-engine-d6vm): it skips with `status=skipped` and exit 0 while a Discogs or MusicBrainz `loader_extraction_latch` is unrefreshed and under 48 h old, or while the embeddings pipeline holds its run lock. The embeddings pipeline now holds a session advisory lock (key 7419286113405217) for its whole run, so two embeddings runs serialise, and its pool grows from 1 to 2 connections (`embedding_pipeline_login` allows 5 on lux). **Paired floor:** the load check reads `public.loader_extraction_latch` as `embedding_pipeline`, which needs the database-schema release with that grant (gm-database-schema-37cl); without it the job exits failed. Nothing schedules the job yet (gm-deployment-h0d4, homelab-dfmy), and `ENABLE_SIMILAR_ARTISTS` is unchanged.

### Feat

- **similar-artists**: run the exact job on its own, deadline-bounded and resumable
- **similar-artists**: skip the nightly run while a load or embeddings write is running
- **similar-artists**: hold an advisory lock for the embeddings run and check it in the load check

## v0.8.0 (2026-11-10)

### Release notes

The embeddings job's memory estimate now models the whole run, not only FastRP's arrays: it counted about half of the 15.8 GiB peak lux measured at 9,152,168 rows, and is now pinned to that run (gm-analytics-engine-y01w). The edge stream is merged into ~4M-edge mapped arrays so freed batches no longer fragment the heap and keep the stream resident through compute; output edges and embeddings are unchanged (pinned by tests across chunk sizes). Re-measure the peak on the next monthly embeddings run. Adds `python -m insights.similar_artists_sizing`, a read-only sizing and exactness pass for the exact similar-artist job (gm-analytics-engine-3am): it connects with `default_transaction_read_only=on`, writes nothing to PostgreSQL, publishes nothing, and reports per-phase timing, peak RSS and a full-catalog extrapolation for a slice of artists, plus a seeded 2,000-artist check against an independent brute force. `ENABLE_SIMILAR_ARTISTS` stays off; the full-catalog run needs its own operator approval. The PostgreSQL 18 and pgvector integration suites now run in CI. No schema dependency.

### Feat

- **embeddings**: add a read-only slice sizing pass for the exact similar-artist job

### Fix

- **embeddings**: model the whole job's peak and stop the edge stream fragmenting the heap
- **release**: keep breaking changes below 1.0 with major_version_zero

## v0.7.0 (2026-11-07)

### Release notes

Needs the database-schema release with `label_longevity.is_placeholder` first. Stores placeholder labels after the real ranks and adds include_placeholders to /api/insights/label-longevity, off by default. Promotes internal-insights 1.10.0.

### Feat

- **insights**: let callers include placeholder labels in label longevity
- **contracts**: promote internal-insights 1.10.0 and pin database-schema 0810604

## v0.6.2 (2026-11-06)

### Release notes

Stores and serves a null completeness_pct. Needs database-schema v0.8.0 or later, and must be live before catalog-api v0.8.1, which returns null for artists and labels.

### Fix

- **insights**: store and serve a null completeness_pct

## v0.6.1 (2026-11-06)

### Release notes

No schema change. The insights status reports interrupted computations and the last completed time for an unfinished latest run.

### Fix

- **status**: report the last completed time for any unfinished latest run
- **status**: show interrupted computations in the insights status

## v0.6.0 (2026-11-05)

### Feat

- **insights**: write label_ids and artist_ids in the rarity COPY
- **contracts**: promote internal-insights 1.7.0 and pin database-schema v0.7.0

### Fix

- **insights**: walk rarity pages instead of the whole-catalog route

### Perf

- **insights**: store release_rarity with one COPY instead of a round trip per row
- **embeddings**: binary read-back, COPY write and binary similar-artist COPY
- **embeddings**: derive credited artist ids from the edge stream, drop server DISTINCT

## v0.5.0 (2026-10-30)

### Feat

- **cache**: document and test Valkey TLS configuration

## v0.4.0 (2026-10-30)

### Feat

- **cache**: prefix all Valkey keys and scans via python-libraries helpers
- **insights**: store style co-tags and serve them per style
- **contracts**: promote internal-insights 1.5.0 for style-cotags
- **insights**: store landmark masters and serve them in the style atlas
- **contracts**: promote internal-insights 1.4.0 for style-landmarks
- **insights**: mark atlas emergence edges from stored genre years
- **contracts**: promote internal-insights 1.3.0 and test settled vs latest stage
- **insights**: serve GET /api/insights/style-atlas with generation-keyed caching
- **insights**: derive style lifecycle stages and the month-pinned Genre Atlas layout
- **insights**: compute style year counts and dominant parent genres
- **embeddings**: default the monthly pipeline to the self-term config

### Fix

- **deps**: upgrade msgpack past CVE-2026-114613
- **deps**: upgrade filelock to 4.2.1 past CVE-2026-112940
- **deps**: upgrade certifi and opentelemetry-api past PYSEC-2026-4320 and CVE-2026-114603

## v0.3.0 (2026-10-05)

### Feat

- **cache**: migrate analytics client and naming to Valkey
- **embeddings**: run the similar-artist stage after each monthly load
- **embeddings**: spool, COPY and publish exact similar-artist lists
- **embeddings**: default the monthly pipeline to the 0.05 self term
- **embeddings**: add an exact all-pairs top-K cosine kernel

### Fix

- **build**: align analytics image runtime wheel with Valkey helper pin
- **embeddings**: keep unaccepted similar artist publishing off by default
- **embeddings**: write compact release arrays and retain displaced current

### Perf

- **embeddings**: narrow exact column candidate segments

## v0.2.0 (2026-10-03)

### Feat

- **scripts**: thread --self-weight through embeddings_from_dump
- **embeddings**: add a deterministic FastRP self term
- **embeddings**: harden the sweep pipeline and drop the m=32 recall variant
- **embeddings**: measure tie-tolerant ANN recall on the September embeddings
- **embeddings**: add track-level credits and track performers to the edge set
- **embeddings**: apply the real database-schema to the pg19 integration tier
- **embeddings**: checkpoint August's result so a restart can skip it
- **embeddings**: configurable per-month maintenance_work_mem + container restart
- **embeddings**: add the Docker/pgvector recall and churn measurement script
- **embeddings**: add release-level credited-artist edges to the graph
- **embeddings**: stream-parse dumps for real-embedding recall/churn measurement
- **embeddings**: load embeddings monthly with lineage under the pipeline role
- **telemetry**: add embedding-pipeline rows-written and failure counters
- **embeddings**: add deterministic column-blocked FastRP
- **insights**: compute and expose the activity summary
- **insights**: add the consent-aware activity read path
- **telemetry**: trace insights computations and sample the event loop
- **insights**: persist and expose media families, family signals, and medium rarity
- **telemetry**: adopt common.telemetry and record insights domain metrics

### Fix

- **logging**: pin thread-safe runtime context
- **scripts**: use the monotonic getrusage peak for the memory guard
- **scripts**: persist self_weight in the shipped npz
- **embeddings**: use memorystatus_level, not free swap, for the host-pressure guard
- **embeddings**: stream a month's vectors instead of holding them all in RAM
- **embeddings**: build the real per-model_version PARTIAL HNSW index
- **embeddings**: drop the COPY fallback's explicit NULL computed_at
- **embeddings**: compose the real stored model_version, not the bare one
- **embeddings**: self-exclude ANN results client-side, not via a WHERE filter
- **embeddings**: align credited_by_artist with ieu.6's real name-join rule
- **embeddings**: derive the HNSW index name from a hash, not a truncated slug
- **embeddings**: key the stored model_version per dump, not per algorithm
- **ci**: accept commitizen's no-eligible-commits bump-preview state
- **ci**: use public python libraries

### Refactor

- **insights**: simplify computation lifecycle

## v0.1.1 (2026-08-31)

### Fix

- **ci**: accept release-boundary bump states

## v0.1.0 (2026-08-31)

The `v0.1.0` workflow failed before publishing artifacts or images. The tag is retained as an immutable record of that release attempt.
