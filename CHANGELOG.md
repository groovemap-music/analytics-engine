# Changelog

All notable changes to this project will be documented here by Commitizen.

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
