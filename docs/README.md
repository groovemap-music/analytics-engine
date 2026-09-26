# Analytics-engine documentation

This directory documents the responsibilities and operating contract of the `analytics-engine` repository.

- [Architecture](architecture.md) — service boundaries, data flow, precomputation, and cache consistency.
- [Operations](operations.md) — configuration, health, scheduling, shutdown, and failure behavior.
- [Release compliance](release-compliance.md) — package, image, automation, dependency, and publication-readiness checks.
- [FastRP artist embeddings](embeddings.md) — the deterministic embedding computation, its determinism guarantees, and its measured scaling.
- [ANN recall and churn](recall_and_churn.md) — real-embedding recall@10 vs. `ef_search`, month-over-month churn, and the graph-scope finding that led to gm-analytics-engine-ieu.6.
- [Extraction provenance](extraction.md) — what moved from the monolith and which repositories own adjacent responsibilities.
- [Embedding quality](embedding_quality.md) — shipped FastRP similar-artist recall on the chw.2 proxy benchmark, vs. the design spike and the production heuristic.

The complete HTTP request and response schema consumed from `catalog-api` is the promoted [OpenAPI contract](../contracts/catalog-api/internal-insights/v1/openapi.yaml). Its [provenance record](../contracts/catalog-api/internal-insights/v1/source.json) pins the producer revision and file digests. Historical implementation plans are preserved in the private organization planning archive rather than published as current product documentation.
