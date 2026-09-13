# Analytics-engine documentation

This directory documents the responsibilities and operating contract of the `analytics-engine` repository.

- [Architecture](architecture.md) — service boundaries, data flow, precomputation, and cache consistency.
- [Operations](operations.md) — configuration, health, scheduling, shutdown, and failure behavior.
- [Release compliance](release-compliance.md) — package, image, automation, dependency, and publication-readiness checks.
- [Extraction provenance](extraction.md) — what moved from the monolith and which repositories own adjacent responsibilities.

The complete HTTP request and response schema consumed from `catalog-api` is the promoted [OpenAPI contract](../contracts/catalog-api/internal-insights/v1/openapi.yaml). Its [provenance record](../contracts/catalog-api/internal-insights/v1/source.json) pins the producer revision and file digests. Historical implementation plans are preserved in the private organization planning archive rather than published as current product documentation.
