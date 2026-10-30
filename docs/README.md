# Analytics-engine documentation

This directory documents what the `analytics-engine` repository owns and how to operate it. It is written for operators, maintainers, and developers of neighbouring GrooveMap services. Start with the README at the repository root for setup and the `just` recipes.

## Start here

- [Architecture](architecture.md) — service boundaries, data flow, precomputation, the Genre Atlas, and cache consistency.
- [Operations](operations.md) — configuration, health, telemetry, scheduling, shutdown, and failure behavior.
- [Release compliance](release-compliance.md) — package, image, automation, dependency, and publication-readiness checks.

## Embeddings and similar artists

The monthly `analytics-engine-embeddings` job is separate from the always-on service. These pages cover it:

- [FastRP artist embeddings](embeddings.md) — the algorithm, its configuration, and the monthly load pipeline.
- [Exact similar-artist lists](similar_artists.md) — the batch job that computes and publishes top-K lists.

## Measurement records

These pages record completed measurements and the decisions they supported. They are history, not operating guidance.

- [Recall and churn on real embeddings](recall_and_churn.md)
- [Edges-v3 step-0 weight sweep](embedding_weight_sweep.md)
- [Self term tie-break and serving-mode verdict](embedding_tie_break.md)
- [Shipped embedding quality on the chw.2 proxy benchmark](embedding_quality.md)

## History

- [Extraction provenance](extraction.md) — how the service was split out of the former monolith.
