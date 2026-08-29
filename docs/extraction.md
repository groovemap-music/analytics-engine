# Extraction provenance

`analytics-engine` was extracted from the former GrooveMap monolith as an independently versioned service. The extraction retained service-owned implementation and tests while replacing source-tree imports with explicit repository contracts.

## Ownership after extraction

- This repository owns scheduled computation, PostgreSQL result persistence, Redis cache behavior, and read-only analytics endpoints.
- `catalog-api` owns raw graph and catalog queries. This service consumes its promoted internal HTTP contract from `contracts/catalog-api/internal-insights/v1/`.
- `database-schema` owns PostgreSQL schema initialization.
- `python-libraries` owns shared runtime, resilience, configuration, and health-server helpers.
- `deployment` owns service orchestration, secret delivery, and environment-specific configuration.

The service does not import from sibling checkouts or require a monorepo build context. `scripts/prepare-runtime-wheel.sh` verifies the exact `python-libraries` revision before staging its wheel for installation and image builds.

Historical licensing is recorded in [NOTICE](../NOTICE). Historical implementation plans are preserved privately; the current architecture decisions that remain operative are documented in [architecture.md](architecture.md).
