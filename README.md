# GrooveMap analytics engine

Private, independently versioned service for scheduled music analytics. It obtains raw graph and catalog inputs from `catalog-api`, stores precomputed results in PostgreSQL, uses Redis for cache-aside reads, and exposes an internal FastAPI service on port 8008 with health checks on port 8009.

This source is available under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires a separate license.

## Development

Prerequisites are pinned in `.mise.toml`. The private `groovemap-runtime` dependency is fixed to an immutable `python-libraries` commit.

```bash
mise install
just setup
just check
```

The stable repository interface is:

- `just setup` — install the locked development environment.
- `just check` — run the authoritative pre-merge gate.
- `just test` — run the isolated test suite with coverage.
- `just build` — create wheel and source distributions.
- `just image` — build and inspect the non-root production image.
- `just release-dry-run` — build checksums, SBOM, notices, and provenance without publishing.
- `just bump-preview` — preview the Conventional Commits version and changelog without changing files.

## Runtime

Run locally after setup:

```bash
uv run analytics-engine
```

Required configuration includes PostgreSQL credentials (`POSTGRES_HOST`, `POSTGRES_USERNAME`, `POSTGRES_PASSWORD`, and `POSTGRES_DATABASE`). Optional settings include `API_BASE_URL`, `REDIS_HOST`, `INSIGHTS_INTERNAL_SECRET`, `INSIGHTS_SCHEDULE_HOURS`, `INSIGHTS_MILESTONE_YEARS`, and `LOG_LEVEL`. Supply real secrets through the deployment layer; never commit them here.

## Repository boundary

`catalog-api` owns the internal HTTP interface and query implementations. This repository consumes the promoted contract in `contracts/catalog-api/internal-insights/v1/`; it does not import API source or rely on a sibling checkout. Database schema ownership belongs to `database-schema`; runtime and resilience helpers belong to `python-libraries`; service orchestration and secret examples belong to `deployment`.

The Docker build only needs this repository plus a locally prepared wheel for the pinned private runtime. `scripts/prepare-runtime-wheel.sh` verifies the adjacent runtime checkout is both clean and at the expected commit before staging that wheel in the ignored `.build/` directory.

## Releases

The project is independently versioned from PEP 621 metadata with Commitizen and `v$version` annotated tags. Migration verification is deliberately non-publishing. A hosted release workflow remains disabled until a short-lived GitHub App installation token can read the private runtime repository and an approved artifact publishing identity is designed.

See [docs/extraction.md](docs/extraction.md) for source-history provenance and the retained path set.
