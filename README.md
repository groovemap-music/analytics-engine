# GrooveMap analytics engine

Independently versioned service for scheduled music analytics. It obtains raw graph and catalog inputs from `catalog-api`, stores precomputed results in PostgreSQL, uses Redis for cache-aside reads, and exposes a FastAPI service on port 8008 with health checks on port 8009.

This project is licensed under the [GNU Affero General Public License v3.0 only](LICENSE). Commercial use is permitted under the AGPL when its terms are followed; [alternative commercial terms may be negotiated](COMMERCIAL-LICENSING.md).

External contributions are temporarily paused until a relicensing-capable contributor agreement is approved. See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes.

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

OpenTelemetry metrics and traces export to a collector when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Alongside the HTTP, database, and computation metrics, the service reports the process view and its event-loop lag, and opens one `insights {computation}` root span per scheduled computation so a slow insight is attributable to the calls it made. See [docs/operations.md](docs/operations.md#telemetry) for the full variable list, the metrics, and the spans this service emits. Each signal can be turned off on its own, and with no endpoint configured telemetry is a no-op and the service behaves exactly as it does today.

## Repository boundary

`catalog-api` owns the internal HTTP interface and query implementations. This repository consumes the promoted contract in `contracts/catalog-api/internal-insights/v1/`; it does not import API source or rely on a sibling checkout. Database schema ownership belongs to `database-schema`; runtime and resilience helpers belong to `python-libraries`; service orchestration and secret examples belong to `deployment`.

The Docker build only needs this repository plus a locally prepared wheel for the pinned private runtime. `scripts/prepare-runtime-wheel.sh` verifies the adjacent runtime checkout is both clean and at the expected commit before staging that wheel in the ignored `.build/` directory.

## Releases

The project is independently versioned from PEP 621 metadata with Commitizen and approved `v$version` annotated tags. Migration and release-readiness verification are deliberately non-publishing; the hosted workflow only responds to an explicitly created version tag.

See the [documentation index](docs/README.md) for architecture, operations, extraction provenance, and release-compliance guidance.
