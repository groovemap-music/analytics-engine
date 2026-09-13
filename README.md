# GrooveMap analytics engine

Independently versioned service for scheduled music analytics. It obtains raw graph and catalog inputs from `catalog-api`, stores precomputed results in PostgreSQL, uses Redis for cache-aside reads, and exposes a FastAPI service on port 8008 with a separate health listener on port 8009.

This project is licensed under the [GNU Affero General Public License v3.0 only](LICENSE). Commercial use is permitted under the AGPL when its terms are followed; [alternative commercial terms may be negotiated](COMMERCIAL-LICENSING.md).

External contributions are temporarily paused until a relicensing-capable contributor agreement is approved. See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes.

## Development

**Public-library cutover: complete.** Prerequisites are pinned in `.mise.toml`, and the
first-party `groovemap-runtime` dependency comes from the public `python-libraries` repository at
an immutable commit. Local setup and CI require no private-package credentials.

```bash
mise install
just setup
just check
```

The stable repository interface is:

- `just setup` — install the locked development environment.
- `just check` — run the authoritative pre-merge gate.
- `just test` — run the isolated test suite with coverage.
- `just coverage` — produce the same test coverage report used by CI.
- `just contract-check` — verify the promoted `catalog-api` contract and generated binding.
- `just secret-scan` — scan Git history and the working tree with Gitleaks.
- `just build` — create wheel and source distributions.
- `just image` — build and inspect the non-root production image.
- `just release-dry-run` — build checksums, SBOM, notices, and provenance without publishing.
- `just bump-preview` — preview the Conventional Commits version and changelog without changing files.

## Runtime

Run locally after setup:

```bash
uv run analytics-engine
```

Required configuration includes `POSTGRES_HOST`, `POSTGRES_USERNAME`, `POSTGRES_PASSWORD`, and `POSTGRES_DATABASE`. Username, password, Redis password, and the internal API secret support the shared `_FILE` secret convention. The complete runtime and telemetry variable table, including defaults and validation behavior, is in [the operations guide](docs/operations.md#configuration). Supply real secrets through the deployment layer; never commit them here.

OpenTelemetry metrics and traces export to a collector when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Alongside the HTTP, database, and computation metrics, the service reports the process view and its event-loop lag, and opens one `insights {computation}` root span per scheduled computation so a slow insight is attributable to the calls it made. See [docs/operations.md](docs/operations.md#telemetry) for the full variable list, the metrics, and the spans this service emits. Each signal can be turned off on its own, and with no endpoint configured telemetry is a no-op and the service behaves exactly as it does today.

## Repository boundary

`catalog-api` owns the internal HTTP interface and query implementations. This repository consumes the promoted contract in `contracts/catalog-api/internal-insights/v1/`; it does not import API source or rely on a sibling checkout. Database schema ownership belongs to `database-schema`; runtime and resilience helpers belong to `python-libraries`; service orchestration and secret examples belong to `deployment`.

The Docker build only needs this repository plus a locally prepared wheel for the pinned public
runtime. `scripts/prepare-runtime-wheel.sh` verifies the source checkout is clean and at the
expected commit before staging that wheel in the ignored `.build/` directory. Its optional
`GROOVEMAP_RUNTIME_REPO` override accepts an explicit matching checkout; without the override, the
script uses a matching adjacent checkout or creates a temporary checkout from the public source.

## Releases

The project is independently versioned from PEP 621 metadata with Commitizen and approved `v$version` annotated tags. Migration and release-readiness verification are deliberately non-publishing; the hosted workflow only responds to an explicitly created version tag.

See the [documentation index](docs/README.md) for architecture, operations, extraction provenance, and release-compliance guidance.
