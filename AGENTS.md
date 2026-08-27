# Repository instructions

- Run `just check` before proposing a change.
- Treat `contracts/catalog-api/internal-insights/v1/` as a promoted, immutable producer contract. Update its `source.json`, generated Python binding, and compatibility check together.
- Never add a relative import or Docker build-context dependency on another GrooveMap repository.
- Do not commit credentials, local state, build output, or decrypted secret material.
- Releases are versioned with Commitizen and require an approved `v$version` tag. Migration work must not publish artifacts.
