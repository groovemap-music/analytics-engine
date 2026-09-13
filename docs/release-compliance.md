# Release compliance

No migration or validation command publishes a package, image, tag, or release. Publication requires an approved annotated version tag and the separately controlled hosted release workflow.

## Validation surfaces

`just check` verifies formatting, linting, promoted contracts, repository policy, secret scans, types, tests and coverage, wheel installation, legal metadata, dependency policy, release artifacts, and the version preview. `just image` verifies the local OCI image. `just audit` performs the network-backed dependency vulnerability scan.

The wheel carries the AGPL license expression and every repository legal file. The image carries the repository URL, exact source revision, version, creation time, and `AGPL-3.0-only` annotation. The release dry-run emits checksums, an SBOM, a complete runtime dependency notice inventory, and provenance containing the exact commit without uploading any artifact.

## Automation

The thin CI and release callers pin the public organization reusable workflows by full commit. CI runs for pushes to `main`, all pull requests, manual dispatches, and the two retained weekly schedules. There is one required CI job graph for all pull requests and no actor-specific skip.

Full validation fetches the public `python-libraries` repository at the immutable revision recorded in `pyproject.toml`. The CI caller passes no first-party repository credential and uses only an explicit `CODECOV_TOKEN`; the release caller passes no inherited secrets. Repository policy rejects the retired GitHub App credential markers and `secrets: inherit`.

## Historical publication note

Before this repository became public, historical implementation plans were preserved in the private `planning-archive` and `.planning/**`, `docs/superpowers/plans/**`, and `docs/superpowers/specs/**` were removed from every published ref. The retained `scripts/rehearse-history-sanitization.sh` documents and tests that one-time boundary against a separate clone; it is not part of ordinary runtime or release operation.

No validation or release recipe rewrites repository history or changes remote visibility.
