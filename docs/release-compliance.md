# Release compliance

No migration or validation command publishes a package, image, tag, or release. Publication requires an approved annotated version tag and the separately controlled hosted release workflow.

## Validation surfaces

`just check` verifies formatting, linting, promoted contracts, repository policy, secret scans, types, tests and coverage, wheel installation, legal metadata, dependency policy, release artifacts, and the version preview. `just image` verifies the local OCI image. `just audit` performs the network-backed dependency vulnerability scan.

The wheel carries the AGPL license expression and every repository legal file. The image carries the repository URL, exact source revision, version, creation time, and `AGPL-3.0-only` annotation. The release dry-run emits checksums, an SBOM, a complete runtime dependency notice inventory, and provenance containing the exact commit without uploading any artifact.

## Automation

The thin CI and release callers pin the public organization reusable workflows by full commit. CI runs for pushes to `main`, ordinary and Dependabot-authored pull requests, manual dispatches, and the two weekly full/security schedules inherited from the monolith. There is one required CI job graph for all pull requests and no actor-specific skip.

Full validation currently requires read access to the pinned `python-libraries` revision. The GitHub App client ID must be available as an Actions variable, and its private key must be configured in both Actions secrets and Dependabot secrets. Without the Dependabot copy, GitHub withholds the Actions secret from Dependabot-authored pull requests and the shared workflow can only run its credential-free fallback; that is not release-compliant parity.

## Historical planning privacy

Historical implementation plans are preserved in the private `planning-archive` before they are removed from this repository. Deleting them from the current tree is not sufficient: before publication, a backed-up separate clone must remove `.planning/**`, `docs/superpowers/plans/**`, and `docs/superpowers/specs/**` from every ref. The rehearsal must retain an old-to-new commit map and pass complete reachable-object and secret scans.

The separate filtered clone is the only permissible rewrite target. Replacing the private remote from that clone and making the repository public are distinct operator-approved actions; neither is performed by repository validation.
