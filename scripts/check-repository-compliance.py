"""Validate repository identity, documentation, and automation policy."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTOMATION_REVISION = "833cb464507678c38ab78bd4718ce697399463e9"
CI_AUTOMATION_REVISION = "dbf22ed83d486e9484fc8c6385c7c3e5995700e4"
PYTHON_LIBRARIES_REVISION = "5f35c03f68b0dd2c1955c731886c0ad8d8f9de63"
DATABASE_SCHEMA_REVISION = "ba047aed60064662b129807c4a730e2102ccfb97"


def workflow_jobs(text: str) -> set[str]:
    """Return top-level job ids from a workflow's jobs section."""
    jobs = text.split("\njobs:\n", 1)[1]
    return set(re.findall(r"(?m)^  ([a-zA-Z0-9_-]+):\s*$", jobs))


ci = (ROOT / ".github/workflows/ci.yml").read_text()
assert re.search(r"(?m)^  pull_request:\s*$", ci)
assert 'cron: "0 1 * * 6"' in ci
assert 'cron: "0 4 * * 1"' in ci
assert "github.actor" not in ci
assert "dependabot" not in ci.lower()
assert "fallback-command" not in ci
assert workflow_jobs(ci) == {"required"}
ci_target = re.search(r"groovemap-music/automation/\.github/workflows/reusable-ci\.yml@([^\s]+)", ci)
assert ci_target is not None and ci_target.group(1) == CI_AUTOMATION_REVISION
assert "CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}" in ci
assert "secrets: inherit" not in ci

release = (ROOT / ".github/workflows/release.yml").read_text()
assert "attestations: write" in release
release_target = re.search(r"groovemap-music/automation/\.github/workflows/reusable-release\.yml@([^\s]+)", release)
assert release_target is not None and release_target.group(1) == AUTOMATION_REVISION
assert "secrets: inherit" not in release

private_library_markers = (
    "requires-private-library",
    "private-library-client-id",
    "private-library-revision",
    "PRIVATE_LIBRARY_PRIVATE_KEY",
    "GROOVEMAP_CI_APP_CLIENT_ID",
    "GROOVEMAP_CI_APP_PRIVATE_KEY",
)
assert not any(marker in ci for marker in private_library_markers)
assert not any(marker in release for marker in private_library_markers)

pyproject = (ROOT / "pyproject.toml").read_text()
assert 'git = "https://github.com/groovemap-music/python-libraries.git"' in pyproject
assert f'rev = "{PYTHON_LIBRARIES_REVISION}"' in pyproject
runtime_wheel_script = (ROOT / "scripts/prepare-runtime-wheel.sh").read_text()
assert f'expected="{PYTHON_LIBRARIES_REVISION}"' in runtime_wheel_script
# database-schema is a first-party runtime dependency: the lock and the staged wheel share one pin.
assert (
    f'groovemap-database-schema = {{ git = "https://github.com/groovemap-music/database-schema.git", rev = "{DATABASE_SCHEMA_REVISION}" }}'
    in pyproject
)
assert f'schema_expected="{DATABASE_SCHEMA_REVISION}"' in runtime_wheel_script
assert "--no-emit-package groovemap-database-schema" in runtime_wheel_script

release_script = (ROOT / "scripts/release-dry-run.sh").read_text()
assert "--require-hashes" in release_script
assert "--requirements .build/requirements.txt" in release_script
assert "--no-deps" in release_script
assert '"${notices_tmp}/venv/bin/python"' in release_script

workflow_names = {path.name.lower() for path in (ROOT / ".github/workflows").iterdir()}
assert not any("renovate" in name or "claude" in name for name in workflow_names)
assert not any(path.name.lower().startswith("renovate") for path in ROOT.iterdir())

build_image = (ROOT / "scripts/build-image.sh").read_text()
assert 'bash "${repo_root}/scripts/check-image-source.sh"' in build_image

private_planning = (
    ROOT / ".planning",
    ROOT / "docs/superpowers/plans",
    ROOT / "docs/superpowers/specs",
)
assert not any(path.exists() for path in private_planning)
assert (ROOT / "scripts/rehearse-history-sanitization.sh").is_file()

readme = (ROOT / "README.md").read_text()
assert "docs/README.md" in readme
assert "Private, independently" not in readme
assert "**Public-library cutover: complete.**" in readme
assert "public `python-libraries` repository" in readme
assert "require no private-package credentials" in readme
assert "pinned private runtime" not in readme
assert "`GROOVEMAP_RUNTIME_REPO` override" in readme

active_docs = "\n".join(path.read_text() for path in sorted((ROOT / "docs").glob("*.md")))
for legacy_service in ("Graphinator", "Tableinator", "Brainzgraphinator", "Brainztableinator", "Dashboard Service"):
    assert legacy_service not in active_docs

source = (ROOT / "insights/insights.py").read_text()
assert 'SERVICE_NAME = "analytics-engine"' in source
assert '"User-Agent": USER_AGENT' in source
assert "Insights service" not in source
