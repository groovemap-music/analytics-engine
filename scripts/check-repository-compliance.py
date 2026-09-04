"""Validate repository identity, documentation, and automation policy."""

import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTOMATION_REVISION = "2f34a4da5c552bc23c75edd3d8d81be0a4b3271c"
PYTHON_LIBRARIES_REVISION = "41805b62520785f412e8f5d0db90f8d83838ec56"
GIT = shutil.which("git")
if GIT is None:
    raise RuntimeError("git is required to establish the tracked first-party source boundary")


def tracked_files(root: Path = ROOT) -> tuple[Path, ...]:
    """Return the repository's committed-source candidates in Git index order."""
    result = subprocess.run(  # noqa: S603 - resolved executable and fixed arguments
        [GIT, "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(root / os.fsdecode(name) for name in result.stdout.split(b"\0") if name)


def legacy_branding_violations(root: Path = ROOT) -> tuple[Path, ...]:
    """Return tracked first-party text files that retain the legacy project name."""
    legacy_project_name = "discogs" + "ography"
    ignored_scan_directories = {".build", ".git", ".venv", "dist"}
    text_suffixes = {".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
    text_names = {"Dockerfile", "Justfile", "LICENSE", "NOTICE"}
    violations = []
    for path in tracked_files(root):
        relative_path = path.relative_to(root)
        if not path.is_file() or ignored_scan_directories.intersection(relative_path.parts):
            continue
        if path.suffix not in text_suffixes and path.name not in text_names:
            continue
        if legacy_project_name in path.read_text(errors="ignore").lower():
            violations.append(relative_path)
    return tuple(violations)


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
assert ci_target is not None and ci_target.group(1) == AUTOMATION_REVISION
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

release_script = (ROOT / "scripts/release-dry-run.sh").read_text()
assert "--require-hashes" in release_script
assert "--requirements .build/requirements.txt" in release_script
assert "--no-deps" in release_script
assert '"${notices_tmp}/venv/bin/python"' in release_script

workflow_names = {path.name.lower() for path in (ROOT / ".github/workflows").iterdir()}
assert not any("renovate" in name or "claude" in name for name in workflow_names)
assert not any(path.name.lower().startswith("renovate") for path in ROOT.iterdir())

assert not legacy_branding_violations(), legacy_branding_violations()

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

active_docs = "\n".join(path.read_text() for path in sorted((ROOT / "docs").glob("*.md")))
for legacy_service in ("Graphinator", "Tableinator", "Brainzgraphinator", "Brainztableinator", "Dashboard Service"):
    assert legacy_service not in active_docs

source = (ROOT / "insights/insights.py").read_text()
assert 'SERVICE_NAME = "analytics-engine"' in source
assert '"User-Agent": USER_AGENT' in source
assert "Insights service" not in source
