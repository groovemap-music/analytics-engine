"""Validate repository identity, documentation, and automation policy."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")


def workflow_jobs(text: str) -> set[str]:
    """Return top-level job ids from a workflow's jobs section."""
    jobs = text.split("\njobs:\n", 1)[1]
    return set(re.findall(r"(?m)^  ([a-zA-Z0-9_-]+):\s*$", jobs))


ci = (ROOT / ".github/workflows/ci.yml").read_text()
assert re.search(r"(?m)^  pull_request:\s*$", ci)
assert 'cron: "0 1 * * 6"' in ci
assert 'cron: "0 4 * * 1"' in ci
assert workflow_jobs(ci) == {"ci"}
assert "github.actor" not in ci
assert "dependabot" not in ci.lower()
ci_target = re.search(r"reusable-ci\.yml@([^\s]+)", ci)
assert ci_target is not None and FULL_REVISION.fullmatch(ci_target.group(1))

release = (ROOT / ".github/workflows/release.yml").read_text()
release_target = re.search(r"reusable-image-release\.yml@([^\s]+)", release)
assert release_target is not None and FULL_REVISION.fullmatch(release_target.group(1))

workflow_names = {path.name.lower() for path in (ROOT / ".github/workflows").iterdir()}
assert not any("renovate" in name or "claude" in name for name in workflow_names)
assert not any(path.name.lower().startswith("renovate") for path in ROOT.iterdir())

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
