"""Validate the local analytics-engine image's immutable identity metadata."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> str:
    """Run a required local executable and return stripped stdout."""
    executable = shutil.which(arguments[0])
    if executable is None:
        raise RuntimeError(f"required executable not found: {arguments[0]}")
    return subprocess.run(  # noqa: S603 -- executable is resolved from the operator's PATH
        [executable, *arguments[1:]],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    ).stdout.strip()


image = sys.argv[1] if len(sys.argv) == 2 else "analytics-engine:local"
inspection = cast("list[dict[str, Any]]", json.loads(run("docker", "image", "inspect", image)))
assert len(inspection) == 1
labels = inspection[0]["Config"]["Labels"]
expected = {
    "org.opencontainers.image.licenses": "AGPL-3.0-only",
    "org.opencontainers.image.revision": run("git", "rev-parse", "HEAD"),
    "org.opencontainers.image.source": "https://github.com/groovemap-music/analytics-engine",
    "org.opencontainers.image.title": "analytics-engine",
}
for key, value in expected.items():
    assert labels.get(key) == value, f"{key}: expected {value!r}, got {labels.get(key)!r}"
