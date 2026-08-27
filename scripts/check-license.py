"""Validate the first-party license and synchronized package version."""

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "pyproject.toml").open("rb") as source:
    project = tomllib.load(source)["project"]

version_match = re.search(r'^__version__ = "([^"]+)"$', (ROOT / "insights/__init__.py").read_text(), re.MULTILINE)
assert version_match is not None
assert project["license"] == "PolyForm-Noncommercial-1.0.0"
assert project["version"] == version_match.group(1)
license_text = (ROOT / "LICENSE").read_text()
assert "PolyForm Noncommercial License 1.0.0" in license_text
assert "Required Notice:" in license_text
