"""Validate the first-party license and synchronized package version."""

import hashlib
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "pyproject.toml").open("rb") as source:
    project = tomllib.load(source)["project"]

version_match = re.search(r'^__version__ = "([^"]+)"$', (ROOT / "insights/__init__.py").read_text(), re.MULTILINE)
assert version_match is not None
assert project["license"] == "AGPL-3.0-only"
assert "License :: OSI Approved :: GNU Affero General Public License v3" in project["classifiers"]
assert all("Proprietary" not in classifier for classifier in project["classifiers"])
assert project["version"] == version_match.group(1)
license_digest = hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest()
assert license_digest == "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"

readme = (ROOT / "README.md").read_text().lower()
commercial_terms = (ROOT / "COMMERCIAL-LICENSING.md").read_text().lower()
notice = (ROOT / "NOTICE").read_text()
contributing = (ROOT / "CONTRIBUTING.md").read_text().lower()
assert "commercial use is permitted under the agpl" in readme
assert "alternative commercial terms may be negotiated" in readme
assert "commercial use is permitted" in commercial_terms
assert "alternative commercial terms" in commercial_terms
assert "MIT License" in notice
assert "PolyForm Noncommercial License 1.0.0" in notice
assert "external contributions are temporarily paused" in contributing
assert "relicensing-capable contributor license agreement" in contributing
