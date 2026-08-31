"""Regression tests for the tracked first-party compliance boundary."""

import runpy
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent
GIT = shutil.which("git")
if GIT is None:  # pragma: no cover - Git is a repository test prerequisite
    raise RuntimeError("git is required for repository compliance tests")


def _checker_functions() -> dict[str, object]:
    return runpy.run_path(str(ROOT / "scripts" / "check-repository-compliance.py"))


def _indexed_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run([GIT, "init", "--quiet"], cwd=repository, check=True)  # noqa: S603
    (repository / "README.md").write_text("# GrooveMap\n")
    subprocess.run([GIT, "add", "README.md"], cwd=repository, check=True)  # noqa: S603
    return repository


def test_legacy_scan_ignores_workflow_injected_dependency_checkout(tmp_path: Path) -> None:
    repository = _indexed_repository(tmp_path)
    dependency = repository / "python-libraries"
    dependency.mkdir()
    (dependency / "README.md").write_text("legacy " + "discogs" + "ography dependency documentation\n")

    violations = _checker_functions()["legacy_branding_violations"](repository)

    assert violations == ()


def test_legacy_scan_rejects_tracked_first_party_branding(tmp_path: Path) -> None:
    repository = _indexed_repository(tmp_path)
    (repository / "README.md").write_text("legacy " + "discogs" + "ography documentation\n")

    violations = _checker_functions()["legacy_branding_violations"](repository)

    assert violations == (Path("README.md"),)


def test_tracked_source_boundary_fails_closed_outside_git(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _checker_functions()["tracked_files"](tmp_path)
