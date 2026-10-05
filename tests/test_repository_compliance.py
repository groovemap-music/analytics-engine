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


def test_current_ci_and_existing_release_pass_actual_checker() -> None:
    _checker_functions()


@pytest.mark.parametrize(
    ("workflow", "old", "replacement"),
    [
        ("ci.yml", "2f890111657d9f3e6f55d8bd5a5e7b8f9ca97b26", "833cb464507678c38ab78bd4718ce697399463e9"),
        ("ci.yml", "2f890111657d9f3e6f55d8bd5a5e7b8f9ca97b26", "a" * 40),
        ("ci.yml", "groovemap-music/automation/", "foreign/automation/"),
        ("release.yml", "833cb464507678c38ab78bd4718ce697399463e9", "2f890111657d9f3e6f55d8bd5a5e7b8f9ca97b26"),
        ("release.yml", "833cb464507678c38ab78bd4718ce697399463e9", "a" * 40),
        ("release.yml", "groovemap-music/automation/", "foreign/automation/"),
    ],
)
def test_actual_checker_rejects_stale_or_foreign_workflow_pins(monkeypatch: pytest.MonkeyPatch, workflow: str, old: str, replacement: str) -> None:
    fixture_path = ROOT / ".github/workflows" / workflow
    original_read = Path.read_text
    original = original_read(fixture_path)
    assert old in original
    fixture = original.replace(old, replacement)

    def read_fixture(path: Path, *args: object, **kwargs: object) -> str:
        if path == fixture_path:
            return fixture
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_fixture)
    with pytest.raises(AssertionError):
        _checker_functions()


@pytest.mark.parametrize("replacement", ["77e5b53f469ddcf0b96fc559306aeefa4a4e7008", "a" * 40])
def test_actual_checker_rejects_stale_or_foreign_runtime_wheel_pin(monkeypatch: pytest.MonkeyPatch, replacement: str) -> None:
    fixture_path = ROOT / "scripts/prepare-runtime-wheel.sh"
    original_read = Path.read_text
    original = original_read(fixture_path)
    expected = "6c3802035e9c973c6598dadbd3e4377daee613d4"
    assert expected in original
    fixture = original.replace(expected, replacement)

    def read_fixture(path: Path, *args: object, **kwargs: object) -> str:
        if path == fixture_path:
            return fixture
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_fixture)
    with pytest.raises(AssertionError):
        _checker_functions()
