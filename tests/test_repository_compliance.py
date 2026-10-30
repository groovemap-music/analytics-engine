"""Regression tests for the tracked first-party compliance boundary."""

import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent


def _checker_functions() -> dict[str, object]:
    return runpy.run_path(str(ROOT / "scripts" / "check-repository-compliance.py"))


def test_current_ci_and_existing_release_pass_actual_checker() -> None:
    _checker_functions()


@pytest.mark.parametrize(
    ("workflow", "old", "replacement"),
    [
        ("ci.yml", "dbf22ed83d486e9484fc8c6385c7c3e5995700e4", "833cb464507678c38ab78bd4718ce697399463e9"),
        ("ci.yml", "dbf22ed83d486e9484fc8c6385c7c3e5995700e4", "a" * 40),
        ("ci.yml", "groovemap-music/automation/", "foreign/automation/"),
        ("release.yml", "833cb464507678c38ab78bd4718ce697399463e9", "dbf22ed83d486e9484fc8c6385c7c3e5995700e4"),
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
    expected = "6d0425f908e01a446b040e2993045e2afe30460e"
    assert expected in original
    fixture = original.replace(expected, replacement)

    def read_fixture(path: Path, *args: object, **kwargs: object) -> str:
        if path == fixture_path:
            return fixture
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_fixture)
    with pytest.raises(AssertionError):
        _checker_functions()


@pytest.mark.parametrize("path", ["scripts/prepare-runtime-wheel.sh", "pyproject.toml"])
def test_actual_checker_rejects_a_schema_pin_that_differs_between_lock_and_staged_wheel(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    fixture_path = ROOT / path
    original_read = Path.read_text
    original = original_read(fixture_path)
    expected = "ba047aed60064662b129807c4a730e2102ccfb97"
    assert expected in original
    fixture = original.replace(expected, "4e9720d838c7da8a6bde139c64a69d781c0f67f0")

    def read_fixture(path: Path, *args: object, **kwargs: object) -> str:
        if path == fixture_path:
            return fixture
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_fixture)
    with pytest.raises(AssertionError):
        _checker_functions()
