"""The `analytics-engine` dispatcher (gm-analytics-engine-ej6s)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from insights import cli, embedding_pipeline, insights, similar_artists_job


def test_no_args_starts_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(insights, "main", lambda: calls.append("serve"))
    cli.main([])
    assert calls == ["serve"]


def test_similar_artists_forwards_argv_verbatim_and_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake(argv: list[str] | None = None, **_: Any) -> None:
        seen.append(list(argv or []))
        raise SystemExit(75)

    monkeypatch.setattr(similar_artists_job, "main", fake)
    argv = ["--spool-dir", "/spool", "--max-runtime", "19800", "--threads", "8"]
    with pytest.raises(SystemExit) as exc:
        cli.main(["similar-artists", *argv])
    assert exc.value.code == 75
    assert seen == [argv]


def test_similar_artists_status_line_and_exit_match_script(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Through the real job `main` with an injected run: same JSON status line and exit code as the script."""
    status = similar_artists_job.failed_status(RuntimeError("boom"), 1.0)

    async def run(*_: Any) -> Any:
        return status

    real = similar_artists_job.main
    monkeypatch.setattr(similar_artists_job, "main", lambda argv=None: real(argv, run=run))
    monkeypatch.setattr(similar_artists_job, "setup_logging", lambda *a, **k: None)
    argv = ["--spool-dir", str(tmp_path / "spool")]
    with pytest.raises(SystemExit) as via_cli:
        cli.main(["similar-artists", *argv])
    line_cli = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    with pytest.raises(SystemExit) as direct:
        similar_artists_job.main(argv)
    line_direct = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert via_cli.value.code == direct.value.code
    line_cli.pop("elapsed_s", None)
    line_direct.pop("elapsed_s", None)
    assert line_cli == line_direct


def test_embeddings_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(embedding_pipeline, "main", lambda: calls.append("embed"))
    cli.main(["embeddings"])
    assert calls == ["embed"]


def test_embeddings_rejects_extra_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_pipeline, "main", lambda: pytest.fail("must not run"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["embeddings", "--x"])
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["bogus"], ["bogus", "--help"], ["--version"]])
def test_unknown_subcommand_exits_2_without_serving(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    monkeypatch.setattr(insights, "main", lambda: pytest.fail("must not start the server"))
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert "usage: analytics-engine" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_exits_0(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flag: str) -> None:
    monkeypatch.setattr(insights, "main", lambda: pytest.fail("must not start the server"))
    with pytest.raises(SystemExit) as exc:
        cli.main([flag])
    assert exc.value.code == 0
    assert "usage: analytics-engine" in capsys.readouterr().out


def test_argv_defaults_to_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["analytics-engine", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0


def test_script_names_are_kept() -> None:
    scripts = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == {
        "analytics-engine": "insights.cli:main",
        "analytics-engine-embeddings": "insights.embedding_pipeline:main",
        "analytics-engine-similar-artists": "insights.similar_artists_job:main",
    }
