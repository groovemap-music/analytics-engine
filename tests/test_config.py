"""The consumer uses the reviewed Valkey URL builder and real client URL parser."""

from pathlib import Path

import pytest
from valkey.asyncio import Valkey

from insights.config import InsightsConfig


@pytest.fixture
def configured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "POSTGRES_HOST": "postgres-fixture",
        "POSTGRES_USERNAME": "fixture",
        "POSTGRES_PASSWORD": "fixture",
        "POSTGRES_DATABASE": "fixture",
        "VALKEY_HOST": "cache-fixture",
        "VALKEY_PORT": "6380",
        "VALKEY_PASSWORD": "",
        "VALKEY_PASSWORD_FILE": "",
    }.items():
        monkeypatch.setenv(name, value)


def test_consumer_builds_valkey_url_accepted_by_actual_client(configured_environment: None) -> None:
    config = InsightsConfig.from_env()
    assert config.valkey_url == "valkey://cache-fixture:6380/0"
    client = Valkey.from_url(config.valkey_url, decode_responses=True)
    connection = client.connection_pool.connection_kwargs
    assert connection["host"] == "cache-fixture"
    assert connection["port"] == 6380
    assert connection["db"] == 0
    assert connection["decode_responses"] is True


def test_file_password_precedence_and_url_escaping(configured_environment: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    password_file = tmp_path / "fixture-password"
    password_file.write_text("fixture:@/ value\n")
    monkeypatch.setenv("VALKEY_PASSWORD", "not-selected")
    monkeypatch.setenv("VALKEY_PASSWORD_FILE", str(password_file))
    config = InsightsConfig.from_env()
    assert config.valkey_url == "valkey://:fixture%3A%40%2F%20value@cache-fixture:6380/0"
    client = Valkey.from_url(config.valkey_url)
    assert client.connection_pool.connection_kwargs["password"] == "fixture:@/ value"


def test_unreadable_password_file_does_not_fall_back(configured_environment: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VALKEY_PASSWORD_FILE", str(tmp_path / "absent"))
    monkeypatch.setenv("VALKEY_PASSWORD", "not-selected")
    with pytest.raises(ValueError, match="Cannot read secret file for VALKEY_PASSWORD"):
        InsightsConfig.from_env()
