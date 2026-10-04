"""Prove this service's file log sink is bounded, not an unbounded append.

``insights.insights`` and ``insights.embedding_pipeline`` both call
``common.config.setup_logging(SERVICE_NAME, log_file=Path("/logs/<service>.log"))`` on
startup (see the ``mock_setup_logging.assert_called_once_with`` assertions in
``tests/test_insights.py``). ``groovemap-runtime`` itself proves that call
(``log_file=...``) always builds a size-capped ``RotatingFileHandler`` via
``common.log_rotation.build_rotating_file_handler``
(``tests/test_log_rotation.py::test_setup_logging_uses_shared_rotating_handler`` in
python-libraries). This test exercises the real, unmocked ``setup_logging`` with this
service's own log-file path shape end to end, so a future regression that stops routing
through the rotating handler fails here even if the mocked lifespan tests still pass.
"""

import io
import json
import logging
from contextlib import redirect_stdout
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import structlog
from common.config import setup_logging
from common.health_server import HealthServer
from common.log_rotation import DEFAULT_LOG_FILE_BACKUP_COUNT, DEFAULT_LOG_FILE_MAX_BYTES


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _restore_logging_state() -> Iterator[None]:
    """setup_logging mutates process-global logging/structlog state via basicConfig(force=True)."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        yield
    finally:
        for handler in root.handlers:
            if handler not in original_handlers:
                handler.close()
        root.handlers = original_handlers
        root.setLevel(original_level)
        structlog.reset_defaults()


def test_service_log_file_sink_is_a_size_capped_rotating_handler(tmp_path: Path) -> None:
    """The handler backing analytics-engine's /logs/analytics-engine.log file rotates."""
    log_file = tmp_path / "analytics-engine.log"

    setup_logging("analytics-engine", log_file=log_file)

    file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1, "expected exactly one rotating file handler on the root logger"
    handler = file_handlers[0]
    assert Path(handler.baseFilename) == log_file
    assert handler.maxBytes == DEFAULT_LOG_FILE_MAX_BYTES
    assert handler.backupCount == DEFAULT_LOG_FILE_BACKUP_COUNT


def test_service_log_file_sink_honors_deployment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LOG_FILE_MAX_BYTES / LOG_FILE_BACKUP_COUNT retune the bound without a rebuild."""
    monkeypatch.setenv("LOG_FILE_MAX_BYTES", "2048")
    monkeypatch.setenv("LOG_FILE_BACKUP_COUNT", "3")
    log_file = tmp_path / "analytics-engine.log"

    setup_logging("analytics-engine", log_file=log_file)

    (handler,) = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
    assert handler.maxBytes == 2048
    assert handler.backupCount == 3


@pytest.mark.parametrize(("environment", "expected"), [("production", "production"), (None, "development")])
def test_health_server_background_log_keeps_deployment_context_and_event_fields(
    environment: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real health-server thread emits complete JSON with stable process context."""
    if environment is None:
        monkeypatch.delenv("ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("ENVIRONMENT", environment)

    output = io.StringIO()
    with redirect_stdout(output):
        setup_logging("analytics-engine")

    server = HealthServer(0, lambda: {"status": "healthy"})
    try:
        server.start_background()
        assert server.thread is not None
        server.thread.join(timeout=0.05)
    finally:
        server.stop()

    record = next(json.loads(line) for line in output.getvalue().splitlines() if "Health server listening" in json.loads(line)["event"])
    assert record["service"] == "analytics-engine"
    assert record["environment"] == expected
    assert record["logger"] == "common.health_server"
    assert record["level"] == "info"
    assert isinstance(record["timestamp"], str)
    assert isinstance(record["lineno"], int)
