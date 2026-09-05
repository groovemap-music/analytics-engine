"""Shared fixtures for insights tests.

The telemetry suites assert on what an in-memory OpenTelemetry provider recorded, so they
must not inherit the ambient OpenTelemetry configuration. `OTEL_SDK_DISABLED=true` in
particular turns every SDK meter into a no-op, which makes those assertions fail with an
empty collection and no error anywhere. Continuous-integration runners set these variables
to keep their own instrumentation quiet, so an unisolated suite passes on a developer's
machine and fails there.

The same applies to the tracing half: `spans` installs an in-memory TracerProvider so the
`insights {computation}` spans can be read back without a collector.
"""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from common import telemetry as _common_telemetry
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from insights import telemetry as _telemetry


if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.trace import ReadableSpan


# Arbitrary non-zero generation — proves endpoints use the value they read.
TEST_CACHE_GENERATION = 4

# Every standard OpenTelemetry variable that changes what the SDK records or exports. The
# tracing bootstrap also *writes* OTEL_TRACES_SAMPLER and OTEL_TRACES_SAMPLER_ARG when they
# are unset, so leaving them in place would let one test's sampler decide another's spans.
OTEL_ENVIRONMENT = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_METRICS_EXEMPLAR_FILTER",
    "OTEL_METRICS_EXPORTER",
    "OTEL_METRIC_EXPORT_INTERVAL",
    "OTEL_PROPAGATORS",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_SDK_DISABLED",
    "OTEL_SERVICE_NAME",
    "OTEL_TRACES_EXPORTER",
    "OTEL_TRACES_SAMPLER",
    "OTEL_TRACES_SAMPLER_ARG",
)


class SpanCollector:
    """An in-memory TracerProvider whose finished spans can be read back by name."""

    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.provider = SdkTracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def spans(self) -> tuple[ReadableSpan, ...]:
        """Return every span that has finished so far, in completion order."""
        return self.exporter.get_finished_spans()

    def names(self) -> list[str]:
        """Return the finished span names, in completion order."""
        return [span.name for span in self.spans()]

    def named(self, name: str) -> list[ReadableSpan]:
        """Return every finished span carrying one name."""
        return [span for span in self.spans() if span.name == name]

    def only(self, name: str) -> ReadableSpan:
        """Return the one span carrying ``name``, asserting there is exactly one."""
        matching = self.named(name)
        assert len(matching) == 1, f"expected exactly one {name!r} span, saw {self.names()}"
        return matching[0]


@pytest.fixture(autouse=True)
def isolated_otel_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run every test against a known-empty OpenTelemetry configuration."""
    for name in OTEL_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[SpanCollector]:
    """Make the domain span helpers record into an in-memory provider, not the global one."""
    collector = SpanCollector()
    monkeypatch.setattr(_common_telemetry, "_tracer_provider", collector.provider)
    assert _common_telemetry.tracer_provider() is collector.provider
    yield collector
    monkeypatch.setattr(_common_telemetry, "_tracer_provider", None)


@pytest.fixture(autouse=True)
def isolated_insights_telemetry() -> Iterator[None]:
    """Give every test a pristine domain-instrument cache and last-success state."""
    _telemetry.reset_instruments()
    yield
    _telemetry.reset_instruments()


@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Mock httpx.AsyncClient for API calls."""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def mock_pg_pool() -> AsyncMock:
    """Mock PostgreSQL pool."""
    mock_cursor = AsyncMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.fetchone = AsyncMock(return_value=None)
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = AsyncMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    # Support conn.transaction() as an async context manager
    mock_tx_cm = AsyncMock()
    mock_tx_cm.__aenter__ = AsyncMock(return_value=None)
    mock_tx_cm.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=mock_tx_cm)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    pool = AsyncMock()
    pool.connection = MagicMock(return_value=mock_conn)
    return pool


@pytest.fixture
def test_client(mock_http_client: AsyncMock, mock_pg_pool: AsyncMock) -> TestClient:
    """Create a test client with mocked dependencies (no cache)."""
    import insights.insights as _module

    _module._http_client = mock_http_client
    _module._pool = mock_pg_pool
    _module._cache = None

    from insights.insights import app

    return TestClient(app)


@pytest.fixture
def mock_cache() -> AsyncMock:
    """Mock InsightsCache."""
    cache = AsyncMock()
    # A real int, not an AsyncMock sentinel: endpoints must thread this exact
    # value from generation() through get() and set() (cache-generation regression).
    cache.generation = AsyncMock(return_value=TEST_CACHE_GENERATION)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.invalidate_all = AsyncMock()
    return cache


@pytest.fixture
def test_client_with_cache(
    mock_http_client: AsyncMock,
    mock_pg_pool: AsyncMock,
    mock_cache: AsyncMock,
) -> TestClient:
    """Create a test client with mocked dependencies and cache enabled."""
    import insights.insights as _module

    _module._http_client = mock_http_client
    _module._pool = mock_pg_pool
    _module._cache = mock_cache

    from insights.insights import app

    return TestClient(app)
