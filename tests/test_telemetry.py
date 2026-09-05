"""Metric and span contracts for the analytics-engine domain telemetry.

Uses in-memory OpenTelemetry providers (no network, no collector) to assert the instrument
name, unit, attribute keys, and recorded values the collector and dashboards depend on, and
the name, kind, and attributes of the `insights {computation}` root span the spanmetrics
connector turns into per-computation call and duration series — mirroring the technique
`common`'s own `tests/test_runtime_metrics.py` and `tests/test_tracing_wrappers.py` use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from common import telemetry
from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    MetricExporter,
    MetricExportResult,
)
from opentelemetry.trace import NoOpTracerProvider, SpanKind, StatusCode

from insights import telemetry as insights_telemetry


if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.metrics.export import Metric

    from tests.conftest import SpanCollector


SPAN_EXPORTER_IMPORT_PATH = "opentelemetry.exporter.otlp.proto.http.trace_exporter"
METRIC_EXPORTER_IMPORT_PATH = "opentelemetry.exporter.otlp.proto.http.metric_exporter"
ENDPOINT = "http://otel-collector:4318"


class Collector:
    """An in-memory provider plus helpers for reading what the recorders wrote."""

    def __init__(self) -> None:
        self.reader = InMemoryMetricReader()
        self.provider = SdkMeterProvider(metric_readers=[self.reader])

    def metrics(self) -> dict[str, Metric]:
        """Collect once and return every recorded metric by name."""
        data = self.reader.get_metrics_data()
        if data is None:
            return {}
        return {
            metric.name: metric
            for resource_metrics in data.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
        }

    def points(self, name: str) -> list[Any]:
        """Return the data points recorded for one metric name."""
        metric = self.metrics().get(name)
        return [] if metric is None else list(metric.data.data_points)

    def attributes(self, name: str) -> list[dict[str, Any]]:
        """Return the attribute dicts recorded for one metric name."""
        return [dict(point.attributes) for point in self.points(name)]


@pytest.fixture
def collector(monkeypatch: pytest.MonkeyPatch) -> Iterator[Collector]:
    """Install an in-memory provider and make the recorders build instruments against it."""
    active = Collector()
    monkeypatch.setattr(telemetry, "_provider", active.provider)
    monkeypatch.setattr(telemetry, "_generation", telemetry.provider_generation() + 1)
    insights_telemetry.reset_instruments()
    assert telemetry._active_provider() is active.provider
    yield active
    monkeypatch.setattr(telemetry, "_provider", None)
    insights_telemetry.reset_instruments()


class TestComputationDuration:
    def test_is_a_seconds_histogram_recording_the_computation_and_success_outcome(self, collector: Collector) -> None:
        insights_telemetry.record_computation("artist_centrality", 1.5, success=True)

        metric = collector.metrics()[insights_telemetry.COMPUTATION_DURATION]
        assert metric.unit == "s"
        points = collector.points(insights_telemetry.COMPUTATION_DURATION)
        assert len(points) == 1
        assert dict(points[0].attributes) == {"computation": "artist_centrality", "outcome": "success"}
        assert points[0].sum == pytest.approx(1.5)
        assert points[0].count == 1

    def test_records_the_failure_outcome(self, collector: Collector) -> None:
        insights_telemetry.record_computation("data_completeness", 0.25, success=False)

        assert collector.attributes(insights_telemetry.COMPUTATION_DURATION) == [{"computation": "data_completeness", "outcome": "failure"}]

    def test_a_second_computation_gets_its_own_attribute_set(self, collector: Collector) -> None:
        insights_telemetry.record_computation("genre_trends", 0.1, success=True)
        insights_telemetry.record_computation("release_rarity", 0.2, success=False)

        attrs = collector.attributes(insights_telemetry.COMPUTATION_DURATION)
        assert {"computation": "genre_trends", "outcome": "success"} in attrs
        assert {"computation": "release_rarity", "outcome": "failure"} in attrs


class TestLastSuccessGauge:
    def test_has_no_points_before_any_success(self, collector: Collector) -> None:
        assert collector.points(insights_telemetry.LAST_SUCCESS) == []

    def test_a_failed_run_reports_no_point(self, collector: Collector) -> None:
        insights_telemetry.record_computation("label_longevity", 0.3, success=False)

        assert collector.points(insights_telemetry.LAST_SUCCESS) == []

    def test_reports_the_unix_time_of_the_successful_run(self, collector: Collector, monkeypatch: pytest.MonkeyPatch) -> None:
        fixed_time = 1_725_000_000.0
        monkeypatch.setattr(insights_telemetry.time, "time", lambda: fixed_time)

        insights_telemetry.record_computation("anniversaries", 0.4, success=True)

        points = collector.points(insights_telemetry.LAST_SUCCESS)
        assert len(points) == 1
        assert dict(points[0].attributes) == {"computation": "anniversaries"}
        assert points[0].value == fixed_time

    def test_a_later_success_overwrites_the_earlier_one_for_the_same_computation(self, collector: Collector, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(insights_telemetry.time, "time", lambda: 1_000.0)
        insights_telemetry.record_computation("artist_centrality", 0.1, success=True)

        monkeypatch.setattr(insights_telemetry.time, "time", lambda: 2_000.0)
        insights_telemetry.record_computation("artist_centrality", 0.1, success=True)

        points = collector.points(insights_telemetry.LAST_SUCCESS)
        assert len(points) == 1
        assert points[0].value == 2_000.0

    def test_reports_one_point_per_live_computation(self, collector: Collector) -> None:
        insights_telemetry.record_computation("artist_centrality", 0.1, success=True)
        insights_telemetry.record_computation("genre_trends", 0.1, success=True)

        attrs = collector.attributes(insights_telemetry.LAST_SUCCESS)
        assert {"computation": "artist_centrality"} in attrs
        assert {"computation": "genre_trends"} in attrs


class TestCacheReads:
    def test_records_a_hit(self, collector: Collector) -> None:
        insights_telemetry.record_cache_read(hit=True)

        points = collector.points(insights_telemetry.CACHE)
        assert len(points) == 1
        assert dict(points[0].attributes) == {"outcome": "hit", "cache": "insights"}
        assert points[0].value == 1

    def test_records_a_miss(self, collector: Collector) -> None:
        insights_telemetry.record_cache_read(hit=False)

        assert collector.attributes(insights_telemetry.CACHE) == [{"outcome": "miss", "cache": "insights"}]

    def test_counts_repeated_reads(self, collector: Collector) -> None:
        insights_telemetry.record_cache_read(hit=True)
        insights_telemetry.record_cache_read(hit=True)
        insights_telemetry.record_cache_read(hit=False)

        points = {dict(p.attributes)["outcome"]: p.value for p in collector.points(insights_telemetry.CACHE)}
        assert points == {"hit": 2, "miss": 1}


class TestResetInstruments:
    def test_clears_cached_instruments_and_last_success_state(self, collector: Collector) -> None:
        insights_telemetry.record_computation("artist_centrality", 0.1, success=True)
        assert insights_telemetry._last_success

        insights_telemetry.reset_instruments()

        assert insights_telemetry._last_success == {}
        assert insights_telemetry._instruments == {}
        assert insights_telemetry._instrument_generation == -1


# ============================================================
# Span contracts
# ============================================================


@pytest.fixture
def tracing_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Install the no-op TracerProvider — the state a service without an endpoint runs in."""
    monkeypatch.setattr(telemetry, "_tracer_provider", NoOpTracerProvider())
    yield
    monkeypatch.setattr(telemetry, "_tracer_provider", None)


class TestComputationSpan:
    def test_names_the_span_after_the_computation_and_marks_a_successful_outcome(self, spans: SpanCollector) -> None:
        with insights_telemetry.computation_span("artist_centrality"):
            pass

        span = spans.only("insights artist_centrality")
        assert span.kind is SpanKind.INTERNAL
        assert dict(span.attributes) == {"computation": "artist_centrality", "outcome": "success"}
        assert span.status.status_code is not StatusCode.ERROR

    def test_is_the_root_of_its_own_trace(self, spans: SpanCollector) -> None:
        with insights_telemetry.computation_span("genre_trends"):
            pass

        assert spans.only("insights genre_trends").parent is None

    def test_a_failure_sets_error_status_and_error_type_only(self, spans: SpanCollector) -> None:
        with pytest.raises(RuntimeError), insights_telemetry.computation_span("data_completeness"):
            raise RuntimeError("boom")

        span = spans.only("insights data_completeness")
        assert dict(span.attributes) == {
            "computation": "data_completeness",
            "outcome": "failure",
            "error.type": "RuntimeError",
        }
        assert span.status.status_code is StatusCode.ERROR
        # The message is deliberately absent: the conventions allow a status and an error.type.
        assert span.status.description is None
        assert span.events == ()

    def test_re_raises_so_the_caller_decides_what_a_failure_costs(self, spans: SpanCollector) -> None:
        with pytest.raises(ValueError, match="bad"), insights_telemetry.computation_span("release_rarity"):
            raise ValueError("bad")

        assert spans.names() == ["insights release_rarity"]

    def test_yields_the_live_span_so_a_caller_can_read_its_context(self, spans: SpanCollector) -> None:
        with insights_telemetry.computation_span("label_longevity") as span:
            assert span is not None
            assert span.is_recording()

    def test_one_span_per_computation(self, spans: SpanCollector) -> None:
        for computation in ("artist_centrality", "genre_trends", "anniversaries"):
            with insights_telemetry.computation_span(computation):
                pass

        assert spans.names() == [
            "insights artist_centrality",
            "insights genre_trends",
            "insights anniversaries",
        ]

    def test_span_name_is_built_from_the_documented_prefix(self) -> None:
        assert insights_telemetry.computation_span_name("community_enrichment") == "insights community_enrichment"
        assert insights_telemetry.COMPUTATION_SPAN_PREFIX == "insights"

    @pytest.mark.usefixtures("tracing_off")
    def test_records_nothing_and_still_runs_the_body_when_tracing_is_off(self) -> None:
        ran = False

        with insights_telemetry.computation_span("artist_centrality") as span:
            ran = True
            assert not span.is_recording()

        assert ran

    @pytest.mark.usefixtures("tracing_off")
    def test_still_re_raises_when_tracing_is_off(self) -> None:
        with pytest.raises(RuntimeError), insights_telemetry.computation_span("genre_trends"):
            raise RuntimeError("boom")

    def test_a_span_that_rejects_its_outcome_does_not_fail_the_computation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Telemetry must never turn a working computation into a failure."""

        class ExplodingSpan:
            def __enter__(self) -> ExplodingSpan:
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def set_attribute(self, key: str, value: object) -> None:
                raise RuntimeError("no attributes")

        class ExplodingTracer:
            def start_as_current_span(self, *_args: object, **_kwargs: object) -> ExplodingSpan:
                return ExplodingSpan()

        monkeypatch.setattr(insights_telemetry, "get_tracer", lambda _name: ExplodingTracer())
        ran = False

        with insights_telemetry.computation_span("anniversaries"):
            ran = True

        assert ran

    def test_a_span_that_rejects_a_failure_outcome_still_re_raises_the_original_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class ExplodingSpan:
            def __enter__(self) -> ExplodingSpan:
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def set_attribute(self, key: str, value: object) -> None:
                raise RuntimeError("no attributes")

        class ExplodingTracer:
            def start_as_current_span(self, *_args: object, **_kwargs: object) -> ExplodingSpan:
                return ExplodingSpan()

        monkeypatch.setattr(insights_telemetry, "get_tracer", lambda _name: ExplodingTracer())

        with pytest.raises(ValueError, match="original"), insights_telemetry.computation_span("anniversaries"):
            raise ValueError("original")

    def test_a_tracer_that_cannot_start_a_span_does_not_fail_the_computation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def exploding_tracer(_name: str) -> Any:
            raise RuntimeError("no tracer")

        monkeypatch.setattr(insights_telemetry, "get_tracer", exploding_tracer)
        ran = False

        with insights_telemetry.computation_span("release_rarity") as span:
            ran = True
            assert span is None

        assert ran


class CapturingMetricExporter(MetricExporter):
    """Stands in for the OTLP metric exporter so this suite never opens a socket."""

    def __init__(self, **_kwargs: Any) -> None:
        super().__init__(preferred_temporality={}, preferred_aggregation={})
        self.exported: list[str] = []

    def export(self, metrics_data: Any, timeout_millis: float = 10_000, **_kwargs: Any) -> MetricExportResult:
        self.exported.extend(
            metric.name
            for resource_metrics in metrics_data.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
        )
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        return True

    def shutdown(self, timeout_millis: float = 30_000, **_kwargs: Any) -> None:
        """Discard the shutdown."""


class TestSignalsAreConfiguredIndependently:
    """The env-var contract this service is deployed under, exercised through setup_telemetry."""

    @pytest.fixture(autouse=True)
    def pristine_providers(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """Give each test unconfigured provider handles and leave none behind."""
        handles = ("_provider", "_sdk_provider", "_tracer_provider", "_sdk_tracer_provider")
        for name in handles:
            monkeypatch.setattr(telemetry, name, None)
        yield
        for name in handles:
            monkeypatch.setattr(telemetry, name, None)

    @pytest.fixture
    def captured_metrics(self, monkeypatch: pytest.MonkeyPatch) -> list[CapturingMetricExporter]:
        """Replace the OTLP metric exporter the bootstrap constructs with a capturing one."""
        built: list[CapturingMetricExporter] = []

        def factory(**kwargs: Any) -> CapturingMetricExporter:
            exporter = CapturingMetricExporter(**kwargs)
            built.append(exporter)
            return exporter

        monkeypatch.setattr(f"{METRIC_EXPORTER_IMPORT_PATH}.OTLPMetricExporter", factory)
        return built

    @pytest.fixture
    def built_span_exporters(self, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        """Record every OTLP span exporter the bootstrap builds, without opening a socket."""
        built: list[Any] = []

        def factory(**_kwargs: Any) -> Any:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

            exporter = InMemorySpanExporter()
            built.append(exporter)
            return exporter

        monkeypatch.setattr(f"{SPAN_EXPORTER_IMPORT_PATH}.OTLPSpanExporter", factory)
        return built

    def test_traces_off_with_an_endpoint_set_leaves_metrics_flowing_and_creates_no_spans(
        self,
        monkeypatch: pytest.MonkeyPatch,
        captured_metrics: list[CapturingMetricExporter],
        built_span_exporters: list[Any],
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", ENDPOINT)
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")

        provider = telemetry.setup_telemetry("analytics-engine")
        try:
            assert isinstance(provider, SdkMeterProvider)
            assert isinstance(telemetry.tracer_provider(), NoOpTracerProvider)
            # No span exporter is even constructed, so nothing can reach the collector.
            assert built_span_exporters == []

            insights_telemetry.record_computation("artist_centrality", 1.5, success=True)
            with insights_telemetry.computation_span("artist_centrality") as span:
                assert not span.is_recording()

            provider.force_flush()
        finally:
            provider.shutdown()

        assert len(captured_metrics) == 1
        assert insights_telemetry.COMPUTATION_DURATION in captured_metrics[0].exported

    def test_no_endpoint_leaves_both_signals_no_ops(self) -> None:
        provider = telemetry.setup_telemetry("analytics-engine")

        assert not isinstance(provider, SdkMeterProvider)
        assert isinstance(telemetry.tracer_provider(), NoOpTracerProvider)

        insights_telemetry.record_computation("genre_trends", 0.1, success=True)
        with insights_telemetry.computation_span("genre_trends") as span:
            assert not span.is_recording()
