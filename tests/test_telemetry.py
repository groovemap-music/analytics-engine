"""Metric contracts for the analytics-engine domain instruments.

Uses an in-memory OpenTelemetry provider (no network, no collector) to assert the
instrument name, unit, attribute keys, and recorded values the collector and dashboards
depend on — mirroring the technique `common`'s own `tests/test_runtime_metrics.py` uses for
the shared resilience-wrapper instruments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from common import telemetry
from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from insights import telemetry as insights_telemetry


if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.metrics.export import Metric


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
