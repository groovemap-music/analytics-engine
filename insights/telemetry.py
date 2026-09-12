"""Domain metrics and spans for scheduled analytics work.

Telemetry failures never affect computations or cache reads.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from threading import RLock
from typing import TYPE_CHECKING, Any

import structlog
from common.telemetry import get_meter, provider_generation
from common.tracing import get_tracer


if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.metrics import CallbackOptions, Observation


logger = structlog.get_logger(__name__)

INSTRUMENTATION_SCOPE = "groovemap.insights"

COMPUTATION_DURATION = "groovemap.insights.computation.duration"
LAST_SUCCESS = "groovemap.insights.last_success"
CACHE = "groovemap.api.cache"

COMPUTATION_SPAN_PREFIX = "insights"
CACHE_NAME = "insights"

_lock = RLock()
_instruments: dict[str, Any] = {}
_instrument_generation = -1

# Process-local by design; PostgreSQL owns the durable computation record.
_last_success: dict[str, float] = {}


def _observe_last_success(_options: CallbackOptions) -> Iterator[Observation]:
    """Yield the last successful-run time for every computation that has completed at least once."""
    from opentelemetry.metrics import Observation  # noqa: PLC0415

    with _lock:
        snapshot = dict(_last_success)
    for computation, unix_time in snapshot.items():
        yield Observation(unix_time, {"computation": computation})


def _build_instruments() -> dict[str, Any]:
    """Create one instrument per domain metric from the current provider."""
    meter = get_meter(INSTRUMENTATION_SCOPE)
    instruments: dict[str, Any] = {
        COMPUTATION_DURATION: meter.create_histogram(
            COMPUTATION_DURATION,
            unit="s",
            description="Duration of one scheduled insight computation.",
        ),
        CACHE: meter.create_counter(
            CACHE,
            description="Insights cache reads by outcome.",
        ),
    }
    instruments[LAST_SUCCESS] = meter.create_observable_gauge(
        LAST_SUCCESS,
        callbacks=[_observe_last_success],
        unit="s",
        description="Unix time of the last successful run of a computation.",
    )
    return instruments


def _instrument(name: str) -> Any:
    """Return one cached instrument, rebuilding the cache when the provider changed."""
    global _instrument_generation

    generation = provider_generation()
    with _lock:
        if _instrument_generation != generation or not _instruments:
            _instruments.clear()
            _instruments.update(_build_instruments())
            _instrument_generation = generation
        return _instruments[name]


def reset_instruments() -> None:
    """Drop the instrument cache and in-memory gauge state. Test seam only."""
    global _instrument_generation

    with _lock:
        _instruments.clear()
        _instrument_generation = -1
        _last_success.clear()


def record_computation(computation: str, duration_s: float, *, success: bool) -> None:
    """Record duration and the latest successful completion time."""
    outcome = "success" if success else "failure"
    try:
        _instrument(COMPUTATION_DURATION).record(duration_s, {"computation": computation, "outcome": outcome})
    except Exception:
        logger.debug("⚠️ Could not record computation duration", computation=computation)

    if success:
        with _lock:
            _last_success[computation] = time.time()
        # Register the observable callback even when the first event is a success.
        try:
            _instrument(LAST_SUCCESS)
        except Exception:
            logger.debug("⚠️ Could not register last-success gauge", computation=computation)


def record_cache_read(*, hit: bool) -> None:
    """Record one insights cache read. A Redis error counts as a miss (see InsightsCache)."""
    outcome = "hit" if hit else "miss"
    try:
        _instrument(CACHE).add(1, {"outcome": outcome, "cache": CACHE_NAME})
    except Exception:
        logger.debug("⚠️ Could not record cache read", outcome=outcome)


def computation_span_name(computation: str) -> str:
    """Return the span name for one computation: ``insights {computation}``."""
    return f"{COMPUTATION_SPAN_PREFIX} {computation}"


def _mark_outcome(span: Any, outcome: str, exc: BaseException | None = None) -> None:
    """Record only the permitted outcome and error type on a span."""
    try:
        span.set_attribute("outcome", outcome)
        if exc is not None:
            from opentelemetry.trace import Status, StatusCode  # noqa: PLC0415

            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
    except Exception:
        logger.debug("⚠️ Could not record the computation span outcome", outcome=outcome)


@contextmanager
def computation_span(computation: str) -> Iterator[Any]:
    """Trace one computation, preserving its exception for scheduler policy."""
    attributes = {"computation": computation}
    try:
        manager = get_tracer(INSTRUMENTATION_SCOPE).start_as_current_span(
            computation_span_name(computation),
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        )
    except Exception:
        logger.debug("⚠️ Could not start the computation span", computation=computation)
        yield None
        return

    with manager as span:
        try:
            yield span
        except BaseException as exc:
            _mark_outcome(span, "failure", exc)
            raise
        else:
            _mark_outcome(span, "success")
