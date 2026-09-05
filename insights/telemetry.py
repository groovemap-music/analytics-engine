"""GrooveMap OpenTelemetry domain instruments and spans for analytics-engine.

Registers the ``groovemap.insights.*`` and ``groovemap.api.cache`` instruments once
against the meter ``common.telemetry.get_meter`` returns, and exposes small recording
helpers called from the scheduler's computation loop (:mod:`insights.computations`) and
the cache-aside read path (:mod:`insights.cache`). :func:`computation_span` is the tracing
half: the ``insights {computation}`` root span the same loop opens around each computation.

Every recorder swallows its own errors — telemetry must never turn a working computation
or a working cache read into a failure — and every instrument is a local no-op until the
``otel`` extra is installed and ``OTEL_EXPORTER_OTLP_ENDPOINT`` is configured (see
``common.telemetry``). Postgres ``db.client.operation.duration`` metrics and the ``session
postgresql`` client spans are emitted automatically by the shared ``AsyncPostgreSQLPool``
wrapper, and the process view and the event-loop lag histogram by ``setup_telemetry`` and
``start_event_loop_monitor``; none of them need code here.

Instruments are built lazily from a meter obtained once at import — the OpenTelemetry API
hands back a proxy meter before ``setup_telemetry`` runs and transparently upgrades it once
the real provider is installed, so the instruments created here keep recording correctly
regardless of import order. The cache is still keyed by ``provider_generation()`` so a test
that swaps the provider directly (bypassing the API's own proxy mechanism) rebuilds it too.
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

# The domain root span the GrooveMap conventions give this service: `insights {computation}`,
# one per scheduled computation, over the same closed computation set the metrics use.
COMPUTATION_SPAN_PREFIX = "insights"

# The `cache` attribute value on the shared groovemap.api.cache instrument: this service's
# own Redis-backed cache, distinct from the API service's insights:data-completeness cache.
CACHE_NAME = "insights"

_lock = RLock()
_instruments: dict[str, Any] = {}
_instrument_generation = -1

# computation name -> unix time of its last successful run. In-memory only, by design: the
# observable gauge below reports this process's view, not a durable record.
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
    """Record one scheduled computation's duration and, on success, its completion time.

    ``computation`` matches the names ``run_all_computations`` already uses (``artist_centrality``,
    ``genre_trends``, ...) — a closed, low-cardinality set.
    """
    outcome = "success" if success else "failure"
    try:
        _instrument(COMPUTATION_DURATION).record(duration_s, {"computation": computation, "outcome": outcome})
    except Exception:
        logger.debug("⚠️ Could not record computation duration", computation=computation)

    if success:
        with _lock:
            _last_success[computation] = time.time()
        # Ensure the observable gauge instrument exists so a computation that never fails
        # still gets its callback registered on the first successful run.
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
    """Record the outcome on a span, failing it with ``error.type`` only when one is given.

    The conventions allow a status and an ``error.type``; never a message, a stack trace, or a
    span event carrying a payload. Swallows its own errors like every recorder here, including
    the ``None`` a tracer that could not start a span leaves behind.
    """
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
    """Open the root span for one scheduled computation: ``insights {computation}``.

    ``computation`` is the same closed, low-cardinality set :func:`record_computation` uses, so
    the span name stays low-cardinality and a computation's span and its duration histogram
    carry the identical attribute value. The span kind is left at the default ``INTERNAL`` the
    conventions give domain root spans, which also keeps this working with no ``opentelemetry``
    package installed at all.

    ``outcome`` is written on the way out — ``success``, or ``failure`` with an ``error.type``
    attribute and status ``ERROR`` when the body raised. The exception is always re-raised: the
    caller decides whether a failed computation stops the cycle. Yields ``None`` only when the
    tracer could not be started, and never raises on its own account.
    """
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
