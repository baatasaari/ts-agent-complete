"""
ts_agent.observability.signals
==============================
EAMGP (Enterprise Agentic Multi-modal Governance Platform) signal taxonomy.

Every signal carries standard envelope fields (session_id, timestamp_utc,
zone, service_name) plus signal-specific attributes.

Design decisions
----------------
- Uses structlog for structured JSON logging (Cloud Operations compatible).
- Uses OpenTelemetry spans for latency tracing.
- Signals are emitted synchronously — callers are never blocked by telemetry.
- The ``emit`` function is the single egress point; mock it in tests with
  ``pytest-mock`` to assert signals without side effects.
- Listeners can be registered via ``register_listener`` / ``deregister_listener``
  to receive a copy of every emitted payload.  This is the hook used by
  ``SessionStore`` to capture signals without monkey-patching.  Existing
  tests that mock the module-bound ``eamgp.emit`` name are unaffected because
  the mock replaces the reference before ``emit`` dispatches to listeners.

Listener contract
-----------------
A listener is any callable ``(payload: dict[str, Any]) -> None``.
Listeners are called synchronously after structlog and OTEL dispatch.
Exceptions raised by a listener are caught and logged; they never propagate
to the caller of ``emit()``.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_logger = logging.getLogger(__name__)

# ── Tracer setup (override in production with OTLP exporter) ─────────────────
_provider = TracerProvider()
_in_memory_exporter = InMemorySpanExporter()          # captured in tests
_provider.add_span_processor(SimpleSpanProcessor(_in_memory_exporter))
trace.set_tracer_provider(_provider)
_tracer = trace.get_tracer("ts_agent")

# ── Structured logger ─────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
_log = structlog.get_logger("ts_agent.eamgp")

# ── Signal level constants ────────────────────────────────────────────────────
INFO  = "INFO"
WARN  = "WARN"
ERROR = "ERROR"

# Standard envelope keys present on every signal
_ENVELOPE_KEYS = frozenset({
    "signal", "level", "zone", "service_name",
    "session_id", "timestamp_utc",
})

# ── Listener registry (thread-safe) ──────────────────────────────────────────
#
# A listener is any callable ``(payload: dict[str, Any]) -> None``.
# The registry is a plain list protected by a reentrant lock so that
# register/deregister from one thread cannot race with emit from another.
#
# Design: we copy the list under lock before iterating so a listener that
# calls deregister_listener during its own invocation does not cause a
# concurrent-modification error.

ListenerFn = Callable[[dict[str, Any]], None]

_listeners_lock: threading.RLock = threading.RLock()
_listeners: list[ListenerFn] = []


def register_listener(fn: ListenerFn) -> None:
    """
    Register a callable to receive every emitted signal payload.

    The callable is invoked synchronously after structlog and OTEL dispatch.
    Exceptions raised by the callable are caught and logged; they never
    propagate to the original ``emit()`` caller.

    Registering the same callable twice results in it being called twice
    per signal.  Use :func:`deregister_listener` to remove it.

    Parameters
    ----------
    fn:
        ``(payload: dict[str, Any]) -> None``.  The payload dict is a
        shallow copy — mutating it does not affect other listeners.
    """
    with _listeners_lock:
        _listeners.append(fn)


def deregister_listener(fn: ListenerFn) -> None:
    """
    Remove the first occurrence of ``fn`` from the listener registry.

    If ``fn`` was never registered, this is a no-op (no exception raised).
    """
    with _listeners_lock:
        try:
            _listeners.remove(fn)
        except ValueError:
            pass


def clear_listeners() -> None:
    """Remove all registered listeners.  Useful in test teardown."""
    with _listeners_lock:
        _listeners.clear()


def listener_count() -> int:
    """Return the number of currently registered listeners."""
    with _listeners_lock:
        return len(_listeners)


def emit(
    signal: str,
    level: str,
    zone: str,
    *,
    session_id: str = "",
    service_name: str = "ts-agent",
    **attributes: Any,
) -> dict[str, Any]:
    """
    Emit one EAMGP signal.

    Returns the full signal payload (useful for assertions in tests).

    Parameters
    ----------
    signal:       EAMGP signal name, e.g. ``"GRAPH_BUILD_COMPLETE"``.
    level:        ``"INFO"`` | ``"WARN"`` | ``"ERROR"``.
    zone:         Owning zone, e.g. ``"Zone1"``.
    session_id:   Current TS session identifier.
    service_name: Kubernetes service name label.
    **attributes: Signal-specific key/value pairs.
    """
    payload: dict[str, Any] = {
        "signal":        signal,
        "level":         level,
        "zone":          zone,
        "service_name":  service_name,
        "session_id":    session_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **attributes,
    }
    _emit_log(level, payload)
    _emit_span(signal, zone, payload)
    _dispatch_listeners(payload)
    return payload


def _dispatch_listeners(payload: dict[str, Any]) -> None:
    """Fan out payload to all registered listeners.  Exceptions are caught."""
    with _listeners_lock:
        current = list(_listeners)   # snapshot under lock; iterate outside lock
    for fn in current:
        try:
            fn(dict(payload))        # shallow copy — listeners cannot corrupt each other
        except Exception:            # noqa: BLE001
            _logger.exception("EAMGP listener %r raised an exception", fn)


def _emit_log(level: str, payload: dict[str, Any]) -> None:
    fn = {"INFO": _log.info, "WARN": _log.warning, "ERROR": _log.error}.get(
        level, _log.info
    )
    fn(payload["signal"], **{k: v for k, v in payload.items() if k != "signal"})


def _emit_span(signal: str, zone: str, payload: dict[str, Any]) -> None:
    with _tracer.start_as_current_span(signal) as span:
        span.set_attribute("zone", zone)
        for k, v in payload.items():
            if k not in _ENVELOPE_KEYS:
                span.set_attribute(k, str(v) if not isinstance(v, (bool, int, float, str)) else v)


# ── Latency context manager ───────────────────────────────────────────────────

class timed:  # noqa: N801  — lowercase intentional (used as context manager)
    """
    Context manager that measures wall-clock latency and emits a signal.

    Usage::

        with timed("GRAPH_BUILD_COMPLETE", "Zone1", session_id=sid) as ctx:
            result = do_work()
        # signal automatically emitted with latency_ms
    """

    def __init__(self, signal: str, zone: str, **kwargs: Any) -> None:
        self._signal = signal
        self._zone = zone
        self._kwargs = kwargs
        self._start: float = 0.0
        self.latency_ms: int = 0

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.latency_ms = int((time.perf_counter() - self._start) * 1_000)
        emit(
            self._signal,
            ERROR if exc_type else INFO,
            self._zone,
            latency_ms=self.latency_ms,
            error=str(exc_val) if exc_val else None,
            **self._kwargs,
        )
        return False   # never suppress exceptions


def get_emitted_spans() -> list:
    """Return in-memory spans — useful in unit tests."""
    return _in_memory_exporter.get_finished_spans()


def clear_spans() -> None:
    """Reset the in-memory exporter between tests."""
    _in_memory_exporter.clear()
