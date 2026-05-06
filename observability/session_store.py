"""
ts_agent.observability.session_store
=====================================
In-memory session store for the regulatory visualiser.

Architecture
------------
``SessionStore`` registers a single listener with ``signals.emit()`` via the
``register_listener`` API added in v1.4.  Every emitted payload is routed by
``session_id`` into a ``SessionRecord``.

``SessionIndex`` maintains the ``party_ref → [session_id, ...]`` mapping.
It is populated when the store receives a signal that carries ``party_ref``
(``SESSION_RESUMED``, ``SESSION_DISCONNECTED``) or is updated explicitly by
the pipeline bootstrap via ``SessionIndex.add(party_ref, session_id)``.

Thread safety
-------------
All public methods acquire ``_lock`` (``threading.RLock``) before reading or
writing shared state.  The listener callback itself holds the lock only for the
minimal dict-append operation; the rest of the routing logic runs without
holding the lock to avoid potential deadlocks if ``emit()`` is called
re-entrantly from inside a listener.

Codex review notes
------------------
- No import from ``tests/``.
- No import from ``visualiser/``.
- ``SessionRecord`` is a plain dataclass — serialisable to/from plain dicts
  for future persistence (e.g. Cloud Spanner or BigQuery).
- ``SignalEvent``, ``ConversationTurn``, ``PredictionSnapshot`` are all
  frozen or value-like to prevent accidental mutation by the visualiser layer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ts_agent.observability import signals as eamgp


# ──────────────────────────────────────────────────────────────────────────────
# Value types (one per data kind)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalEvent:
    """
    One captured EAMGP signal payload, verbatim from ``emit()``.

    ``elapsed_ms`` is the number of milliseconds since the session's first
    signal — computed at append time.  This lets the visualiser draw a
    timeline without depending on wall-clock timestamps.
    """
    signal:        str
    level:         str
    zone:          str
    session_id:    str
    timestamp_utc: str
    elapsed_ms:    float
    attributes:    dict[str, Any]   # all fields except the envelope keys


@dataclass(frozen=True)
class ConversationTurn:
    """One question–answer exchange in Zone 2."""
    turn_number:    int
    char_id:        str
    question_text:  str             # human-readable; filled by DataAdapter
    value_hash:     str | None      # SHA-256 of consumer answer (never raw value)
    source:         str             # CONSUMER_INPUT | BANK_DATA
    completeness:   float           # 0–1 graph completeness after this answer
    elapsed_ms:     float


@dataclass(frozen=True)
class PredictionSnapshot:
    """One ML prediction turn captured from Zone 1.5 signals."""
    turn:               int
    top_segment_id:     str | None
    top_confidence:     float
    model_version:      str
    model_algorithm:    str
    known_trait_count:  int
    disposition:        str         # ACTIVE | UNDECIDABLE | FAILED | LOW_CONFIDENCE
    shap_features_json: str         # raw JSON string from signal attribute
    elapsed_ms:         float


@dataclass
class SessionRecord:
    """
    Complete audit record for one TS session.

    Populated incrementally as signals are received.  All lists are
    append-only after the record is created.

    ``is_complete`` is set to True when any terminal signal is received
    (``AUDIT_WRITE_CONFIRMED``, ``AUDIT_WRITE_FAILED``, ``SEGMENT_NO_MATCH``
    with no subsequent signals, or ``SESSION_EXPIRED``).
    """
    session_id:       str
    party_ref:        str = ""
    intent_id:        str = ""
    situation_id:     str = ""
    channel:          str = ""

    # Ordered signal log — source of truth for the trace waterfall
    signals: list[SignalEvent] = field(default_factory=list)

    # Zone 2 conversation
    conversation: list[ConversationTurn] = field(default_factory=list)

    # Zone 1.5 ML prediction chain
    prediction_chain: list[PredictionSnapshot] = field(default_factory=list)

    # Zone 1 trait state at completion (populated from GRAPH_BUILD_COMPLETE)
    known_trait_count:    int = 0
    missing_trait_count:  int = 0
    excluded_trait_count: int = 0

    # Zone 2 outcome
    matched_segment_id:  str   = ""
    segment_confidence:  float = 0.0
    gap_fill_turns:      int   = 0
    fill_strategy:       str   = ""

    # Zone 3 rule evaluation (from suggestion engine signals)
    rule_evaluations: list[dict[str, Any]] = field(default_factory=list)
    candidates_evaluated: int = 0
    validated_candidates: int = 0

    # Zone 4 outcome
    gate_disposition:         str  = ""
    audit_confirmed:          bool = False
    communication_text_hash:  str  = ""
    consumer_explanation:     str  = ""

    # Session lifecycle
    is_complete:   bool = False
    has_error:     bool = False
    error_signals: list[str] = field(default_factory=list)

    # Timing
    started_at:    str  = ""    # ISO8601 timestamp of first signal
    completed_at:  str  = ""    # ISO8601 timestamp of terminal signal
    total_ms:      float = 0.0

    def duration_seconds(self) -> float:
        return self.total_ms / 1000.0

    def signal_count(self) -> int:
        return len(self.signals)

    def error_count(self) -> int:
        return len(self.error_signals)


# ──────────────────────────────────────────────────────────────────────────────
# Session index: party_ref → session_ids
# ──────────────────────────────────────────────────────────────────────────────

class SessionIndex:
    """
    Thread-safe mapping from ``party_ref`` to a list of ``session_id`` values.

    Populated by ``SessionStore`` when a party-aware signal is received,
    or explicitly via ``add()``.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._index: dict[str, list[str]] = {}

    def add(self, party_ref: str, session_id: str) -> None:
        """Register a session under a party reference."""
        if not party_ref or not session_id:
            return
        with self._lock:
            if party_ref not in self._index:
                self._index[party_ref] = []
            if session_id not in self._index[party_ref]:
                self._index[party_ref].append(session_id)

    def sessions_for(self, party_ref: str) -> list[str]:
        """Return all session_ids for a party, oldest first."""
        with self._lock:
            return list(self._index.get(party_ref, []))

    def all_party_refs(self) -> list[str]:
        """Return all known party references, sorted."""
        with self._lock:
            return sorted(self._index.keys())

    def party_for_session(self, session_id: str) -> str | None:
        """Reverse lookup: session_id → party_ref."""
        with self._lock:
            for party, sessions in self._index.items():
                if session_id in sessions:
                    return party
        return None

    def total_sessions(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._index.values())


# ──────────────────────────────────────────────────────────────────────────────
# Signal envelope keys (excluded from SignalEvent.attributes)
# ──────────────────────────────────────────────────────────────────────────────

_ENVELOPE_KEYS = frozenset({
    "signal", "level", "zone", "session_id",
    "service_name", "timestamp_utc",
})

# Signals that carry party_ref so we can update the index
_PARTY_REF_SIGNALS = frozenset({
    "SESSION_RESUMED", "SESSION_DISCONNECTED",
    "SESSION_PAUSED", "SESSION_EXPIRED",
})

# Signals that mark a session as complete
_TERMINAL_SIGNALS = frozenset({
    "AUDIT_WRITE_CONFIRMED",
    "AUDIT_WRITE_FAILED",
    "SESSION_EXPIRED",
})

# Signals whose arrival means the rule evaluation should be recorded
_RULE_EVAL_SIGNAL = "SUGGESTION_RULE_EVALUATED"


# ──────────────────────────────────────────────────────────────────────────────
# SessionStore
# ──────────────────────────────────────────────────────────────────────────────

class SessionStore:
    """
    In-memory store for all ``SessionRecord`` objects.

    Lifecycle
    ---------
    1. ``store = SessionStore()``  — registers a listener with ``signals.emit``.
    2. Run the pipeline.  Every ``emit()`` call fans out to the store listener.
    3. ``store.get_record(session_id)``  — retrieve a fully populated record.
    4. ``store.close()``  — deregisters the listener (call in test teardown).

    The store is designed for a single process.  In production, a separate
    BigQuery sink would replace the in-memory list.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._records: dict[str, SessionRecord] = {}
        self.index: SessionIndex = SessionIndex()
        eamgp.register_listener(self._on_signal)

    # ── Listener callback ─────────────────────────────────────────────────────

    def _on_signal(self, payload: dict[str, Any]) -> None:
        """Called by signals.emit for every emitted payload."""
        session_id = payload.get("session_id", "")
        if not session_id:
            return   # infrastructure signals without session scope are skipped

        record = self._get_or_create(session_id)
        self._append_signal(record, payload)
        self._update_record(record, payload)

    def _get_or_create(self, session_id: str) -> SessionRecord:
        with self._lock:
            if session_id not in self._records:
                self._records[session_id] = SessionRecord(session_id=session_id)
            return self._records[session_id]

    def _append_signal(
        self, record: SessionRecord, payload: dict[str, Any]
    ) -> None:
        ts = payload.get("timestamp_utc", "")
        with self._lock:
            if not record.started_at:
                record.started_at = ts

            # elapsed_ms relative to session start
            elapsed = self._elapsed_ms(record.started_at, ts)

            event = SignalEvent(
                signal=payload["signal"],
                level=payload.get("level", "INFO"),
                zone=payload.get("zone", ""),
                session_id=payload.get("session_id", ""),
                timestamp_utc=ts,
                elapsed_ms=elapsed,
                attributes={
                    k: v for k, v in payload.items()
                    if k not in _ENVELOPE_KEYS
                },
            )
            record.signals.append(event)

    def _update_record(
        self, record: SessionRecord, payload: dict[str, Any]
    ) -> None:
        signal = payload["signal"]
        with self._lock:
            # Error tracking
            if payload.get("level") == "ERROR":
                record.has_error = True
                record.error_signals.append(signal)

            # Zone 1 — graph build
            if signal == "GRAPH_BUILD_COMPLETE":
                record.known_trait_count    = payload.get("known_count", 0)
                record.missing_trait_count  = payload.get("missing_count", 0)
                record.excluded_trait_count = payload.get("excluded_count", 0)
                record.situation_id         = payload.get("situation_id", "")
                record.intent_id            = payload.get("intent_id", "")
            elif signal == "GRAPH_BUILD_START":
                record.intent_id    = payload.get("intent_id", "")
                record.situation_id = payload.get("situation_id", "")

            # Zone 1.5 — ML prediction
            elif signal == "SEG_PREDICT_COMPLETE":
                self._append_prediction(record, payload, "ACTIVE")
            elif signal == "SEG_PREDICT_UNDECIDABLE":
                self._append_prediction(record, payload, "UNDECIDABLE")
            elif signal == "SEG_PREDICT_FAILED":
                self._append_prediction(record, payload, "FAILED")
            elif signal == "SEG_PREDICT_LOW_CONFIDENCE":
                self._append_prediction(record, payload, "LOW_CONFIDENCE")
            elif signal == "SEG_GAP_REORDERED":
                record.fill_strategy = "ML_IG"

            # Zone 2 — gap-fill
            elif signal == "GAP_FILL_ANSWERED":
                self._append_conversation_turn(record, payload)
                record.gap_fill_turns += 1
            elif signal == "SEGMENT_MATCHED":
                record.matched_segment_id = payload.get("segment_id", "")
                record.segment_confidence = payload.get("confidence", 0.0)

            # Zone 3 — suggestion validation
            elif signal == _RULE_EVAL_SIGNAL:
                record.rule_evaluations.append({
                    "rule_id":   payload.get("rule_id", ""),
                    "rule_type": payload.get("rule_type", ""),
                    "outcome":   payload.get("outcome", ""),
                    "suggestion_id": payload.get("suggestion_id", ""),
                })
            elif signal == "SUGGESTION_CANDIDATES_RETRIEVED":
                record.candidates_evaluated = payload.get("candidate_count", 0)
            elif signal == "SUGGESTION_VALIDATED":
                record.validated_candidates += 1

            # Zone 4 — delivery
            elif signal == "OUTPUT_CONSTRUCTED":
                record.gate_disposition = payload.get("gate_disposition", "")
            elif signal == "AUDIT_WRITE_CONFIRMED":
                record.audit_confirmed = True
                record.gate_disposition = record.gate_disposition or "EMIT"
                self._mark_complete(record, payload)
            elif signal == "AUDIT_WRITE_FAILED":
                self._mark_complete(record, payload)
            elif signal == "CONSUMER_EXPLAIN_SERVED":
                pass  # explanation text is in ExplainabilityBundle, not signals

            # Session lifecycle
            elif signal == "SESSION_EXPIRED":
                # SESSION_EXPIRED is also in _PARTY_REF_SIGNALS so must be
                # checked first to reach the mark_complete branch.
                self._mark_complete(record, payload)
            elif signal in _PARTY_REF_SIGNALS:
                party_ref = payload.get("party_ref", "")
                if party_ref:
                    record.party_ref = party_ref
                    self.index.add(party_ref, record.session_id)

    def _append_prediction(
        self,
        record: SessionRecord,
        payload: dict[str, Any],
        disposition: str,
    ) -> None:
        elapsed = self._elapsed_ms(
            record.started_at, payload.get("timestamp_utc", "")
        )
        snap = PredictionSnapshot(
            turn=payload.get("turn", len(record.prediction_chain)),
            top_segment_id=payload.get("top_segment_id"),
            top_confidence=payload.get("top_confidence", 0.0),
            model_version=payload.get("model_version", ""),
            model_algorithm=payload.get("model_algorithm", ""),
            known_trait_count=payload.get("known_trait_count", 0),
            disposition=disposition,
            shap_features_json=str(payload.get("shap_features_json", "[]")),
            elapsed_ms=elapsed,
        )
        record.prediction_chain.append(snap)

    def _append_conversation_turn(
        self,
        record: SessionRecord,
        payload: dict[str, Any],
    ) -> None:
        elapsed = self._elapsed_ms(
            record.started_at, payload.get("timestamp_utc", "")
        )
        turn = ConversationTurn(
            turn_number=payload.get("turn", len(record.conversation) + 1),
            char_id=payload.get("char_id", ""),
            question_text="",      # filled by DataAdapter using QUESTION_TEXT_MAP
            value_hash=payload.get("value_hash"),
            source=payload.get("source", "CONSUMER_INPUT"),
            completeness=0.0,      # not in signal; DataAdapter fills from graph
            elapsed_ms=elapsed,
        )
        record.conversation.append(turn)

    def _mark_complete(
        self, record: SessionRecord, payload: dict[str, Any]
    ) -> None:
        record.is_complete   = True
        record.completed_at  = payload.get("timestamp_utc", "")
        record.total_ms      = self._elapsed_ms(
            record.started_at, record.completed_at
        )

    @staticmethod
    def _elapsed_ms(start_iso: str, end_iso: str) -> float:
        """Return milliseconds between two ISO8601 timestamps.  Returns 0 on error."""
        try:
            start = datetime.fromisoformat(start_iso)
            end   = datetime.fromisoformat(end_iso)
            return max(0.0, (end - start).total_seconds() * 1000)
        except (ValueError, TypeError):
            return 0.0

    # ── Public read API ───────────────────────────────────────────────────────

    def get_record(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._records.get(session_id)

    def all_session_ids(self) -> list[str]:
        with self._lock:
            return list(self._records.keys())

    def all_records(self) -> list[SessionRecord]:
        with self._lock:
            return list(self._records.values())

    def record_count(self) -> int:
        with self._lock:
            return len(self._records)

    def register_session(
        self,
        session_id: str,
        party_ref: str,
        intent_id: str = "",
        situation_id: str = "",
        channel: str = "mobile",
    ) -> SessionRecord:
        """
        Explicitly register a session before signals arrive.

        Called by the pipeline bootstrap (``LeadOrchestrator``) so that
        ``party_ref`` is in the index from the start — before any zone
        signal is emitted.
        """
        record = self._get_or_create(session_id)
        with self._lock:
            record.party_ref    = party_ref
            record.intent_id    = intent_id
            record.situation_id = situation_id
            record.channel      = channel
            if not record.started_at:
                record.started_at = datetime.now(timezone.utc).isoformat()
        self.index.add(party_ref, session_id)
        return record

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Deregister the listener.  Call in test teardown or app shutdown."""
        eamgp.deregister_listener(self._on_signal)

    def clear(self) -> None:
        """Clear all records and index.  Useful in tests."""
        with self._lock:
            self._records.clear()
        with self.index._lock:
            self.index._index.clear()
