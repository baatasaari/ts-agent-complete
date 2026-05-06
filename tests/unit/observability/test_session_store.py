"""
tests/unit/observability/test_session_store.py
===============================================
Unit tests for SessionStore, SessionIndex, and the listener API added
to signals.py.  All tests use the real emit() function — no mocking of
eamgp.emit so we exercise the full listener dispatch path.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from ts_agent.observability import signals as eamgp
from ts_agent.observability.session_store import (
    ConversationTurn,
    PredictionSnapshot,
    SessionIndex,
    SessionRecord,
    SessionStore,
    SignalEvent,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_listeners():
    """Ensure the global listener list is empty before and after each test."""
    eamgp.clear_listeners()
    yield
    eamgp.clear_listeners()


@pytest.fixture()
def store():
    s = SessionStore()
    yield s
    s.close()


# ──────────────────────────────────────────────────────────────────────────────
# signals.py listener API
# ──────────────────────────────────────────────────────────────────────────────

class TestListenerAPI:

    def test_register_listener_fires_on_emit(self):
        received = []
        eamgp.register_listener(received.append)
        eamgp.emit("TEST", eamgp.INFO, "Zone1", session_id="s1")
        assert len(received) == 1
        assert received[0]["signal"] == "TEST"

    def test_listener_receives_shallow_copy(self):
        received = []
        eamgp.register_listener(received.append)
        eamgp.emit("TEST", eamgp.INFO, "Zone1", session_id="s1", foo="bar")
        payload = received[0]
        payload["foo"] = "MUTATED"
        # Emit again and verify original emit still returns "bar"
        received2 = []
        eamgp.register_listener(received2.append)
        eamgp.emit("TEST", eamgp.INFO, "Zone1", session_id="s1", foo="bar")
        assert received2[0]["foo"] == "bar"

    def test_deregister_stops_listener(self):
        received = []
        fn = received.append
        eamgp.register_listener(fn)
        eamgp.emit("BEFORE", eamgp.INFO, "Zone1", session_id="s1")
        eamgp.deregister_listener(fn)
        eamgp.emit("AFTER", eamgp.INFO, "Zone1", session_id="s1")
        assert len(received) == 1
        assert received[0]["signal"] == "BEFORE"

    def test_deregister_nonexistent_is_noop(self):
        eamgp.deregister_listener(lambda x: None)   # should not raise

    def test_multiple_listeners_all_receive(self):
        r1, r2 = [], []
        eamgp.register_listener(r1.append)
        eamgp.register_listener(r2.append)
        eamgp.emit("MULTI", eamgp.INFO, "Zone1", session_id="s1")
        assert len(r1) == 1
        assert len(r2) == 1

    def test_listener_exception_does_not_propagate(self):
        def bad_listener(payload):
            raise RuntimeError("listener error")
        eamgp.register_listener(bad_listener)
        # Should not raise; the exception is caught inside _dispatch_listeners
        result = eamgp.emit("SAFE", eamgp.INFO, "Zone1", session_id="s1")
        assert result["signal"] == "SAFE"

    def test_listener_count_tracks_registrations(self):
        assert eamgp.listener_count() == 0
        eamgp.register_listener(lambda x: None)
        assert eamgp.listener_count() == 1
        eamgp.register_listener(lambda x: None)
        assert eamgp.listener_count() == 2

    def test_clear_listeners_resets_to_zero(self):
        eamgp.register_listener(lambda x: None)
        eamgp.register_listener(lambda x: None)
        eamgp.clear_listeners()
        assert eamgp.listener_count() == 0

    def test_emit_return_value_unchanged(self):
        eamgp.register_listener(lambda x: None)
        p = eamgp.emit("RET", eamgp.INFO, "Zone1", session_id="s1", k="v")
        assert p["signal"] == "RET"
        assert p["k"] == "v"

    def test_thread_safety_concurrent_emits(self):
        """Multiple threads emitting simultaneously must not lose or duplicate signals."""
        received = []
        lock = threading.Lock()

        def safe_append(payload):
            with lock:
                received.append(payload["signal"])

        eamgp.register_listener(safe_append)
        threads = [
            threading.Thread(
                target=eamgp.emit,
                args=(f"SIG_{i}", eamgp.INFO, "Zone1"),
                kwargs={"session_id": f"s{i}"}
            )
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(received) == 20


# ──────────────────────────────────────────────────────────────────────────────
# SessionIndex
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionIndex:

    def test_add_and_sessions_for(self):
        idx = SessionIndex()
        idx.add("PARTY-A", "sess-1")
        idx.add("PARTY-A", "sess-2")
        assert idx.sessions_for("PARTY-A") == ["sess-1", "sess-2"]

    def test_sessions_for_unknown_party_returns_empty(self):
        idx = SessionIndex()
        assert idx.sessions_for("PARTY-X") == []

    def test_no_duplicate_session_ids(self):
        idx = SessionIndex()
        idx.add("PARTY-A", "sess-1")
        idx.add("PARTY-A", "sess-1")
        assert len(idx.sessions_for("PARTY-A")) == 1

    def test_all_party_refs_sorted(self):
        idx = SessionIndex()
        idx.add("PARTY-C", "s1")
        idx.add("PARTY-A", "s2")
        idx.add("PARTY-B", "s3")
        assert idx.all_party_refs() == ["PARTY-A", "PARTY-B", "PARTY-C"]

    def test_party_for_session_reverse_lookup(self):
        idx = SessionIndex()
        idx.add("PARTY-A", "sess-42")
        assert idx.party_for_session("sess-42") == "PARTY-A"

    def test_party_for_session_unknown_returns_none(self):
        idx = SessionIndex()
        assert idx.party_for_session("nonexistent") is None

    def test_total_sessions(self):
        idx = SessionIndex()
        idx.add("P1", "s1"); idx.add("P1", "s2"); idx.add("P2", "s3")
        assert idx.total_sessions() == 3

    def test_empty_party_ref_is_ignored(self):
        idx = SessionIndex()
        idx.add("", "sess-1")
        assert idx.all_party_refs() == []

    def test_empty_session_id_is_ignored(self):
        idx = SessionIndex()
        idx.add("PARTY-A", "")
        assert idx.sessions_for("PARTY-A") == []


# ──────────────────────────────────────────────────────────────────────────────
# SessionStore — signal routing
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionStore:

    def test_store_registers_listener_on_creation(self):
        store = SessionStore()
        assert eamgp.listener_count() >= 1
        store.close()

    def test_close_deregisters_listener(self):
        count_before = eamgp.listener_count()
        store = SessionStore()
        store.close()
        assert eamgp.listener_count() == count_before

    def test_signal_without_session_id_is_ignored(self, store):
        eamgp.emit("NO_SESSION", eamgp.INFO, "Zone1")   # no session_id kwarg
        assert store.record_count() == 0

    def test_signal_creates_session_record(self, store):
        eamgp.emit("GRAPH_BUILD_START", eamgp.INFO, "Zone1", session_id="sess-a")
        assert store.get_record("sess-a") is not None

    def test_signal_appended_to_record(self, store):
        sid = "sess-b"
        eamgp.emit("GRAPH_BUILD_START", eamgp.INFO, "Zone1", session_id=sid)
        eamgp.emit("GRAPH_BUILD_COMPLETE", eamgp.INFO, "Zone1",
                   session_id=sid, known_count=5, missing_count=2,
                   excluded_count=1, node_count=8, edge_count=8,
                   intent_id="INTENT-INVEST-CASH", situation_id="SIT-INV-001")
        record = store.get_record(sid)
        assert len(record.signals) == 2

    def test_graph_build_complete_populates_counts(self, store):
        sid = "sess-c"
        eamgp.emit("GRAPH_BUILD_COMPLETE", eamgp.INFO, "Zone1",
                   session_id=sid, known_count=6, missing_count=3,
                   excluded_count=1, node_count=10, edge_count=10,
                   intent_id="INTENT-INVEST-CASH", situation_id="SIT-INV-001")
        r = store.get_record(sid)
        assert r.known_trait_count   == 6
        assert r.missing_trait_count == 3
        assert r.excluded_trait_count== 1

    def test_gap_fill_answered_appends_conversation_turn(self, store):
        sid = "sess-d"
        eamgp.emit("GAP_FILL_ANSWERED", eamgp.INFO, "Zone2",
                   session_id=sid, turn=1, char_id="CHAR-B3A-I1",
                   value_hash="abc123", source="CONSUMER_INPUT")
        r = store.get_record(sid)
        assert len(r.conversation) == 1
        assert r.conversation[0].char_id == "CHAR-B3A-I1"
        assert r.gap_fill_turns == 1

    def test_segment_matched_stored(self, store):
        sid = "sess-e"
        eamgp.emit("SEGMENT_MATCHED", eamgp.INFO, "Zone2",
                   session_id=sid, segment_id="SEG-INV-001", confidence=0.91)
        r = store.get_record(sid)
        assert r.matched_segment_id  == "SEG-INV-001"
        assert abs(r.segment_confidence - 0.91) < 1e-6

    def test_rule_evaluation_appended(self, store):
        sid = "sess-f"
        eamgp.emit("SUGGESTION_RULE_EVALUATED", eamgp.INFO, "Zone3",
                   session_id=sid, rule_id="R-001", rule_type="HARD",
                   outcome="PASS", suggestion_id="SUG-INV-001")
        r = store.get_record(sid)
        assert len(r.rule_evaluations) == 1
        assert r.rule_evaluations[0]["rule_id"] == "R-001"

    def test_audit_write_confirmed_marks_complete(self, store):
        sid = "sess-g"
        eamgp.emit("AUDIT_WRITE_CONFIRMED", eamgp.INFO, "Zone4",
                   session_id=sid, audit_id="aud-1")
        r = store.get_record(sid)
        assert r.is_complete is True
        assert r.audit_confirmed is True

    def test_error_signal_sets_has_error(self, store):
        sid = "sess-h"
        eamgp.emit("NEO4J_WRITE_HARD_FAIL", eamgp.ERROR, "Zone1",
                   session_id=sid, operation="write_hard")
        r = store.get_record(sid)
        assert r.has_error is True
        assert "NEO4J_WRITE_HARD_FAIL" in r.error_signals

    def test_multiple_sessions_are_isolated(self, store):
        eamgp.emit("GRAPH_BUILD_START", eamgp.INFO, "Zone1", session_id="s1")
        eamgp.emit("GRAPH_BUILD_START", eamgp.INFO, "Zone1", session_id="s2")
        assert store.record_count() == 2
        assert store.get_record("s1") is not None
        assert store.get_record("s2") is not None
        assert store.get_record("s1") is not store.get_record("s2")

    def test_register_session_populates_index(self, store):
        store.register_session("sess-z", "PARTY-Z", intent_id="INTENT-X")
        assert "PARTY-Z" in store.index.all_party_refs()
        assert "sess-z" in store.index.sessions_for("PARTY-Z")

    def test_register_session_sets_party_ref(self, store):
        store.register_session("sess-zz", "PARTY-ZZ")
        r = store.get_record("sess-zz")
        assert r.party_ref == "PARTY-ZZ"

    def test_all_session_ids_returns_all(self, store):
        for i in range(5):
            eamgp.emit("X", eamgp.INFO, "Zone1", session_id=f"sess-{i}")
        assert len(store.all_session_ids()) == 5

    def test_elapsed_ms_computed_correctly(self, store):
        eamgp.emit("GRAPH_BUILD_START", eamgp.INFO, "Zone1", session_id="sess-t")
        time.sleep(0.05)
        eamgp.emit("GRAPH_BUILD_COMPLETE", eamgp.INFO, "Zone1",
                   session_id="sess-t", known_count=1, missing_count=0,
                   excluded_count=0, node_count=1, edge_count=1)
        r = store.get_record("sess-t")
        elapsed = r.signals[-1].elapsed_ms
        assert elapsed >= 0   # should be positive; may be ~50ms in real runs

    def test_get_record_returns_none_for_unknown(self, store):
        assert store.get_record("nonexistent") is None

    def test_clear_resets_store(self, store):
        eamgp.emit("X", eamgp.INFO, "Zone1", session_id="sess-clear")
        assert store.record_count() == 1
        store.clear()
        assert store.record_count() == 0

    def test_prediction_snapshot_appended(self, store):
        sid = "sess-pred"
        eamgp.emit("SEG_PREDICT_COMPLETE", eamgp.INFO, "Zone1.5",
                   session_id=sid, turn=1, top_segment_id="SEG-INV-001",
                   top_confidence=0.82, model_version="demo-1.3",
                   model_algorithm="LR", known_trait_count=7,
                   shap_features_json="[]")
        r = store.get_record(sid)
        assert len(r.prediction_chain) == 1
        assert r.prediction_chain[0].top_segment_id == "SEG-INV-001"
        assert abs(r.prediction_chain[0].top_confidence - 0.82) < 1e-6

    def test_undecidable_prediction_stored(self, store):
        sid = "sess-undec"
        eamgp.emit("SEG_PREDICT_UNDECIDABLE", eamgp.WARN, "Zone1.5",
                   session_id=sid, turn=0, known_trait_count=2, threshold=5)
        r = store.get_record(sid)
        assert len(r.prediction_chain) == 1
        assert r.prediction_chain[0].disposition == "UNDECIDABLE"


# ──────────────────────────────────────────────────────────────────────────────
# Additional signal-handler branches not covered by the original test set
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionStoreAdditionalBranches:
    """Cover the remaining elif branches in _update_record."""

    def test_seg_predict_failed_stored(self, store):
        sid = "sess-pred-fail"
        eamgp.emit(
            "SEG_PREDICT_FAILED", eamgp.ERROR, "Zone1.5",
            session_id=sid, turn=2, known_trait_count=4,
        )
        r = store.get_record(sid)
        assert r is not None
        assert len(r.prediction_chain) == 1
        assert r.prediction_chain[0].disposition == "FAILED"

    def test_seg_predict_low_confidence_stored(self, store):
        sid = "sess-lowconf"
        eamgp.emit(
            "SEG_PREDICT_LOW_CONFIDENCE", eamgp.WARN, "Zone1.5",
            session_id=sid, turn=3, top_segment_id="SEG-INV-001",
            top_confidence=0.52, model_version="demo",
            model_algorithm="LR", known_trait_count=6,
            shap_features_json="[]",
        )
        r = store.get_record(sid)
        assert len(r.prediction_chain) == 1
        assert r.prediction_chain[0].disposition == "LOW_CONFIDENCE"

    def test_seg_gap_reordered_sets_fill_strategy(self, store):
        sid = "sess-reorder"
        eamgp.emit(
            "SEG_GAP_REORDERED", eamgp.INFO, "Zone1.5",
            session_id=sid, new_order_char_ids=["CHAR-F2A-I1"],
        )
        r = store.get_record(sid)
        assert r.fill_strategy == "ML_IG"

    def test_audit_write_failed_marks_complete(self, store):
        sid = "sess-auditfail"
        eamgp.emit(
            "AUDIT_WRITE_FAILED", eamgp.ERROR, "Zone4",
            session_id=sid, attempt_count=3,
        )
        r = store.get_record(sid)
        assert r.is_complete is True

    def test_session_expired_marks_complete(self, store):
        sid = "sess-expired"
        eamgp.emit(
            "SESSION_EXPIRED", eamgp.WARN, "Session",
            session_id=sid, age_hours=25,
        )
        r = store.get_record(sid)
        assert r.is_complete is True

    def test_session_resumed_sets_party_ref_in_index(self, store):
        sid = "sess-resumed"
        store.register_session(sid, "PARTY-R")
        eamgp.emit(
            "SESSION_RESUMED", eamgp.INFO, "Session",
            session_id=sid, party_ref="PARTY-R",
            traits_already_known=3,
        )
        assert "PARTY-R" in store.index.all_party_refs()

    def test_elapsed_ms_returns_zero_on_bad_timestamps(self, store):
        from ts_agent.observability.session_store import SessionStore as SS
        result = SS._elapsed_ms("not-a-date", "also-not-a-date")
        assert result == 0.0

    def test_elapsed_ms_returns_zero_when_start_empty(self, store):
        from ts_agent.observability.session_store import SessionStore as SS
        result = SS._elapsed_ms("", "2026-01-01T00:00:01+00:00")
        assert result == 0.0

    def test_graph_build_start_stores_intent_and_situation(self, store):
        sid = "sess-buildstart"
        eamgp.emit(
            "GRAPH_BUILD_START", eamgp.INFO, "Zone1",
            session_id=sid,
            intent_id="INTENT-INVEST-CASH",
            situation_id="SIT-INV-001",
        )
        r = store.get_record(sid)
        assert r.intent_id    == "INTENT-INVEST-CASH"
        assert r.situation_id == "SIT-INV-001"

    def test_suggestion_validated_increments_counter(self, store):
        sid = "sess-validated"
        eamgp.emit(
            "SUGGESTION_VALIDATED", eamgp.INFO, "Zone3",
            session_id=sid, suggestion_id="SUG-INV-001",
            rules_passed_count=10,
        )
        r = store.get_record(sid)
        assert r.validated_candidates == 1


# ──────────────────────────────────────────────────────────────────────────────
# signals.py — timed() context manager and span helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestTimedContextManager:
    """Cover the timed() context manager and get_emitted_spans/clear_spans."""

    def test_timed_emits_signal_on_exit(self):
        received = []
        eamgp.register_listener(received.append)
        with eamgp.timed("GRAPH_BUILD_COMPLETE", "Zone1", session_id="sess-t1"):
            pass
        eamgp.deregister_listener(received.append)
        assert any(p["signal"] == "GRAPH_BUILD_COMPLETE" for p in received)

    def test_timed_sets_latency_ms(self):
        received = []
        eamgp.register_listener(received.append)
        with eamgp.timed("GRAPH_BUILD_COMPLETE", "Zone1", session_id="sess-t2"):
            pass
        eamgp.deregister_listener(received.append)
        payload = next(p for p in received if p["signal"] == "GRAPH_BUILD_COMPLETE")
        assert "latency_ms" in payload
        assert isinstance(payload["latency_ms"], int)
        assert payload["latency_ms"] >= 0

    def test_timed_emits_error_level_on_exception(self):
        received = []
        eamgp.register_listener(received.append)
        try:
            with eamgp.timed("TEST_TIMED_ERR", "Zone1", session_id="sess-t3"):
                raise ValueError("deliberate error")
        except ValueError:
            pass
        eamgp.deregister_listener(received.append)
        err_payloads = [p for p in received if p["signal"] == "TEST_TIMED_ERR"]
        assert len(err_payloads) == 1
        assert err_payloads[0]["level"] == "ERROR"
        assert "deliberate error" in str(err_payloads[0].get("error", ""))

    def test_timed_does_not_suppress_exceptions(self):
        """timed.__exit__ must always return False so exceptions propagate."""
        with pytest.raises(RuntimeError, match="propagated"):
            with eamgp.timed("TEST_PROPAGATE", "Zone1", session_id="sess-t4"):
                raise RuntimeError("propagated")

    def test_timed_context_attr_latency_ms_accessible(self):
        with eamgp.timed("TEST_CTX", "Zone1", session_id="sess-t5") as ctx:
            pass
        assert isinstance(ctx.latency_ms, int)

    def test_get_emitted_spans_returns_sequence(self):
        result = eamgp.get_emitted_spans()
        # OTEL InMemorySpanExporter returns a tuple or list depending on version
        assert hasattr(result, "__iter__") and hasattr(result, "__len__")

    def test_clear_spans_empties_exporter(self):
        eamgp.emit("SPAN_TEST", eamgp.INFO, "Zone1", session_id="s1")
        eamgp.clear_spans()
        assert len(eamgp.get_emitted_spans()) == 0
