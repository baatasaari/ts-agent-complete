"""
tests/unit/visualiser/test_data_adapter.py
==========================================
Unit tests for DataAdapter — all pure enrichment functions.
"""
from __future__ import annotations

import pytest

from ts_agent.observability.session_store import (
    ConversationTurn,
    PredictionSnapshot,
    SessionRecord,
    SignalEvent,
)
from ts_agent.visualiser.data_adapter import (
    CHAR_BRANCH_MAP,
    CHAR_SHORT_LABEL,
    QUESTION_TEXT_MAP,
    DataAdapter,
)


# ──────────────────────────────────────────────────────────────────────────────
# QUESTION_TEXT_MAP
# ──────────────────────────────────────────────────────────────────────────────

class TestQuestionTextMap:

    def test_all_chars_have_question_text(self):
        # v2 char_ids — PS25/22 ontology
        known_chars = [
            # Personal
            "CHAR-P1A-I1", "CHAR-P1B-I1", "CHAR-P1C-I1",
            "CHAR-P1D-I1", "CHAR-P1E-I1",
            # Financial — investment domain
            "CHAR-F2A-I1", "CHAR-F2B-I1", "CHAR-F2G-I1", "CHAR-F2H-I1",
            "CHAR-F2I-I1", "CHAR-F2J-I1", "CHAR-F2K-I1", "CHAR-F2L-I1",
            "CHAR-F2M-I1", "CHAR-F2N-I1", "CHAR-F2O-I1", "CHAR-F2P-I1",
            "CHAR-F2Q-I1", "CHAR-F2R-I1",
            # Pension — accumulation
            "CHAR-P2A-I1", "CHAR-P2B-I1", "CHAR-P2C-I1", "CHAR-P2D-I1",
            "CHAR-P2E-I1", "CHAR-P2F-I1", "CHAR-P2G-I1", "CHAR-P2H-I1",
            # Pension — decumulation
            "CHAR-P2I-I1", "CHAR-P2J-I1", "CHAR-P2K-I1", "CHAR-P2L-I1",
            "CHAR-P2M-I1", "CHAR-P2N-I1", "CHAR-P2O-I1", "CHAR-P2P-I1",
            # Behavioural
            "CHAR-B3A-I1", "CHAR-B3B-I1", "CHAR-B3C-I1",
        ]
        for char_id in known_chars:
            assert char_id in QUESTION_TEXT_MAP, f"{char_id} missing from QUESTION_TEXT_MAP"
            assert len(QUESTION_TEXT_MAP[char_id]) > 10

    def test_question_text_is_non_empty_string(self):
        for char_id, text in QUESTION_TEXT_MAP.items():
            assert isinstance(text, str) and text.strip(), f"{char_id} has empty text"

    def test_char_short_label_covers_same_chars(self):
        assert set(CHAR_SHORT_LABEL.keys()) == set(QUESTION_TEXT_MAP.keys())

    def test_char_branch_map_covers_same_chars(self):
        assert set(CHAR_BRANCH_MAP.keys()) == set(QUESTION_TEXT_MAP.keys())

    def test_all_branches_are_valid(self):
        valid = {"Personal", "Financial", "Behavioural", "Pension", "Temporal"}
        for char_id, branch in CHAR_BRANCH_MAP.items():
            assert branch in valid, f"{char_id} has invalid branch '{branch}'"


# ──────────────────────────────────────────────────────────────────────────────
# DataAdapter.session_summary
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionSummary:

    def _make_record(self, **kwargs) -> SessionRecord:
        r = SessionRecord(session_id="sess-test")
        for k, v in kwargs.items():
            setattr(r, k, v)
        return r

    def test_summary_has_required_keys(self):
        r = self._make_record(situation_id="SIT-INV-001", gate_disposition="EMIT")
        summary = DataAdapter.session_summary(r)
        required = {
            "session_id", "party_ref", "situation_label", "gate_disposition",
            "completeness_pct", "gap_fill_turns", "signal_count",
        }
        for key in required:
            assert key in summary, f"Key '{key}' missing from summary"

    def test_unknown_situation_returns_raw_id(self):
        r = self._make_record(situation_id="SIT-UNKNOWN-999")
        summary = DataAdapter.session_summary(r)
        assert summary["situation_label"] == "SIT-UNKNOWN-999"

    def test_known_situation_returns_label(self):
        r = self._make_record(situation_id="SIT-INV-001")
        summary = DataAdapter.session_summary(r)
        # SIT-INV-001 label contains "Cash Drag" or "Invest"
        assert summary["situation_label"] != "SIT-INV-001"  # must look up label

    def test_completeness_zero_when_all_missing(self):
        r = self._make_record(known_trait_count=0, missing_trait_count=5)
        pct = DataAdapter.completeness_pct(r)
        assert pct == 0.0

    def test_completeness_100_when_all_known(self):
        r = self._make_record(known_trait_count=8, missing_trait_count=0)
        pct = DataAdapter.completeness_pct(r)
        assert pct == 100.0

    def test_completeness_excludes_excluded_from_denominator(self):
        r = self._make_record(
            known_trait_count=5,
            missing_trait_count=5,
            excluded_trait_count=3,
        )
        # eligible = known + missing = 10; completeness = 5/10 = 50%
        pct = DataAdapter.completeness_pct(r)
        assert abs(pct - 50.0) < 0.1

    def test_gate_disposition_in_summary(self):
        r = self._make_record(gate_disposition="HUMAN_REVIEW")
        summary = DataAdapter.session_summary(r)
        assert summary["gate_disposition"] == "HUMAN_REVIEW"


# ──────────────────────────────────────────────────────────────────────────────
# DataAdapter.enrich_conversation
# ──────────────────────────────────────────────────────────────────────────────

class TestEnrichConversation:

    def _make_turn(self, char_id: str, turn: int = 1) -> ConversationTurn:
        return ConversationTurn(
            turn_number=turn,
            char_id=char_id,
            question_text="",
            value_hash="abc123hash",
            source="CONSUMER_INPUT",
            completeness=0.75,
            elapsed_ms=120.0,
        )

    def test_known_char_id_has_question_text(self):
        turns = [self._make_turn("CHAR-B3A-I1")]
        enriched = DataAdapter.enrich_conversation(turns)
        assert enriched[0]["question_text"] != ""
        assert "risk" in enriched[0]["question_text"].lower()

    def test_unknown_char_id_has_fallback_text(self):
        turns = [self._make_turn("CHAR-UNKNOWN-99")]
        enriched = DataAdapter.enrich_conversation(turns)
        assert "CHAR-UNKNOWN-99" in enriched[0]["question_text"]

    def test_branch_populated(self):
        turns = [self._make_turn("CHAR-F2A-I1")]
        enriched = DataAdapter.enrich_conversation(turns)
        assert enriched[0]["branch"] == "Financial"

    def test_value_hash_truncated_safely(self):
        """Enrichment does not truncate — that is the component's job."""
        turns = [self._make_turn("CHAR-P1A-I1")]
        enriched = DataAdapter.enrich_conversation(turns)
        assert enriched[0]["value_hash"] == "abc123hash"

    def test_empty_turns_returns_empty_list(self):
        assert DataAdapter.enrich_conversation([]) == []

    def test_short_label_populated(self):
        turns = [self._make_turn("CHAR-F2A-I1")]
        enriched = DataAdapter.enrich_conversation(turns)
        assert enriched[0]["short_label"] == "Monthly Surplus £"


# ──────────────────────────────────────────────────────────────────────────────
# DataAdapter.enrich_signals
# ──────────────────────────────────────────────────────────────────────────────

class TestEnrichSignals:

    def _make_signal(self, signal: str, zone: str, level: str = "INFO",
                     elapsed: float = 0.0) -> SignalEvent:
        return SignalEvent(
            signal=signal, level=level, zone=zone,
            session_id="sess-x", timestamp_utc="2026-01-01T00:00:00+00:00",
            elapsed_ms=elapsed, attributes={},
        )

    def test_zone_order_assigned(self):
        sigs = [self._make_signal("X", "Zone3")]
        enriched = DataAdapter.enrich_signals(sigs)
        assert enriched[0]["zone_order"] == 4

    def test_info_colour_is_green(self):
        sigs = [self._make_signal("X", "Zone1", "INFO")]
        enriched = DataAdapter.enrich_signals(sigs)
        assert enriched[0]["colour"] == "#2ECC71"

    def test_error_colour_is_red(self):
        sigs = [self._make_signal("X", "Zone1", "ERROR")]
        enriched = DataAdapter.enrich_signals(sigs)
        assert enriched[0]["colour"] == "#E74C3C"

    def test_signals_sorted_by_elapsed_ms(self):
        sigs = [
            self._make_signal("LATE",  "Zone3", elapsed=200.0),
            self._make_signal("EARLY", "Zone1", elapsed=10.0),
        ]
        enriched = DataAdapter.enrich_signals(sigs)
        assert enriched[0]["signal"] == "EARLY"
        assert enriched[1]["signal"] == "LATE"

    def test_empty_signals_returns_empty_list(self):
        assert DataAdapter.enrich_signals([]) == []


# ──────────────────────────────────────────────────────────────────────────────
# DataAdapter.enrich_rules
# ──────────────────────────────────────────────────────────────────────────────

class TestEnrichRules:

    def test_known_rule_id_has_description(self):
        evals = [{"rule_id": "R-001", "rule_type": "HARD", "outcome": "PASS",
                  "suggestion_id": "SUG-INV-001"}]
        enriched = DataAdapter.enrich_rules(evals)
        r001 = next(r for r in enriched if r["rule_id"] == "R-001")
        assert "segment" in r001["description"].lower()

    def test_known_rule_has_fca_ref(self):
        evals = [{"rule_id": "R-009", "rule_type": "GATE", "outcome": "GATE",
                  "suggestion_id": "SUG-INV-001"}]
        enriched = DataAdapter.enrich_rules(evals)
        r009 = next(r for r in enriched if r["rule_id"] == "R-009")
        assert r009["fca_ref"] != "—"

    def test_unevaluated_rules_added_as_not_reached(self):
        # Only supply R-001; all others should appear as NOT_REACHED
        evals = [{"rule_id": "R-001", "rule_type": "HARD", "outcome": "PASS",
                  "suggestion_id": ""}]
        enriched = DataAdapter.enrich_rules(evals)
        not_reached = [r for r in enriched if r["outcome"] == "NOT_REACHED"]
        assert len(not_reached) == 11   # 12 total - 1 evaluated

    def test_all_twelve_rules_present(self):
        enriched = DataAdapter.enrich_rules([])
        rule_ids = {r["rule_id"] for r in enriched}
        for i in range(1, 13):
            assert f"R-{i:03d}" in rule_ids

    def test_deduplication_on_rule_id(self):
        evals = [
            {"rule_id": "R-001", "rule_type": "HARD", "outcome": "PASS", "suggestion_id": ""},
            {"rule_id": "R-001", "rule_type": "HARD", "outcome": "FAIL", "suggestion_id": ""},
        ]
        enriched = DataAdapter.enrich_rules(evals)
        r001_count = sum(1 for r in enriched if r["rule_id"] == "R-001")
        assert r001_count == 1

    def test_result_sorted_by_rule_id(self):
        enriched = DataAdapter.enrich_rules([])
        ids = [r["rule_id"] for r in enriched]
        assert ids == sorted(ids)


# ──────────────────────────────────────────────────────────────────────────────
# DataAdapter.enrich_predictions
# ──────────────────────────────────────────────────────────────────────────────

class TestEnrichPredictions:

    def _make_snap(self, turn: int, seg_id: str, conf: float,
                   disposition: str = "ACTIVE") -> PredictionSnapshot:
        return PredictionSnapshot(
            turn=turn, top_segment_id=seg_id, top_confidence=conf,
            model_version="1.0", model_algorithm="LR",
            known_trait_count=5, disposition=disposition,
            shap_features_json="[]", elapsed_ms=0.0,
        )

    def test_known_segment_has_label(self):
        snaps = [self._make_snap(1, "SEG-INV-001", 0.88)]
        enriched = DataAdapter.enrich_predictions(snaps)
        assert enriched[0]["segment_label"] != "SEG-INV-001"
        assert len(enriched[0]["segment_label"]) > 3  # must have a meaningful label

    def test_unknown_segment_returns_raw_id(self):
        snaps = [self._make_snap(1, "SEG-UNKNOWN-99", 0.50)]
        enriched = DataAdapter.enrich_predictions(snaps)
        assert enriched[0]["segment_label"] == "SEG-UNKNOWN-99"

    def test_none_segment_returns_dash(self):
        snap = PredictionSnapshot(
            turn=0, top_segment_id=None, top_confidence=0.0,
            model_version="1.0", model_algorithm="LR",
            known_trait_count=2, disposition="UNDECIDABLE",
            shap_features_json="[]", elapsed_ms=0.0,
        )
        enriched = DataAdapter.enrich_predictions([snap])
        assert enriched[0]["segment_label"] == "—"

    def test_confidence_rounded_to_4dp(self):
        snaps = [self._make_snap(1, "SEG-INV-001", 0.8888888)]
        enriched = DataAdapter.enrich_predictions(snaps)
        assert enriched[0]["top_confidence"] == round(0.8888888, 4)

    def test_empty_chain_returns_empty_list(self):
        assert DataAdapter.enrich_predictions([]) == []


# ──────────────────────────────────────────────────────────────────────────────
# DataAdapter.sankey_for_session
# ──────────────────────────────────────────────────────────────────────────────

class TestSankeyForSession:

    def test_returns_labels_and_links(self):
        r = SessionRecord(session_id="s", situation_id="SIT-INV-001",
                         intent_id="INTENT-INVEST-CASH", matched_segment_id="SEG-INV-001",
                         gate_disposition="EMIT")
        data = DataAdapter.sankey_for_session(r)
        assert "labels" in data
        assert "links"  in data

    def test_labels_has_five_nodes(self):
        r = SessionRecord(session_id="s", situation_id="SIT-INV-001",
                         intent_id="INTENT-INVEST-CASH", matched_segment_id="SEG-INV-001",
                         gate_disposition="EMIT")
        data = DataAdapter.sankey_for_session(r)
        assert len(data["labels"]) == 5

    def test_links_has_four_edges(self):
        r = SessionRecord(session_id="s", situation_id="SIT-INV-001",
                         intent_id="INTENT-INVEST-CASH", matched_segment_id="SEG-INV-001",
                         gate_disposition="EMIT")
        data = DataAdapter.sankey_for_session(r)
        assert len(data["links"]) == 4

    def test_gate_label_is_last_node(self):
        r = SessionRecord(session_id="s", situation_id="SIT-INV-001",
                         intent_id="INTENT-INVEST-CASH", matched_segment_id="SEG-INV-001",
                         gate_disposition="SUPPRESS")
        data = DataAdapter.sankey_for_session(r)
        assert data["labels"][-1] == "SUPPRESS"


# ──────────────────────────────────────────────────────────────────────────────
# DataAdapter.sankey_aggregate
# ──────────────────────────────────────────────────────────────────────────────

class TestSankeyAggregate:

    def _make_record(self, sit: str, seg: str, gate: str,
                     intent: str = "INTENT-INVEST-CASH") -> SessionRecord:
        r = SessionRecord(session_id=f"s-{sit}-{seg}-{gate}")
        r.situation_id       = sit
        r.intent_id          = intent
        r.matched_segment_id = seg
        r.gate_disposition   = gate
        return r

    def test_aggregate_returns_labels_and_links(self):
        records = [
            self._make_record("SIT-INV-001", "SEG-INV-001", "EMIT"),
            self._make_record("SIT-PEN-001", "SEG-PEN-001", "SUPPRESS"),
        ]
        data = DataAdapter.sankey_aggregate(records)
        assert "labels" in data
        assert "links"  in data

    def test_link_count_increases_with_sessions(self):
        r1 = [self._make_record("SIT-INV-001", "SEG-INV-001", "EMIT")]
        r2 = r1 + [self._make_record("SIT-INV-001", "SEG-INV-002", "SUPPRESS")]
        d1 = DataAdapter.sankey_aggregate(r1)
        d2 = DataAdapter.sankey_aggregate(r2)
        assert len(d2["links"]) >= len(d1["links"])

    def test_empty_records_returns_empty_labels(self):
        data = DataAdapter.sankey_aggregate([])
        assert data["labels"] == []
        assert data["links"]  == []

    def test_emit_links_have_green_tint(self):
        records = [self._make_record("SIT-INV-001", "SEG-INV-001", "EMIT")]
        data = DataAdapter.sankey_aggregate(records)
        emit_links = [lk for lk in data["links"] if "EMIT" in
                      data["labels"][lk["target"]]]
        assert all("46, 204, 113" in lk["colour"] for lk in emit_links)


# ──────────────────────────────────────────────────────────────────────────────
# Component empty-state branches (conversation, waterfall, sankey, ML)
# ──────────────────────────────────────────────────────────────────────────────

class TestComponentEmptyStates:
    """Verify every component returns a valid figure when given empty input."""

    def test_conversation_panel_empty_turns(self):
        from ts_agent.visualiser.components.conversation import build_conversation_panel
        fig = build_conversation_panel([])
        # Empty state renders an annotation, not a table
        annotations = fig.layout.annotations
        assert len(annotations) > 0
        assert any("No conversation" in str(a.text) for a in annotations)

    def test_trace_waterfall_empty_signals(self):
        from ts_agent.visualiser.components.trace_waterfall import build_trace_waterfall
        fig = build_trace_waterfall([])
        annotations = fig.layout.annotations
        assert len(annotations) > 0
        assert any("No signals" in str(a.text) for a in annotations)

    def test_sankey_flow_empty_data(self):
        from ts_agent.visualiser.components.sankey_flow import build_sankey
        fig = build_sankey({"labels": [], "links": []})
        annotations = fig.layout.annotations
        assert len(annotations) > 0
        assert any("Insufficient" in str(a.text) for a in annotations)

    def test_ml_panel_empty_predictions(self):
        from ts_agent.visualiser.components.ml_prediction import build_ml_panel
        fig = build_ml_panel([])
        annotations = fig.layout.annotations
        assert len(annotations) > 0
        assert any("No ML" in str(a.text) for a in annotations)

    def test_session_overview_renders_zero_completeness(self):
        from ts_agent.visualiser.components.session_overview import build_session_overview
        summary = {
            "session_id": "sess-empty",
            "party_ref": "PARTY-X",
            "situation_label": "Savings",
            "gate_disposition": "SUPPRESS",
            "audit_confirmed": False,
            "known_traits": 0,
            "missing_traits": 0,
            "excluded_traits": 0,
            "total_traits": 0,
            "completeness_pct": 0.0,
            "gap_fill_turns": 0,
            "fill_strategy": "STATIC",
            "signal_count": 0,
            "error_count": 0,
            "has_error": False,
            "is_complete": False,
            "started_at": "",
            "total_ms": 0.0,
            "duration_s": 0.0,
            "segment_label": "—",
        }
        fig = build_session_overview(summary)
        assert fig is not None
        assert len(fig.data) > 0

    def test_sankey_single_session_missing_segment(self):
        """Sankey with no matched segment should still produce 5 labels."""
        from ts_agent.observability.session_store import SessionRecord
        r = SessionRecord(
            session_id="sess-nomatch",
            situation_id="SIT-INV-001",
            intent_id="INTENT-INVEST-CASH",
            matched_segment_id="",   # no match
            gate_disposition="SUPPRESS",
        )
        data = DataAdapter.sankey_for_session(r)
        assert len(data["labels"]) == 5
        assert data["labels"][-1] == "SUPPRESS"
