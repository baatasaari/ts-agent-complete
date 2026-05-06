"""
tests/unit/explainability/test_explainer.py
===========================================
Unit tests for ts_agent.explainability.explainer

Tests cover:
- Consumer explanation rendering (suggestion and no-suggestion paths)
- INV-06: output is always from an approved template, never an LLM
- Symbolic trace construction (INV-10)
- ExplainabilityBundle population helpers
- SHA-256 communication hash
- EAMGP signal emission
"""

import hashlib
import json

import pytest

from ts_agent.domain.models import (
    ExplainabilityBundle,
    GateDisposition,
    ModelAlgorithm,
    RuleEvaluation,
    RuleRejection,
    RuleType,
)
from ts_agent.explainability.explainer import (
    CONSUMER_REASON_MAP,
    ConsumerExplainer,
    SuggestionContext,
    build_symbolic_trace,
    populate_zone1_fields,
    populate_zone15_fields,
    populate_zone3_fields,
)
from tests.fixtures.factories import (
    make_complete_graph,
    make_hypothesis,
    make_incomplete_graph,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bundle(session_id: str = "sess-001") -> ExplainabilityBundle:
    return ExplainabilityBundle(session_id=session_id)


def _suggestion_ctx() -> SuggestionContext:
    return SuggestionContext(
        suggestion_name="LBG Easy Saver ISA",
        characteristic_descriptions=[
            "regular monthly savings between £50 and £500",
            "not currently holding a Cash ISA",
        ],
        advisor_url="https://www.lloydsbank.com/financial-advice",
        fca_firm_ref="FRN-119278",
    )


def _rule_eval(
    rule_id: str = "R-001",
    outcome: str = "PASS",
    rule_type: RuleType = RuleType.HARD,
) -> RuleEvaluation:
    return RuleEvaluation(
        rule_id=rule_id,
        rule_type=rule_type,
        input_value="SEG-I1-A",
        expected_value="SEG-I1-A",
        operator="==",
        outcome=outcome,
        consumer_reason=CONSUMER_REASON_MAP.get(rule_id),
    )


# ──────────────────────────────────────────────────────────────────────────────
# ConsumerExplainer — suggestion path
# ──────────────────────────────────────────────────────────────────────────────

class TestConsumerExplainerSuggestion:

    @pytest.fixture()
    def explainer(self) -> ConsumerExplainer:
        return ConsumerExplainer(
            advisor_url="https://advisor.test",
            fca_firm_ref="FRN-999",
        )

    def test_suggestion_text_contains_product_name(self, explainer):
        bundle = _bundle()
        ctx    = _suggestion_ctx()
        text   = explainer.explain_suggestion(bundle, ctx)
        assert "LBG Easy Saver ISA" in text

    def test_suggestion_text_contains_characteristics(self, explainer):
        bundle = _bundle()
        ctx    = _suggestion_ctx()
        text   = explainer.explain_suggestion(bundle, ctx)
        assert "regular monthly savings" in text

    def test_suggestion_text_contains_audit_id(self, explainer):
        bundle = _bundle()
        text   = explainer.explain_suggestion(bundle, _suggestion_ctx())
        assert bundle.audit_id in text

    def test_suggestion_text_contains_not_personal_advice_disclaimer(self, explainer):
        text = explainer.explain_suggestion(_bundle(), _suggestion_ctx())
        assert "not personalised financial advice" in text

    def test_suggestion_text_contains_advisor_url(self, explainer):
        bundle = _bundle()
        ctx    = _suggestion_ctx()
        text   = explainer.explain_suggestion(bundle, ctx)
        assert "https://www.lloydsbank.com/financial-advice" in text

    def test_suggestion_text_is_plain_string(self, explainer):
        text = explainer.explain_suggestion(_bundle(), _suggestion_ctx())
        assert isinstance(text, str)
        assert len(text) > 0

    def test_suggestion_text_does_not_contain_raw_rule_expressions(self, explainer):
        text = explainer.explain_suggestion(_bundle(), _suggestion_ctx())
        # Must not expose internal identifiers like "R-001", "SEG-I1-A", etc.
        for internal in ["R-001", "R-002", "SEG-I1-A", "HARD", "rule_id"]:
            assert internal not in text, f"Internal term {internal!r} leaked into consumer text"


# ──────────────────────────────────────────────────────────────────────────────
# ConsumerExplainer — no-suggestion path
# ──────────────────────────────────────────────────────────────────────────────

class TestConsumerExplainerNoSuggestion:

    @pytest.fixture()
    def explainer(self) -> ConsumerExplainer:
        return ConsumerExplainer()

    def test_no_suggestion_text_is_string(self, explainer):
        text = explainer.explain_no_suggestion(_bundle(), [])
        assert isinstance(text, str)

    def test_no_suggestion_text_contains_audit_id(self, explainer):
        bundle = _bundle()
        text   = explainer.explain_no_suggestion(bundle, [])
        assert bundle.audit_id in text

    def test_no_suggestion_text_omits_top_reason_when_no_rejections(self, explainer):
        text = explainer.explain_no_suggestion(_bundle(), [])
        assert "The primary reason" not in text

    def test_no_suggestion_text_includes_safe_reason_from_map(self, explainer):
        rejection = RuleRejection(
            suggestion_id="SUGG-001",
            rule_evaluation=_rule_eval(rule_id="R-002", outcome="FAIL"),
        )
        text = explainer.explain_no_suggestion(_bundle(), [rejection])
        assert "minimum age requirement" in text   # CONSUMER_REASON_MAP["R-002"]

    def test_no_suggestion_reason_uses_fallback_for_unknown_rule(self, explainer):
        """Unknown rule_id → generic fallback text, no exception."""
        rejection = RuleRejection(
            suggestion_id="SUGG-001",
            rule_evaluation=_rule_eval(rule_id="R-999", outcome="FAIL"),
        )
        text = explainer.explain_no_suggestion(_bundle(), [rejection])
        assert "current financial profile" in text

    def test_consumer_reason_map_covers_all_twelve_rules(self):
        for i in range(1, 13):
            rule_id = f"R-{i:03d}"
            assert rule_id in CONSUMER_REASON_MAP, f"{rule_id} missing from CONSUMER_REASON_MAP"

    def test_no_suggestion_does_not_expose_internal_ids(self, explainer):
        rejection = RuleRejection(
            suggestion_id="SUGG-001",
            rule_evaluation=_rule_eval(rule_id="R-005", outcome="FAIL"),
        )
        text = explainer.explain_no_suggestion(_bundle(), [rejection])
        for internal in ["R-005", "HARD", "GATE", "SOFT", "segment_id"]:
            assert internal not in text


# ──────────────────────────────────────────────────────────────────────────────
# Communication text hash
# ──────────────────────────────────────────────────────────────────────────────

class TestCommunicationHash:

    def test_hash_is_sha256_hex(self):
        h = ConsumerExplainer.hash_communication_text("hello")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_is_deterministic(self):
        h1 = ConsumerExplainer.hash_communication_text("same text")
        h2 = ConsumerExplainer.hash_communication_text("same text")
        assert h1 == h2

    def test_hash_differs_for_different_texts(self):
        h1 = ConsumerExplainer.hash_communication_text("text A")
        h2 = ConsumerExplainer.hash_communication_text("text B")
        assert h1 != h2

    def test_hash_matches_manual_sha256(self):
        text = "The quick brown fox"
        expected = hashlib.sha256(text.encode()).hexdigest()
        assert ConsumerExplainer.hash_communication_text(text) == expected


# ──────────────────────────────────────────────────────────────────────────────
# Symbolic trace (INV-10)
# ──────────────────────────────────────────────────────────────────────────────

class TestSymbolicTrace:

    def test_trace_contains_one_entry_per_evaluation(self):
        evals = [_rule_eval("R-001"), _rule_eval("R-002", "FAIL")]
        trace = build_symbolic_trace(evals)
        assert len(trace) == 2

    def test_trace_entry_has_required_keys(self):
        trace = build_symbolic_trace([_rule_eval()])
        entry = trace[0]
        for key in ("rule_id", "rule_type", "input_value",
                    "expected_value", "operator", "outcome"):
            assert key in entry

    def test_trace_preserves_rule_id(self):
        trace = build_symbolic_trace([_rule_eval("R-007")])
        assert trace[0]["rule_id"] == "R-007"

    def test_trace_preserves_outcome(self):
        trace = build_symbolic_trace([_rule_eval(outcome="GATE")])
        assert trace[0]["outcome"] == "GATE"

    def test_trace_is_json_serialisable(self):
        evals = [_rule_eval("R-001"), _rule_eval("R-005", "FAIL")]
        trace = build_symbolic_trace(evals)
        # Must not raise — Cloud Spanner stores as JSON
        serialised = json.dumps(trace)
        assert isinstance(serialised, str)

    def test_empty_evaluations_returns_empty_trace(self):
        assert build_symbolic_trace([]) == []


# ──────────────────────────────────────────────────────────────────────────────
# Bundle population helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestBundlePopulationHelpers:

    def test_populate_zone1_sets_known_traits(self):
        graph  = make_complete_graph()
        bundle = _bundle()
        populate_zone1_fields(bundle, graph, latency_ms=42)
        assert len(bundle.known_traits) == len(graph.known_nodes())

    def test_populate_zone1_known_trait_contains_char_id(self):
        graph  = make_complete_graph()
        bundle = _bundle()
        populate_zone1_fields(bundle, graph, latency_ms=0)
        char_ids = [t["char_id"] for t in bundle.known_traits]
        assert "CHAR-P1A-I1" in char_ids

    def test_populate_zone1_known_trait_does_not_expose_raw_value(self):
        """Only value_hash must appear in the audit bundle, not the raw value."""
        graph  = make_complete_graph()
        bundle = _bundle()
        populate_zone1_fields(bundle, graph, latency_ms=0)
        for trait in bundle.known_traits:
            assert "value" not in trait or trait.get("value") is None
            assert "value_hash" in trait

    def test_populate_zone1_sets_excluded_traits(self):
        graph  = make_complete_graph()
        bundle = _bundle()
        populate_zone1_fields(bundle, graph, latency_ms=0)
        assert len(bundle.excluded_traits) == len(graph.excluded_nodes())

    def test_populate_zone1_sets_missing_traits(self):
        graph  = make_incomplete_graph()
        bundle = _bundle()
        populate_zone1_fields(bundle, graph, latency_ms=0)
        assert "CHAR-F2A-I1" in bundle.missing_traits

    def test_populate_zone1_sets_latency(self):
        graph  = make_complete_graph()
        bundle = _bundle()
        populate_zone1_fields(bundle, graph, latency_ms=123)
        assert bundle.zone1_latency_ms == 123

    def test_populate_zone15_sets_model_version(self):
        hyp    = make_hypothesis()
        bundle = _bundle()
        populate_zone15_fields(bundle, hyp)
        assert bundle.model_version == hyp.model_version

    def test_populate_zone15_sets_shap_values(self):
        hyp    = make_hypothesis()
        bundle = _bundle()
        populate_zone15_fields(bundle, hyp)
        assert len(bundle.shap_values) == len(hyp.shap_top_features)

    def test_populate_zone3_sets_symbolic_trace(self):
        evals  = [_rule_eval("R-001"), _rule_eval("R-002", "PASS")]
        bundle = _bundle()
        populate_zone3_fields(
            bundle, evals,
            GateDisposition.EMIT, [], [], [],
        )
        assert len(bundle.symbolic_trace) == 2

    def test_populate_zone3_sets_gate_disposition(self):
        bundle = _bundle()
        populate_zone3_fields(
            bundle, [],
            GateDisposition.HUMAN_REVIEW, [], [], [],
        )
        assert bundle.gate_disposition == GateDisposition.HUMAN_REVIEW

    def test_populate_zone3_rejected_candidates_use_safe_reason(self):
        rejection = RuleRejection(
            suggestion_id="SUGG-001",
            rule_evaluation=_rule_eval("R-003", "GATE", RuleType.GATE),
        )
        bundle = _bundle()
        populate_zone3_fields(
            bundle, [_rule_eval("R-003", "GATE")],
            GateDisposition.HUMAN_REVIEW,
            [], [], [rejection],
        )
        assert len(bundle.rejected_candidates) == 1
        rc = bundle.rejected_candidates[0]
        # Must use consumer-safe reason, not raw rule expression
        assert "specialist" in rc["consumer_reason"]   # R-003 → specialist journey


# ──────────────────────────────────────────────────────────────────────────────
# EAMGP signals
# ──────────────────────────────────────────────────────────────────────────────

class TestExplainerSignals:

    def test_consumer_explain_served_emitted_on_suggestion(self, mocker):
        mock_emit = mocker.patch("ts_agent.explainability.explainer.eamgp.emit")
        ConsumerExplainer().explain_suggestion(_bundle(), _suggestion_ctx())
        signals = [c.args[0] for c in mock_emit.call_args_list]
        assert "CONSUMER_EXPLAIN_SERVED" in signals

    def test_consumer_explain_served_emitted_on_no_suggestion(self, mocker):
        mock_emit = mocker.patch("ts_agent.explainability.explainer.eamgp.emit")
        ConsumerExplainer().explain_no_suggestion(_bundle(), [])
        signals = [c.args[0] for c in mock_emit.call_args_list]
        assert "CONSUMER_EXPLAIN_SERVED" in signals
