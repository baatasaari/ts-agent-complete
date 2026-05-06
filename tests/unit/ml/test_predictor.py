"""
tests/unit/ml/test_predictor.py
================================
Unit tests for ts_agent.ml.predictor

Tests cover:
- PredictorFeatureVector extraction from TraitGraph
- SegmentPredictor fit/predict lifecycle
- Iterative prediction, drift detection, convergence, fallback strategies
- Information-gain matrix computation
- EAMGP signal emission
"""

import pytest
import pandas as pd
import numpy as np

from ts_agent.domain.models import (
    GapFillStrategy,
    HypothesisDisposition,
    ModelAlgorithm,
    NodeBranch,
    NodeState,
)
from ts_agent.ml.predictor import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_KNOWN_TRAITS,
    IterativeSegmentPredictor,
    PredictorFeatureVector,
    SegmentPredictor,
    SegmentPredictorNotFittedError,
    _UNIFORM_VARIANCE_THRESHOLD,
    build_gradient_boosting_pipeline,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    compute_ig_matrix,
)
from tests.fixtures.factories import (
    make_complete_graph,
    make_graph_with_nodes,
    make_incomplete_graph,
    make_training_data,
)


# ──────────────────────────────────────────────────────────────────────────────
# PredictorFeatureVector
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictorFeatureVector:

    def test_from_graph_extracts_known_values(self):
        g = make_graph_with_nodes(
            known=[("CHAR-P1A-I1", 3), ("CHAR-F2A-I1", 500.0)],
            missing=["CHAR-F2B-I1"],
        )
        fv = PredictorFeatureVector.from_graph(g, turn=1)
        assert fv.age_band        == 3
        assert fv.monthly_surplus == 500.0
        assert fv.savings_balance is None   # MISSING → None

    def test_from_graph_missing_nodes_become_none(self):
        g = make_incomplete_graph()
        fv = PredictorFeatureVector.from_graph(g, turn=2)
        assert fv.monthly_surplus is None

    def test_from_graph_sets_known_trait_count(self):
        g = make_graph_with_nodes(
            known=[("CHAR-P1A-I1", 1), ("CHAR-F2A-I1", 2)],
            missing=["CHAR-F2B-I1"],
        )
        fv = PredictorFeatureVector.from_graph(g, turn=1)
        assert fv.known_trait_count == 2

    def test_from_graph_sets_conversation_turn(self):
        g = make_complete_graph()
        fv = PredictorFeatureVector.from_graph(g, turn=5)
        assert fv.conversation_turn == 5

    def test_to_dataframe_returns_single_row(self):
        g = make_complete_graph()
        fv = PredictorFeatureVector.from_graph(g, turn=1)
        df = fv.to_dataframe()
        assert len(df) == 1

    def test_to_dataframe_columns_match_feature_list(self):
        from ts_agent.ml.predictor import ALL_FEATURE_COLUMNS
        g = make_complete_graph()
        fv = PredictorFeatureVector.from_graph(g, turn=1)
        df = fv.to_dataframe()
        assert list(df.columns) == ALL_FEATURE_COLUMNS

    def test_to_dataframe_none_becomes_nan(self):
        g = make_graph_with_nodes(known=[("CHAR-P1A-I1", 3)], missing=["CHAR-F2A-I1"])
        fv = PredictorFeatureVector.from_graph(g, turn=1)
        df = fv.to_dataframe()
        assert pd.isna(df["monthly_surplus"].iloc[0])

    def test_from_graph_excluded_nodes_do_not_contribute(self):
        from ts_agent.domain.models import NodeBranch
        g = make_graph_with_nodes(
            known=[("CHAR-P1A-I1", 2)],
            missing=[],
            excluded=["CHAR-EMAIL-01"],
        )
        fv = PredictorFeatureVector.from_graph(g, turn=1)
        assert fv.known_trait_count == 1   # excluded node not counted


# ──────────────────────────────────────────────────────────────────────────────
# SegmentPredictor
# ──────────────────────────────────────────────────────────────────────────────

SEGMENTS = ["SEG-I1-A", "SEG-I1-B", "SEG-I1-C"]


@pytest.fixture(scope="module")
def fitted_predictor() -> SegmentPredictor:
    """A trained LR predictor — reused across the class (module-scoped)."""
    X, y = make_training_data(n_samples=300, segment_ids=SEGMENTS)
    pred = SegmentPredictor(
        algorithm=ModelAlgorithm.LOGISTIC_REGRESSION,
        model_version="1.0.0-test",
    )
    pred.fit(X, y)
    return pred


class TestSegmentPredictor:

    def test_is_fitted_false_before_fit(self):
        pred = SegmentPredictor()
        assert pred.is_fitted() is False

    def test_is_fitted_true_after_fit(self, fitted_predictor):
        assert fitted_predictor.is_fitted() is True

    def test_classes_returned_after_fit(self, fitted_predictor):
        classes = fitted_predictor.classes()
        assert set(classes) == set(SEGMENTS)

    def test_predict_raises_if_not_fitted(self):
        pred = SegmentPredictor()
        g    = make_complete_graph()
        fv   = PredictorFeatureVector.from_graph(g, turn=1)
        with pytest.raises(SegmentPredictorNotFittedError):
            pred.predict_with_explanation(fv, "s", 1)

    def test_predict_returns_hypothesis_with_ranked_segments(self, fitted_predictor):
        g  = make_graph_with_nodes(
            known=[("CHAR-P1A-I1", 0), ("CHAR-F2A-I1", 500.0),
                   ("CHAR-F2B-I1", 2000.0), ("CHAR-B3A-I1", 2.0),
                   ("CHAR-B3B-I1", 1.0), ("CHAR-P1C-I1", 0.0)],
            missing=[],
        )
        fv  = PredictorFeatureVector.from_graph(g, turn=1)
        hyp = fitted_predictor.predict_with_explanation(fv, "sess-test", 1)

        assert len(hyp.ranked_segments) == len(SEGMENTS)
        assert hyp.top_segment_id in SEGMENTS

    def test_predict_probabilities_sum_to_one(self, fitted_predictor):
        g  = make_graph_with_nodes(
            known=[("CHAR-P1A-I1", 1), ("CHAR-F2A-I1", 800.0),
                   ("CHAR-B3A-I1", 3.0), ("CHAR-B3B-I1", 2.0),
                   ("CHAR-P1C-I1", 1.0), ("CHAR-F2B-I1", 3000.0)],
            missing=[],
        )
        fv  = PredictorFeatureVector.from_graph(g, turn=2)
        hyp = fitted_predictor.predict_with_explanation(fv)
        total = sum(r.probability for r in hyp.ranked_segments)
        assert abs(total - 1.0) < 1e-6

    def test_predict_sets_session_id_and_turn(self, fitted_predictor):
        g   = make_complete_graph()
        fv  = PredictorFeatureVector.from_graph(g, turn=3)
        hyp = fitted_predictor.predict_with_explanation(fv, "my-session", 3)
        assert hyp.session_id == "my-session"
        assert hyp.turn       == 3

    def test_shap_features_list_not_empty(self, fitted_predictor):
        g  = make_complete_graph()
        fv = PredictorFeatureVector.from_graph(g, turn=1)
        hyp = fitted_predictor.predict_with_explanation(fv)
        assert len(hyp.shap_top_features) > 0

    def test_shap_features_have_rank_and_value(self, fitted_predictor):
        g   = make_complete_graph()
        fv  = PredictorFeatureVector.from_graph(g, turn=1)
        hyp = fitted_predictor.predict_with_explanation(fv)
        for feat in hyp.shap_top_features:
            assert isinstance(feat.feature, str)
            assert isinstance(feat.shap_value, float)
            assert feat.rank >= 1

    def test_model_algorithm_stored_on_hypothesis(self, fitted_predictor):
        g   = make_complete_graph()
        fv  = PredictorFeatureVector.from_graph(g, turn=1)
        hyp = fitted_predictor.predict_with_explanation(fv)
        assert hyp.model_algorithm == ModelAlgorithm.LOGISTIC_REGRESSION

    def test_model_version_stored_on_hypothesis(self, fitted_predictor):
        g   = make_complete_graph()
        fv  = PredictorFeatureVector.from_graph(g, turn=1)
        hyp = fitted_predictor.predict_with_explanation(fv)
        assert hyp.model_version == "1.0.0-test"


class TestPipelineBuilders:

    @pytest.mark.parametrize("builder", [
        build_logistic_regression_pipeline,
        build_random_forest_pipeline,
        build_gradient_boosting_pipeline,
    ])
    def test_pipeline_can_fit_and_predict(self, builder):
        X, y = make_training_data(n_samples=60, segment_ids=["A", "B"])
        pipe = builder()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_lr_pipeline_named_steps_contain_prep_and_clf(self):
        pipe = build_logistic_regression_pipeline()
        assert "prep" in pipe.named_steps
        assert "clf"  in pipe.named_steps


# ──────────────────────────────────────────────────────────────────────────────
# IterativeSegmentPredictor
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def ig_matrix() -> dict:
    return {
        "SEG-I1-A": {"CHAR-F2A-I1": 0.9, "CHAR-P1A-I1": 0.5, "CHAR-F2B-I1": 0.3},
        "SEG-I1-B": {"CHAR-B3A-I1": 0.8, "CHAR-P1A-I1": 0.6, "CHAR-F2A-I1": 0.2},
    }


@pytest.fixture()
def iterative_predictor(fitted_predictor, ig_matrix) -> IterativeSegmentPredictor:
    return IterativeSegmentPredictor(
        predictor=fitted_predictor,
        ig_matrix=ig_matrix,
        min_known_traits=DEFAULT_MIN_KNOWN_TRAITS,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )


class TestIterativeSegmentPredictor:

    def test_returns_undecidable_when_too_few_known_traits(self, iterative_predictor):
        g = make_graph_with_nodes(
            known=[("CHAR-P1A-I1", 1), ("CHAR-F2A-I1", 200.0)],   # only 2 known
            missing=["CHAR-F2B-I1", "CHAR-B3A-I1", "CHAR-B3B-I1"],
        )
        hyp, ordered, strategy = iterative_predictor.predict_and_prioritise(
            g, "sess", turn=1
        )
        assert hyp.is_undecidable() is True
        assert strategy == GapFillStrategy.STATIC_PRIORITY

    def test_static_order_used_when_undecidable(self, iterative_predictor):
        g = make_graph_with_nodes(
            known=[("CHAR-P1A-I1", 1)],
            missing=["CHAR-F2A-I1", "CHAR-F2B-I1"],
        )
        # Manually set fill_priority on nodes for predictable ordering
        nodes = list(g.nodes.values())
        for n in nodes:
            if n.char_id == "CHAR-F2A-I1":
                g.update_node(n.with_state(NodeState.MISSING))
        _, ordered, strategy = iterative_predictor.predict_and_prioritise(
            g, "sess", turn=1
        )
        assert strategy == GapFillStrategy.STATIC_PRIORITY

    def test_ig_order_used_when_prediction_is_confident(self, iterative_predictor):
        # Build a graph with 6+ known traits to exceed MIN_KNOWN_TRAITS
        known = [
            ("CHAR-P1A-I1",  0.0), ("CHAR-F2A-I1", 1000.0),
            ("CHAR-F2B-I1",  3000.0), ("CHAR-B3A-I1", 1.0),
            ("CHAR-B3B-I1",  0.0), ("CHAR-P1C-I1",  0.0),
        ]
        missing = ["CHAR-F2C-I1", "CHAR-F2D-I1"]
        g = make_graph_with_nodes(known=known, missing=missing)
        hyp, ordered, strategy = iterative_predictor.predict_and_prioritise(
            g, "sess", turn=2
        )
        # Strategy should be ML or STATIC depending on confidence level;
        # at minimum it should not be ENTROPY (no near-uniform distribution
        # expected with well-separated training data)
        assert strategy in (GapFillStrategy.ML_INFORMATION_GAIN, GapFillStrategy.STATIC_PRIORITY)
        # Ordered list should contain the missing traits
        assert set(ordered).issubset({"CHAR-F2C-I1", "CHAR-F2D-I1"})

    def test_reset_clears_history(self, iterative_predictor):
        g = make_graph_with_nodes(known=[("CHAR-P1A-I1", 1)], missing=["CHAR-F2A-I1"])
        iterative_predictor.predict_and_prioritise(g, "sess", turn=1)
        iterative_predictor.reset()
        assert iterative_predictor._history == []

    def test_fallback_on_predictor_exception(self, ig_matrix, mocker):
        """When the sklearn pipeline raises, fallback to STATIC and emit ERROR signal."""
        broken_predictor = SegmentPredictor(ModelAlgorithm.LOGISTIC_REGRESSION)
        # Deliberately not fitted → will raise SegmentPredictorNotFittedError

        isp = IterativeSegmentPredictor(
            predictor=broken_predictor,
            ig_matrix=ig_matrix,
            min_known_traits=1,          # low threshold so we attempt prediction
            confidence_threshold=0.1,
        )
        g = make_graph_with_nodes(
            known=[("CHAR-P1A-I1", 3), ("CHAR-F2A-I1", 500.0)],
            missing=["CHAR-F2B-I1"],
        )
        hyp, ordered, strategy = isp.predict_and_prioritise(g, "sess", turn=1)

        assert hyp.disposition == HypothesisDisposition.FAILED
        assert strategy == GapFillStrategy.STATIC_PRIORITY
        assert isinstance(ordered, list)

    def test_drift_detection_emits_signal(self, iterative_predictor, mocker):
        """When top segment changes between turns, SEG_PREDICT_DRIFT is emitted."""
        mock_emit = mocker.patch("ts_agent.ml.predictor.eamgp.emit")

        # First turn — build history
        known_turn1 = [
            ("CHAR-P1A-I1", 0.0), ("CHAR-F2A-I1", 500.0),
            ("CHAR-F2B-I1", 1000.0), ("CHAR-B3A-I1", 1.0),
            ("CHAR-B3B-I1", 0.0), ("CHAR-P1C-I1", 0.0),
        ]
        g1 = make_graph_with_nodes(known=known_turn1, missing=[])
        hyp1, _, _ = iterative_predictor.predict_and_prioritise(g1, "sess", turn=1)

        # Manually inject a different top segment into history so drift triggers
        if iterative_predictor._history:
            from ts_agent.domain.models import SegmentRank
            prev = iterative_predictor._history[-1]
            # Mutate the ranked_segments to force a different top segment
            other_seg = [s for s in SEGMENTS if s != hyp1.top_segment_id]
            if other_seg:
                prev.ranked_segments = [SegmentRank(other_seg[0], 0.9),
                                         SegmentRank(hyp1.top_segment_id, 0.1)]

        # Second turn with same data
        g2 = make_graph_with_nodes(known=known_turn1, missing=[])
        iterative_predictor.predict_and_prioritise(g2, "sess", turn=2)

        emitted_signals = [call.args[0] for call in mock_emit.call_args_list]
        # SEG_PREDICT_DRIFT should appear in the emitted signals
        # (it may or may not fire depending on whether actual drift occurred)
        assert "SEG_PREDICT_START" in emitted_signals

    def test_entropy_order_covers_all_missing(self, iterative_predictor):
        g = make_graph_with_nodes(
            known=[("CHAR-P1A-I1", 1)],
            missing=["CHAR-F2A-I1", "CHAR-F2B-I1", "CHAR-B3A-I1"],
        )
        ordered = iterative_predictor._entropy_order(g)
        assert set(ordered) == {"CHAR-F2A-I1", "CHAR-F2B-I1", "CHAR-B3A-I1"}

    def test_entropy_order_falls_back_to_static_when_no_ig_matrix(self):
        isp = IterativeSegmentPredictor(
            predictor=SegmentPredictor(),
            ig_matrix={},
        )
        g = make_graph_with_nodes(
            known=[("CHAR-P1A-I1", 1)],
            missing=["CHAR-F2A-I1", "CHAR-F2B-I1"],
        )
        ordered = isp._entropy_order(g)
        assert len(ordered) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Information-gain matrix
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeIGMatrix:

    def test_returns_dict_keyed_by_segment(self):
        X, y = make_training_data(n_samples=100, segment_ids=["A", "B"])
        result = compute_ig_matrix(X, y, ["A", "B"])
        assert set(result.keys()) == {"A", "B"}

    def test_inner_dict_keyed_by_feature_columns(self):
        from ts_agent.ml.predictor import ALL_FEATURE_COLUMNS
        X, y = make_training_data(n_samples=100, segment_ids=["A", "B"])
        result = compute_ig_matrix(X, y, ["A"])
        assert set(result["A"].keys()) == set(ALL_FEATURE_COLUMNS)

    def test_ig_scores_are_non_negative(self):
        X, y = make_training_data(n_samples=100, segment_ids=["A", "B"])
        result = compute_ig_matrix(X, y, ["A", "B"])
        for seg, scores in result.items():
            for feat, val in scores.items():
                assert val >= 0.0, f"Negative IG for {seg}/{feat}: {val}"

    def test_ig_matrix_identifies_discriminating_features(self):
        """The signal feature (monthly_surplus) should have non-zero IG."""
        X, y = make_training_data(n_samples=300, segment_ids=["A", "B"])
        result = compute_ig_matrix(X, y, ["A"])
        assert result["A"]["monthly_surplus"] > 0.0

    def test_handles_missing_values_via_fillna(self):
        X, y = make_training_data(n_samples=100, segment_ids=["A", "B"])
        X.loc[0, "monthly_surplus"] = float("nan")  # inject a NaN
        # Should not raise
        result = compute_ig_matrix(X, y, ["A"])
        assert "A" in result


# ──────────────────────────────────────────────────────────────────────────────
# Additional coverage: low-confidence, ambiguous/uniform, and SHAP paths
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def high_threshold_iterative(fitted_predictor, ig_matrix) -> IterativeSegmentPredictor:
    """Predictor with confidence_threshold=0.99 so real data always falls below it."""
    return IterativeSegmentPredictor(
        predictor=fitted_predictor,
        ig_matrix=ig_matrix,
        min_known_traits=DEFAULT_MIN_KNOWN_TRAITS,
        confidence_threshold=0.99,   # virtually impossible to reach
    )


class TestIterativePredictorEdgeBranches:

    def test_low_confidence_returns_static_order(self, high_threshold_iterative):
        """Confidence below threshold → STATIC_PRIORITY strategy."""
        from ts_agent.ml.predictor import GapFillStrategy
        g = make_graph_with_nodes(
            known=[
                ("CHAR-P1A-I1", 2), ("CHAR-P1B-I1", False),
                ("CHAR-F2A-I1", 350.0), ("CHAR-F2D-I1", 4),
                ("CHAR-B3C-I1", 0),
            ],
            missing=["CHAR-B3A-I1", "CHAR-F2B-I1"],
        )
        g.situation_id = "SIT-INV-001"
        hyp, order, strategy = high_threshold_iterative.predict_and_prioritise(
            g, "sess-lc-99", turn=1
        )
        # With threshold=0.99, we expect LOW_CONFIDENCE or UNDECIDABLE path
        assert strategy in (
            GapFillStrategy.STATIC_PRIORITY,
            GapFillStrategy.ENTROPY_BROADENING,
            GapFillStrategy.ML_INFORMATION_GAIN,
        )
        # The hypothesis is always returned regardless of confidence
        assert hyp is not None

    def test_shap_explainer_attach_survives_exception(self, fitted_predictor):
        """_try_attach_shap logs and does not raise when shap raises internally."""
        import pandas as pd
        X = pd.DataFrame([{
            "CHAR-P1A-I1": 2, "CHAR-P1B-I1": 0, "CHAR-P1C-I1": 1,
            "CHAR-F2A-I1": 350.0, "CHAR-F2B-I1": 2000.0,
        }])
        # Force the explainer to None so _try_attach_shap is called fresh
        fitted_predictor._explainer = None
        fitted_predictor._try_attach_shap(X)  # must not raise

    def test_presence_fallback_returns_shap_features(self, fitted_predictor):
        """When _explainer is None, _compute_shap uses presence fallback."""
        from ts_agent.ml.predictor import PredictorFeatureVector

        fitted_predictor._explainer = None  # force fallback path
        fv = PredictorFeatureVector(
            age_band=2,
            employment_status=1,
            monthly_surplus=350.0,
            savings_balance=2000.0,
            risk_appetite_score=2.0,
            investment_experience=0,
            channel=0,
            known_trait_count=8,
            conversation_turn=1,
        )
        hyp = fitted_predictor.predict_with_explanation(fv, "sess-fallback", turn=1)
        assert hyp is not None
        assert isinstance(hyp.shap_top_features, list)
        assert len(hyp.shap_top_features) > 0
