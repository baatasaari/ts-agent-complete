"""
ts_agent.ml.predictor
=====================
ML Segment Predictor — Zone 1.5

Advisory only: output influences gap-fill question ordering, never the
final deterministic segment classification performed by Zone 2.

Design decisions
----------------
- ``OrdinalEncoder(unknown_value=-1)`` — integer, not float; sklearn 1.5+
  requires an integer when dtype is not float.
- SHAP is imported lazily; a no-op fallback is used when unavailable.
- Module-scope pytest fixtures should use ``scope="module"`` only if the
  pipeline build is guaranteed to succeed; we use function scope in tests
  for the iterative predictor and only module-scope for the fitted predictor.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from ts_agent.config.settings import settings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from ts_agent.domain.models import (
    GapFillStrategy,
    HypothesisDisposition,
    ModelAlgorithm,
    NodeState,
    SegmentHypothesis,
    SegmentRank,
    ShapFeature,
    TraitGraph,
)
from ts_agent.observability import signals as eamgp

logger = logging.getLogger(__name__)

# ── Feature column definitions ────────────────────────────────────────────────

NUMERIC_FEATURES: list[str] = [
    "age_band",              # CHAR-P1A-I1: age band 1–5
    # v2 financial features
    "savings_balance",       # CHAR-F2B-I1: cash savings / deposit value
    "monthly_surplus",       # CHAR-F2A-I1: monthly disposable income
    "risk_appetite_score",   # CHAR-B3A-I1: 1.0–5.0 scale
    "investment_experience", # CHAR-B3B-I1: 0=none, 1=basic, 2=intermediate, 3=experienced
    # v2 investment-specific features
    "account_tenure_months", # CHAR-F2L-I1: account tenure in months (INV-001 criterion)
    "lump_sum_amount",       # CHAR-F2M-I1: one-off capital event (INV-004)
    "regular_saving_amount", # CHAR-F2O-I1: regular monthly saving amount (INV-006)
    # v2 pension-specific features
    "pension_contribution_pct", # CHAR-P2A-I1: combined DC contribution rate %
    "pension_pot_value",        # CHAR-P2L-I1: DC pot value £
    "years_to_retirement",      # CHAR-P2D-I1: years to state pension age
    "months_since_review",      # CHAR-P2N-I1: months since last drawdown review
    # Shared
    "known_trait_count",
    "conversation_turn",
]
CATEGORICAL_FEATURES: list[str] = [
    "employment_status",   # CHAR-P1C-I1: 0=unemployed, 1=employed, 2=self-emp, 3=retired
    "channel",             # CHAR-B3C-I1: 0=mobile, 1=web, 2=IVR
]
ALL_FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Maps char_id → feature column name (v2 ontology)
CHAR_ID_TO_FEATURE: dict[str, str] = {
    # Personal
    "CHAR-P1A-I1": "age_band",
    "CHAR-P1C-I1": "employment_status",
    # Financial — investment domain
    "CHAR-F2A-I1": "monthly_surplus",
    "CHAR-F2B-I1": "savings_balance",
    "CHAR-F2L-I1": "account_tenure_months",
    "CHAR-F2M-I1": "lump_sum_amount",
    "CHAR-F2O-I1": "regular_saving_amount",
    # Behavioural
    "CHAR-B3A-I1": "risk_appetite_score",
    "CHAR-B3B-I1": "investment_experience",
    "CHAR-B3C-I1": "channel",
    # Pension — accumulation
    "CHAR-P2A-I1": "pension_contribution_pct",
    "CHAR-P2D-I1": "years_to_retirement",
    # Pension — decumulation
    "CHAR-P2L-I1": "pension_pot_value",
    "CHAR-P2N-I1": "months_since_review",
}

_SHAP_TOP_N: int = settings.shap_top_n
_UNIFORM_VARIANCE_THRESHOLD: float = 0.10  # internal; not tuned per-deployment

# ── Feature vector ────────────────────────────────────────────────────────────

@dataclass
class PredictorFeatureVector:
    """
    Input to the ML predictor, extracted from a TraitGraph at each turn.
    MISSING / EXCLUDED nodes map to ``None`` → NaN after DataFrame conversion.

    v2: Updated for PS25/22 ontology. Old savings/debt/insurance/mortgage
    features removed; pension and investment-specific features added.
    See CHAR_ID_TO_FEATURE for the full char_id → column mapping.
    """
    # Personal
    age_band:                float | None = None
    employment_status:       float | None = None
    # Financial — investment domain
    savings_balance:         float | None = None
    monthly_surplus:         float | None = None
    risk_appetite_score:     float | None = None
    investment_experience:   float | None = None
    account_tenure_months:   float | None = None
    lump_sum_amount:         float | None = None
    regular_saving_amount:   float | None = None
    # Pension — accumulation
    pension_contribution_pct: float | None = None
    years_to_retirement:      float | None = None
    # Pension — decumulation
    pension_pot_value:        float | None = None
    months_since_review:      float | None = None
    # Categorical
    channel:                 float | None = None
    # Session context
    known_trait_count:       int = 0
    conversation_turn:       int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame matching ALL_FEATURE_COLUMNS (v2)."""
        def _f(v):
            return float(v) if v is not None else float("nan")
        row = {
            "age_band":                 _f(self.age_band),
            "savings_balance":          _f(self.savings_balance),
            "monthly_surplus":          _f(self.monthly_surplus),
            "risk_appetite_score":      _f(self.risk_appetite_score),
            "investment_experience":    _f(self.investment_experience),
            "account_tenure_months":    _f(self.account_tenure_months),
            "lump_sum_amount":          _f(self.lump_sum_amount),
            "regular_saving_amount":    _f(self.regular_saving_amount),
            "pension_contribution_pct": _f(self.pension_contribution_pct),
            "years_to_retirement":      _f(self.years_to_retirement),
            "pension_pot_value":        _f(self.pension_pot_value),
            "months_since_review":      _f(self.months_since_review),
            "known_trait_count":        float(self.known_trait_count),
            "conversation_turn":        float(self.conversation_turn),
            "employment_status":        _f(self.employment_status),
            "channel":                  _f(self.channel),
        }
        return pd.DataFrame([row], columns=ALL_FEATURE_COLUMNS)

    @classmethod
    def from_graph(
        cls,
        graph: "TraitGraph",
        turn: int = 0,
    ) -> "PredictorFeatureVector":
        """
        Build a PredictorFeatureVector from a live TraitGraph.

        v2: Maps PS25/22 char_ids to the v2 feature vector columns.
        Only KNOWN nodes contribute values; MISSING/EXCLUDED → None → NaN.
        """
        def _get(char_id: str):
            n = graph.node_by_char_id(char_id)
            return n.value if (n is not None and n.state.value == "KNOWN") else None

        known_count = sum(
            1 for n in graph.nodes.values()
            if n.state.value == "KNOWN"
        )
        return cls(
            # Personal
            age_band=_get("CHAR-P1A-I1"),
            employment_status=_get("CHAR-P1C-I1"),
            # Financial — investment
            savings_balance=_get("CHAR-F2B-I1"),
            monthly_surplus=_get("CHAR-F2A-I1"),
            risk_appetite_score=_get("CHAR-B3A-I1"),
            investment_experience=_get("CHAR-B3B-I1"),
            account_tenure_months=_get("CHAR-F2L-I1"),
            lump_sum_amount=_get("CHAR-F2M-I1"),
            regular_saving_amount=_get("CHAR-F2O-I1"),
            # Pension
            pension_contribution_pct=_get("CHAR-P2A-I1"),
            years_to_retirement=_get("CHAR-P2D-I1"),
            pension_pot_value=_get("CHAR-P2L-I1"),
            months_since_review=_get("CHAR-P2N-I1"),
            # Behavioural
            channel=_get("CHAR-B3C-I1"),
            # Session context
            known_trait_count=known_count,
            conversation_turn=turn,
        )


# ── Pipeline factories ────────────────────────────────────────────────────────

def _build_preprocessor() -> ColumnTransformer:
    """Shared numeric + categorical preprocessing block."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale",  RobustScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    # unknown_value MUST be an integer for non-float dtype
                    ("encode", OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    )),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_logistic_regression_pipeline() -> Pipeline:
    """Option C — maximum FCA explainability.  Recommended for initial launch."""
    return Pipeline([
        ("prep", _build_preprocessor()),
        ("clf",  LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=0.5,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def build_random_forest_pipeline() -> Pipeline:
    """Option B — feature importances + SHAP."""
    return Pipeline([
        ("prep", _build_preprocessor()),
        ("clf",  CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight="balanced",
                random_state=42,
            ),
            method="isotonic",
            cv=3,
        )),
    ])


def build_gradient_boosting_pipeline() -> Pipeline:
    """Option A — highest accuracy."""
    return Pipeline([
        ("prep", _build_preprocessor()),
        ("clf",  CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            ),
            method="isotonic",
            cv=3,
        )),
    ])


PIPELINE_REGISTRY: dict[ModelAlgorithm, Any] = {
    ModelAlgorithm.LOGISTIC_REGRESSION: build_logistic_regression_pipeline,
    ModelAlgorithm.RANDOM_FOREST:       build_random_forest_pipeline,
    ModelAlgorithm.GRADIENT_BOOSTING:   build_gradient_boosting_pipeline,
}


# ── SegmentPredictor ──────────────────────────────────────────────────────────

class SegmentPredictorNotFittedError(RuntimeError):
    pass


class SegmentPredictor:
    """
    Wraps a fitted sklearn Pipeline; produces ``SegmentHypothesis`` objects.

    Call ``fit()`` before ``predict_with_explanation()``.
    """

    def __init__(
        self,
        algorithm: ModelAlgorithm = ModelAlgorithm.LOGISTIC_REGRESSION,
        model_version: str = "0.0.0",
    ) -> None:
        self.algorithm     = algorithm
        self.model_version = model_version
        self._pipeline: Pipeline | None = None
        self._classes:  list[str]       = []
        self._explainer: Any            = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SegmentPredictor":
        factory = PIPELINE_REGISTRY[self.algorithm]
        self._pipeline = factory()
        self._pipeline.fit(X, y)
        self._classes = list(self._pipeline.classes_)
        self._try_attach_shap(X)
        return self

    def _try_attach_shap(self, X_train: pd.DataFrame) -> None:
        try:
            import shap  # noqa: PLC0415
            clf = self._pipeline.named_steps["clf"]
            self._explainer = shap.Explainer(clf, X_train)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SHAP explainer skipped: %s", exc)

    def predict_with_explanation(
        self,
        fv: PredictorFeatureVector,
        session_id: str = "",
        turn: int = 0,
    ) -> SegmentHypothesis:
        if self._pipeline is None:
            raise SegmentPredictorNotFittedError(
                "Call SegmentPredictor.fit() before predict_with_explanation()"
            )
        X = fv.to_dataframe()
        probs: np.ndarray = self._pipeline.predict_proba(X)[0]
        ranked_segments = [
            SegmentRank(seg_id, float(prob))
            for seg_id, prob in sorted(
                zip(self._classes, probs.tolist()), key=lambda p: -p[1]
            )
        ]
        shap_features = self._compute_shap(X)
        return SegmentHypothesis(
            session_id=session_id,
            turn=turn,
            model_version=self.model_version,
            model_algorithm=self.algorithm,
            known_trait_count=fv.known_trait_count,
            ranked_segments=ranked_segments,
            shap_top_features=shap_features,
            disposition=HypothesisDisposition.ACTIVE,
        )

    def _compute_shap(self, X: pd.DataFrame) -> list[ShapFeature]:
        if self._explainer is None:
            return self._presence_fallback(X)
        try:
            sv = self._explainer(X)
            # multi-class → mean abs across classes
            vals = sv.values[0]
            if vals.ndim == 2:
                vals = np.abs(vals).mean(axis=1)
            pairs = sorted(
                zip(X.columns, np.abs(vals).tolist()),
                key=lambda p: -p[1],
            )
            return [
                ShapFeature(feature=name, shap_value=round(val, 6), rank=i + 1)
                for i, (name, val) in enumerate(pairs[:_SHAP_TOP_N])
            ]
        except Exception as exc:  # noqa: BLE001
            logger.debug("SHAP compute failed: %s", exc)
            return self._presence_fallback(X)

    @staticmethod
    def _presence_fallback(X: pd.DataFrame) -> list[ShapFeature]:
        return [
            ShapFeature(feature=col, shap_value=0.0, rank=i + 1)
            for i, col in enumerate(X.columns)
            if not X[col].isna().all()
        ][:_SHAP_TOP_N]

    def classes(self) -> list[str]:
        return list(self._classes)

    def is_fitted(self) -> bool:
        return self._pipeline is not None


# ── IterativeSegmentPredictor ─────────────────────────────────────────────────

DEFAULT_MIN_KNOWN_TRAITS    = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
CONVERGENCE_STABLE_TURNS    = 3


class IterativeSegmentPredictor:
    """
    Zone 1.5 orchestrator: predict → detect drift/convergence → reorder MISSING.

    Advisory only — does not affect Zone 2 deterministic classification.
    """

    def __init__(
        self,
        predictor: SegmentPredictor,
        ig_matrix: dict[str, dict[str, float]],
        min_known_traits: int = DEFAULT_MIN_KNOWN_TRAITS,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        convergence_stable_turns: int = CONVERGENCE_STABLE_TURNS,
    ) -> None:
        self._predictor            = predictor
        self._ig_matrix            = ig_matrix
        self._min_known_traits     = min_known_traits
        self._confidence_threshold = confidence_threshold
        self._convergence_turns    = convergence_stable_turns
        self._history: list[SegmentHypothesis] = []

    def predict_and_prioritise(
        self,
        graph: TraitGraph,
        session_id: str,
        turn: int,
    ) -> tuple[SegmentHypothesis, list[str], GapFillStrategy]:
        """
        Returns (hypothesis, ordered_missing_char_ids, strategy).
        Never raises — failures fall back to STATIC order.
        """
        eamgp.emit(
            "SEG_PREDICT_START", eamgp.INFO, "Zone1.5",
            session_id=session_id, turn=turn,
            known_trait_count=len(graph.known_nodes()),
        )
        known_count = len(graph.known_nodes())

        # ── Insufficient data ─────────────────────────────────────────────────
        if known_count < self._min_known_traits:
            hyp = self._undecidable_hyp(session_id, turn, known_count)
            eamgp.emit(
                "SEG_PREDICT_UNDECIDABLE", eamgp.WARN, "Zone1.5",
                session_id=session_id, turn=turn,
                known_trait_count=known_count,
                threshold=self._min_known_traits,
            )
            self._history.append(hyp)
            return hyp, self._static_order(graph), GapFillStrategy.STATIC_PRIORITY

        # ── Predict ───────────────────────────────────────────────────────────
        fv = PredictorFeatureVector.from_graph(graph, turn)
        try:
            hyp = self._predictor.predict_with_explanation(fv, session_id, turn)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Predictor error: %s", exc)
            eamgp.emit(
                "SEG_PREDICT_FAILED", eamgp.ERROR, "Zone1.5",
                session_id=session_id, turn=turn,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            failed = SegmentHypothesis(
                session_id=session_id, turn=turn,
                model_version=self._predictor.model_version,
                model_algorithm=self._predictor.algorithm,
                known_trait_count=known_count,
                disposition=HypothesisDisposition.FAILED,
            )
            self._history.append(failed)
            return failed, self._static_order(graph), GapFillStrategy.STATIC_PRIORITY

        # ── Low confidence ────────────────────────────────────────────────────
        if hyp.top_confidence < self._confidence_threshold:
            eamgp.emit(
                "SEG_PREDICT_LOW_CONFIDENCE", eamgp.WARN, "Zone1.5",
                session_id=session_id, turn=turn,
                top_segment_id=hyp.top_segment_id,
                confidence=hyp.top_confidence,
                threshold=self._confidence_threshold,
            )
            self._history.append(hyp)
            return hyp, self._static_order(graph), GapFillStrategy.STATIC_PRIORITY

        # ── Near-uniform distribution ─────────────────────────────────────────
        if self._is_uniform(hyp):
            eamgp.emit(
                "SEG_PREDICT_AMBIGUOUS", eamgp.WARN, "Zone1.5",
                session_id=session_id, turn=turn,
            )
            self._history.append(hyp)
            return hyp, self._entropy_order(graph), GapFillStrategy.ENTROPY_BROADENING

        # ── Drift / convergence ───────────────────────────────────────────────
        self._check_drift(hyp, session_id, turn)
        self._check_convergence(hyp, session_id, turn)

        # ── IG-ordered MISSING ────────────────────────────────────────────────
        ordered = self._ig_order(graph, hyp.top_segment_id)
        eamgp.emit(
            "SEG_GAP_REORDERED", eamgp.INFO, "Zone1.5",
            session_id=session_id, turn=turn,
            segment_id=hyp.top_segment_id,
            missing_traits_reordered_count=len(ordered),
        )
        eamgp.emit(
            "SEG_PREDICT_COMPLETE", eamgp.INFO, "Zone1.5",
            session_id=session_id, turn=turn,
            top_segment_id=hyp.top_segment_id,
            top_confidence=round(hyp.top_confidence, 4),
            model_version=hyp.model_version,
        )
        self._history.append(hyp)
        return hyp, ordered, GapFillStrategy.ML_INFORMATION_GAIN

    def reset(self) -> None:
        self._history.clear()

    # ── Ordering helpers ──────────────────────────────────────────────────────

    def _static_order(self, graph: TraitGraph) -> list[str]:
        return [
            n.char_id
            for n in sorted(graph.missing_nodes(), key=lambda n: n.fill_priority)
        ]

    def _ig_order(self, graph: TraitGraph, segment_id: str) -> list[str]:
        ig = self._ig_matrix.get(segment_id, {})
        return [
            n.char_id
            for n in sorted(
                graph.missing_nodes(),
                key=lambda n: ig.get(n.char_id, 0.0),
                reverse=True,
            )
        ]

    def _entropy_order(self, graph: TraitGraph) -> list[str]:
        if not self._ig_matrix:
            return self._static_order(graph)
        all_segs = list(self._ig_matrix.values())

        def mean_ig(char_id: str) -> float:
            vals = [s.get(char_id, 0.0) for s in all_segs]
            return sum(vals) / len(vals) if vals else 0.0

        return [
            n.char_id
            for n in sorted(
                graph.missing_nodes(),
                key=lambda n: mean_ig(n.char_id),
                reverse=True,
            )
        ]

    # ── Drift / convergence detection ─────────────────────────────────────────

    @staticmethod
    def _is_uniform(hyp: SegmentHypothesis) -> bool:
        if not hyp.ranked_segments:
            return True
        probs = [r.probability for r in hyp.ranked_segments]
        return (max(probs) - (sum(probs) / len(probs))) < _UNIFORM_VARIANCE_THRESHOLD

    def _check_drift(
        self, hyp: SegmentHypothesis, session_id: str, turn: int
    ) -> None:
        decisive = [
            h for h in reversed(self._history)
            if h.disposition not in (
                HypothesisDisposition.UNDECIDABLE,
                HypothesisDisposition.FAILED,
            )
        ]
        if decisive and decisive[0].top_segment_id != hyp.top_segment_id:
            eamgp.emit(
                "SEG_PREDICT_DRIFT", eamgp.WARN, "Zone1.5",
                session_id=session_id, turn=turn,
                prev_segment_id=decisive[0].top_segment_id,
                new_segment_id=hyp.top_segment_id,
            )

    def _check_convergence(
        self, hyp: SegmentHypothesis, session_id: str, turn: int
    ) -> None:
        decisive = [
            h for h in self._history
            if h.disposition not in (
                HypothesisDisposition.UNDECIDABLE,
                HypothesisDisposition.FAILED,
            )
        ]
        if len(decisive) >= self._convergence_turns - 1:
            recent = decisive[-(self._convergence_turns - 1):]
            if all(h.top_segment_id == hyp.top_segment_id for h in recent):
                eamgp.emit(
                    "SEG_PREDICT_CONVERGED", eamgp.INFO, "Zone1.5",
                    session_id=session_id, turn=turn,
                    segment_id=hyp.top_segment_id,
                    confidence=round(hyp.top_confidence, 4),
                    consecutive_stable_turns=len(recent) + 1,
                )

    # ── Private constructor helpers ───────────────────────────────────────────

    def _undecidable_hyp(
        self, session_id: str, turn: int, known_count: int
    ) -> SegmentHypothesis:
        return SegmentHypothesis(
            session_id=session_id, turn=turn,
            model_version=self._predictor.model_version,
            model_algorithm=self._predictor.algorithm,
            known_trait_count=known_count,
            disposition=HypothesisDisposition.UNDECIDABLE,
        )


# ── IG matrix utility ─────────────────────────────────────────────────────────

def compute_ig_matrix(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    segment_ids: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-segment information gain via mutual_info_classif (one-vs-rest)."""
    ig_matrix: dict[str, dict[str, float]] = {}
    for seg_id in segment_ids:
        y_binary = (y_train == seg_id).astype(int)
        scores = mutual_info_classif(
            X_train.fillna(-1.0),
            y_binary,
            discrete_features="auto",
            random_state=42,
        )
        ig_matrix[seg_id] = dict(zip(X_train.columns, scores.tolist()))
    return ig_matrix
