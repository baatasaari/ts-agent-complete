"""
tests/fixtures/factories.py
============================
Centralised factories for all domain objects used across the test suite.

All factory functions take explicit keyword arguments for the fields
that tests care about, defaulting everything else to valid values.
This makes test intent clear without repeating boilerplate.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ts_agent.domain.models import (
    GapFillStrategy,
    HypothesisDisposition,
    ModelAlgorithm,
    NodeBranch,
    NodeState,
    SegmentHypothesis,
    SegmentRank,
    ShapFeature,
    TraitGraph,
    TraitNode,
)
from ts_agent.zones.zone1_graph_builder import (
    ClassifiedIntent,
    ConflictPair,
    SituationConfig,
    TraitConfig,
    UserProfile,
)


# ── TraitNode factories ───────────────────────────────────────────────────────

def make_trait_node(
    char_id: str = "CHAR-P1A-I1",
    branch: NodeBranch = NodeBranch.PERSONAL,
    state: NodeState = NodeState.MISSING,
    fill_priority: int = 1,
    value: Any = None,
    node_id: str | None = None,
    email_test_node: bool = False,
    **kwargs: Any,
) -> TraitNode:
    return TraitNode(
        node_id=node_id or f"sess::{char_id}",
        char_id=char_id,
        branch=branch,
        label=f"Label for {char_id}",
        op="==",
        target_value=True,
        data_sources=("OCIS",),
        aging="90d",
        fill_priority=fill_priority,
        email_test_node=email_test_node,
        state=state,
        value=value,
        **kwargs,
    )


def make_known_node(
    char_id: str = "CHAR-P1A-I1",
    value: Any = 3,
    fill_priority: int = 1,
    branch: NodeBranch = NodeBranch.PERSONAL,
    **kw: Any,
) -> TraitNode:
    return make_trait_node(
        char_id=char_id, state=NodeState.KNOWN,
        value=value, fill_priority=fill_priority, branch=branch, **kw,
    )


def make_missing_node(
    char_id: str = "CHAR-F2A-I1",
    fill_priority: int = 2,
    branch: NodeBranch = NodeBranch.FINANCIAL,
    **kw: Any,
) -> TraitNode:
    return make_trait_node(
        char_id=char_id, state=NodeState.MISSING,
        fill_priority=fill_priority, branch=branch, **kw,
    )


def make_excluded_node(
    char_id: str = "CHAR-EMAIL-01",
    **kw: Any,
) -> TraitNode:
    return make_trait_node(
        char_id=char_id, state=NodeState.EXCLUDED,
        email_test_node=True, **kw,
    )


# ── TraitGraph factories ──────────────────────────────────────────────────────

def make_complete_graph(session_id: str = "sess-001") -> TraitGraph:
    """All nodes KNOWN or EXCLUDED → passes is_complete()."""
    g = TraitGraph(session_id=session_id, party_ref="PARTY-001")
    g.add_node(make_known_node("CHAR-P1A-I1", value=3, fill_priority=1))
    g.add_node(
        make_known_node(
            "CHAR-F2A-I1", value=500.0,
            fill_priority=2, branch=NodeBranch.FINANCIAL,
        )
    )
    g.add_node(make_excluded_node("CHAR-EMAIL-01"))
    return g


def make_incomplete_graph(session_id: str = "sess-002") -> TraitGraph:
    """At least one MISSING node → fails is_complete()."""
    g = TraitGraph(session_id=session_id, party_ref="PARTY-002")
    g.add_node(make_known_node("CHAR-P1A-I1", value=3))
    g.add_node(make_missing_node("CHAR-F2A-I1"))
    return g


def make_graph_with_nodes(
    known:    list[tuple[str, Any]],
    missing:  list[str],
    excluded: list[str] | None = None,
) -> TraitGraph:
    """Flexible factory: pass lists of (char_id, value), char_ids, char_ids."""
    g = TraitGraph()
    for i, (char_id, val) in enumerate(known):
        g.add_node(make_known_node(char_id, value=val, fill_priority=i + 1))
    for i, char_id in enumerate(missing):
        g.add_node(make_missing_node(char_id, fill_priority=i + 1))
    for char_id in excluded or []:
        g.add_node(make_excluded_node(char_id))
    return g


# ── Intent / config factories ─────────────────────────────────────────────────

def make_intent(
    intent_id: str = "INTENT-INVEST-CASH",
    confidence: float = 0.92,
    channel: str = "mobile",
) -> ClassifiedIntent:
    return ClassifiedIntent(
        intent_id=intent_id, confidence=confidence, channel=channel,
    )


def make_trait_config(
    char_id: str = "CHAR-P1A-I1",
    branch: NodeBranch = NodeBranch.PERSONAL,
    fill_priority: int = 1,
    data_sources: tuple[str, ...] = ("OCIS",),
    email_test_node: bool = False,
) -> TraitConfig:
    return TraitConfig(
        char_id=char_id,
        branch=branch,
        label=f"Trait {char_id}",
        op="==",
        target_value=True,
        data_sources=data_sources,
        aging="90d",
        fill_priority=fill_priority,
        email_test_node=email_test_node,
    )


def make_situation_config(
    trait_configs: list[TraitConfig] | None = None,
    conflict_pairs: list[ConflictPair] | None = None,
) -> SituationConfig:
    return SituationConfig(
        situation_id="SIT-INV-001",
        intent_ids=["INTENT-INVEST-CASH"],
        trait_configs=trait_configs or [
            make_trait_config("CHAR-P1A-I1", fill_priority=1),
            make_trait_config(
                "CHAR-F2A-I1", NodeBranch.FINANCIAL, fill_priority=2,
            ),
        ],
        conflict_pairs=conflict_pairs or [],
    )


def make_user_profile(
    party_ref: str = "PARTY-001",
    bank_data: dict[str, Any] | None = None,
) -> UserProfile:
    return UserProfile(
        party_ref=party_ref,
        bank_data=bank_data if bank_data is not None else {"CHAR-P1A-I1": 3},
    )


# ── ML factories ──────────────────────────────────────────────────────────────

def make_hypothesis(
    session_id: str = "sess-001",
    turn: int = 1,
    top_segment_id: str = "SEG-I1-A",
    top_confidence: float = 0.82,
    algorithm: ModelAlgorithm = ModelAlgorithm.LOGISTIC_REGRESSION,
) -> SegmentHypothesis:
    second_prob = round(1.0 - top_confidence, 6)
    return SegmentHypothesis(
        session_id=session_id,
        turn=turn,
        model_version="1.0.0",
        model_algorithm=algorithm,
        known_trait_count=6,
        ranked_segments=[
            SegmentRank(top_segment_id, top_confidence),
            SegmentRank("SEG-I1-B", second_prob),
        ],
        shap_top_features=[
            ShapFeature("monthly_surplus", 0.42, 1),
            ShapFeature("age_band",        0.21, 2),
        ],
    )


SEGMENT_IDS = ["SEG-I1-A", "SEG-I1-B", "SEG-I1-C"]


def make_training_data(
    n_samples: int = 300,
    segment_ids: list[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Synthetic training data for SegmentPredictor tests.
    Designed to be well-separated so LR achieves > 0.5 confidence.
    """
    from ts_agent.ml.predictor import ALL_FEATURE_COLUMNS  # noqa: PLC0415

    rng  = np.random.default_rng(seed)
    segs = segment_ids or SEGMENT_IDS
    n    = len(segs)
    rows, labels = [], []

    for i in range(n_samples):
        s_idx = i % n
        # Clear separation: each segment occupies a distinct region
        base = float(s_idx * 3)
        row = {
            "age_band":              base + rng.normal(0, 0.2),
            "monthly_surplus":       (s_idx + 1) * 500.0 + rng.normal(0, 30),
            "savings_balance":       (s_idx + 1) * 1000.0 + rng.normal(0, 50),
            "pension_contribution_pct": max(0.04, 0.04 + s_idx * 0.01 + rng.normal(0, 0.005)),
            "years_to_retirement":   max(1.0, 30.0 - s_idx * 2 + rng.normal(0, 1)),
            "risk_appetite_score":   float(s_idx + 1) + rng.normal(0, 0.05),
            "investment_experience": float(s_idx % 4),
            "known_trait_count":     float(6 + s_idx),
            "conversation_turn":     1.0,
            "employment_status":     float(s_idx % 3),
            "account_tenure_months": float(12 + s_idx * 3 + int(rng.integers(0, 6))),
            "channel":               0.0,
        }
        rows.append(row)
        labels.append(segs[s_idx])

    X = pd.DataFrame(rows, columns=ALL_FEATURE_COLUMNS)
    y = pd.Series(labels, name="segment_id")
    return X, y
