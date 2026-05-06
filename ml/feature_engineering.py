"""
ts_agent.ml.feature_engineering
===============================
Feature engineering components for ML segment prediction.

Extracted from predictor.py for better modularity and maintainability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from ts_agent.config.settings import settings
from ts_agent.domain.models import NodeState, TraitGraph


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
            "pension_pot_value":        _f(self.pension_pot_value),
            "years_to_retirement":      _f(self.years_to_retirement),
            "months_since_review":      _f(self.months_since_review),
            "employment_status":        _f(self.employment_status),
            "channel":                  _f(self.channel),
            "known_trait_count":        float(self.known_trait_count),
            "conversation_turn":        float(self.conversation_turn),
        }
        return pd.DataFrame([row])

    @classmethod
    def from_graph(cls, graph: TraitGraph, turn: int = 0) -> "PredictorFeatureVector":
        """Extract features from a TraitGraph."""
        features = cls(conversation_turn=turn)
        
        known_count = 0
        for node in graph.nodes.values():
            if node.state != NodeState.KNOWN:
                continue
            
            known_count += 1
            feature_name = CHAR_ID_TO_FEATURE.get(node.char_id)
            if not feature_name:
                continue
                
            value = node.value
            if isinstance(value, (int, float)):
                setattr(features, feature_name, float(value))
            elif isinstance(value, bool):
                setattr(features, feature_name, float(value))
        
        features.known_trait_count = known_count
        return features


def extract_features_from_traits(traits: dict[str, Any], turn: int = 0) -> PredictorFeatureVector:
    """
    Extract features from a traits dictionary.
    
    Args:
        traits: Dictionary mapping char_id to values
        turn: Conversation turn number
        
    Returns:
        PredictorFeatureVector ready for ML prediction
    """
    features = PredictorFeatureVector(conversation_turn=turn)
    
    known_count = len(traits)
    for char_id, value in traits.items():
        feature_name = CHAR_ID_TO_FEATURE.get(char_id)
        if not feature_name:
            continue
            
        if isinstance(value, (int, float)):
            setattr(features, feature_name, float(value))
        elif isinstance(value, bool):
            setattr(features, feature_name, float(value))
    
    features.known_trait_count = known_count
    return features


def build_feature_matrix(graphs: list[TraitGraph]) -> pd.DataFrame:
    """
    Build a feature matrix from multiple graphs for training.
    
    Args:
        graphs: List of TraitGraph instances
        
    Returns:
        DataFrame with features for all graphs
    """
    rows = []
    for graph in graphs:
        features = PredictorFeatureVector.from_graph(graph)
        rows.append(features.to_dataframe().iloc[0])
    
    return pd.DataFrame(rows)