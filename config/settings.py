"""
ts_agent.config.settings
=========================
Centralised runtime configuration for the LBG TS Agent Platform.

All tuneable constants live here.  Every module that previously used
hardcoded literals now imports from this module.  Production deployments
override via environment variables.

Environment variable precedence:
    env var > default defined here

Usage
-----
    from ts_agent.config.settings import settings

    threshold = settings.ml_confidence_threshold  # 0.75 unless overridden
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    """
    Immutable settings object constructed once at import time.

    All thresholds, model identifiers, and behavioural constants are defined
    here and referenced by name throughout the codebase.  No magic numbers
    anywhere else.

    PS25/22 references are included where a threshold has a regulatory basis.
    """

    # ── LLM / inference ───────────────────────────────────────────────────────
    gemini_model: str = field(
        default_factory=lambda: _env_str(
            "GEMINI_MODEL", "gemini-1.5-flash"
        )
    )
    """
    Gemini model identifier used by the Zone 2 ADK LlmAgent.
    Override: GEMINI_MODEL=gemini-2.0-flash-lite
    """

    vertex_location: str = field(
        default_factory=lambda: _env_str("VERTEX_LOCATION", "europe-west2")
    )
    """GCP region for Vertex AI / Agent Engine."""

    google_cloud_project: str = field(
        default_factory=lambda: _env_str("GOOGLE_CLOUD_PROJECT", "")
    )

    # ── ML predictor ──────────────────────────────────────────────────────────
    ml_confidence_threshold: float = field(
        default_factory=lambda: _env_float(
            "TS_ML_CONFIDENCE_THRESHOLD", 0.95
        )
    )
    """
    Minimum ML prediction confidence for automated EMIT delivery (PDC-007).
    Below this threshold: R-009 GATE → HUMAN_REVIEW.
    PS25/22 paras 3.28–3.29 (medium-materiality assumption consumer check).
    Updated to 95% for high regulatory confidence requirement.
    Override: TS_ML_CONFIDENCE_THRESHOLD=0.95
    """

    ml_min_known_traits: int = field(
        default_factory=lambda: _env_int("TS_ML_MIN_KNOWN_TRAITS", 3)
    )
    """Minimum known traits before ML prediction is considered reliable."""

    shap_top_n: int = field(
        default_factory=lambda: _env_int("TS_SHAP_TOP_N", 5)
    )
    """Number of top SHAP features to include in the ExplainabilityBundle."""

    # ── Zone 2 — gap-fill ────────────────────────────────────────────────────
    graph_completeness_threshold: float = field(
        default_factory=lambda: _env_float(
            "TS_GRAPH_COMPLETENESS_THRESHOLD", 0.90
        )
    )
    """
    Proportion of non-excluded trait nodes that must be KNOWN before
    Zone 2 attempts segment matching.
    Override: TS_GRAPH_COMPLETENESS_THRESHOLD=0.85
    """

    segment_match_score_floor: float = field(
        default_factory=lambda: _env_float(
            "TS_SEGMENT_MATCH_SCORE_FLOOR", 0.90
        )
    )
    """
    Minimum proportion of including characteristics that must be satisfied
    for a segment to be considered a match in Zone 2.
    Override: TS_SEGMENT_MATCH_SCORE_FLOOR=0.85
    """

    # ── Data accuracy ────────────────────────────────────────────────────────
    default_stale_data_days: int = field(
        default_factory=lambda: _env_int(
            "TS_STALE_DATA_DAYS", 30
        )
    )
    """
    Default age (in days) beyond which trait data is considered stale and
    must be verified with the consumer (PDC-002 / UK GDPR Art 5(1)(d)).
    Segments may override this per-segment via data_accuracy_stale_days.
    Override: TS_STALE_DATA_DAYS=14
    """

    # ── Resilience ────────────────────────────────────────────────────────────
    neo4j_circuit_breaker_threshold: int = field(
        default_factory=lambda: _env_int(
            "TS_NEO4J_CB_THRESHOLD", 3
        )
    )
    """Number of consecutive Neo4j failures before circuit opens."""

    neo4j_circuit_breaker_timeout_s: float = field(
        default_factory=lambda: _env_float(
            "TS_NEO4J_CB_TIMEOUT_S", 30.0
        )
    )
    """Seconds the Neo4j circuit breaker stays open before half-opening."""

    neo4j_max_retries: int = field(
        default_factory=lambda: _env_int("TS_NEO4J_MAX_RETRIES", 3)
    )

    # ── FCA audit ─────────────────────────────────────────────────────────────
    fca_ref: str = field(
        default_factory=lambda: _env_str("TS_FCA_REF", "PS25/22")
    )
    """FCA policy statement reference used in audit records."""

    fca_firm_ref: str = field(
        default_factory=lambda: _env_str("TS_FCA_FIRM_REF", "FRN-119278")
    )
    """Firm FRN included in consumer-facing explanation (INV-06)."""

    advisor_url: str = field(
        default_factory=lambda: _env_str(
            "TS_ADVISOR_URL",
            "https://www.lloydsbank.com/financial-advice",
        )
    )
    """URL for full regulated advice, included in every consumer message (DEL-002)."""

    moneyhelper_url: str = field(
        default_factory=lambda: _env_str(
            "TS_MONEYHELPER_URL", "https://www.moneyhelper.org.uk"
        )
    )
    """MoneyHelper URL — mandatory signpost on all suggestions (DEL-006)."""

    pension_wise_url: str = field(
        default_factory=lambda: _env_str(
            "TS_PENSION_WISE_URL",
            "https://www.moneyhelper.org.uk/pensionwise",
        )
    )
    """Pension Wise URL — mandatory for all pension suggestions (DEL-006 / COBS 19)."""

    # ── Observability ─────────────────────────────────────────────────────────
    service_name: str = field(
        default_factory=lambda: _env_str("TS_SERVICE_NAME", "ts-agent")
    )

    log_level: str = field(
        default_factory=lambda: _env_str("TS_LOG_LEVEL", "INFO")
    )
    
    # ── Backward compatibility aliases ────────────────────────────────────────
    @property
    def segment_confidence_threshold(self) -> float:
        """Alias for ml_confidence_threshold (backward compatibility)."""
        return self.ml_confidence_threshold


# Singleton — imported everywhere as `settings`
settings = Settings()
