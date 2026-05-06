"""
ts_agent.domain.models
======================
Core domain value objects for the LBG Targeted Support Agent Platform.

v2 — Updated for FCA PS25/22 (live 6 April 2026) ontology.

Design decisions
----------------
- ``TraitNode`` is a frozen dataclass; all optional fields have defaults.
- ``TraitGraph`` uses ``__post_init__`` to initialise ``_frozen`` outside
  the constructor, so the mutation gate never appears in call-sites.
- Enums inherit from ``str`` for transparent JSON / Neo4j serialisation.
- ``CheckSeverity`` and ``CheckPhase`` model the 4-phase compliance
  architecture defined in fca_ts_compliance_checks.yml (PS25/22).
- ``GateDisposition`` outcomes map to YAML check severity levels:
    HARD_BLOCK  → SUPPRESS (or exit-TS where mandated)
    SOFT_WARNING → HUMAN_REVIEW
    DISCLOSURE_REQUIRED / INFORMATION_REQUIRED → HUMAN_REVIEW (if absent)
    LOGGING_REQUIRED → EMIT (audit-only; does not block delivery)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


# ── Enumerations ──────────────────────────────────────────────────────────────

class NodeBranch(str, Enum):
    """Characteristic branch — maps to YAML segment characteristic types."""
    PERSONAL     = "PERSONAL"      # age, employment, life events
    FINANCIAL    = "FINANCIAL"     # balances, rates, pot values, contributions
    PRODUCT      = "PRODUCT"       # holdings, ISA status, fund allocation
    BEHAVIOURAL  = "BEHAVIOURAL"   # risk appetite, declared interests
    TEMPORAL     = "TEMPORAL"      # time-based signals (tax year, maturity date)
    PENSION      = "PENSION"       # pension-specific: contribution rate, pot, drawdown


class NodeState(str, Enum):
    KNOWN    = "KNOWN"
    MISSING  = "MISSING"
    EXCLUDED = "EXCLUDED"


class EdgeType(str, Enum):
    HAS_TRAIT         = "HAS_TRAIT"
    DEPENDS_ON        = "DEPENDS_ON"
    CONFLICTS_WITH    = "CONFLICTS_WITH"
    POPULATED_FROM    = "POPULATED_FROM"
    BELONGS_TO_SEG    = "BELONGS_TO_SEG"
    BELONGS_TO_SIT    = "BELONGS_TO_SIT"
    LINKED_TO_SUGG    = "LINKED_TO_SUGG"
    MATCHED_SEGMENT   = "MATCHED_SEGMENT"
    SIMILAR_TO        = "SIMILAR_TO"
    HAS_PREDICTION    = "HAS_PREDICTION"
    SUPERSEDES        = "SUPERSEDES"
    PREDICTED_SEGMENT = "PREDICTED_SEGMENT"


class HypothesisDisposition(str, Enum):
    ACTIVE      = "ACTIVE"
    SUPERSEDED  = "SUPERSEDED"
    TERMINAL    = "TERMINAL"
    FAILED      = "FAILED"
    UNDECIDABLE = "UNDECIDABLE"


class GapFillStrategy(str, Enum):
    ML_INFORMATION_GAIN = "ML_IG"
    STATIC_PRIORITY     = "STATIC"
    ENTROPY_BROADENING  = "ENTROPY"


class GateDisposition(str, Enum):
    """
    Final pipeline outcome.

    Maps to YAML compliance check severity levels:
        HARD_BLOCK + BOUNDARY_CHECK (exit-TS) → SUPPRESS
        SOFT_WARNING + missing DISCLOSURE_REQUIRED → HUMAN_REVIEW
        All checks pass → EMIT
    """
    EMIT         = "EMIT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    SUPPRESS     = "SUPPRESS"


class SessionDisposition(str, Enum):
    ACTIVE       = "ACTIVE"
    PAUSED       = "PAUSED"
    DISCONNECTED = "DISCONNECTED"
    FAILED       = "FAILED"
    COMPLETED    = "COMPLETED"


class CheckSeverity(str, Enum):
    """
    PS25/22 compliance check severity levels (fca_ts_compliance_checks.yml).

    HARD_BLOCK         : Absolute prohibition — halt and escalate. Maps → SUPPRESS.
    SOFT_WARNING       : Material risk — log, apply safeguard, or route to review.
                         Maps → HUMAN_REVIEW.
    DISCLOSURE_REQUIRED: Mandatory disclosure must be present before delivery.
                         Absent disclosure → HUMAN_REVIEW.
    LOGGING_REQUIRED   : Audit capture mandatory. Agent continues. EMIT allowed.
    BOUNDARY_CHECK     : Advice/guidance perimeter test. Exit-TS if triggered.
                         Maps → SUPPRESS (exit journey).
    INFORMATION_REQUIRED: Consumer information must be collected before proceeding.
                           If unanswered → HUMAN_REVIEW.
    """
    HARD_BLOCK          = "HARD_BLOCK"
    SOFT_WARNING        = "SOFT_WARNING"
    DISCLOSURE_REQUIRED = "DISCLOSURE_REQUIRED"
    LOGGING_REQUIRED    = "LOGGING_REQUIRED"
    BOUNDARY_CHECK      = "BOUNDARY_CHECK"
    INFORMATION_REQUIRED = "INFORMATION_REQUIRED"


class CheckPhase(str, Enum):
    """
    Pipeline phase in which a compliance check is evaluated.
    (fca_ts_compliance_checks.yml — 4 operational phases + pre-launch.)
    """
    PRE_LAUNCH   = "pre_launch"
    DESIGN       = "design"
    PRE_DELIVERY = "pre_delivery"
    DELIVERY     = "delivery"
    MONITORING   = "monitoring"


class CheckOutcome(str, Enum):
    """Runtime outcome of evaluating one compliance check."""
    PASS            = "PASS"
    HARD_BLOCK      = "HARD_BLOCK"       # → SUPPRESS
    SOFT_BLOCK      = "SOFT_BLOCK"       # → HUMAN_REVIEW
    DISCLOSURE_MISS = "DISCLOSURE_MISS"  # → HUMAN_REVIEW
    INFO_REQUIRED   = "INFO_REQUIRED"    # → HUMAN_REVIEW
    BOUNDARY_EXIT   = "BOUNDARY_EXIT"    # → SUPPRESS (exit TS)
    LOGGED          = "LOGGED"           # audit-only; EMIT continues


class AlternativeSupportType(str, Enum):
    """
    Action to take when an excluding characteristic is triggered.
    Defined per-characteristic in fca_ts_segmentations.yml.
    """
    SIGNPOST          = "SIGNPOST"           # MoneyHelper / MoneyHelper + adviser
    SPECIALIST_JOURNEY = "SPECIALIST_JOURNEY" # Firm's vulnerable customer team
    SEGMENT_REDIRECT  = "SEGMENT_REDIRECT"   # Route to a different segment
    MANDATORY_REFERRAL = "MANDATORY_REFERRAL" # Regulated specialist (DB pensions)
    NONE_REQUIRED     = "NONE_REQUIRED"      # Documented rationale; no action


class ModelAlgorithm(str, Enum):
    LOGISTIC_REGRESSION = "LR"
    RANDOM_FOREST       = "RF"
    GRADIENT_BOOSTING   = "GBT"


# Legacy alias — kept so existing tests importing RuleType continue to work.
# New code should use CheckSeverity.
class RuleType(str, Enum):
    HARD = "HARD"
    GATE = "GATE"
    SOFT = "SOFT"


# ── TraitNode ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TraitNode:
    """
    A single characteristic node in the session property graph.

    Immutable after construction.  State transitions produce new nodes
    via ``with_state()``.  All optional fields default to safe values so
    factories and tests need not repeat boilerplate.
    """
    node_id:           str
    char_id:           str
    branch:            NodeBranch
    label:             str
    op:                str
    target_value:      Any
    data_sources:      tuple[str, ...]
    aging:             str
    fill_priority:     int
    email_test_node:   bool          = False
    fill_question_key: str | None    = None
    fca_ref:           str | None    = None
    state:             NodeState     = NodeState.MISSING
    value:             Any           = None
    populated_source:  str | None    = None

    def with_state(
        self,
        state: NodeState,
        value: Any = None,
        populated_source: str | None = None,
    ) -> "TraitNode":
        from dataclasses import replace
        return replace(self, state=state, value=value,
                       populated_source=populated_source)

    def value_hash(self) -> str | None:
        """SHA-256 hex of the serialised value — safe for audit without PII."""
        if self.value is None:
            return None
        raw = json.dumps(self.value, default=str).encode()
        return hashlib.sha256(raw).hexdigest()


# ── GraphEdge ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraphEdge:
    """A directed, typed edge in the property graph."""
    from_id:    str
    to_id:      str
    edge_type:  EdgeType
    edge_id:    str             = field(default_factory=lambda: str(uuid4()))
    properties: dict[str, Any] = field(default_factory=dict)


# ── TraitGraph ────────────────────────────────────────────────────────────────

class TraitGraphIncompleteError(RuntimeError):
    """Raised when ``freeze()`` is called before ``is_complete()``."""


@dataclass
class TraitGraph:
    """
    Session-level property graph — mutable during Zone 1, frozen before Zone 2.

    ``_frozen`` is intentionally *not* a dataclass field; it is set via
    ``object.__setattr__`` in ``__post_init__`` so it never appears in the
    constructor signature or ``__repr__``.
    """
    session_id:    str = field(default_factory=lambda: str(uuid4()))
    party_ref:     str = ""
    intent_id:     str = ""
    situation_id:  str = ""
    graph_version: str = "2.0.0"
    nodes: dict[str, TraitNode] = field(default_factory=dict)
    edges: list[GraphEdge]      = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_frozen", False)

    # ── Internal guard ───────────────────────────────────────────────────────

    def _assert_mutable(self) -> None:
        if self._frozen:  # type: ignore[attr-defined]
            raise RuntimeError("Cannot mutate a frozen TraitGraph")

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_node(self, node: TraitNode) -> None:
        self._assert_mutable()
        self.nodes[node.node_id] = node
        self.edges.append(GraphEdge(
            from_id=self.session_id,
            to_id=node.node_id,
            edge_type=EdgeType.HAS_TRAIT,
            properties={
                "branch":        node.branch.value,
                "fill_priority": node.fill_priority,
            },
        ))

    def add_edge(self, edge: GraphEdge) -> None:
        self._assert_mutable()
        self.edges.append(edge)

    def update_node(self, node: TraitNode) -> None:
        self._assert_mutable()
        if node.node_id not in self.nodes:
            raise KeyError(f"Node {node.node_id!r} not found in graph")
        self.nodes[node.node_id] = node

    # ── Queries ──────────────────────────────────────────────────────────────

    def is_complete(self) -> bool:
        return all(n.state != NodeState.MISSING for n in self.nodes.values())

    def missing_nodes(self) -> list[TraitNode]:
        return [n for n in self.nodes.values() if n.state == NodeState.MISSING]

    def known_nodes(self) -> list[TraitNode]:
        return [n for n in self.nodes.values() if n.state == NodeState.KNOWN]

    def excluded_nodes(self) -> list[TraitNode]:
        return [n for n in self.nodes.values() if n.state == NodeState.EXCLUDED]

    def node_by_char_id(self, char_id: str) -> TraitNode | None:
        return next(
            (n for n in self.nodes.values() if n.char_id == char_id), None
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def freeze(self) -> None:
        """Assert INV-01 then mark graph immutable."""
        if not self.is_complete():
            missing = [n.char_id for n in self.missing_nodes()]
            raise TraitGraphIncompleteError(
                f"INV-01 violated — {len(missing)} MISSING node(s): {missing}"
            )
        object.__setattr__(self, "_frozen", True)

    @property
    def is_frozen(self) -> bool:
        return self._frozen  # type: ignore[attr-defined]

    # ── Metrics ──────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        return {
            "total":    len(self.nodes),
            "known":    sum(1 for n in self.nodes.values()
                            if n.state == NodeState.KNOWN),
            "missing":  sum(1 for n in self.nodes.values()
                            if n.state == NodeState.MISSING),
            "excluded": sum(1 for n in self.nodes.values()
                            if n.state == NodeState.EXCLUDED),
            "edges":    len(self.edges),
        }


# ── ML prediction types ───────────────────────────────────────────────────────

@dataclass
class SegmentRank:
    segment_id:  str
    probability: float


@dataclass
class ShapFeature:
    feature:    str
    shap_value: float
    rank:       int


@dataclass
class SegmentHypothesis:
    """
    ML predictor output for one conversation turn.

    Persisted as ``:PredictedSegment`` in Neo4j; chained via ``SUPERSEDES``.
    """
    session_id:         str
    turn:               int
    model_version:      str
    model_algorithm:    ModelAlgorithm
    known_trait_count:  int
    ranked_segments:    list[SegmentRank]     = field(default_factory=list)
    shap_top_features:  list[ShapFeature]     = field(default_factory=list)
    disposition:        HypothesisDisposition = HypothesisDisposition.ACTIVE
    hypothesis_id:      str = field(default_factory=lambda: str(uuid4()))
    created_ts: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def top_segment_id(self) -> str | None:
        return self.ranked_segments[0].segment_id if self.ranked_segments else None

    @property
    def top_confidence(self) -> float:
        return self.ranked_segments[0].probability if self.ranked_segments else 0.0

    def is_undecidable(self) -> bool:
        return self.disposition == HypothesisDisposition.UNDECIDABLE

    def to_neo4j_params(self) -> dict[str, Any]:
        return {
            "hypothesis_id":     self.hypothesis_id,
            "session_id":        self.session_id,
            "turn":              self.turn,
            "top_segment_id":    self.top_segment_id,
            "top_confidence":    self.top_confidence,
            "shap_json":         json.dumps([
                {"feature": f.feature,
                 "shap_value": f.shap_value,
                 "rank": f.rank}
                for f in self.shap_top_features
            ]),
            "model_version":     self.model_version,
            "model_algorithm":   self.model_algorithm.value,
            "known_trait_count": self.known_trait_count,
            "disposition":       self.disposition.value,
        }


# ── Compliance check evaluation ───────────────────────────────────────────────

@dataclass(frozen=True)
class ComplianceCheckResult:
    """
    Runtime result of evaluating one compliance check from the YAML.

    ``check_id`` maps directly to check IDs in fca_ts_compliance_checks.yml
    (e.g. PDC-001, DEL-006, DC-002).
    ``outcome`` determines gate disposition contribution:
        HARD_BLOCK / BOUNDARY_EXIT → SUPPRESS
        SOFT_BLOCK / DISCLOSURE_MISS / INFO_REQUIRED → HUMAN_REVIEW
        PASS / LOGGED → continue (EMIT if no blocks)
    ``consumer_reason`` is the FCA-approved consumer-facing explanation
    drawn from CONSUMER_REASON_MAP in explainer.py.
    """
    check_id:        str
    check_phase:     CheckPhase
    severity:        CheckSeverity
    outcome:         CheckOutcome
    description:     str
    fca_ref:         str
    input_summary:   str | None = None   # what was evaluated (no raw PII)
    consumer_reason: str | None = None


# Legacy alias — RuleEvaluation kept so existing test assertions pass.
# New code should use ComplianceCheckResult.
@dataclass(frozen=True)
class RuleEvaluation:
    rule_id:         str
    rule_type:       RuleType
    input_value:     Any
    expected_value:  Any
    operator:        str
    outcome:         str           # "PASS" | "FAIL" | "GATE"
    consumer_reason: str | None = None

    @property
    def rule_def(self) -> "RuleEvaluation":
        """Self-reference so suggestion_engine can access rule_def.rule_id."""
        return self


@dataclass
class RuleRejection:
    suggestion_id:   str
    rule_evaluation: RuleEvaluation

    @property
    def consumer_reason(self) -> str:
        return (self.rule_evaluation.consumer_reason
                or "your current financial profile")


# ── Alternative support (excluding characteristics) ───────────────────────────

@dataclass(frozen=True)
class AlternativeSupport:
    """
    Action to take when an excluding characteristic is triggered.
    Defined per-characteristic in fca_ts_segmentations.yml (PS25/22 para 3.27).
    """
    support_type:   AlternativeSupportType
    destination:    str = ""   # e.g. "MoneyHelper — finding a financial adviser"
    action:         str = ""   # e.g. "Signpost to regulated adviser"
    segment_id:     str = ""   # populated for SEGMENT_REDIRECT


@dataclass(frozen=True)
class ExcludingCharacteristic:
    """
    A single excluding characteristic from fca_ts_segmentations.yml.

    When a consumer meets this characteristic, they are excluded from the
    segment and routed per ``alternative_support``.
    """
    char_id:            str
    label:              str
    definition:         str
    rationale:          str
    alternative_support: AlternativeSupport


# ── Explainability bundle ─────────────────────────────────────────────────────

@dataclass
class ExplainabilityBundle:
    """
    Assembled across all zones; written to Spanner before delivery (INV-05).

    v2: ``compliance_checks`` replaces the old ``symbolic_trace`` as the
    primary audit record of check evaluations.  ``symbolic_trace`` is
    retained as a derived view for backward compatibility with the
    visualiser's Layer 4 trace table.
    """
    audit_id:   str = field(default_factory=lambda: str(uuid4()))
    session_id: str = ""

    intent_id:         str   = ""
    intent_confidence: float = 0.0
    intent_model_ver:  str   = ""
    intent_top_k:      list[dict[str, Any]] = field(default_factory=list)

    known_traits:     list[dict[str, Any]] = field(default_factory=list)
    missing_traits:   list[str]            = field(default_factory=list)
    excluded_traits:  list[dict[str, Any]] = field(default_factory=list)
    zone1_latency_ms: int = 0

    prediction_chain: list[dict[str, Any]] = field(default_factory=list)
    final_hypothesis: dict[str, Any]       = field(default_factory=dict)
    shap_values:      list[dict[str, Any]] = field(default_factory=list)
    model_version:    str = ""
    model_algorithm:  str = ""

    matched_segment_id: str   = ""
    segment_match_conf: float = 0.0
    gap_fill_turns:     int   = 0
    gap_fill_strategy:  str   = ""

    # v2 compliance check results (PS25/22 phase-based)
    compliance_checks:    list[ComplianceCheckResult] = field(default_factory=list)

    # Legacy trace retained for visualiser Layer 4 backward compatibility
    candidates_evaluated: list[dict[str, Any]] = field(default_factory=list)
    validated_candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected_candidates:  list[dict[str, Any]] = field(default_factory=list)
    symbolic_trace:       list[dict[str, Any]] = field(default_factory=list)

    gate_disposition:     GateDisposition = GateDisposition.SUPPRESS

    communication_text_hash: str = ""
    consumer_explanation:     str = ""
    kg_version:               str = ""
    rule_engine_ver:          str = ""
    total_latency_ms:         int = 0
    created_ts: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
