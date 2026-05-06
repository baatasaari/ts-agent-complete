"""
ts_agent.explainability.explainer
==================================
End-to-end explainability — Layers L1–L5 (Section 12).

L1  Graph path        — Neo4j traversal (handled by graph layer)
L2  ML explanation    — SHAP values in SegmentHypothesis
L3  Symbolic trace    — rule evaluation log in ExplainabilityBundle
L4  Rationale doc     — human-readable PDF (assembled by RationaleAssembler)
L5  Consumer explain  — PS25/22 §8.4 template-rendered plain English

INV-06: Consumer explanation MUST be generated from a PS25/22-approved
        Jinja2 template.  Free-text LLM output MUST NOT be served.

Improvements:
- Type safety with Protocol and validation
- Comprehensive error handling
- Detailed logging for audit trails
- Template validation on initialization
- Enum for template types
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from jinja2 import Environment, StrictUndefined, TemplateError

from ts_agent.domain.models import (
    ExplainabilityBundle,
    GateDisposition,
    RuleEvaluation,
    RuleRejection,
    SegmentHypothesis,
    TraitGraph,
)
from ts_agent.observability import signals as eamgp

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Template types and validation
# ──────────────────────────────────────────────────────────────────────────────

class TemplateType(str, Enum):
    """FCA-approved template types (INV-06)."""
    SUGGESTION = "SUGGESTION_TEMPLATE"
    NO_SUGGESTION = "NO_SUGGESTION_TEMPLATE"


# ──────────────────────────────────────────────────────────────────────────────
# FCA-approved template strings (INV-06)
# ──────────────────────────────────────────────────────────────────────────────

_SUGGESTION_TEMPLATE_SRC = """\
⚠ TARGETED SUPPORT — This is targeted support, not personalised financial advice.

Based on information about customers who share similar financial \
characteristics to yours, we thought you might find the following helpful:

  {{ suggestion_name }}

Customers in a similar situation often share these characteristics:
{%- for desc in characteristic_descriptions %}
  • {{ desc }}
{%- endfor %}

This recommendation has been designed for a group of customers who share \
similar financial characteristics. It has not been made based on a \
comprehensive review of your individual financial circumstances.
{%- if capital_at_risk %}

⚠ Capital at risk: The value of investments can fall as well as rise. \
You may get back less than you invest.
{%- endif %}
{%- if moneyhelper_signpost %}

Free impartial guidance: For free guidance on this topic, visit \
MoneyHelper at moneyhelper.org.uk
{%- endif %}
{%- if pension_wise_signpost %}
Pension guidance: For free impartial pensions guidance, visit \
Pension Wise at moneyhelper.org.uk/pensionwise
{%- endif %}
{%- if no_consolidation_statement %}

This suggestion does not involve combining or moving any pension pots.
{%- endif %}

For advice tailored to your individual circumstances, please contact: \
{{ advisor_url }}

Reference: {{ audit_id }} | FCA firm: {{ fca_firm_ref }}
"""

_NO_SUGGESTION_TEMPLATE_SRC = """\
Based on your current financial profile, we were unable to identify a \
targeted support suggestion appropriate for customers sharing your \
characteristics at this time.
{%- if top_reason %}

The primary reason is: {{ top_reason }}
{%- endif %}

For tailored guidance, please speak with one of our advisors: {{ advisor_url }}

Reference: {{ audit_id }}
"""


def _create_jinja_environment() -> Environment:
    """
    Create Jinja2 environment with strict settings for regulatory compliance.
    
    Returns:
        Configured Jinja2 environment
    """
    return Environment(
        undefined=StrictUndefined,  # Fail on undefined variables
        autoescape=True,            # XSS protection
        trim_blocks=True,           # Clean output
        lstrip_blocks=True,
    )


# Initialize templates with validation
_jinja_env = _create_jinja_environment()

try:
    _SUGGESTION_TEMPLATE = _jinja_env.from_string(_SUGGESTION_TEMPLATE_SRC)
    _NO_SUGGESTION_TEMPLATE = _jinja_env.from_string(_NO_SUGGESTION_TEMPLATE_SRC)
    logger.info("FCA-approved templates loaded successfully")
except TemplateError as e:
    logger.error(f"Template initialization failed: {e}")
    raise RuntimeError(f"Critical: FCA template validation failed: {e}") from e


# Consumer-safe rejection reasons (INV-06 / PS25/22 DEL-002).
from ts_agent.config.settings import settings
from ts_agent.config.segments import CONSUMER_REASON_MAP


# ──────────────────────────────────────────────────────────────────────────────
# Protocols for type safety
# ──────────────────────────────────────────────────────────────────────────────

class ExplainerProtocol(Protocol):
    """Protocol for explainer implementations."""
    
    def explain_suggestion(
        self,
        bundle: ExplainabilityBundle,
        suggestion_context: SuggestionContext,
    ) -> str:
        """Generate consumer-facing suggestion explanation."""
        ...
    
    def explain_no_suggestion(
        self,
        bundle: ExplainabilityBundle,
        rejections: list[RuleRejection],
    ) -> str:
        """Generate consumer-facing no-suggestion explanation."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Symbolic trace builder
# ──────────────────────────────────────────────────────────────────────────────

def build_symbolic_trace(evaluations: list[RuleEvaluation]) -> list[dict[str, Any]]:
    """
    Convert rule evaluation results to a structured trace (INV-10).

    The trace records every rule — PASS, FAIL, and GATE — so the FCA can
    reconstruct the full decision path.
    
    Args:
        evaluations: List of rule evaluations from Zone 3
        
    Returns:
        Structured trace for audit purposes
        
    Raises:
        ValueError: If evaluations list is malformed
    """
    if not isinstance(evaluations, list):
        raise ValueError(f"evaluations must be a list, got {type(evaluations)}")
    
    try:
        trace = [
            {
                "rule_id":        ev.rule_id,
                "rule_type":      ev.rule_type.value,
                "input_value":    str(ev.input_value),
                "expected_value": str(ev.expected_value),
                "operator":       ev.operator,
                "outcome":        ev.outcome,
            }
            for ev in evaluations
        ]
        
        logger.debug(f"Built symbolic trace with {len(trace)} evaluations")
        return trace
        
    except (AttributeError, TypeError) as e:
        logger.error(f"Failed to build symbolic trace: {e}")
        raise ValueError(f"Invalid evaluation object in list: {e}") from e


# ──────────────────────────────────────────────────────────────────────────────
# Consumer explainer (L5)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SuggestionContext:
    """
    Data needed to render the L5 suggestion template.

    v2: Includes PS25/22 mandatory disclosure flags (DEL-004 through DEL-008).
    All boolean flags default to False; set to True based on suggestion domain
    and product type.
    
    Attributes:
        suggestion_name: Display name of the suggestion
        characteristic_descriptions: List of consumer-safe characteristic descriptions
        advisor_url: URL for human advisor contact
        fca_firm_ref: FCA firm reference number (FRN)
        capital_at_risk: DEL-005 disclosure for investment products
        moneyhelper_signpost: DEL-006 mandatory signpost (always True)
        pension_wise_signpost: DEL-006 for pension domains
        no_consolidation_statement: DEL-014 affirmative check
    """
    suggestion_name: str
    characteristic_descriptions: list[str]
    advisor_url: str
    fca_firm_ref: str
    # PS25/22 mandatory disclosure flags
    capital_at_risk: bool = False
    moneyhelper_signpost: bool = True
    pension_wise_signpost: bool = False
    no_consolidation_statement: bool = False
    
    def __post_init__(self):
        """Validate context data."""
        if not self.suggestion_name:
            raise ValueError("suggestion_name cannot be empty")
        if not self.characteristic_descriptions:
            raise ValueError("characteristic_descriptions cannot be empty")
        if not self.advisor_url:
            raise ValueError("advisor_url cannot be empty")
        if not self.fca_firm_ref:
            raise ValueError("fca_firm_ref cannot be empty")
        
        # Log warnings for regulatory compliance
        if not self.moneyhelper_signpost:
            logger.warning("MoneyHelper signpost disabled - regulatory violation risk")
        
        # Validate characteristic descriptions
        for desc in self.characteristic_descriptions:
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError(f"Invalid characteristic description: {desc}")


class ConsumerExplainer:
    """
    Generates L5 consumer-facing explanation text (INV-06).

    Template rendering is synchronous and deterministic. Never calls an LLM.
    All explanations are generated from FCA-approved Jinja2 templates.
    
    Thread-safe and stateless after initialization.
    
    Attributes:
        _advisor_url: Default advisor contact URL
        _fca_firm_ref: FCA firm reference number
    """

    def __init__(
        self,
        advisor_url: str = settings.advisor_url,
        fca_firm_ref: str = settings.fca_firm_ref,
    ) -> None:
        """
        Initialize explainer with firm-specific settings.
        
        Args:
            advisor_url: URL for human advisor contact
            fca_firm_ref: FCA firm reference number
            
        Raises:
            ValueError: If settings are invalid
        """
        if not advisor_url:
            raise ValueError("advisor_url cannot be empty")
        if not fca_firm_ref:
            raise ValueError("fca_firm_ref cannot be empty")
        
        self._advisor_url = advisor_url
        self._fca_firm_ref = fca_firm_ref
        
        logger.info(
            f"ConsumerExplainer initialized: firm={fca_firm_ref}, "
            f"advisor_url={advisor_url}"
        )

    def explain_suggestion(
        self,
        bundle: ExplainabilityBundle,
        suggestion_context: SuggestionContext,
    ) -> str:
        """
        Render the suggestion explanation template.
        
        Emits CONSUMER_EXPLAIN_SERVED signal for observability.
        
        Args:
            bundle: Explainability data bundle
            suggestion_context: Suggestion presentation context
            
        Returns:
            Rendered consumer-facing explanation text
            
        Raises:
            TemplateError: If template rendering fails
            ValueError: If inputs are invalid
        """
        if not bundle.audit_id:
            raise ValueError("ExplainabilityBundle must have audit_id")
        
        try:
            text = _SUGGESTION_TEMPLATE.render(
                suggestion_name=suggestion_context.suggestion_name,
                characteristic_descriptions=suggestion_context.characteristic_descriptions,
                advisor_url=suggestion_context.advisor_url,
                fca_firm_ref=suggestion_context.fca_firm_ref,
                audit_id=bundle.audit_id,
                # PS25/22 mandatory disclosure flags (DEL-004 – DEL-008)
                capital_at_risk=suggestion_context.capital_at_risk,
                moneyhelper_signpost=suggestion_context.moneyhelper_signpost,
                pension_wise_signpost=suggestion_context.pension_wise_signpost,
                no_consolidation_statement=suggestion_context.no_consolidation_statement,
            )
            
            # Validate output
            if not text or len(text) < 100:
                raise ValueError("Rendered explanation is suspiciously short")
            
            # Emit observability signal
            eamgp.emit(
                "CONSUMER_EXPLAIN_SERVED",
                eamgp.INFO,
                "Zone4",
                session_id=bundle.session_id,
                audit_id=bundle.audit_id,
                explanation_template=TemplateType.SUGGESTION.value,
                text_length=len(text),
            )
            
            logger.info(
                f"Suggestion explanation rendered: session={bundle.session_id}, "
                f"audit_id={bundle.audit_id}, length={len(text)}"
            )
            
            return text
            
        except TemplateError as e:
            logger.error(f"Template rendering failed: {e}")
            eamgp.emit(
                "TEMPLATE_RENDER_FAILED",
                eamgp.ERROR,
                "Zone4",
                session_id=bundle.session_id,
                template_type=TemplateType.SUGGESTION.value,
                error=str(e),
            )
            raise

    def explain_no_suggestion(
        self,
        bundle: ExplainabilityBundle,
        rejections: list[RuleRejection],
    ) -> str:
        """
        Render the no-suggestion explanation template.
        
        Safe rejection reason is looked up from CONSUMER_REASON_MAP to ensure
        no internal/technical details are exposed to consumers.
        
        Args:
            bundle: Explainability data bundle
            rejections: List of rule rejections (ordered by priority)
            
        Returns:
            Rendered consumer-facing explanation text
            
        Raises:
            TemplateError: If template rendering fails
            ValueError: If inputs are invalid
        """
        if not bundle.audit_id:
            raise ValueError("ExplainabilityBundle must have audit_id")
        
        # Extract consumer-safe reason from top rejection
        top_reason: str | None = None
        if rejections:
            rule_id = rejections[0].rule_evaluation.rule_id
            top_reason = CONSUMER_REASON_MAP.get(rule_id)
            
            if not top_reason:
                logger.warning(
                    f"No consumer reason mapped for rule_id={rule_id}. "
                    f"Add to CONSUMER_REASON_MAP."
                )

        try:
            text = _NO_SUGGESTION_TEMPLATE.render(
                top_reason=top_reason,
                advisor_url=self._advisor_url,
                audit_id=bundle.audit_id,
            )
            
            # Validate output
            if not text or len(text) < 50:
                raise ValueError("Rendered explanation is suspiciously short")
            
            # Emit observability signal
            eamgp.emit(
                "CONSUMER_EXPLAIN_SERVED",
                eamgp.INFO,
                "Zone4",
                session_id=bundle.session_id,
                audit_id=bundle.audit_id,
                explanation_template=TemplateType.NO_SUGGESTION.value,
                rejection_count=len(rejections),
                text_length=len(text),
            )
            
            logger.info(
                f"No-suggestion explanation rendered: session={bundle.session_id}, "
                f"audit_id={bundle.audit_id}, rejections={len(rejections)}"
            )
            
            return text
            
        except TemplateError as e:
            logger.error(f"Template rendering failed: {e}")
            eamgp.emit(
                "TEMPLATE_RENDER_FAILED",
                eamgp.ERROR,
                "Zone4",
                session_id=bundle.session_id,
                template_type=TemplateType.NO_SUGGESTION.value,
                error=str(e),
            )
            raise

    @staticmethod
    def hash_communication_text(text: str) -> str:
        """
        Generate SHA-256 hash of consumer message.
        
        The hash is stored in audit logs instead of the raw text to protect
        consumer privacy while maintaining audit trail integrity.
        
        Args:
            text: Consumer-facing message text
            
        Returns:
            SHA-256 hex digest
            
        Raises:
            ValueError: If text is empty
        """
        if not text:
            raise ValueError("Cannot hash empty text")
        
        hash_value = hashlib.sha256(text.encode('utf-8')).hexdigest()
        logger.debug(f"Generated message hash: {hash_value[:16]}...")
        return hash_value


# ──────────────────────────────────────────────────────────────────────────────
# ExplainabilityBundle builder helpers
# ──────────────────────────────────────────────────────────────────────────────

def populate_zone1_fields(
    bundle: ExplainabilityBundle,
    graph: TraitGraph,
    latency_ms: int,
) -> ExplainabilityBundle:
    """
    Attach Zone 1 data to the explainability bundle.
    
    Args:
        bundle: Bundle to populate
        graph: Trait graph from Zone 1
        latency_ms: Zone 1 processing latency
        
    Returns:
        Updated bundle with Zone 1 data
        
    Raises:
        ValueError: If inputs are invalid
    """
    if latency_ms < 0:
        raise ValueError(f"latency_ms cannot be negative: {latency_ms}")
    
    try:
        bundle.known_traits = [
            {
                "char_id":          n.char_id,
                "value_hash":       n.value_hash(),
                "populated_source": n.populated_source,
            }
            for n in graph.known_nodes()
        ]
        bundle.missing_traits = [n.char_id for n in graph.missing_nodes()]
        bundle.excluded_traits = [
            {
                "char_id":          n.char_id,
                "exclusion_reason": n.populated_source,
                "fca_ref":          n.fca_ref,
            }
            for n in graph.excluded_nodes()
        ]
        bundle.zone1_latency_ms = latency_ms
        
        logger.debug(
            f"Zone 1 fields populated: known={len(bundle.known_traits)}, "
            f"missing={len(bundle.missing_traits)}, "
            f"excluded={len(bundle.excluded_traits)}"
        )
        
        return bundle
        
    except (AttributeError, TypeError) as e:
        logger.error(f"Failed to populate Zone 1 fields: {e}")
        raise ValueError(f"Invalid graph or bundle structure: {e}") from e


def populate_zone15_fields(
    bundle: ExplainabilityBundle,
    hypothesis: SegmentHypothesis,
) -> ExplainabilityBundle:
    """
    Attach Zone 1.5 (ML predictor) data to the bundle.
    
    Args:
        bundle: Bundle to populate
        hypothesis: ML segment prediction hypothesis
        
    Returns:
        Updated bundle with Zone 1.5 data
        
    Raises:
        ValueError: If inputs are invalid
    """
    try:
        bundle.final_hypothesis = hypothesis.to_neo4j_params()
        bundle.shap_values = [
            {"feature": f.feature, "shap_value": f.shap_value, "rank": f.rank}
            for f in hypothesis.shap_top_features
        ]
        bundle.model_version = hypothesis.model_version
        bundle.model_algorithm = hypothesis.model_algorithm.value
        
        logger.debug(
            f"Zone 1.5 fields populated: segment={hypothesis.top_segment_id}, "
            f"confidence={hypothesis.top_confidence:.2%}, "
            f"shap_features={len(hypothesis.shap_top_features)}"
        )
        
        return bundle
        
    except (AttributeError, TypeError) as e:
        logger.error(f"Failed to populate Zone 1.5 fields: {e}")
        raise ValueError(f"Invalid hypothesis structure: {e}") from e


def populate_zone3_fields(
    bundle: ExplainabilityBundle,
    evaluations: list[RuleEvaluation],
    gate_disposition: GateDisposition,
    candidates_evaluated: list[dict[str, Any]],
    validated_candidates: list[dict[str, Any]],
    rejections: list[RuleRejection],
) -> ExplainabilityBundle:
    """
    Attach Zone 3 validation data to the bundle (INV-10).
    
    Args:
        bundle: Bundle to populate
        evaluations: All rule evaluations
        gate_disposition: Final gate decision
        candidates_evaluated: All candidate suggestions considered
        validated_candidates: Candidates that passed validation
        rejections: Rejected candidates with reasons
        
    Returns:
        Updated bundle with Zone 3 data
        
    Raises:
        ValueError: If inputs are invalid
    """
    try:
        bundle.symbolic_trace = build_symbolic_trace(evaluations)  # INV-10
        bundle.gate_disposition = gate_disposition
        bundle.candidates_evaluated = candidates_evaluated
        bundle.validated_candidates = validated_candidates
        bundle.rejected_candidates = [
            {
                "suggestion_id":  r.suggestion_id,
                "rule_failed":    r.rule_evaluation.rule_id,
                "consumer_reason": r.consumer_reason,
            }
            for r in rejections
        ]
        
        logger.debug(
            f"Zone 3 fields populated: gate={gate_disposition.value}, "
            f"evaluated={len(candidates_evaluated)}, "
            f"validated={len(validated_candidates)}, "
            f"rejected={len(rejections)}"
        )
        
        return bundle
        
    except (AttributeError, TypeError, ValueError) as e:
        logger.error(f"Failed to populate Zone 3 fields: {e}")
        raise ValueError(f"Invalid Zone 3 data: {e}") from e
