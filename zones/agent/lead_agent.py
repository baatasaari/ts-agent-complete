"""
ts_agent.zones.agent.lead_agent
================================
Lead Orchestrator Agent (ADK SequentialAgent + sub-agents)

The Lead Agent coordinates the full zone pipeline:

    Zone 0  — Intent classification (injected upstream, not an ADK agent)
    Zone 1  — TraitGraphBuilder (Python service, called before ADK session)
    Zone 1.5— IterativeSegmentPredictor (called inside Zone 2 loop)
    Zone 2  — GapFillAgent (ADK LlmAgent, conversational loop)
    Zone 3  — SuggestionEngine + DeliveryCoordinator (Python services)

The Lead Agent is primarily an orchestration boundary — it holds references
to the sub-agent and the Python service instances, and provides the
``run_pipeline`` async method that the API layer calls per request.

ADK Runner integration
-----------------------
In production, the Lead Agent is registered with an ADK ``Runner`` and
``InMemorySessionService`` (or GCP-backed session service).  The gap-fill
conversation is driven by the Runner's ``run_async`` loop until the agent
signals completion via the ``ts_complete`` state flag.

The lead agent itself does not subclass ``LlmAgent`` — it is a plain
Python class that wires the ADK and Python layers together.  This keeps
the orchestration logic testable without a live LLM.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from ts_agent.domain.models import (
    ExplainabilityBundle,
    GapFillStrategy,
    NodeState,
    SegmentHypothesis,
    TraitGraph,
)
from ts_agent.ml.predictor import IterativeSegmentPredictor
from ts_agent.observability import signals as eamgp
from ts_agent.zones.zone2.tools import (
    STATE_COMPLETE,
    STATE_FILL_ORDER,
    STATE_FILL_STRATEGY,
    STATE_GRAPH,
    STATE_HYPOTHESIS,
    STATE_PARTY_REF,
    STATE_SEGMENT_ID,
    STATE_SESSION_ID,
    STATE_TURN,
    _graph_to_dict,
)
from ts_agent.zones.zone3.delivery_agent import DeliveryCoordinator, DeliveryResult
from ts_agent.zones.zone3.suggestion_engine import SuggestionEngine, SuggestionResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """
    All data produced across zones for one TS session.
    Passed through the pipeline and mutated in place.
    """
    session_id:       str = field(default_factory=lambda: str(uuid.uuid4()))
    party_ref:        str = ""
    graph:            TraitGraph | None = None
    hypothesis:       SegmentHypothesis | None = None
    fill_strategy:    GapFillStrategy = GapFillStrategy.STATIC_PRIORITY
    fill_order:       list[str] = field(default_factory=list)
    matched_segment:  str | None = None
    suggestion_result: SuggestionResult | None = None
    delivery_result:  DeliveryResult | None = None
    bundle:           ExplainabilityBundle = field(
        default_factory=ExplainabilityBundle
    )
    adk_session_state: dict[str, Any] = field(default_factory=dict)


class LeadOrchestrator:
    """
    Wires together all zones into one coherent pipeline.

    Zone 2 (conversational gap-fill) is externalised to the ADK runner
    and operates through ``adk_session_state``.  Zones 3 and beyond are
    called synchronously once Zone 2 signals completion.
    """

    def __init__(
        self,
        iterative_predictor: IterativeSegmentPredictor,
        suggestion_engine: SuggestionEngine | None = None,
        delivery_coordinator: DeliveryCoordinator | None = None,
    ) -> None:
        self._predictor    = iterative_predictor
        self._suggestion   = suggestion_engine or SuggestionEngine()
        self._delivery     = delivery_coordinator or DeliveryCoordinator()

    # ── Zone 1.5: predict and set ADK state ──────────────────────────────────

    def prepare_adk_session_state(self, ctx: PipelineContext) -> dict[str, Any]:
        """
        Convert a PipelineContext into ADK session state ready for the
        GapFillAgent.  Called before handing off to the ADK runner.
        """
        graph = ctx.graph
        if graph is None:
            raise ValueError("Graph must be built (Zone 1) before Zone 2")

        # Run initial prediction (turn=0) to get fill order
        hyp, fill_order, strategy = self._predictor.predict_and_prioritise(
            graph, ctx.session_id, turn=0
        )
        ctx.hypothesis   = hyp
        ctx.fill_order   = fill_order
        ctx.fill_strategy = strategy

        state: dict[str, Any] = {
            STATE_SESSION_ID:    ctx.session_id,
            STATE_PARTY_REF:     ctx.party_ref,
            STATE_TURN:          0,
            STATE_GRAPH:         _graph_to_dict(graph),
            STATE_FILL_ORDER:    fill_order,
            STATE_FILL_STRATEGY: strategy.value,
            STATE_COMPLETE:      False,
            STATE_SEGMENT_ID:    None,
            STATE_HYPOTHESIS:    None,
        }
        ctx.adk_session_state = state
        return state

    # ── Zone 3: process after gap-fill completes ──────────────────────────────

    def run_zone3(self, ctx: PipelineContext) -> DeliveryResult:
        """
        Called after Zone 2 (ADK gap-fill) completes and sets
        ``ts_matched_segment_id`` in ADK session state.

        Parameters
        ----------
        ctx.adk_session_state must contain ``ts_matched_segment_id``.

        Returns
        -------
        DeliveryResult — caller must call ``confirm_audit()`` before
        surfacing the message to the consumer.
        """
        state       = ctx.adk_session_state
        segment_id  = state.get(STATE_SEGMENT_ID)

        if not segment_id:
            logger.warning("Zone 3 entered without a matched segment — suppressing")
            eamgp.emit(
                "SEGMENT_NO_MATCH", eamgp.WARN, "Zone3",
                session_id=ctx.session_id,
                reason="NO_SEGMENT_IN_STATE",
            )
            from ts_agent.domain.models import GateDisposition
            return DeliveryResult(
                session_id=ctx.session_id,
                audit_id=ctx.bundle.audit_id,
                gate_disposition=GateDisposition.SUPPRESS,
                consumer_message=None,
                communication_hash=None,
                audit_confirmed=False,
                error="No segment matched",
            )

        # Reconstruct graph from ADK state
        from ts_agent.zones.zone2.tools import _graph_from_dict
        graph_dict = state.get(STATE_GRAPH)
        graph = _graph_from_dict(graph_dict) if graph_dict else ctx.graph

        ctx.matched_segment = segment_id
        ctx.bundle.session_id = ctx.session_id

        result = self._suggestion.evaluate(
            segment_id=segment_id,
            graph=graph,
            hypothesis=ctx.hypothesis,
            bundle=ctx.bundle,
        )
        ctx.suggestion_result = result

        delivery = self._delivery.deliver(result, ctx.bundle)
        ctx.delivery_result = delivery
        return delivery
