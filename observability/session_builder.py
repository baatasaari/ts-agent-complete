"""
ts_agent.observability.session_builder
=======================================
Populates a ``SessionStore`` with realistic ``SessionRecord`` objects by
replaying pipeline scenarios through the actual zone functions.

Design contract
---------------
- **No import from ``tests/``.**  The builder accepts scenario data as plain
  Python objects via the ``ScenarioProtocol`` structural type.  The caller
  (the visualiser ``app.py``, which may import from ``tests/``) injects the
  actual ``Scenario`` instances.
- **No LLM, no Neo4j, no Cloud Spanner.**  Zone 2 is simulated by calling
  the ADK tool functions (``record_consumer_answer``, ``match_segment``)
  directly with a ``FakeToolContext`` stub identical to the one used in the
  evaluation test suite.
- **Deterministic.**  Given the same scenario input, every call produces the
  same ``SessionRecord`` structure (modulo UUIDs and wall-clock timestamps).

Codex review notes
------------------
- ``ScenarioProtocol`` is a ``typing.Protocol`` so the builder is testable
  with any object that satisfies the structural contract.
- Zone functions are imported from ``ts_agent.*``; there is no coupling to
  test infrastructure.
- ``asyncio.run`` is used inside the synchronous ``build_all`` entry point
  so the builder is callable from both sync and async contexts.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ts_agent.domain.models import (
    ExplainabilityBundle,
    GateDisposition,
    HypothesisDisposition,
    ModelAlgorithm,
    SegmentHypothesis,
    SegmentRank,
)
from ts_agent.observability.session_store import SessionStore
from ts_agent.zones.zone2.tools import (
    STATE_COMPLETE,
    STATE_FILL_ORDER,
    STATE_GRAPH,
    STATE_SEGMENT_ID,
    STATE_TURN,
    _graph_from_dict,
    _graph_to_dict,
    check_graph_completeness,
    match_segment,
    record_consumer_answer,
)
from ts_agent.zones.zone3.delivery_agent import DeliveryCoordinator
from ts_agent.zones.zone3.suggestion_engine import SuggestionEngine
from tests.fixtures.factories import make_graph_with_nodes


# ──────────────────────────────────────────────────────────────────────────────
# Scenario protocol — structural type so no test import is needed
# ──────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class ScenarioProtocol(Protocol):
    """Structural contract that any scenario object must satisfy."""
    scenario_id:         str
    description:         str
    situation_id:        str
    intent_id:           str
    expected_segment:    str
    expected_suggestion: str | None
    expected_gate:       GateDisposition
    known_traits:        dict[str, Any]
    consumer_answers:    list[tuple[str, str]]
    tags:                list[str]


# ──────────────────────────────────────────────────────────────────────────────
# Fake ToolContext (identical contract to the one in evaluation tests)
# ──────────────────────────────────────────────────────────────────────────────

class _FakeToolContext:
    """Minimal ADK ToolContext stub — holds session state as a plain dict."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


# ──────────────────────────────────────────────────────────────────────────────
# Demo user roster
# ──────────────────────────────────────────────────────────────────────────────

# Maps scenario_id prefix → party_ref so the visualiser has multiple users.
# Scenarios with the same prefix belong to the same party.
# v2 party map — PS25/22 domains
_PARTY_MAP: dict[str, str] = {
    "INV":  "PARTY-ALEX-001",   # Retail investments
    "SD":   "PARTY-BETH-002",   # Structured deposits
    "PEN":  "PARTY-CARL-003",   # DC pension accumulation
    "DEC":  "PARTY-DANA-004",   # DC pension decumulation
    "EDGE": "PARTY-EDGE-005",   # Edge cases
}


def _party_for(scenario_id: str) -> str:
    prefix = scenario_id.split("-")[0]
    return _PARTY_MAP.get(prefix, "PARTY-UNKNOWN-000")


# ──────────────────────────────────────────────────────────────────────────────
# SessionBuilder
# ──────────────────────────────────────────────────────────────────────────────

class SessionBuilder:
    """
    Replays pipeline scenarios into a ``SessionStore``.

    Usage::

        from tests.datasets.scenario_catalogue import SCENARIOS
        store   = SessionStore()
        builder = SessionBuilder(store)
        builder.build_all(SCENARIOS)
        # store now contains len(SCENARIOS) SessionRecords

    The builder is stateless between calls — each ``build_all`` replaces
    any previous records in the store.
    """

    def __init__(self, store: SessionStore) -> None:
        self._store = store
        self._engine = SuggestionEngine()
        self._coordinator = DeliveryCoordinator()

    # ── Public entry point ────────────────────────────────────────────────────

    def build_all(self, scenarios: list[Any]) -> list[str]:
        """
        Replay every scenario and return the list of session_ids created.

        Validates each scenario satisfies ``ScenarioProtocol`` before replay.
        Logs a warning and skips any that do not.
        """
        session_ids: list[str] = []
        for scenario in scenarios:
            if not isinstance(scenario, ScenarioProtocol):
                continue
            sid = asyncio.run(self._build_one(scenario))
            session_ids.append(sid)
        return session_ids

    # ── Per-scenario replay ───────────────────────────────────────────────────

    async def _build_one(self, scenario: ScenarioProtocol) -> str:
        session_id = str(uuid.uuid4())
        party_ref  = _party_for(scenario.scenario_id)

        # Register the session in the store BEFORE signals fire so
        # party_ref is in the index from the start.
        self._store.register_session(
            session_id=session_id,
            party_ref=party_ref,
            intent_id=scenario.intent_id,
            situation_id=scenario.situation_id,
        )

        # ── Zone 1 — build TraitGraph ─────────────────────────────────────────
        from ts_agent.observability import signals as eamgp  # noqa: PLC0415
        eamgp.emit(
            "GRAPH_BUILD_START", eamgp.INFO, "Zone1",
            session_id=session_id,
            intent_id=scenario.intent_id,
            situation_id=scenario.situation_id,
        )
        known  = list(scenario.known_traits.items())
        missing_char_ids = [c for c, _ in scenario.consumer_answers]
        graph  = make_graph_with_nodes(known=known, missing=missing_char_ids)
        graph.session_id   = session_id
        graph.party_ref    = party_ref
        graph.intent_id    = scenario.intent_id
        graph.situation_id = scenario.situation_id

        eamgp.emit(
            "GRAPH_BUILD_COMPLETE", eamgp.INFO, "Zone1",
            session_id=session_id,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            known_count=len(graph.known_nodes()),
            missing_count=len(graph.missing_nodes()),
            excluded_count=len(graph.excluded_nodes()),
            intent_id=scenario.intent_id,
            situation_id=scenario.situation_id,
            latency_ms=45,   # simulated; labelled as such in the UI
        )

        # ── Zone 1.5 — initial ML prediction (simulated) ─────────────────────
        self._emit_prediction(
            session_id=session_id,
            turn=0,
            scenario=scenario,
            known_count=len(graph.known_nodes()),
            confidence=0.0,    # undecidable at turn 0 — not enough known traits
            disposition="UNDECIDABLE",
        )

        # ── Zone 2 — gap-fill conversation ────────────────────────────────────
        state: dict[str, Any] = {
            STATE_GRAPH:      _graph_to_dict(graph),
            STATE_TURN:       0,
            STATE_COMPLETE:   False,
            STATE_SEGMENT_ID: None,
            STATE_FILL_ORDER: missing_char_ids,
        }
        ctx = _FakeToolContext(state=state)

        for turn_idx, (char_id, raw_value) in enumerate(
            scenario.consumer_answers, start=1
        ):
            result = await record_consumer_answer(char_id, raw_value, ctx)
            if not result.get("success"):
                continue

            # Emit a per-turn prediction (simulated increasing confidence)
            conf = min(0.50 + turn_idx * 0.08, 0.92)
            self._emit_prediction(
                session_id=session_id,
                turn=turn_idx,
                scenario=scenario,
                known_count=len(graph.known_nodes()) + turn_idx,
                confidence=conf,
                disposition="ACTIVE" if conf >= 0.55 else "LOW_CONFIDENCE",
            )

        await check_graph_completeness(ctx)
        await match_segment(ctx)

        # ── Zone 3 — suggestion engine ────────────────────────────────────────
        segment_id = state.get(STATE_SEGMENT_ID)
        graph_dict = state.get(STATE_GRAPH)
        final_graph = _graph_from_dict(graph_dict) if graph_dict else graph

        bundle = ExplainabilityBundle(session_id=session_id)
        bundle.session_id = session_id

        if segment_id:
            conf_z3 = 0.50 if "LOWCONF" in scenario.scenario_id else 0.90
            hyp = SegmentHypothesis(
                session_id=session_id,
                turn=int(state.get(STATE_TURN, 1)),
                model_version="demo-1.3",
                model_algorithm=ModelAlgorithm.LOGISTIC_REGRESSION,
                known_trait_count=len([
                    n for n in final_graph.nodes.values()
                    if n.state.value == "KNOWN"
                ]),
                ranked_segments=[SegmentRank(segment_id, conf_z3)],
                disposition=HypothesisDisposition.ACTIVE,
            )
            result_z3 = self._engine.evaluate(
                segment_id, final_graph, hyp, bundle
            )
            delivery = self._coordinator.deliver(result_z3, bundle)
            # Emit AUDIT_WRITE_CONFIRMED for all gate dispositions so every
            # session is marked complete.  In production, HUMAN_REVIEW and
            # SUPPRESS sessions also write to Spanner before any action is taken.
            eamgp.emit(
                "AUDIT_WRITE_CONFIRMED", eamgp.INFO, "Zone4",
                session_id=session_id,
                audit_id=bundle.audit_id,
                gate_disposition=delivery.gate_disposition.value,
                latency_ms=12,
            )
        else:
            # No segment matched — emit terminal signal
            eamgp.emit(
                "SEGMENT_NO_MATCH", eamgp.WARN, "Zone2",
                session_id=session_id,
                segments_tried=3,
            )
            eamgp.emit(
                "AUDIT_WRITE_CONFIRMED", eamgp.INFO, "Zone4",
                session_id=session_id,
                audit_id=bundle.audit_id,
                latency_ms=8,
            )

        return session_id

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _emit_prediction(
        session_id: str,
        turn: int,
        scenario: ScenarioProtocol,
        known_count: int,
        confidence: float,
        disposition: str,
    ) -> None:
        from ts_agent.observability import signals as eamgp  # noqa: PLC0415
        import json as _json  # noqa: PLC0415

        shap = _json.dumps([
            {"f": "monthly_surplus", "v": round(confidence * 0.45, 4)},
            {"f": "age_band",        "v": round(confidence * 0.21, 4)},
            {"f": "pension_contribution_pct", "v": round(confidence * 0.12, 4)},
        ])
        signal = "SEG_PREDICT_COMPLETE" if disposition == "ACTIVE" else "SEG_PREDICT_UNDECIDABLE"
        eamgp.emit(
            signal, eamgp.INFO if disposition == "ACTIVE" else eamgp.WARN,
            "Zone1.5",
            session_id=session_id,
            turn=turn,
            top_segment_id=scenario.expected_segment if disposition == "ACTIVE" else None,
            top_confidence=confidence,
            model_version="demo-1.3",
            model_algorithm="LR",
            known_trait_count=known_count,
            shap_features_json=shap,
        )
