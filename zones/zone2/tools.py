"""
ts_agent.zones.zone2.tools
===========================
ADK FunctionTools used by the Gap-Fill Conversational Agent.

Every tool writes its result into ``tool_context.state`` (ADK session state),
which is the authoritative in-flight data store for a live conversation.
The graph is reconstructed from state on each turn so the agent is stateless
between calls to the LLM.

State schema (keys prefixed ``ts_``)
--------------------------------------
``ts_graph``         : serialised TraitGraph (JSON)
``ts_session_id``    : TS session UUID
``ts_party_ref``     : consumer party reference
``ts_turn``          : conversation turn counter
``ts_hypothesis``    : last SegmentHypothesis JSON (or null)
``ts_fill_strategy`` : GapFillStrategy value
``ts_complete``      : bool — set True when graph is complete / threshold met
``ts_matched_segment_id`` : segment_id after Zone 2 match (or null)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

# google.adk is imported lazily so the module loads without it installed.
# FunctionTool wrappers at the bottom of this file call _get_function_tool()
# which imports at first use — only needed in production ADK runner mode.
try:
    from google.adk.tools import FunctionTool
    from google.adk.tools.tool_context import ToolContext
    _ADK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ADK_AVAILABLE = False
    FunctionTool = None  # type: ignore[assignment,misc]
    ToolContext = object  # type: ignore[assignment,misc]

from ts_agent.config.settings import settings
from ts_agent.config.segments import SEGMENT_TO_SUGGESTIONS, SEGMENTS, RULES
from ts_agent.domain.models import (
    EdgeType,
    GapFillStrategy,
    GraphEdge,
    NodeState,
    TraitGraph,
    TraitNode,
)
from ts_agent.observability import signals as eamgp

logger = logging.getLogger(__name__)

# ── State key constants ───────────────────────────────────────────────────────

STATE_GRAPH        = "ts_graph"
STATE_SESSION_ID   = "ts_session_id"
STATE_PARTY_REF    = "ts_party_ref"
STATE_TURN         = "ts_turn"
STATE_HYPOTHESIS   = "ts_hypothesis"
STATE_FILL_STRATEGY = "ts_fill_strategy"
STATE_COMPLETE     = "ts_complete"
STATE_SEGMENT_ID   = "ts_matched_segment_id"
STATE_FILL_ORDER   = "ts_fill_order"

# ── Serialisation helpers ─────────────────────────────────────────────────────

def _graph_to_dict(graph: TraitGraph) -> dict[str, Any]:
    """Serialise TraitGraph to a JSON-safe dict for ADK session state."""
    return {
        "session_id":   graph.session_id,
        "party_ref":    graph.party_ref,
        "intent_id":    graph.intent_id,
        "situation_id": graph.situation_id,
        "nodes": {
            node_id: {
                "node_id":          n.node_id,
                "char_id":          n.char_id,
                "branch":           n.branch.value,
                "label":            n.label,
                "op":               n.op,
                "target_value":     n.target_value,
                "data_sources":     list(n.data_sources),
                "aging":            n.aging,
                "fill_priority":    n.fill_priority,
                "email_test_node":  n.email_test_node,
                "fill_question_key": n.fill_question_key,
                "fca_ref":          n.fca_ref,
                "state":            n.state.value,
                "value":            n.value,
                "populated_source": n.populated_source,
            }
            for node_id, n in graph.nodes.items()
        },
        "edges": [
            {
                "edge_id":    e.edge_id,
                "from_id":    e.from_id,
                "to_id":      e.to_id,
                "edge_type":  e.edge_type.value,
                "properties": e.properties,
            }
            for e in graph.edges
        ],
    }


def _graph_from_dict(d: dict[str, Any]) -> TraitGraph:
    """Deserialise TraitGraph from ADK session state dict."""
    from ts_agent.domain.models import NodeBranch

    graph = TraitGraph(
        session_id=d["session_id"],
        party_ref=d.get("party_ref", ""),
        intent_id=d.get("intent_id", ""),
        situation_id=d.get("situation_id", ""),
    )
    for node_id, nd in d.get("nodes", {}).items():
        node = TraitNode(
            node_id=nd["node_id"],
            char_id=nd["char_id"],
            branch=NodeBranch(nd["branch"]),
            label=nd["label"],
            op=nd["op"],
            target_value=nd["target_value"],
            data_sources=tuple(nd["data_sources"]),
            aging=nd["aging"],
            fill_priority=nd["fill_priority"],
            email_test_node=nd.get("email_test_node", False),
            fill_question_key=nd.get("fill_question_key"),
            fca_ref=nd.get("fca_ref"),
            state=NodeState(nd["state"]),
            value=nd.get("value"),
            populated_source=nd.get("populated_source"),
        )
        # Add node directly without re-triggering edge creation
        graph.nodes[node.node_id] = node

    for ed in d.get("edges", []):
        graph.edges.append(GraphEdge(
            edge_id=ed["edge_id"],
            from_id=ed["from_id"],
            to_id=ed["to_id"],
            edge_type=EdgeType(ed["edge_type"]),
            properties=ed.get("properties", {}),
        ))
    return graph


# ──────────────────────────────────────────────────────────────────────────────
# Tool 1: record_consumer_answer
# ──────────────────────────────────────────────────────────────────────────────

async def record_consumer_answer(
    char_id: str,
    value: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """
    Record a consumer's answer for a specific trait characteristic.

    This tool is called by the Gap-Fill Agent after the consumer provides
    a value in conversation.  It updates the TraitGraph node to KNOWN,
    increments the turn counter, and checks whether the graph has reached
    the completeness threshold (90 % fill).

    Args:
        char_id : The characteristic identifier being answered,
                  e.g. ``"CHAR-F2A-I1"`` for monthly_surplus.
        value   : The consumer's answer as a string (parsed by this tool).

    Returns:
        dict with keys:
            ``success``       : bool
            ``char_id``       : echoed back
            ``node_label``    : human-readable label of the answered trait
            ``completeness``  : float 0–1, proportion of non-excluded nodes that are KNOWN
            ``ready_for_match`` : bool — True when threshold reached
            ``next_char_id``  : next MISSING char_id to ask, or null
    """
    state = tool_context.state
    graph_dict = state.get(STATE_GRAPH)
    if not graph_dict:
        return {"success": False, "error": "No graph in session state"}

    graph = _graph_from_dict(graph_dict)
    node  = graph.node_by_char_id(char_id)
    if node is None:
        return {"success": False, "error": f"Unknown char_id: {char_id}"}
    if node.state != NodeState.MISSING:
        return {"success": False, "error": f"{char_id} is already {node.state.value}"}

    # ── Parse and coerce the value ────────────────────────────────────────────
    parsed = _coerce_value(value, node.op, node.target_value)

    updated = node.with_state(
        NodeState.KNOWN,
        value=parsed,
        populated_source="CONSUMER_INPUT",
    )
    graph.update_node(updated)

    # ── Turn increment ────────────────────────────────────────────────────────
    turn = int(state.get(STATE_TURN, 0)) + 1
    state[STATE_TURN]  = turn
    state[STATE_GRAPH] = _graph_to_dict(graph)

    # ── Completeness check ────────────────────────────────────────────────────
    eligible = [n for n in graph.nodes.values() if n.state != NodeState.EXCLUDED]
    known    = [n for n in eligible if n.state == NodeState.KNOWN]
    completeness = len(known) / len(eligible) if eligible else 1.0
    ready = completeness >= settings.graph_completeness_threshold or graph.is_complete()

    if ready:
        state[STATE_COMPLETE] = True

    # ── Next missing trait ────────────────────────────────────────────────────
    fill_order  = state.get(STATE_FILL_ORDER, [])
    remaining   = [c for c in fill_order if c != char_id and
                   graph.node_by_char_id(c) and
                   graph.node_by_char_id(c).state == NodeState.MISSING]
    next_char_id = remaining[0] if remaining else None

    eamgp.emit(
        "GAP_FILL_ANSWERED", eamgp.INFO, "Zone2",
        session_id=graph.session_id, turn=turn, char_id=char_id,
        value_hash=updated.value_hash(), source="CONSUMER_INPUT",
    )

    return {
        "success":        True,
        "char_id":        char_id,
        "node_label":     node.label,
        "completeness":   round(completeness, 3),
        "ready_for_match": ready,
        "next_char_id":   next_char_id,
    }


def _coerce_value(raw: str, op: str, target: Any) -> Any:
    """
    Coerce a consumer's string answer to the most useful Python type.

    Decision tree
    -------------
    1. If ``raw`` is a recognised boolean word (yes/no/true/false/y/n) → bool.
    2. If ``raw`` looks like a whole number → int.
    3. If ``raw`` looks like a decimal → float.
    4. Otherwise → str as-is (handles values like "OWNER", "RENTER", etc.).

    The ``target`` and ``op`` parameters are kept for API compatibility but
    no longer drive the dispatch — the raw value's own lexical form is the
    sole discriminator.  This prevents the factory's generic
    ``target_value=True`` from forcing all string answers through the bool
    path and producing incorrect ``False`` for inputs like "2" or "OWNER".
    """
    raw = raw.strip()

    _BOOL_TRUE  = {"yes", "true", "y"}
    _BOOL_FALSE = {"no",  "false", "n"}
    _BOOL_WORDS = _BOOL_TRUE | _BOOL_FALSE

    lower = raw.lower()

    # Rule 1: explicit boolean words
    if lower in _BOOL_WORDS:
        return lower in _BOOL_TRUE

    # Rule 2: numeric — try int then float
    try:
        as_float = float(raw)
        if as_float == int(as_float):
            return int(as_float)
        return as_float
    except ValueError:
        pass

    # Rule 3: string pass-through (e.g. "OWNER", "RENTER", "unknown")
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# Tool 2: get_next_question
# ──────────────────────────────────────────────────────────────────────────────

async def get_next_question(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """
    Return the next trait to ask the consumer about.

    Reads the ordered fill list from state (set by the IterativeSegmentPredictor
    in Zone 1.5) and returns the first remaining MISSING trait with its
    question key and human-readable label.

    Returns:
        dict with keys:
            ``char_id``     : e.g. ``"CHAR-F2A-I1"``
            ``label``       : human-readable trait label
            ``question_key``: question template key for the UI
            ``remaining``   : number of MISSING traits still to fill
            ``done``        : True if no more MISSING traits
    """
    state = tool_context.state
    graph_dict = state.get(STATE_GRAPH)
    if not graph_dict:
        return {"done": True, "error": "No graph"}

    graph      = _graph_from_dict(graph_dict)
    fill_order = state.get(STATE_FILL_ORDER, [])

    missing_set = {n.char_id for n in graph.missing_nodes()}
    ordered_missing = [c for c in fill_order if c in missing_set]

    # Fall back to graph's own ordering if predictor hasn't set one
    if not ordered_missing:
        ordered_missing = [
            n.char_id
            for n in sorted(graph.missing_nodes(), key=lambda n: n.fill_priority)
        ]

    if not ordered_missing:
        return {"done": True, "remaining": 0, "char_id": None}

    next_char = ordered_missing[0]
    node = graph.node_by_char_id(next_char)

    eamgp.emit(
        "GAP_FILL_QUESTION_ASKED", eamgp.INFO, "Zone2",
        session_id=graph.session_id,
        char_id=next_char,
        question_key=node.fill_question_key if node else None,
        remaining=len(ordered_missing),
    )

    return {
        "done":         False,
        "char_id":      next_char,
        "label":        node.label if node else next_char,
        "question_key": node.fill_question_key if node else None,
        "remaining":    len(ordered_missing),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tool 3: check_graph_completeness
# ──────────────────────────────────────────────────────────────────────────────

async def check_graph_completeness(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """
    Report the current completeness of the TraitGraph.

    Returns:
        dict with keys:
            ``completeness``    : float 0–1
            ``total_nodes``     : int
            ``known_nodes``     : int
            ``missing_nodes``   : int
            ``excluded_nodes``  : int
            ``ready_for_match`` : bool (completeness >= 0.90)
            ``is_fully_complete`` : bool (all nodes KNOWN or EXCLUDED)
    """
    state      = tool_context.state
    graph_dict = state.get(STATE_GRAPH)
    if not graph_dict:
        return {"completeness": 0.0, "ready_for_match": False}

    graph    = _graph_from_dict(graph_dict)
    eligible = [n for n in graph.nodes.values() if n.state != NodeState.EXCLUDED]
    known    = [n for n in eligible if n.state == NodeState.KNOWN]
    missing  = [n for n in eligible if n.state == NodeState.MISSING]
    comp     = len(known) / len(eligible) if eligible else 1.0

    return {
        "completeness":       round(comp, 3),
        "total_nodes":        len(graph.nodes),
        "known_nodes":        len(known),
        "missing_nodes":      len(missing),
        "excluded_nodes":     len(graph.excluded_nodes()),
        "ready_for_match":    comp >= settings.graph_completeness_threshold or graph.is_complete(),
        "is_fully_complete":  graph.is_complete(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tool 4: match_segment   (deterministic Zone 2 match)
# ──────────────────────────────────────────────────────────────────────────────

async def match_segment(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """
    Run the deterministic SegmentMatcher against the current TraitGraph.

    Called when ``ready_for_match`` is True.  Evaluates all segment criteria
    against KNOWN trait values and returns the first matching segment.

    Returns:
        dict with keys:
            ``matched``        : bool
            ``segment_id``     : matched segment id or null
            ``segment_label``  : human-readable label
            ``confidence``     : proportion of criteria that passed (0–1)
            ``failed_criteria``: list of TraitCriterion specs that failed
    """
    state      = tool_context.state
    graph_dict = state.get(STATE_GRAPH)
    if not graph_dict:
        return {"matched": False, "error": "No graph"}

    graph        = _graph_from_dict(graph_dict)
    situation_id = graph.situation_id

    # Find candidate segments for this situation
    candidates = [
        seg for seg in SEGMENTS.values()
        if seg.situation_id == situation_id
    ]
    if not candidates:
        return {"matched": False, "segment_id": None, "reason": "NO_SEGMENTS_FOR_SITUATION"}

    known_values: dict[str, Any] = {
        n.char_id: n.value
        for n in graph.nodes.values()
        if n.state == NodeState.KNOWN
    }

    best_seg     = None
    best_score   = -1.0
    failed_specs = []

    for seg in candidates:
        # PS25/22 para 3.49 / PDC-001: check excluding characteristics FIRST.
        # If any excluding characteristic is met, this segment must not match.
        has_excluding = False
        for exc in seg.excluding:
            exc_val = known_values.get(exc.char_id)
            if exc_val is None:
                continue
            # Excluding chars are boolean flags or threshold checks.
            # Simple True/presence check covers the majority of cases.
            # For threshold/range exclusions we check the definition numerically
            # via the segment criteria pattern — here we use a basic check:
            # if the excluding char_id value matches the definition trigger.
            # The full predicate logic is in the compliance engine (PDC-001).
            # For zone2 matching, we apply basic excluding characteristic logic:
            if exc.char_id == "CHAR-P1B-I1" and exc_val:
                # Active vulnerability flag
                has_excluding = True; break
            elif exc.char_id == "CHAR-F2G-I1" and exc_val:
                # Active high-cost debt
                has_excluding = True; break
            elif exc.char_id == "CHAR-F2H-I1" and exc_val:
                # Financial hardship / arrears
                has_excluding = True; break
            elif exc.char_id == "CHAR-P2K-I1" and exc_val:
                # DB transfer applicable
                has_excluding = True; break
            elif exc.char_id == "CHAR-P2P-I1" and exc_val:
                # Recent independent advice
                has_excluding = True; break
            elif exc.char_id == "CHAR-F2M-I1" and exc_val is not None:
                # Lump sum amount — check if above threshold (£75k for INV-004)
                # Threshold comes from the segment's excluding characteristic definition
                if float(exc_val) > 75000 and seg.segment_id == "SEG-INV-004":
                    has_excluding = True; break
            elif exc.char_id == "CHAR-F2B-I1" and exc_val is not None:
                # Cash/deposit amount — check segment-specific thresholds
                if seg.segment_id == "SEG-SD-001" and float(exc_val) > 100000:  # excl char threshold
                    has_excluding = True; break
                elif seg.segment_id == "SEG-INV-001" and float(exc_val) > 100000:  # excl char threshold
                    has_excluding = True; break
            elif exc.char_id == "CHAR-P2L-I1" and exc_val is not None:
                # DC pot value — check ceiling for DEC-002
                if seg.segment_id == "SEG-DEC-002" and float(exc_val) > 30000:  # excl char threshold
                    has_excluding = True; break
            elif exc.char_id == "CHAR-P2J-I1" and exc_val and seg.segment_id == "SEG-DEC-001":
                # Already in drawdown → doesn't apply to pre-retirement segment
                has_excluding = True; break

        if has_excluding:
            eamgp.emit(
                "SEGMENT_EXCLUDED", eamgp.INFO, "Zone2",
                session_id=graph.session_id,
                segment_id=seg.segment_id,
                reason="excluding_characteristic_met",
            )
            continue

        passed, failed = _evaluate_segment(seg, known_values)
        score = passed / len(seg.criteria) if seg.criteria else 1.0
        if score > best_score and score >= settings.segment_match_score_floor:
            best_score = score
            best_seg   = seg
            failed_specs = failed

    if best_seg is None:
        eamgp.emit(
            "SEGMENT_NO_MATCH", eamgp.WARN, "Zone2",
            session_id=graph.session_id,
            segments_tried=len(candidates),
        )
        # Mark conversation as complete even on no-match so the demo exits
        state[STATE_COMPLETE] = True
        return {
            "matched":    False,
            "segment_id": None,
            "confidence": 0.0,
            "reason":     "NO_MATCHING_SEGMENT",
        }

    # Write matched segment to state
    state[STATE_SEGMENT_ID] = best_seg.segment_id

    eamgp.emit(
        "SEGMENT_MATCHED", eamgp.INFO, "Zone2",
        session_id=graph.session_id,
        segment_id=best_seg.segment_id,
        confidence=round(best_score, 3),
    )

    return {
        "matched":         True,
        "segment_id":      best_seg.segment_id,
        "segment_label":   best_seg.label,
        "confidence":      round(best_score, 3),
        "failed_criteria": [
            {"char_id": c.char_id, "op": c.op, "expected": c.value}
            for c in failed_specs
        ],
    }


def _evaluate_segment(
    seg,
    known_values: dict[str, Any],
) -> tuple[int, list]:
    """Return (count_passed, list_of_failed_criteria)."""
    passed, failed = 0, []
    for criterion in seg.criteria:
        actual = known_values.get(criterion.char_id)
        if actual is None:
            # MISSING trait → treat as fail for matching purposes
            failed.append(criterion)
            continue
        if _op_eval(actual, criterion.op, criterion.value):
            passed += 1
        else:
            failed.append(criterion)
    return passed, failed


def _op_eval(actual: Any, op: str, expected: Any) -> bool:
    """Evaluate a single criterion operator."""
    try:
        if op == "==":   return actual == expected
        if op == "!=":   return actual != expected
        if op == ">":    return float(actual) >  float(expected)
        if op == ">=":   return float(actual) >= float(expected)
        if op == "<":    return float(actual) <  float(expected)
        if op == "<=":   return float(actual) <= float(expected)
        if op == "in":   return actual in expected
    except (TypeError, ValueError):
        pass
    return False


# ──────────────────────────────────────────────────────────────────────────────
# ADK FunctionTool wrappers — lazily constructed so the module is importable
# without google-adk installed (used by demo and unit tests).
# Production ADK runner imports these; they raise ImportError with a clear
# message if google-adk is not installed.
# ──────────────────────────────────────────────────────────────────────────────

def _make_tool(fn):
    """Create an ADK FunctionTool, raising clearly if ADK is not installed."""
    if not _ADK_AVAILABLE:
        raise ImportError(
            "google-adk is required to create FunctionTool wrappers. "
            "Install it with: uv add google-adk==1.31.1"
        )
    return FunctionTool(fn)


class _LazyTool:
    """
    Proxy that constructs the real FunctionTool on first attribute access.
    Allows the module to be imported without google-adk; the tool is only
    materialised when the ADK runner calls it.
    """
    def __init__(self, fn):
        self._fn   = fn
        self._tool = None

    def _ensure(self):
        if self._tool is None:
            self._tool = _make_tool(self._fn)
        return self._tool

    def __getattr__(self, name):
        return getattr(self._ensure(), name)

    def __call__(self, *args, **kwargs):
        return self._ensure()(*args, **kwargs)


RECORD_ANSWER_TOOL      = _LazyTool(record_consumer_answer)
GET_NEXT_QUESTION_TOOL  = _LazyTool(get_next_question)
CHECK_COMPLETENESS_TOOL = _LazyTool(check_graph_completeness)
MATCH_SEGMENT_TOOL      = _LazyTool(match_segment)
