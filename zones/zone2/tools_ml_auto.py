"""
ts_agent.zones.zone2.tools_ml_auto
===================================
ML-DRIVEN Gap-Fill Tools with AUTOMATIC Iterative Prediction

Key Design: ML prediction happens AUTOMATICALLY after each answer, not requiring
LLM to explicitly call a prediction tool. The LLM just asks questions and records
answers - the system handles prediction and prioritization automatically.

Features:
- Automatic ML prediction after every answer recorded
- Full logging of prediction changes for explainability
- SHAP-based question prioritization baked into get_next_question
- Confidence delta tracking (how much prediction improved)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

try:
    from google.adk.tools import FunctionTool
    from google.adk.tools.tool_context import ToolContext
    _ADK_AVAILABLE = True
except ImportError:
    _ADK_AVAILABLE = False
    FunctionTool = None  # type: ignore[assignment,misc]
    ToolContext = object  # type: ignore[assignment,misc]

from ts_agent.config.settings import settings
from ts_agent.config.segments import SEGMENTS
from ts_agent.config.segment_fill_priorities import (
    get_priority_traits_for_segment,
    get_global_trait_priority,
    get_segment_rationale,
)
from ts_agent.domain.models import NodeState
from ts_agent.ml.predictor import IterativeSegmentPredictor
from ts_agent.observability import signals as eamgp

# Import base tools
from ts_agent.zones.zone2.tools import (
    STATE_GRAPH,
    STATE_SESSION_ID,
    STATE_TURN,
    STATE_HYPOTHESIS,
    STATE_COMPLETE,
    STATE_SEGMENT_ID,
    _graph_from_dict,
    _graph_to_dict,
    _coerce_value,
    _evaluate_segment,
    _op_eval,
    match_segment as _base_match_segment,
)

logger = logging.getLogger(__name__)

# Additional state keys for ML tracking
STATE_PREV_CONFIDENCE = "ts_prev_confidence"
STATE_PREDICTION_HISTORY = "ts_prediction_history"


# ──────────────────────────────────────────────────────────────────────────────
# Core ML Prediction Function (called automatically)
# ──────────────────────────────────────────────────────────────────────────────

def _run_ml_prediction(
    graph,
    turn: int,
    session_id: str,
    state: dict,
) -> dict[str, Any]:
    """
    Run ML segment prediction and log everything for explainability.
    
    Returns prediction result with confidence delta tracking.
    """
    # Extract known traits
    known_traits = {
        n.char_id: n.value
        for n in graph.nodes.values()
        if n.state == NodeState.KNOWN
    }
    
    if not known_traits:
        # No prediction possible yet
        eamgp.emit(
            "SEG_PREDICT_UNDECIDABLE", eamgp.WARN, "Zone1.5",
            session_id=session_id,
            turn=turn,
            top_segment_id=None,
            top_confidence=0.0,
            model_version="demo-1.3",
            model_algorithm="LR",
            known_trait_count=0,
            shap_features_json="[]",
        )
        return {
            "top_segment_id": None,
            "top_confidence": 0.0,
            "confidence_delta": 0.0,
            "segments_predicted": [],
            "high_confidence": False,
            "prioritized_missing": [],
        }
    
    # Run predictor
    predictor = IterativeSegmentPredictor()
    try:
        hypothesis, ordered, strategy = predictor.predict(
            graph=graph,
            session_id=session_id,
            turn=turn,
        )
        
        # Get previous confidence for delta calculation
        prev_confidence = float(state.get(STATE_PREV_CONFIDENCE, 0.0))
        confidence_delta = hypothesis.top_confidence - prev_confidence
        
        # Store current confidence for next iteration
        state[STATE_PREV_CONFIDENCE] = hypothesis.top_confidence
        
        # Build prediction history entry
        prediction_entry = {
            "turn": turn,
            "top_segment_id": hypothesis.top_segment_id,
            "top_confidence": round(hypothesis.top_confidence, 4),
            "confidence_delta": round(confidence_delta, 4),
            "all_scores": {k: round(v, 4) for k, v in hypothesis.all_scores.items()},
            "shap_features": hypothesis.shap_features[:5],
            "known_trait_count": len(known_traits),
        }
        
        # Append to prediction history
        history_json = state.get(STATE_PREDICTION_HISTORY, "[]")
        history = json.loads(history_json)
        history.append(prediction_entry)
        state[STATE_PREDICTION_HISTORY] = json.dumps(history)
        
        # Store latest hypothesis
        state[STATE_HYPOTHESIS] = json.dumps(prediction_entry)
        
        # Emit comprehensive signal for observability
        eamgp.emit(
            "SEG_PREDICT_COMPLETE", eamgp.INFO, "Zone1.5",
            session_id=session_id,
            turn=turn,
            top_segment_id=hypothesis.top_segment_id,
            top_confidence=round(hypothesis.top_confidence, 4),
            confidence_delta=round(confidence_delta, 4),
            model_version=hypothesis.model_version,
            model_algorithm=hypothesis.model_algorithm,
            known_trait_count=len(known_traits),
            shap_features_json=json.dumps(hypothesis.shap_features[:3]),
        )
        
        # Log prediction change if significant
        if abs(confidence_delta) > 0.10:
            logger.info(
                f"ML prediction updated: {hypothesis.top_segment_id} "
                f"confidence {prev_confidence:.2f} → {hypothesis.top_confidence:.2f} "
                f"(Δ{confidence_delta:+.2f})"
            )
        
        # Rank missing traits by SHAP importance
        prioritized_missing = _rank_missing_traits_by_shap(
            graph, hypothesis.shap_features
        )
        
        # Determine if confidence is high enough to stop
        high_conf = hypothesis.top_confidence >= settings.segment_confidence_threshold
        
        return {
            "top_segment_id": hypothesis.top_segment_id,
            "top_confidence": round(hypothesis.top_confidence, 4),
            "confidence_delta": round(confidence_delta, 4),
            "segments_predicted": [
                {
                    "segment_id": seg_id,
                    "label": SEGMENTS[seg_id].label if seg_id in SEGMENTS else seg_id,
                    "score": round(score, 4),
                }
                for seg_id, score in sorted(
                    hypothesis.all_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
            ],
            "high_confidence": high_conf,
            "prioritized_missing": prioritized_missing,
        }
        
    except Exception as e:
        logger.debug(f"ML prediction failed: {e}")
        eamgp.emit(
            "SEG_PREDICT_ERROR", eamgp.ERROR, "Zone1.5",
            session_id=session_id,
            turn=turn,
            error=str(e),
        )
        return {
            "top_segment_id": None,
            "top_confidence": 0.0,
            "confidence_delta": 0.0,
            "segments_predicted": [],
            "high_confidence": False,
            "prioritized_missing": [],
        }


def _rank_missing_traits_by_shap(
    graph,
    shap_features: list[dict],
) -> list[str]:
    """
    Rank MISSING traits by SHAP feature importance.
    
    Returns char_ids ordered by information gain potential (most important first).
    """
    # Build importance map from SHAP
    importance_map = {
        feat["f"]: abs(feat["v"])
        for feat in shap_features
    }
    
    # Score each missing trait
    char_scores = {}
    for node in graph.missing_nodes():
        # Map char_id to feature name (via node label)
        feature_name = node.label.lower().replace(" ", "_").replace("-", "_")
        importance = importance_map.get(feature_name, 0.0)
        
        # Also consider fill_priority as tiebreaker
        combined_score = importance * 100 + (10 - node.fill_priority)
        char_scores[node.char_id] = combined_score
    
    # Sort by score descending
    ranked = sorted(
        char_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return [char_id for char_id, _ in ranked]


# ──────────────────────────────────────────────────────────────────────────────
# ENHANCED TOOL: record_consumer_answer (with automatic ML prediction)
# ──────────────────────────────────────────────────────────────────────────────

async def record_consumer_answer_ml_auto(
    char_id: str,
    value: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """
    Record consumer answer and AUTOMATICALLY run ML prediction.
    
    This is the core enhancement: after recording the answer, the system
    immediately runs ML prediction without requiring LLM to call it.
    
    Args:
        char_id: Trait characteristic being answered
        value: Consumer's answer
    
    Returns:
        dict with:
            success, char_id, node_label (standard fields)
            ml_prediction: Current segment hypothesis
            confidence_delta: How much confidence improved
            ready_for_match: True if ML confidence >= 75%
            next_char_id: Highest priority question to ask next
    """
    state = tool_context.state
    graph_dict = state.get(STATE_GRAPH)
    if not graph_dict:
        return {"success": False, "error": "No graph in session state"}
    
    graph = _graph_from_dict(graph_dict)
    node = graph.node_by_char_id(char_id)
    if node is None:
        return {"success": False, "error": f"Unknown char_id: {char_id}"}
    if node.state != NodeState.MISSING:
        return {"success": False, "error": f"{char_id} is already {node.state.value}"}
    
    # Parse and record answer
    parsed = _coerce_value(value, node.op, node.target_value)
    updated = node.with_state(
        NodeState.KNOWN,
        value=parsed,
        populated_source="CONSUMER_INPUT",
    )
    graph.update_node(updated)
    
    # Increment turn
    turn = int(state.get(STATE_TURN, 0)) + 1
    state[STATE_TURN] = turn
    state[STATE_GRAPH] = _graph_to_dict(graph)
    
    # Emit answer recorded signal
    eamgp.emit(
        "GAP_FILL_ANSWERED", eamgp.INFO, "Zone2",
        session_id=graph.session_id,
        turn=turn,
        char_id=char_id,
        value_hash=updated.value_hash(),
        source="CONSUMER_INPUT",
    )
    
    # === AUTOMATIC ML PREDICTION ===
    ml_result = _run_ml_prediction(graph, turn, graph.session_id, state)
    
    # Calculate graph completeness
    eligible = [n for n in graph.nodes.values() if n.state != NodeState.EXCLUDED]
    known = [n for n in eligible if n.state == NodeState.KNOWN]
    completeness = len(known) / len(eligible) if eligible else 1.0
    
    # Determine readiness: high ML confidence OR traditional threshold
    ready = (
        ml_result["high_confidence"] or
        completeness >= settings.graph_completeness_threshold or
        graph.is_complete()
    )
    
    if ready:
        state[STATE_COMPLETE] = True
    
    # Get next question from prioritized list
    prioritized = ml_result.get("prioritized_missing", [])
    next_char_id = prioritized[0] if prioritized else None
    
    return {
        "success": True,
        "char_id": char_id,
        "node_label": node.label,
        "completeness": round(completeness, 3),
        "ready_for_match": ready,
        "next_char_id": next_char_id,
        "ml_prediction": {
            "segment_id": ml_result["top_segment_id"],
            "confidence": ml_result["top_confidence"],
            "confidence_delta": ml_result["confidence_delta"],
            "high_confidence": ml_result["high_confidence"],
        },
        "top_3_segments": ml_result["segments_predicted"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENHANCED TOOL: get_next_question (with ML prioritization)
# ──────────────────────────────────────────────────────────────────────────────

async def get_next_question_ml_prioritized(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """
    Get next question using multi-strategy prioritization:
    
    1. **SHAP features** (from ML prediction) - highest priority
    2. **Segment-specific config** (from segment_fill_priorities.py)
    3. **Global trait importance** (from segment_fill_priorities.py)
    4. **Node fill_priority** (fallback)
    
    Returns the most informative question to ask based on available intelligence.
    
    Returns:
        dict with:
            char_id: Most important trait to ask about
            label: Human-readable trait label
            question_key: Question template key
            remaining: Number of missing traits
            priority_reason: Why this trait was prioritized
            current_ml_confidence: Latest ML confidence
    """
    state = tool_context.state
    graph_dict = state.get(STATE_GRAPH)
    if not graph_dict:
        return {"done": True, "error": "No graph"}
    
    graph = _graph_from_dict(graph_dict)
    missing_nodes = list(graph.missing_nodes())
    
    if not missing_nodes:
        return {"done": True, "char_id": None, "remaining": 0}
    
    # Get latest ML prediction
    hypothesis_json = state.get(STATE_HYPOTHESIS)
    current_hypothesis = None
    if hypothesis_json:
        current_hypothesis = json.loads(hypothesis_json)
    
    # Strategy 1: SHAP features (most dynamic, ML-driven)
    if current_hypothesis:
        shap_features = current_hypothesis.get("shap_features", [])
        if shap_features:
            # Find first missing trait from SHAP ranking
            feature_names = [f["f"] for f in shap_features]
            for feat_name in feature_names:
                for node in missing_nodes:
                    node_feat = node.label.lower().replace(" ", "_").replace("-", "_")
                    if node_feat == feat_name:
                        logger.info(f"Prioritizing {node.char_id} via SHAP features")
                        return {
                            "done": False,
                            "char_id": node.char_id,
                            "label": node.label,
                            "question_key": node.fill_question_key,
                            "remaining": len(missing_nodes),
                            "priority_reason": f"High SHAP importance for {current_hypothesis.get('top_segment_id')}",
                            "current_ml_confidence": current_hypothesis.get("top_confidence", 0.0),
                        }
    
    # Strategy 2: Segment-specific configuration
    if current_hypothesis and current_hypothesis.get("top_segment_id"):
        segment_id = current_hypothesis["top_segment_id"]
        priority_traits = get_priority_traits_for_segment(segment_id)
        rationale = get_segment_rationale(segment_id)
        
        if priority_traits:
            # Find first missing trait from config priorities
            for char_id in priority_traits:
                for node in missing_nodes:
                    if node.char_id == char_id:
                        logger.info(
                            f"Prioritizing {node.char_id} via segment config for {segment_id}"
                        )
                        return {
                            "done": False,
                            "char_id": node.char_id,
                            "label": node.label,
                            "question_key": node.fill_question_key,
                            "remaining": len(missing_nodes),
                            "priority_reason": f"Config priority for {segment_id}: {rationale}",
                            "current_ml_confidence": current_hypothesis.get("top_confidence", 0.0),
                        }
    
    # Strategy 3: Global trait importance
    missing_with_global_priority = [
        (node, get_global_trait_priority(node.char_id))
        for node in missing_nodes
    ]
    missing_with_global_priority.sort(key=lambda x: x[1], reverse=True)
    
    if missing_with_global_priority:
        next_node, global_priority = missing_with_global_priority[0]
        if global_priority > 5:  # Only use if significantly important
            logger.info(
                f"Prioritizing {next_node.char_id} via global importance ({global_priority})"
            )
            return {
                "done": False,
                "char_id": next_node.char_id,
                "label": next_node.label,
                "question_key": next_node.fill_question_key,
                "remaining": len(missing_nodes),
                "priority_reason": f"Global trait importance: {global_priority}/10",
                "current_ml_confidence": current_hypothesis.get("top_confidence", 0.0) if current_hypothesis else 0.0,
            }
    
    # Strategy 4: Fallback to standard fill_priority
    missing_sorted = sorted(missing_nodes, key=lambda n: n.fill_priority)
    next_node = missing_sorted[0]
    
    eamgp.emit(
        "GAP_FILL_QUESTION_ASKED", eamgp.INFO, "Zone2",
        session_id=graph.session_id,
        char_id=next_node.char_id,
        question_key=next_node.fill_question_key,
        remaining=len(missing_sorted),
    )
    
    return {
        "done": False,
        "char_id": next_node.char_id,
        "label": next_node.label,
        "question_key": next_node.fill_question_key,
        "remaining": len(missing_sorted),
        "priority_reason": f"Standard fill_priority: {next_node.fill_priority}",
        "current_ml_confidence": 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENHANCED TOOL: check_graph_completeness (with ML readiness)
# ──────────────────────────────────────────────────────────────────────────────

async def check_graph_completeness_ml_aware(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """
    Check completeness with ML confidence awareness.
    
    Returns readiness based on EITHER:
    - ML confidence >= 75% (intelligent stopping)
    - OR traditional 90% graph completeness
    """
    state = tool_context.state
    graph_dict = state.get(STATE_GRAPH)
    if not graph_dict:
        return {"completeness": 0.0, "ready_for_match": False}
    
    graph = _graph_from_dict(graph_dict)
    eligible = [n for n in graph.nodes.values() if n.state != NodeState.EXCLUDED]
    known = [n for n in eligible if n.state == NodeState.KNOWN]
    missing = [n for n in eligible if n.state == NodeState.MISSING]
    comp = len(known) / len(eligible) if eligible else 1.0
    
    # Check ML confidence
    hypothesis_json = state.get(STATE_HYPOTHESIS)
    ml_confidence = 0.0
    ml_ready = False
    if hypothesis_json:
        hypothesis = json.loads(hypothesis_json)
        ml_confidence = hypothesis.get("top_confidence", 0.0)
        ml_ready = ml_confidence >= settings.segment_confidence_threshold
    
    # Ready if either condition met
    traditional_ready = comp >= settings.graph_completeness_threshold or graph.is_complete()
    ready = ml_ready or traditional_ready
    
    return {
        "completeness": round(comp, 3),
        "total_nodes": len(graph.nodes),
        "known_nodes": len(known),
        "missing_nodes": len(missing),
        "excluded_nodes": len(graph.excluded_nodes()),
        "ready_for_match": ready,
        "is_fully_complete": graph.is_complete(),
        "ml_confidence": round(ml_confidence, 4),
        "ml_ready": ml_ready,
        "traditional_ready": traditional_ready,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pass-through: match_segment (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

match_segment = _base_match_segment


# ──────────────────────────────────────────────────────────────────────────────
# ADK FunctionTool wrappers
# ──────────────────────────────────────────────────────────────────────────────

def _make_tool(fn):
    if not _ADK_AVAILABLE:
        raise ImportError(
            "google-adk is required. Install: uv add google-adk==1.31.1"
        )
    return FunctionTool(fn)


class _LazyTool:
    def __init__(self, fn):
        self._fn = fn
        self._tool = None
    
    def _ensure(self):
        if self._tool is None:
            self._tool = _make_tool(self._fn)
        return self._tool
    
    def __getattr__(self, name):
        return getattr(self._ensure(), name)
    
    def __call__(self, *args, **kwargs):
        return self._ensure()(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions for testing/standalone use
# ──────────────────────────────────────────────────────────────────────────────

def _build_features_for_prediction(graph) -> dict[str, Any]:
    """
    Build feature dictionary from graph for ML prediction testing.
    
    Returns a dictionary of features that can be used with the predictor.
    """
    features = {}
    
    # Extract known traits
    for node in graph.nodes.values():
        if node.state == NodeState.KNOWN:
            feature_name = node.label.lower().replace(" ", "_").replace("-", "_")
            features[feature_name] = node.value
    
    # Add metadata
    features["situation_id"] = graph.situation_id
    features["known_count"] = len([n for n in graph.nodes.values() if n.state == NodeState.KNOWN])
    features["missing_count"] = len([n for n in graph.nodes.values() if n.state == NodeState.MISSING])
    
    return features


def get_ml_prediction(graph, turn: int) -> str:
    """
    Standalone ML prediction function for testing.
    
    Returns JSON string with prediction results.
    """
    try:
        # Extract known traits
        known_traits = {
            n.char_id: n.value
            for n in graph.nodes.values()
            if n.state == NodeState.KNOWN
        }
        
        if not known_traits:
            return json.dumps({
                "top_segment_id": None,
                "top_confidence": 0.0,
                "segments": [],
                "features_used": 0,
            })
        
        # Run predictor (for testing - create a temporary graph-like state)
        from ts_agent.ml.predictor import SegmentPredictor
        predictor = SegmentPredictor()
        hypothesis = predictor.predict(
            traits=known_traits,
            situation_id=graph.situation_id,
        )
        
        return json.dumps({
            "top_segment_id": hypothesis.top_segment_id,
            "top_confidence": round(hypothesis.top_confidence, 4),
            "segments": [
                {"segment_id": seg_id, "score": round(score, 4)}
                for seg_id, score in sorted(
                    hypothesis.all_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            ],
            "features_used": len(known_traits),
            "shap_features": hypothesis.shap_features[:3],
        })
        
    except Exception as e:
        return json.dumps({
            "top_segment_id": None,
            "top_confidence": 0.0,
            "error": str(e),
            "segments": [],
            "features_used": 0,
        })


# Export ML-automatic tools
RECORD_ANSWER_TOOL = _LazyTool(record_consumer_answer_ml_auto)
GET_NEXT_QUESTION_TOOL = _LazyTool(get_next_question_ml_prioritized)
CHECK_COMPLETENESS_TOOL = _LazyTool(check_graph_completeness_ml_aware)
MATCH_SEGMENT_TOOL = _LazyTool(match_segment)
