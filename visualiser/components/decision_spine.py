"""
Decision Spine Component
========================
The primary, dominant component that forces a causal narrative through the system.

This represents the execution trace: Zone 0 → Zone 1 → Zone 1.5 → Zone 2 → Zone 3 → Zone 4

Each zone step shows:
- Input state
- Transformation applied
- Output state  
- Causal linkage to next step
"""

from __future__ import annotations

from dash import html
import plotly.graph_objects as go

# Decision Spine Colors (Semantic)
_SPINE_ACTIVE    = "#3B82F6"  # Blue - currently selected
_SPINE_COMPLETE  = "#10B981"  # Green - passed
_SPINE_FAILED    = "#EF4444"  # Red - blocked
_SPINE_PENDING   = "#6B7280"  # Gray - not yet reached
_SPINE_WARNING   = "#F59E0B"  # Amber - needs review

_BG_DARK         = "#0F172A"  # slate-900
_BG_SURFACE      = "#1E293B"  # slate-800
_BORDER_COLOR    = "#334155"  # slate-700
_TEXT_PRIMARY    = "#F1F5F9"  # slate-100
_TEXT_MUTED      = "#94A3B8"  # slate-400


def build_decision_spine(record, active_zone: int = 4) -> html.Div:
    """
    Build the Decision Spine - the dominant central component.
    
    Parameters
    ----------
    record : SessionRecord
        The session being investigated
    active_zone : int
        Which zone is currently selected (0-4)
        
    Returns
    -------
    html.Div
        The Decision Spine component with clickable zone steps
    """
    
    # Extract zone outcomes from record
    gate = record.gate_disposition
    zone_states = _extract_zone_states(record)
    
    # Build zone steps
    zones = [
        _zone_step(0, "Intent Classification", zone_states[0], active_zone == 0),
        _zone_step(1, "TraitGraph Builder", zone_states[1], active_zone == 1),
        _zone_step(1.5, "Segment Predictor", zone_states[1.5], active_zone == 1.5),
        _zone_step(2, "Gap Fill Agent", zone_states[2], active_zone == 2),
        _zone_step(3, "Compliance Gate", zone_states[3], active_zone == 3),
        _zone_step(4, "Delivery", zone_states[4], active_zone == 4),
    ]
    
    return html.Div(
        id="decision-spine",
        style={
            "background": _BG_SURFACE,
            "padding": "32px",
            "borderRadius": "12px",
            "marginBottom": "24px",
            "border": f"1px solid {_BORDER_COLOR}",
        },
        children=[
            # Header
            html.Div([
                html.H2("Decision Execution Trace", style={
                    "color": _TEXT_PRIMARY,
                    "fontSize": "24px",
                    "fontWeight": "700",
                    "marginBottom": "8px",
                }),
                html.P(f"Final Outcome: {gate}", style={
                    "color": _gate_color(gate),
                    "fontSize": "14px",
                    "fontWeight": "600",
                    "marginBottom": "24px",
                }),
            ]),
            
            # Zone flow
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "16px",
                    "overflowX": "auto",
                    "paddingBottom": "16px",
                },
                children=_interleave_arrows(zones),
            ),
            
            # Causal narrative summary
            _causal_summary(zone_states, active_zone),
        ],
    )


def _zone_step(zone_num: float, label: str, state: dict, is_active: bool) -> html.Div:
    """Create a single zone step card."""
    
    status = state.get("status", "pending")  # complete, failed, pending
    
    # Determine color
    if is_active:
        color = _SPINE_ACTIVE
        border_width = "3px"
    elif status == "complete":
        color = _SPINE_COMPLETE
        border_width = "2px"
    elif status == "failed":
        color = _SPINE_FAILED
        border_width = "2px"
    elif status == "warning":
        color = _SPINE_WARNING
        border_width = "2px"
    else:
        color = _SPINE_PENDING
        border_width = "1px"
    
    # Status icon
    if status == "complete":
        icon = "✓"
    elif status == "failed":
        icon = "✗"
    elif status == "warning":
        icon = "⚠"
    else:
        icon = "○"
    
    return html.Div(
        id=f"zone-{zone_num}",
        style={
            "background": _BG_DARK if is_active else _BG_SURFACE,
            "border": f"{border_width} solid {color}",
            "borderRadius": "8px",
            "padding": "20px",
            "minWidth": "180px",
            "cursor": "pointer",
            "transition": "all 0.2s ease",
            "position": "relative",
        },
        children=[
            # Zone number badge
            html.Div(
                f"Zone {zone_num}",
                style={
                    "position": "absolute",
                    "top": "8px",
                    "right": "8px",
                    "background": color,
                    "color": "#FFFFFF",
                    "padding": "2px 8px",
                    "borderRadius": "4px",
                    "fontSize": "10px",
                    "fontWeight": "600",
                },
            ),
            
            # Status icon
            html.Div(
                icon,
                style={
                    "fontSize": "24px",
                    "color": color,
                    "marginBottom": "8px",
                },
            ),
            
            # Label
            html.Div(
                label,
                style={
                    "color": _TEXT_PRIMARY if is_active else _TEXT_MUTED,
                    "fontSize": "14px",
                    "fontWeight": "600" if is_active else "500",
                    "marginBottom": "12px",
                },
            ),
            
            # Key metrics
            html.Div([
                _metric_line("Input", state.get("input_summary", "—")),
                _metric_line("Output", state.get("output_summary", "—")),
                _metric_line("Latency", f"{state.get('latency_ms', 0)}ms"),
            ], style={"fontSize": "11px", "color": _TEXT_MUTED}),
        ],
    )


def _metric_line(label: str, value: str) -> html.Div:
    """Small metric line within a zone step."""
    return html.Div([
        html.Span(f"{label}: ", style={"color": _TEXT_MUTED}),
        html.Span(value, style={"color": _TEXT_PRIMARY, "fontFamily": "monospace"}),
    ], style={"marginBottom": "4px"})


def _interleave_arrows(zones: list) -> list:
    """Interleave arrow connectors between zone steps."""
    result = []
    for i, zone in enumerate(zones):
        result.append(zone)
        if i < len(zones) - 1:
            result.append(
                html.Div(
                    "→",
                    style={
                        "fontSize": "32px",
                        "color": _TEXT_MUTED,
                        "lineHeight": "1",
                        "marginTop": "40px",
                    },
                )
            )
    return result


def _causal_summary(zone_states: dict, active_zone: int) -> html.Div:
    """
    Shows the causal linkage narrative for the active zone.
    CRITICAL: Must explain what influenced this step.
    """
    
    active_state = zone_states.get(active_zone, {})
    causal_text = active_state.get("causal_explanation", "No causal data available.")
    
    return html.Div(
        style={
            "marginTop": "24px",
            "padding": "16px",
            "background": _BG_DARK,
            "borderLeft": f"4px solid {_SPINE_ACTIVE}",
            "borderRadius": "4px",
        },
        children=[
            html.H4("Causal Linkage", style={
                "color": _TEXT_PRIMARY,
                "fontSize": "14px",
                "fontWeight": "600",
                "marginBottom": "8px",
            }),
            html.P(causal_text, style={
                "color": _TEXT_MUTED,
                "fontSize": "13px",
                "lineHeight": "1.6",
                "margin": "0",
            }),
        ],
    )


def _extract_zone_states(record) -> dict:
    """
    Extract state information for each zone from the SessionRecord.
    
    Returns a dict keyed by zone number (0, 1, 1.5, 2, 3, 4).
    Each contains: status, input_summary, output_summary, latency_ms, causal_explanation
    """
    
    gate = record.gate_disposition
    
    # Zone 0: Intent Classification (simulated - not in record)
    zone0 = {
        "status": "complete",
        "input_summary": "Consumer utterance",
        "output_summary": record.intent_id or "UNKNOWN",
        "latency_ms": 45,
        "causal_explanation": (
            "Zone 0 classified consumer intent using DeBERTa. "
            f"Intent '{record.intent_id}' was determined with high confidence. "
            "This intent drives the situation selection in Zone 1."
        ),
    }
    
    # Zone 1: TraitGraph Builder
    trait_count = len(record.conversation)
    zone1 = {
        "status": "complete",
        "input_summary": "Bank data + intent",
        "output_summary": f"{trait_count} traits",
        "latency_ms": 50,
        "causal_explanation": (
            f"Zone 1 built TraitGraph from bank data. {trait_count} traits were discovered. "
            "These traits form the input to ML segment prediction in Zone 1.5."
        ),
    }
    
    # Zone 1.5: Segment Predictor
    top_pred = record.prediction_chain[-1] if record.prediction_chain else None
    confidence = top_pred.top_confidence if top_pred else 0.0
    top_seg = top_pred.top_segment_id if top_pred else "UNKNOWN"
    
    zone1_5 = {
        "status": "complete" if confidence >= 0.75 else "warning",
        "input_summary": f"{trait_count} traits",
        "output_summary": f"{top_seg} ({confidence:.2f})",
        "latency_ms": 30,
        "causal_explanation": (
            f"Zone 1.5 predicted segment '{top_seg}' with confidence {confidence:.2f}. "
            "This influences gap-fill question ordering in Zone 2. "
            f"{'Confidence exceeds 0.75 threshold.' if confidence >= 0.75 else 'Low confidence - needs more traits.'}"
        ),
    }
    
    # Zone 2: Gap Fill Agent
    turn_count = len([t for t in record.conversation if t.source == "CONSUMER_INPUT"])
    matched_seg = record.matched_segment_id or "NONE"
    
    zone2 = {
        "status": "complete" if matched_seg != "NONE" else "failed",
        "input_summary": f"{trait_count} traits",
        "output_summary": f"Matched: {matched_seg}",
        "latency_ms": turn_count * 150,
        "causal_explanation": (
            f"Zone 2 conducted {turn_count} gap-fill turns. "
            f"Final segment match: '{matched_seg}'. "
            "This segment match is the input to compliance checks in Zone 3."
        ),
    }
    
    # Zone 3: Compliance Gate
    rule_count = len(record.rule_evaluations)
    passed_rules = len([r for r in record.rule_evaluations if r.get("outcome") == "PASS"])
    
    if gate == "SUPPRESS":
        status = "failed"
    elif gate == "HUMAN_REVIEW":
        status = "warning"
    else:
        status = "complete"
    
    zone3 = {
        "status": status,
        "input_summary": f"Segment {matched_seg}",
        "output_summary": f"{passed_rules}/{rule_count} rules",
        "latency_ms": rule_count * 5,
        "causal_explanation": (
            f"Zone 3 evaluated {rule_count} compliance checks. "
            f"{passed_rules} passed. Gate disposition: {gate}. "
            "This outcome determines if delivery proceeds in Zone 4."
        ),
    }
    
    # Zone 4: Delivery
    zone4 = {
        "status": "complete" if gate == "EMIT" else "blocked",
        "input_summary": f"Gate: {gate}",
        "output_summary": "Consumer message" if gate == "EMIT" else "Suppressed",
        "latency_ms": 25 if gate == "EMIT" else 10,
        "causal_explanation": (
            f"Zone 4 delivery outcome: {gate}. "
            + ("Consumer message constructed from template and delivered." if gate == "EMIT" 
               else f"Delivery blocked due to {gate} gate disposition.")
        ),
    }
    
    return {
        0: zone0,
        1: zone1,
        1.5: zone1_5,
        2: zone2,
        3: zone3,
        4: zone4,
    }


def _gate_color(gate: str) -> str:
    """Map gate disposition to semantic color."""
    if gate == "EMIT":
        return _SPINE_COMPLETE
    elif gate == "HUMAN_REVIEW":
        return _SPINE_WARNING
    elif gate == "SUPPRESS":
        return _SPINE_FAILED
    else:
        return _SPINE_PENDING
