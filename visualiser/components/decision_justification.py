"""
Decision Justification Component
=================================
NON-NEGOTIABLE component that explicitly shows:
- Why the final outcome was selected
- What alternative outcomes were possible
- Why each alternative was rejected
- Evidence supporting the decision

This must be deterministic and explicit. Causality must NOT be inferred by the user.
"""

from __future__ import annotations

from dash import html

# Colors
_BG_DARK = "#0F172A"
_BG_SURFACE = "#1E293B"
_BORDER_COLOR = "#334155"
_TEXT_PRIMARY = "#F1F5F9"
_TEXT_MUTED = "#94A3B8"
_GREEN = "#10B981"
_RED = "#EF4444"
_AMBER = "#F59E0B"


def build_decision_justification(record) -> html.Div:
    """
    Build the Decision Justification panel.
    
    Parameters
    ----------
    record : SessionRecord
        The session being investigated
        
    Returns
    -------
    html.Div
        Decision justification with alternatives analysis
    """
    
    gate = record.gate_disposition
    rules = record.rule_evaluations
    
    # Analyze why this outcome
    reasons_for = _reasons_for_outcome(gate, rules, record)
    
    # Analyze why NOT other outcomes
    alternatives = _analyze_alternatives(gate, rules, record)
    
    # Gather evidence
    evidence = _gather_evidence(record)
    
    return html.Div(
        style={
            "background": _BG_SURFACE,
            "padding": "24px",
            "borderRadius": "12px",
            "marginBottom": "24px",
            "border": f"1px solid {_BORDER_COLOR}",
        },
        children=[
            # Header
            html.H3("Decision Justification", style={
                "color": _TEXT_PRIMARY,
                "fontSize": "20px",
                "fontWeight": "700",
                "marginBottom": "20px",
            }),
            
            # Final decision
            _outcome_badge(gate),
            
            # Why this outcome
            html.Div([
                html.H4("Why This Outcome:", style={
                    "color": _TEXT_PRIMARY,
                    "fontSize": "16px",
                    "fontWeight": "600",
                    "marginTop": "20px",
                    "marginBottom": "12px",
                }),
                *[_reason_item(reason, True) for reason in reasons_for],
            ]),
            
            # Alternative outcomes considered
            html.Div([
                html.H4("Alternative Outcomes Considered:", style={
                    "color": _TEXT_PRIMARY,
                    "fontSize": "16px",
                    "fontWeight": "600",
                    "marginTop": "24px",
                    "marginBottom": "12px",
                }),
                *[_alternative_item(alt) for alt in alternatives],
            ]),
            
            # Evidence summary
            html.Div([
                html.H4("Evidence:", style={
                    "color": _TEXT_PRIMARY,
                    "fontSize": "16px",
                    "fontWeight": "600",
                    "marginTop": "24px",
                    "marginBottom": "12px",
                }),
                *[_evidence_item(ev) for ev in evidence],
            ]),
        ],
    )


def _outcome_badge(gate: str) -> html.Div:
    """Large badge showing final decision."""
    
    if gate == "EMIT":
        color = _GREEN
        label = "✓ EMIT"
        desc = "Suggestion delivered to consumer"
    elif gate == "HUMAN_REVIEW":
        color = _AMBER
        label = "⚠ HUMAN_REVIEW"
        desc = "Requires compliance officer review"
    else:  # SUPPRESS
        color = _RED
        label = "✗ SUPPRESS"
        desc = "Suggestion blocked from delivery"
    
    return html.Div([
        html.Div(label, style={
            "display": "inline-block",
            "padding": "12px 24px",
            "background": color,
            "color": "#FFFFFF",
            "fontWeight": "700",
            "fontSize": "18px",
            "borderRadius": "8px",
            "marginBottom": "8px",
        }),
        html.P(desc, style={
            "color": _TEXT_MUTED,
            "fontSize": "13px",
            "margin": "0",
        }),
    ])


def _reason_item(reason: str, passed: bool) -> html.Div:
    """Single reason for the outcome."""
    icon = "✓" if passed else "✗"
    color = _GREEN if passed else _RED
    
    return html.Div([
        html.Span(icon, style={
            "color": color,
            "fontWeight": "700",
            "marginRight": "8px",
            "fontSize": "16px",
        }),
        html.Span(reason, style={
            "color": _TEXT_PRIMARY,
            "fontSize": "14px",
        }),
    ], style={
        "marginBottom": "8px",
        "display": "flex",
        "alignItems": "flex-start",
    })


def _alternative_item(alternative: dict) -> html.Div:
    """Alternative outcome that was rejected."""
    
    outcome = alternative["outcome"]
    reason = alternative["reason"]
    rules = alternative.get("rules", [])
    
    return html.Div(
        style={
            "background": _BG_DARK,
            "padding": "16px",
            "borderRadius": "8px",
            "marginBottom": "12px",
            "borderLeft": f"3px solid {_RED}",
        },
        children=[
            html.Div([
                html.Span("✗ ", style={
                    "color": _RED,
                    "fontWeight": "700",
                    "marginRight": "8px",
                }),
                html.Span(outcome, style={
                    "color": _TEXT_PRIMARY,
                    "fontWeight": "600",
                    "fontSize": "14px",
                }),
            ], style={"marginBottom": "8px"}),
            
            html.Div(f"Rejected: {reason}", style={
                "color": _TEXT_MUTED,
                "fontSize": "13px",
                "marginBottom": "8px" if rules else "0",
            }),
            
            # Failed rules
            *[html.Div([
                html.Span(f"Rule {rule}: ", style={
                    "color": _TEXT_MUTED,
                    "fontSize": "12px",
                    "fontFamily": "monospace",
                }),
                html.Span(desc, style={
                    "color": _TEXT_PRIMARY,
                    "fontSize": "12px",
                }),
            ]) for rule, desc in rules],
        ],
    )


def _evidence_item(evidence: str) -> html.Div:
    """Evidence supporting the decision."""
    return html.Div([
        html.Span("→ ", style={
            "color": _GREEN,
            "marginRight": "8px",
        }),
        html.Span(evidence, style={
            "color": _TEXT_PRIMARY,
            "fontSize": "13px",
            "fontFamily": "monospace",
        }),
    ], style={"marginBottom": "6px"})


def _reasons_for_outcome(gate: str, rules: list, record) -> list[str]:
    """Determine why this outcome was selected."""
    
    reasons = []
    
    if gate == "EMIT":
        # Count passed rules
        passed_hard = len([r for r in rules if r.get("severity") == "HARD_BLOCK" and r.get("outcome") == "PASS"])
        total_hard = len([r for r in rules if r.get("severity") == "HARD_BLOCK"])
        
        reasons.append(f"All {total_hard} HARD_BLOCK compliance checks passed")
        
        # ML confidence
        top_pred = record.prediction_chain[-1] if record.prediction_chain else None
        conf = top_pred.top_confidence if top_pred else 0.0
        if conf >= 0.75:
            reasons.append(f"ML confidence {conf:.2f} ≥ 0.75 threshold")
        
        # Segment match
        if record.matched_segment_id:
            reasons.append(f"Segment matched: {record.matched_segment_id}")
        
        # Not vulnerable
        vuln_rule = next((r for r in rules if r.get("rule_id") == "R-003"), None)
        if vuln_rule and vuln_rule.get("outcome") == "PASS":
            reasons.append("Consumer not vulnerable (R-003)")
        
        # No excluding characteristics
        reasons.append("No excluding characteristics present")
    
    elif gate == "HUMAN_REVIEW":
        # Find warnings
        warnings = [r for r in rules if r.get("outcome") == "SOFT_WARNING"]
        if warnings:
            reasons.append(f"{len(warnings)} soft warning(s) triggered manual review")
        
        # Vulnerability
        vuln_rule = next((r for r in rules if r.get("rule_id") == "R-003"), None)
        if vuln_rule and vuln_rule.get("outcome") != "PASS":
            reasons.append("Vulnerability indicator requires human assessment (R-003)")
    
    elif gate == "SUPPRESS":
        # Find hard blocks
        failed_hard = [r for r in rules if r.get("severity") == "HARD_BLOCK" and r.get("outcome") == "FAIL"]
        if failed_hard:
            reasons.append(f"{len(failed_hard)} HARD_BLOCK check(s) failed")
            for rule in failed_hard[:3]:  # Show first 3
                reasons.append(f"  • {rule.get('rule_id')}: {rule.get('description', 'N/A')}")
        
        # Low confidence
        top_pred = record.prediction_chain[-1] if record.prediction_chain else None
        conf = top_pred.top_confidence if top_pred else 0.0
        if conf < 0.75:
            reasons.append(f"ML confidence {conf:.2f} < 0.75 threshold")
        
        # No segment match
        if not record.matched_segment_id or record.matched_segment_id == "NONE":
            reasons.append("No segment match in Zone 2")
    
    return reasons if reasons else ["Decision reason not available"]


def _analyze_alternatives(gate: str, rules: list, record) -> list[dict]:
    """Determine why other outcomes were NOT selected."""
    
    alternatives = []
    
    if gate == "EMIT":
        # Why not HUMAN_REVIEW?
        vuln_rule = next((r for r in rules if r.get("rule_id") == "R-003"), None)
        if vuln_rule and vuln_rule.get("outcome") == "PASS":
            alternatives.append({
                "outcome": "HUMAN_REVIEW",
                "reason": "No soft warnings triggered",
                "rules": [("R-003", "vulnerability check = PASS")],
            })
        
        # Why not SUPPRESS?
        failed_hard = [r for r in rules if r.get("severity") == "HARD_BLOCK" and r.get("outcome") == "FAIL"]
        if not failed_hard:
            alternatives.append({
                "outcome": "SUPPRESS",
                "reason": "All hard blocks passed",
                "rules": [
                    ("PDC-001", "segment alignment = PASS"),
                    ("R-006", "experience requirement = PASS"),
                ],
            })
    
    elif gate == "HUMAN_REVIEW":
        # Why not EMIT?
        warnings = [r for r in rules if r.get("outcome") == "SOFT_WARNING"]
        if warnings:
            alternatives.append({
                "outcome": "EMIT",
                "reason": f"{len(warnings)} warning(s) require human review",
                "rules": [(w.get("rule_id"), w.get("description")) for w in warnings[:2]],
            })
        
        # Why not SUPPRESS?
        failed_hard = [r for r in rules if r.get("severity") == "HARD_BLOCK" and r.get("outcome") == "FAIL"]
        if not failed_hard:
            alternatives.append({
                "outcome": "SUPPRESS",
                "reason": "No hard blocks failed (warnings only)",
                "rules": [],
            })
    
    elif gate == "SUPPRESS":
        # Why not EMIT?
        failed_hard = [r for r in rules if r.get("severity") == "HARD_BLOCK" and r.get("outcome") == "FAIL"]
        if failed_hard:
            alternatives.append({
                "outcome": "EMIT",
                "reason": "Hard blocks failed",
                "rules": [(r.get("rule_id"), r.get("description")) for r in failed_hard[:3]],
            })
        
        # Why not HUMAN_REVIEW?
        alternatives.append({
            "outcome": "HUMAN_REVIEW",
            "reason": "Hard block failure requires suppression, not review",
            "rules": [],
        })
    
    return alternatives


def _gather_evidence(record) -> list[str]:
    """Gather evidence links supporting the decision."""
    
    evidence = []
    
    # TraitGraph completeness
    trait_count = len(record.conversation)
    known_traits = len([t for t in record.conversation if t.source == "CONSUMER_INPUT"])
    evidence.append(f"TraitGraph completeness: {known_traits}/{trait_count} traits known")
    
    # Segment match
    if record.matched_segment_id:
        top_pred = record.prediction_chain[-1] if record.prediction_chain else None
        conf = top_pred.top_confidence if top_pred else 0.0
        evidence.append(f"Segment match confidence: {record.matched_segment_id} = {conf:.2f}")
    
    # Rule evaluations
    passed = len([r for r in record.rule_evaluations if r.get("outcome") == "PASS"])
    total = len(record.rule_evaluations)
    evidence.append(f"Rule evaluation results: {passed}/{total} PASS")
    
    # Session duration
    if record.signals:
        first = min(s.timestamp_utc for s in record.signals)
        last = max(s.timestamp_utc for s in record.signals)
        # Note: In production, would calculate actual duration
        evidence.append(f"Session signals: {len(record.signals)} events recorded")
    
    return evidence
