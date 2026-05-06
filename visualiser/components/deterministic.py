"""Deterministic Zone 2 trait heatmap + Zone 3 rule table — Layer 4."""
from __future__ import annotations
from typing import Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_HEX_TO_RGBA = {
    "#2ECC71": "rgba(46,204,113,0.27)",
    "#E74C3C": "rgba(231,76,60,0.27)",
    "#F39C12": "rgba(243,156,18,0.27)",
    "#BDC3C7": "rgba(189,195,199,0.27)",
}
_OUTCOME_SYMBOL = {"PASS": "✓", "FAIL": "✗", "GATE": "⚠", "NOT_REACHED": "—"}
_TYPE_BADGE     = {"HARD": "#E74C3C", "GATE": "#F39C12", "SOFT": "#3498DB"}


def build_deterministic_panel(
    rules: list[dict[str, Any]],
    summary: dict[str, Any],
) -> go.Figure:
    """
    Two-part figure:
    - Top: trait completeness bar (known / missing / excluded counts)
    - Bottom: 12-rule compliance table with FCA refs and outcome badges
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.25, 0.75],
        vertical_spacing=0.06,
        subplot_titles=["Trait Graph — Knowledge State", "12-Rule Compliance Gate"],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    # ── Trait bar ─────────────────────────────────────────────────────────────
    known    = summary.get("known_traits", 0)
    missing  = summary.get("missing_traits", 0)
    excluded = summary.get("excluded_traits", 0)

    for label, val, colour in [
        ("KNOWN",    known,    "#2ECC71"),
        ("MISSING",  missing,  "#E74C3C"),
        ("EXCLUDED", excluded, "#BDC3C7"),
    ]:
        fig.add_trace(go.Bar(
            x=[val], y=["Traits"],
            orientation="h",
            name=label,
            marker_color=colour,
            text=[f"{label}: {val}"],
            textposition="inside" if val > 0 else "none",
            insidetextanchor="middle",
            hovertemplate=f"{label}: {val}<extra></extra>",
        ), row=1, col=1)

    # ── Rule table ────────────────────────────────────────────────────────────
    row_colours: list[str] = []
    for r in rules:
        oc  = r.get("outcome", "NOT_REACHED")
        hex_c = {"PASS": "#2ECC71", "FAIL": "#E74C3C",
                 "GATE": "#F39C12", "NOT_REACHED": "#BDC3C7"}.get(oc, "#BDC3C7")
        row_colours.append(_HEX_TO_RGBA.get(hex_c, "rgba(189,195,199,0.27)"))

    fig.add_trace(go.Table(
        columnwidth=[60, 60, 280, 120, 80, 70],
        header=dict(
            values=[
                "<b>Rule ID</b>", "<b>Type</b>", "<b>Description</b>",
                "<b>FCA Reference</b>", "<b>Outcome</b>", "<b>Symbol</b>",
            ],
            fill_color="#2C3E50",
            font=dict(color="white", size=12),
            align="left", height=32,
        ),
        cells=dict(
            values=[
                [r["rule_id"]     for r in rules],
                [r["rule_type"]   for r in rules],
                [r["description"] for r in rules],
                [r["fca_ref"]     for r in rules],
                [r["outcome"]     for r in rules],
                [_OUTCOME_SYMBOL.get(r["outcome"], "—") for r in rules],
            ],
            fill_color=row_colours,  # one colour per row — plotly 6 compatible
            font=dict(color="#2C3E50", size=11),
            align="left", height=28,
        ),
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#FAFAFA",
        height=560,
        margin=dict(l=10, r=10, t=60, b=10),
        barmode="stack",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, row=1, col=1)
    return fig
