"""Session overview panel — Layer 2."""
from __future__ import annotations
from typing import Any
import plotly.graph_objects as go


def build_session_overview(summary: dict[str, Any]) -> go.Figure:
    """
    Gauge + KPI cards for the session overview.
    Returns a Plotly Figure with a completeness gauge and key metrics.
    """
    gate = summary.get("gate_disposition", "SUPPRESS")
    gate_colour = {"EMIT": "#2ECC71", "HUMAN_REVIEW": "#F39C12", "SUPPRESS": "#E74C3C"}.get(
        gate, "#95A5A6"
    )
    completeness = summary.get("completeness_pct", 0.0)

    fig = go.Figure()

    # Completeness gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=completeness,
        number={"suffix": "%", "font": {"size": 36, "color": "#2C3E50"}},
        delta={"reference": 90, "suffix": "%"},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#7F8C8D"},
            "bar":   {"color": "#3498DB"},
            "steps": [
                {"range": [0, 50],  "color": "#FADBD8"},
                {"range": [50, 90], "color": "#FEF9E7"},
                {"range": [90, 100],"color": "#D5F5E3"},
            ],
            "threshold": {
                "line":  {"color": "#27AE60", "width": 4},
                "thickness": 0.85,
                "value": 90,
            },
        },
        title={"text": "Graph Completeness", "font": {"size": 16}},
        domain={"x": [0, 0.45], "y": [0, 1]},
    ))

    # Gate disposition indicator
    fig.add_trace(go.Indicator(
        mode="number",
        value=1,
        number={"valueformat": "", "font": {"size": 24, "color": gate_colour}},
        title={"text": f"Gate: <b>{gate}</b>", "font": {"size": 18, "color": gate_colour}},
        domain={"x": [0.55, 1.0], "y": [0.5, 1.0]},
    ))

    # Turn count
    fig.add_trace(go.Indicator(
        mode="number",
        value=summary.get("gap_fill_turns", 0),
        title={"text": "Conversation Turns", "font": {"size": 14, "color": "#7F8C8D"}},
        number={"font": {"size": 28, "color": "#2C3E50"}},
        domain={"x": [0.55, 1.0], "y": [0.0, 0.45]},
    ))

    fig.update_layout(
        paper_bgcolor="#FAFAFA",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        annotations=[
            dict(
                text=(
                    f"<b>Session:</b> {summary.get('session_id','')[:16]}…  "
                    f"| <b>Party:</b> {summary.get('party_ref','')}  "
                    f"| <b>Situation:</b> {summary.get('situation_label','')}  "
                    f"| <b>Segment:</b> {summary.get('segment_label','')}  "
                    f"| <b>Signals:</b> {summary.get('signal_count', 0)}  "
                    f"| <b>Duration:</b> {summary.get('duration_s', 0):.2f}s"
                ),
                xref="paper", yref="paper",
                x=0, y=1.12, showarrow=False,
                font=dict(size=11, color="#555"),
                align="left",
            )
        ],
    )
    return fig
