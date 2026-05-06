"""EAMGP signal trace waterfall — Layer 3."""
from __future__ import annotations
from typing import Any
import plotly.graph_objects as go


_ZONE_COLOURS = {
    "Zone1":   "#3498DB",
    "Zone1.5": "#9B59B6",
    "Zone2":   "#1ABC9C",
    "Zone3":   "#E67E22",
    "Zone4":   "#E74C3C",
    "Session": "#95A5A6",
    "Infra":   "#7F8C8D",
}
_LEVEL_SYMBOL = {"INFO": "circle", "WARN": "diamond", "ERROR": "x"}
_LEVEL_SIZE   = {"INFO": 10, "WARN": 13, "ERROR": 15}


def build_trace_waterfall(signals: list[dict[str, Any]]) -> go.Figure:
    """
    Gantt-style signal timeline.  Each signal is a marker on the x-axis
    (elapsed ms from session start) positioned on its zone row.
    Hover shows the full signal name and all attributes.
    """
    if not signals:
        fig = go.Figure()
        fig.add_annotation(
            text="No signals recorded.", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#7F8C8D"),
        )
        fig.update_layout(paper_bgcolor="#FAFAFA", height=300)
        return fig

    zones = ["Zone1", "Zone1.5", "Zone2", "Zone3", "Zone4", "Session", "Infra"]
    zone_y = {z: i for i, z in enumerate(zones)}

    traces = []
    # One scatter trace per (zone, level) combination for a clean legend
    for zone in zones:
        for level in ("INFO", "WARN", "ERROR"):
            subset = [s for s in signals if s["zone"] == zone and s["level"] == level]
            if not subset:
                continue
            traces.append(go.Scatter(
                x=[s["elapsed_ms"]  for s in subset],
                y=[zone_y.get(zone, 8) for _ in subset],
                mode="markers+text",
                name=f"{zone} / {level}",
                marker=dict(
                    color=_ZONE_COLOURS.get(zone, "#95A5A6"),
                    symbol=_LEVEL_SYMBOL.get(level, "circle"),
                    size=_LEVEL_SIZE.get(level, 10),
                    line=dict(width=1, color="white"),
                    opacity=0.9,
                ),
                text=[s["signal"] for s in subset],
                textposition="top center",
                textfont=dict(size=8, color="#2C3E50"),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Zone: " + zone + "<br>"
                    "Level: " + level + "<br>"
                    "Elapsed: %{x:.1f} ms<br>"
                    "<extra></extra>"
                ),
                legendgroup=zone,
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#FAFAFA",
        plot_bgcolor="#FFFFFF",
        height=420,
        margin=dict(l=20, r=20, t=40, b=50),
        xaxis=dict(
            title="Elapsed time (ms from session start)",
            showgrid=True, gridcolor="#ECF0F1",
            zeroline=True, zerolinecolor="#BDC3C7",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(zones))),
            ticktext=zones,
            showgrid=True, gridcolor="#ECF0F1",
            range=[-0.5, len(zones) - 0.5],
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35,
            xanchor="left", x=0, font=dict(size=10),
        ),
        hoverlabel=dict(bgcolor="white", font_size=11),
    )
    return fig
