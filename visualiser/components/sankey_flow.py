"""End-to-end Sankey decision flow — Layer 6."""
from __future__ import annotations
from typing import Any
import plotly.graph_objects as go


def build_sankey(sankey_data: dict[str, Any], title: str = "Decision Flow") -> go.Figure:
    """
    Build a Plotly Sankey figure from pre-computed node/link data.

    ``sankey_data`` must have:
        ``labels`` : list[str]  — node labels
        ``links``  : list[dict] — each with source, target, value, colour
    """
    labels = sankey_data.get("labels", [])
    links  = sankey_data.get("links", [])

    if not labels or not links:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data to render decision flow.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color="#7F8C8D"),
        )
        fig.update_layout(paper_bgcolor="#FAFAFA", height=400)
        return fig

    node_colours = []
    gate_node_colours = {
        "EMIT":         "#2ECC71",
        "HUMAN_REVIEW": "#F39C12",
        "SUPPRESS":     "#E74C3C",
    }
    for label in labels:
        node_colours.append(gate_node_colours.get(label, "#3498DB"))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=24,
            line=dict(color="#2C3E50", width=0.5),
            label=labels,
            color=node_colours,
            hovertemplate="%{label}<br>Flow: %{value}<extra></extra>",
        ),
        link=dict(
            source=[lk["source"] for lk in links],
            target=[lk["target"] for lk in links],
            value= [lk["value"]  for lk in links],
            color= [lk["colour"] for lk in links],
            hovertemplate=(
                "%{source.label} → %{target.label}<br>"
                "Sessions: %{value}<extra></extra>"
            ),
        ),
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#2C3E50")),
        paper_bgcolor="#FAFAFA",
        height=420,
        font=dict(size=11, color="#2C3E50"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
