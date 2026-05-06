"""Conversation transcript panel — Layer 1."""
from __future__ import annotations
from typing import Any
import plotly.graph_objects as go


def build_conversation_panel(turns: list[dict[str, Any]]) -> go.Figure:
    """
    Styled table showing each Q&A turn in the consumer conversation.
    Value hashes are shown, never raw values — PII-safe.
    """
    if not turns:
        fig = go.Figure()
        fig.add_annotation(
            text="No conversation turns recorded for this session.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#7F8C8D"),
        )
        fig.update_layout(paper_bgcolor="#FAFAFA", height=300)
        return fig

    branch_colour = {
        "Personal":    "#D6EAF8",
        "Financial":   "#D5F5E3",
        "Behavioural": "#FEF9E7",
        "Unknown":     "#F2F3F4",
    }
    fill_colours = [
        [branch_colour.get(t["branch"], "#F2F3F4") for t in turns]
    ]

    fig = go.Figure(data=[go.Table(
        columnwidth=[50, 120, 350, 220, 80, 80],
        header=dict(
            values=[
                "<b>Turn</b>", "<b>Trait (char_id)</b>",
                "<b>Question asked to consumer</b>",
                "<b>Answer hash (SHA-256, first 16)</b>",
                "<b>Branch</b>", "<b>Elapsed ms</b>",
            ],
            fill_color="#2C3E50",
            font=dict(color="white", size=12),
            align="left",
            height=32,
        ),
        cells=dict(
            values=[
                [t["turn"]        for t in turns],
                [t["char_id"]     for t in turns],
                [t["question_text"] for t in turns],
                [
                    (t["value_hash"][:16] + "…") if t["value_hash"] != "—" else "—"
                    for t in turns
                ],
                [t["branch"]      for t in turns],
                [t["elapsed_ms"]  for t in turns],
            ],
            fill_color=fill_colours * 6,
            font=dict(color="#2C3E50", size=11),
            align="left",
            height=28,
        ),
    )])
    fig.update_layout(
        paper_bgcolor="#FAFAFA",
        margin=dict(l=0, r=0, t=30, b=0),
        height=max(300, 60 + 30 * len(turns)),
    )
    return fig
