"""ML prediction chain + SHAP panel — Layer 5."""
from __future__ import annotations
from typing import Any
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_DISPOSITION_COLOUR = {
    "ACTIVE":         "#2ECC71",
    "UNDECIDABLE":    "#95A5A6",
    "FAILED":         "#E74C3C",
    "LOW_CONFIDENCE": "#F39C12",
}


def build_ml_panel(predictions: list[dict[str, Any]]) -> go.Figure:
    """
    Two sub-figures:
    - Left:  line chart of top_confidence across conversation turns
    - Right: SHAP feature importance bar chart (from the final prediction)
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=["Segment Confidence Per Turn", "SHAP Feature Attribution (Final Turn)"],
        horizontal_spacing=0.10,
    )

    if not predictions:
        fig.add_annotation(
            text="No ML predictions recorded.", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=13, color="#7F8C8D"),
        )
        fig.update_layout(paper_bgcolor="#FAFAFA", height=360)
        return fig

    # ── Confidence line chart ─────────────────────────────────────────────────
    turns  = [p["turn"]           for p in predictions]
    confs  = [p["top_confidence"] for p in predictions]
    labels = [p["segment_label"]  for p in predictions]
    colours= [
        _DISPOSITION_COLOUR.get(p["disposition"], "#3498DB")
        for p in predictions
    ]

    fig.add_trace(go.Scatter(
        x=turns, y=confs,
        mode="lines+markers",
        name="Top Segment Confidence",
        line=dict(color="#3498DB", width=2),
        marker=dict(color=colours, size=10, line=dict(color="white", width=1)),
        hovertemplate=(
            "Turn %{x}<br>"
            "Confidence: %{y:.2%}<br>"
            "<extra></extra>"
        ),
        text=labels,
    ), row=1, col=1)

    # Threshold line at 0.75 (R-009 auto-emit threshold)
    fig.add_hline(
        y=0.75, line_dash="dash", line_color="#E74C3C",
        annotation_text="R-009 threshold (0.75)",
        annotation_position="bottom right",
        row=1, col=1,
    )

    # ── SHAP bar chart — use final prediction with valid shap data ────────────
    final = next(
        (p for p in reversed(predictions) if p["disposition"] == "ACTIVE"),
        predictions[-1] if predictions else None,
    )
    if final:
        try:
            shap_data = json.loads(final["shap_json"])
        except (json.JSONDecodeError, TypeError):
            shap_data = []

        if shap_data:
            features = [d.get("f", d.get("feature", "?")) for d in shap_data]
            values   = [abs(d.get("v", d.get("shap_value", 0.0))) for d in shap_data]
            fig.add_trace(go.Bar(
                x=values,
                y=features,
                orientation="h",
                marker_color="#9B59B6",
                name="SHAP |value|",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="#FAFAFA",
        plot_bgcolor="#FFFFFF",
        height=380,
        margin=dict(l=20, r=20, t=50, b=30),
        showlegend=False,
    )
    fig.update_yaxes(title_text="Confidence", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Turn", row=1, col=1)
    fig.update_xaxes(title_text="|SHAP value|", row=1, col=2)
    return fig
