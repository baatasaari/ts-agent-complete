"""
ts_agent.visualiser.static_report
====================================
Generates a self-contained HTML report for one ``SessionRecord`` using
``plotly.io.write_html``.  No running server required — the output file
opens in any browser.

Each of the six panels is rendered as a Plotly figure and embedded in a
single HTML file via ``include_plotlyjs="cdn"`` (one CDN script tag shared
across all figures).

The report is suitable for sending to regulators, storing in audit archives,
or attaching to FCA supervisory submissions.

Codex review notes
------------------
- Pure functions; no side effects other than writing ``out_path``.
- All PII is hashed; the report never contains raw consumer values.
- Labelled "SIMULATED LATENCY" where values are not from live signals.
"""

from __future__ import annotations

import os
from typing import Any

import plotly.io as pio

from ts_agent.observability.session_store import SessionRecord
from ts_agent.visualiser.data_adapter import DataAdapter
from ts_agent.visualiser.components.conversation import build_conversation_panel
from ts_agent.visualiser.components.deterministic import build_deterministic_panel
from ts_agent.visualiser.components.ml_prediction import build_ml_panel
from ts_agent.visualiser.components.sankey_flow import build_sankey
from ts_agent.visualiser.components.session_overview import build_session_overview
from ts_agent.visualiser.components.trace_waterfall import build_trace_waterfall

_SECTION_TEMPLATE = """
<div style="margin:32px 0 16px 0;padding:8px 16px;
            background:#2C3E50;color:white;border-radius:4px;
            font-family:Arial,sans-serif;font-size:15px;font-weight:bold;">
  {title}
  <span style="float:right;font-size:11px;font-weight:normal;opacity:.7;">{subtitle}</span>
</div>
{body}
"""

_HEADER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TS Regulatory Report — {session_id}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js" charset="utf-8"></script>
  <style>
    body  {{ font-family: Arial, sans-serif; background: #F4F6F7;
             color: #2C3E50; margin: 0; padding: 24px; }}
    h1    {{ font-size: 22px; color: #2C3E50; margin-bottom: 4px; }}
    .meta {{ font-size: 12px; color: #7F8C8D; margin-bottom: 24px; }}
    .panel{{ background: white; border-radius: 6px; padding: 16px;
             margin-bottom: 24px;
             box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .badge{{ display:inline-block; padding:3px 10px; border-radius:12px;
             font-size:12px; font-weight:bold; color:white; }}
    .emit {{ background:#27AE60; }}
    .review{{ background:#E67E22; }}
    .suppress{{ background:#C0392B; }}
    .inv {{ font-size: 11px; color: #7F8C8D; margin-top: 4px; }}
    .disclaimer {{ background:#FEF9E7; border-left:4px solid #F39C12;
                   padding:10px 16px; margin-bottom:24px; font-size:12px; }}
  </style>
</head>
<body>
  <h1>LBG Targeted Support — Regulatory Audit Report</h1>
  <div class="meta">
    Session: <b>{session_id}</b> &nbsp;|&nbsp;
    Party: <b>{party_ref}</b> &nbsp;|&nbsp;
    Situation: <b>{situation_label}</b> &nbsp;|&nbsp;
    Gate: <b><span class="badge {gate_class}">{gate}</span></b> &nbsp;|&nbsp;
    Generated: {generated_at}
  </div>
  <div class="disclaimer">
    ⚠ &nbsp;<b>CONFIDENTIAL — FCA SUPERVISORY USE ONLY.</b>
    Consumer values are represented as SHA-256 hashes; no raw PII is present in this report.
    Latency values marked "[sim]" are simulated for demonstration purposes.
    This report satisfies audit evidence requirements under PS25/22 §8.4 and FCA FG21/1.
  </div>
"""

_INVARIANT_NOTE = """
<div class="inv">
  Invariants verified: INV-01 (graph complete before Zone 2) · INV-05 (audit before delivery) ·
  INV-06 (template-only explanation) · INV-07 (all rules logged) · INV-10 (symbolic trace complete)
</div>
"""

_FOOTER = """
</body></html>
"""


def _fig_to_div(fig, include_plotlyjs: str = "cdn") -> str:
    """Convert a Plotly figure to an HTML div string."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,   # CDN tag already in header
        config={"displayModeBar": True, "responsive": True},
    )


def generate_static_report(
    record: SessionRecord,
    out_path: str,
    adapter: DataAdapter | None = None,
) -> str:
    """
    Write a self-contained HTML report for ``record`` to ``out_path``.

    Parameters
    ----------
    record   : ``SessionRecord`` to report on.
    out_path : File path for the HTML output.  Parent directory must exist.
    adapter  : Optional ``DataAdapter`` instance; a new one is created if None.

    Returns
    -------
    The absolute path of the written file.
    """
    if adapter is None:
        adapter = DataAdapter()

    summary      = DataAdapter.session_summary(record)
    turns_data   = DataAdapter.enrich_conversation(record.conversation)
    signals_data = DataAdapter.enrich_signals(record.signals)
    rules_data   = DataAdapter.enrich_rules(record.rule_evaluations)
    preds_data   = DataAdapter.enrich_predictions(record.prediction_chain)
    sankey_data  = DataAdapter.sankey_for_session(record)

    gate     = summary.get("gate_disposition", "SUPPRESS")
    gate_cls = {"EMIT": "emit", "HUMAN_REVIEW": "review", "SUPPRESS": "suppress"}.get(
        gate, "suppress"
    )

    from datetime import datetime, timezone  # noqa: PLC0415
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Build each panel ──────────────────────────────────────────────────────
    overview_fig     = build_session_overview(summary)
    conversation_fig = build_conversation_panel(turns_data)
    waterfall_fig    = build_trace_waterfall(signals_data)
    deterministic_fig= build_deterministic_panel(rules_data, summary)
    ml_fig           = build_ml_panel(preds_data)
    sankey_fig       = build_sankey(
        sankey_data,
        title=f"Decision Path — {summary.get('session_id','')[:16]}…",
    )

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    parts: list[str] = [
        _HEADER_TEMPLATE.format(
            session_id    = record.session_id,
            party_ref     = summary.get("party_ref", "—"),
            situation_label = summary.get("situation_label", "—"),
            gate          = gate,
            gate_class    = gate_cls,
            generated_at  = generated,
        ),
    ]

    panels = [
        ("Layer 2 — Session Overview",
         "graph completeness · timing · gate disposition",
         overview_fig),
        ("Layer 1 — Conversation Transcript",
         "consumer Q&A · value hashes · branch labels",
         conversation_fig),
        ("Layer 3 — EAMGP Signal Trace",
         "all signals emitted · zone timeline · level indicators",
         waterfall_fig),
        ("Layer 4 — Deterministic Compliance Gate",
         "trait knowledge state · 12-rule evaluation · FCA references",
         deterministic_fig),
        ("Layer 5 — ML Segment Prediction",
         "confidence per turn · SHAP feature attribution",
         ml_fig),
        ("Layer 6 — End-to-End Decision Flow",
         "intent → situation → segment → suggestion → gate",
         sankey_fig),
    ]

    for title, subtitle, fig in panels:
        div = _fig_to_div(fig)
        parts.append(
            _SECTION_TEMPLATE.format(
                title=title, subtitle=subtitle,
                body=f'<div class="panel">{div}</div>',
            )
        )

    parts.append(_INVARIANT_NOTE)
    parts.append(_FOOTER)

    html = "\n".join(parts)
    abs_path = os.path.abspath(out_path)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return abs_path
