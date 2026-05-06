"""
ts_agent.visualiser.app
========================
Dash application — LBG Targeted Support Regulatory Visualiser.

Architecture
------------
``create_app(store)`` is the factory function.  It returns a configured
``dash.Dash`` instance that reads exclusively from the injected
``SessionStore`` — no direct database calls, no signals import.

The app has two views:
1. **Session detail** — party-ref dropdown → session selector → six panels
2. **Aggregate Sankey** — population flow across all sessions in the store

Running
-------
    from ts_agent.visualiser.app import create_app, build_demo_store
    store = build_demo_store()    # populates with all 19 test scenarios
    app   = create_app(store)
    app.run(debug=False, port=8050)

Codex review notes
------------------
- No import of ``tests/`` at module level.  ``build_demo_store`` imports
  lazily (inside the function) so the module is importable without the
  test tree on PYTHONPATH.
- All callbacks are pure functions of their inputs and the injected store.
- No authentication; a disclaimer banner makes scope explicit.
- Designed for read-only regulatory inspection; no write endpoints.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from ts_agent.observability.session_store import SessionRecord, SessionStore
from ts_agent.visualiser.data_adapter import DataAdapter
from ts_agent.visualiser.components.conversation import build_conversation_panel
from ts_agent.visualiser.components.deterministic import build_deterministic_panel
from ts_agent.visualiser.components.ml_prediction import build_ml_panel
from ts_agent.visualiser.components.sankey_flow import build_sankey
from ts_agent.visualiser.components.session_overview import build_session_overview
from ts_agent.visualiser.components.trace_waterfall import build_trace_waterfall
from ts_agent.visualiser.components.decision_spine import build_decision_spine
from ts_agent.visualiser.components.decision_justification import build_decision_justification

# ── Brand Colours (Scottish Widows / LBG) ────────────────────────────────────
_SW_TEAL      = "#006B6E"  # Scottish Widows primary teal
_SW_ORANGE    = "#E86C25"  # Scottish Widows accent orange
_LBG_CYAN     = "#009CA6"  # LBG primary cyan
_LBG_GREEN    = "#00A758"  # LBG secondary green
_NAVY         = "#003C5A"  # Deep navy for headers
_DARK_BG      = "#0A1929"  # Rich dark background
_CARD_BG      = "#FFFFFF"  # Clean white cards
_SUBTLE_GRAY  = "#F8F9FA"  # Very light gray for alternating
_TEXT_PRIMARY = "#212529"  # Primary text
_TEXT_MUTED   = "#6C757D"  # Muted text
_BORDER_LIGHT = "#E9ECEF"  # Light borders


# ──────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sidebar(party_options: list[dict]) -> html.Div:
    return html.Div(
        style={
            "width": "320px", "minHeight": "100vh",
            "background": f"linear-gradient(180deg, {_DARK_BG} 0%, {_NAVY} 100%)",
            "padding": "32px 24px",
            "position": "fixed", "top": 0, "left": 0, "bottom": 0,
            "overflowY": "auto", "zIndex": 100,
            "boxShadow": "4px 0 16px rgba(0, 0, 0, 0.15)",
        },
        children=[
            # Brand Header
            html.Div([
                html.Div("SCOTTISH WIDOWS", style={
                    "color": _SW_TEAL, "fontWeight": "800",
                    "fontSize": "11px", "letterSpacing": "2px",
                    "marginBottom": "6px", "textTransform": "uppercase",
                }),
                html.Div("Lloyds Banking Group", style={
                    "color": _LBG_CYAN, "fontWeight": "600",
                    "fontSize": "10px", "letterSpacing": "1.5px",
                    "marginBottom": "20px", "textTransform": "uppercase",
                    "opacity": "0.9",
                }),
            ]),
            html.H1("Regulatory Visualiser", style={
                "color": "#FFFFFF", "fontSize": "22px", "marginBottom": "8px",
                "fontWeight": "700", "lineHeight": "1.2",
            }),
            html.P("FCA Compliance Dashboard", style={
                "color": "#B0BEC5", "fontSize": "13px", "marginBottom": "32px",
                "fontWeight": "400",
            }),
            
            html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "0 0 24px 0"}),

            # Consumer Selection
            html.Label("Consumer (party_ref)", style={
                "color": "#E0E0E0", "fontSize": "11px", "fontWeight": "600",
                "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "8px",
                "display": "block",
            }),
            dcc.Dropdown(
                id="party-dropdown",
                options=party_options,
                placeholder="Select consumer…",
                style={
                    "marginBottom": "20px", "fontSize": "13px",
                    "borderRadius": "6px",
                },
                clearable=False,
            ),

            # Session Selection
            html.Label("Session", style={
                "color": "#E0E0E0", "fontSize": "11px", "fontWeight": "600",
                "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "8px",
                "display": "block",
            }),
            dcc.Dropdown(
                id="session-dropdown",
                placeholder="Select session…",
                style={
                    "marginBottom": "28px", "fontSize": "13px",
                    "borderRadius": "6px",
                },
                clearable=False,
            ),

            html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "0 0 24px 0"}),
            
            # Aggregate Toggle
            html.Label("View Options", style={
                "color": "#E0E0E0", "fontSize": "11px", "fontWeight": "600",
                "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "12px",
                "display": "block",
            }),
            dcc.Checklist(
                id="aggregate-toggle",
                options=[{"label": " Show Population Sankey", "value": "agg"}],
                value=[],
                style={
                    "color": "#FFFFFF", "fontSize": "13px", "marginBottom": "24px",
                    "fontWeight": "400",
                },
            ),

            html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "0 0 20px 0"}),
            
            # Session Meta
            html.Div(id="session-meta", style={
                "color": "#90A4AE", "fontSize": "11px", "lineHeight": "1.6",
                "padding": "12px", "background": "rgba(255,255,255,0.05)",
                "borderRadius": "6px", "fontFamily": "monospace",
            }),
        ],
    )


def _disclaimer() -> html.Div:
    return html.Div(
        style={
            "background": f"linear-gradient(135deg, {_SW_ORANGE} 0%, {_SW_TEAL} 100%)",
            "padding": "16px 24px",
            "borderRadius": "8px",
            "marginBottom": "24px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
        },
        children=[
            html.Div([
                html.Span("⚠ ", style={"fontSize": "18px", "marginRight": "8px"}),
                html.B("CONFIDENTIAL — FCA SUPERVISORY USE ONLY", style={"color": "#FFFFFF"}),
            ], style={"marginBottom": "8px"}),
            html.P([
                "Consumer values are represented as SHA-256 hashes; no raw PII is present. "
                "This interface is a read-only regulatory audit tool. "
                "Access should be restricted to authorised FCA supervisors and LBG compliance officers. "
                "Production deployments must sit behind Identity-Aware Proxy (IAP) or equivalent SSO.",
            ], style={"fontSize": "12px", "color": "#FFFFFF", "margin": "0", "opacity": "0.95"}),
        ],
    )


def _panel_card(title: str, subtitle: str, panel_id: str) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div([
                    html.Span(title, style={
                        "fontWeight": "700", "fontSize": "15px", "color": _NAVY,
                    }),
                    html.Span(f" · {subtitle}", style={
                        "color": _TEXT_MUTED, "fontSize": "12px", "fontWeight": "400",
                    }),
                ]),
                style={
                    "background": _CARD_BG,
                    "padding": "16px 20px",
                    "borderBottom": f"2px solid {_SW_TEAL}",
                },
            ),
            dbc.CardBody(
                dcc.Graph(id=panel_id, config={
                    "displayModeBar": True,
                    "responsive": True,
                    "displaylogo": False,
                }),
                style={"padding": "16px"},
            ),
        ],
        style={
            "marginBottom": "20px",
            "borderRadius": "10px",
            "border": "none",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
            "overflow": "hidden",
        },
    )


def _content_area() -> html.Div:
    return html.Div(
        style={
            "marginLeft": "336px",
            "padding": "32px",
            "minHeight": "100vh",
            "background": _SUBTLE_GRAY,
        },
        children=[
            _disclaimer(),
            html.Div(
                id="no-selection-msg",
                style={
                    "color": _TEXT_MUTED,
                    "fontSize": "15px",
                    "textAlign": "center",
                    "padding": "48px 24px",
                    "background": _CARD_BG,
                    "borderRadius": "10px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                },
            ),

            # Session detail panels (hidden until a session is selected)
            html.Div(
                id="session-panels",
                style={"display": "none"},
                children=[
                    # NEW: Decision Spine & Justification (at the top)
                    html.Div(id="panel-decision-spine", style={"marginBottom": "20px"}),
                    html.Div(id="panel-decision-justification", style={"marginBottom": "20px"}),
                    
                    # Original panels
                    _panel_card("Layer 2 — Session Overview",
                                "completeness · timing · gate disposition",
                                "panel-overview"),
                    _panel_card("Layer 1 — Conversation Transcript",
                                "consumer Q&A · value hashes · branch",
                                "panel-conversation"),
                    _panel_card("Layer 3 — EAMGP Signal Trace",
                                "all 63 signal types · zone timeline · error indicators",
                                "panel-trace"),
                    _panel_card("Layer 4 — Deterministic Compliance Gate",
                                "trait knowledge state · 12 rules · FCA references",
                                "panel-deterministic"),
                    _panel_card("Layer 5 — ML Segment Prediction",
                                "confidence per turn · SHAP attribution",
                                "panel-ml"),
                    _panel_card("Layer 6 — Decision Flow",
                                "single session path / aggregate population flow",
                                "panel-sankey"),
                ],
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────

def create_app(store: SessionStore) -> dash.Dash:
    """
    Create and configure the Dash application.

    Parameters
    ----------
    store :
        Populated ``SessionStore``.  The app reads from this store on every
        callback; the store is not modified.
    """
    party_options = [
        {"label": p, "value": p}
        for p in store.index.all_party_refs()
    ]

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
        ],
        title="Scottish Widows · LBG TS — Regulatory Visualiser",
        suppress_callback_exceptions=True,
    )
    app.layout = html.Div(
        style={
            "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            "margin": 0,
            "padding": 0,
        },
        children=[
            _sidebar(party_options),
            _content_area(),
        ],
    )

    # ── Callback 1: populate session dropdown from party selection ────────────
    @app.callback(
        Output("session-dropdown", "options"),
        Output("session-dropdown", "value"),
        Input("party-dropdown", "value"),
    )
    def update_session_list(party_ref: str | None):
        if not party_ref:
            return [], None
        session_ids = store.index.sessions_for(party_ref)
        options = [
            {"label": f"{sid[:16]}… ({_session_gate(store, sid)})", "value": sid}
            for sid in session_ids
        ]
        return options, (session_ids[0] if session_ids else None)

    # ── Callback 2: render all panels (including new decision components) ─────
    @app.callback(
        Output("session-panels",              "style"),
        Output("no-selection-msg",            "children"),
        Output("session-meta",                "children"),
        Output("panel-decision-spine",        "children"),
        Output("panel-decision-justification", "children"),
        Output("panel-overview",              "figure"),
        Output("panel-conversation",          "figure"),
        Output("panel-trace",                 "figure"),
        Output("panel-deterministic",         "figure"),
        Output("panel-ml",                    "figure"),
        Output("panel-sankey",                "figure"),
        Input("session-dropdown",             "value"),
        Input("aggregate-toggle",             "value"),
    )
    def render_panels(session_id: str | None, aggregate_toggle: list):
        import plotly.graph_objects as go  # noqa: PLC0415

        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor="#FAFAFA", height=300)

        if not session_id:
            return (
                {"display": "none"},
                "← Select a consumer and session from the sidebar to begin.",
                "",
                html.Div(),  # empty decision spine
                html.Div(),  # empty decision justification
                *([empty_fig] * 6),
            )

        record = store.get_record(session_id)
        if record is None:
            return (
                {"display": "none"},
                f"Session '{session_id}' not found in store.",
                "",
                html.Div(),  # empty decision spine
                html.Div(),  # empty decision justification
                *([empty_fig] * 6),
            )

        summary      = DataAdapter.session_summary(record)
        turns_data   = DataAdapter.enrich_conversation(record.conversation)
        signals_data = DataAdapter.enrich_signals(record.signals)
        rules_data   = DataAdapter.enrich_rules(record.rule_evaluations)
        preds_data   = DataAdapter.enrich_predictions(record.prediction_chain)

        # Sankey: aggregate or per-session
        if "agg" in (aggregate_toggle or []):
            sankey_data = DataAdapter.sankey_aggregate(store.all_records())
            sankey_title = f"Population Flow — {store.record_count()} sessions"
        else:
            sankey_data  = DataAdapter.sankey_for_session(record)
            sankey_title = f"Decision Path — {session_id[:16]}…"

        gate = summary.get("gate_disposition", "SUPPRESS")
        meta = (
            f"Gate: {gate} | "
            f"Signals: {summary.get('signal_count', 0)} | "
            f"Errors: {summary.get('error_count', 0)} | "
            f"Turns: {summary.get('gap_fill_turns', 0)} | "
            f"Duration: {summary.get('duration_s', 0):.2f}s"
        )

        return (
            {"display": "block"},
            "",
            meta,
            build_decision_spine(record),              # NEW
            build_decision_justification(record),      # NEW
            build_session_overview(summary),
            build_conversation_panel(turns_data),
            build_trace_waterfall(signals_data),
            build_deterministic_panel(rules_data, summary),
            build_ml_panel(preds_data),
            build_sankey(sankey_data, title=sankey_title),
        )

    return app


def _session_gate(store: SessionStore, session_id: str) -> str:
    r = store.get_record(session_id)
    return r.gate_disposition if r else "?"


# ──────────────────────────────────────────────────────────────────────────────
# Demo store factory
# ──────────────────────────────────────────────────────────────────────────────

def build_demo_store() -> SessionStore:
    """
    Build and populate a ``SessionStore`` with all 19 test scenarios.
    Imports lazily from ``tests/`` so the visualiser module is importable
    without the test tree when running in production.
    """
    from tests.datasets.scenario_catalogue import SCENARIOS  # noqa: PLC0415
    from ts_agent.observability.session_builder import SessionBuilder  # noqa: PLC0415

    store   = SessionStore()
    builder = SessionBuilder(store)
    builder.build_all(SCENARIOS)
    return store


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building demo store from 19 scenarios…")
    _store = build_demo_store()
    print(f"Loaded {_store.record_count()} sessions across "
          f"{len(_store.index.all_party_refs())} consumers.")
    _app = create_app(_store)
    print("Visualiser running at http://localhost:8050")
    _app.run(debug=False, port=8050)
