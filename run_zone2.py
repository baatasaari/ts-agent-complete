"""
Run Zone 2 with Gemini Conversational AI — ADK 1.22.1
======================================================

Full conversational AI demo using Gemini 2.0 Flash with:
- Natural, friendly conversation (not robotic Q&A)
- Automatic ML prediction after each answer
- Adaptive questioning — stops when confidence ≥ 75%
- Early exclusion detection (high-cost debt, vulnerability)
- Full EAMGP observability signals
- Zones 1 → 2 (→ optional 3 & 4)

Prerequisites
-------------
1. Authenticate with GCP:
       gcloud auth application-default login

2. Project is auto-detected from gcloud config (tvr-ai-projects).

3. Enable Vertex AI API (once per project):
       gcloud services enable aiplatform.googleapis.com

4. Run:
       python3 run_zone2.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Suppress log noise for clean conversational output ───────────────────────
import logging
import warnings
logging.disable(logging.WARNING)          # hide INFO/WARNING JSON logs
warnings.filterwarnings("ignore")         # hide ADK experimental feature warnings

# ── ADK imports (1.22.1) ──────────────────────────────────────────────────────
try:
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.events import Event
    from google.genai import types as genai_types
except ImportError:
    print("❌  google-adk not installed.")
    print("    pip3 install google-adk google-cloud-aiplatform")
    sys.exit(1)

# ── Project imports ───────────────────────────────────────────────────────────
from ts_agent.zones.agent.lead_agent import PipelineContext
from ts_agent.zones.zone2.agent_ml_auto import create_ml_auto_gap_fill_agent
from ts_agent.zones.zone2.tools import (
    STATE_SEGMENT_ID,
    STATE_COMPLETE,
    STATE_HYPOTHESIS,
    STATE_GRAPH,
    STATE_FILL_ORDER,
    STATE_TURN,
    _graph_to_dict,
    _graph_from_dict,
)
from ts_agent.zones.zone3.suggestion_engine import SuggestionEngine
from ts_agent.zones.zone3.delivery_agent import DeliveryCoordinator
from ts_agent.domain.models import (
    ExplainabilityBundle,
    NodeBranch,
    NodeState,
    TraitGraph,
    TraitNode,
)
from ts_agent.visualiser.data_adapter import CHAR_BRANCH_MAP, QUESTION_TEXT_MAP, CHAR_SHORT_LABEL
import ts_agent.observability.signals as _signals_mod

# Silence console JSON logs but keep data collection for visualiser
def _silent_emit_log(level: str, payload: dict) -> None:
    """Silent version that doesn't print to console"""
    pass

_signals_mod._emit_log = _silent_emit_log
eamgp = _signals_mod   # local alias still works

# ── Constants ─────────────────────────────────────────────────────────────────
_APP_NAME = "ts-agent"
_USER_ID  = "demo-user"

# ── Colour helpers ────────────────────────────────────────────────────────────
_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"; _B = "\033[94m"
_BOLD = "\033[1m"; _RST = "\033[0m"


def _hdr(t: str) -> None:
    print(f"\n{_B}{_BOLD}{'═'*70}{_RST}")
    print(f"{_B}{_BOLD}  {t}{_RST}")
    print(f"{_B}{_BOLD}{'═'*70}{_RST}\n")


def _sec(t: str) -> None:
    print(f"\n{_BOLD}── {t} {'─'*(60-len(t))}{_RST}")


def _ok(t: str)   -> None: print(f"  {_G}✓{_RST}  {t}")
def _warn(t: str) -> None: print(f"  {_Y}⚠{_RST}  {t}")
def _err(t: str)  -> None: print(f"  {_R}✗{_RST}  {t}")


# ── All 13 scenarios (matching run_demo.py) ───────────────────────────────────

# ── Traits that must be gap-filled for each situation ────────────────────────
# These are loaded as MISSING nodes so Gemini has questions to ask.

_GAP_FILL_TRAITS: dict[str, list[str]] = {
    "SIT-INV-001": ["CHAR-F2I-I1", "CHAR-F2G-I1", "CHAR-B3A-I1", "CHAR-B3B-I1", "CHAR-F2A-I1"],
    "SIT-INV-002": ["CHAR-F2I-I1", "CHAR-F2G-I1", "CHAR-B3A-I1", "CHAR-F2A-I1"],
    "SIT-INV-003": ["CHAR-B3A-I1", "CHAR-F2A-I1"],
    "SIT-INV-004": ["CHAR-F2M-I1", "CHAR-F2I-I1", "CHAR-B3A-I1", "CHAR-F2A-I1"],
    "SIT-SD-001":  ["CHAR-F2A-I1"],
    "SIT-PEN-001": ["CHAR-P2A-I1", "CHAR-F2A-I1"],
    "SIT-PEN-002": ["CHAR-F2A-I1"],
    "SIT-PEN-003": ["CHAR-P2A-I1", "CHAR-F2A-I1"],
    "SIT-DEC-001": ["CHAR-F2A-I1"],
    "SIT-DEC-002": ["CHAR-F2A-I1"],
    "SIT-DEC-003": ["CHAR-F2A-I1"],
    "SIT-DEC-004": ["CHAR-F2A-I1"],
}

SCENARIOS: dict[str, dict] = {
    "1": {
        "label":        "Investments — Cash-Heavy Non-Investor (SEG-INV-001)",
        "situation_id": "SIT-INV-001",
        "intent_id":    "INTENT-INVEST-CASH",
        "bank_traits": {
            "CHAR-P1A-I1": 2,
            "CHAR-P1B-I1": False,
            "CHAR-P1C-I1": 1,
            "CHAR-F2B-I1": 18000.0,
            "CHAR-F2L-I1": 24,
        },
    },
    "2": {
        "label":        "Investments — First-Time Investor (SEG-INV-002)",
        "situation_id": "SIT-INV-002",
        "intent_id":    "INTENT-FIRST-INVEST",
        "bank_traits": {
            "CHAR-P1A-I1": 2,
            "CHAR-P1B-I1": False,
            "CHAR-P1C-I1": 1,
            "CHAR-F2B-I1": 4000.0,
        },
    },
    "3": {
        "label":        "Investments — ISA Allowance Window (SEG-INV-003)",
        "situation_id": "SIT-INV-003",
        "intent_id":    "INTENT-ISA-ALLOWANCE",
        "bank_traits": {
            "CHAR-P1A-I1": 3,
            "CHAR-P1B-I1": False,
            "CHAR-F2B-I1": 5000.0,
            "CHAR-F2J-I1": False,
            "CHAR-F2K-I1": True,
        },
    },
    "4": {
        "label":        "Investments — Lump Sum Recipient (SEG-INV-004)",
        "situation_id": "SIT-INV-004",
        "intent_id":    "INTENT-LUMP-SUM",
        "bank_traits": {
            "CHAR-P1A-I1": 3,
            "CHAR-P1B-I1": False,
            "CHAR-F2H-I1": False,
        },
    },
    "5": {
        "label":        "Structured Deposits — Maturing Deposit (SEG-SD-001)",
        "situation_id": "SIT-SD-001",
        "intent_id":    "INTENT-DEPOSIT-MATURITY",
        "bank_traits": {
            "CHAR-P1A-I1": 4,
            "CHAR-P1B-I1": False,
            "CHAR-F2Q-I1": 45,
            "CHAR-F2R-I1": False,
            "CHAR-F2B-I1": 35000.0,
        },
    },
    "6": {
        "label":        "DC Pension Accum. — Under-saving (SEG-PEN-001)",
        "situation_id": "SIT-PEN-001",
        "intent_id":    "INTENT-PENSION-CONTRIBUTIONS",
        "bank_traits": {
            "CHAR-P1A-I1": 2,
            "CHAR-P1B-I1": False,
            "CHAR-P2B-I1": True,
            "CHAR-P2C-I1": True,
            "CHAR-P2D-I1": 28,
            "CHAR-F2H-I1": False,
        },
    },
    "7": {
        "label":        "DC Pension Accum. — Default Fund Disengaged (SEG-PEN-002)",
        "situation_id": "SIT-PEN-002",
        "intent_id":    "INTENT-PENSION-FUNDS",
        "bank_traits": {
            "CHAR-P1A-I1": 2,
            "CHAR-P1B-I1": False,
            "CHAR-P2E-I1": True,
            "CHAR-P2F-I1": False,
            "CHAR-P2B-I1": True,
            "CHAR-P2G-I1": False,
        },
    },
    "8": {
        "label":        "DC Pension Accum. — Life Event (SEG-PEN-003)",
        "situation_id": "SIT-PEN-003",
        "intent_id":    "INTENT-LIFE-EVENT",
        "bank_traits": {
            "CHAR-P1A-I1": 3,
            "CHAR-P1B-I1": False,
            "CHAR-P2H-I1": True,
            "CHAR-P2B-I1": True,
            "CHAR-F2H-I1": False,
            "CHAR-P2D-I1": 20,
        },
    },
    "9": {
        "label":        "DC Pension Decum. — Pre-retirement Non-Planner (SEG-DEC-001)",
        "situation_id": "SIT-DEC-001",
        "intent_id":    "INTENT-RETIREMENT-PLAN",
        "bank_traits": {
            "CHAR-P1A-I1": 5,
            "CHAR-P1B-I1": False,
            "CHAR-P2B-I1": True,
            "CHAR-P2I-I1": False,
            "CHAR-P2J-I1": False,
            "CHAR-P2K-I1": False,
        },
    },
    "10": {
        "label":        "DC Pension Decum. — Small Pot Holder (SEG-DEC-002)",
        "situation_id": "SIT-DEC-002",
        "intent_id":    "INTENT-TAKE-PENSION",
        "bank_traits": {
            "CHAR-P1A-I1": 5,
            "CHAR-P1B-I1": False,
            "CHAR-P2L-I1": 18000.0,
            "CHAR-P2D-I1": 3,
            "CHAR-P2I-I1": False,
        },
    },
    "11": {
        "label":        "DC Pension Decum. — Annuity Enquirer (SEG-DEC-003)",
        "situation_id": "SIT-DEC-003",
        "intent_id":    "INTENT-ANNUITY",
        "bank_traits": {
            "CHAR-P1A-I1": 5,
            "CHAR-P1B-I1": False,
            "CHAR-P2M-I1": True,
            "CHAR-P2B-I1": True,
            "CHAR-P2K-I1": False,
        },
    },
    "12": {
        "label":        "DC Pension Decum. — Drawdown Review (SEG-DEC-004)",
        "situation_id": "SIT-DEC-004",
        "intent_id":    "INTENT-DRAWDOWN-REVIEW",
        "bank_traits": {
            "CHAR-P1A-I1": 5,
            "CHAR-P1B-I1": False,
            "CHAR-P2J-I1": True,
            "CHAR-P2N-I1": 22,
            "CHAR-P2O-I1": True,
            "CHAR-P2P-I1": False,
        },
    },
    "13": {
        "label":        "[Custom] — Enter your own answers",
        "situation_id": "SIT-INV-001",
        "intent_id":    "INTENT-INVEST-CASH",
        "bank_traits": {
            "CHAR-P1A-I1": 2,
            "CHAR-P1B-I1": False,
            "CHAR-P1C-I1": 1,
        },
    },
    "14": {
        "label":        "[CSV Profile] — Load a consumer from ts_agent_consumer_profiles.csv",
        "situation_id": None,   # resolved at runtime
        "intent_id":    None,
        "bank_traits":  None,   # resolved at runtime
    },
}


# ── CSV column → char_id mapping ─────────────────────────────────────────────

_CSV_COL_TO_CHAR: dict[str, str] = {
    "CHAR_P1A_I1_age_band":          "CHAR-P1A-I1",
    "CHAR_P1B_I1_vulnerability":     "CHAR-P1B-I1",
    "CHAR_P1C_I1_employment":        "CHAR-P1C-I1",
    "CHAR_F2A_I1_monthly_surplus":   "CHAR-F2A-I1",
    "CHAR_F2B_I1_savings_balance":   "CHAR-F2B-I1",
    "CHAR_F2G_I1_high_cost_debt":    "CHAR-F2G-I1",
    "CHAR_F2H_I1_financial_hardship":"CHAR-F2H-I1",
    "CHAR_F2I_I1_has_investment":    "CHAR-F2I-I1",
    "CHAR_F2J_I1_isa_sub_this_year": "CHAR-F2J-I1",
    "CHAR_F2K_I1_within_90d_tax_yr": "CHAR-F2K-I1",
    "CHAR_F2L_I1_account_tenure_mo": "CHAR-F2L-I1",
    "CHAR_F2M_I1_lump_sum":          "CHAR-F2M-I1",
    "CHAR_F2Q_I1_days_to_maturity":  "CHAR-F2Q-I1",
    "CHAR_F2R_I1_reinvest_instr":    "CHAR-F2R-I1",
    "CHAR_P2A_I1_contrib_pct":       "CHAR-P2A-I1",
    "CHAR_P2B_I1_active_dc":         "CHAR-P2B-I1",
    "CHAR_P2C_I1_shortfall_flag":    "CHAR-P2C-I1",
    "CHAR_P2D_I1_yrs_to_ret":        "CHAR-P2D-I1",
    "CHAR_P2E_I1_default_fund":      "CHAR-P2E-I1",
    "CHAR_P2F_I1_no_selection":      "CHAR-P2F-I1",
    "CHAR_P2G_I1_recent_engage":     "CHAR-P2G-I1",
    "CHAR_P2H_I1_life_event":        "CHAR-P2H-I1",
    "CHAR_P2I_I1_access_plan":       "CHAR-P2I-I1",
    "CHAR_P2J_I1_in_drawdown":       "CHAR-P2J-I1",
    "CHAR_P2K_I1_db_transfer":       "CHAR-P2K-I1",
    "CHAR_P2L_I1_pot_value":         "CHAR-P2L-I1",
    "CHAR_P2M_I1_annuity_interest":  "CHAR-P2M-I1",
    "CHAR_P2N_I1_months_review":     "CHAR-P2N-I1",
    "CHAR_P2O_I1_cash_drag":         "CHAR-P2O-I1",
    "CHAR_P2P_I1_recent_adviser":    "CHAR-P2P-I1",
    "CHAR_B3A_I1_risk_appetite":     "CHAR-B3A-I1",
    "CHAR_B3B_I1_invest_exp":        "CHAR-B3B-I1",
    "CHAR_B3C_I1_channel":           "CHAR-B3C-I1",
}

# Traits that come from bank systems (bank-known, loaded as KNOWN)
_BANK_KNOWN_CHARS = {
    "CHAR-P1A-I1", "CHAR-P1B-I1", "CHAR-P1C-I1",
    "CHAR-F2B-I1", "CHAR-F2H-I1", "CHAR-F2J-I1", "CHAR-F2K-I1",
    "CHAR-F2L-I1", "CHAR-F2Q-I1", "CHAR-F2R-I1",
    "CHAR-P2B-I1", "CHAR-P2C-I1", "CHAR-P2D-I1", "CHAR-P2E-I1",
    "CHAR-P2F-I1", "CHAR-P2G-I1", "CHAR-P2H-I1", "CHAR-P2I-I1",
    "CHAR-P2J-I1", "CHAR-P2K-I1", "CHAR-P2L-I1", "CHAR-P2M-I1",
    "CHAR-P2N-I1", "CHAR-P2O-I1", "CHAR-P2P-I1",
}


def _load_csv_scenario(n: int = 1) -> dict | None:
    """Load consumer profile #n from ts_agent_consumer_profiles.csv."""
    import csv
    csv_path = Path(__file__).parent / "ts_agent_consumer_profiles.csv"
    if not csv_path.exists():
        _err(f"CSV not found: {csv_path}")
        return None

    profiles = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            profiles.append(row)

    if not profiles:
        _err("CSV is empty.")
        return None

    # Show a selection of profiles
    print()
    _sec(f"Consumer Profiles from CSV ({len(profiles)} total)")
    for i, p in enumerate(profiles[:10], 1):
        gate = p.get("gate_disposition", "?")
        seg  = p.get("segment_id", "?")
        sit  = p.get("situation_id", "?")
        age  = p.get("CHAR_P1A_I1_age_band", "?")
        print(f"  {i:2}. {sit}/{seg}  age_band={age}  gate={gate}")
    if len(profiles) > 10:
        print(f"      ... ({len(profiles) - 10} more)")
    print()

    try:
        idx = int(input(f"  {_BOLD}Enter profile number (1–{min(10, len(profiles))}): {_RST}").strip()) - 1
        if not 0 <= idx < len(profiles):
            raise ValueError
    except (ValueError, EOFError):
        _err("Invalid choice.")
        return None

    row = profiles[idx]
    situation_id = row.get("situation_id", "SIT-INV-001")

    # Split traits: bank-known vs gap-fill
    bank_traits: dict = {}
    gap_fill_chars = set(_GAP_FILL_TRAITS.get(situation_id, ["CHAR-F2A-I1"]))

    for col, char_id in _CSV_COL_TO_CHAR.items():
        raw = row.get(col, "")
        if raw == "" or raw is None:
            continue
        # Coerce type
        if raw.lower() in ("true", "false"):
            val: object = raw.lower() == "true"
        else:
            try:
                val = float(raw)
                if val == int(val):
                    val = int(val)
            except ValueError:
                val = raw

        if char_id in _BANK_KNOWN_CHARS and char_id not in gap_fill_chars:
            bank_traits[char_id] = val

    party_ref = row.get("party_ref", "DEMO-CSV-CONSUMER")
    seg_id    = row.get("segment_id", "")
    gate      = row.get("gate_disposition", "")

    _ok(f"Loaded:  {party_ref}  →  {situation_id}/{seg_id}  [expected gate: {gate}]")
    return {
        "label":        f"CSV: {party_ref} — {seg_id}",
        "situation_id": situation_id,
        "intent_id":    f"INTENT-CSV-{situation_id}",
        "bank_traits":  bank_traits,
    }


# ── Graph builder (characteristic IDs → TraitGraph) ──────────────────────────

_BRANCH_MAP: dict[str, NodeBranch] = {
    "Personal":    NodeBranch.PERSONAL,
    "Financial":   NodeBranch.FINANCIAL,
    "Pension":     NodeBranch.PENSION,
    "Temporal":    NodeBranch.TEMPORAL,
    "Behavioural": NodeBranch.BEHAVIOURAL,
    "Product":     NodeBranch.PRODUCT,
}


def _build_graph_from_traits(bank_traits: dict, situation_id: str) -> TraitGraph:
    """
    Construct a TraitGraph from bank-known char_id → value pairs.

    Also adds MISSING nodes for the gap-fill traits of this situation so
    Gemini has something to ask the consumer about.
    """
    g = TraitGraph(
        session_id=str(uuid.uuid4()),
        party_ref="DEMO-CONSUMER-001",
        intent_id="DEMO",
        situation_id=situation_id,
    )

    # ── KNOWN nodes — from bank data ──────────────────────────────────────────
    for i, (char_id, value) in enumerate(bank_traits.items()):
        branch_str = CHAR_BRANCH_MAP.get(char_id, "Financial")
        branch = _BRANCH_MAP.get(branch_str, NodeBranch.FINANCIAL)
        node = TraitNode(
            node_id=f"node-{i:03d}",
            char_id=char_id,
            branch=branch,
            label=QUESTION_TEXT_MAP.get(char_id, char_id)[:60],
            op="==",
            target_value=value,
            data_sources=("BANK_DATA",),
            aging="30d",
            fill_priority=i + 1,
            state=NodeState.KNOWN,
            value=value,
            populated_source="BANK_DATA",
        )
        g.add_node(node)

    # ── MISSING nodes — gap-fill traits Gemini will ask about ─────────────────
    gap_fill = _GAP_FILL_TRAITS.get(situation_id, ["CHAR-F2A-I1"])
    known_ids = set(bank_traits.keys())
    for j, char_id in enumerate(gap_fill):
        if char_id in known_ids:
            continue   # already known from bank data
        branch_str = CHAR_BRANCH_MAP.get(char_id, "Financial")
        branch = _BRANCH_MAP.get(branch_str, NodeBranch.FINANCIAL)
        node = TraitNode(
            node_id=f"gap-{j:03d}",
            char_id=char_id,
            branch=branch,
            label=QUESTION_TEXT_MAP.get(char_id, char_id)[:60],
            op="==",
            target_value=None,
            data_sources=("CONSUMER_INPUT",),
            aging="session",
            fill_priority=len(bank_traits) + j + 1,
            state=NodeState.MISSING,
            value=None,
            populated_source=None,
        )
        g.add_node(node)

    return g


# ── Display helpers ───────────────────────────────────────────────────────────

def _extract_text_from_events(events: list[Event]) -> str:
    """Return the final agent text from a completed run_async event stream."""
    for ev in reversed(events):
        if ev.is_final_response() and ev.content and ev.content.parts:
            texts = [p.text for p in ev.content.parts if p.text]
            if texts:
                return " ".join(texts)
    return ""


def _display_graph(graph: TraitGraph) -> None:
    known   = list(graph.known_nodes())
    missing = list(graph.missing_nodes())
    total   = len(graph.nodes)
    pct     = len(known) / total if total else 0
    bar     = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
    print(f"   Completeness: [{bar}] {pct:.0%}  ({len(known)}/{total} traits known)")
    if missing:
        ids = ", ".join(n.char_id for n in missing[:5])
        suffix = f" + {len(missing)-5} more" if len(missing) > 5 else ""
        print(f"   Missing: {ids}{suffix}")
    print()


def _display_ml(hypothesis_json: str, turn: int) -> None:
    try:
        hyp  = json.loads(hypothesis_json)
        seg  = hyp.get("top_segment_id", "N/A")
        conf = hyp.get("top_confidence", 0)
        bars = "█" * int(conf * 20)
        print(f"   🤖 ML (turn {turn}): {seg}  {conf:.1%} [{bars:<20}]")
        if "all_scores" in hyp:
            top3 = sorted(hyp["all_scores"].items(), key=lambda x: x[1], reverse=True)[:3]
            print("      Top-3: " + " | ".join(f"{s}: {v:.1%}" for s, v in top3))
    except Exception:
        pass


# ── Main demo ─────────────────────────────────────────────────────────────────

class GeminiConversationDemo:

    def __init__(self, run_full_pipeline: bool = False) -> None:
        self.run_full_pipeline = run_full_pipeline
        self.turn_times: list[float] = []

    # ── Setup check ───────────────────────────────────────────────────────────

    def _check_setup(self) -> bool:
        # ── Option A: Google AI Studio API Key ────────────────────────────────
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if api_key:
            # Remove Vertex AI flag so ADK uses Google AI Studio directly
            os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
            _ok(f"Auth mode:    Google AI Studio (API key)")
            _ok(f"Model:        gemini-2.0-flash")
            return True

        # ── Option B: Vertex AI (GCP project + ADC) ───────────────────────────
        _warn("GOOGLE_API_KEY not set — falling back to Vertex AI.")
        print()
        print(f"  {_BOLD}FASTEST: Get a free AI Studio API key:{_RST}")
        print(f"  1. Visit {_Y}https://aistudio.google.com/app/apikey{_RST}")
        print(f"  2. Create API key for project tvr-ai-projects")
        print(f"  3. Run:  {_Y}export GOOGLE_API_KEY=<your-key>{_RST}")
        print(f"  4. Re-run:  python3 run_zone2.py")
        print()

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not project:
            try:
                import subprocess
                r = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    capture_output=True, text=True, timeout=5,
                )
                project = r.stdout.strip()
                if project:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = project
            except Exception:
                pass

        if not project:
            _err("No API key and no GCP project. Set GOOGLE_API_KEY to continue.")
            return False

        _ok(f"GCP Project:  {project}")

        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west2")
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
        _ok(f"Location:     {location}")
        _ok("Vertex AI backend: enabled")

        adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if not adc.exists():
            _err("Application Default Credentials not found.")
            print(f"  Run:  {_Y}gcloud auth application-default login{_RST}")
            return False

        _ok("Application Default Credentials: found")
        return True

    # ── Zone 1: build graph ───────────────────────────────────────────────────

    def _build_zone1(self, scenario: dict) -> tuple[PipelineContext, dict, object]:
        _sec("Zone 1 — Bank-Known Traits")
        t0 = time.time()

        for char_id, value in scenario["bank_traits"].items():
            lbl = CHAR_SHORT_LABEL.get(char_id, char_id)
            print(f"   {lbl:<30} {value}")
        print()

        graph = _build_graph_from_traits(scenario["bank_traits"], scenario["situation_id"])
        _display_graph(graph)
        _ok(f"Zone 1 complete ({time.time() - t0:.2f}s)")

        session_id = f"gemini-demo-{int(time.time())}"
        ctx = PipelineContext(
            session_id=session_id,
            party_ref="DEMO-CONSUMER-001",
            graph=graph,
            bundle=ExplainabilityBundle(session_id=session_id),
        )

        _sec("Zone 1.5 — Initial State (static fill order)")
        # Build ADK session state directly — IterativeSegmentPredictor requires a
        # fitted sklearn model which is not available in the demo. Gemini's
        # ML auto-tools run prediction automatically after each answer.
        fill_order = [
            n.char_id
            for n in sorted(graph.missing_nodes(), key=lambda n: n.fill_priority)
        ]
        state = {
            STATE_GRAPH:      _graph_to_dict(graph),
            STATE_FILL_ORDER: fill_order,
            STATE_TURN:       0,
            STATE_COMPLETE:   False,
            STATE_SEGMENT_ID: None,
        }
        _ok(f"Graph nodes:  {len(graph.nodes)} ({len(graph.known_nodes())} known, "
            f"{len(list(graph.missing_nodes()))} missing)")
        _ok(f"Fill order:   {', '.join(fill_order[:5])}"
            + (f" + {len(fill_order)-5} more" if len(fill_order) > 5 else ""))
        print()
        return ctx, state, None

    # ── Zone 2: Gemini conversation ───────────────────────────────────────────

    async def _run_zone2(self, ctx: PipelineContext, state: dict, fallback_segment_id: str | None = None) -> dict | None:
        _hdr("GEMINI CONVERSATIONAL AI — Zone 2 Gap-Fill")
        print("  Gemini has a natural conversation to understand your situation.")
        print("  ML runs after every answer and stops when confidence ≥ 75%.")
        print()
        input(f"  {_BOLD}Press Enter to start...{_RST}")
        print()

        agent           = create_ml_auto_gap_fill_agent()
        session_service = InMemorySessionService()

        await session_service.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            state=state,
            session_id=ctx.session_id,
        )

        runner = Runner(
            agent=agent,
            session_service=session_service,
            app_name=_APP_NAME,
        )

        eamgp.emit("ZONE2_START", eamgp.INFO, "Zone2", session_id=ctx.session_id)

        print(f"  {_Y}Note: ADK Web UI not available in version 1.22.1{_RST}")
        print(f"  {_G}Using enhanced console interface with full ML capabilities{_RST}")
        print()
        
        return await self._run_console_mode(runner, ctx, fallback_segment_id)
    
    async def _run_console_mode(self, runner: Runner, ctx: PipelineContext, fallback_segment_id: str | None = None) -> dict | None:
        """Fallback console mode if web interface fails"""
        turn = 0
        while turn <= 15:
            t0 = time.time()
            turn += 1

            user_text = input(f"  {_BOLD}You:{_RST} ").strip()
            if not user_text:
                if turn == 1:
                    user_text = "Hi, I need some financial support"
                    print(f"     {_Y}→ Using default: {user_text}{_RST}")
                else:
                    _warn("No input — ending conversation.")
                    break
            print()

            new_msg = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_text)],
            )

            collected: list[Event] = []
            try:
                async for event in runner.run_async(
                    user_id=_USER_ID,
                    session_id=ctx.session_id,
                    new_message=new_msg,
                ):
                    collected.append(event)
            except Exception as api_err:
                elapsed = time.time() - t0
                self.turn_times.append(elapsed)
                err_str = str(api_err)
                if "503" in err_str or "UNAVAILABLE" in err_str:
                    _warn("Gemini is busy right now — please wait a moment and try again.")
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    _warn("API rate limit hit — please wait a moment and try again.")
                else:
                    _err(f"API error: {err_str[:120]}")
                continue

            elapsed = time.time() - t0
            self.turn_times.append(elapsed)

            agent_text = _extract_text_from_events(collected)
            if agent_text:
                print(f"  {_BOLD}{_G}Gemini:{_RST} {agent_text}\n")
            else:
                print(f"  {_BOLD}{_G}Gemini:{_RST} [processing...]\n")

            session = await runner.session_service.get_session(
                app_name=_APP_NAME,
                user_id=_USER_ID,
                session_id=ctx.session_id,
            )
            current = session.state if session else {}

            if STATE_HYPOTHESIS in current:
                _display_ml(current[STATE_HYPOTHESIS], turn)
                print()

            eamgp.emit("GAP_FILL_TURN", eamgp.INFO, "Zone2",
                       turn=turn, duration_ms=int(elapsed * 1000))

            if current.get(STATE_COMPLETE):
                seg = current.get(STATE_SEGMENT_ID)
                if not seg and fallback_segment_id:
                    seg = fallback_segment_id
                    current[STATE_SEGMENT_ID] = seg
                    _warn(f"No exact match — using scenario segment: {_BOLD}{seg}{_RST}")
                else:
                    _ok(f"Matched segment:    {_BOLD}{seg}{_RST}")
                _ok(f"Conversation turns: {turn}")
                return current

        _warn("Conversation reached turn limit without completing.")
        return None

    # ── Zones 3 & 4 ──────────────────────────────────────────────────────────

    def _run_zone3_4(self, ctx: PipelineContext, final_state: dict) -> None:
        seg_id = final_state.get(STATE_SEGMENT_ID)
        if not seg_id:
            _warn("No segment matched — skipping Zone 3.")
            return

        _sec("Zone 3 — Compliance Evaluation")
        graph_dict = final_state.get(STATE_GRAPH)
        graph = _graph_from_dict(graph_dict) if graph_dict else ctx.graph

        engine = SuggestionEngine()
        result = engine.evaluate(
            segment_id=seg_id,
            graph=graph,
            hypothesis=ctx.hypothesis,
            bundle=ctx.bundle,
        )

        gate = result.gate_disposition.value
        colour = {"EMIT": _G, "HUMAN_REVIEW": _Y, "SUPPRESS": _R}.get(gate, _RST)
        print(f"  Gate disposition: {colour}{_BOLD}{gate}{_RST}")

        for ev in result.all_evaluations:
            for r in ev.rule_results:
                fn = _ok if r.outcome == "PASS" else (_warn if r.outcome == "GATE" else _err)
                fn(f"{r.rule_def.rule_id:<8} {r.rule_def.description[:55]}")

        if self.run_full_pipeline and gate == "EMIT":
            _sec("Zone 4 — Consumer Message Delivery")
            delivery = DeliveryCoordinator().deliver(result, ctx.bundle)
            if delivery.consumer_message:
                for line in delivery.consumer_message.splitlines():
                    print(f"  {line}")
                print(f"\n  Communication hash: {delivery.communication_hash}")

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        t_start = time.time()

        _hdr("LBG Targeted Support — Gemini Conversational AI Demo")
        print(f"  ADK version:  google-adk 1.22.1")
        print(f"  Model:        gemini-1.5-flash  (via Vertex AI)")
        zones = "Zones 1 → 2 → 3 → 4" if self.run_full_pipeline else "Zones 1 → 2"
        print(f"  Pipeline:     {zones}")
        print()

        if not self._check_setup():
            return

        _sec("Select a scenario")
        for k, v in SCENARIOS.items():
            print(f"  {_BOLD}{k:>2}.{_RST} {v['label']}")
        print()

        choice = input(f"  {_BOLD}Enter 1–{len(SCENARIOS)}:{_RST} ").strip()
        if choice not in SCENARIOS:
            _err("Invalid choice. Exiting.")
            return

        if choice == "14":
            sc = _load_csv_scenario()
            if sc is None:
                return
        else:
            sc = SCENARIOS[choice]
        _hdr(sc["label"])
        print(f"  Situation: {sc['situation_id']}  |  Intent: {sc['intent_id']}")

        eamgp.emit("DEMO_START", eamgp.INFO, "Demo",
                   scenario=sc["situation_id"], mode="gemini_conversational")

        # Derive expected segment as fallback: SIT-INV-001 → SEG-INV-001
        fallback_seg = sc.get("segment_id") or (
            sc["situation_id"].replace("SIT-", "SEG-")
            if sc.get("situation_id") else None
        )
        ctx, state, _orch = self._build_zone1(sc)
        final_state = await self._run_zone2(ctx, state, fallback_segment_id=fallback_seg)

        if final_state and self.run_full_pipeline:
            self._run_zone3_4(ctx, final_state)

        total    = time.time() - t_start
        avg_turn = sum(self.turn_times) / len(self.turn_times) if self.turn_times else 0

        _sec("Session Summary")
        _ok(f"Total time:   {total:.1f}s")
        _ok(f"Turns:        {len(self.turn_times)}")
        _ok(f"Avg turn:     {avg_turn:.1f}s  (Gemini API + tools)")
        _ok(f"Session ID:   {ctx.session_id}")

        eamgp.emit("DEMO_COMPLETE", eamgp.INFO, "Demo",
                   duration_s=total, turns=len(self.turn_times))

        print(f"\n  {_BOLD}Run again:{_RST}  python3 run_zone2.py\n")


# ── CLI entry ─────────────────────────────────────────────────────────────────

async def main() -> None:
    _hdr("LBG TS Agent — Gemini Conversational AI")
    print("  1. Zone 2 only  (Gemini conversation + ML)")
    print("  2. Full pipeline (Zones 1 → 4 including compliance)")
    print()
    choice = input(f"  {_BOLD}Enter 1–2:{_RST} ").strip()
    demo = GeminiConversationDemo(run_full_pipeline=(choice == "2"))
    await demo.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n  {_Y}Interrupted.{_RST}\n")
    except Exception as exc:
        print(f"\n\n  {_R}Error: {exc}{_RST}\n")
        import traceback
        traceback.print_exc()
