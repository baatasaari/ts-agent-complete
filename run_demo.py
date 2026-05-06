"""
run_demo.py
============
Interactive CLI demo for the LBG Targeted Support Agent Platform v2.

Runs the full pipeline — Zone 1 through Zone 4 — without a live LLM or
Neo4j connection.  Zone 2 gap-fill is simulated by asking you questions
in the terminal.

Domains covered (PS25/22 live 6 April 2026):
  Retail Investments  (INV-001 … INV-006)
  Structured Deposits (SD-001)
  DC Pension Accumulation (PEN-001 … PEN-003)
  DC Pension Decumulation (DEC-001 … DEC-004)

Usage
-----
    python run_demo.py

No Google Cloud credentials, no Neo4j, no API key required.
"""
from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import uuid
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

from ts_agent.config.segments import SEGMENT_TO_SUGGESTIONS, SEGMENTS, SITUATIONS
from ts_agent.config.settings import settings
from ts_agent.domain.models import (
    ExplainabilityBundle,
    GapFillStrategy,
    HypothesisDisposition,
    ModelAlgorithm,
    NodeBranch,
    NodeState,
    SegmentHypothesis,
    SegmentRank,
    TraitGraph,
    TraitNode,
)
from ts_agent.explainability.explainer import ConsumerExplainer, SuggestionContext
from ts_agent.zones.zone2.tools_ml_auto import (
    STATE_COMPLETE,
    STATE_GRAPH,
    STATE_SEGMENT_ID,
    STATE_TURN,
    check_graph_completeness_ml_aware as check_graph_completeness,
    match_segment,
    record_consumer_answer_ml_auto as record_consumer_answer,
)
from ts_agent.zones.zone2.tools import (
    STATE_FILL_ORDER,
    _graph_to_dict,
)
from ts_agent.zones.zone3.delivery_agent import DeliveryCoordinator
from ts_agent.zones.zone3.suggestion_engine import SuggestionEngine
from ts_agent.observability.session_store import SessionStore
from ts_agent.observability.session_builder import SessionBuilder

# ── Colour helpers ────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BLUE   = "\033[94m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _banner(text: str, colour: str = _BLUE) -> None:
    width = 70
    print(f"\n{colour}{_BOLD}{'═' * width}{_RESET}")
    print(f"{colour}{_BOLD}  {text}{_RESET}")
    print(f"{colour}{_BOLD}{'═' * width}{_RESET}\n")


def _section(text: str) -> None:
    print(f"\n{_BOLD}── {text} {'─' * (60 - len(text))}{_RESET}")


def _ok(text: str)   -> None: print(f"  {_GREEN}✓{_RESET}  {text}")
def _warn(text: str) -> None: print(f"  {_YELLOW}⚠{_RESET}  {text}")
def _err(text: str)  -> None: print(f"  {_RED}✗{_RESET}  {text}")


# ── Demo scenario definitions (v2 ontology) ───────────────────────────────────

DEMO_SCENARIOS: list[dict[str, Any]] = [
    {
        "label":       "Investments — Cash-Heavy Non-Investor (SEG-INV-001)",
        "situation_id": "SIT-INV-001",
        "intent_id":   "INTENT-INVEST-CASH",
        "segment_id":  "SEG-INV-001",
        "bank_traits": {
            "CHAR-P1A-I1": 2,        # age band 30–39
            "CHAR-P1B-I1": False,    # not vulnerable
            "CHAR-P1C-I1": 1,        # employed
            "CHAR-F2B-I1": 18000.0,  # £18,000 in cash savings
            "CHAR-F2L-I1": 24,       # 24 months tenure
        },
        "gap_fill_questions": [
            ("CHAR-F2I-I1", "Do you currently hold any investment products (S&S ISA, funds, GIA)?", bool),
            ("CHAR-F2G-I1", "Do you have any active high-cost debt (overdraft > £1k, payday loan)?", bool),
            ("CHAR-B3A-I1", "On a scale of 1–5, how comfortable are you with investment risk?", float),
            ("CHAR-B3B-I1", "Investment experience: 0=none, 1=basic, 2=some, 3=experienced?", int),
            ("CHAR-F2A-I1", "Roughly how much do you have left each month after outgoings (£)?", float),
        ],
        "suggest_answers": ["no", "no", "3", "1", "600"],
    },
    {
        "label":       "Investments — First-Time Investor (SEG-INV-002)",
        "situation_id": "SIT-INV-002",
        "intent_id":   "INTENT-FIRST-INVEST",
        "segment_id":  "SEG-INV-002",
        "bank_traits": {
            "CHAR-P1A-I1": 2,        # age band 30–39
            "CHAR-P1B-I1": False,
            "CHAR-P1C-I1": 1,
            "CHAR-F2B-I1": 4000.0,   # £4,000 savings
        },
        "gap_fill_questions": [
            ("CHAR-F2I-I1", "Have you ever held an investment product?", bool),
            ("CHAR-F2G-I1", "Active high-cost debt?", bool),
            ("CHAR-B3A-I1", "Risk appetite 1–5?", float),
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["no", "no", "2", "350"],
    },
    {
        "label":       "Investments — ISA Allowance Window (SEG-INV-003)",
        "situation_id": "SIT-INV-003",
        "intent_id":   "INTENT-ISA-ALLOWANCE",
        "segment_id":  "SEG-INV-003",
        "bank_traits": {
            "CHAR-P1A-I1": 3,
            "CHAR-P1B-I1": False,
            "CHAR-F2B-I1": 5000.0,
            "CHAR-F2J-I1": False,    # no ISA sub this year
            "CHAR-F2K-I1": True,     # within 90d of 5 April
        },
        "gap_fill_questions": [
            ("CHAR-B3A-I1", "Risk appetite 1–5?", float),
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["3", "450"],
    },
    {
        "label":       "Investments — Lump Sum Recipient (SEG-INV-004)",
        "situation_id": "SIT-INV-004",
        "intent_id":   "INTENT-LUMP-SUM",
        "segment_id":  "SEG-INV-004",
        "bank_traits": {
            "CHAR-P1A-I1": 3,
            "CHAR-P1B-I1": False,
            "CHAR-F2H-I1": False,
        },
        "gap_fill_questions": [
            ("CHAR-F2M-I1", "How large is the lump sum (£)?  [Enter > 75000 to see exit-TS]", float),
            ("CHAR-F2I-I1", "Already have an investment instruction placed?", bool),
            ("CHAR-B3A-I1", "Risk appetite 1–5?", float),
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["20000", "no", "3", "700"],
    },
    {
        "label":       "Structured Deposits — Maturing Deposit (SEG-SD-001)",
        "situation_id": "SIT-SD-001",
        "intent_id":   "INTENT-DEPOSIT-MATURITY",
        "segment_id":  "SEG-SD-001",
        "bank_traits": {
            "CHAR-P1A-I1": 4,
            "CHAR-P1B-I1": False,
            "CHAR-F2Q-I1": 45,       # 45 days to maturity
            "CHAR-F2R-I1": False,    # no reinvestment instruction
            "CHAR-F2B-I1": 35000.0,
        },
        "gap_fill_questions": [
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["800"],
    },
    {
        "label":       "DC Pension Accum. — Under-saving (SEG-PEN-001)",
        "situation_id": "SIT-PEN-001",
        "intent_id":   "INTENT-PENSION-CONTRIBUTIONS",
        "segment_id":  "SEG-PEN-001",
        "bank_traits": {
            "CHAR-P1A-I1": 2,
            "CHAR-P1B-I1": False,
            "CHAR-P2B-I1": True,     # active DC member
            "CHAR-P2C-I1": True,     # projected shortfall
            "CHAR-P2D-I1": 28,       # 28 years to retirement
            "CHAR-F2H-I1": False,
        },
        "gap_fill_questions": [
            ("CHAR-P2A-I1", "Current combined pension contribution rate (% e.g. 8)?", float),
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["7", "500"],
    },
    {
        "label":       "DC Pension Accum. — Default Fund Disengaged (SEG-PEN-002)",
        "situation_id": "SIT-PEN-002",
        "intent_id":   "INTENT-PENSION-FUNDS",
        "segment_id":  "SEG-PEN-002",
        "bank_traits": {
            "CHAR-P1A-I1": 2,
            "CHAR-P1B-I1": False,
            "CHAR-P2E-I1": True,     # 100% default fund
            "CHAR-P2F-I1": False,    # no active selection ever
            "CHAR-P2B-I1": True,
            "CHAR-P2G-I1": False,    # no recent engagement
        },
        "gap_fill_questions": [
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["400"],
    },
    {
        "label":       "DC Pension Accum. — Life Event (SEG-PEN-003)",
        "situation_id": "SIT-PEN-003",
        "intent_id":   "INTENT-LIFE-EVENT",
        "segment_id":  "SEG-PEN-003",
        "bank_traits": {
            "CHAR-P1A-I1": 3,
            "CHAR-P1B-I1": False,
            "CHAR-P2H-I1": True,     # life event signal
            "CHAR-P2B-I1": True,
            "CHAR-F2H-I1": False,
            "CHAR-P2D-I1": 20,
        },
        "gap_fill_questions": [
            ("CHAR-P2A-I1", "Current contribution rate (%)?", float),
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["9", "700"],
    },
    {
        "label":       "DC Pension Decum. — Pre-retirement Non-Planner (SEG-DEC-001)",
        "situation_id": "SIT-DEC-001",
        "intent_id":   "INTENT-RETIREMENT-PLAN",
        "segment_id":  "SEG-DEC-001",
        "bank_traits": {
            "CHAR-P1A-I1": 5,
            "CHAR-P1B-I1": False,
            "CHAR-P2B-I1": True,
            "CHAR-P2I-I1": False,    # no access plan
            "CHAR-P2J-I1": False,    # not in drawdown
            "CHAR-P2K-I1": False,    # no DB transfer
        },
        "gap_fill_questions": [
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["600"],
    },
    {
        "label":       "DC Pension Decum. — Small Pot Holder (SEG-DEC-002)",
        "situation_id": "SIT-DEC-002",
        "intent_id":   "INTENT-TAKE-PENSION",
        "segment_id":  "SEG-DEC-002",
        "bank_traits": {
            "CHAR-P1A-I1": 5,
            "CHAR-P1B-I1": False,
            "CHAR-P2L-I1": 18000.0,  # £18k pot
            "CHAR-P2D-I1": 3,         # 3 years to retirement
            "CHAR-P2I-I1": False,
        },
        "gap_fill_questions": [
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["300"],
    },
    {
        "label":       "DC Pension Decum. — Annuity Enquirer (SEG-DEC-003)",
        "situation_id": "SIT-DEC-003",
        "intent_id":   "INTENT-ANNUITY",
        "segment_id":  "SEG-DEC-003",
        "bank_traits": {
            "CHAR-P1A-I1": 5,
            "CHAR-P1B-I1": False,
            "CHAR-P2M-I1": True,     # expressed annuity interest
            "CHAR-P2B-I1": True,
            "CHAR-P2K-I1": False,
        },
        "gap_fill_questions": [
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["400"],
    },
    {
        "label":       "DC Pension Decum. — Drawdown Review (SEG-DEC-004)",
        "situation_id": "SIT-DEC-004",
        "intent_id":   "INTENT-DRAWDOWN-REVIEW",
        "segment_id":  "SEG-DEC-004",
        "bank_traits": {
            "CHAR-P1A-I1": 5,
            "CHAR-P1B-I1": False,
            "CHAR-P2J-I1": True,     # in active drawdown
            "CHAR-P2N-I1": 22,       # 22 months since review
            "CHAR-P2O-I1": True,     # cash drag flag
            "CHAR-P2P-I1": False,    # no recent adviser
        },
        "gap_fill_questions": [
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
        ],
        "suggest_answers": ["600"],
    },
    {
        "label":       "[Custom] — Enter your own answers",
        "situation_id": "SIT-INV-001",
        "intent_id":   "INTENT-INVEST-CASH",
        "segment_id":  "SEG-INV-001",
        "bank_traits": {
            "CHAR-P1A-I1": 2,
            "CHAR-P1B-I1": False,
            "CHAR-P1C-I1": 1,
        },
        "gap_fill_questions": [
            ("CHAR-F2B-I1", "Cash savings balance (£)?", float),
            ("CHAR-F2I-I1", "Hold any investment product? (yes/no)", bool),
            ("CHAR-F2L-I1", "Account tenure in months?", int),
            ("CHAR-F2G-I1", "Active high-cost debt? (yes/no)", bool),
            ("CHAR-B3A-I1", "Risk appetite 1–5?", float),
            ("CHAR-B3B-I1", "Investment experience 0–3?", int),
            ("CHAR-F2A-I1", "Monthly surplus (£)?", float),
            ("CHAR-P1B-I1", "Vulnerability indicator? (yes/no — triggers HUMAN_REVIEW)", bool),
        ],
        "suggest_answers": None,  # prompt user for each
    },
]


# ── FakeToolContext ───────────────────────────────────────────────────────────

class _FakeToolContext:
    def __init__(self, state: dict) -> None:
        self.state = state


# ── Coercion ──────────────────────────────────────────────────────────────────

def _coerce(raw: str, py_type: type) -> Any:
    """Convert a terminal string to the required Python type."""
    s = raw.strip().lower()
    if py_type == bool:
        return s in ("yes", "y", "true", "1")
    if py_type == float:
        return float(s)
    if py_type == int:
        return int(float(s))
    return s


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph(bank_traits: dict[str, Any], situation_id: str) -> TraitGraph:
    """Populate a TraitGraph from bank-known traits."""
    from ts_agent.visualiser.data_adapter import CHAR_BRANCH_MAP, QUESTION_TEXT_MAP

    _BRANCH_MAP = {
        "Personal": NodeBranch.PERSONAL,
        "Financial": NodeBranch.FINANCIAL,
        "Pension": NodeBranch.PENSION,
        "Temporal": NodeBranch.TEMPORAL,
        "Behavioural": NodeBranch.BEHAVIOURAL,
        "Product": NodeBranch.PRODUCT,
    }

    g = TraitGraph(
        session_id=str(uuid.uuid4()),
        party_ref="DEMO-CONSUMER-001",
        intent_id="DEMO",
        situation_id=situation_id,
    )
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
    return g


# ── Zone 2 simulation ─────────────────────────────────────────────────────────

async def _run_gap_fill(
    g: TraitGraph,
    questions: list[tuple[str, str, type]],
    preset_answers: list[str] | None,
) -> str | None:
    """
    Simulate Zone 2 gap-fill.  Returns matched segment_id or None.
    """
    from ts_agent.visualiser.data_adapter import QUESTION_TEXT_MAP

    ctx = _FakeToolContext({
        STATE_GRAPH:      _graph_to_dict(g),
        STATE_FILL_ORDER: [],
        STATE_TURN:       0,
        STATE_COMPLETE:   False,
        STATE_SEGMENT_ID: None,
    })

    _section("Zone 2 — Gap-Fill Conversation")
    for idx, (char_id, question, py_type) in enumerate(questions):
        if preset_answers is not None:
            suggested = preset_answers[idx]
            print(f"  Q: {question}")
            raw = input(f"  A: [{_YELLOW}{suggested}{_RESET}] ").strip()
            if not raw:
                raw = suggested
                print(f"     → Using suggested: {_YELLOW}{raw}{_RESET}")
        else:
            raw = input(f"  {_BOLD}Q:{_RESET} {question}\n  {_BOLD}A:{_RESET} ").strip()
            if not raw:
                raw = "no" if py_type == bool else "0"

        value = _coerce(raw, py_type)
        await record_consumer_answer(char_id, str(value), ctx)

    await check_graph_completeness(ctx)
    complete = ctx.state.get(STATE_COMPLETE, False)
    print(f"\n  Graph complete: {_GREEN if complete else _YELLOW}{complete}{_RESET}")

    await match_segment(ctx)
    seg = ctx.state.get(STATE_SEGMENT_ID)
    return seg


# ── Zone 3 + 4 ───────────────────────────────────────────────────────────────

def _run_engine(
    g: TraitGraph,
    segment_id: str,
    confidence: float = 0.88,
) -> tuple:
    hyp = SegmentHypothesis(
        session_id=g.session_id,
        turn=4,
        model_version=settings.gemini_model,
        model_algorithm=ModelAlgorithm.LOGISTIC_REGRESSION,
        known_trait_count=len(g.known_nodes()),
        ranked_segments=[SegmentRank(segment_id, confidence)],
        disposition=HypothesisDisposition.ACTIVE,
    )
    bundle = ExplainabilityBundle(session_id=g.session_id)
    engine = SuggestionEngine()
    result = engine.evaluate(segment_id, g, hyp, bundle)

    _section("Zone 3 — Compliance Check Results")
    for ev in result.all_evaluations:
        for r in ev.rule_results:
            if r.outcome == "PASS":
                _ok(f"{r.rule_def.rule_id:<8} {r.rule_def.description[:55]}")
            elif r.outcome == "GATE":
                _warn(f"{r.rule_def.rule_id:<8} {r.rule_def.description[:55]}")
            else:
                _err(f"{r.rule_def.rule_id:<8} {r.rule_def.description[:55]}")

    delivery = DeliveryCoordinator().deliver(result, bundle)
    return result, bundle, delivery


# ── Result display ────────────────────────────────────────────────────────────

def _display(scenario: dict, result, bundle, delivery) -> None:
    gate = result.gate_disposition.value
    gate_colour = {
        "EMIT": _GREEN, "HUMAN_REVIEW": _YELLOW, "SUPPRESS": _RED
    }.get(gate, _RESET)

    _section("Zone 4 — Pipeline Result")
    print(f"  Gate disposition: {gate_colour}{_BOLD}{gate}{_RESET}")

    if result.top_suggestion:
        s = result.top_suggestion
        print(f"  Suggestion:       {s.suggestion_id} — {s.product_name}")
        print(f"  Product type:     {s.product_type}")
        print(f"  FCA ref:          {s.fca_ref}")

    if delivery.consumer_message:
        _section("Consumer Message (Zone 4 — Jinja2 template, no LLM)")
        for line in delivery.consumer_message.splitlines():
            print(f"  {line}")

    print(f"\n  {_BOLD}Audit confirmed:{_RESET} {delivery.audit_confirmed}  "
          f"(False until Spanner write — INV-05)")
    print(f"  {_BOLD}Symbolic trace entries:{_RESET} {len(bundle.symbolic_trace)}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    _banner("LBG Targeted Support Agent Platform — Local Demo v2.0", _BLUE)
    print(f"  Ontology: PS25/22 (live 6 April 2026)")
    print(f"  Model:    {settings.gemini_model}  (Zone 2, not called in demo)")
    print(f"  Domains:  Retail Investments · Structured Deposits · DC Pension")
    print()

    for i, sc in enumerate(DEMO_SCENARIOS, 1):
        print(f"  {_BOLD}{i:2}.{_RESET} {sc['label']}")

    try:
        choice = int(input(f"\n  {_BOLD}Enter 1–{len(DEMO_SCENARIOS)}:{_RESET} ").strip())
        if not 1 <= choice <= len(DEMO_SCENARIOS):
            raise ValueError
    except (ValueError, EOFError):
        print("Invalid choice. Exiting.")
        return

    sc = DEMO_SCENARIOS[choice - 1]
    preset = sc.get("suggest_answers")   # None = custom mode

    _banner(sc["label"], _BLUE)
    print(f"  Situation: {sc['situation_id']}  |  Intent: {sc['intent_id']}")

    _section("Zone 1 — Bank-Known Traits")
    from ts_agent.visualiser.data_adapter import CHAR_SHORT_LABEL
    for char_id, value in sc["bank_traits"].items():
        lbl = CHAR_SHORT_LABEL.get(char_id, char_id)
        print(f"  {lbl:<28} {value}")

    g = _build_graph(sc["bank_traits"], sc["situation_id"])

    if preset is not None:
        print(f"\n  {_YELLOW}Suggested answers shown.  Press Enter to use them "
              f"or type your own.{_RESET}")

    seg_id = await _run_gap_fill(g, sc["gap_fill_questions"], preset)

    if seg_id is None:
        _warn("Zone 2 found no matching segment — checking engine with expected segment.")
        seg_id = sc["segment_id"]

    print(f"\n  Matched segment: {_BOLD}{seg_id}{_RESET}")
    if seg_id in SEGMENTS:
        print(f"  Label:           {SEGMENTS[seg_id].label}")

    # Zone 2 → Zone 3 transition narrative
    print(f"\n  {_GREEN}✓{_RESET}  Zone 2 gap-fill conversation complete!")
    print(f"  {_BLUE}➤{_RESET}  Handing over to Suggestion Agent (Zone 3)...")
    print(f"      • Reviewing profile against 12 compliance rules")
    print(f"      • Identifying suitable financial products") 
    print(f"      • Delivery Agent will prepare personalized message")
    print()

    result, bundle, delivery = _run_engine(g, seg_id)
    _display(sc, result, bundle, delivery)

    print(f"\n{_BOLD}Run again?{_RESET}  python run_demo.py\n")


if __name__ == "__main__":
    asyncio.run(main())
