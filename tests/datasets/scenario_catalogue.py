"""
tests/datasets/scenario_catalogue.py
======================================
Canonical scenario catalogue — PS25/22 v2 ontology (live 6 April 2026).

Covers every (Situation × Segment × Suggestion) in the v2 ontology:
  14 happy-path EMIT scenarios  (one per segment)
   2 HUMAN_REVIEW scenarios     (vulnerability GATE, low-confidence GATE)
   6 SUPPRESS scenarios         (excluding characteristics triggered)
  ──
  22 total scenarios

All IDs are v2.  SAV/DEBT/INS/MORT IDs are removed — those domains are
explicitly out of scope per PS25/22 Ch.3 (DC-001).

Codex review notes
------------------
- No import from production code zones (pure data + catalogue lookups).
- ``tags`` used for selective test marking (zone2_suppress, vulnerability, etc.)
- ``consumer_answers`` are raw strings — coerced by _coerce_value in Zone 2.
- ``known_traits`` use v2 char_ids mapped to the v2 segment criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ts_agent.config.segments import (
    RULES,
    SEGMENT_TO_SUGGESTIONS,
    SEGMENTS,
    SITUATIONS,
    SUGGESTIONS,
)
from ts_agent.domain.models import GateDisposition


@dataclass
class Scenario:
    """A complete, self-contained test scenario."""
    scenario_id:         str
    description:         str
    situation_id:        str
    intent_id:           str
    expected_segment:    str
    expected_suggestion: str | None
    expected_gate:       GateDisposition
    known_traits:        dict[str, Any]
    consumer_answers:    list[tuple[str, str]]
    tags:                list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# RETAIL INVESTMENT SCENARIOS
# ──────────────────────────────────────────────────────────────────────────────

INV_001_HAPPY = Scenario(
    scenario_id="INV-001-EMIT-001",
    description="SEG-INV-001: £15k cash, no investment, 18mo tenure → SUG-INV-001 EMIT",
    situation_id="SIT-INV-001", intent_id="INTENT-INVEST-CASH",
    expected_segment="SEG-INV-001", expected_suggestion="SUG-INV-001",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 2, "CHAR-P1B-I1": False, "CHAR-P1C-I1": 1,
        "CHAR-F2B-I1": 15000.0, "CHAR-F2I-I1": False, "CHAR-F2L-I1": 18,
        "CHAR-F2A-I1": 600.0, "CHAR-F2G-I1": False,
    },
    consumer_answers=[("CHAR-B3A-I1", "3"), ("CHAR-B3B-I1", "1")],
    tags=["invest", "emit"],
)

INV_001_VULN = Scenario(
    scenario_id="INV-001-REVIEW-001",
    description="SEG-INV-001: vulnerability flag active → R-003 GATE → HUMAN_REVIEW",
    situation_id="SIT-INV-001", intent_id="INTENT-INVEST-CASH",
    expected_segment="SEG-INV-001", expected_suggestion="SUG-INV-001",
    expected_gate=GateDisposition.HUMAN_REVIEW,
    known_traits={
        "CHAR-P1A-I1": 2, "CHAR-P1B-I1": True,   # VULNERABLE
        "CHAR-P1C-I1": 1, "CHAR-F2B-I1": 15000.0,
        "CHAR-F2I-I1": False, "CHAR-F2L-I1": 18,
        "CHAR-F2A-I1": 600.0, "CHAR-F2G-I1": False,
    },
    consumer_answers=[("CHAR-B3A-I1", "3"), ("CHAR-B3B-I1", "1")],
    tags=["invest", "review", "vulnerability"],
)

INV_001_LOWCONF = Scenario(
    scenario_id="INV-001-REVIEW-LOWCONF",
    description="SEG-INV-001: ML confidence 0.52 < 0.75 → R-009 GATE → HUMAN_REVIEW",
    situation_id="SIT-INV-001", intent_id="INTENT-INVEST-CASH",
    expected_segment="SEG-INV-001", expected_suggestion="SUG-INV-001",
    expected_gate=GateDisposition.HUMAN_REVIEW,
    known_traits={
        "CHAR-P1A-I1": 2, "CHAR-P1B-I1": False, "CHAR-F2B-I1": 12000.0,
        "CHAR-F2I-I1": False, "CHAR-F2L-I1": 15,
        "CHAR-F2A-I1": 400.0, "CHAR-F2G-I1": False,
    },
    consumer_answers=[("CHAR-B3A-I1", "2"), ("CHAR-B3B-I1", "0")],
    tags=["invest", "review", "low-confidence", "LOWCONF"],
)

INV_002_HAPPY = Scenario(
    scenario_id="INV-002-EMIT-001",
    description="SEG-INV-002: young employed, £5k savings, no prior investment → SUG-INV-002 EMIT",
    situation_id="SIT-INV-002", intent_id="INTENT-FIRST-INVEST",
    expected_segment="SEG-INV-002", expected_suggestion="SUG-INV-002",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 2, "CHAR-P1B-I1": False, "CHAR-P1C-I1": 1,
        "CHAR-F2B-I1": 5000.0, "CHAR-F2I-I1": False,
        "CHAR-F2A-I1": 400.0, "CHAR-F2G-I1": False,
    },
    consumer_answers=[("CHAR-B3A-I1", "2"), ("CHAR-B3B-I1", "0")],
    tags=["invest", "emit", "first-time"],
)

INV_003_HAPPY = Scenario(
    scenario_id="INV-003-EMIT-001",
    description="SEG-INV-003: no ISA sub this tax year, £2k cash, within 90 days → SUG-INV-003 EMIT",
    situation_id="SIT-INV-003", intent_id="INTENT-ISA-ALLOWANCE",
    expected_segment="SEG-INV-003", expected_suggestion="SUG-INV-003",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 3, "CHAR-P1B-I1": False, "CHAR-F2B-I1": 2000.0,
        "CHAR-F2J-I1": False, "CHAR-F2K-I1": True, "CHAR-F2A-I1": 500.0,
    },
    consumer_answers=[("CHAR-B3A-I1", "3")],
    tags=["invest", "emit", "isa", "tax-year"],
)

INV_004_HAPPY = Scenario(
    scenario_id="INV-004-EMIT-001",
    description="SEG-INV-004: £20k lump sum, no investment instruction → SUG-INV-004 EMIT",
    situation_id="SIT-INV-004", intent_id="INTENT-LUMP-SUM",
    expected_segment="SEG-INV-004", expected_suggestion="SUG-INV-004",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 3, "CHAR-P1B-I1": False, "CHAR-F2M-I1": 20000.0,
        "CHAR-F2I-I1": False, "CHAR-F2H-I1": False, "CHAR-F2A-I1": 800.0,
    },
    consumer_answers=[("CHAR-B3A-I1", "3"), ("CHAR-B3B-I1", "1")],
    tags=["invest", "emit", "lump-sum"],
)

INV_004_ABOVE_THRESHOLD = Scenario(
    scenario_id="INV-004-SUPPRESS-001",
    description="SEG-INV-004: £80k lump sum > £75k threshold → EC-INV-004-01 → SUPPRESS (exit TS)",
    situation_id="SIT-INV-004", intent_id="INTENT-LUMP-SUM",
    expected_segment="SEG-INV-004", expected_suggestion=None,
    expected_gate=GateDisposition.SUPPRESS,
    known_traits={
        "CHAR-P1A-I1": 3, "CHAR-P1B-I1": False, "CHAR-F2M-I1": 80000.0,
        "CHAR-F2I-I1": False, "CHAR-F2H-I1": False, "CHAR-F2A-I1": 800.0,
    },
    consumer_answers=[],
    tags=["invest", "suppress", "lump-sum", "zone2_suppress"],
)

INV_005_HAPPY = Scenario(
    scenario_id="INV-005-EMIT-001",
    description="SEG-INV-005: investment inactive 14 months → SUG-INV-005 re-engagement EMIT",
    situation_id="SIT-INV-005", intent_id="INTENT-REVIEW-INVESTMENT",
    expected_segment="SEG-INV-005", expected_suggestion="SUG-INV-005",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 3, "CHAR-P1B-I1": False,
        "CHAR-F2I-I1": True, "CHAR-F2N-I1": 14, "CHAR-F2A-I1": 400.0,
    },
    consumer_answers=[],
    tags=["invest", "emit", "dormant"],
)

INV_006_HAPPY = Scenario(
    scenario_id="INV-006-EMIT-001",
    description="SEG-INV-006: regular £100/mo saving 8 months, no investment → SUG-INV-006 EMIT",
    situation_id="SIT-INV-006", intent_id="INTENT-REGULAR-INVEST",
    expected_segment="SEG-INV-006", expected_suggestion="SUG-INV-006",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 2, "CHAR-P1B-I1": False,
        "CHAR-F2O-I1": 100.0, "CHAR-F2P-I1": 8,
        "CHAR-F2I-I1": False, "CHAR-F2G-I1": False, "CHAR-F2A-I1": 350.0,
    },
    consumer_answers=[("CHAR-B3A-I1", "2")],
    tags=["invest", "emit", "regular-saver"],
)

# ──────────────────────────────────────────────────────────────────────────────
# STRUCTURED DEPOSIT SCENARIOS
# ──────────────────────────────────────────────────────────────────────────────

SD_001_HAPPY = Scenario(
    scenario_id="SD-001-EMIT-001",
    description="SEG-SD-001: deposit maturing 45 days, no reinvestment instruction → SUG-SD-001 EMIT",
    situation_id="SIT-SD-001", intent_id="INTENT-DEPOSIT-MATURITY",
    expected_segment="SEG-SD-001", expected_suggestion="SUG-SD-001",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 4, "CHAR-P1B-I1": False,
        "CHAR-F2Q-I1": 45, "CHAR-F2R-I1": False,
        "CHAR-F2B-I1": 30000.0, "CHAR-F2A-I1": 800.0,
    },
    consumer_answers=[],
    tags=["structured-deposit", "emit"],
)

SD_001_ABOVE_THRESHOLD = Scenario(
    scenario_id="SD-001-SUPPRESS-001",
    description="SEG-SD-001: deposit £120k > £100k → EC-SD-001-01 → SUPPRESS",
    situation_id="SIT-SD-001", intent_id="INTENT-DEPOSIT-MATURITY",
    expected_segment="SEG-SD-001", expected_suggestion=None,
    expected_gate=GateDisposition.SUPPRESS,
    known_traits={
        "CHAR-P1A-I1": 4, "CHAR-P1B-I1": False,
        "CHAR-F2Q-I1": 30, "CHAR-F2R-I1": False,
        "CHAR-F2B-I1": 120000.0, "CHAR-F2A-I1": 800.0,
    },
    consumer_answers=[],
    tags=["structured-deposit", "suppress", "zone2_suppress"],
)

# ──────────────────────────────────────────────────────────────────────────────
# DC PENSION ACCUMULATION SCENARIOS
# ──────────────────────────────────────────────────────────────────────────────

PEN_001_HAPPY = Scenario(
    scenario_id="PEN-001-EMIT-001",
    description="SEG-PEN-001: 35yo DC member, 8% AE minimum contribution, projected shortfall → SUG-PEN-001 EMIT",
    situation_id="SIT-PEN-001", intent_id="INTENT-PENSION-CONTRIBUTIONS",
    expected_segment="SEG-PEN-001", expected_suggestion="SUG-PEN-001",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 2, "CHAR-P1B-I1": False, "CHAR-P1C-I1": 1,
        "CHAR-P2A-I1": 8, "CHAR-P2B-I1": True, "CHAR-P2C-I1": True,
        "CHAR-P2D-I1": 30, "CHAR-F2H-I1": False, "CHAR-F2A-I1": 500.0,
    },
    consumer_answers=[("CHAR-P2A-I1", "8")],
    tags=["pension", "accumulation", "emit"],
)

PEN_001_HARDSHIP = Scenario(
    scenario_id="PEN-001-SUPPRESS-001",
    description="SEG-PEN-001: financial hardship flag active → EC-PEN-001-02 → SUPPRESS",
    situation_id="SIT-PEN-001", intent_id="INTENT-PENSION-CONTRIBUTIONS",
    expected_segment="SEG-PEN-001", expected_suggestion=None,
    expected_gate=GateDisposition.SUPPRESS,
    known_traits={
        "CHAR-P1A-I1": 2, "CHAR-P1B-I1": False,
        "CHAR-P2A-I1": 8, "CHAR-P2B-I1": True, "CHAR-P2C-I1": True,
        "CHAR-P2D-I1": 25, "CHAR-F2H-I1": True,   # hardship → EC
        "CHAR-F2A-I1": 50.0,
    },
    consumer_answers=[],
    tags=["pension", "accumulation", "suppress", "zone2_suppress"],
)

PEN_002_HAPPY = Scenario(
    scenario_id="PEN-002-EMIT-001",
    description="SEG-PEN-002: 100% default fund, no active selection, age band 2 → SUG-PEN-002 EMIT",
    situation_id="SIT-PEN-002", intent_id="INTENT-PENSION-FUNDS",
    expected_segment="SEG-PEN-002", expected_suggestion="SUG-PEN-002",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 2, "CHAR-P1B-I1": False,
        "CHAR-P2E-I1": True, "CHAR-P2F-I1": False,
        "CHAR-P2B-I1": True, "CHAR-P2G-I1": False, "CHAR-F2A-I1": 400.0,
    },
    consumer_answers=[],
    tags=["pension", "accumulation", "emit", "fund-switch"],
)

PEN_003_HAPPY = Scenario(
    scenario_id="PEN-003-EMIT-001",
    description="SEG-PEN-003: salary increase life event, contribution 9% < 12% → SUG-PEN-003 EMIT",
    situation_id="SIT-PEN-003", intent_id="INTENT-LIFE-EVENT",
    expected_segment="SEG-PEN-003", expected_suggestion="SUG-PEN-003",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 3, "CHAR-P1B-I1": False,
        "CHAR-P2H-I1": True, "CHAR-P2A-I1": 9,
        "CHAR-P2B-I1": True, "CHAR-F2H-I1": False,
        "CHAR-P2D-I1": 20, "CHAR-F2A-I1": 700.0,
    },
    consumer_answers=[("CHAR-P2H-I1", "yes")],
    tags=["pension", "accumulation", "emit", "life-event"],
)

# ──────────────────────────────────────────────────────────────────────────────
# DC PENSION DECUMULATION SCENARIOS
# ──────────────────────────────────────────────────────────────────────────────

DEC_001_HAPPY = Scenario(
    scenario_id="DEC-001-EMIT-001",
    description="SEG-DEC-001: 55yo DC member, no access plan → SUG-DEC-001 pathway direction EMIT",
    situation_id="SIT-DEC-001", intent_id="INTENT-RETIREMENT-PLAN",
    expected_segment="SEG-DEC-001", expected_suggestion="SUG-DEC-001",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 5, "CHAR-P1B-I1": False,
        "CHAR-P2B-I1": True, "CHAR-P2I-I1": False,
        "CHAR-P2J-I1": False, "CHAR-P2K-I1": False, "CHAR-F2A-I1": 600.0,
    },
    consumer_answers=[],
    tags=["pension", "decumulation", "emit", "pension-wise"],
)

DEC_001_DB_TRANSFER = Scenario(
    scenario_id="DEC-001-SUPPRESS-001",
    description="SEG-DEC-001: DB transfer flag active → EC-DEC-001-02 mandatory referral → SUPPRESS",
    situation_id="SIT-DEC-001", intent_id="INTENT-RETIREMENT-PLAN",
    expected_segment="SEG-DEC-001", expected_suggestion=None,
    expected_gate=GateDisposition.SUPPRESS,
    known_traits={
        "CHAR-P1A-I1": 5, "CHAR-P1B-I1": False,
        "CHAR-P2B-I1": True, "CHAR-P2I-I1": False,
        "CHAR-P2J-I1": False, "CHAR-P2K-I1": True,   # DB transfer → EC
        "CHAR-F2A-I1": 600.0,
    },
    consumer_answers=[],
    tags=["pension", "decumulation", "suppress", "db-transfer", "zone2_suppress"],
)

DEC_002_HAPPY = Scenario(
    scenario_id="DEC-002-EMIT-001",
    description="SEG-DEC-002: £18k DC pot, 3 years to retirement, no access → SUG-DEC-002 EMIT",
    situation_id="SIT-DEC-002", intent_id="INTENT-TAKE-PENSION",
    expected_segment="SEG-DEC-002", expected_suggestion="SUG-DEC-002",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 5, "CHAR-P1B-I1": False,
        "CHAR-P2L-I1": 18000.0, "CHAR-P2D-I1": 3,
        "CHAR-P2I-I1": False, "CHAR-F2A-I1": 300.0,
    },
    consumer_answers=[],
    tags=["pension", "decumulation", "emit", "small-pot"],
)

DEC_002_ABOVE_CEILING = Scenario(
    scenario_id="DEC-002-SUPPRESS-001",
    description="SEG-DEC-002: £35k pot > £30k ceiling → EC-DEC-002-02 → SUPPRESS",
    situation_id="SIT-DEC-002", intent_id="INTENT-SMALL-POT",
    expected_segment="SEG-DEC-002", expected_suggestion=None,
    expected_gate=GateDisposition.SUPPRESS,
    known_traits={
        "CHAR-P1A-I1": 5, "CHAR-P1B-I1": False,
        "CHAR-P2L-I1": 35000.0,   # > £30k ceiling → EC
        "CHAR-P2D-I1": 3, "CHAR-P2I-I1": False, "CHAR-F2A-I1": 400.0,
    },
    consumer_answers=[],
    tags=["pension", "decumulation", "suppress", "zone2_suppress"],
)

DEC_003_HAPPY = Scenario(
    scenario_id="DEC-003-EMIT-001",
    description="SEG-DEC-003: annuity interest expressed, DC pension, age ≥50 → "
                "SUG-DEC-003 features + MoneyHelper EMIT (no product recommendation)",
    situation_id="SIT-DEC-003", intent_id="INTENT-ANNUITY",
    expected_segment="SEG-DEC-003", expected_suggestion="SUG-DEC-003",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 5, "CHAR-P1B-I1": False,
        "CHAR-P2M-I1": True, "CHAR-P2B-I1": True,
        "CHAR-P2K-I1": False, "CHAR-F2A-I1": 400.0,
    },
    consumer_answers=[],
    tags=["pension", "decumulation", "emit", "annuity", "moneyhelper-mandatory"],
)

DEC_004_HAPPY = Scenario(
    scenario_id="DEC-004-EMIT-001",
    description="SEG-DEC-004: 20 months no drawdown review, cash drag flag → "
                "SUG-DEC-004 re-engagement EMIT (do-nothing valid)",
    situation_id="SIT-DEC-004", intent_id="INTENT-DRAWDOWN-REVIEW",
    expected_segment="SEG-DEC-004", expected_suggestion="SUG-DEC-004",
    expected_gate=GateDisposition.EMIT,
    known_traits={
        "CHAR-P1A-I1": 5, "CHAR-P1B-I1": False,
        "CHAR-P2J-I1": True, "CHAR-P2N-I1": 20,
        "CHAR-P2O-I1": True, "CHAR-P2P-I1": False, "CHAR-F2A-I1": 600.0,
    },
    consumer_answers=[],
    tags=["pension", "decumulation", "emit", "drawdown"],
)

# ──────────────────────────────────────────────────────────────────────────────
# EDGE CASE
# ──────────────────────────────────────────────────────────────────────────────

EDGE_NO_MATCH = Scenario(
    scenario_id="EDGE-NO-MATCH-001",
    description="Edge case: non-existent segment → SUPPRESS (zero candidates)",
    situation_id="SIT-INV-001", intent_id="INTENT-INVEST-CASH",
    expected_segment="SEG-UNKNOWN-999",
    expected_suggestion=None,
    expected_gate=GateDisposition.SUPPRESS,
    known_traits={"CHAR-P1A-I1": 2, "CHAR-P1B-I1": False},
    consumer_answers=[],
    tags=["edge", "suppress", "no-match"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Master list
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS: list[Scenario] = [
    # Retail investments
    INV_001_HAPPY, INV_001_VULN, INV_001_LOWCONF,
    INV_002_HAPPY, INV_003_HAPPY, INV_004_HAPPY, INV_004_ABOVE_THRESHOLD,
    INV_005_HAPPY, INV_006_HAPPY,
    # Structured deposits
    SD_001_HAPPY, SD_001_ABOVE_THRESHOLD,
    # DC pension accumulation
    PEN_001_HAPPY, PEN_001_HARDSHIP, PEN_002_HAPPY, PEN_003_HAPPY,
    # DC pension decumulation
    DEC_001_HAPPY, DEC_001_DB_TRANSFER, DEC_002_HAPPY, DEC_002_ABOVE_CEILING,
    DEC_003_HAPPY, DEC_004_HAPPY,
    # Edge cases
    EDGE_NO_MATCH,
]

# Derived lookups
SCENARIOS_BY_ID:      dict[str, Scenario]       = {s.scenario_id: s for s in SCENARIOS}
SCENARIOS_BY_SEGMENT: dict[str, list[Scenario]] = {}
for _s in SCENARIOS:
    SCENARIOS_BY_SEGMENT.setdefault(_s.expected_segment, []).append(_s)
SCENARIOS_BY_TAG: dict[str, list[Scenario]] = {}
for _s in SCENARIOS:
    for _t in _s.tags:
        SCENARIOS_BY_TAG.setdefault(_t, []).append(_s)


def emit_scenarios()         -> list[Scenario]:
    return [s for s in SCENARIOS if s.expected_gate == GateDisposition.EMIT]

def suppress_scenarios()     -> list[Scenario]:
    return [s for s in SCENARIOS if s.expected_gate == GateDisposition.SUPPRESS]

def review_scenarios()       -> list[Scenario]:
    return [s for s in SCENARIOS if s.expected_gate == GateDisposition.HUMAN_REVIEW]

def zone2_suppress_scenarios() -> list[Scenario]:
    """Scenarios that suppress at Zone 2 (no segment match) before Zone 3."""
    return [s for s in SCENARIOS if "zone2_suppress" in s.tags]

def scenarios_for_situation(situation_id: str) -> list[Scenario]:
    return [s for s in SCENARIOS if s.situation_id == situation_id]

def scenarios_for_domain(domain: str) -> list[Scenario]:
    prefix_map = {
        "invest": "SIT-INV", "structured_deposit": "SIT-SD",
        "pension_accumulation": "SIT-PEN", "pension_decumulation": "SIT-DEC",
    }
    prefix = prefix_map.get(domain, domain)
    return [s for s in SCENARIOS if s.situation_id.startswith(prefix)]
