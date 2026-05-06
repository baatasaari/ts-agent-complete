"""
generate_test_data.py
=====================
Generates three test datasets for the LBG TS Agent Platform v2.

All data uses the PS25/22 live ontology (6 April 2026):
  Domains: Retail Investments, Structured Deposits,
           DC Pension Accumulation, DC Pension Decumulation

OUT-OF-SCOPE (not generated): Savings, Debt, Insurance, Mortgage.

Outputs (written to project root, i.e. the directory containing this script):
  ts_agent_consumer_profiles.csv  — 3,000 consumer records, all v2 traits,
                                     gate_disposition label, 12 rule outcomes.
  ts_agent_ml_training.csv        — 3,000 labelled records for ML training.
                                     Columns = ALL_FEATURE_COLUMNS exactly.
  ts_agent_sample_prompts.csv     — Consumer utterances for every v2 trait
                                     and intent. 3 paraphrase variants each.

Usage
-----
  cd ts_agent
  python generate_test_data.py
"""
from __future__ import annotations

import math
import os
import random
import uuid

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT = os.path.dirname(os.path.abspath(__file__))

rng = np.random.default_rng(42)
random.seed(42)


def uid() -> str:
    return str(uuid.uuid4())


def choice(lst):
    return random.choice(lst)


def norm(mu, sigma, lo=None, hi=None) -> float:
    v = float(rng.normal(mu, sigma))
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    return round(v, 2)


def boolchoice(p_true: float = 0.5) -> bool:
    return rng.random() < p_true


# ──────────────────────────────────────────────────────────────────────────────
# Per-segment trait distributions
# ──────────────────────────────────────────────────────────────────────────────

SEG_PROFILES = {

    # ── RETAIL INVESTMENT SEGMENTS ────────────────────────────────────────────

    "SEG-INV-001": dict(
        situation_id="SIT-INV-001",
        suggestion_id="SUG-INV-001",
        product_type="STOCKS_SHARES_ISA",
        intents=["INTENT-INVEST-CASH", "INTENT-ISA-OPEN"],
        # Including characteristics
        age_band=lambda: int(norm(2.8, 0.7, 1, 5)),
        employment_status=lambda: choice([1, 1, 1, 2, 3]),
        savings_balance=lambda: norm(28000, 12000, 10000, 100000),
        has_investment=lambda: False,               # CHAR-F2I-I1 == False (IC)
        account_tenure_months=lambda: int(norm(28, 10, 12, 120)),
        monthly_surplus=lambda: norm(550, 200, 100, 2500),
        has_high_cost_debt=lambda: False,           # EC if True
        has_hardship=lambda: False,
        # Other v2 traits (not directly in criteria but in graph)
        risk_appetite_score=lambda: norm(2.8, 0.8, 1.5, 4.5),
        investment_experience=lambda: choice([0, 1, 1, 2]),
        channel=lambda: choice([0, 0, 1]),
        vulnerability=lambda: boolchoice(0.05),
    ),

    "SEG-INV-002": dict(
        situation_id="SIT-INV-002",
        suggestion_id="SUG-INV-002",
        product_type="STOCKS_SHARES_ISA_REGULAR",
        intents=["INTENT-FIRST-INVEST", "INTENT-HOW-TO-INVEST"],
        age_band=lambda: int(norm(1.8, 0.5, 1, 3)),   # 18–40
        employment_status=lambda: choice([1, 1, 2]),
        savings_balance=lambda: norm(6000, 4000, 500, 25000),
        has_investment=lambda: False,
        account_tenure_months=lambda: int(norm(18, 8, 3, 60)),
        monthly_surplus=lambda: norm(380, 150, 50, 1500),
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        risk_appetite_score=lambda: norm(2.2, 0.7, 1.0, 4.0),
        investment_experience=lambda: 0,              # no experience (IC)
        channel=lambda: choice([0, 0, 1]),
        vulnerability=lambda: boolchoice(0.05),
    ),

    "SEG-INV-003": dict(
        situation_id="SIT-INV-003",
        suggestion_id="SUG-INV-003",
        product_type="STOCKS_SHARES_ISA_TAX_YEAR",
        intents=["INTENT-ISA-ALLOWANCE", "INTENT-TAX-YEAR-END"],
        age_band=lambda: int(norm(2.5, 0.9, 1, 5)),
        employment_status=lambda: choice([1, 1, 2, 3]),
        savings_balance=lambda: norm(4500, 3000, 500, 40000),
        has_investment=lambda: boolchoice(0.3),       # may or may not hold investment
        isa_sub_this_year=lambda: False,              # CHAR-F2J-I1 == False (IC)
        within_90d_tax_year=lambda: True,             # CHAR-F2K-I1 == True (IC)
        account_tenure_months=lambda: int(norm(24, 12, 6, 120)),
        monthly_surplus=lambda: norm(450, 200, 100, 2000),
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        risk_appetite_score=lambda: norm(2.5, 0.9, 1.0, 5.0),
        investment_experience=lambda: choice([0, 1, 1, 2]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.05),
    ),

    "SEG-INV-004": dict(
        situation_id="SIT-INV-004",
        suggestion_id="SUG-INV-004",
        product_type="ISA_AND_GIA_LUMP_SUM",
        intents=["INTENT-LUMP-SUM", "INTENT-WINDFALL"],
        age_band=lambda: int(norm(3.0, 0.9, 1, 5)),
        employment_status=lambda: choice([1, 1, 2, 3]),
        savings_balance=lambda: norm(15000, 8000, 2000, 50000),
        lump_sum_amount=lambda: norm(22000, 10000, 5000, 75000),  # IC: £5k–£75k
        has_investment=lambda: False,
        account_tenure_months=lambda: int(norm(24, 12, 6, 120)),
        monthly_surplus=lambda: norm(600, 250, 100, 2500),
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        risk_appetite_score=lambda: norm(3.0, 0.9, 1.5, 5.0),
        investment_experience=lambda: choice([0, 1, 1, 2]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.05),
    ),

    "SEG-INV-005": dict(
        situation_id="SIT-INV-005",
        suggestion_id="SUG-INV-005",
        product_type="REVIEW_PROMPT_DO_NOTHING",
        intents=["INTENT-REVIEW-INVESTMENT"],
        age_band=lambda: int(norm(3.0, 0.9, 1, 5)),
        employment_status=lambda: choice([1, 1, 2, 3]),
        savings_balance=lambda: norm(12000, 8000, 1000, 60000),
        has_investment=lambda: True,                  # IC: holds investment
        months_inactive=lambda: int(norm(18, 6, 12, 48)),  # IC: >= 12 months
        account_tenure_months=lambda: int(norm(36, 12, 12, 120)),
        monthly_surplus=lambda: norm(450, 200, 100, 2000),
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        risk_appetite_score=lambda: norm(2.5, 0.9, 1.0, 5.0),
        investment_experience=lambda: choice([1, 1, 2, 3]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.05),
    ),

    "SEG-INV-006": dict(
        situation_id="SIT-INV-006",
        suggestion_id="SUG-INV-006",
        product_type="STOCKS_SHARES_ISA_REGULAR_ADDITION",
        intents=["INTENT-REGULAR-INVEST", "INTENT-ISA-OPEN"],
        age_band=lambda: int(norm(2.2, 0.7, 1, 4)),
        employment_status=lambda: choice([1, 1, 2]),
        savings_balance=lambda: norm(5000, 3000, 500, 25000),
        regular_saving_amount=lambda: norm(130, 60, 50, 500),   # IC: >= £50
        consecutive_saving_months=lambda: int(norm(10, 3, 6, 36)),  # IC: >= 6
        has_investment=lambda: False,
        account_tenure_months=lambda: int(norm(20, 8, 6, 60)),
        monthly_surplus=lambda: norm(380, 150, 80, 1500),
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        risk_appetite_score=lambda: norm(2.3, 0.7, 1.0, 4.0),
        investment_experience=lambda: choice([0, 0, 1]),
        channel=lambda: choice([0, 0, 1]),
        vulnerability=lambda: boolchoice(0.05),
    ),

    # ── STRUCTURED DEPOSIT ────────────────────────────────────────────────────

    "SEG-SD-001": dict(
        situation_id="SIT-SD-001",
        suggestion_id="SUG-SD-001",
        product_type="MATURITY_REINVESTMENT_OPTIONS",
        intents=["INTENT-DEPOSIT-MATURITY", "INTENT-REINVEST"],
        age_band=lambda: int(norm(3.5, 0.8, 2, 5)),
        employment_status=lambda: choice([1, 2, 3, 3]),
        savings_balance=lambda: norm(35000, 20000, 2000, 100000),
        days_to_maturity=lambda: int(norm(30, 15, 1, 60)),  # IC: <= 60 days
        has_reinvestment_instruction=lambda: False,           # IC: no instruction
        account_tenure_months=lambda: int(norm(40, 15, 12, 120)),
        monthly_surplus=lambda: norm(700, 300, 100, 3000),
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        risk_appetite_score=lambda: norm(2.5, 0.8, 1.0, 4.5),
        investment_experience=lambda: choice([0, 1, 1, 2]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.04),
    ),

    # ── DC PENSION ACCUMULATION ───────────────────────────────────────────────

    "SEG-PEN-001": dict(
        situation_id="SIT-PEN-001",
        suggestion_id="SUG-PEN-001",
        product_type="DC_PENSION_CONTRIBUTION_INCREASE",
        intents=["INTENT-PENSION-CONTRIBUTIONS", "INTENT-SAVE-MORE"],
        age_band=lambda: int(norm(2.8, 0.6, 2, 4)),   # 25–57
        employment_status=lambda: choice([1, 1, 2]),
        savings_balance=lambda: norm(5000, 4000, 0, 30000),
        pension_contribution_pct=lambda: norm(6, 1.5, 2, 8),  # IC: <= 8%
        is_active_dc_member=lambda: True,
        has_projected_shortfall=lambda: True,
        years_to_retirement=lambda: int(norm(22, 8, 6, 40)),  # EC if <= 5
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        monthly_surplus=lambda: norm(450, 180, 50, 2000),
        account_tenure_months=lambda: int(norm(36, 15, 12, 120)),
        risk_appetite_score=lambda: norm(2.5, 0.8, 1.0, 4.5),
        investment_experience=lambda: choice([0, 1, 1]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.05),
    ),

    "SEG-PEN-002": dict(
        situation_id="SIT-PEN-002",
        suggestion_id="SUG-PEN-002",
        product_type="DC_PENSION_FUND_SWITCH",
        intents=["INTENT-PENSION-FUNDS", "INTENT-FUND-SWITCH"],
        age_band=lambda: int(norm(2.5, 0.8, 1, 5)),
        employment_status=lambda: choice([1, 1, 2]),
        savings_balance=lambda: norm(4000, 3000, 0, 20000),
        in_100pct_default_fund=lambda: True,           # IC
        no_active_fund_selection=lambda: True,         # IC: no selection ever
        is_active_dc_member=lambda: True,
        has_recent_engagement=lambda: False,           # EC if True
        years_to_retirement=lambda: int(norm(20, 8, 5, 40)),
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        monthly_surplus=lambda: norm(400, 180, 50, 2000),
        pension_contribution_pct=lambda: norm(8, 2, 4, 15),
        account_tenure_months=lambda: int(norm(36, 15, 12, 120)),
        risk_appetite_score=lambda: norm(2.3, 0.7, 1.0, 4.5),
        investment_experience=lambda: choice([0, 0, 1]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.05),
    ),

    "SEG-PEN-003": dict(
        situation_id="SIT-PEN-003",
        suggestion_id="SUG-PEN-003",
        product_type="DC_PENSION_CONTRIBUTION_REVIEW",
        intents=["INTENT-LIFE-EVENT", "INTENT-PENSION-REVIEW"],
        age_band=lambda: int(norm(2.8, 0.7, 1, 5)),
        employment_status=lambda: choice([1, 1, 2]),
        savings_balance=lambda: norm(6000, 4000, 0, 30000),
        life_event_signal=lambda: True,                # IC
        pension_contribution_pct=lambda: norm(8, 2, 4, 12),  # IC: <= 12%
        is_active_dc_member=lambda: True,
        years_to_retirement=lambda: int(norm(18, 8, 6, 40)),  # EC if <= 5
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        monthly_surplus=lambda: norm(550, 200, 100, 2500),
        account_tenure_months=lambda: int(norm(32, 12, 12, 120)),
        risk_appetite_score=lambda: norm(2.5, 0.8, 1.0, 4.5),
        investment_experience=lambda: choice([0, 1]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.05),
    ),

    # ── DC PENSION DECUMULATION ───────────────────────────────────────────────

    "SEG-DEC-001": dict(
        situation_id="SIT-DEC-001",
        suggestion_id="SUG-DEC-001",
        product_type="DECUMULATION_PATHWAY_DIRECTION",
        intents=["INTENT-RETIREMENT-PLAN", "INTENT-PENSION-ACCESS"],
        age_band=lambda: int(norm(4.2, 0.6, 3, 5)),   # 45–65
        employment_status=lambda: choice([1, 2, 3, 3]),
        savings_balance=lambda: norm(15000, 10000, 1000, 80000),
        is_active_dc_member=lambda: True,
        has_retirement_access_plan=lambda: False,      # IC: no plan
        already_in_drawdown=lambda: False,             # EC if True
        has_db_transfer=lambda: False,                 # EC if True (mandatory referral)
        years_to_retirement=lambda: int(norm(10, 5, 1, 25)),
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        monthly_surplus=lambda: norm(500, 200, 50, 2500),
        pension_contribution_pct=lambda: norm(10, 3, 4, 20),
        pension_pot_value=lambda: norm(60000, 35000, 5000, 300000),
        account_tenure_months=lambda: int(norm(48, 18, 12, 120)),
        risk_appetite_score=lambda: norm(2.5, 0.9, 1.0, 5.0),
        investment_experience=lambda: choice([0, 1, 2]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.05),
    ),

    "SEG-DEC-002": dict(
        situation_id="SIT-DEC-002",
        suggestion_id="SUG-DEC-002",
        product_type="DC_PENSION_SMALL_POT_ACCESS",
        intents=["INTENT-TAKE-PENSION", "INTENT-SMALL-POT"],
        age_band=lambda: int(norm(4.5, 0.5, 4, 5)),   # 55+
        employment_status=lambda: choice([1, 2, 3, 3]),
        savings_balance=lambda: norm(8000, 5000, 500, 30000),
        pension_pot_value=lambda: norm(16000, 6000, 5000, 30000),  # IC: £5k–£30k
        years_to_retirement=lambda: int(norm(3, 1.5, 0, 5)),      # IC: <= 5
        has_retirement_access_plan=lambda: False,
        already_in_drawdown=lambda: False,
        has_db_transfer=lambda: False,
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        monthly_surplus=lambda: norm(350, 150, 50, 1500),
        is_active_dc_member=lambda: True,
        account_tenure_months=lambda: int(norm(48, 18, 12, 120)),
        risk_appetite_score=lambda: norm(2.0, 0.8, 1.0, 4.0),
        investment_experience=lambda: choice([0, 0, 1]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.06),
    ),

    "SEG-DEC-003": dict(
        situation_id="SIT-DEC-003",
        suggestion_id="SUG-DEC-003",
        product_type="ANNUITY_FEATURES_REFERRAL",
        intents=["INTENT-ANNUITY", "INTENT-GUARANTEED-INCOME"],
        age_band=lambda: int(norm(4.5, 0.4, 4, 5)),   # >= 50 (band 4+)
        employment_status=lambda: choice([2, 3, 3]),
        savings_balance=lambda: norm(20000, 15000, 1000, 100000),
        expressed_annuity_interest=lambda: True,       # IC
        is_active_dc_member=lambda: True,
        has_db_transfer=lambda: False,                 # EC (mandatory referral if True)
        years_to_retirement=lambda: int(norm(5, 3, 0, 15)),
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        monthly_surplus=lambda: norm(400, 180, 50, 2000),
        pension_pot_value=lambda: norm(45000, 25000, 5000, 200000),
        pension_contribution_pct=lambda: norm(10, 3, 4, 20),
        account_tenure_months=lambda: int(norm(48, 18, 12, 120)),
        risk_appetite_score=lambda: norm(2.0, 0.8, 1.0, 4.0),
        investment_experience=lambda: choice([0, 1]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.06),
    ),

    "SEG-DEC-004": dict(
        situation_id="SIT-DEC-004",
        suggestion_id="SUG-DEC-004",
        product_type="DRAWDOWN_REVIEW_PROMPT",
        intents=["INTENT-DRAWDOWN-REVIEW"],
        age_band=lambda: int(norm(4.5, 0.5, 4, 5)),
        employment_status=lambda: choice([2, 3, 3]),
        savings_balance=lambda: norm(10000, 8000, 500, 60000),
        already_in_drawdown=lambda: True,             # IC
        months_since_review=lambda: int(norm(24, 6, 18, 60)),  # IC: >= 18
        has_cash_drag_flag=lambda: True,              # IC
        has_recent_adviser=lambda: False,             # EC if True
        years_to_retirement=lambda: int(norm(3, 2, 0, 10)),
        has_high_cost_debt=lambda: False,
        has_hardship=lambda: False,
        monthly_surplus=lambda: norm(400, 200, 50, 2000),
        pension_pot_value=lambda: norm(55000, 30000, 10000, 250000),
        is_active_dc_member=lambda: True,
        pension_contribution_pct=lambda: 0,
        account_tenure_months=lambda: int(norm(48, 18, 12, 120)),
        risk_appetite_score=lambda: norm(2.2, 0.8, 1.0, 4.5),
        investment_experience=lambda: choice([1, 1, 2]),
        channel=lambda: choice([0, 1]),
        vulnerability=lambda: boolchoice(0.06),
    ),
}

SEGMENT_IDS = list(SEG_PROFILES.keys())


def _call(v):
    """Call lambdas; return other values as-is."""
    return v() if callable(v) else v


def _get(profile, key, default=None):
    val = profile.get(key, default)
    return _call(val)


# ──────────────────────────────────────────────────────────────────────────────
# Rule evaluation — maps to v2 compliance checks
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_rules(row: dict) -> tuple[dict, str]:
    """Evaluate 12 legacy rule IDs; return (outcomes, gate_disposition)."""
    seg      = row["segment_id"]
    conf     = float(row.get("ml_confidence", 0.88))
    vuln     = bool(row.get("CHAR_P1B_I1_vulnerability", False))
    age      = int(row.get("CHAR_P1A_I1_age_band", 2))
    surplus  = float(row.get("CHAR_F2A_I1_monthly_surplus", 500))
    risk     = float(row.get("CHAR_B3A_I1_risk_appetite", 2.5))
    exp      = int(row.get("CHAR_B3B_I1_invest_exp", 0))
    pt       = SEG_PROFILES[seg]["product_type"]

    # R-001: segment match — always PASS (generated from segment profile)
    # R-002: age >= 18 (band >= 1)
    # R-003: vulnerability → GATE
    # R-004: existing product (ISA or investment) — for INV segments
    # R-005: risk appetite appropriate
    # R-006: investment experience (ISA products need >= 1)
    # R-007: surplus >= 0
    # R-008: no duplicate contact (soft — always PASS)
    # R-009: ML confidence >= 0.75 → GATE if below
    # R-010: product in scope (always PASS for v2 catalogue)
    # R-011: Consumer Duty (GATE if vulnerable + high-risk product)
    # R-012: FCA eligibility cross-check (always PASS)

    r = {}
    r["R-001"] = "PASS"
    r["R-002"] = "PASS" if age >= 1 else "FAIL"
    r["R-003"] = "GATE" if vuln else "PASS"

    # R-004: check existing product holding (PS25/22 excluding characteristic)
    # INV-003 (tax-year-end ISA allowance): excluding char is already subscribed (CHAR-F2J-I1)
    # Other ISA products: excluding char is already holding investment product (CHAR-F2I-I1)
    if pt == "STOCKS_SHARES_ISA_TAX_YEAR":
        # R-004 FAIL if consumer has already used the annual ISA allowance this year
        holds = bool(row.get("CHAR_F2J_I1_isa_sub_this_year", False))
        r["R-004"] = "FAIL" if holds else "PASS"
    elif pt in {
        "STOCKS_SHARES_ISA", "STOCKS_SHARES_ISA_REGULAR",
        "STOCKS_SHARES_ISA_REGULAR_ADDITION", "ISA_AND_GIA_LUMP_SUM",
    }:
        holds = bool(row.get("CHAR_F2I_I1_has_investment", False))
        r["R-004"] = "FAIL" if holds else "PASS"
    else:
        r["R-004"] = "PASS"

    # R-005: risk appetite floor (2.0 for ISA products)
    # TAX_YEAR ISA prompt has no risk_appetite criterion (SEG-INV-003 design)
    _risk_floor_products = {
        "STOCKS_SHARES_ISA", "STOCKS_SHARES_ISA_REGULAR",
        "STOCKS_SHARES_ISA_REGULAR_ADDITION", "ISA_AND_GIA_LUMP_SUM",
    }
    min_risk = 2.0 if pt in _risk_floor_products else 1.0
    r["R-005"] = "PASS" if risk >= min_risk else "FAIL"

    # R-006: investment experience (ISA products need >= 1, except INV-002/INV-003/INV-006)
    inv_exp_required = {
        "STOCKS_SHARES_ISA": 1, "ISA_AND_GIA_LUMP_SUM": 1,
    }
    req_exp = inv_exp_required.get(pt, 0)
    r["R-006"] = "FAIL" if exp < req_exp else "PASS"

    # R-007: positive surplus (or zero cost)
    cost_map = {
        "STOCKS_SHARES_ISA": 50, "STOCKS_SHARES_ISA_REGULAR": 25,
        "STOCKS_SHARES_ISA_TAX_YEAR": 50,
        "STOCKS_SHARES_ISA_REGULAR_ADDITION": 25,
    }
    cost = cost_map.get(pt, 0)
    r["R-007"] = "PASS" if surplus - cost >= 0 else "FAIL"

    r["R-008"] = "PASS"  # soft — logging only (MON-001)
    r["R-009"] = "GATE" if conf < 0.75 else "PASS"
    r["R-010"] = "PASS"  # all v2 products in-scope (DC-001 checked at design time)

    # R-011: Consumer Duty — vulnerable + high risk product → GATE
    # Must match _HIGH_RISK_PRODUCT_TYPES in zone3/suggestion_engine.py exactly
    high_risk = {
        "STOCKS_SHARES_ISA", "ISA_AND_GIA_LUMP_SUM",
        "STOCKS_SHARES_ISA_REGULAR",   # capital at risk — Consumer Duty review required
        "DC_PENSION_FUND_SWITCH",       # lifecycle allocation — material pension decision
        "ANNUITY_FEATURES_REFERRAL", "DC_PENSION_SMALL_POT_ACCESS",
        "DECUMULATION_PATHWAY_DIRECTION", "DRAWDOWN_REVIEW_PROMPT",
    }
    r["R-011"] = "GATE" if (vuln and pt in high_risk) else "PASS"
    r["R-012"] = "PASS"

    # Gate disposition
    hard_fail  = any(r[k] == "FAIL" for k in
                     ["R-001","R-002","R-004","R-005","R-006","R-007","R-010","R-012"])
    gate_block = any(r[k] == "GATE" for k in ["R-003","R-009","R-011"])

    if hard_fail:
        gate = "SUPPRESS"
    elif gate_block:
        gate = "HUMAN_REVIEW"
    else:
        gate = "EMIT"

    return r, gate


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 1 — Consumer Profiles (3,000 records)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating consumer profiles...")

N = 3000
records = []

for i in range(N):
    seg_id = SEGMENT_IDS[i % len(SEGMENT_IDS)]
    p      = SEG_PROFILES[seg_id]

    # Inject edge cases: ~9% low-confidence, ~14% have a rule violation for SUPPRESS
    low_conf      = (i % 11 == 0)
    vuln_inject   = (i % 19 == 0)   # ~5% vulnerable (never EMIT)

    row = {
        "record_id":    uid(),
        "party_ref":    f"PARTY-{i+1:05d}",
        "session_id":   uid(),
        "situation_id": p["situation_id"],
        "segment_id":   seg_id,
        "suggestion_id": p["suggestion_id"],
        "product_type": p["product_type"],
        # Core trait fields (char_id columns)
        "CHAR_P1A_I1_age_band":         _get(p, "age_band"),
        "CHAR_P1B_I1_vulnerability":    vuln_inject or _get(p, "vulnerability"),
        "CHAR_P1C_I1_employment":       _get(p, "employment_status"),
        "CHAR_F2A_I1_monthly_surplus":  _get(p, "monthly_surplus"),
        "CHAR_F2B_I1_savings_balance":  _get(p, "savings_balance"),
        "CHAR_F2I_I1_has_investment":   _get(p, "has_investment", False),
        "CHAR_F2G_I1_high_cost_debt":   _get(p, "has_high_cost_debt", False),
        "CHAR_F2H_I1_financial_hardship": _get(p, "has_hardship", False),
        "CHAR_F2J_I1_isa_sub_this_year":  _get(p, "isa_sub_this_year", None),
        "CHAR_F2K_I1_within_90d_tax_yr":  _get(p, "within_90d_tax_year", None),
        "CHAR_F2L_I1_account_tenure_mo":  _get(p, "account_tenure_months"),
        "CHAR_F2M_I1_lump_sum":           _get(p, "lump_sum_amount", None),
        "CHAR_F2N_I1_months_inactive":    _get(p, "months_inactive", None),
        "CHAR_F2O_I1_regular_saving":     _get(p, "regular_saving_amount", None),
        "CHAR_F2P_I1_consec_months":      _get(p, "consecutive_saving_months", None),
        "CHAR_F2Q_I1_days_to_maturity":   _get(p, "days_to_maturity", None),
        "CHAR_F2R_I1_reinvest_instr":     _get(p, "has_reinvestment_instruction", None),
        "CHAR_P2A_I1_contrib_pct":        _get(p, "pension_contribution_pct", None),
        "CHAR_P2B_I1_active_dc":          _get(p, "is_active_dc_member", None),
        "CHAR_P2C_I1_shortfall_flag":     _get(p, "has_projected_shortfall", None),
        "CHAR_P2D_I1_yrs_to_ret":         _get(p, "years_to_retirement", None),
        "CHAR_P2E_I1_default_fund":       _get(p, "in_100pct_default_fund", None),
        "CHAR_P2F_I1_no_selection":       _get(p, "no_active_fund_selection", None),
        "CHAR_P2G_I1_recent_engage":      _get(p, "has_recent_engagement", None),
        "CHAR_P2H_I1_life_event":         _get(p, "life_event_signal", None),
        "CHAR_P2I_I1_access_plan":        _get(p, "has_retirement_access_plan", None),
        "CHAR_P2J_I1_in_drawdown":        _get(p, "already_in_drawdown", None),
        "CHAR_P2K_I1_db_transfer":        _get(p, "has_db_transfer", None),
        "CHAR_P2L_I1_pot_value":          _get(p, "pension_pot_value", None),
        "CHAR_P2M_I1_annuity_interest":   _get(p, "expressed_annuity_interest", None),
        "CHAR_P2N_I1_months_review":      _get(p, "months_since_review", None),
        "CHAR_P2O_I1_cash_drag":          _get(p, "has_cash_drag_flag", None),
        "CHAR_P2P_I1_recent_adviser":     _get(p, "has_recent_adviser", None),
        "CHAR_B3A_I1_risk_appetite":      _get(p, "risk_appetite_score"),
        "CHAR_B3B_I1_invest_exp":         _get(p, "investment_experience"),
        "CHAR_B3C_I1_channel":            _get(p, "channel"),
        # ML metadata
        "ml_confidence":    round(norm(0.50 if low_conf else 0.87, 0.06, 0.40, 0.99), 3),
        "ml_model_version": "2.0.0",
        "gap_fill_turns":   int(norm(4, 1.5, 1, 10)),
        "graph_completeness": round(norm(0.94, 0.04, 0.85, 1.0), 3),
        "fca_ref": f"PS25/22-§{p['situation_id']}",
    }

    # Evaluate rules and gate
    rules, gate = evaluate_rules(row)
    for r_id, outcome in rules.items():
        row[f"rule_{r_id.replace('-','_')}_outcome"] = outcome
    row["gate_disposition"] = gate

    records.append(row)

df_profiles = pd.DataFrame(records)

print(f"  Total: {len(df_profiles)}")
print(f"  EMIT:         {(df_profiles.gate_disposition=='EMIT').sum()}")
print(f"  HUMAN_REVIEW: {(df_profiles.gate_disposition=='HUMAN_REVIEW').sum()}")
print(f"  SUPPRESS:     {(df_profiles.gate_disposition=='SUPPRESS').sum()}")
print(f"  Segments: {df_profiles.segment_id.nunique()} unique")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 2 — ML Training Data (3,000 records)
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating ML training data...")

# Import inside to avoid path issues
import sys
sys.path.insert(0, OUT)
from ts_agent.ml.predictor import ALL_FEATURE_COLUMNS

ml_rows = []
for i in range(3000):
    seg_id = SEGMENT_IDS[i % len(SEGMENT_IDS)]
    p      = SEG_PROFILES[seg_id]
    turn   = int(norm(3, 2, 0, 8))

    def _v(key, default=None):
        val = p.get(key, default)
        v = _call(val)
        if v is None:
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    row = {
        # Numeric features
        "age_band":                 _v("age_band"),
        "savings_balance":          _v("savings_balance"),
        "monthly_surplus":          _v("monthly_surplus"),
        "risk_appetite_score":      _v("risk_appetite_score"),
        "investment_experience":    _v("investment_experience"),
        "account_tenure_months":    _v("account_tenure_months"),
        "lump_sum_amount":          _v("lump_sum_amount"),
        "regular_saving_amount":    _v("regular_saving_amount"),
        "pension_contribution_pct": _v("pension_contribution_pct"),
        "years_to_retirement":      _v("years_to_retirement"),
        "pension_pot_value":        _v("pension_pot_value"),
        "months_since_review":      _v("months_since_review"),
        "known_trait_count":        float(min(6 + turn, 14)),
        "conversation_turn":        float(turn),
        # Categorical features
        "employment_status":        _v("employment_status"),
        "channel":                  _v("channel"),
        # Labels
        "segment_id":   seg_id,
        "situation_id": p["situation_id"],
        "suggestion_id": p["suggestion_id"],
        # Metadata
        "record_id":    uid(),
        "party_ref":    f"ML-{i+1:05d}",
        "data_split":   "TRAIN" if i < 2400 else ("VAL" if i < 2700 else "TEST"),
    }

    # Inject NaN for early turns (realistic partial-knowledge simulation)
    if turn < 3:
        for feat in ["risk_appetite_score", "pension_contribution_pct",
                     "years_to_retirement", "pension_pot_value"]:
            if random.random() < 0.40:
                row[feat] = float("nan")
    if turn < 5:
        for feat in ["savings_balance", "monthly_surplus", "lump_sum_amount",
                     "regular_saving_amount", "months_since_review"]:
            if random.random() < 0.25:
                row[feat] = float("nan")

    # Features not applicable to non-pension segments get NaN
    if not seg_id.startswith("SEG-PEN") and not seg_id.startswith("SEG-DEC"):
        for feat in ["pension_contribution_pct", "years_to_retirement",
                     "pension_pot_value", "months_since_review"]:
            row[feat] = float("nan")
    if not seg_id.startswith("SEG-INV"):
        for feat in ["lump_sum_amount", "regular_saving_amount", "account_tenure_months"]:
            if seg_id != "SEG-SD-001":
                row[feat] = float("nan")

    ml_rows.append(row)

df_ml = pd.DataFrame(ml_rows)

feature_cols = ALL_FEATURE_COLUMNS
label_cols   = ["segment_id", "situation_id", "suggestion_id"]
meta_cols    = ["record_id", "party_ref", "data_split"]
df_ml = df_ml[feature_cols + label_cols + meta_cols]

nan_frac = df_ml[feature_cols].isna().mean().mean()
print(f"  Total: {len(df_ml)}")
print(f"  Train/Val/Test: {(df_ml.data_split=='TRAIN').sum()}/{(df_ml.data_split=='VAL').sum()}/{(df_ml.data_split=='TEST').sum()}")
print(f"  NaN fraction: {nan_frac:.1%}")
print(f"  Segments: {df_ml.segment_id.nunique()} unique")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 3 — Sample Consumer Prompts (v2)
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating sample prompts...")

PROMPTS = [
    # ── Intent utterances (one per intent_id) ─────────────────────────────────
    {"prompt_id":"INT-INV-001","category":"INTENT","char_id":"","intent_id":"INTENT-INVEST-CASH","situation_id":"SIT-INV-001",
     "consumer_utterance":"I have some cash savings I'd like to invest",
     "paraphrase_1":"I want to grow my savings rather than leaving them in cash",
     "paraphrase_2":"What can I do with my savings to beat inflation?",
     "paraphrase_3":"I'm thinking about investing some of my money",
     "expected_coerced_value":"","notes":"Firm-initiative trigger for SIT-INV-001 cash drag situation"},
    {"prompt_id":"INT-INV-002","category":"INTENT","char_id":"","intent_id":"INTENT-FIRST-INVEST","situation_id":"SIT-INV-002",
     "consumer_utterance":"I've never invested before and want to start",
     "paraphrase_1":"I'm new to investing and don't really know where to begin",
     "paraphrase_2":"How do I start investing for the first time?",
     "paraphrase_3":"I want to try investing but I'm a bit nervous about it",
     "expected_coerced_value":"","notes":"Consumer-request trigger for SIT-INV-002"},
    {"prompt_id":"INT-INV-003","category":"INTENT","char_id":"","intent_id":"INTENT-ISA-ALLOWANCE","situation_id":"SIT-INV-003",
     "consumer_utterance":"I want to use my ISA allowance before April",
     "paraphrase_1":"I haven't used my ISA this year and it's nearly April",
     "paraphrase_2":"Can I still put money in an ISA before the tax year ends?",
     "paraphrase_3":"I want to make the most of this year's ISA allowance",
     "expected_coerced_value":"","notes":"Tax year end trigger for SIT-INV-003"},
    {"prompt_id":"INT-INV-004","category":"INTENT","char_id":"","intent_id":"INTENT-LUMP-SUM","situation_id":"SIT-INV-004",
     "consumer_utterance":"I've just received some money and I'm not sure what to do with it",
     "paraphrase_1":"I got a windfall and want some ideas on what to do",
     "paraphrase_2":"I have a lump sum I need to invest",
     "paraphrase_3":"I recently inherited some money and would like some direction",
     "expected_coerced_value":"","notes":"Lump sum capital event trigger for SIT-INV-004"},
    {"prompt_id":"INT-INV-005","category":"INTENT","char_id":"","intent_id":"INTENT-REVIEW-INVESTMENT","situation_id":"SIT-INV-005",
     "consumer_utterance":"I have an old investment account I haven't looked at in ages",
     "paraphrase_1":"I think I have an ISA somewhere that I forgot about",
     "paraphrase_2":"Can you show me my investment account? I haven't logged in for a while",
     "paraphrase_3":"I want to review my dormant investment",
     "expected_coerced_value":"","notes":"Firm-initiative dormant account trigger for SIT-INV-005"},
    {"prompt_id":"INT-INV-006","category":"INTENT","char_id":"","intent_id":"INTENT-REGULAR-INVEST","situation_id":"SIT-INV-006",
     "consumer_utterance":"I save a bit every month and want to do more with it",
     "paraphrase_1":"I've been saving regularly and want to invest instead",
     "paraphrase_2":"Can I turn my regular savings into investments?",
     "paraphrase_3":"I put £100 aside every month and want to make it work harder",
     "expected_coerced_value":"","notes":"Regular saver investment upgrade trigger for SIT-INV-006"},
    {"prompt_id":"INT-SD-001","category":"INTENT","char_id":"","intent_id":"INTENT-DEPOSIT-MATURITY","situation_id":"SIT-SD-001",
     "consumer_utterance":"My fixed-rate savings bond is about to mature",
     "paraphrase_1":"My deposit account matures next month",
     "paraphrase_2":"My structured deposit is coming to an end, what are my options?",
     "paraphrase_3":"My fixed term is ending and I need to decide what to do",
     "expected_coerced_value":"","notes":"Maturing deposit trigger for SIT-SD-001"},
    {"prompt_id":"INT-PEN-001","category":"INTENT","char_id":"","intent_id":"INTENT-PENSION-CONTRIBUTIONS","situation_id":"SIT-PEN-001",
     "consumer_utterance":"I'm worried I'm not saving enough for retirement",
     "paraphrase_1":"I don't think my pension contributions are enough",
     "paraphrase_2":"How can I save more into my pension?",
     "paraphrase_3":"My pension pot seems too small for what I'll need",
     "expected_coerced_value":"","notes":"Under-saving trigger for SIT-PEN-001"},
    {"prompt_id":"INT-PEN-002","category":"INTENT","char_id":"","intent_id":"INTENT-PENSION-FUNDS","situation_id":"SIT-PEN-002",
     "consumer_utterance":"I want to choose where my pension money is invested",
     "paraphrase_1":"Can I pick my own pension funds?",
     "paraphrase_2":"I don't know what funds my pension is in",
     "paraphrase_3":"I want to move my pension out of the default fund",
     "expected_coerced_value":"","notes":"Fund disengagement trigger for SIT-PEN-002"},
    {"prompt_id":"INT-PEN-003","category":"INTENT","char_id":"","intent_id":"INTENT-LIFE-EVENT","situation_id":"SIT-PEN-003",
     "consumer_utterance":"I recently got a pay rise and want to save more for retirement",
     "paraphrase_1":"I've just finished paying off my car loan and have more money free",
     "paraphrase_2":"My mortgage is ending next month — I want to put more into my pension",
     "paraphrase_3":"Things have changed financially and I want to review my contributions",
     "expected_coerced_value":"","notes":"Life event trigger for SIT-PEN-003"},
    {"prompt_id":"INT-DEC-001","category":"INTENT","char_id":"","intent_id":"INTENT-RETIREMENT-PLAN","situation_id":"SIT-DEC-001",
     "consumer_utterance":"I'm approaching retirement and need to think about accessing my pension",
     "paraphrase_1":"I'm in my late fifties and haven't planned how to take my pension",
     "paraphrase_2":"I need help deciding how to draw down my pension",
     "paraphrase_3":"What are my options for taking money from my pension?",
     "expected_coerced_value":"","notes":"Pre-retirement planning trigger for SIT-DEC-001"},
    {"prompt_id":"INT-DEC-002","category":"INTENT","char_id":"","intent_id":"INTENT-TAKE-PENSION","situation_id":"SIT-DEC-002",
     "consumer_utterance":"I'm nearly at retirement age and want to start taking my pension",
     "paraphrase_1":"I retire in a few years and have a small pension pot",
     "paraphrase_2":"How do I take money from my small pension?",
     "paraphrase_3":"I need to start accessing my pension soon",
     "expected_coerced_value":"","notes":"Small pot access trigger for SIT-DEC-002"},
    {"prompt_id":"INT-DEC-003","category":"INTENT","char_id":"","intent_id":"INTENT-ANNUITY","situation_id":"SIT-DEC-003",
     "consumer_utterance":"I'm thinking about getting an annuity for a guaranteed income",
     "paraphrase_1":"Could an annuity be right for me?",
     "paraphrase_2":"I want a guaranteed income in retirement — is an annuity the answer?",
     "paraphrase_3":"I'd like to explore annuity options",
     "expected_coerced_value":"","notes":"Annuity enquiry trigger for SIT-DEC-003"},
    {"prompt_id":"INT-DEC-004","category":"INTENT","char_id":"","intent_id":"INTENT-DRAWDOWN-REVIEW","situation_id":"SIT-DEC-004",
     "consumer_utterance":"I'm already in drawdown but haven't reviewed my pension in a while",
     "paraphrase_1":"I draw from my pension but haven't checked my strategy for ages",
     "paraphrase_2":"My drawdown pension needs a review",
     "paraphrase_3":"I've been in drawdown for two years and have never changed anything",
     "expected_coerced_value":"","notes":"Drawdown review trigger for SIT-DEC-004"},
    # ── Trait gap-fill prompts ─────────────────────────────────────────────────
    {"prompt_id":"TRAIT-P1A-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-P1A-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I'm 38","paraphrase_1":"Thirty-eight","paraphrase_2":"38 years old","paraphrase_3":"I'm in my late thirties",
     "expected_coerced_value":"2","notes":"age_band 2 (30–39). Band 2 qualifies for SEG-INV-002 (<=3)"},
    {"prompt_id":"TRAIT-P1A-002","category":"TRAIT_GAP_FILL","char_id":"CHAR-P1A-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I'm 55","paraphrase_1":"Fifty-five","paraphrase_2":"Just turned 55","paraphrase_3":"55",
     "expected_coerced_value":"4","notes":"age_band 4 (55–64). Required for DEC-001/003 (age >= 4)"},
    {"prompt_id":"TRAIT-P1B-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-P1B-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"No I don't have any vulnerability flags","paraphrase_1":"Not as far as I know","paraphrase_2":"No","paraphrase_3":"Nothing noted on my account",
     "expected_coerced_value":"False","notes":"vulnerability=False. No excluding characteristic triggered."},
    {"prompt_id":"TRAIT-P1B-002","category":"TRAIT_GAP_FILL","char_id":"CHAR-P1B-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"Yes I'm flagged as a vulnerable customer","paraphrase_1":"Yes I am","paraphrase_2":"Yes there is a flag","paraphrase_3":"Yes",
     "expected_coerced_value":"True","notes":"vulnerability=True → R-003 GATE → HUMAN_REVIEW. FCA FG21/1."},
    {"prompt_id":"TRAIT-F2B-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-F2B-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I have about £12,000 in savings","paraphrase_1":"Roughly twelve thousand","paraphrase_2":"Around 12k","paraphrase_3":"£12,000",
     "expected_coerced_value":"12000","notes":"savings_balance=12000. >= £10k qualifies for SEG-INV-001"},
    {"prompt_id":"TRAIT-F2B-002","category":"TRAIT_GAP_FILL","char_id":"CHAR-F2B-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I only have £800 saved","paraphrase_1":"About eight hundred pounds","paraphrase_2":"Not much, around 800","paraphrase_3":"£800",
     "expected_coerced_value":"800","notes":"savings_balance=800. >= £500 qualifies for SEG-INV-002"},
    {"prompt_id":"TRAIT-F2I-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-F2I-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"No I don't have any investments","paraphrase_1":"Not got any investment accounts","paraphrase_2":"No","paraphrase_3":"Never had one",
     "expected_coerced_value":"False","notes":"has_investment=False. IC for INV-001/002/003/004/006."},
    {"prompt_id":"TRAIT-F2I-002","category":"TRAIT_GAP_FILL","char_id":"CHAR-F2I-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"Yes I have a Stocks and Shares ISA","paraphrase_1":"Yes I've got one","paraphrase_2":"Yes","paraphrase_3":"I hold an investment ISA already",
     "expected_coerced_value":"True","notes":"has_investment=True. IC for INV-005 (holds dormant investment)."},
    {"prompt_id":"TRAIT-F2L-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-F2L-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I've been with you for about 2 years","paraphrase_1":"Two years or so","paraphrase_2":"Around 24 months","paraphrase_3":"About two years",
     "expected_coerced_value":"24","notes":"account_tenure_months=24. >= 12 qualifies for SEG-INV-001"},
    {"prompt_id":"TRAIT-F2M-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-F2M-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I received about £20,000","paraphrase_1":"Twenty thousand","paraphrase_2":"£20k","paraphrase_3":"Around 20,000 pounds",
     "expected_coerced_value":"20000","notes":"lump_sum=20000. £5k-£75k qualifies for SEG-INV-004"},
    {"prompt_id":"TRAIT-F2M-002","category":"TRAIT_GAP_FILL","char_id":"CHAR-F2M-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"It's about £80,000","paraphrase_1":"Eighty thousand","paraphrase_2":"Around 80k","paraphrase_3":"£80,000",
     "expected_coerced_value":"80000","notes":"lump_sum=80000. > £75k → EC-INV-004-01 → SUPPRESS (exit TS)"},
    {"prompt_id":"TRAIT-F2Q-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-F2Q-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"It matures in about 45 days","paraphrase_1":"Six weeks or so","paraphrase_2":"45 days","paraphrase_3":"About a month and a half",
     "expected_coerced_value":"45","notes":"days_to_maturity=45. <= 60 qualifies for SEG-SD-001"},
    {"prompt_id":"TRAIT-P2A-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-P2A-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I think I pay in about 8 percent","paraphrase_1":"Eight percent","paraphrase_2":"8%","paraphrase_3":"Around 8 per cent",
     "expected_coerced_value":"8","notes":"pension_contribution_pct=8. <= 8% AE minimum qualifies for SEG-PEN-001"},
    {"prompt_id":"TRAIT-P2A-002","category":"TRAIT_GAP_FILL","char_id":"CHAR-P2A-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"My total contribution is about 15 percent","paraphrase_1":"Fifteen percent","paraphrase_2":"15%","paraphrase_3":"About 15 per cent",
     "expected_coerced_value":"15","notes":"pension_contribution_pct=15. > 12% fails SEG-PEN-003 (IC: <= 12%)"},
    {"prompt_id":"TRAIT-P2B-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-P2B-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"Yes I'm in a workplace pension scheme","paraphrase_1":"Yes I have a work pension","paraphrase_2":"Yes","paraphrase_3":"Yes I'm a member",
     "expected_coerced_value":"True","notes":"active_dc_member=True. IC for SEG-PEN-001/002/003, SEG-DEC-001/002/003"},
    {"prompt_id":"TRAIT-P2D-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-P2D-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"About 25 years until I retire","paraphrase_1":"Twenty-five years","paraphrase_2":"25 years","paraphrase_3":"A long way off",
     "expected_coerced_value":"25","notes":"years_to_retirement=25. > 5 so EC-PEN-001-03 not triggered"},
    {"prompt_id":"TRAIT-P2D-002","category":"TRAIT_GAP_FILL","char_id":"CHAR-P2D-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I plan to retire in 3 years","paraphrase_1":"Three years","paraphrase_2":"About 3 years","paraphrase_3":"I'm retiring soon",
     "expected_coerced_value":"3","notes":"years_to_retirement=3. <= 5 triggers EC-PEN-001-03 → redirect to DEC segment"},
    {"prompt_id":"TRAIT-P2L-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-P2L-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"My pension pot is about £18,000","paraphrase_1":"Eighteen thousand","paraphrase_2":"Around £18k","paraphrase_3":"£18,000",
     "expected_coerced_value":"18000","notes":"pension_pot_value=18000. £5k-£30k qualifies for SEG-DEC-002"},
    {"prompt_id":"TRAIT-P2L-002","category":"TRAIT_GAP_FILL","char_id":"CHAR-P2L-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I have around £60,000 in my pension","paraphrase_1":"About sixty thousand","paraphrase_2":"Roughly £60k","paraphrase_3":"60,000 pounds",
     "expected_coerced_value":"60000","notes":"pension_pot_value=60000. > £30k ceiling → EC-DEC-002-02 → SUPPRESS"},
    {"prompt_id":"TRAIT-P2M-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-P2M-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"Yes I'm interested in annuities","paraphrase_1":"Yes I've been thinking about getting an annuity","paraphrase_2":"Yes","paraphrase_3":"I want to know more about annuities",
     "expected_coerced_value":"True","notes":"expressed_annuity_interest=True. IC for SEG-DEC-003"},
    {"prompt_id":"TRAIT-P2N-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-P2N-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"It's been about 20 months since I last checked","paraphrase_1":"Around 20 months","paraphrase_2":"Nearly two years","paraphrase_3":"20 months",
     "expected_coerced_value":"20","notes":"months_since_review=20. >= 18 qualifies for SEG-DEC-004"},
    {"prompt_id":"TRAIT-B3A-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-B3A-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I'd say about a 3 out of 5","paraphrase_1":"Moderate risk, maybe a 3","paraphrase_2":"3","paraphrase_3":"Middle of the road",
     "expected_coerced_value":"3","notes":"risk_appetite=3.0. >= 2.0 minimum for ISA products (R-005 PASS)"},
    {"prompt_id":"TRAIT-B3A-002","category":"TRAIT_GAP_FILL","char_id":"CHAR-B3A-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I'm very cautious, maybe a 1","paraphrase_1":"Very low, about 1","paraphrase_2":"1 out of 5","paraphrase_3":"1",
     "expected_coerced_value":"1","notes":"risk_appetite=1.0. < 2.0 → R-005 FAIL for ISA products"},
    {"prompt_id":"TRAIT-B3B-001","category":"TRAIT_GAP_FILL","char_id":"CHAR-B3B-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I've never invested before so no experience","paraphrase_1":"Zero experience","paraphrase_2":"None at all","paraphrase_3":"0",
     "expected_coerced_value":"0","notes":"investment_experience=0. Qualifies for INV-002 (entry-level); fails R-006 for advanced ISA"},
    {"prompt_id":"TRAIT-B3B-002","category":"TRAIT_GAP_FILL","char_id":"CHAR-B3B-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"Some experience, I've had a stocks and shares ISA before","paraphrase_1":"Basic experience","paraphrase_2":"I've invested a bit","paraphrase_3":"1",
     "expected_coerced_value":"1","notes":"investment_experience=1. R-006 PASS for ISA products"},
    # ── Edge cases ─────────────────────────────────────────────────────────────
    {"prompt_id":"EDGE-001","category":"EDGE_CASE","char_id":"CHAR-F2B-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"I'm not sure exactly how much I have","paraphrase_1":"Hard to say","paraphrase_2":"I'd need to check","paraphrase_3":"Not sure",
     "expected_coerced_value":"unknown","notes":"Consumer declines. System stores 'unknown'. Node stays MISSING."},
    {"prompt_id":"EDGE-002","category":"EDGE_CASE","char_id":"CHAR-B3A-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"What do you mean by risk appetite?","paraphrase_1":"I don't understand that question","paraphrase_2":"Could you explain?","paraphrase_3":"I'm not sure what that means",
     "expected_coerced_value":"","notes":"Agent must clarify before re-asking. No coercion."},
    {"prompt_id":"EDGE-003","category":"EDGE_CASE","char_id":"CHAR-P2L-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"Somewhere between £20k and £40k I think","paraphrase_1":"Around thirty thousand","paraphrase_2":"I'd guess about 30k","paraphrase_3":"In that range",
     "expected_coerced_value":"30000","notes":"Approximate range — agent takes midpoint estimate."},
    {"prompt_id":"EDGE-004","category":"EDGE_CASE","char_id":"","intent_id":"","situation_id":"",
     "consumer_utterance":"I need some help with my money but I'm not sure what","paraphrase_1":"Not sure where to start","paraphrase_2":"General financial direction please","paraphrase_3":"I just want some guidance",
     "expected_coerced_value":"","notes":"Ambiguous intent. Agent asks clarifying question before situation routing."},
    {"prompt_id":"EDGE-005","category":"EDGE_CASE","char_id":"CHAR-P2K-I1","intent_id":"","situation_id":"",
     "consumer_utterance":"Yes I have a final salary pension from a previous employer","paraphrase_1":"I have a defined benefit pension","paraphrase_2":"Yes, I'm in a DB scheme","paraphrase_3":"Yes I have a final salary scheme",
     "expected_coerced_value":"True","notes":"DB transfer applicable → EC-DEC-001-02 → SUPPRESS (mandatory specialist referral). COBS 9/9A."},
]

df_prompts = pd.DataFrame(PROMPTS)
print(f"  Total prompts: {len(df_prompts)}")
print(f"  Categories: {df_prompts.category.value_counts().to_dict()}")


# ──────────────────────────────────────────────────────────────────────────────
# Write CSV outputs
# ──────────────────────────────────────────────────────────────────────────────
csv1 = os.path.join(OUT, "ts_agent_consumer_profiles.csv")
csv2 = os.path.join(OUT, "ts_agent_ml_training.csv")
csv3 = os.path.join(OUT, "ts_agent_sample_prompts.csv")

df_profiles.to_csv(csv1, index=False)
df_ml.to_csv(csv2, index=False)
df_prompts.to_csv(csv3, index=False)

print(f"\nOutputs written to {OUT}/")
print(f"  {os.path.basename(csv1)}: {len(df_profiles)} rows, {df_profiles.shape[1]} cols ({os.path.getsize(csv1)//1024} KB)")
print(f"  {os.path.basename(csv2)}: {len(df_ml)} rows, {df_ml.shape[1]} cols ({os.path.getsize(csv2)//1024} KB)")
print(f"  {os.path.basename(csv3)}: {len(df_prompts)} rows, {df_prompts.shape[1]} cols ({os.path.getsize(csv3)//1024} KB)")
print(f"\nTotal records generated: {len(df_profiles) + len(df_ml) + len(df_prompts)}")
