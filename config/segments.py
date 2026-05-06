"""
ts_agent.config.segments
=========================
FCA PS25/22 Targeted Support Ontology Catalogue — v2.0

Live: 6 April 2026  |  Regulation: PS25/22 confirmed 26 Feb 2026
Rulebook: COBS 9B (new chapter), PRIN 2A (Consumer Duty)

This module is the single source of truth for:
  - 4 domains (Retail Investments, Structured Deposits,
    DC Pension Accumulation, DC Pension Decumulation)
  - 13 situations  (SIT-INV-001–006, SIT-SD-001,
                    SIT-PEN-001–003, SIT-DEC-001–004)
  - 13 segments    (SEG-INV-001–006, SEG-SD-001,
                    SEG-PEN-001–003, SEG-DEC-001–004)
  - 13 suggestions (SUG-INV-001–006, SUG-SD-001,
                    SUG-PEN-001–003, SUG-DEC-001–004)
  - 28 compliance checks across 5 phases (fca_ts_compliance_checks.yml)

OUT-OF-SCOPE PRODUCTS (hard exclusions per PS25/22 Ch.3):
  Mortgages, pure protection insurance, debt/credit products,
  DB pension transfers, pension consolidation, specific annuity products,
  non-mass-market investments, restricted mass-market investments,
  qualifying cryptoassets, leveraged products where loss > investment.

Codex review notes
------------------
- Every ID cross-references the upstream YAML files exactly.
- All FCA citations use the format PS25/22-§<section> or COBS<rule>.
- RuleType is imported for backward compatibility with existing tests;
  new code uses CheckSeverity from domain.models.
- SEGMENT_TO_SUGGESTIONS and SITUATION_INTENTS are derived dicts for
  fast lookups without duplicating data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ts_agent.domain.models import (
    AlternativeSupport,
    AlternativeSupportType,
    CheckPhase,
    CheckSeverity,
    ExcludingCharacteristic,
    RuleType,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TraitCriterion:
    """One including characteristic condition for a segment."""
    char_id:  str
    op:       str   # "==", "!=", ">=", "<=", ">", "<", "in"
    value:    Any


@dataclass(frozen=True)
class SituationDef:
    """PS25/22 pre-defined situation (COBS 9B.3)."""
    situation_id:    str
    domain:          str
    label:           str
    trigger_type:    str      # "CONSUMER_REQUEST" | "FIRM_INITIATIVE" | "BOTH"
    segment_ids:     tuple[str, ...]
    fca_ref:         str = "PS25/22-COBS9B.3"
    description:     str = ""
    intent_ids:      tuple[str, ...] = field(default_factory=tuple)
    out_of_scope_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SegmentDef:
    """
    PS25/22 pre-defined consumer segment (COBS 9B.4).

    Every segment has at least one including AND one excluding characteristic
    per PS25/22 para 3.22 / COBS 9B.4.
    """
    segment_id:               str
    situation_id:             str
    label:                    str
    description:              str
    criteria:                 tuple[TraitCriterion, ...]   # including characteristics
    excluding:                tuple[ExcludingCharacteristic, ...]
    fca_ref:                  str = "PS25/22-COBS9B.4"
    characteristic_descriptions: tuple[str, ...] = field(default_factory=tuple)
    risk_questionnaire_permitted: bool = True
    data_accuracy_stale_days:     int  = 30


@dataclass(frozen=True)
class SuggestionDef:
    """
    PS25/22 ready-made suggestion (COBS 9B.5).

    ``pre_delivery_checks`` and ``delivery_checks`` reference check IDs from
    fca_ts_compliance_checks.yml.  The engine evaluates them in order;
    the first HARD_BLOCK or BOUNDARY_EXIT terminates the evaluation.
    """
    suggestion_id:       str
    segment_ids:         tuple[str, ...]
    product_name:        str
    product_type:        str
    domain:              str
    fca_ref:             str
    cobs_9b_suitability: str = ""
    description:         str = ""

    # Phase-2 (pre-delivery) checks from compliance YAML
    pre_delivery_checks: tuple[str, ...] = (
        "PDC-001", "PDC-002", "PDC-003", "PDC-004",
    )
    # Phase-3 (delivery) checks from compliance YAML
    delivery_checks:     tuple[str, ...] = (
        "DEL-001", "DEL-002", "DEL-003", "DEL-004",
        "DEL-005", "DEL-006", "DEL-012", "DEL-013",
    )
    # Absolute prohibitions — evaluated before any other checks
    hard_prohibitions:   tuple[str, ...] = field(default_factory=tuple)

    # Legacy fields — retained so existing engine code compiles
    eligibility_rules:  tuple[str, ...] = field(default_factory=tuple)
    suitability_rules:  tuple[str, ...] = field(default_factory=tuple)
    compliance_rules:   tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ComplianceCheckDef:
    """
    One compliance check from fca_ts_compliance_checks.yml.

    Mapped into the engine as a first-class object so the neurosymbolic
    3S engine can evaluate predicates symbolically.
    """
    check_id:          str
    phase:             CheckPhase
    severity:          CheckSeverity
    label:             str
    rule_text:         str
    regulatory_source: str
    failure_action:    str
    # Maps to GateDisposition contribution
    gate_on_fail:      str   # "SUPPRESS" | "HUMAN_REVIEW" | "AUDIT_ONLY"


# ──────────────────────────────────────────────────────────────────────────────
# Reusable excluding characteristics (shared across multiple segments)
# ──────────────────────────────────────────────────────────────────────────────

_EC_VULNERABILITY = ExcludingCharacteristic(
    char_id="CHAR-P1B-I1",
    label="Active vulnerability indicator",
    definition="Any active vulnerability flag on the consumer's record "
               "(bereavement, PoA, financial difficulty, health)",
    rationale="Standard segment route not appropriate without enhanced care. "
              "PS25/22 para 3.26; FCA FG21/1.",
    alternative_support=AlternativeSupport(
        support_type=AlternativeSupportType.SPECIALIST_JOURNEY,
        destination="Firm's vulnerable customer team; MoneyHelper",
        action="Route to specialist support with enhanced care obligations",
    ),
)

_EC_HIGH_COST_DEBT = ExcludingCharacteristic(
    char_id="CHAR-F2G-I1",
    label="Active high-cost debt",
    definition="Active overdraft > £1,000 or high-cost short-term credit "
               "product active (PS25/22 para 3.27; SEG-INV-001 exclusion)",
    rationale="Investing before clearing high-cost debt may not be suitable.",
    alternative_support=AlternativeSupport(
        support_type=AlternativeSupportType.SIGNPOST,
        destination="MoneyHelper debt guidance",
        action="Direct to MoneyHelper before re-assessing for investment TS",
    ),
)

_EC_HARDSHIP = ExcludingCharacteristic(
    char_id="CHAR-F2H-I1",
    label="Financial hardship or difficulty flag",
    definition="Active financial difficulty marker, debt management plan, "
               "or payment deferral",
    rationale="Increasing contributions may not be suitable under financial stress.",
    alternative_support=AlternativeSupport(
        support_type=AlternativeSupportType.SIGNPOST,
        destination="MoneyHelper debt and money guidance",
        action="Signpost to MoneyHelper before re-assessing",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# SITUATIONS — 13 pre-defined (PS25/22 COBS 9B.3)
# ──────────────────────────────────────────────────────────────────────────────

SITUATIONS: dict[str, SituationDef] = {

    # ── RETAIL INVESTMENTS (6 situations) ────────────────────────────────────

    "SIT-INV-001": SituationDef(
        situation_id="SIT-INV-001",
        domain="RETAIL_INVESTMENTS",
        label="Cash Drag — Under-invested Saver with Investible Assets",
        description="Consumer holds ≥ £10,000 in cash with no investment product. "
                    "Real-terms purchasing power at risk from inflation.",
        trigger_type="FIRM_INITIATIVE",
        segment_ids=("SEG-INV-001",),
        fca_ref="PS25/22-para1.2-COBS9B.3",
        intent_ids=("INTENT-INVEST-CASH", "INTENT-ISA-OPEN",),
    ),

    "SIT-INV-002": SituationDef(
        situation_id="SIT-INV-002",
        domain="RETAIL_INVESTMENTS",
        label="First-Time Investor — Knowledge Barrier",
        description="Consumer has never held an investment product. "
                    "Cites knowledge or confidence as primary barrier.",
        trigger_type="CONSUMER_REQUEST",
        segment_ids=("SEG-INV-002",),
        fca_ref="PS25/22-para2.2-COBS9B.3",
        intent_ids=("INTENT-FIRST-INVEST", "INTENT-HOW-TO-INVEST",),
    ),

    "SIT-INV-003": SituationDef(
        situation_id="SIT-INV-003",
        domain="RETAIL_INVESTMENTS",
        label="ISA Allowance Non-Utilisation — Tax Year End",
        description="Consumer has unused annual ISA allowance within 90 days "
                    "of 5 April with cash savings available.",
        trigger_type="FIRM_INITIATIVE",
        segment_ids=("SEG-INV-003",),
        fca_ref="PS25/22-para3.17-COBS9B.3",
        intent_ids=("INTENT-ISA-ALLOWANCE", "INTENT-TAX-YEAR-END",),
    ),

    "SIT-INV-004": SituationDef(
        situation_id="SIT-INV-004",
        domain="RETAIL_INVESTMENTS",
        label="Lump Sum Capital Event — Investment Direction",
        description="Consumer has received or expects a one-off capital sum "
                    "(£5,000–£75,000) and has not placed an investment instruction.",
        trigger_type="BOTH",
        segment_ids=("SEG-INV-004",),
        fca_ref="PS25/22-para3.15-COBS9B.3",
        intent_ids=("INTENT-LUMP-SUM", "INTENT-WINDFALL",),
        out_of_scope_flags=("LUMP_SUM_ABOVE_75K_EXIT_TS",),
    ),

    "SIT-INV-005": SituationDef(
        situation_id="SIT-INV-005",
        domain="RETAIL_INVESTMENTS",
        label="Dormant / Lapsed Investment Account — Re-engagement",
        description="Consumer holds an investment account inactive for ≥ 12 months.",
        trigger_type="FIRM_INITIATIVE",
        segment_ids=("SEG-INV-005",),
        fca_ref="PS25/22-para3.30-COBS9B.10",
        intent_ids=("INTENT-REVIEW-INVESTMENT",),
    ),

    "SIT-INV-006": SituationDef(
        situation_id="SIT-INV-006",
        domain="RETAIL_INVESTMENTS",
        label="Regular Saver Seeking Investment Upgrade",
        description="Consumer has regular cash savings habit (≥ 6 months, "
                    "≥ £50/month) with no investment counterpart.",
        trigger_type="FIRM_INITIATIVE",
        segment_ids=("SEG-INV-006",),
        fca_ref="PS25/22-COBS9B.3",
        intent_ids=("INTENT-REGULAR-INVEST", "INTENT-ISA-OPEN",),
    ),

    # ── STRUCTURED DEPOSITS (1 situation) ────────────────────────────────────

    "SIT-SD-001": SituationDef(
        situation_id="SIT-SD-001",
        domain="STRUCTURED_DEPOSITS",
        label="Maturing Fixed-Rate Deposit — Reinvestment Direction",
        description="Consumer holds a fixed-rate or structured deposit maturing "
                    "within 60 days with no reinvestment instruction placed.",
        trigger_type="FIRM_INITIATIVE",
        segment_ids=("SEG-SD-001",),
        fca_ref="PS25/22-Ch3-COBS9B.3",
        intent_ids=("INTENT-DEPOSIT-MATURITY", "INTENT-REINVEST",),
    ),

    # ── DC PENSION ACCUMULATION (3 situations) ────────────────────────────────

    "SIT-PEN-001": SituationDef(
        situation_id="SIT-PEN-001",
        domain="DC_PENSION_ACCUMULATION",
        label="Under-saving for Retirement — Below Adequacy Benchmark",
        description="Working-age DC pension member whose projected retirement "
                    "income is materially below adequacy benchmark, contributing "
                    "at or below auto-enrolment minimum.",
        trigger_type="FIRM_INITIATIVE",
        segment_ids=("SEG-PEN-001",),
        fca_ref="PS25/22-para1.2-COBS9B.3",
        intent_ids=("INTENT-PENSION-CONTRIBUTIONS", "INTENT-SAVE-MORE",),
        out_of_scope_flags=("PENSION_CONSOLIDATION",),
    ),

    "SIT-PEN-002": SituationDef(
        situation_id="SIT-PEN-002",
        domain="DC_PENSION_ACCUMULATION",
        label="Default Fund Disengagement — Lifecycle Mismatch",
        description="DC pension member invested 100% in the default fund with "
                    "no active fund selection, where age-band suggests an "
                    "actively selected fund may be more appropriate.",
        trigger_type="FIRM_INITIATIVE",
        segment_ids=("SEG-PEN-002",),
        fca_ref="PS25/22-Ch3-COBS9B.3-PROD",
        intent_ids=("INTENT-PENSION-FUNDS", "INTENT-FUND-SWITCH",),
        out_of_scope_flags=("PENSION_CONSOLIDATION",),
    ),

    "SIT-PEN-003": SituationDef(
        situation_id="SIT-PEN-003",
        domain="DC_PENSION_ACCUMULATION",
        label="Life Event — Contribution Review Opportunity",
        description="Consumer has experienced a life event (salary increase, "
                    "debt clearance, mortgage end) creating headroom to review "
                    "pension contributions.",
        trigger_type="FIRM_INITIATIVE",
        segment_ids=("SEG-PEN-003",),
        fca_ref="PS25/22-para3.14-COBS9B.3",
        intent_ids=("INTENT-LIFE-EVENT", "INTENT-PENSION-REVIEW",),
        out_of_scope_flags=("PENSION_CONSOLIDATION",),
    ),

    # ── DC PENSION DECUMULATION (4 situations) ────────────────────────────────

    "SIT-DEC-001": SituationDef(
        situation_id="SIT-DEC-001",
        domain="DC_PENSION_DECUMULATION",
        label="Approaching Retirement — No Decumulation Plan (45–65)",
        description="DC pension holder aged 45–65 with no retirement access plan. "
                    "75% of DC holders 45+ fall into this group (FLS 2024).",
        trigger_type="FIRM_INITIATIVE",
        segment_ids=("SEG-DEC-001",),
        fca_ref="PS25/22-para2.2-COBS9B.3-COBS19",
        intent_ids=("INTENT-RETIREMENT-PLAN", "INTENT-PENSION-ACCESS",),
        out_of_scope_flags=("PENSION_CONSOLIDATION", "SPECIFIC_ANNUITY_PRODUCT"),
    ),

    "SIT-DEC-002": SituationDef(
        situation_id="SIT-DEC-002",
        domain="DC_PENSION_DECUMULATION",
        label="Small Pot — Imminent Retirement Decision",
        description="Consumer holds DC pot £5,000–£30,000, within 5 years of "
                    "retirement, no access method nominated.",
        trigger_type="BOTH",
        segment_ids=("SEG-DEC-002",),
        fca_ref="PS25/22-Ch3-COBS9B.3-COBS19",
        intent_ids=("INTENT-TAKE-PENSION", "INTENT-SMALL-POT",),
        out_of_scope_flags=("PENSION_CONSOLIDATION", "SPECIFIC_ANNUITY_PRODUCT"),
    ),

    "SIT-DEC-003": SituationDef(
        situation_id="SIT-DEC-003",
        domain="DC_PENSION_DECUMULATION",
        label="Annuity Consideration — Guaranteed Income Exploration",
        description="Consumer in or approaching retirement wishing to explore "
                    "whether an annuity is appropriate and seeking direction to "
                    "comparison tools.",
        trigger_type="CONSUMER_REQUEST",
        segment_ids=("SEG-DEC-003",),
        fca_ref="PS25/22-paras3.35-3.39-COBS9B-COBS19",
        intent_ids=("INTENT-ANNUITY", "INTENT-GUARANTEED-INCOME",),
        out_of_scope_flags=("SPECIFIC_ANNUITY_PRODUCT",),
    ),

    "SIT-DEC-004": SituationDef(
        situation_id="SIT-DEC-004",
        domain="DC_PENSION_DECUMULATION",
        label="Drawdown Review — Consumer Already in Drawdown",
        description="Consumer in flexi-access drawdown who has not reviewed "
                    "withdrawal rate or investment strategy for ≥ 18 months.",
        trigger_type="FIRM_INITIATIVE",
        segment_ids=("SEG-DEC-004",),
        fca_ref="PS25/22-COBS9B.3-COBS9B.10-PRIN2A",
        intent_ids=("INTENT-DRAWDOWN-REVIEW",),
        out_of_scope_flags=("PENSION_CONSOLIDATION", "SPECIFIC_ANNUITY_PRODUCT"),
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# SEGMENTS — 13 pre-defined (PS25/22 COBS 9B.4)
# ──────────────────────────────────────────────────────────────────────────────

SEGMENTS: dict[str, SegmentDef] = {

    # ── RETAIL INVESTMENT SEGMENTS ────────────────────────────────────────────

    "SEG-INV-001": SegmentDef(
        segment_id="SEG-INV-001",
        situation_id="SIT-INV-001",
        label="Cash-Heavy Non-Investor — Mass Retail Band",
        description="Consumers holding significant investible cash (£10k–£100k) "
                    "with no existing S&S ISA, GIA, or investment fund.",
        fca_ref="PS25/22-para3.22-COBS9B.4",
        criteria=(
            TraitCriterion("CHAR-F2B-I1", ">=", 10000),   # cash savings ≥ £10,000
            TraitCriterion("CHAR-F2B-I1", "<=", 100000),  # cash savings ≤ £100,000
            TraitCriterion("CHAR-F2I-I1", "==", False),   # no existing investment product
            TraitCriterion("CHAR-F2L-I1", ">=", 12),      # account tenure ≥ 12 months
        ),
        excluding=(
            _EC_VULNERABILITY,
            _EC_HIGH_COST_DEBT,
            ExcludingCharacteristic(
                char_id="CHAR-F2B-I1",
                label="High net worth threshold",
                definition="Total assets with firm > £100,000",
                rationale="Scale of assets may warrant full regulated advice.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.SIGNPOST,
                    destination="MoneyHelper — finding a financial adviser",
                    action="Signpost to regulated adviser and MoneyHelper",
                ),
            ),
        ),
        characteristic_descriptions=(
            "holds £10,000–£100,000 in cash savings",
            "does not currently hold a Stocks and Shares ISA or investment fund",
            "has held an account with us for at least 12 months",
        ),
        data_accuracy_stale_days=30,
    ),

    "SEG-INV-002": SegmentDef(
        segment_id="SEG-INV-002",
        situation_id="SIT-INV-002",
        label="First-Time Investor — Young Working Adult",
        description="Young working-age consumers who have never held an investment "
                    "product, have declared a knowledge barrier, and hold modest savings.",
        fca_ref="PS25/22-para3.22-COBS9B.4",
        criteria=(
            TraitCriterion("CHAR-P1A-I1", ">=", 1),   # age band ≥ 18 (band 1)
            TraitCriterion("CHAR-P1A-I1", "<=", 3),   # age band ≤ 40 (band 3)
            TraitCriterion("CHAR-P1C-I1", "in", (1, 2)),   # employed or self-employed
            TraitCriterion("CHAR-F2I-I1", "==", False),    # no prior investment product
            TraitCriterion("CHAR-F2B-I1", ">=", 500),      # savings ≥ £500
            TraitCriterion("CHAR-F2B-I1", "<=", 25000),    # savings ≤ £25,000
        ),
        excluding=(
            _EC_VULNERABILITY,
            ExcludingCharacteristic(
                char_id="CHAR-F2G-I1",
                label="Active high-cost debt",
                definition="Active payday loan, overdraft > £500, or BNPL arrears",
                rationale="Investing before clearing high-cost debt likely unsuitable.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.SIGNPOST,
                    destination="MoneyHelper debt guidance",
                    action="Signpost before re-entering TS",
                ),
            ),
        ),
        characteristic_descriptions=(
            "aged 18–40",
            "employed or self-employed with a regular income",
            "have not previously held an investment product with us",
            "hold savings between £500 and £25,000",
        ),
        risk_questionnaire_permitted=True,
    ),

    "SEG-INV-003": SegmentDef(
        segment_id="SEG-INV-003",
        situation_id="SIT-INV-003",
        label="ISA Allowance Non-Utiliser — Tax Year End Window",
        description="Consumer with unused ISA allowance, available cash ≥ £500, "
                    "and journey date in the final 90 days of the UK tax year.",
        fca_ref="PS25/22-para3.22-COBS9B.4",
        criteria=(
            TraitCriterion("CHAR-F2J-I1", "==", False),   # no S&S ISA sub this tax year
            TraitCriterion("CHAR-F2B-I1", ">=", 500),     # cash ≥ £500
            TraitCriterion("CHAR-F2K-I1", "==", True),    # within final 90 days of tax year
        ),
        excluding=(
            _EC_VULNERABILITY,
            ExcludingCharacteristic(
                char_id="CHAR-F2J-I1",
                label="Annual ISA limit already reached",
                definition="Consumer has subscribed to the full annual ISA allowance",
                rationale="Suggestion has no effect.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.NONE_REQUIRED,
                    action="Document rationale — consumer already maximised allowance",
                ),
            ),
        ),
        characteristic_descriptions=(
            "have not made a Stocks and Shares ISA contribution this tax year",
            "hold savings of at least £500",
            "the tax year end is approaching (within 90 days of 5 April)",
        ),
        data_accuracy_stale_days=1,   # ISA sub status must be real-time
    ),

    "SEG-INV-004": SegmentDef(
        segment_id="SEG-INV-004",
        situation_id="SIT-INV-004",
        label="Mid-Range Lump Sum Recipient — No Investment Plan",
        description="Consumer who has received or declared a lump sum of "
                    "£5,000–£75,000 and has not placed an investment instruction.",
        fca_ref="PS25/22-para3.22-COBS9B.4",
        criteria=(
            TraitCriterion("CHAR-F2M-I1", ">=", 5000),    # lump sum ≥ £5,000
            TraitCriterion("CHAR-F2M-I1", "<=", 75000),   # lump sum ≤ £75,000
            TraitCriterion("CHAR-F2I-I1", "==", False),   # no investment instruction placed
        ),
        excluding=(
            _EC_VULNERABILITY,
            ExcludingCharacteristic(
                char_id="CHAR-F2M-I1",
                label="Amount above advice threshold",
                definition="Declared or identified lump sum > £75,000",
                rationale="Scale warrants full regulated advice. "
                          "PS25/22 SIT-INV-004 escalation_threshold.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.SIGNPOST,
                    destination="Regulated financial adviser; MoneyHelper",
                    action="Exit TS; provide signpost before journey ends",
                ),
            ),
            ExcludingCharacteristic(
                char_id="CHAR-F2H-I1",
                label="Active debt or arrears",
                definition="Active mortgage arrears, debt management plan, or CCJ",
                rationale="Lump sum may be better applied to debt clearance.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.SIGNPOST,
                    destination="MoneyHelper debt guidance; StepChange",
                    action="Signpost to debt guidance",
                ),
            ),
        ),
        characteristic_descriptions=(
            "have recently received or declared a lump sum of between £5,000 and £75,000",
            "have not placed an investment instruction for these funds",
        ),
        data_accuracy_stale_days=7,
    ),

    "SEG-INV-005": SegmentDef(
        segment_id="SEG-INV-005",
        situation_id="SIT-INV-005",
        label="Dormant Investment Account Holder",
        description="Consumer holds an active investment account (S&S ISA, GIA, "
                    "or investment fund) with no activity for ≥ 12 months.",
        fca_ref="PS25/22-para3.30-COBS9B.4",
        criteria=(
            TraitCriterion("CHAR-F2I-I1", "==", True),    # holds investment product
            TraitCriterion("CHAR-F2N-I1", ">=", 12),      # months inactive ≥ 12
        ),
        excluding=(
            _EC_VULNERABILITY,
            ExcludingCharacteristic(
                char_id="CHAR-F2N-I1",
                label="Consumer has recently engaged",
                definition="Consumer logged in or transacted in past 90 days",
                rationale="Not dormant; situation does not apply.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.NONE_REQUIRED,
                    action="Consumer is engaged; no TS action required",
                ),
            ),
        ),
        characteristic_descriptions=(
            "hold an investment account with us that has had no activity for over 12 months",
        ),
        risk_questionnaire_permitted=False,
    ),

    "SEG-INV-006": SegmentDef(
        segment_id="SEG-INV-006",
        situation_id="SIT-INV-006",
        label="Regular Cash Saver — No Investment Counterpart",
        description="Consumer with regular cash savings habit (≥ £50/month "
                    "for ≥ 6 months) and no investment product.",
        fca_ref="PS25/22-COBS9B.4",
        criteria=(
            TraitCriterion("CHAR-F2O-I1", ">=", 50),      # regular saving ≥ £50/month
            TraitCriterion("CHAR-F2P-I1", ">=", 6),       # consecutive months ≥ 6
            TraitCriterion("CHAR-F2I-I1", "==", False),   # no investment product
        ),
        excluding=(
            _EC_VULNERABILITY,
            _EC_HIGH_COST_DEBT,
        ),
        characteristic_descriptions=(
            "save regularly into a cash account (≥ £50/month for ≥ 6 months)",
            "do not currently hold a Stocks and Shares ISA with us",
        ),
    ),

    # ── STRUCTURED DEPOSIT SEGMENT ────────────────────────────────────────────

    "SEG-SD-001": SegmentDef(
        segment_id="SEG-SD-001",
        situation_id="SIT-SD-001",
        label="Maturing Deposit Holder — No Reinvestment Plan",
        description="Consumer holding a fixed-rate or structured deposit maturing "
                    "within 60 days with no reinvestment instruction placed.",
        fca_ref="PS25/22-COBS9B.4",
        criteria=(
            TraitCriterion("CHAR-F2Q-I1", "<=", 60),   # days to deposit maturity ≤ 60
            TraitCriterion("CHAR-F2R-I1", "==", False), # no reinvestment instruction placed
        ),
        excluding=(
            _EC_VULNERABILITY,
            ExcludingCharacteristic(
                char_id="CHAR-F2B-I1",
                label="Amount above advice threshold",
                definition="Maturing deposit > £100,000",
                rationale="Scale may warrant regulated advice.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.SIGNPOST,
                    destination="MoneyHelper; regulated financial adviser",
                    action="Signpost before journey continues",
                ),
            ),
        ),
        characteristic_descriptions=(
            "hold a fixed-rate or structured deposit maturing within 60 days",
            "have not yet placed a reinvestment instruction",
        ),
    ),

    # ── DC PENSION ACCUMULATION SEGMENTS ──────────────────────────────────────

    "SEG-PEN-001": SegmentDef(
        segment_id="SEG-PEN-001",
        situation_id="SIT-PEN-001",
        label="Under-Saving DC Member — Working Age with Projected Shortfall",
        description="Active DC member aged 25–57, contributing at or below the "
                    "auto-enrolment minimum, with segment-level projection below "
                    "67% of salary at state pension age.",
        fca_ref="PS25/22-para3.22-COBS9B.4",
        criteria=(
            TraitCriterion("CHAR-P1A-I1", ">=", 2),   # age band ≥ 25 (band 2)
            TraitCriterion("CHAR-P1A-I1", "<=", 4),   # age band ≤ 57 (band 4)
            TraitCriterion("CHAR-P2A-I1", "<=", 8),   # contribution rate ≤ 8%
            TraitCriterion("CHAR-P2B-I1", "==", True), # active DC scheme member
            TraitCriterion("CHAR-P2C-I1", "==", True), # projected shortfall (model output)
        ),
        excluding=(
            _EC_VULNERABILITY,
            _EC_HARDSHIP,
            ExcludingCharacteristic(
                char_id="CHAR-P2D-I1",
                label="Within 5 years of retirement",
                definition="Within 5 years of stated or estimated state pension age",
                rationale="Accumulation suggestion less relevant; redirect to decumulation.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.SEGMENT_REDIRECT,
                    destination="SEG-DEC-001 or SEG-DEC-002",
                    segment_id="SEG-DEC-001",
                    action="Redirect to decumulation segment",
                ),
            ),
        ),
        characteristic_descriptions=(
            "aged 25–57",
            "your current pension contribution rate is at or below the auto-enrolment minimum",
            "your projected retirement income appears below a reasonable target",
        ),
        risk_questionnaire_permitted=False,
    ),

    "SEG-PEN-002": SegmentDef(
        segment_id="SEG-PEN-002",
        situation_id="SIT-PEN-002",
        label="Default Fund Disengaged DC Member — Age-Lifecycle Mismatch",
        description="DC member invested 100% in default fund with no active fund "
                    "selection history, where age-band creates a lifecycle mismatch.",
        fca_ref="PS25/22-para3.22-COBS9B.4-PROD",
        criteria=(
            TraitCriterion("CHAR-P2E-I1", "==", True),   # 100% in default fund
            TraitCriterion("CHAR-P2F-I1", "==", False),  # no active fund selection ever
            TraitCriterion("CHAR-P2B-I1", "==", True),   # active DC member
        ),
        excluding=(
            _EC_VULNERABILITY,
            ExcludingCharacteristic(
                char_id="CHAR-P2G-I1",
                label="Recent active engagement",
                definition="Active fund switch or investment choice interaction within 24 months",
                rationale="Consumer recently engaged; suggestion not appropriate now.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.NONE_REQUIRED,
                    action="Document: consumer recently engaged. No TS required now.",
                ),
            ),
        ),
        characteristic_descriptions=(
            "invested 100% in the default fund",
            "have never made an active fund choice",
            "are in the [age band] stage of your pension journey",
        ),
        risk_questionnaire_permitted=True,
    ),

    "SEG-PEN-003": SegmentDef(
        segment_id="SEG-PEN-003",
        situation_id="SIT-PEN-003",
        label="Life-Event Triggered Contribution Review Candidate",
        description="Active DC member who has experienced a life event creating "
                    "headroom, with contribution rate below voluntary threshold.",
        fca_ref="PS25/22-para3.22-COBS9B.4",
        criteria=(
            TraitCriterion("CHAR-P2H-I1", "==", True),   # life event signal detected
            TraitCriterion("CHAR-P2A-I1", "<=", 12),     # contribution rate < 12%
            TraitCriterion("CHAR-P2B-I1", "==", True),   # active DC member
        ),
        excluding=(
            _EC_VULNERABILITY,
            _EC_HARDSHIP,
            ExcludingCharacteristic(
                char_id="CHAR-P2D-I1",
                label="Within 5 years of retirement",
                definition="Within 5 years of state pension age",
                rationale="Accumulation less relevant; redirect to decumulation.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.SEGMENT_REDIRECT,
                    destination="SEG-DEC-001",
                    segment_id="SEG-DEC-001",
                    action="Redirect to decumulation segment",
                ),
            ),
        ),
        characteristic_descriptions=(
            "a recent change in your financial position suggests you may have "
            "more capacity to save for retirement",
        ),
        risk_questionnaire_permitted=False,
    ),

    # ── DC PENSION DECUMULATION SEGMENTS ──────────────────────────────────────

    "SEG-DEC-001": SegmentDef(
        segment_id="SEG-DEC-001",
        situation_id="SIT-DEC-001",
        label="Pre-Retirement Non-Planner — 45 to 65",
        description="DC pension holder aged 45–65 with no recorded retirement "
                    "access plan and no engagement with decumulation tools.",
        fca_ref="PS25/22-para3.22-COBS9B.4-COBS19",
        criteria=(
            TraitCriterion("CHAR-P1A-I1", ">=", 3),    # age band ≥ 45 (band 3)
            TraitCriterion("CHAR-P1A-I1", "<=", 5),    # age band ≤ 65 (band 5)
            TraitCriterion("CHAR-P2B-I1", "==", True),  # DC pension held
            TraitCriterion("CHAR-P2I-I1", "==", False), # no retirement access plan on record
        ),
        excluding=(
            _EC_VULNERABILITY,
            ExcludingCharacteristic(
                char_id="CHAR-P2J-I1",
                label="Already in drawdown or annuity",
                definition="Consumer has already commenced pension access in any form",
                rationale="Access decision already made; situation does not apply.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.NONE_REQUIRED,
                    action="Route to SEG-DEC-004 if in drawdown",
                ),
            ),
            ExcludingCharacteristic(
                char_id="CHAR-P2K-I1",
                label="DB transfer in progress or applicable",
                definition="Consumer has DB pension requiring specialist advice",
                rationale="Mandatory specialist advice regime (COBS 9/9A); TS not appropriate.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.MANDATORY_REFERRAL,
                    destination="Regulated pension transfer specialist; Pension Wise",
                    action="Mandatory referral — do not proceed with TS",
                ),
            ),
        ),
        characteristic_descriptions=(
            "have a DC pension with us",
            "are aged 45–65",
            "have not yet chosen how to access your pension savings",
        ),
        risk_questionnaire_permitted=False,
    ),

    "SEG-DEC-002": SegmentDef(
        segment_id="SEG-DEC-002",
        situation_id="SIT-DEC-002",
        label="Small Pot Holder — Imminent Retirement, No Access Plan",
        description="DC pension holder with pot £5,000–£30,000, within 5 years "
                    "of retirement, with no access method nominated.",
        fca_ref="PS25/22-COBS9B.4-COBS19",
        criteria=(
            TraitCriterion("CHAR-P2L-I1", ">=", 5000),    # pot value ≥ £5,000
            TraitCriterion("CHAR-P2L-I1", "<=", 30000),   # pot value ≤ £30,000
            TraitCriterion("CHAR-P2D-I1", "<=", 5),       # years to retirement ≤ 5
            TraitCriterion("CHAR-P2I-I1", "==", False),   # no access nomination
        ),
        excluding=(
            _EC_VULNERABILITY,
            ExcludingCharacteristic(
                char_id="CHAR-P2L-I1",
                label="Trivial commutation range",
                definition="DC pot value < £2,000",
                rationale="Different regulatory rules apply to very small pots.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.SIGNPOST,
                    destination="MoneyHelper small pots guidance",
                    action="Direct to MoneyHelper trivial commutation guidance",
                ),
            ),
            ExcludingCharacteristic(
                char_id="CHAR-P2L-I1",
                label="Above segment ceiling",
                definition="Pot value > £30,000",
                rationale="Material financial decision; may require comprehensive support.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.SIGNPOST,
                    destination="Pension Wise full guidance appointment; SEG-DEC-001",
                    action="Signpost to Pension Wise; consider SEG-DEC-001",
                ),
            ),
        ),
        characteristic_descriptions=(
            "your pension pot is between £5,000 and £30,000",
            "you are within 5 years of retirement",
            "you have not chosen an access method",
        ),
        risk_questionnaire_permitted=False,
    ),

    "SEG-DEC-003": SegmentDef(
        segment_id="SEG-DEC-003",
        situation_id="SIT-DEC-003",
        label="Annuity Enquirer — Retirement Stage",
        description="Consumer aged ≥ 50 who has explicitly asked about annuities "
                    "or guaranteed income, holding a DC pension.",
        fca_ref="PS25/22-paras3.35-3.39-COBS9B.4-COBS19",
        criteria=(
            TraitCriterion("CHAR-P2M-I1", "==", True),   # expressed annuity interest
            TraitCriterion("CHAR-P2B-I1", "==", True),   # DC pension held
            TraitCriterion("CHAR-P1A-I1", ">=", 4),      # age band ≥ 50 (band 4)
        ),
        excluding=(
            _EC_VULNERABILITY,
            ExcludingCharacteristic(
                char_id="CHAR-P2K-I1",
                label="DB safeguarded benefits",
                definition="Consumer has safeguarded benefits requiring independent advice",
                rationale="Specialist regulated advice mandatory.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.MANDATORY_REFERRAL,
                    destination="Regulated pension transfer specialist",
                    action="Mandatory referral",
                ),
            ),
        ),
        characteristic_descriptions=(
            "have asked about annuities or guaranteed retirement income",
            "hold a DC pension with us",
        ),
        risk_questionnaire_permitted=False,
    ),

    "SEG-DEC-004": SegmentDef(
        segment_id="SEG-DEC-004",
        situation_id="SIT-DEC-004",
        label="Existing Drawdown Member — Lapsed Engagement",
        description="Consumer in active flexi-access drawdown with no strategy "
                    "review in ≥ 18 months and a cash drag or withdrawal concern.",
        fca_ref="PS25/22-COBS9B.4-COBS9B.10-PRIN2A",
        criteria=(
            TraitCriterion("CHAR-P2J-I1", "==", True),   # in active drawdown
            TraitCriterion("CHAR-P2N-I1", ">=", 18),     # months since last review ≥ 18
            # At least one of: cash > 20% of pot OR withdrawal rate > 6%
            # (engine evaluates compound OR conditions via separate criteria)
            TraitCriterion("CHAR-P2O-I1", "==", True),   # cash drag or withdrawal concern flag
        ),
        excluding=(
            _EC_VULNERABILITY,
            ExcludingCharacteristic(
                char_id="CHAR-P2P-I1",
                label="Recent independent advice",
                definition="Declared engagement with regulated adviser in past 12 months",
                rationale="Consumer has appropriate support.",
                alternative_support=AlternativeSupport(
                    support_type=AlternativeSupportType.NONE_REQUIRED,
                    action="Consumer advised; no TS required",
                ),
            ),
        ),
        characteristic_descriptions=(
            "you are in drawdown with us",
            "your account has had no activity for over 18 months",
            "your account shows a potential cash drag or elevated withdrawal rate",
        ),
        risk_questionnaire_permitted=False,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# SUGGESTIONS — 13 pre-defined (PS25/22 COBS 9B.5)
# ──────────────────────────────────────────────────────────────────────────────

SUGGESTIONS: dict[str, SuggestionDef] = {

    # ── RETAIL INVESTMENT SUGGESTIONS ─────────────────────────────────────────

    "SUG-INV-001": SuggestionDef(
        suggestion_id="SUG-INV-001",
        segment_ids=("SEG-INV-001",),
        product_name="Stocks and Shares ISA — Multi-Asset Balanced Fund",
        product_type="STOCKS_SHARES_ISA",
        domain="RETAIL_INVESTMENTS",
        fca_ref="PS25/22-COBS9B.5-COBS4",
        description="Open a S&S ISA and contribute to a diversified multi-asset "
                    "balanced fund to begin growing savings above cash real return.",
        cobs_9b_suitability="Consumers in SEG-INV-001 hold material investible cash "
                            "(£10k–£100k), have no existing investment product, are not "
                            "in financial hardship, are not vulnerable, and are not HNW. "
                            "A diversified multi-asset balanced fund in an ISA is "
                            "appropriate at segment level. COBS 9B — not COBS 9/9A.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003", "DEL-004",
            "DEL-005", "DEL-006", "DEL-012", "DEL-013",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-012"),
    ),

    "SUG-INV-002": SuggestionDef(
        suggestion_id="SUG-INV-002",
        segment_ids=("SEG-INV-002",),
        product_name="Begin Regular Investment — Entry-Level ISA Starter Plan",
        product_type="STOCKS_SHARES_ISA_REGULAR",
        domain="RETAIL_INVESTMENTS",
        fca_ref="PS25/22-COBS9B.5-COBS4",
        description="Open a S&S ISA and begin a regular monthly contribution into "
                    "a diversified entry-level fund, building an investing habit via "
                    "pound-cost averaging.",
        cobs_9b_suitability="Consumers in SEG-INV-002 are young, employed, have no "
                            "prior investment, hold modest savings. A low monthly "
                            "contribution into a diversified starter fund presents "
                            "minimal complexity, no leverage, appropriate for a long "
                            "accumulation horizon. Suitable at segment level under COBS 9B.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003", "DEL-004",
            "DEL-005", "DEL-006", "DEL-012", "DEL-013",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-012"),
    ),

    "SUG-INV-003": SuggestionDef(
        suggestion_id="SUG-INV-003",
        segment_ids=("SEG-INV-003",),
        product_name="Use Your ISA Allowance Before Tax Year End",
        product_type="STOCKS_SHARES_ISA_TAX_YEAR",
        domain="RETAIL_INVESTMENTS",
        fca_ref="PS25/22-COBS9B.5-COBS4",
        description="Open a S&S ISA or contribute to an existing one before 5 April "
                    "to use the remaining annual ISA allowance.",
        cobs_9b_suitability="Consumer has cash, no S&S ISA, and allowance about to "
                            "lapse permanently. Diversified fund is not high-risk. "
                            "Time constraint makes prompt action the clearly suitable "
                            "direction for consumers in this segment.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003", "DEL-004",
            "DEL-005", "DEL-006", "DEL-012", "DEL-013",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-012"),
    ),

    "SUG-INV-004": SuggestionDef(
        suggestion_id="SUG-INV-004",
        segment_ids=("SEG-INV-004",),
        product_name="Invest Your Lump Sum — ISA and General Investment Account",
        product_type="ISA_AND_GIA_LUMP_SUM",
        domain="RETAIL_INVESTMENTS",
        fca_ref="PS25/22-COBS9B.5-COBS4",
        description="Invest the lump sum into a S&S ISA (up to annual allowance) "
                    "and a GIA for amounts exceeding the allowance, using a "
                    "diversified multi-asset fund.",
        cobs_9b_suitability="Segment holds mid-range lump sum (£5k–£75k), no "
                            "investment plan, no debt stress, no vulnerability. "
                            "Diversified multi-asset fund in ISA/GIA is appropriate "
                            "for medium-term lump sum at this scale under COBS 9B.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003", "DEL-004",
            "DEL-005", "DEL-006", "DEL-012", "DEL-013",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-012"),
    ),

    "SUG-INV-005": SuggestionDef(
        suggestion_id="SUG-INV-005",
        segment_ids=("SEG-INV-005",),
        product_name="Review Your Investment — Do Nothing or Re-engage",
        product_type="REVIEW_PROMPT_DO_NOTHING",
        domain="RETAIL_INVESTMENTS",
        fca_ref="PS25/22-para3.30-COBS9B.5-COBS4",
        description="Re-engagement prompt for dormant investment account. "
                    "PS25/22 para 3.30 explicitly permits 'do nothing' as a "
                    "valid ready-made suggestion.",
        cobs_9b_suitability="Consumer holds an existing investment with no "
                            "activity. A re-engagement prompt with a 'do nothing "
                            "is valid' pathway is suitable for all consumers in "
                            "SEG-INV-005 under COBS 9B.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003",
            "DEL-005", "DEL-006", "DEL-012", "DEL-013",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-012"),
    ),

    "SUG-INV-006": SuggestionDef(
        suggestion_id="SUG-INV-006",
        segment_ids=("SEG-INV-006",),
        product_name="Add a Regular Investment to Your Savings Habit",
        product_type="STOCKS_SHARES_ISA_REGULAR_ADDITION",
        domain="RETAIL_INVESTMENTS",
        fca_ref="PS25/22-COBS9B.5-COBS4",
        description="Add a regular monthly contribution into a S&S ISA alongside "
                    "existing cash savings habit.",
        cobs_9b_suitability="Consumer has demonstrated sustained surplus cash; "
                            "no high-cost debt; no vulnerability. A modest incremental "
                            "investment contribution alongside existing savings is "
                            "appropriate at segment level under COBS 9B.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003", "DEL-004",
            "DEL-005", "DEL-006", "DEL-012", "DEL-013",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-012"),
    ),

    # ── STRUCTURED DEPOSIT SUGGESTION ─────────────────────────────────────────

    "SUG-SD-001": SuggestionDef(
        suggestion_id="SUG-SD-001",
        segment_ids=("SEG-SD-001",),
        product_name="Reinvest Your Maturing Deposit — Cash, Structured Deposit, or Investment",
        product_type="MATURITY_REINVESTMENT_OPTIONS",
        domain="STRUCTURED_DEPOSITS",
        fca_ref="PS25/22-COBS9B.5-COBS4",
        description="At deposit maturity: Option A (new fixed/structured deposit), "
                    "Option B (S&S ISA or investment fund), "
                    "Option C (easy-access savings temporarily). "
                    "Multiple-options per PS25/22 para 3.32.",
        cobs_9b_suitability="Consumer holds a maturing deposit with no reinvestment "
                            "plan. Presenting structured reinvestment options is "
                            "suitable for all consumers in SEG-SD-001. Options are "
                            "mass-market; capital at risk disclosed for investment option.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003", "DEL-004",
            "DEL-005", "DEL-006", "DEL-012", "DEL-013",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-012"),
    ),

    # ── DC PENSION ACCUMULATION SUGGESTIONS ───────────────────────────────────

    "SUG-PEN-001": SuggestionDef(
        suggestion_id="SUG-PEN-001",
        segment_ids=("SEG-PEN-001",),
        product_name="Increase Your Pension Contribution Rate",
        product_type="DC_PENSION_CONTRIBUTION_INCREASE",
        domain="DC_PENSION_ACCUMULATION",
        fca_ref="PS25/22-COBS9B.5-COBS19",
        description="Increase employee pension contribution rate by 2–4% to "
                    "materially improve projected retirement income. Includes "
                    "standardised COBS illustration and employer match check.",
        cobs_9b_suitability="Segment is working-age, at/below AE minimum, projected "
                            "shortfall evident at segment level, not in hardship, not "
                            "vulnerable. Increasing contributions within the same "
                            "existing scheme carries no product change risk. Suitable "
                            "for all correctly aligned consumers in SEG-PEN-001 under COBS 9B.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004", "PDC-007"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003",
            "DEL-006", "DEL-007", "DEL-012", "DEL-013", "DEL-014",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-007"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-007", "DEL-012", "DEL-014"),
    ),

    "SUG-PEN-002": SuggestionDef(
        suggestion_id="SUG-PEN-002",
        segment_ids=("SEG-PEN-002",),
        product_name="Switch to an Age-Appropriate Pension Fund",
        product_type="DC_PENSION_FUND_SWITCH",
        domain="DC_PENSION_ACCUMULATION",
        fca_ref="PS25/22-COBS9B.5-PROD-COBS19",
        description="Fund switch within existing DC pension scheme to an "
                    "age-appropriate risk-rated fund: Growth (25–35), "
                    "Balanced (36–50), or Cautious/Preservation (51–60). "
                    "Do-nothing pathway valid per PS25/22 para 3.30.",
        cobs_9b_suitability="Segment uses three objective criteria to produce "
                            "three lifecycle sub-groups. Fund options are firm's "
                            "regulated pension funds — not high-risk, not leveraged. "
                            "Suitability confirmed at segment level under COBS 9B.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003",
            "DEL-004", "DEL-005", "DEL-006", "DEL-012", "DEL-013", "DEL-014",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-012", "DEL-014"),
    ),

    "SUG-PEN-003": SuggestionDef(
        suggestion_id="SUG-PEN-003",
        segment_ids=("SEG-PEN-003",),
        product_name="Review Your Pension Contributions After Your Life Change",
        product_type="DC_PENSION_CONTRIBUTION_REVIEW",
        domain="DC_PENSION_ACCUMULATION",
        fca_ref="PS25/22-COBS9B.5-COBS19",
        description="Contribution rate review prompt triggered by a life event "
                    "(salary increase, loan clearance, etc.). Suggest 1–3% increase "
                    "and check for employer matching. Medium-materiality assumption "
                    "consumer check required.",
        cobs_9b_suitability="Segment is life-event triggered, objective transactional "
                            "evidence, active DC member, not in hardship, not "
                            "vulnerable. Contribution rate review within same scheme "
                            "is low-risk and appropriate for all correctly aligned "
                            "consumers under COBS 9B.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004", "PDC-007"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003",
            "DEL-006", "DEL-007", "DEL-012", "DEL-013", "DEL-014",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-007"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-007", "DEL-012", "DEL-014"),
    ),

    # ── DC PENSION DECUMULATION SUGGESTIONS ───────────────────────────────────

    "SUG-DEC-001": SuggestionDef(
        suggestion_id="SUG-DEC-001",
        segment_ids=("SEG-DEC-001",),
        product_name="Start Planning How You'll Access Your Pension",
        product_type="DECUMULATION_PATHWAY_DIRECTION",
        domain="DC_PENSION_DECUMULATION",
        fca_ref="PS25/22-COBS9B.5-COBS19-COBS4",
        description="Directional prompt covering drawdown and annuity pathways "
                    "with mandatory Pension Wise signpost. "
                    "Specific annuity product recommendation PROHIBITED. "
                    "Pension consolidation PROHIBITED. "
                    "MoneyHelper annuity tool referral MANDATORY if annuity discussed.",
        cobs_9b_suitability="Suggestion is directional (pathway exploration + Pension "
                            "Wise referral), not a final product selection. Consumer "
                            "retains full decision-making responsibility. Suitable for "
                            "all consumers in SEG-DEC-001 under COBS 9B.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003",
            "DEL-005", "DEL-006", "DEL-008", "DEL-012", "DEL-013", "DEL-014",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-008", "DEL-012", "DEL-014"),
    ),

    "SUG-DEC-002": SuggestionDef(
        suggestion_id="SUG-DEC-002",
        segment_ids=("SEG-DEC-002",),
        product_name="Access Your Small Pension Pot — UFPLS or Drawdown",
        product_type="DC_PENSION_SMALL_POT_ACCESS",
        domain="DC_PENSION_DECUMULATION",
        fca_ref="PS25/22-COBS9B.5-COBS19",
        description="For small pots (£5k–£30k): Option A (UFPLS — 25% tax-free, "
                    "75% taxable income), Option B (flexi-access drawdown). "
                    "Pension Wise signpost MANDATORY. "
                    "Specific annuity recommendation and pension consolidation PROHIBITED.",
        cobs_9b_suitability="Small pot (£5k–£30k), within 5 years of retirement, "
                            "no access plan. UFPLS and flexi-access drawdown are "
                            "standard mass-market DC access routes. No annuity product "
                            "recommended. Do-nothing pathway offered. COBS 9B.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003",
            "DEL-006", "DEL-008", "DEL-012", "DEL-013", "DEL-014",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-008", "DEL-012", "DEL-014"),
    ),

    "SUG-DEC-003": SuggestionDef(
        suggestion_id="SUG-DEC-003",
        segment_ids=("SEG-DEC-003",),
        product_name="Explore Annuity Options — Features and Comparison Referral",
        product_type="ANNUITY_FEATURES_REFERRAL",
        domain="DC_PENSION_DECUMULATION",
        fca_ref="PS25/22-paras3.35-3.39-COBS9B.5-COBS19",
        description="Annuity feature guidance (joint/single life, escalation, "
                    "guaranteed periods, enhanced). "
                    "Specific named annuity product PROHIBITED. "
                    "Quotation including illustrative average rate PROHIBITED. "
                    "MoneyHelper annuity comparison tool MANDATORY. "
                    "Whole-of-market brokerage referral PERMITTED after TS journey ends.",
        cobs_9b_suitability="Consumer has expressed annuity interest; holds DC pension; "
                            "aged ≥ 50. Suggestion covers features and comparison "
                            "direction only — no product recommendation. Suitable for "
                            "all correctly aligned consumers in SEG-DEC-003 under COBS 9B.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003",
            "DEL-006", "DEL-008", "DEL-010", "DEL-011", "DEL-012", "DEL-013", "DEL-014",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=(
            "DEL-001", "DEL-006", "DEL-008", "DEL-010", "DEL-011", "DEL-012", "DEL-014"
        ),
    ),

    "SUG-DEC-004": SuggestionDef(
        suggestion_id="SUG-DEC-004",
        segment_ids=("SEG-DEC-004",),
        product_name="Review Your Drawdown Strategy — Cash Drag or Withdrawal Rate Check",
        product_type="DRAWDOWN_REVIEW_PROMPT",
        domain="DC_PENSION_DECUMULATION",
        fca_ref="PS25/22-COBS9B.5-COBS9B.10-PRIN2A-COBS19",
        description="One-off re-engagement prompt for lapsed drawdown consumers. "
                    "Option A (review and adjust), Option B (do nothing — confirm). "
                    "PS25/22 simplified monitoring: no ongoing individual suitability. "
                    "Pension consolidation PROHIBITED.",
        cobs_9b_suitability="Consumer in drawdown; demonstrable inactivity and "
                            "financial signal. Re-engagement prompt with do-nothing "
                            "pathway suitable for all consumers in SEG-DEC-004. "
                            "One-off nature consistent with PS25/22 simplified "
                            "monitoring rules.",
        pre_delivery_checks=("PDC-001", "PDC-002", "PDC-003", "PDC-004"),
        delivery_checks=(
            "DEL-001", "DEL-002", "DEL-003",
            "DEL-005", "DEL-006", "DEL-012", "DEL-013", "DEL-014",
        ),
        hard_prohibitions=("DC-001", "DC-002", "DC-003"),
        eligibility_rules=("PDC-001", "PDC-002"),
        suitability_rules=("PDC-003", "PDC-004"),
        compliance_rules=("DEL-001", "DEL-006", "DEL-012", "DEL-014"),
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# COMPLIANCE CHECKS — 28 checks across 5 phases (fca_ts_compliance_checks.yml)
# ──────────────────────────────────────────────────────────────────────────────

COMPLIANCE_CHECKS: dict[str, ComplianceCheckDef] = {

    # Phase 0: Pre-launch (not evaluated per-session; firm-level checks)
    "PL-001": ComplianceCheckDef(
        check_id="PL-001", phase=CheckPhase.PRE_LAUNCH,
        severity=CheckSeverity.HARD_BLOCK,
        label="FCA Targeted Support Permission — VoP Required",
        rule_text="Firm must hold FCA variation of permission for TS (RAO new "
                  "specified activity). ARs cannot deliver TS.",
        regulatory_source="PS25/22-para1.7; RAO new specified activity; "
                          "FSMA 2000 s.19",
        failure_action="Halt ALL targeted support. Escalate to CCO.",
        gate_on_fail="SUPPRESS",
    ),
    "PL-002": ComplianceCheckDef(
        check_id="PL-002", phase=CheckPhase.PRE_LAUNCH,
        severity=CheckSeverity.HARD_BLOCK,
        label="Situations, Segments, and Suggestions Pre-Defined",
        rule_text="All situations, segments, and suggestions must be pre-defined "
                  "before any TS delivery. Dynamic individual construction prohibited.",
        regulatory_source="PS25/22-paras3.14-3.15; COBS9B.3-9B.5; "
                          "AI-tension: PS25/22-para3.13",
        failure_action="Halt delivery. Match to pre-defined segment or exit TS.",
        gate_on_fail="SUPPRESS",
    ),

    # Phase 1: Design checks (evaluated at segment/suggestion design time)
    "DC-001": ComplianceCheckDef(
        check_id="DC-001", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.HARD_BLOCK,
        label="In-Scope Products Only",
        rule_text="Suggestions must only involve mass-market securities, S&S ISA, "
                  "GIA (mass-market), structured deposits, investment-based life "
                  "insurance, DC pension contributions/fund switches/access routes. "
                  "Mortgages, pure protection insurance, and debt/credit products "
                  "are explicitly out of scope.",
        regulatory_source="PS25/22-paras3.45-3.48; COBS9B; "
                          "Mondaq/A&O-Shearman-Dec2025",
        failure_action="Remove prohibited product. Halt suggestion.",
        gate_on_fail="SUPPRESS",
    ),
    "DC-002": ComplianceCheckDef(
        check_id="DC-002", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.HARD_BLOCK,
        label="Absolute Prohibition — No Pension Consolidation",
        rule_text="TS must not include any suggestion to consolidate any pension "
                  "arrangements in any form.",
        regulatory_source="PS25/22-paras3.40-3.44; COBS9B",
        failure_action="Remove consolidation element entirely.",
        gate_on_fail="SUPPRESS",
    ),
    "DC-003": ComplianceCheckDef(
        check_id="DC-003", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.HARD_BLOCK,
        label="Absolute Prohibition — No Specific Annuity Product Recommendation",
        rule_text="TS must not include a recommendation of a specific named annuity "
                  "product or a quotation (including illustrative average rates). "
                  "Annuity features and MoneyHelper tool referral are permitted.",
        regulatory_source="PS25/22-paras3.35-3.39",
        failure_action="Replace with annuity features guidance + MoneyHelper referral.",
        gate_on_fail="SUPPRESS",
    ),
    "DC-004": ComplianceCheckDef(
        check_id="DC-004", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.HARD_BLOCK,
        label="Granularity Ceiling — No Individual Profiling",
        rule_text="Segments must not be designed with a level of detail that "
                  "constitutes a comprehensive individual assessment.",
        regulatory_source="PS25/22-para3.27; COBS9B.4",
        failure_action="Redesign segment. Remove comprehensive profiling characteristics.",
        gate_on_fail="SUPPRESS",
    ),
    "DC-005": ComplianceCheckDef(
        check_id="DC-005", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.HARD_BLOCK,
        label="Granularity Floor — Segment Must Support Suitability",
        rule_text="Segments must not be so broad that no suitable suggestion can "
                  "be designed for all consumers in the segment.",
        regulatory_source="PS25/22-para3.27; COBS9B.4",
        failure_action="Add differentiating including characteristics.",
        gate_on_fail="SUPPRESS",
    ),
    "DC-006": ComplianceCheckDef(
        check_id="DC-006", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.HARD_BLOCK,
        label="Including AND Excluding Characteristics — Both Required",
        rule_text="All segments must specify at least one including and one excluding "
                  "characteristic.",
        regulatory_source="PS25/22-para3.22; COBS9B.4",
        failure_action="Add missing characteristics before segment goes live.",
        gate_on_fail="SUPPRESS",
    ),
    "DC-007": ComplianceCheckDef(
        check_id="DC-007", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.SOFT_WARNING,
        label="Excluding Characteristics — Alternative Support Documented",
        rule_text="When defining an excluding characteristic, the firm must consider "
                  "creating a new segment or providing alternative support, or document "
                  "a clear rationale.",
        regulatory_source="PS25/22-para3.27; COBS9B.4; PRIN2A",
        failure_action="Document rationale. Flag for compliance review.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "DC-008": ComplianceCheckDef(
        check_id="DC-008", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.HARD_BLOCK,
        label="No Material Assumptions",
        rule_text="Assumptions material to the suitability of the suggestion "
                  "are prohibited. Suitability must rest on segment characteristics.",
        regulatory_source="PS25/22-paras3.28-3.29; COBS9B",
        failure_action="Convert material assumption to explicit including characteristic.",
        gate_on_fail="SUPPRESS",
    ),
    "DC-009": ComplianceCheckDef(
        check_id="DC-009", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.SOFT_WARNING,
        label="UK GDPR Data Minimisation",
        rule_text="Only data adequate, relevant, and limited to what is necessary "
                  "for the TS purpose should be used.",
        regulatory_source="PS25/22-para3.24; UK-GDPR-Art5(1)(c); DPA2018",
        failure_action="Review with DPO. Complete DPIA for special category data.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "DC-010": ComplianceCheckDef(
        check_id="DC-010", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.BOUNDARY_CHECK,
        label="Advice/Guidance Boundary",
        rule_text="A ready-made suggestion must constitute a personal recommendation. "
                  "Pure guidance must not be labelled as TS.",
        regulatory_source="PS25/22-paras3.31-3.33; RAO-Art53",
        failure_action="Reclassify as guidance. Do not apply TS label.",
        gate_on_fail="SUPPRESS",
    ),
    "DC-011": ComplianceCheckDef(
        check_id="DC-011", phase=CheckPhase.DESIGN,
        severity=CheckSeverity.INFORMATION_REQUIRED,
        label="Consumer Check Question — Medium-Materiality Assumptions",
        rule_text="Where a segment uses a MEDIUM-materiality assumption, a consumer "
                  "check question must be present in the journey to validate it before "
                  "segment alignment is finalised.",
        regulatory_source="PS25/22-paras3.28-3.29; COBS9B",
        failure_action="Add consumer check question. Segment cannot go live without it.",
        gate_on_fail="HUMAN_REVIEW",
    ),

    # Phase 2: Pre-delivery checks (evaluated per consumer session)
    "PDC-001": ComplianceCheckDef(
        check_id="PDC-001", phase=CheckPhase.PRE_DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="Segment Alignment Verified",
        rule_text="Consumer must meet ALL including characteristics and NONE of "
                  "the excluding characteristics of the matched segment, based on "
                  "accurate up-to-date data.",
        regulatory_source="PS25/22-para3.49; COBS9B.4; GDPR-Art5(1)(d)",
        failure_action="Do not deliver. Route per alternative_support specification.",
        gate_on_fail="SUPPRESS",
    ),
    "PDC-002": ComplianceCheckDef(
        check_id="PDC-002", phase=CheckPhase.PRE_DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="Data Accuracy — Verify Before Segment Alignment",
        rule_text="If data used for segment alignment is suspected to be inaccurate "
                  "or stale (> stale_data_threshold), the firm must verify with the "
                  "consumer before proceeding.",
        regulatory_source="PS25/22-paras3.49-3.51; UK-GDPR-Art5(1)(d)",
        failure_action="Present key data to consumer for confirmation. "
                       "Re-run alignment with corrected data.",
        gate_on_fail="SUPPRESS",
    ),
    "PDC-003": ComplianceCheckDef(
        check_id="PDC-003", phase=CheckPhase.PRE_DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="Known Unsuitability Override",
        rule_text="Must not provide suggestion if firm is, or ought reasonably to "
                  "be, aware of information indicating unsuitability — including "
                  "information held prior to and volunteered during the journey.",
        regulatory_source="PS25/22-paras3.52-3.55; COBS9B; Principle9",
        failure_action="Halt immediately. Acknowledge information. Route to alternative support.",
        gate_on_fail="SUPPRESS",
    ),
    "PDC-004": ComplianceCheckDef(
        check_id="PDC-004", phase=CheckPhase.PRE_DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="Retail Client Classification Confirmed",
        rule_text="Consumer must be treated as a retail client for TS, even if "
                  "otherwise classifiable as professional.",
        regulatory_source="PS25/22; COBS-client-categorisation",
        failure_action="Set classification to retail client for this interaction.",
        gate_on_fail="SUPPRESS",
    ),
    "PDC-005": ComplianceCheckDef(
        check_id="PDC-005", phase=CheckPhase.PRE_DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="No PDS Post-View Service Delivery",
        rule_text="TS must not be offered as a post-view service (PVS) within the "
                  "Pension Dashboard Service context.",
        regulatory_source="PS25/22; PDS-regulations",
        failure_action="Halt. Route to standalone TS journey outside pension dashboard.",
        gate_on_fail="SUPPRESS",
    ),
    "PDC-006": ComplianceCheckDef(
        check_id="PDC-006", phase=CheckPhase.PRE_DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="Appointed Representative — Cannot Deliver TS",
        rule_text="TS may only be delivered by directly authorised firms. "
                  "Appointed representatives are excluded.",
        regulatory_source="HMT-consultation-Dec2025; PS25/22",
        failure_action="Halt. Route consumer to directly authorised firm.",
        gate_on_fail="SUPPRESS",
    ),
    "PDC-007": ComplianceCheckDef(
        check_id="PDC-007", phase=CheckPhase.PRE_DELIVERY,
        severity=CheckSeverity.INFORMATION_REQUIRED,
        label="Medium-Materiality Assumption Consumer Check Completed",
        rule_text="Where the matched segment uses a MEDIUM-materiality assumption, "
                  "the consumer check question must have been answered before delivery.",
        regulatory_source="PS25/22-paras3.28-3.29; COBS9B",
        failure_action="Present check question. If response invalidates match, exit TS.",
        gate_on_fail="HUMAN_REVIEW",
    ),

    # Phase 3: Delivery checks (evaluated at moment of suggestion delivery)
    "DEL-001": ComplianceCheckDef(
        check_id="DEL-001", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="Mandatory Service Label — 'Targeted Support'",
        rule_text="The firm MUST explicitly label the service as 'targeted support' "
                  "at the point of delivering a ready-made suggestion. Clear and prominent.",
        regulatory_source="PS25/22-para2.8; COBS9B",
        failure_action="Halt. Add mandatory 'targeted support' label.",
        gate_on_fail="SUPPRESS",
    ),
    "DEL-002": ComplianceCheckDef(
        check_id="DEL-002", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="Not Personalised Advice — Nature and Limitations Disclosure",
        rule_text="Must communicate: (i) this is TS not individualised advice, "
                  "(ii) designed for a group with common characteristics, "
                  "(iii) not based on comprehensive individual circumstances.",
        regulatory_source="PS25/22-paras1.9-2.7; COBS9B; COBS4",
        failure_action="Add nature and limitations disclosure.",
        gate_on_fail="SUPPRESS",
    ),
    "DEL-003": ComplianceCheckDef(
        check_id="DEL-003", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="Segment Characteristics Disclosed to Consumer",
        rule_text="When delivering TS, the firm must disclose the common "
                  "characteristics of the segment on which the suggestion is based.",
        regulatory_source="PS25/22-para2.7; COBS9B",
        failure_action="Add segment characteristics disclosure.",
        gate_on_fail="SUPPRESS",
    ),
    "DEL-004": ComplianceCheckDef(
        check_id="DEL-004", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.DISCLOSURE_REQUIRED,
        label="Product Scope Limitations Disclosed",
        rule_text="Where suggestion is limited to firm's own range rather than "
                  "whole-of-market, this must be disclosed.",
        regulatory_source="PS25/22; COBS9B; COBS16",
        failure_action="Add product scope limitation disclosure.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "DEL-005": ComplianceCheckDef(
        check_id="DEL-005", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.DISCLOSURE_REQUIRED,
        label="Capital at Risk Disclosure",
        rule_text="Where suggestion involves a product where capital is at risk, "
                  "a prominent capital-at-risk disclosure must be included.",
        regulatory_source="COBS4; PRIN2A",
        failure_action="Add capital at risk disclosure.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "DEL-006": ComplianceCheckDef(
        check_id="DEL-006", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="MoneyHelper Signpost — Mandatory on All Suggestions",
        rule_text="All TS delivery communications must include a MoneyHelper "
                  "signpost. For pension suggestions, Pension Wise signpost also "
                  "mandatory under COBS 19.",
        regulatory_source="PS25/22-Ch4; COBS16; COBS19; MaPS",
        failure_action="Add MoneyHelper and/or Pension Wise signpost.",
        gate_on_fail="SUPPRESS",
    ),
    "DEL-007": ComplianceCheckDef(
        check_id="DEL-007", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.DISCLOSURE_REQUIRED,
        label="Pension Projection Disclaimer — Standardised Assumptions",
        rule_text="Where suggestion includes a pension income projection, it must "
                  "state clearly that projections use standardised assumptions and "
                  "are not a personalised forecast or guarantee.",
        regulatory_source="COBS-illustration-rules; PS25/22; PRIN2A",
        failure_action="Add projection disclaimer.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "DEL-008": ComplianceCheckDef(
        check_id="DEL-008", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.DISCLOSURE_REQUIRED,
        label="Annuity Irrevocability Disclosure",
        rule_text="Where suggestion involves an annuity pathway, must disclose "
                  "that an annuity is generally an irreversible decision.",
        regulatory_source="PS25/22-Ch3-annuity-rules; COBS19; PRIN2A",
        failure_action="Add annuity irrevocability disclosure.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "DEL-009": ComplianceCheckDef(
        check_id="DEL-009", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.DISCLOSURE_REQUIRED,
        label="Charge Variation Disclosure",
        rule_text="If product charges vary by access route (TS vs other), "
                  "this must be disclosed.",
        regulatory_source="PS25/22-Ch5; COBS9B",
        failure_action="Add charge variation disclosure.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "DEL-010": ComplianceCheckDef(
        check_id="DEL-010", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.DISCLOSURE_REQUIRED,
        label="Annuity Brokerage Referral Fee Disclosure",
        rule_text="Where firm refers consumer to annuity brokerage post-journey "
                  "and receives a referral fee, this must be disclosed per "
                  "existing FCA inducement rules.",
        regulatory_source="PS25/22-paras3.38-3.39; FCA-inducement-rules",
        failure_action="Add referral fee disclosure before providing brokerage info.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "DEL-011": ComplianceCheckDef(
        check_id="DEL-011", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="Annuity Journey End Communication Required",
        rule_text="Where firm provides annuity brokerage information after a TS "
                  "journey, it must first communicate clearly that the TS journey "
                  "has ENDED.",
        regulatory_source="PS25/22-para3.39",
        failure_action="Insert journey-end communication before brokerage information.",
        gate_on_fail="SUPPRESS",
    ),
    "DEL-012": ComplianceCheckDef(
        check_id="DEL-012", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.HARD_BLOCK,
        label="COBS 4 — Fair, Clear, and Not Misleading",
        rule_text="All TS communications must be fair, clear, and not misleading: "
                  "no guaranteed returns implied, no TS presented as full advice, "
                  "no undue urgency, no material omissions, risk warnings prominent.",
        regulatory_source="COBS4.2; PS25/22; PRIN2A",
        failure_action="Revise communication. Submit to compliance review.",
        gate_on_fail="SUPPRESS",
    ),
    "DEL-013": ComplianceCheckDef(
        check_id="DEL-013", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.LOGGING_REQUIRED,
        label="Delivery Audit Record — Mandatory Capture",
        rule_text="Every suggestion delivery must create an audit record with: "
                  "pseudonymised consumer ID, segment ID, situation ID, suggestion ID, "
                  "disclosures presented, consumer action, timestamp, channel, "
                  "override events, data accuracy verification record.",
        regulatory_source="PS25/22-Ch8; SYSC; FOS-joint-statement-Dec2025",
        failure_action="Flag incomplete record. Escalate if systematic.",
        gate_on_fail="AUDIT_ONLY",
    ),
    "DEL-014": ComplianceCheckDef(
        check_id="DEL-014", phase=CheckPhase.DELIVERY,
        severity=CheckSeverity.INFORMATION_REQUIRED,
        label="No Pension Consolidation — Affirmative Journey Check",
        rule_text="In the pensions domain, the delivery communication must "
                  "affirmatively state that pension consolidation is not being suggested.",
        regulatory_source="PS25/22-paras3.40-3.44; COBS9B",
        failure_action="Add explicit no-consolidation statement.",
        gate_on_fail="HUMAN_REVIEW",
    ),

    # Phase 4: Monitoring checks (ongoing obligations)
    "MON-001": ComplianceCheckDef(
        check_id="MON-001", phase=CheckPhase.MONITORING,
        severity=CheckSeverity.LOGGING_REQUIRED,
        label="Segment-Level Outcome Monitoring — Regular Review",
        rule_text="Firms must regularly review TS outcomes at segment level. "
                  "No ongoing individual suitability obligation. "
                  "Minimum review cadence: 12 months.",
        regulatory_source="PS25/22-para2.8; COBS9B.10; PROD; PRIN2A",
        failure_action="Schedule overdue segment review. Escalate to product governance.",
        gate_on_fail="AUDIT_ONLY",
    ),
    "MON-002": ComplianceCheckDef(
        check_id="MON-002", phase=CheckPhase.MONITORING,
        severity=CheckSeverity.SOFT_WARNING,
        label="Significant Product Adaptation — Re-suitability Assessment",
        rule_text="If a product forming part of a suggestion is significantly adapted, "
                  "the firm must assess impact on segment-level suitability.",
        regulatory_source="PS25/22-Ch6; PROD; COBS9B.10",
        failure_action="Pause affected suggestion. Conduct suitability review.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "MON-003": ComplianceCheckDef(
        check_id="MON-003", phase=CheckPhase.MONITORING,
        severity=CheckSeverity.SOFT_WARNING,
        label="Adverse Outcome Signal — Consumer Harm Detection",
        rule_text="Under Consumer Duty, firms must monitor for and respond to "
                  "signals of consumer harm at segment level. Adverse outcome rate "
                  "> 5% or complaint rate > 0.5% triggers urgent review.",
        regulatory_source="PS25/22-Ch3-Ch6; PRIN2A; COBS9B",
        failure_action="Flag for urgent review. Consider suspension. Report to SM.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "MON-004": ComplianceCheckDef(
        check_id="MON-004", phase=CheckPhase.MONITORING,
        severity=CheckSeverity.SOFT_WARNING,
        label="Vulnerable Consumer Outcome Monitoring",
        rule_text="Firms must monitor outcomes for consumers with characteristics "
                  "of vulnerability. Disproportionate exclusion or worse outcomes "
                  "must trigger review.",
        regulatory_source="PS25/22-para3.26; PRIN2A; FCA-FG21/1",
        failure_action="Review excluding characteristics. Review alternative support.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "MON-005": ComplianceCheckDef(
        check_id="MON-005", phase=CheckPhase.MONITORING,
        severity=CheckSeverity.LOGGING_REQUIRED,
        label="FOS Complaint Logging and Systemic Signal Detection",
        rule_text="All TS complaints must be logged and assessed for systemic "
                  "indicators. Systemic issues may require FCA engagement.",
        regulatory_source="PS25/22-Ch7; FCA-FOS-Joint-Statement-Dec2025; DISP",
        failure_action="Log complaint. Assess for systemic signal. Engage CCO if systemic.",
        gate_on_fail="AUDIT_ONLY",
    ),
    "MON-006": ComplianceCheckDef(
        check_id="MON-006", phase=CheckPhase.MONITORING,
        severity=CheckSeverity.LOGGING_REQUIRED,
        label="Direct Marketing Compliance — Firm-Initiative Communications",
        rule_text="All proactive firm-initiative TS communications must have a "
                  "documented legal basis under PECR.",
        regulatory_source="PS25/22; PECR-Reg22; ICO-FCA-Joint-Statement-Dec2025",
        failure_action="Document legal basis. DPO sign-off before channel goes live.",
        gate_on_fail="AUDIT_ONLY",
    ),
    "MON-007": ComplianceCheckDef(
        check_id="MON-007", phase=CheckPhase.MONITORING,
        severity=CheckSeverity.LOGGING_REQUIRED,
        label="FCA Post-Implementation Review Data Readiness — 2028",
        rule_text="Firms must retain TS data sufficient to support the FCA PIR "
                  "(due March 2028) from day one of operation.",
        regulatory_source="PS25/22-para2.16; FCA-PIR-commitment",
        failure_action="Implement PIR data capture. Retain ≥ 6 years.",
        gate_on_fail="AUDIT_ONLY",
    ),
    "MON-008": ComplianceCheckDef(
        check_id="MON-008", phase=CheckPhase.MONITORING,
        severity=CheckSeverity.SOFT_WARNING,
        label="Consumer Duty Annual Board Report — TS Section",
        rule_text="Annual board report assessing good outcomes must include a TS "
                  "section covering all four Consumer Duty outcomes for TS segments.",
        regulatory_source="PRIN2A; PS22/9",
        failure_action="Include TS outcomes in Consumer Duty annual board report.",
        gate_on_fail="HUMAN_REVIEW",
    ),
    "MON-009": ComplianceCheckDef(
        check_id="MON-009", phase=CheckPhase.MONITORING,
        severity=CheckSeverity.LOGGING_REQUIRED,
        label="AI System Change — Re-run Design Checks",
        rule_text="When this AI agent is updated in a way that changes segment "
                  "matching, suggestion selection, or disclosure assembly, all "
                  "design-stage checks (DC-001 through DC-011) must be re-run "
                  "before re-deployment.",
        regulatory_source="PS25/22-para3.13; SYSC; PRIN2A; FCA-AI-Lab",
        failure_action="Pause AI update. Re-run all design checks. Get compliance sign-off.",
        gate_on_fail="AUDIT_ONLY",
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Derived lookup structures (fast access without data duplication)
# ──────────────────────────────────────────────────────────────────────────────

# segment_id → list[SuggestionDef]
SEGMENT_TO_SUGGESTIONS: dict[str, list[SuggestionDef]] = {}
for _sugg in SUGGESTIONS.values():
    for _sid in _sugg.segment_ids:
        SEGMENT_TO_SUGGESTIONS.setdefault(_sid, []).append(_sugg)

# situation_id → list of intent_id strings
SITUATION_INTENTS: dict[str, list[str]] = {
    sit_id: list(sit.intent_ids)
    for sit_id, sit in SITUATIONS.items()
}

# All unique intent_ids → situation_id (for intent classifier routing)
INTENT_TO_SITUATION: dict[str, str] = {}
for _sit_id, _sit in SITUATIONS.items():
    for _intent in _sit.intent_ids:
        INTENT_TO_SITUATION[_intent] = _sit_id

# compliance check_id → GateDisposition contribution
CHECK_GATE_MAP: dict[str, str] = {
    cid: chk.gate_on_fail
    for cid, chk in COMPLIANCE_CHECKS.items()
}

# Legacy: RULES dict — maps old R-00x IDs to ComplianceCheckDef
# for backward compatibility with existing explainer and test code.
# Maps: R-001 → PDC-001, R-002 → PDC-001, R-003 → PDC-003,
#       R-004 → PDC-001, R-005 → PDC-001, R-006 → PDC-001,
#       R-007 → PDC-003, R-008 → MON-001, R-009 → PDC-007,
#       R-010 → DC-001,  R-011 → DEL-012, R-012 → PDC-001
# This allows zone3/suggestion_engine.py and explainer.py to continue
# referencing R-00x IDs during the migration period.
_LEGACY_RULE_MAP: dict[str, str] = {
    "R-001": "PDC-001",  # Segment match
    "R-002": "PDC-001",  # Age requirement (including characteristic)
    "R-003": "PDC-003",  # Vulnerability / known unsuitability override
    "R-004": "PDC-001",  # Existing product (excluding characteristic)
    "R-005": "PDC-001",  # Risk appetite (including characteristic)
    "R-006": "PDC-001",  # Investment experience (including characteristic)
    "R-007": "PDC-003",  # Surplus after cost (unsuitability signal)
    "R-008": "MON-001",  # Duplicate contact (monitoring)
    "R-009": "PDC-007",  # ML confidence (medium-materiality assumption)
    "R-010": "DC-001",   # Product in scope
    "R-011": "DEL-012",  # COBS 4 — fair, clear, not misleading
    "R-012": "PDC-001",  # FCA segment eligibility cross-check
}


@dataclass(frozen=True)
class _LegacyRuleDef:
    """
    Thin wrapper so existing tests that do RULES['R-001'].rule_type,
    RULES['R-001'].description, and RULES['R-001'].fca_ref continue to work
    during the v1 → v2 migration period.
    """
    rule_id:     str
    rule_type:   RuleType
    description: str
    fca_ref:     str

    @property
    def check_id(self) -> str:
        return _LEGACY_RULE_MAP.get(self.rule_id, self.rule_id)

    @property
    def severity(self) -> CheckSeverity:
        chk = COMPLIANCE_CHECKS.get(self.check_id)
        return chk.severity if chk else CheckSeverity.HARD_BLOCK


# Legacy RULES dict — preserved for backward compatibility
RULES: dict[str, _LegacyRuleDef] = {
    "R-001": _LegacyRuleDef("R-001", RuleType.HARD,
        "Consumer must match the target segment (all including characteristics "
        "met, no excluding characteristics triggered). COBS 9B.4; PDC-001.",
        "PS25/22-COBS9B.4-PDC-001"),
    "R-002": _LegacyRuleDef("R-002", RuleType.HARD,
        "Consumer age must meet product minimum age requirement (age band ≥ 1, "
        "i.e. aged ≥ 18). PS25/22 para 3.22; PDC-001.",
        "PS25/22-COBS9B.4-PDC-001"),
    "R-003": _LegacyRuleDef("R-003", RuleType.GATE,
        "Vulnerability flag must be reviewed before delivery — active vulnerability "
        "indicator triggers specialist journey routing. PDC-003; FCA FG21/1.",
        "PS25/22-para3.26-PDC-003-FG21/1"),
    "R-004": _LegacyRuleDef("R-004", RuleType.HARD,
        "Consumer must not already hold this product (excluding characteristic "
        "check). PDC-001; COBS 9B.4.",
        "PS25/22-COBS9B.4-PDC-001"),
    "R-005": _LegacyRuleDef("R-005", RuleType.HARD,
        "Risk appetite must be appropriate for product risk level (including "
        "characteristic check). PDC-001; COBS 9B.4.",
        "PS25/22-COBS9B.4-PDC-001"),
    "R-006": _LegacyRuleDef("R-006", RuleType.HARD,
        "Investment experience must meet product threshold (including "
        "characteristic check). PDC-001; COBS 9B.4.",
        "PS25/22-COBS9B.4-PDC-001"),
    "R-007": _LegacyRuleDef("R-007", RuleType.HARD,
        "Consumer monthly surplus must be positive after product cost — "
        "financial affordability check. PDC-003.",
        "PS25/22-COBS9B-PDC-003"),
    "R-008": _LegacyRuleDef("R-008", RuleType.SOFT,
        "No duplicate targeted support contact within last 30 days. "
        "Logged but non-blocking. MON-001.",
        "PS25/22-COBS9B.10-MON-001"),
    "R-009": _LegacyRuleDef("R-009", RuleType.GATE,
        "ML prediction confidence must exceed 0.75 for automated delivery — "
        "medium-materiality assumption consumer check. PDC-007.",
        "PS25/22-paras3.28-3.29-PDC-007"),
    "R-010": _LegacyRuleDef("R-010", RuleType.HARD,
        "Product must be currently available and within the PS25/22 in-scope "
        "product list. DC-001.",
        "PS25/22-paras3.45-3.48-DC-001"),
    "R-011": _LegacyRuleDef("R-011", RuleType.HARD,
        "Consumer Duty outcome assessment must pass — communication must be "
        "fair, clear, and not misleading. DEL-012; COBS 4.2.",
        "PS25/22-COBS4.2-PRIN2A-DEL-012"),
    "R-012": _LegacyRuleDef("R-012", RuleType.HARD,
        "FCA segment eligibility cross-check — consumer correctly aligned to "
        "pre-defined segment per COBS 9B.4 and PDC-001.",
        "PS25/22-COBS9B.4-PDC-001"),
}


# Safe consumer-facing reason messages — used by explainer.py
# Maps rule_id (legacy) to a FCA-approved consumer-facing explanation string.
CONSUMER_REASON_MAP: dict[str, str] = {
    "R-001": "your financial profile does not match the characteristics "
             "of customers in this group",
    "R-002": "the minimum age requirement for this service is not met",
    "R-003": "we would like one of our specialist advisers to assist you",
    "R-004": "you already hold a product of this type",
    "R-005": "the investment risk level of this product may not be suitable "
             "for your stated risk appetite",
    "R-006": "this product is designed for customers with some prior investment "
             "experience",
    "R-007": "the cost of this product may exceed what is comfortable given "
             "your current financial position",
    "R-008": "we have recently been in contact with you about a similar topic",
    "R-009": "we need a little more information before we can make a "
             "directional suggestion",
    "R-010": "this product is not currently available",
    "R-011": "we are unable to provide this suggestion at this time",
    "R-012": "you do not meet the eligibility criteria for this service",
    # v2 compliance check IDs
    "PDC-001": "your profile does not meet the criteria for this group of customers",
    "PDC-002": "we need to verify some information before we can proceed",
    "PDC-003": "based on information we hold, this suggestion may not be "
               "appropriate for you at this time",
    "PDC-007": "we need a little more information to confirm the right direction for you",
    "DC-001":  "this product type is outside the scope of our targeted support service",
    "DC-002":  "pension consolidation cannot be suggested through this service",
    "DC-003":  "we cannot recommend a specific annuity product through this service; "
               "we can direct you to an independent comparison tool",
    "DEL-006": "you will be directed to MoneyHelper for free impartial guidance",
    "DEL-012": "this suggestion cannot be made at this time",
}


# Backward-compatibility alias — old code imports RuleDef from this module.
# _LegacyRuleDef IS RuleDef; exposed under both names.
RuleDef = _LegacyRuleDef
