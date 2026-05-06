"""
ts_agent.visualiser.data_adapter
==================================
Enriches raw ``SessionRecord`` data with catalogue lookups so the visualiser
components receive fully human-readable, FCA-annotated data without having to
know about the domain catalogue themselves.

All functions are pure — no I/O, no mutation of the input record.

Codex review notes
------------------
- Every function accepts a ``SessionRecord`` (or a subset of its fields) and
  returns plain Python primitives / dicts / lists.
- Unknown IDs (rule_id, char_id, segment_id) always return safe defaults
  so the visualiser never raises a ``KeyError``.
- ``QUESTION_TEXT_MAP`` is the canonical source of human-readable question
  text for each ``char_id``.  In production this would live in the
  characteristics YAML; here it is a typed constant.
"""

from __future__ import annotations

from typing import Any

from ts_agent.config.segments import RULES, SEGMENTS, SITUATIONS, SUGGESTIONS
from ts_agent.observability.session_store import (
    ConversationTurn,
    PredictionSnapshot,
    SessionRecord,
    SignalEvent,
)

# ──────────────────────────────────────────────────────────────────────────────
# Question text map  (char_id → natural-language question shown to consumer)
# ──────────────────────────────────────────────────────────────────────────────

# v2 char_ids — PS25/22 ontology (live 6 April 2026).
# Covers all characteristics used in SEGMENTS, SITUATIONS, and SUGGESTIONS.
# This is the canonical source of human-readable question text for Zone 2 gap-fill.
QUESTION_TEXT_MAP: dict[str, str] = {
    # ── Personal ──────────────────────────────────────────────────────────────
    "CHAR-P1A-I1": "What is your age band? (1=18–29, 2=30–39, 3=40–54, 4=55–64, 5=65+)",
    "CHAR-P1B-I1": "Do you have any active vulnerability indicators on your account? (yes/no)",
    "CHAR-P1C-I1": "What is your employment status? (0=not employed, 1=employed, 2=self-employed, 3=retired)",
    # ── Personal (additional) ────────────────────────────────────────────────
    "CHAR-P1D-I1": "Do you own or rent your home? (OWNER or RENTER)",
    "CHAR-P1E-I1": "How many financial dependants do you have? (e.g. 0, 1, 2)",
    # ── Financial — investment domain ─────────────────────────────────────────
    "CHAR-F2A-I1": "Roughly how much do you have left each month after all your regular outgoings? (in £)",
    "CHAR-F2B-I1": "Approximately how much do you have in cash savings at the moment? (in £)",
    "CHAR-F2G-I1": "Do you have any active high-cost debt (overdraft over £1,000, payday loan, or BNPL arrears)? (yes/no)",
    "CHAR-F2H-I1": "Do you have any active financial difficulties such as a debt management plan or mortgage arrears? (yes/no)",
    "CHAR-F2I-I1": "Do you currently hold an investment product such as a Stocks and Shares ISA or investment fund? (yes/no)",
    "CHAR-F2J-I1": "Have you made a Stocks and Shares ISA contribution this tax year? (yes/no)",
    "CHAR-F2K-I1": "Is the current date within 90 days of the 5th of April? (yes/no)",
    "CHAR-F2L-I1": "How many months have you held an account with us?",
    "CHAR-F2M-I1": "Have you recently received or are you expecting a one-off lump sum? If yes, approximately how much? (in £, or 0 if no)",
    "CHAR-F2N-I1": "How many months has your investment account been inactive (no deposits, withdrawals or fund switches)?",
    "CHAR-F2O-I1": "How much do you save into a cash account each month on average? (in £)",
    "CHAR-F2P-I1": "How many consecutive months have you saved that amount?",
    "CHAR-F2Q-I1": "How many days until your fixed-rate or structured deposit matures?",
    "CHAR-F2R-I1": "Have you already placed a reinvestment instruction for your maturing deposit? (yes/no)",
    # ── Pension — accumulation ─────────────────────────────────────────────────
    "CHAR-P2A-I1": "What is your current combined pension contribution rate, including employer contributions? (as a percentage, e.g. 8 for 8%)",
    "CHAR-P2B-I1": "Are you currently an active member of a workplace or personal DC pension scheme? (yes/no)",
    "CHAR-P2C-I1": "Has your pension provider indicated that your projected retirement income may fall below target? (yes/no)",
    "CHAR-P2D-I1": "How many years do you have until your planned or state retirement age?",
    "CHAR-P2E-I1": "Is 100% of your pension invested in the scheme's default fund, with no changes made by you? (yes/no)",
    "CHAR-P2F-I1": "Have you ever made an active fund choice in your pension — for example, chosen a specific fund or risk level? (yes/no)",
    "CHAR-P2G-I1": "Have you made any changes to your pension funds or contributions in the last two years? (yes/no)",
    "CHAR-P2H-I1": "Have you recently experienced a significant financial life event such as a pay rise, paying off a debt, or your mortgage ending? (yes/no)",
    # ── Pension — decumulation ─────────────────────────────────────────────────
    "CHAR-P2I-I1": "Have you chosen how you would like to access your pension savings when you retire? (yes/no)",
    "CHAR-P2J-I1": "Are you currently drawing an income from your pension (for example through drawdown)? (yes/no)",
    "CHAR-P2K-I1": "Do you have a defined benefit (final salary) pension or any safeguarded pension benefits? (yes/no)",
    "CHAR-P2L-I1": "Approximately how much is in your pension pot at the moment? (in £)",
    "CHAR-P2M-I1": "Have you expressed an interest in, or asked about, annuities or guaranteed income in retirement? (yes/no)",
    "CHAR-P2N-I1": "How many months has it been since you last reviewed your drawdown strategy or made changes to your pension?",
    "CHAR-P2O-I1": "Does your pension account show a high proportion of cash (over 20%) or a withdrawal rate your provider has flagged as elevated? (yes/no)",
    "CHAR-P2P-I1": "Have you spoken to a regulated financial adviser about your pension in the last 12 months? (yes/no)",
    # ── Behavioural ───────────────────────────────────────────────────────────
    "CHAR-B3A-I1": "On a scale of 1 to 5, how comfortable are you with investment risk? (1=very low, 5=very high)",
    "CHAR-B3B-I1": "How would you describe your investment experience? (0=none, 1=basic, 2=some, 3=experienced)",
    "CHAR-B3C-I1": "Which channel are you using today? (0=mobile app, 1=web browser, 2=telephone)",
}

CHAR_BRANCH_MAP: dict[str, str] = {
    # Personal
    "CHAR-P1A-I1": "Personal",  "CHAR-P1B-I1": "Personal",
    "CHAR-P1C-I1": "Personal",
    "CHAR-P1D-I1": "Personal",  "CHAR-P1E-I1": "Personal",
    # Financial
    "CHAR-F2A-I1": "Financial", "CHAR-F2B-I1": "Financial",
    "CHAR-F2G-I1": "Financial", "CHAR-F2H-I1": "Financial",
    "CHAR-F2I-I1": "Financial", "CHAR-F2J-I1": "Financial",
    "CHAR-F2K-I1": "Temporal",  "CHAR-F2L-I1": "Financial",
    "CHAR-F2M-I1": "Financial", "CHAR-F2N-I1": "Financial",
    "CHAR-F2O-I1": "Financial", "CHAR-F2P-I1": "Temporal",
    "CHAR-F2Q-I1": "Temporal",  "CHAR-F2R-I1": "Financial",
    # Pension — accumulation
    "CHAR-P2A-I1": "Pension",   "CHAR-P2B-I1": "Pension",
    "CHAR-P2C-I1": "Pension",   "CHAR-P2D-I1": "Pension",
    "CHAR-P2E-I1": "Pension",   "CHAR-P2F-I1": "Pension",
    "CHAR-P2G-I1": "Pension",   "CHAR-P2H-I1": "Pension",
    # Pension — decumulation
    "CHAR-P2I-I1": "Pension",   "CHAR-P2J-I1": "Pension",
    "CHAR-P2K-I1": "Pension",   "CHAR-P2L-I1": "Pension",
    "CHAR-P2M-I1": "Pension",   "CHAR-P2N-I1": "Pension",
    "CHAR-P2O-I1": "Pension",   "CHAR-P2P-I1": "Pension",
    # Behavioural
    "CHAR-B3A-I1": "Behavioural", "CHAR-B3B-I1": "Behavioural",
    "CHAR-B3C-I1": "Behavioural",
}

CHAR_SHORT_LABEL: dict[str, str] = {
    # Personal
    "CHAR-P1A-I1": "Age Band",          "CHAR-P1B-I1": "Vulnerability",
    "CHAR-P1C-I1": "Employment",
    "CHAR-P1D-I1": "Tenure",          "CHAR-P1E-I1": "Dependants",
    # Financial — investment
    "CHAR-F2A-I1": "Monthly Surplus £", "CHAR-F2B-I1": "Savings Balance £",
    "CHAR-F2G-I1": "High-Cost Debt",    "CHAR-F2H-I1": "Financial Hardship",
    "CHAR-F2I-I1": "Holds Investment",  "CHAR-F2J-I1": "ISA Sub This Year",
    "CHAR-F2K-I1": "Tax Year Window",   "CHAR-F2L-I1": "Account Tenure (mo)",
    "CHAR-F2M-I1": "Lump Sum £",        "CHAR-F2N-I1": "Months Inactive",
    "CHAR-F2O-I1": "Regular Saving £",  "CHAR-F2P-I1": "Consecutive Months",
    "CHAR-F2Q-I1": "Days to Maturity",  "CHAR-F2R-I1": "Reinvest Instruction",
    # Pension — accumulation
    "CHAR-P2A-I1": "Contribution %",    "CHAR-P2B-I1": "Active DC Member",
    "CHAR-P2C-I1": "Shortfall Flag",    "CHAR-P2D-I1": "Yrs to Retirement",
    "CHAR-P2E-I1": "100% Default Fund", "CHAR-P2F-I1": "No Active Selection",
    "CHAR-P2G-I1": "Recent Engagement", "CHAR-P2H-I1": "Life Event Signal",
    # Pension — decumulation
    "CHAR-P2I-I1": "Access Plan",       "CHAR-P2J-I1": "In Drawdown",
    "CHAR-P2K-I1": "DB/Safeguarded",    "CHAR-P2L-I1": "Pot Value £",
    "CHAR-P2M-I1": "Annuity Interest",  "CHAR-P2N-I1": "Months Since Review",
    "CHAR-P2O-I1": "Cash Drag Flag",    "CHAR-P2P-I1": "Recent Adviser",
    # Behavioural
    "CHAR-B3A-I1": "Risk Appetite",     "CHAR-B3B-I1": "Invest. Experience",
    "CHAR-B3C-I1": "Channel",
}


# ──────────────────────────────────────────────────────────────────────────────
# DataAdapter — all pure enrichment functions
# ──────────────────────────────────────────────────────────────────────────────

class DataAdapter:
    """
    Stateless enrichment layer between ``SessionRecord`` and visualiser
    components.

    All methods are ``@staticmethod`` so they can be called without an
    instance; the class is used purely for namespace organisation.
    """

    # ── Session overview ──────────────────────────────────────────────────────

    @staticmethod
    def session_summary(record: SessionRecord) -> dict[str, Any]:
        """Return a flat dict of display-ready session metadata."""
        sit  = SITUATIONS.get(record.situation_id)
        seg  = SEGMENTS.get(record.matched_segment_id)

        # Look up top suggestion for this segment
        top_sugg = None
        if seg:
            from ts_agent.config.segments import SEGMENT_TO_SUGGESTIONS  # noqa: PLC0415
            candidates = SEGMENT_TO_SUGGESTIONS.get(seg.segment_id, [])
            top_sugg   = candidates[0] if candidates else None

        return {
            "session_id":         record.session_id,
            "party_ref":          record.party_ref or "Unknown",
            "intent_id":          record.intent_id or "Unknown",
            "situation_id":       record.situation_id or "Unknown",
            "situation_label":    sit.label if sit else record.situation_id,
            "channel":            record.channel or "mobile",
            "matched_segment":    seg.segment_id if seg else "—",
            "segment_label":      seg.label if seg else "No match",
            "suggestion_id":      top_sugg.suggestion_id if top_sugg else "—",
            "suggestion_name":    top_sugg.product_name if top_sugg else "—",
            "gate_disposition":   record.gate_disposition or "SUPPRESS",
            "audit_confirmed":    record.audit_confirmed,
            "known_traits":       record.known_trait_count,
            "missing_traits":     record.missing_trait_count,
            "excluded_traits":    record.excluded_trait_count,
            "total_traits":       (
                record.known_trait_count
                + record.missing_trait_count
                + record.excluded_trait_count
            ),
            "completeness_pct":   DataAdapter.completeness_pct(record),
            "gap_fill_turns":     record.gap_fill_turns,
            "fill_strategy":      record.fill_strategy or "STATIC",
            "signal_count":       record.signal_count(),
            "error_count":        record.error_count(),
            "has_error":          record.has_error,
            "is_complete":        record.is_complete,
            "started_at":         record.started_at,
            "total_ms":           round(record.total_ms, 1),
            "duration_s":         round(record.duration_seconds(), 2),
        }

    @staticmethod
    def completeness_pct(record: SessionRecord) -> float:
        total = (
            record.known_trait_count
            + record.missing_trait_count
            + record.excluded_trait_count
        )
        eligible = record.known_trait_count + record.missing_trait_count
        if eligible == 0:
            return 100.0
        return round(record.known_trait_count / eligible * 100, 1)

    # ── Conversation panel ────────────────────────────────────────────────────

    @staticmethod
    def enrich_conversation(
        turns: list[ConversationTurn],
    ) -> list[dict[str, Any]]:
        """Return conversation turns with human-readable question text."""
        return [
            {
                "turn":          t.turn_number,
                "char_id":       t.char_id,
                "branch":        CHAR_BRANCH_MAP.get(t.char_id, "Unknown"),
                "short_label":   CHAR_SHORT_LABEL.get(t.char_id, t.char_id),
                "question_text": QUESTION_TEXT_MAP.get(
                    t.char_id, f"Question for {t.char_id}"
                ),
                "value_hash":    t.value_hash or "—",
                "source":        t.source,
                "elapsed_ms":    round(t.elapsed_ms, 1),
            }
            for t in turns
        ]

    # ── Trace waterfall ───────────────────────────────────────────────────────

    @staticmethod
    def enrich_signals(signals: list[SignalEvent]) -> list[dict[str, Any]]:
        """Return signals annotated with display colour and zone order."""
        zone_order = {
            "Zone1": 1, "Zone1.5": 2, "Zone2": 3,
            "Zone3": 4, "Zone4": 5, "Session": 6, "Infra": 7,
        }
        level_colour = {
            "INFO": "#2ECC71", "WARN": "#F39C12", "ERROR": "#E74C3C",
        }
        return [
            {
                "signal":      s.signal,
                "level":       s.level,
                "zone":        s.zone,
                "zone_order":  zone_order.get(s.zone, 9),
                "elapsed_ms":  round(s.elapsed_ms, 1),
                "colour":      level_colour.get(s.level, "#95A5A6"),
                "attributes":  s.attributes,
                "session_id":  s.session_id,
                "timestamp":   s.timestamp_utc,
            }
            for s in sorted(signals, key=lambda x: x.elapsed_ms)
        ]

    # ── Deterministic rule panel ──────────────────────────────────────────────

    @staticmethod
    def enrich_rules(
        rule_evals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Enrich raw rule_evaluation dicts with catalogue metadata."""
        enriched = []
        seen: set[str] = set()
        for ev in rule_evals:
            rule_id = ev.get("rule_id", "")
            if not rule_id or rule_id in seen:
                continue
            seen.add(rule_id)
            rule_def = RULES.get(rule_id)
            enriched.append({
                "rule_id":     rule_id,
                "rule_type":   ev.get("rule_type", rule_def.rule_type if rule_def else "?"),
                "description": rule_def.description if rule_def else rule_id,
                "fca_ref":     rule_def.fca_ref     if rule_def else "—",
                "outcome":     ev.get("outcome", "—"),
                "suggestion_id": ev.get("suggestion_id", ""),
            })
        # Fill in any missing rules (not evaluated = not reached)
        evaluated_ids = {e["rule_id"] for e in enriched}
        for rule_id, rule_def in RULES.items():
            if rule_id not in evaluated_ids:
                enriched.append({
                    "rule_id":     rule_id,
                    "rule_type":   rule_def.rule_type,
                    "description": rule_def.description,
                    "fca_ref":     rule_def.fca_ref or "—",
                    "outcome":     "NOT_REACHED",
                    "suggestion_id": "",
                })
        return sorted(enriched, key=lambda x: x["rule_id"])

    # ── ML prediction panel ───────────────────────────────────────────────────

    @staticmethod
    def enrich_predictions(
        chain: list[PredictionSnapshot],
    ) -> list[dict[str, Any]]:
        """Return prediction chain with segment labels."""
        result = []
        for snap in chain:
            seg = SEGMENTS.get(snap.top_segment_id or "")
            result.append({
                "turn":            snap.turn,
                "top_segment_id":  snap.top_segment_id or "—",
                "segment_label":   seg.label if seg else snap.top_segment_id or "—",
                "top_confidence":  round(snap.top_confidence, 4),
                "model_version":   snap.model_version,
                "model_algorithm": snap.model_algorithm,
                "known_count":     snap.known_trait_count,
                "disposition":     snap.disposition,
                "shap_json":       snap.shap_features_json,
                "elapsed_ms":      round(snap.elapsed_ms, 1),
            })
        return result

    # ── Sankey data ───────────────────────────────────────────────────────────

    @staticmethod
    def sankey_for_session(record: SessionRecord) -> dict[str, Any]:
        """
        Build node/link data for a single-session Sankey diagram.
        Returns a dict with ``nodes`` (list of labels) and ``links``
        (source, target, value=1, colour).
        """
        sit  = SITUATIONS.get(record.situation_id)
        seg  = SEGMENTS.get(record.matched_segment_id)

        from ts_agent.config.segments import SEGMENT_TO_SUGGESTIONS  # noqa: PLC0415
        sugg_candidates = SEGMENT_TO_SUGGESTIONS.get(
            record.matched_segment_id, []
        )
        sugg = sugg_candidates[0] if sugg_candidates else None

        labels = [
            record.intent_id or "Intent",
            sit.label if sit else record.situation_id or "Situation",
            seg.label if seg else (record.matched_segment_id or "No Segment"),
            sugg.product_name if sugg else "No Suggestion",
            record.gate_disposition or "SUPPRESS",
        ]
        gate_colour = {
            "EMIT":         "rgba(46, 204, 113, 0.6)",
            "HUMAN_REVIEW": "rgba(243, 156, 18, 0.6)",
            "SUPPRESS":     "rgba(231, 76, 60, 0.6)",
        }.get(record.gate_disposition, "rgba(149, 165, 166, 0.6)")

        links = [
            {"source": 0, "target": 1, "value": 1, "colour": "rgba(52,152,219,0.4)"},
            {"source": 1, "target": 2, "value": 1, "colour": "rgba(52,152,219,0.4)"},
            {"source": 2, "target": 3, "value": 1, "colour": "rgba(52,152,219,0.4)"},
            {"source": 3, "target": 4, "value": 1, "colour": gate_colour},
        ]
        return {"labels": labels, "links": links}

    @staticmethod
    def sankey_aggregate(records: list[SessionRecord]) -> dict[str, Any]:
        """
        Build aggregate Sankey data across all sessions showing
        population flow through the pipeline.
        """
        from collections import defaultdict  # noqa: PLC0415

        # Count flows between consecutive pipeline stages
        flow_counts: dict[tuple[str, str], int] = defaultdict(int)

        for r in records:
            sit    = SITUATIONS.get(r.situation_id)
            seg    = SEGMENTS.get(r.matched_segment_id)
            gate   = r.gate_disposition or "SUPPRESS"
            intent = r.intent_id or "Unknown Intent"
            sit_l  = sit.label if sit else (r.situation_id or "Unknown Situation")
            seg_l  = seg.label if seg else "No Segment Matched"

            from ts_agent.config.segments import SEGMENT_TO_SUGGESTIONS  # noqa: PLC0415
            sugg_list = SEGMENT_TO_SUGGESTIONS.get(r.matched_segment_id, [])
            sugg_l = sugg_list[0].product_name if sugg_list else "No Suggestion"

            flow_counts[(intent, sit_l)]   += 1
            flow_counts[(sit_l,  seg_l)]   += 1
            flow_counts[(seg_l,  sugg_l)]  += 1
            flow_counts[(sugg_l, gate)]    += 1

        # Build unique ordered node list
        all_nodes: list[str] = []
        for src, tgt in flow_counts:
            if src not in all_nodes:
                all_nodes.append(src)
            if tgt not in all_nodes:
                all_nodes.append(tgt)
        node_idx = {n: i for i, n in enumerate(all_nodes)}

        gate_colours = {
            "EMIT":         "rgba(46, 204, 113, 0.5)",
            "HUMAN_REVIEW": "rgba(243, 156, 18, 0.5)",
            "SUPPRESS":     "rgba(231, 76, 60, 0.5)",
        }
        links = []
        for (src, tgt), count in flow_counts.items():
            colour = gate_colours.get(tgt, "rgba(52,152,219,0.3)")
            links.append({
                "source": node_idx[src],
                "target": node_idx[tgt],
                "value":  count,
                "colour": colour,
            })

        return {"labels": all_nodes, "links": links}


def build_demo_store():
    """
    Build a demo session store with sample data for visualiser testing.
    This creates a store with multiple sessions across different scenarios.
    """
    from ts_agent.observability.session_store import SessionStore, SessionRecord
    from datetime import datetime, timedelta
    import uuid
    
    store = SessionStore()
    
    # Create demo sessions for different scenarios
    base_time = datetime.utcnow() - timedelta(days=1)
    
    demo_scenarios = [
        ("SIT-INV-001", "consumer-001", "INT-INVEST", "SEG-INV-1", 8, 2),
        ("SIT-INV-002", "consumer-002", "INT-INVEST", "SEG-INV-2", 6, 4),
        ("SIT-PEN-001", "consumer-003", "INT-PENSION", "SEG-PEN-1", 7, 3),
        ("SIT-PEN-002", "consumer-004", "INT-PENSION", "SEG-PEN-2", 9, 1),
        ("SIT-DEC-001", "consumer-005", "INT-DRAWDOWN", "SEG-DEC-1", 5, 2),
    ]
    
    for i, (sit_id, party_ref, intent, seg_id, known, missing) in enumerate(demo_scenarios):
        session_id = str(uuid.uuid4())
        
        record = SessionRecord(
            session_id=session_id,
            party_ref=party_ref,
            intent_id=intent,
            situation_id=sit_id,
            channel="web",
            matched_segment_id=seg_id,
            gate_disposition="EMIT" if i % 3 == 0 else "SUPPRESS",
            audit_confirmed=True,
            known_trait_count=known,
            missing_trait_count=missing,
            excluded_trait_count=0,
            gap_fill_turns=missing,
            fill_strategy="ML_AUTO",
            has_error=False,
            is_complete=True,
            started_at=base_time + timedelta(hours=i),
            total_ms=1500 + (i * 200),
        )
        
        store.add_session(record)
    
    return store
