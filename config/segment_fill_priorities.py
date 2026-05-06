"""
ts_agent.config.segment_fill_priorities
========================================
Configuration defining question priorities for gap-filling per segment.

This configuration serves multiple purposes:
1. Guides ML-automatic question ordering when SHAP data is unavailable
2. Provides human-readable documentation of key traits per segment
3. Enables fine-tuning of question priorities per segment
4. Supports regulatory transparency (which questions matter most for each segment)

Structure:
----------
SEGMENT_FILL_PRIORITIES = {
    "segment_id": {
        "priority_traits": [list of char_ids in priority order],
        "rationale": "Why these traits matter for this segment",
        "min_required": int,  # Minimum traits needed for confident match
    }
}
"""
from typing import TypedDict

class SegmentFillPriority(TypedDict):
    """Priority configuration for a segment's gap-fill process."""
    priority_traits: list[str]  # char_ids in descending priority
    rationale: str
    min_required: int


# ══════════════════════════════════════════════════════════════════════════════
# INVESTMENT SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

SEG_INV_001_PRIORITY: SegmentFillPriority = {
    "priority_traits": [
        "CHAR-F2B-I1",  # Cash/Deposits (discriminator for this segment)
        "CHAR-F2A-I1",  # Monthly Surplus
        "CHAR-F2I-I1",  # Current Investments
        "CHAR-P2A-I1",  # Age
        "CHAR-P2B-I1",  # Risk Tolerance
    ],
    "rationale": "Cash Savings Investment: Need cash amount to determine eligibility",
    "min_required": 3,
}

SEG_INV_002_PRIORITY: SegmentFillPriority = {
    "priority_traits": [
        "CHAR-F2I-I1",  # Current Investments (key discriminator: should be False)
        "CHAR-F2A-I1",  # Monthly Surplus
        "CHAR-F2B-I1",  # Cash/Deposits
        "CHAR-P2A-I1",  # Age
        "CHAR-P2B-I1",  # Risk Tolerance
    ],
    "rationale": "First-time Investor: Key is no existing investments",
    "min_required": 3,
}

SEG_INV_003_PRIORITY: SegmentFillPriority = {
    "priority_traits": [
        "CHAR-F2A-I1",  # Monthly Surplus
        "CHAR-F2I-I1",  # Current Investments
        "CHAR-F2J-I1",  # ISA Allowance Used
        "CHAR-P2A-I1",  # Age
        "CHAR-P2B-I1",  # Risk Tolerance
    ],
    "rationale": "ISA Allowance: Focus on income and ISA utilization",
    "min_required": 3,
}

SEG_INV_004_PRIORITY: SegmentFillPriority = {
    "priority_traits": [
        "CHAR-F2M-I1",  # Lump Sum Amount (critical for this segment)
        "CHAR-F2I-I1",  # Current Investments
        "CHAR-P2A-I1",  # Age
        "CHAR-P2B-I1",  # Risk Tolerance
        "CHAR-F2A-I1",  # Monthly Surplus
    ],
    "rationale": "Lump Sum Investment: Lump sum amount is the key discriminator",
    "min_required": 3,
}

# ══════════════════════════════════════════════════════════════════════════════
# SAVINGS & DEPOSITS SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

SEG_SD_001_PRIORITY: SegmentFillPriority = {
    "priority_traits": [
        "CHAR-F2B-I1",  # Cash/Deposits (must be within range)
        "CHAR-F2A-I1",  # Monthly Surplus
        "CHAR-F2K-I1",  # Savings Goal
        "CHAR-P2C-I1",  # Time Horizon
    ],
    "rationale": "Regular Saver: Focus on cash position and savings behavior",
    "min_required": 3,
}

SEG_SD_002_PRIORITY: SegmentFillPriority = {
    "priority_traits": [
        "CHAR-F2B-I1",  # Cash/Deposits
        "CHAR-F2K-I1",  # Savings Goal
        "CHAR-P2C-I1",  # Time Horizon
        "CHAR-F2A-I1",  # Monthly Surplus
    ],
    "rationale": "Emergency Fund Builder: Cash and goal clarity are key",
    "min_required": 2,
}

# ══════════════════════════════════════════════════════════════════════════════
# PENSION/RETIREMENT SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

SEG_DEC_001_PRIORITY: SegmentFillPriority = {
    "priority_traits": [
        "CHAR-P2A-I1",  # Age (must be approaching retirement)
        "CHAR-P2L-I1",  # DC Pot Value
        "CHAR-P2J-I1",  # Already in Drawdown (should be False)
        "CHAR-F2A-I1",  # Monthly Surplus
        "CHAR-P2B-I1",  # Risk Tolerance
    ],
    "rationale": "Pre-retirement Decumulation: Age and pot value critical",
    "min_required": 3,
}

SEG_DEC_002_PRIORITY: SegmentFillPriority = {
    "priority_traits": [
        "CHAR-P2L-I1",  # DC Pot Value (must be within range)
        "CHAR-P2A-I1",  # Age
        "CHAR-P2M-I1",  # Retirement Income Need
        "CHAR-P2J-I1",  # Already in Drawdown
        "CHAR-P2B-I1",  # Risk Tolerance
    ],
    "rationale": "Small Pot Consolidation: Pot size is key discriminator",
    "min_required": 3,
}

SEG_DEC_003_PRIORITY: SegmentFillPriority = {
    "priority_traits": [
        "CHAR-P2K-I1",  # DB Transfer Applicable
        "CHAR-P2A-I1",  # Age
        "CHAR-P2N-I1",  # DB Annual Income
        "CHAR-P2O-I1",  # Has Taken Advice
        "CHAR-F2A-I1",  # Monthly Surplus
    ],
    "rationale": "DB Transfer Consideration: Regulatory-sensitive, need advice status",
    "min_required": 4,
}

# ══════════════════════════════════════════════════════════════════════════════
# MASTER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SEGMENT_FILL_PRIORITIES: dict[str, SegmentFillPriority] = {
    # Investment segments
    "SEG-INV-001": SEG_INV_001_PRIORITY,
    "SEG-INV-002": SEG_INV_002_PRIORITY,
    "SEG-INV-003": SEG_INV_003_PRIORITY,
    "SEG-INV-004": SEG_INV_004_PRIORITY,
    
    # Savings & Deposits
    "SEG-SD-001": SEG_SD_001_PRIORITY,
    "SEG-SD-002": SEG_SD_002_PRIORITY,
    
    # Decumulation/Retirement
    "SEG-DEC-001": SEG_DEC_001_PRIORITY,
    "SEG-DEC-002": SEG_DEC_002_PRIORITY,
    "SEG-DEC-003": SEG_DEC_003_PRIORITY,
}


def get_priority_traits_for_segment(segment_id: str) -> list[str]:
    """
    Get the priority-ordered list of trait char_ids for a segment.
    
    Args:
        segment_id: Segment identifier (e.g., "SEG-INV-002")
    
    Returns:
        List of char_ids in descending priority order.
        Returns empty list if segment not found.
    """
    config = SEGMENT_FILL_PRIORITIES.get(segment_id)
    if not config:
        return []
    return config["priority_traits"]


def get_min_required_traits(segment_id: str) -> int:
    """
    Get the minimum number of traits required for confident segment match.
    
    Args:
        segment_id: Segment identifier
    
    Returns:
        Minimum trait count, or 3 as default.
    """
    config = SEGMENT_FILL_PRIORITIES.get(segment_id)
    if not config:
        return 3  # Default minimum
    return config["min_required"]


def get_segment_rationale(segment_id: str) -> str:
    """
    Get the rationale for why these traits are prioritized for this segment.
    
    Useful for explainability and regulatory documentation.
    
    Args:
        segment_id: Segment identifier
    
    Returns:
        Rationale string, or empty if not found.
    """
    config = SEGMENT_FILL_PRIORITIES.get(segment_id)
    if not config:
        return ""
    return config["rationale"]


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-SEGMENT TRAIT IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

# Global trait importance scores (used when no segment hypothesis available)
GLOBAL_TRAIT_IMPORTANCE = {
    "CHAR-F2A-I1": 10,  # Monthly Surplus - universally important
    "CHAR-F2B-I1": 9,   # Cash/Deposits - key discriminator
    "CHAR-P2A-I1": 8,   # Age - critical for many segments
    "CHAR-F2I-I1": 8,   # Current Investments - major branch point
    "CHAR-P2L-I1": 7,   # DC Pot Value - pension segments
    "CHAR-P2B-I1": 6,   # Risk Tolerance - investment segments
    "CHAR-F2M-I1": 7,   # Lump Sum - specific but important
    "CHAR-F2K-I1": 5,   # Savings Goal
    "CHAR-P2C-I1": 5,   # Time Horizon
    "CHAR-P2J-I1": 6,   # In Drawdown - pension discriminator
    "CHAR-P2K-I1": 8,   # DB Transfer - regulatory critical
    "CHAR-F2J-I1": 4,   # ISA Allowance Used
    "CHAR-P2M-I1": 5,   # Retirement Income Need
    "CHAR-P2N-I1": 6,   # DB Annual Income
    "CHAR-P2O-I1": 7,   # Has Taken Advice - regulatory
}


def get_global_trait_priority(char_id: str) -> int:
    """
    Get the global importance score for a trait (1-10 scale).
    
    Used as fallback when no segment-specific priorities available.
    
    Args:
        char_id: Characteristic identifier
    
    Returns:
        Importance score (1-10), or 5 as default.
    """
    return GLOBAL_TRAIT_IMPORTANCE.get(char_id, 5)
