"""
Elite Traits Module

Position-specific elite thresholds based on NFL Combine/Pro Day data.
Top 10% measurables get meaningful bonuses; average gets no special treatment.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


# Top 10% thresholds by position (based on NFL Combine historical data)
# These are "elite" thresholds - exceeding them indicates exceptional athleticism
ELITE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # Quarterbacks - size + mobility
    "QB": {
        "height": 76,      # 6'4" - elite QB height
        "weight": 225,     # Strong frame
        "forty": 4.65,     # Mobile QB threshold
        "ras": 80,         # Relative Athletic Score
    },
    # Wide Receivers - speed + explosiveness
    "WR": {
        "height": 74,      # 6'2" - elite size
        "weight": 205,     # Optimal build
        "forty": 4.40,     # Elite burner speed
        "ras": 85,         # High bar for WR athleticism
        "vertical": 38,    # Jumping ability
        "broad_jump": 128, # Explosion
    },
    # Running Backs - burst + contact balance
    "RB": {
        "height": 71,      # 5'11" - ideal RB size
        "weight": 215,     # Power back territory
        "forty": 4.45,     # Home run speed
        "ras": 82,
        "vertical": 36,
        "bench": 22,       # Upper body strength
    },
    # Cornerbacks - speed + agility
    "CB": {
        "height": 72,      # 6'0" - modern CB size
        "weight": 195,     # Lean and fast
        "forty": 4.42,     # Elite cover speed
        "ras": 85,
        "vertical": 38,
        "three_cone": 6.85, # Agility
    },
    # Edge Rushers - size + explosion
    "EDGE": {
        "height": 76,      # 6'4" - ideal edge length
        "weight": 260,     # Pass rush power
        "forty": 4.65,     # Speed-to-power
        "ras": 80,
        "vertical": 35,
        "arm_length": 34,  # Reach
    },
    # Defensive End (interior)
    "DE": {
        "height": 76,
        "weight": 275,
        "forty": 4.80,
        "ras": 78,
    },
    # Defensive Tackle
    "DT": {
        "height": 75,      # 6'3"
        "weight": 305,     # Anchor size
        "forty": 5.00,     # Interior speed
        "ras": 75,
        "bench": 28,       # Strength at POA
    },
    # Interior Defensive Line (generic)
    "DL": {
        "height": 76,
        "weight": 295,
        "forty": 4.90,
        "ras": 78,
    },
    # Offensive Tackle
    "OT": {
        "height": 78,      # 6'6" - ideal tackle height
        "weight": 315,     # Elite size
        "forty": 5.10,     # Athleticism for position
        "ras": 75,
        "arm_length": 34,
    },
    # Interior Offensive Line
    "OG": {
        "height": 76,
        "weight": 315,
        "forty": 5.15,
        "ras": 72,
        "bench": 28,
    },
    "C": {
        "height": 75,
        "weight": 305,
        "forty": 5.15,
        "ras": 70,
    },
    # Generic OL
    "OL": {
        "height": 77,
        "weight": 310,
        "forty": 5.15,
        "ras": 73,
    },
    # Linebackers
    "LB": {
        "height": 74,      # 6'2"
        "weight": 235,
        "forty": 4.55,     # Sideline-to-sideline
        "ras": 82,
        "vertical": 36,
        "three_cone": 7.00,
    },
    "ILB": {
        "height": 74,
        "weight": 235,
        "forty": 4.58,
        "ras": 80,
    },
    "OLB": {
        "height": 75,
        "weight": 245,
        "forty": 4.60,
        "ras": 80,
    },
    # Safeties
    "S": {
        "height": 73,      # 6'1"
        "weight": 210,
        "forty": 4.48,
        "ras": 83,
        "vertical": 37,
    },
    "FS": {
        "height": 73,
        "weight": 205,
        "forty": 4.45,
        "ras": 85,
    },
    "SS": {
        "height": 73,
        "weight": 215,
        "forty": 4.50,
        "ras": 82,
    },
    # Tight End
    "TE": {
        "height": 77,      # 6'5" - modern TE size
        "weight": 250,
        "forty": 4.65,     # Athletic TE
        "ras": 80,
        "vertical": 34,
    },
    # Fullback
    "FB": {
        "height": 72,
        "weight": 245,
        "forty": 4.75,
        "ras": 70,
    },
    # Kicker/Punter (less relevant but included)
    "K": {
        "height": 72,
        "weight": 195,
        "ras": 50,  # Athleticism less critical
    },
    "P": {
        "height": 74,
        "weight": 210,
        "ras": 55,
    },
}

# Traits where lower is better
INVERSE_TRAITS = {"forty", "three_cone", "shuttle"}


def normalize_position(position: str) -> str:
    """
    Normalize position abbreviations to standard format.

    Args:
        position: Raw position string

    Returns:
        Normalized position
    """
    if not position:
        return "UNKNOWN"

    pos = position.upper().strip()

    # Common mappings
    mappings = {
        "QUARTERBACK": "QB",
        "RUNNING BACK": "RB",
        "HALFBACK": "RB",
        "TAILBACK": "RB",
        "WIDE RECEIVER": "WR",
        "TIGHT END": "TE",
        "OFFENSIVE TACKLE": "OT",
        "OFFENSIVE GUARD": "OG",
        "CENTER": "C",
        "OFFENSIVE LINE": "OL",
        "OFFENSIVE LINEMAN": "OL",
        "DEFENSIVE END": "DE",
        "DEFENSIVE TACKLE": "DT",
        "NOSE TACKLE": "DT",
        "DEFENSIVE LINE": "DL",
        "DEFENSIVE LINEMAN": "DL",
        "EDGE RUSHER": "EDGE",
        "OUTSIDE LINEBACKER": "OLB",
        "INSIDE LINEBACKER": "ILB",
        "MIDDLE LINEBACKER": "ILB",
        "LINEBACKER": "LB",
        "CORNERBACK": "CB",
        "SAFETY": "S",
        "FREE SAFETY": "FS",
        "STRONG SAFETY": "SS",
        "KICKER": "K",
        "PUNTER": "P",
        "FULLBACK": "FB",
        "ATHLETE": "ATH",
    }

    return mappings.get(pos, pos)


def get_elite_traits(player: Dict[str, Any], position: str) -> List[str]:
    """
    Identify which traits are elite for a player.

    Args:
        player: Player data dict with measurables
        position: Player position

    Returns:
        List of trait names that meet elite thresholds
    """
    pos = normalize_position(position)
    thresholds = ELITE_THRESHOLDS.get(pos, {})

    if not thresholds:
        # Fall back to generic position groups
        if pos in ["DE", "DT", "NT"]:
            thresholds = ELITE_THRESHOLDS.get("DL", {})
        elif pos in ["OT", "OG", "C"]:
            thresholds = ELITE_THRESHOLDS.get("OL", {})
        elif pos in ["ILB", "OLB", "MLB"]:
            thresholds = ELITE_THRESHOLDS.get("LB", {})
        elif pos in ["FS", "SS"]:
            thresholds = ELITE_THRESHOLDS.get("S", {})

    elite_traits = []

    for trait, elite_val in thresholds.items():
        player_val = player.get(trait)

        if player_val is None:
            continue

        try:
            player_val = float(player_val)
        except (ValueError, TypeError):
            continue

        # For inverse traits (forty, three_cone), lower is better
        if trait in INVERSE_TRAITS:
            if player_val <= elite_val:
                elite_traits.append(trait)
        else:
            if player_val >= elite_val:
                elite_traits.append(trait)

    return elite_traits


def calculate_elite_bonus(player: Dict[str, Any], position: str) -> float:
    """
    Calculate NIL/WAR/Draft multiplier based on elite traits.

    Returns 1.0 (no bonus) for average players.
    Returns up to 1.25 (+25%) for players with multiple elite traits.

    Args:
        player: Player data dict with measurables
        position: Player position

    Returns:
        Multiplier (1.0 = no bonus, 1.25 = 25% boost)
    """
    pos = normalize_position(position)
    thresholds = ELITE_THRESHOLDS.get(pos, {})

    if not thresholds:
        # Fall back to generic position groups
        if pos in ["DE", "DT", "NT"]:
            thresholds = ELITE_THRESHOLDS.get("DL", {})
        elif pos in ["OT", "OG", "C"]:
            thresholds = ELITE_THRESHOLDS.get("OL", {})
        elif pos in ["ILB", "OLB", "MLB"]:
            thresholds = ELITE_THRESHOLDS.get("LB", {})
        elif pos in ["FS", "SS"]:
            thresholds = ELITE_THRESHOLDS.get("S", {})

    if not thresholds:
        return 1.0

    elite_traits = 0
    measured_traits = 0

    for trait, elite_val in thresholds.items():
        player_val = player.get(trait)

        if player_val is None:
            continue

        try:
            player_val = float(player_val)
        except (ValueError, TypeError):
            continue

        measured_traits += 1

        # For inverse traits (forty, three_cone), lower is better
        if trait in INVERSE_TRAITS:
            if player_val <= elite_val:
                elite_traits += 1
        else:
            if player_val >= elite_val:
                elite_traits += 1

    if measured_traits == 0:
        return 1.0

    elite_ratio = elite_traits / measured_traits

    # Tiered bonuses - only top performers get meaningful boosts
    if elite_ratio >= 0.80:   # 4/5+ traits elite (top 10%)
        return 1.25           # +25% bonus
    elif elite_ratio >= 0.60: # 3/5 traits elite (top 20%)
        return 1.15           # +15% bonus
    elif elite_ratio >= 0.40: # 2/5 traits elite (top 35%)
        return 1.08           # +8% bonus
    else:
        return 1.0            # No bonus - average or below


def calculate_draft_adjustment(player: Dict[str, Any], position: str) -> int:
    """
    Calculate draft pick adjustment based on elite traits.

    Negative values mean moving UP in the draft (better).

    Args:
        player: Player data dict with measurables
        position: Player position

    Returns:
        Draft pick adjustment (-30 to 0)
    """
    elite_bonus = calculate_elite_bonus(player, position)

    if elite_bonus >= 1.25:
        return -30  # Move up 30 picks (elite athlete)
    elif elite_bonus >= 1.15:
        return -18  # Move up 18 picks
    elif elite_bonus >= 1.08:
        return -8   # Move up 8 picks
    else:
        return 0    # No adjustment


def get_athletic_profile(player: Dict[str, Any], position: str) -> Dict[str, Any]:
    """
    Generate a comprehensive athletic profile for a player.

    Args:
        player: Player data dict with measurables
        position: Player position

    Returns:
        Athletic profile dict
    """
    pos = normalize_position(position)
    thresholds = ELITE_THRESHOLDS.get(pos, {})

    elite_traits = get_elite_traits(player, position)
    elite_bonus = calculate_elite_bonus(player, position)
    draft_adj = calculate_draft_adjustment(player, position)

    # Determine tier
    if elite_bonus >= 1.25:
        tier = "elite"
        tier_label = "Elite Athlete"
    elif elite_bonus >= 1.15:
        tier = "above_average"
        tier_label = "Above Average Athlete"
    elif elite_bonus >= 1.08:
        tier = "good"
        tier_label = "Good Athlete"
    else:
        tier = "average"
        tier_label = "Average Athlete"

    # Extract measurables for display
    measurables = {}
    for trait in ["height", "weight", "forty", "ras", "vertical", "broad_jump",
                  "three_cone", "shuttle", "bench", "arm_length"]:
        if trait in player and player[trait] is not None:
            measurables[trait] = player[trait]

    return {
        "position": pos,
        "tier": tier,
        "tier_label": tier_label,
        "elite_bonus": elite_bonus,
        "draft_adjustment": draft_adj,
        "elite_traits": elite_traits,
        "elite_trait_count": len(elite_traits),
        "measurables": measurables,
        "thresholds": thresholds,
    }


# Convenience function for quick checks
def is_elite_athlete(player: Dict[str, Any], position: str) -> bool:
    """Check if player qualifies as elite athlete (top 10%)."""
    return calculate_elite_bonus(player, position) >= 1.25


def is_above_average_athlete(player: Dict[str, Any], position: str) -> bool:
    """Check if player is above average athletically (top 35%)."""
    return calculate_elite_bonus(player, position) >= 1.08
