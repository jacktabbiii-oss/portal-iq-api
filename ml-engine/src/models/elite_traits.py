"""
Elite Traits Bonus System for Portal IQ

Identifies elite athletes based on top 10% measurable thresholds by position.
Elite traits provide NIL, draft, and WAR bonuses - NOT blanket weighting.

Philosophy:
- Average measurables = no adjustment (compete on performance)
- Elite measurables (top 10%) = meaningful boost
- Position-specific thresholds (6'5" is elite for WR, average for OT)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from ..utils.historical_loader import get_player_measurables, normalize_position
from ..utils.s3_storage import load_csv_with_fallback, R2DataLoadError

logger = logging.getLogger(__name__)


# =============================================================================
# Elite Thresholds by Position (Top 10% based on combine data)
# =============================================================================

ELITE_THRESHOLDS = {
    "QB": {
        "height": 76,      # 6'4"
        "weight": 225,
        "forty": 4.65,     # Lower is better
        "vertical": 32,
        "broad_jump": 112,
    },
    "RB": {
        "height": 71,      # 5'11"
        "weight": 215,
        "forty": 4.45,
        "vertical": 38,
        "broad_jump": 124,
        "bench": 20,
    },
    "WR": {
        "height": 74,      # 6'2"
        "weight": 205,
        "forty": 4.40,
        "vertical": 38,
        "broad_jump": 128,
        "three_cone": 6.85,
    },
    "TE": {
        "height": 77,      # 6'5"
        "weight": 250,
        "forty": 4.65,
        "vertical": 35,
        "bench": 22,
    },
    "OT": {
        "height": 78,      # 6'6"
        "weight": 315,
        "forty": 5.10,
        "bench": 25,
        "arm_length": 34.5,
    },
    "OG": {
        "height": 77,      # 6'5"
        "weight": 315,
        "forty": 5.20,
        "bench": 28,
    },
    "C": {
        "height": 76,      # 6'4"
        "weight": 305,
        "forty": 5.15,
        "bench": 26,
    },
    "OL": {
        "height": 77,
        "weight": 310,
        "forty": 5.15,
        "bench": 26,
    },
    "EDGE": {
        "height": 76,      # 6'4"
        "weight": 260,
        "forty": 4.65,
        "vertical": 36,
        "broad_jump": 120,
        "bench": 22,
    },
    "DL": {
        "height": 76,      # 6'4"
        "weight": 295,
        "forty": 4.90,
        "bench": 28,
        "arm_length": 34,
    },
    "DT": {
        "height": 75,
        "weight": 305,
        "forty": 5.00,
        "bench": 30,
    },
    "LB": {
        "height": 74,      # 6'2"
        "weight": 235,
        "forty": 4.55,
        "vertical": 36,
        "three_cone": 6.95,
    },
    "CB": {
        "height": 72,      # 6'0"
        "weight": 195,
        "forty": 4.42,
        "vertical": 38,
        "three_cone": 6.80,
        "broad_jump": 126,
    },
    "S": {
        "height": 73,      # 6'1"
        "weight": 210,
        "forty": 4.48,
        "vertical": 38,
        "broad_jump": 124,
    },
    "K": {
        "height": 72,
        "weight": 190,
    },
    "P": {
        "height": 74,
        "weight": 210,
    },
}

# Traits where lower values are better
LOWER_IS_BETTER = {"forty", "three_cone", "shuttle"}


# =============================================================================
# Elite Trait Detection
# =============================================================================

def get_elite_traits(
    player_data: Dict[str, Any],
    position: str,
) -> List[str]:
    """Identify which traits are elite for a player.

    Args:
        player_data: Dict with player measurables
        position: Player position

    Returns:
        List of trait names that are elite
    """
    position = normalize_position(position)
    thresholds = ELITE_THRESHOLDS.get(position, {})

    if not thresholds:
        return []

    elite_traits = []

    for trait, elite_value in thresholds.items():
        player_value = player_data.get(trait)

        if player_value is None or pd.isna(player_value):
            continue

        try:
            player_value = float(player_value)
        except (ValueError, TypeError):
            continue

        # Check if elite (lower is better for some traits)
        if trait in LOWER_IS_BETTER:
            if player_value <= elite_value:
                elite_traits.append(trait)
        else:
            if player_value >= elite_value:
                elite_traits.append(trait)

    return elite_traits


def calculate_elite_bonus(
    player_data: Dict[str, Any],
    position: str,
) -> float:
    """Calculate the elite bonus multiplier for a player.

    Args:
        player_data: Dict with player measurables
        position: Player position

    Returns:
        Multiplier (1.0 = no bonus, up to 1.25 = 25% boost)
    """
    position = normalize_position(position)
    thresholds = ELITE_THRESHOLDS.get(position, {})

    if not thresholds:
        return 1.0

    elite_traits = 0
    total_traits = 0

    for trait, elite_value in thresholds.items():
        player_value = player_data.get(trait)

        if player_value is None or pd.isna(player_value):
            continue

        try:
            player_value = float(player_value)
        except (ValueError, TypeError):
            continue

        total_traits += 1

        # Check if elite
        if trait in LOWER_IS_BETTER:
            if player_value <= elite_value:
                elite_traits += 1
        else:
            if player_value >= elite_value:
                elite_traits += 1

    if total_traits == 0:
        return 1.0

    elite_ratio = elite_traits / total_traits

    # Tiered bonus system
    if elite_ratio >= 0.8:      # 4/5+ traits elite
        return 1.25             # +25% bonus
    elif elite_ratio >= 0.6:    # 3/5 traits elite
        return 1.15             # +15% bonus
    elif elite_ratio >= 0.4:    # 2/5 traits elite
        return 1.08             # +8% bonus
    else:
        return 1.0              # No bonus


def calculate_elite_bonus_for_player(player_name: str, position: str) -> Tuple[float, List[str]]:
    """Calculate elite bonus using combine data from R2.

    Args:
        player_name: Player to look up
        position: Player position

    Returns:
        Tuple of (multiplier, list of elite traits)
    """
    measurables = get_player_measurables(player_name)

    if not measurables:
        return 1.0, []

    elite_traits = get_elite_traits(measurables, position)
    bonus = calculate_elite_bonus(measurables, position)

    return bonus, elite_traits


# =============================================================================
# Draft Adjustment
# =============================================================================

def get_draft_adjustment(elite_bonus: float) -> int:
    """Get draft pick adjustment based on elite bonus.

    Args:
        elite_bonus: The elite multiplier (1.0 - 1.25)

    Returns:
        Pick adjustment (negative = move up in draft)
    """
    if elite_bonus >= 1.25:
        return -30      # Move up 30 picks
    elif elite_bonus >= 1.15:
        return -18      # Move up 18 picks
    elif elite_bonus >= 1.08:
        return -8       # Move up 8 picks
    else:
        return 0        # No adjustment


# =============================================================================
# Combine Data Integration
# =============================================================================

def load_combine_data_from_r2() -> pd.DataFrame:
    """Load combine data from R2 storage.

    Returns:
        DataFrame with combine data
    """
    # Try the processed folder first
    try:
        df = load_csv_with_fallback("processed/combine_data.csv", cache_hours=168)
        if not df.empty:
            logger.info(f"Loaded combine data: {len(df)} records")
            return df
    except R2DataLoadError:
        pass

    # Try historical combine folder
    try:
        from ..utils.historical_loader import load_all_combine_data
        df = load_all_combine_data()
        if not df.empty:
            return df
    except Exception as e:
        logger.warning(f"Could not load historical combine data: {e}")

    return pd.DataFrame()


def enrich_player_with_combine_data(
    player_data: Dict[str, Any],
    combine_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Enrich player data with combine measurables.

    Args:
        player_data: Base player data
        combine_df: Optional pre-loaded combine DataFrame

    Returns:
        Player data enriched with measurables
    """
    if combine_df is None:
        combine_df = load_combine_data_from_r2()

    if combine_df.empty:
        return player_data

    player_name = player_data.get("name", "")
    if not player_name:
        return player_data

    # Find player in combine data
    name_col = "name" if "name" in combine_df.columns else "player_name"
    if name_col not in combine_df.columns:
        return player_data

    match = combine_df[
        combine_df[name_col].str.contains(player_name, case=False, na=False)
    ]

    if match.empty:
        return player_data

    # Get most recent combine data
    if "combine_year" in match.columns:
        match = match.sort_values("combine_year", ascending=False)

    combine_row = match.iloc[0]

    # Measurable columns to add
    measurable_cols = [
        "height", "weight", "forty", "vertical", "broad_jump",
        "bench", "three_cone", "shuttle", "arm_length", "hand_size"
    ]

    enriched = player_data.copy()
    for col in measurable_cols:
        if col in combine_row.index and pd.notna(combine_row[col]):
            enriched[col] = combine_row[col]

    return enriched


# =============================================================================
# API-Ready Functions
# =============================================================================

def get_player_elite_profile(player_name: str) -> Dict[str, Any]:
    """Get complete elite athlete profile for a player.

    Args:
        player_name: Player name to look up

    Returns:
        Dict with elite profile data
    """
    # Get player data from NIL valuations
    from ..utils.data_loader import get_nil_players

    nil_df = get_nil_players(limit=50000)
    player_data = {}

    if not nil_df.empty and "name" in nil_df.columns:
        match = nil_df[nil_df["name"].str.contains(player_name, case=False, na=False)]
        if not match.empty:
            player_data = match.iloc[0].to_dict()

    if not player_data:
        return {
            "player": player_name,
            "error": "Player not found",
            "elite_traits": [],
            "elite_bonus": 1.0,
            "measurables": {},
        }

    position = player_data.get("position", "ATH")

    # Get measurables
    measurables = get_player_measurables(player_name)

    # If no combine data, try to enrich from combine DataFrame
    if not measurables:
        player_data = enrich_player_with_combine_data(player_data)
        measurables = {
            k: v for k, v in player_data.items()
            if k in ["height", "weight", "forty", "vertical", "broad_jump",
                     "bench", "three_cone", "shuttle", "arm_length", "hand_size"]
        }

    # Calculate elite traits and bonus
    elite_traits = get_elite_traits(measurables, position)
    elite_bonus = calculate_elite_bonus(measurables, position)
    draft_adjustment = get_draft_adjustment(elite_bonus)

    return {
        "player": player_name,
        "position": position,
        "elite_traits": elite_traits,
        "elite_trait_count": len(elite_traits),
        "elite_bonus": elite_bonus,
        "draft_adjustment": draft_adjustment,
        "measurables": measurables,
        "thresholds": ELITE_THRESHOLDS.get(normalize_position(position), {}),
    }
