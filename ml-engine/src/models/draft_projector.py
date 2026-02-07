"""
NFL Draft Projector for Portal IQ

Uses historical NFL draft data to project draft positions and contract values.
Integrates with elite traits system for athleticism bonuses.

Draft data structure in R2:
    data/nfl_draft_picks.csv - All historical draft picks (2020-2025)

Key Features:
- Position-specific draft modeling
- College performance to draft grade mapping
- Rookie contract estimation based on draft slot
- Elite athlete bonus adjustments
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache

import pandas as pd
import numpy as np

from ..utils.s3_storage import load_csv_with_fallback, R2DataLoadError, get_s3_client
from .elite_traits import calculate_elite_bonus, get_elite_traits

logger = logging.getLogger(__name__)


# =============================================================================
# Draft Value Constants
# =============================================================================

# Rookie contract estimates by draft round (in dollars)
ROOKIE_CONTRACT_BY_ROUND = {
    1: {"min": 10_000_000, "max": 40_000_000, "avg": 20_000_000},
    2: {"min": 5_000_000, "max": 12_000_000, "avg": 7_500_000},
    3: {"min": 3_000_000, "max": 6_000_000, "avg": 4_500_000},
    4: {"min": 2_500_000, "max": 4_500_000, "avg": 3_500_000},
    5: {"min": 2_000_000, "max": 3_500_000, "avg": 2_800_000},
    6: {"min": 1_500_000, "max": 2_500_000, "avg": 2_000_000},
    7: {"min": 1_000_000, "max": 2_000_000, "avg": 1_500_000},
}

# Career earnings multiplier by round (based on historical data)
CAREER_EARNINGS_MULTIPLIER = {
    1: 8.0,   # Top picks earn 8x their rookie deal
    2: 5.0,
    3: 4.0,
    4: 3.5,
    5: 3.0,
    6: 2.5,
    7: 2.0,
}

# Position value tiers (affects draft position)
POSITION_VALUE_TIERS = {
    "QB": 1.0,        # Premium
    "EDGE": 0.95,
    "OT": 0.90,
    "CB": 0.88,
    "WR": 0.85,
    "DL": 0.83,
    "S": 0.80,
    "LB": 0.78,
    "TE": 0.75,
    "RB": 0.70,
    "OG": 0.72,
    "C": 0.70,
    "K": 0.40,
    "P": 0.35,
}

# Map full position names to abbreviations
POSITION_MAP = {
    "Quarterback": "QB",
    "Wide Receiver": "WR",
    "Running Back": "RB",
    "Tight End": "TE",
    "Offensive Tackle": "OT",
    "Offensive Guard": "OG",
    "Center": "C",
    "Defensive End": "EDGE",
    "Defensive Tackle": "DL",
    "Outside Linebacker": "EDGE",
    "Inside Linebacker": "LB",
    "Linebacker": "LB",
    "Cornerback": "CB",
    "Safety": "S",
    "Place Kicker": "K",
    "Punter": "P",
    "Long Snapper": "LS",
}


# =============================================================================
# Draft Data Loading
# =============================================================================

_draft_data_cache = None
_draft_data_loaded = False


def load_draft_data() -> pd.DataFrame:
    """Load NFL draft data from R2.

    Returns:
        DataFrame with historical draft picks
    """
    global _draft_data_cache, _draft_data_loaded

    if _draft_data_loaded and _draft_data_cache is not None:
        return _draft_data_cache

    # Try R2 first
    try:
        df = load_csv_with_fallback("data/nfl_draft_picks.csv", cache_hours=168)
        logger.info(f"Loaded {len(df)} draft picks from R2")
        _draft_data_cache = df
        _draft_data_loaded = True
        return df
    except R2DataLoadError:
        logger.warning("Draft data not found in R2")

    # Try local file
    try:
        import os
        local_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "data", "raw", "nfl_draft_picks.csv"
        )
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            logger.info(f"Loaded {len(df)} draft picks from local file")
            _draft_data_cache = df
            _draft_data_loaded = True
            return df
    except Exception as e:
        logger.warning(f"Could not load local draft data: {e}")

    _draft_data_loaded = True
    _draft_data_cache = pd.DataFrame()
    return _draft_data_cache


def get_draft_history_by_position(position: str) -> pd.DataFrame:
    """Get draft history for a specific position.

    Args:
        position: Position abbreviation (QB, WR, etc.)

    Returns:
        DataFrame with draft picks for that position
    """
    df = load_draft_data()
    if df.empty:
        return df

    # Normalize position
    pos_upper = position.upper()

    # Filter by position (handling full names and abbreviations)
    mask = df["position"].apply(
        lambda x: POSITION_MAP.get(x, x).upper() == pos_upper if pd.notna(x) else False
    )

    return df[mask]


def get_draft_stats_by_school(school: str) -> Dict[str, Any]:
    """Get draft statistics for a school.

    Args:
        school: School name

    Returns:
        Dict with draft stats (total picks, avg round, etc.)
    """
    df = load_draft_data()
    if df.empty:
        return {}

    school_picks = df[df["collegeTeam"].str.contains(school, case=False, na=False)]

    if school_picks.empty:
        return {"school": school, "total_picks": 0}

    return {
        "school": school,
        "total_picks": len(school_picks),
        "first_round_picks": len(school_picks[school_picks["round"] == 1]),
        "avg_round": float(school_picks["round"].mean()),
        "avg_overall": float(school_picks["overall"].mean()),
        "years": sorted(school_picks["year"].unique().tolist()),
        "top_pick": int(school_picks["overall"].min()),
        "positions_drafted": school_picks["position"].value_counts().to_dict(),
    }


# =============================================================================
# Draft Projection Model
# =============================================================================

def normalize_position(position: str) -> str:
    """Normalize position to standard abbreviation."""
    pos = position.strip()
    return POSITION_MAP.get(pos, pos.upper())


def estimate_draft_grade(
    pff_grade: Optional[float] = None,
    stars: Optional[int] = None,
    nil_value: Optional[float] = None,
    position: str = "ATH",
) -> float:
    """Estimate pre-draft grade (0-100 scale).

    Uses available data to estimate what scouts would grade the player.

    Args:
        pff_grade: PFF overall grade (0-100)
        stars: Recruiting stars (1-5)
        nil_value: Current NIL valuation
        position: Player position

    Returns:
        Estimated draft grade (0-100)
    """
    grade = 70.0  # Base grade

    # PFF grade is the strongest signal
    if pff_grade and pff_grade > 0:
        if pff_grade >= 90:
            grade = 92 + (pff_grade - 90) * 0.3
        elif pff_grade >= 80:
            grade = 85 + (pff_grade - 80) * 0.7
        elif pff_grade >= 70:
            grade = 78 + (pff_grade - 70) * 0.7
        else:
            grade = 60 + (pff_grade - 50) * 0.9

    # Recruiting stars as secondary signal
    if stars:
        star_bonus = {5: 8, 4: 4, 3: 0, 2: -3, 1: -5}.get(stars, 0)
        grade += star_bonus

    # NIL value as market signal
    if nil_value and nil_value > 0:
        if nil_value >= 2_000_000:
            grade += 5
        elif nil_value >= 500_000:
            grade += 3
        elif nil_value >= 100_000:
            grade += 1

    # Position value adjustment
    pos_norm = normalize_position(position)
    pos_value = POSITION_VALUE_TIERS.get(pos_norm, 0.75)
    grade = grade * (0.9 + pos_value * 0.1)

    return min(99, max(50, grade))


def project_draft_position(
    player_data: Dict[str, Any],
    draft_grade: Optional[float] = None,
) -> Dict[str, Any]:
    """Project NFL draft position for a player.

    Args:
        player_data: Dict with player info (name, position, stats, etc.)
        draft_grade: Pre-calculated draft grade (or will calculate)

    Returns:
        Dict with draft projection details
    """
    position = normalize_position(player_data.get("position", "ATH"))

    # Calculate draft grade if not provided
    if draft_grade is None:
        draft_grade = estimate_draft_grade(
            pff_grade=player_data.get("pff_grade") or player_data.get("pff_overall"),
            stars=player_data.get("stars"),
            nil_value=player_data.get("nil_value") or player_data.get("valuation"),
            position=position,
        )

    # Get elite bonus
    elite_bonus, elite_traits = 1.0, []
    measurables = {
        k: v for k, v in player_data.items()
        if k in ["height", "weight", "forty", "vertical", "broad_jump",
                 "bench", "three_cone", "shuttle"]
    }
    if measurables:
        elite_traits = get_elite_traits(measurables, position)
        elite_bonus = calculate_elite_bonus(measurables, position)

    # Base projection from grade
    if draft_grade >= 94:
        base_pick = 5
        projected_round = 1
    elif draft_grade >= 91:
        base_pick = 15
        projected_round = 1
    elif draft_grade >= 88:
        base_pick = 28
        projected_round = 1
    elif draft_grade >= 85:
        base_pick = 48
        projected_round = 2
    elif draft_grade >= 82:
        base_pick = 80
        projected_round = 3
    elif draft_grade >= 78:
        base_pick = 120
        projected_round = 4
    elif draft_grade >= 74:
        base_pick = 170
        projected_round = 5
    elif draft_grade >= 70:
        base_pick = 210
        projected_round = 6
    else:
        base_pick = 240
        projected_round = 7

    # Apply elite athlete adjustment
    if elite_bonus >= 1.25:
        pick_adjustment = -30
    elif elite_bonus >= 1.15:
        pick_adjustment = -18
    elif elite_bonus >= 1.08:
        pick_adjustment = -8
    else:
        pick_adjustment = 0

    adjusted_pick = max(1, base_pick + pick_adjustment)

    # Recalculate round from adjusted pick
    if adjusted_pick <= 32:
        projected_round = 1
    elif adjusted_pick <= 64:
        projected_round = 2
    elif adjusted_pick <= 100:
        projected_round = 3
    elif adjusted_pick <= 135:
        projected_round = 4
    elif adjusted_pick <= 176:
        projected_round = 5
    elif adjusted_pick <= 220:
        projected_round = 6
    else:
        projected_round = 7

    # Calculate pick range
    pick_variance = 15 if projected_round <= 2 else 25
    pick_low = max(1, adjusted_pick - pick_variance)
    pick_high = min(262, adjusted_pick + pick_variance)

    # Calculate draft probability
    if draft_grade >= 85:
        draft_probability = 0.95
    elif draft_grade >= 80:
        draft_probability = 0.75
    elif draft_grade >= 75:
        draft_probability = 0.50
    elif draft_grade >= 70:
        draft_probability = 0.30
    else:
        draft_probability = 0.15

    # Estimate contract values
    contract_info = ROOKIE_CONTRACT_BY_ROUND.get(projected_round, ROOKIE_CONTRACT_BY_ROUND[7])
    rookie_contract = contract_info["avg"]

    # Adjust for pick within round
    if projected_round == 1:
        # First round has huge variance
        if adjusted_pick <= 5:
            rookie_contract = 35_000_000
        elif adjusted_pick <= 10:
            rookie_contract = 25_000_000
        elif adjusted_pick <= 15:
            rookie_contract = 18_000_000
        elif adjusted_pick <= 20:
            rookie_contract = 15_000_000
        else:
            rookie_contract = 12_000_000

    career_multiplier = CAREER_EARNINGS_MULTIPLIER.get(projected_round, 2.0)
    career_earnings = rookie_contract * career_multiplier

    # Letter grade
    if draft_grade >= 94:
        letter_grade = "A+"
    elif draft_grade >= 91:
        letter_grade = "A"
    elif draft_grade >= 88:
        letter_grade = "A-"
    elif draft_grade >= 85:
        letter_grade = "B+"
    elif draft_grade >= 82:
        letter_grade = "B"
    elif draft_grade >= 78:
        letter_grade = "B-"
    elif draft_grade >= 74:
        letter_grade = "C+"
    elif draft_grade >= 70:
        letter_grade = "C"
    else:
        letter_grade = "C-"

    return {
        "player": player_data.get("name", "Unknown"),
        "position": position,
        "draft_grade": round(draft_grade, 1),
        "draft_letter_grade": letter_grade,
        "projected_round": projected_round,
        "projected_pick": adjusted_pick,
        "pick_range": f"{pick_low}-{pick_high}",
        "draft_probability": round(draft_probability, 2),
        "elite_bonus": elite_bonus,
        "elite_traits": elite_traits,
        "elite_adjustment": pick_adjustment,
        "rookie_contract_estimate": rookie_contract,
        "career_earnings_estimate": career_earnings,
        "expected_draft_value": int(rookie_contract * draft_probability),
    }


def get_historical_comparables(
    player_data: Dict[str, Any],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Find historical draft comparables for a player.

    Args:
        player_data: Player info dict
        limit: Max comparables to return

    Returns:
        List of comparable players who were drafted
    """
    df = load_draft_data()
    if df.empty:
        return []

    position = normalize_position(player_data.get("position", "ATH"))
    draft_grade = player_data.get("draft_grade") or estimate_draft_grade(
        pff_grade=player_data.get("pff_grade"),
        stars=player_data.get("stars"),
        nil_value=player_data.get("nil_value"),
        position=position,
    )

    # Filter by position
    pos_df = df[df["position"].apply(
        lambda x: POSITION_MAP.get(x, x).upper() == position if pd.notna(x) else False
    )]

    if pos_df.empty:
        return []

    # Find players with similar grades
    if "preDraftGrade" in pos_df.columns:
        pos_df = pos_df.copy()
        pos_df["grade_diff"] = abs(pos_df["preDraftGrade"] - draft_grade)
        pos_df = pos_df.sort_values("grade_diff")
    else:
        pos_df = pos_df.sort_values("overall")

    comparables = []
    for _, row in pos_df.head(limit).iterrows():
        comparables.append({
            "name": row.get("name"),
            "school": row.get("collegeTeam"),
            "year": int(row.get("year", 0)),
            "round": int(row.get("round", 0)),
            "overall_pick": int(row.get("overall", 0)),
            "nfl_team": row.get("nflTeam"),
            "pre_draft_grade": row.get("preDraftGrade"),
            "position": position,
        })

    return comparables


# =============================================================================
# Mock Draft Generation
# =============================================================================

def generate_mock_draft(
    year: int = 2025,
    rounds: int = 3,
    player_pool: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate a mock draft board.

    Args:
        year: Draft year
        rounds: Number of rounds to project
        player_pool: Optional DataFrame of players to rank

    Returns:
        Dict with draft board and analysis
    """
    # If no player pool provided, use NIL top players
    if player_pool is None:
        from ..utils.data_loader import get_nil_players
        player_pool = get_nil_players(limit=500)

    if player_pool.empty:
        return {"year": year, "rounds": rounds, "draft_board": []}

    # Calculate draft projections for each player
    projections = []
    for _, row in player_pool.iterrows():
        player_data = row.to_dict()
        projection = project_draft_position(player_data)
        projection["nil_value"] = player_data.get("nil_value") or player_data.get("valuation", 0)
        projections.append(projection)

    # Sort by projected pick
    projections.sort(key=lambda x: (x["projected_pick"], -x["draft_grade"]))

    # Build draft board
    draft_board = []
    pick_number = 0

    for proj in projections:
        if proj["projected_round"] > rounds:
            continue
        if proj["draft_probability"] < 0.3:
            continue

        pick_number += 1
        if pick_number > rounds * 32:
            break

        draft_board.append({
            "pick": pick_number,
            "round": (pick_number - 1) // 32 + 1,
            "pick_in_round": ((pick_number - 1) % 32) + 1,
            "player": proj["player"],
            "position": proj["position"],
            "draft_grade": proj["draft_grade"],
            "letter_grade": proj["draft_letter_grade"],
            "nil_value": proj.get("nil_value", 0),
            "elite_traits": proj.get("elite_traits", []),
            "projected_contract": proj["rookie_contract_estimate"],
        })

    return {
        "year": year,
        "rounds": rounds,
        "total_picks": len(draft_board),
        "draft_board": draft_board,
        "top_prospects": draft_board[:10] if len(draft_board) >= 10 else draft_board,
        "position_breakdown": _count_by_position(draft_board),
    }


def _count_by_position(draft_board: List[Dict]) -> Dict[str, int]:
    """Count picks by position."""
    counts = {}
    for pick in draft_board:
        pos = pick.get("position", "Unknown")
        counts[pos] = counts.get(pos, 0) + 1
    return counts


def get_player_draft_outcome(player_name: str) -> Optional[Dict[str, Any]]:
    """Get draft outcome for a player.

    Args:
        player_name: Player name to look up

    Returns:
        Dict with draft info or None if not found
    """
    df = load_draft_data()
    if df.empty:
        return None

    # Try exact match
    match = df[df["name"].str.lower() == player_name.lower()]

    # Try contains match
    if match.empty:
        match = df[df["name"].str.contains(player_name, case=False, na=False)]

    if match.empty:
        return None

    # Get most recent (highest year)
    row = match.sort_values("year", ascending=False).iloc[0]

    return {
        "drafted": True,
        "year": int(row.get("year", 0)),
        "round": int(row.get("round", 0)) if pd.notna(row.get("round")) else None,
        "pick": int(row.get("pick", 0)) if pd.notna(row.get("pick")) else None,
        "overall": int(row.get("overall", 0)) if pd.notna(row.get("overall")) else None,
        "team": row.get("team"),
        "college": row.get("college"),
        "position": row.get("position"),
        "contract_value": row.get("contract_value"),
        "aav": row.get("aav"),
        "guaranteed": row.get("guaranteed"),
    }


# =============================================================================
# API-Ready Functions
# =============================================================================

def predict(player_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Predict draft positions for players in a DataFrame.

    This is the main entry point used by the API.

    Args:
        player_df: DataFrame with player data

    Returns:
        List of draft projections
    """
    projections = []

    for _, row in player_df.iterrows():
        player_data = row.to_dict()
        projection = project_draft_position(player_data)
        projections.append(projection)

    return projections


def get_draft_class_stats(year: int) -> Dict[str, Any]:
    """Get statistics for a draft class.

    Args:
        year: Draft year

    Returns:
        Dict with class statistics
    """
    df = load_draft_data()
    if df.empty:
        return {"year": year, "error": "No draft data available"}

    year_df = df[df["year"] == year]
    if year_df.empty:
        return {"year": year, "error": f"No data for {year} draft"}

    return {
        "year": year,
        "total_picks": len(year_df),
        "first_round": len(year_df[year_df["round"] == 1]),
        "by_conference": year_df["collegeConference"].value_counts().to_dict(),
        "by_position": year_df["position"].value_counts().to_dict(),
        "top_schools": year_df["collegeTeam"].value_counts().head(10).to_dict(),
        "avg_grade": float(year_df["preDraftGrade"].mean()) if "preDraftGrade" in year_df.columns else None,
    }
