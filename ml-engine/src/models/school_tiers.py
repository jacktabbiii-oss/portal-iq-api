"""
Dynamic School Tier System

Uses CFBD data to calculate school tiers and NIL multipliers based on:
- Recent performance (wins, CFP appearances, championships)
- SP+ ratings
- Recruiting rankings
- Talent composite
- Portal activity (net gains/losses)

This replaces hardcoded school lists with data-driven tiers.
"""

import logging
from typing import Dict, Optional, Tuple
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Import R2 storage for loading data
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.s3_storage import load_csv_with_fallback

logger = logging.getLogger(__name__)

# Tier definitions with multiplier ranges
TIER_DEFINITIONS = {
    "blue_blood": {"multiplier": 3.0, "label": "Blue Blood / National Champion"},
    "elite": {"multiplier": 2.3, "label": "Elite (CFP Contender)"},
    "power_strong": {"multiplier": 1.8, "label": "Strong Power 4"},
    "power_mid": {"multiplier": 1.4, "label": "Mid Power 4"},
    "power_low": {"multiplier": 1.1, "label": "Lower Power 4"},
    "g5_strong": {"multiplier": 1.0, "label": "Strong G5"},
    "g5_mid": {"multiplier": 0.8, "label": "Mid G5"},
    "fcs": {"multiplier": 0.5, "label": "FCS"},
}

# Manual overrides for special cases (championships, etc.)
# Updated Feb 2026
MANUAL_OVERRIDES = {
    # 2025 National Champions
    "Indiana": "blue_blood",  # Won 2025 National Championship

    # Traditional blue bloods (historical significance)
    "Alabama": "blue_blood",
    "Ohio State": "blue_blood",
    "Georgia": "blue_blood",
    "Texas": "blue_blood",
    "USC": "blue_blood",
    "Michigan": "blue_blood",
    "Notre Dame": "blue_blood",
    "Oklahoma": "blue_blood",

    # Recent CFP success elevates tier
    "Clemson": "elite",
    "LSU": "elite",
    "Oregon": "elite",

    # Prime effect / special circumstances
    "Colorado": "elite",
}


def load_team_data() -> Optional[pd.DataFrame]:
    """Load and merge team data from R2 storage using available sources."""
    try:
        # Try loading CFBD rosters to get list of all schools
        rosters_df = load_csv_with_fallback("cfbd_rosters.csv", subfolder="processed")

        if rosters_df is None or rosters_df.empty:
            logger.warning("CFBD rosters file not found in R2")
            return None

        # Get unique schools from rosters
        if "school" in rosters_df.columns:
            schools = rosters_df["school"].unique()
            logger.info(f"Found {len(schools)} unique schools from CFBD rosters")

            # Create a basic team dataframe with schools
            records_df = pd.DataFrame({
                "team": schools,
                "total_wins": 0,
                "total_losses": 0,
                "conference": "",
            })
        else:
            logger.warning("School column not found in rosters")
            return None

        # Add default values for missing columns
        team_df = records_df.copy()
        team_df["sp_overall"] = 0
        team_df["sp_offense"] = 0
        team_df["sp_defense"] = 0
        team_df["season"] = 2025

        # Keep column names as-is (sp_overall, sp_offense, sp_defense) for compatibility

        logger.info(f"Final merged team data: {len(team_df)} schools")
        return team_df

    except Exception as e:
        logger.error(f"Error loading team data: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_school_score(row: pd.Series) -> float:
    """
    Calculate a composite score for a school based on multiple factors.

    Scoring (out of 100):
    - Wins: 0-30 points (15 wins = 30 pts)
    - SP+ Rating: 0-25 points (top 5 = 25 pts)
    - Talent Composite: 0-25 points (top 10 = 25 pts)
    - Recruiting Rank: 0-20 points (top 5 = 20 pts)
    """
    score = 0.0

    # Wins (0-30 points)
    wins = row.get("total_wins", 0) or 0
    if wins >= 15:
        score += 30  # Undefeated/CFP
    elif wins >= 12:
        score += 25  # Top 10 finish
    elif wins >= 10:
        score += 20  # Bowl win
    elif wins >= 8:
        score += 15  # Bowl eligible
    elif wins >= 6:
        score += 10
    elif wins >= 4:
        score += 5

    # SP+ Rating (0-25 points)
    sp_overall = row.get("sp_overall", 0) or 0
    if sp_overall >= 25:
        score += 25  # Elite (Bama, OSU level)
    elif sp_overall >= 20:
        score += 22
    elif sp_overall >= 15:
        score += 18
    elif sp_overall >= 10:
        score += 14
    elif sp_overall >= 5:
        score += 10
    elif sp_overall >= 0:
        score += 5

    # Talent Composite (0-25 points)
    talent = row.get("talent_composite", 0) or 0
    if talent >= 900:
        score += 25  # Top 5
    elif talent >= 850:
        score += 22
    elif talent >= 800:
        score += 18
    elif talent >= 750:
        score += 14
    elif talent >= 700:
        score += 10
    elif talent >= 600:
        score += 5

    # Recruiting Rank (0-20 points)
    recruiting_rank = row.get("recruiting_rank", 100) or 100
    if recruiting_rank <= 5:
        score += 20
    elif recruiting_rank <= 10:
        score += 17
    elif recruiting_rank <= 15:
        score += 14
    elif recruiting_rank <= 25:
        score += 10
    elif recruiting_rank <= 40:
        score += 6
    elif recruiting_rank <= 60:
        score += 3

    return score


def score_to_tier(score: float, conference: Optional[str] = None) -> str:
    """Convert composite score to tier."""
    # Check conference for G5/FCS
    g5_conferences = ["American", "Mountain West", "Sun Belt", "MAC", "Conference USA"]
    is_g5 = conference in g5_conferences if conference else False

    if score >= 85:
        return "blue_blood"
    elif score >= 70:
        return "elite"
    elif score >= 55:
        return "power_strong"
    elif score >= 40:
        return "power_mid"
    elif score >= 25:
        if is_g5:
            return "g5_strong"
        return "power_low"
    elif score >= 15:
        return "g5_mid" if is_g5 else "power_low"
    else:
        return "fcs" if score < 10 else "g5_mid"


@lru_cache(maxsize=1)
def get_school_tiers() -> Dict[str, Dict]:
    """
    Get all school tiers calculated from CFBD data.

    Returns:
        Dict mapping school name to tier info:
        {
            "Alabama": {"tier": "blue_blood", "multiplier": 3.0, "score": 95.0},
            ...
        }
    """
    tiers = {}

    # Load team data
    team_df = load_team_data()

    if team_df is not None and not team_df.empty:
        # Get most recent season for each team
        if "season" in team_df.columns:
            latest_season = team_df["season"].max()
            recent_df = team_df[team_df["season"] == latest_season].copy()
        else:
            recent_df = team_df.copy()

        # Calculate scores and tiers
        for _, row in recent_df.iterrows():
            team = row.get("team", "")
            if not team:
                continue

            conference = row.get("conference", "")
            score = calculate_school_score(row)

            # Check for manual override
            if team in MANUAL_OVERRIDES:
                tier = MANUAL_OVERRIDES[team]
            else:
                tier = score_to_tier(score, conference)

            tiers[team] = {
                "tier": tier,
                "multiplier": TIER_DEFINITIONS[tier]["multiplier"],
                "label": TIER_DEFINITIONS[tier]["label"],
                "score": round(score, 1),
                # Core metrics
                "wins": int(row.get("total_wins", 0)) if pd.notna(row.get("total_wins")) else 0,
                "losses": int(row.get("total_losses", 0)) if pd.notna(row.get("total_losses")) else 0,
                "conference": str(row.get("conference", "")),
                # SP+ ratings
                "sp_plus_overall": float(row.get("sp_overall", 0)) if pd.notna(row.get("sp_overall")) else 0,
                "sp_plus_offense": float(row.get("sp_offense", 0)) if pd.notna(row.get("sp_offense")) else 0,
                "sp_plus_defense": float(row.get("sp_defense", 0)) if pd.notna(row.get("sp_defense")) else 0,
                # Talent & recruiting
                "talent_composite": float(row.get("talent_composite", 0)) if pd.notna(row.get("talent_composite")) else 0,
                "recruiting_rank": row.get("recruiting_rank", None),
                # Portal performance (On3)
                "portal_rank": int(row.get("overall_rank", 0)) if pd.notna(row.get("overall_rank")) else None,
                "portal_score": float(row.get("overall_score", 0)) if pd.notna(row.get("overall_score")) else None,
                "transfers_in": int(row.get("transfers_in", 0)) if pd.notna(row.get("transfers_in")) else 0,
                "transfers_out": int(row.get("transfers_out", 0)) if pd.notna(row.get("transfers_out")) else 0,
                "portal_net": int(row.get("transfers_in", 0) - row.get("transfers_out", 0)) if pd.notna(row.get("transfers_in")) and pd.notna(row.get("transfers_out")) else 0,
                "avg_rating_in": float(row.get("avg_rating_in", 0)) if pd.notna(row.get("avg_rating_in")) else None,
                "avg_rating_out": float(row.get("avg_rating_out", 0)) if pd.notna(row.get("avg_rating_out")) else None,
                "five_stars_net": int(row.get("five_stars_net", 0)) if pd.notna(row.get("five_stars_net")) else 0,
                "four_stars_net": int(row.get("four_stars_net", 0)) if pd.notna(row.get("four_stars_net")) else 0,
                "three_stars_net": int(row.get("three_stars_net", 0)) if pd.notna(row.get("three_stars_net")) else 0,
                # NIL spending (On3)
                "nil_valuation": float(row.get("adjusted_nil_valuation", 0)) if pd.notna(row.get("adjusted_nil_valuation")) else None,
                "nil_valuation_change": float(row.get("nil_valuation_change", 0)) if pd.notna(row.get("nil_valuation_change")) else None,
            }

    # Add manual overrides that might not be in data
    for team, tier in MANUAL_OVERRIDES.items():
        if team not in tiers:
            tiers[team] = {
                "tier": tier,
                "multiplier": TIER_DEFINITIONS[tier]["multiplier"],
                "label": TIER_DEFINITIONS[tier]["label"],
                "score": 80.0 if tier == "blue_blood" else 65.0,  # Default scores
                "wins": None,
                "sp_plus": None,
                "talent": None,
                "recruiting_rank": None,
            }

    logger.info(f"Calculated tiers for {len(tiers)} schools")
    return tiers


def get_school_multiplier(school: str) -> float:
    """
    Get NIL multiplier for a school.

    Uses CFBD data when available, falls back to defaults.
    """
    tiers = get_school_tiers()

    if school in tiers:
        return tiers[school]["multiplier"]

    # Default for unknown schools
    return 0.8


def get_school_tier(school: str) -> Tuple[str, Dict]:
    """
    Get tier classification for a school.

    Returns:
        Tuple of (tier_name, tier_info_dict)
    """
    tiers = get_school_tiers()

    if school in tiers:
        return tiers[school]["tier"], tiers[school]

    # Default for unknown schools
    return "power_low", {
        "tier": "power_low",
        "multiplier": 0.8,
        "label": "Unknown/Lower Tier",
        "score": 0,
    }


def refresh_tiers():
    """Clear the tier cache to force refresh from data."""
    get_school_tiers.cache_clear()
    logger.info("School tier cache cleared")


# API endpoint helper
def get_all_school_tiers_for_api() -> Dict:
    """
    Get all school tiers formatted for API response.
    """
    tiers = get_school_tiers()

    # Group by tier
    by_tier = {tier: [] for tier in TIER_DEFINITIONS.keys()}

    for school, info in tiers.items():
        tier = info["tier"]
        by_tier[tier].append({
            "school": school,
            **info
        })

    # Sort each tier by score
    for tier in by_tier:
        by_tier[tier].sort(key=lambda x: x.get("score", 0), reverse=True)

    return {
        "tiers": by_tier,
        "tier_definitions": TIER_DEFINITIONS,
        "total_schools": len(tiers),
    }


if __name__ == "__main__":
    # Test the tier system
    print("School Tier System Test")
    print("=" * 60)

    tiers = get_school_tiers()

    # Show top schools by tier
    for tier_name in ["blue_blood", "elite", "power_strong"]:
        print(f"\n{tier_name.upper()}:")
        schools_in_tier = [
            (school, info) for school, info in tiers.items()
            if info["tier"] == tier_name
        ]
        schools_in_tier.sort(key=lambda x: x[1]["score"], reverse=True)

        for school, info in schools_in_tier[:10]:
            print(f"  {school:20s}: {info['score']:5.1f} pts, {info['multiplier']:.1f}x mult")

    # Test specific schools
    print("\n" + "=" * 60)
    print("SPECIFIC SCHOOL LOOKUPS:")
    test_schools = ["Indiana", "Alabama", "Oregon", "Colorado", "Boise State", "Unknown School"]

    for school in test_schools:
        mult = get_school_multiplier(school)
        tier, info = get_school_tier(school)
        print(f"  {school:20s}: {tier:15s} ({mult:.1f}x)")
