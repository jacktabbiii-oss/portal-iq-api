"""
Dynamic School Tier System

Calculates school tiers and NIL multipliers from real CFBD data:
- Wins (cfbd_team_records.csv)
- SP+ ratings (cfbd_sp_ratings.csv)
- Talent composite (cfbd_team_talent.csv)

Every FBS school gets a data-driven tier — no hardcoded school lists.
"""

import logging
from typing import Dict, Optional, Tuple
from functools import lru_cache

import pandas as pd

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

# Power 4 conferences
POWER_4_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC"}

# G5 conferences
G5_CONFERENCES = {"American", "Mountain West", "Sun Belt", "MAC", "Conference USA"}


def load_team_data() -> Optional[pd.DataFrame]:
    """Load and merge CFBD team data from the 3 separate CSV files in R2.

    Merges:
    - cfbd_team_records.csv (wins, losses, conference)
    - cfbd_sp_ratings.csv (SP+ overall, offense, defense)
    - cfbd_team_talent.csv (talent composite)

    Returns a single DataFrame with one row per team (most recent season).
    """
    try:
        from ..utils.data_loader import _load_csv
    except ImportError:
        logger.error("Could not import data_loader")
        return None

    merged = None

    # 1. Team records (wins, losses, conference)
    try:
        records_df = _load_csv("cfbd_team_records.csv")
        if records_df is not None and not records_df.empty:
            if "year" in records_df.columns:
                latest = records_df["year"].max()
                records_df = records_df[records_df["year"] == latest].copy()
            records_df["_key"] = records_df["school"].str.strip().str.lower()
            keep = ["_key", "school", "total_wins", "total_losses", "conference"]
            keep = [c for c in keep if c in records_df.columns]
            merged = records_df[keep].drop_duplicates(subset=["_key"], keep="first")
            logger.info(f"Loaded {len(merged)} team records")
    except Exception as e:
        logger.debug(f"Could not load team records: {e}")

    # 2. SP+ ratings
    try:
        sp_df = _load_csv("cfbd_sp_ratings.csv")
        if sp_df is not None and not sp_df.empty:
            if "year" in sp_df.columns:
                latest = sp_df["year"].max()
                sp_df = sp_df[sp_df["year"] == latest].copy()
            sp_df["_key"] = sp_df["school"].str.strip().str.lower()
            sp_cols = ["_key", "sp_overall", "sp_offense", "sp_defense"]
            if "conference" in sp_df.columns:
                sp_cols.append("conference")
            sp_cols = [c for c in sp_cols if c in sp_df.columns]
            sp_slim = sp_df[sp_cols].drop_duplicates(subset=["_key"], keep="first")

            if merged is not None:
                # Don't duplicate conference column
                join_cols = [c for c in sp_slim.columns if c not in merged.columns or c == "_key"]
                merged = merged.merge(sp_slim[join_cols], on="_key", how="outer")
            else:
                merged = sp_slim
                # Need school name
                if "school" not in merged.columns:
                    merged["school"] = sp_df.set_index(sp_df["_key"])["school"]
            logger.info(f"Loaded SP+ ratings for {len(sp_slim)} teams")
    except Exception as e:
        logger.debug(f"Could not load SP+ ratings: {e}")

    # 3. Talent composite
    try:
        talent_df = _load_csv("cfbd_team_talent.csv")
        if talent_df is not None and not talent_df.empty:
            if "year" in talent_df.columns:
                latest = talent_df["year"].max()
                talent_df = talent_df[talent_df["year"] == latest].copy()
            talent_df["_key"] = talent_df["school"].str.strip().str.lower()
            talent_slim = talent_df[["_key", "talent"]].drop_duplicates(subset=["_key"], keep="first")

            if merged is not None:
                merged = merged.merge(talent_slim, on="_key", how="outer")
            else:
                merged = talent_slim
                if "school" not in merged.columns:
                    merged["school"] = talent_df.set_index(talent_df["_key"])["school"]
            logger.info(f"Loaded talent composites for {len(talent_slim)} teams")
    except Exception as e:
        logger.debug(f"Could not load talent composites: {e}")

    if merged is not None and not merged.empty:
        # Fill school name from key if missing
        if "school" not in merged.columns:
            merged["school"] = merged["_key"].str.title()
        else:
            merged["school"] = merged["school"].fillna(merged["_key"].str.title())
        logger.info(f"Merged CFBD data for {len(merged)} schools")
        return merged

    return None


def calculate_school_score(row: pd.Series) -> float:
    """
    Calculate a composite score for a school based on real CFBD metrics.

    Scoring (out of 100):
    - Wins: 0-30 points (continuous scale)
    - SP+ Rating: 0-30 points (continuous scale)
    - Talent Composite: 0-25 points (continuous scale)
    - Conference strength bonus: 0-15 points
    """
    score = 0.0

    # Wins (0-30 points, continuous)
    wins = float(row.get("total_wins", 0) or 0)
    score += min(30, wins * 2.3)  # 13 wins = 29.9 pts

    # SP+ Rating (0-30 points, continuous)
    sp_overall = float(row.get("sp_overall", 0) or 0)
    # SP+ ranges from about -15 (worst) to +30 (best)
    # Normalize to 0-30 point scale
    sp_normalized = max(0, (sp_overall + 15) / 45 * 30)
    score += min(30, sp_normalized)

    # Talent Composite (0-25 points, continuous)
    talent = float(row.get("talent", 0) or row.get("talent_composite", 0) or 0)
    if talent > 0:
        # Talent ranges from ~400 (lowest FBS) to ~1000 (Alabama/Georgia)
        talent_normalized = max(0, (talent - 400) / 600 * 25)
        score += min(25, talent_normalized)

    # Conference strength bonus (0-15 points)
    conference = str(row.get("conference", ""))
    if conference in POWER_4_CONFERENCES:
        score += 10
        # Extra boost for SEC/Big Ten
        if conference in ("SEC", "Big Ten"):
            score += 5
    elif conference in G5_CONFERENCES:
        score += 3

    return round(score, 1)


def score_to_tier(score: float, conference: Optional[str] = None) -> str:
    """Convert composite score to tier based on data distribution."""
    is_g5 = conference in G5_CONFERENCES if conference else False
    is_p4 = conference in POWER_4_CONFERENCES if conference else False

    if score >= 80:
        return "blue_blood"
    elif score >= 65:
        return "elite"
    elif score >= 50:
        return "power_strong"
    elif score >= 38:
        return "power_mid" if is_p4 else "g5_strong"
    elif score >= 25:
        if is_g5:
            return "g5_strong"
        return "power_low"
    elif score >= 15:
        return "g5_mid" if is_g5 else "power_low"
    else:
        return "fcs" if score < 8 else "g5_mid"


@lru_cache(maxsize=1)
def get_school_tiers() -> Dict[str, Dict]:
    """
    Get all school tiers calculated from real CFBD data.

    Returns:
        Dict mapping school name to tier info:
        {
            "Alabama": {"tier": "blue_blood", "multiplier": 3.0, "score": 95.0, ...},
            ...
        }
    """
    tiers = {}

    # Load real CFBD data
    team_df = load_team_data()

    if team_df is not None and not team_df.empty:
        for _, row in team_df.iterrows():
            team = str(row.get("school", "")).strip()
            if not team or team == "nan":
                continue

            conference = str(row.get("conference", ""))
            score = calculate_school_score(row)
            tier = score_to_tier(score, conference)

            wins = row.get("total_wins")
            sp_plus = row.get("sp_overall")
            talent = row.get("talent", row.get("talent_composite"))

            tiers[team] = {
                "tier": tier,
                "multiplier": TIER_DEFINITIONS[tier]["multiplier"],
                "label": TIER_DEFINITIONS[tier]["label"],
                "score": score,
                "wins": int(wins) if pd.notna(wins) else None,
                "losses": int(row.get("total_losses")) if pd.notna(row.get("total_losses")) else None,
                "sp_plus": round(float(sp_plus), 1) if pd.notna(sp_plus) else None,
                "talent": round(float(talent), 1) if pd.notna(talent) else None,
                "conference": conference if conference and conference != "nan" else None,
            }

        logger.info(f"Calculated data-driven tiers for {len(tiers)} schools")
    else:
        logger.warning("No CFBD data available — school tiers will be empty")

    return tiers


def get_school_multiplier(school: str) -> float:
    """
    Get NIL multiplier for a school from real CFBD data.
    """
    tiers = get_school_tiers()

    if school in tiers:
        return tiers[school]["multiplier"]

    # Try case-insensitive lookup
    school_lower = school.lower().strip()
    for name, info in tiers.items():
        if name.lower() == school_lower:
            return info["multiplier"]

    # Default for unknown schools
    return 0.8


def get_school_tier(school: str) -> Tuple[str, Dict]:
    """
    Get tier classification for a school from real CFBD data.

    Returns:
        Tuple of (tier_name, tier_info_dict)
    """
    tiers = get_school_tiers()

    if school in tiers:
        return tiers[school]["tier"], tiers[school]

    # Try case-insensitive lookup
    school_lower = school.lower().strip()
    for name, info in tiers.items():
        if name.lower() == school_lower:
            return info["tier"], info

    # Default for unknown schools
    return "g5_mid", {
        "tier": "g5_mid",
        "multiplier": 0.8,
        "label": "Unknown / Not in CFBD data",
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

    # Flat list of all schools sorted by score
    all_schools = []
    for school, info in tiers.items():
        all_schools.append({"school": school, **info})
    all_schools.sort(key=lambda x: x.get("score", 0), reverse=True)

    return {
        "tiers": by_tier,
        "tier_definitions": TIER_DEFINITIONS,
        "total_schools": len(tiers),
        "all_schools": all_schools,
    }


if __name__ == "__main__":
    # Test the tier system
    print("School Tier System Test")
    print("=" * 60)

    tiers = get_school_tiers()

    if not tiers:
        print("WARNING: No CFBD data loaded. Tiers are empty.")
        print("Make sure R2 storage has cfbd_team_records.csv, cfbd_sp_ratings.csv, cfbd_team_talent.csv")
    else:
        # Show top schools by tier
        for tier_name in ["blue_blood", "elite", "power_strong", "power_mid", "g5_strong"]:
            schools_in_tier = [
                (school, info) for school, info in tiers.items()
                if info["tier"] == tier_name
            ]
            schools_in_tier.sort(key=lambda x: x[1]["score"], reverse=True)

            print(f"\n{TIER_DEFINITIONS[tier_name]['label']} ({tier_name}) - {len(schools_in_tier)} schools:")
            for school, info in schools_in_tier[:8]:
                wins_str = f"W{info['wins']}" if info.get("wins") is not None else "W?"
                sp_str = f"SP+{info['sp_plus']}" if info.get("sp_plus") is not None else "SP+?"
                talent_str = f"T{info['talent']}" if info.get("talent") is not None else "T?"
                print(f"  {school:22s}: {info['score']:5.1f} pts, {info['multiplier']:.1f}x | {wins_str} {sp_str} {talent_str}")

        # Summary
        print(f"\n{'=' * 60}")
        print(f"Total schools: {len(tiers)}")
        for tier_name, tier_def in TIER_DEFINITIONS.items():
            count = sum(1 for info in tiers.values() if info["tier"] == tier_name)
            print(f"  {tier_def['label']:35s}: {count:3d} schools ({tier_def['multiplier']:.1f}x)")
