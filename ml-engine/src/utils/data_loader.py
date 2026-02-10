"""
Data Loader Utilities

Unified data loading for ML Engine using Cloudflare R2 storage.
R2 is REQUIRED - no local fallback. All data must be in R2.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

import pandas as pd

from .config import get_config
from .s3_storage import (
    load_csv_with_fallback,
    load_data as s3_load_data,
    is_s3_configured,
    get_s3_client,
    R2NotConfiguredError,
    R2DataLoadError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Core Data Loading Functions
# =============================================================================

def _load_csv(filename: str, subdir: str = "processed", cache_hours: int = 1, **kwargs) -> pd.DataFrame:
    """Load CSV from R2 storage (NO local fallback).

    This is the primary way to load data files.
    R2 storage MUST be configured - there is no local fallback.

    Args:
        filename: CSV filename (e.g., "portal_nil_valuations.csv")
        subdir: Subdirectory in R2 bucket (default: "processed")
        cache_hours: How long to cache R2 data locally
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame

    Raises:
        R2NotConfiguredError: If R2 is not configured
        R2DataLoadError: If data cannot be loaded from R2
    """
    s3_key = f"{subdir}/{filename}"
    return load_csv_with_fallback(s3_key, None, cache_hours, **kwargs)


def _merge_headshots(df: pd.DataFrame) -> pd.DataFrame:
    """Merge headshot_url from multiple sources with fallback chain.

    Priority:
        1. on3_transfer_portal.csv (94%+ coverage for portal players)
        2. on3_all_nil_rankings.csv (backup On3 data)
        3. espn_rosters.csv (ESPN rosters - fills remaining gaps)
    """
    if "name" not in df.columns:
        return df

    # Remove existing headshot_url column if empty
    if "headshot_url" in df.columns:
        if df["headshot_url"].notna().any():
            return df  # Already has headshots
        df = df.drop(columns=["headshot_url"])

    all_headshots = {}

    # Source 1: On3 Transfer Portal (highest priority)
    try:
        source_df = _load_csv("on3_transfer_portal.csv")
        if not source_df.empty and "headshot_url" in source_df.columns and "name" in source_df.columns:
            for _, row in source_df.iterrows():
                name = row.get("name")
                url = row.get("headshot_url")
                if name and pd.notna(url) and str(url).startswith("http"):
                    all_headshots[name] = url
            logger.debug(f"Portal headshots: {len(all_headshots)}")
    except Exception as e:
        logger.warning(f"Failed to load headshots from portal data: {e}")

    # Source 2: On3 NIL Rankings (backup)
    try:
        source_df = _load_csv("on3_all_nil_rankings.csv")
        if not source_df.empty and "headshot_url" in source_df.columns and "name" in source_df.columns:
            for _, row in source_df.iterrows():
                name = row.get("name")
                url = row.get("headshot_url")
                if name and name not in all_headshots and pd.notna(url) and str(url).startswith("http"):
                    all_headshots[name] = url
    except Exception as e:
        logger.warning(f"Failed to load headshots from NIL rankings: {e}")

    # Source 3: ESPN Rosters (fills remaining gaps)
    try:
        espn_df = _load_csv("espn_rosters.csv")
        if not espn_df.empty and "headshot_url" in espn_df.columns and "name" in espn_df.columns:
            for _, row in espn_df.iterrows():
                name = row.get("name")
                url = row.get("headshot_url")
                if name and name not in all_headshots and pd.notna(url) and str(url).startswith("http"):
                    all_headshots[name] = url
    except Exception as e:
        logger.warning(f"Failed to load headshots from ESPN rosters: {e}")

    # Create headshot lookup DataFrame and merge
    if all_headshots:
        headshot_df = pd.DataFrame([
            {"name": name, "headshot_url": url}
            for name, url in all_headshots.items()
        ])
        df = df.merge(headshot_df, on="name", how="left")

    logger.info(f"Merged {len(all_headshots)} headshots")
    return df


# =============================================================================
# NIL Data Functions
# =============================================================================

def get_nil_players(
    position: Optional[str] = None,
    school: Optional[str] = None,
    conference: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    limit: int = 50000,  # Default to all data
) -> pd.DataFrame:
    """Get NIL player data with valuations and headshots.

    Args:
        position: Filter by position (e.g., "QB", "WR")
        school: Filter by school
        conference: Filter by conference
        min_value: Minimum NIL value
        max_value: Maximum NIL value
        limit: Max number of results

    Returns:
        DataFrame with NIL players
    """
    # Try proprietary valuations first
    df = _load_csv("portal_nil_valuations.csv")

    if not df.empty:
        # Standardize column names
        df = df.rename(columns={
            "nil_value_predicted": "nil_value",
            "recruiting_stars": "stars",
            "nil_tier": "tier",
        })

        # Add source indicator
        if "is_predicted" in df.columns:
            df["valuation_source"] = df["is_predicted"].apply(
                lambda x: "Predicted" if x else "On3 Actual"
            )
    else:
        # Fallback to On3 NIL rankings
        df = _load_csv("on3_all_nil_rankings.csv")

        if df.empty:
            logger.warning("No NIL data found")
            return pd.DataFrame()

        # Standardize column names
        df = df.rename(columns={
            "nil_valuation": "nil_value",
            "recruiting_stars": "stars",
            "recruiting_rating": "overall_rating",
        })

        # Add tier based on value
        df["tier"] = df["nil_value"].apply(_get_nil_tier)
        df["valuation_source"] = "On3 Actual"

    # Merge headshots
    df = _merge_headshots(df)

    # Apply filters
    if position and "position" in df.columns:
        df = df[df["position"].str.upper() == position.upper()]

    if school and "school" in df.columns:
        df = df[df["school"].str.contains(school, case=False, na=False)]

    if conference and "conference" in df.columns:
        df = df[df["conference"].str.contains(conference, case=False, na=False)]

    if min_value is not None and "nil_value" in df.columns:
        df = df[df["nil_value"] >= min_value]

    if max_value is not None and "nil_value" in df.columns:
        df = df[df["nil_value"] <= max_value]

    # Sort by NIL value descending
    if "nil_value" in df.columns:
        df = df.sort_values("nil_value", ascending=False)

    return df.head(limit)


def get_nil_leaderboard(
    position: Optional[str] = None,
    school: Optional[str] = None,
    conference: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get NIL leaderboard as list of dicts for API response.

    Args:
        position: Filter by position
        school: Filter by school
        conference: Filter by conference
        limit: Max results

    Returns:
        List of player dicts
    """
    df = get_nil_players(
        position=position,
        school=school,
        conference=conference,
        limit=limit,
    )

    if df.empty:
        return []

    # Select and format columns for API response
    output_cols = [
        "name", "position", "school", "conference", "nil_value",
        "stars", "tier", "headshot_url", "valuation_source",
    ]
    available_cols = [c for c in output_cols if c in df.columns]
    result_df = df[available_cols].copy()

    # Calculate change (placeholder - would need historical data)
    result_df["change"] = 0.0

    # Convert to list of dicts
    return result_df.to_dict(orient="records")


def _get_nil_tier(value: float) -> str:
    """Get NIL tier based on value. Must match calibrated_valuator.py thresholds."""
    if pd.isna(value) or value == 0:
        return "unknown"
    if value >= 2_000_000:
        return "mega"
    if value >= 500_000:
        return "premium"
    if value >= 100_000:
        return "solid"
    if value >= 25_000:
        return "moderate"
    return "entry"


# =============================================================================
# PFF Stats Functions
# =============================================================================

_pff_cache: Optional[pd.DataFrame] = None


def get_pff_grades() -> pd.DataFrame:
    """Load PFF player grades from R2.

    Returns:
        DataFrame with all PFF grades and stats
    """
    global _pff_cache

    if _pff_cache is not None:
        return _pff_cache

    df = _load_csv("pff_player_grades.csv")
    if not df.empty:
        _pff_cache = df
        logger.info(f"Loaded {len(df)} PFF records")
    return df


def get_player_pff_stats(player_name: str, season: int = 2025) -> Optional[Dict[str, Any]]:
    """Get detailed PFF stats for a specific player.

    Args:
        player_name: Player name to search for
        season: Season year (default 2025)

    Returns:
        Dict with PFF stats or None if not found
    """
    df = get_pff_grades()
    if df.empty:
        return None

    player_name_lower = player_name.lower()

    # Filter by season if column exists
    if "season" in df.columns:
        season_df = df[df["season"] == season]
        if season_df.empty:
            season_df = df  # Fall back to all seasons
    else:
        season_df = df

    # Try exact match first
    mask = season_df["name"].str.lower() == player_name_lower
    matches = season_df[mask]

    # Try contains match if no exact match
    if matches.empty:
        mask = season_df["name"].str.lower().str.contains(player_name_lower, na=False)
        matches = season_df[mask]

    if matches.empty:
        return None

    # Get the most recent / highest graded record
    if "pff_overall" in matches.columns:
        row = matches.sort_values("pff_overall", ascending=False).iloc[0]
    else:
        row = matches.iloc[0]

    def safe_float(val):
        """Convert value to float, return None if invalid."""
        if pd.isna(val) or val == "" or val == 0:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # Build comprehensive stats dict
    stats = {
        # Basic info
        "name": str(row.get("name", player_name)),
        "position": str(row.get("position", "")),
        "team": str(row.get("team", "")),
        "season": int(row.get("season", season)) if pd.notna(row.get("season")) else season,
        "games_played": int(row.get("games_played", 0)) if pd.notna(row.get("games_played")) else None,

        # Core PFF grades
        "pff_overall": safe_float(row.get("pff_overall")),
        "pff_offense": safe_float(row.get("pff_offense")),
        "pff_defense": safe_float(row.get("pff_defense")),
        "pff_passing": safe_float(row.get("pff_passing")),
        "pff_rushing": safe_float(row.get("pff_rushing")),
        "pff_receiving": safe_float(row.get("pff_receiving")),
        "pff_pass_block": safe_float(row.get("pff_pass_block")),
        "pff_run_block": safe_float(row.get("pff_run_block")),
        "pff_pass_rush": safe_float(row.get("pff_pass_rush")),
        "pff_coverage": safe_float(row.get("pff_coverage")),
        "pff_run_defense": safe_float(row.get("pff_run_defense")),
        "pff_tackling": safe_float(row.get("pff_tackling")),

        # QB stats
        "passer_rating": safe_float(row.get("passer_rating")),
        "completion_pct": safe_float(row.get("completion_pct")),
        "big_time_throws": safe_float(row.get("big_time_throws")),
        "big_time_throw_pct": safe_float(row.get("big_time_throw_pct")),
        "turnover_worthy_plays": safe_float(row.get("turnover_worthy_plays")),
        "turnover_worthy_play_pct": safe_float(row.get("turnover_worthy_play_pct")),
        "avg_time_to_throw": safe_float(row.get("avg_time_to_throw")),
        "pressure_to_sack_rate": safe_float(row.get("pressure_to_sack_rate")),
        "pressure_completion_pct": safe_float(row.get("pressure_completion_percent")),
        "pressure_qb_rating": safe_float(row.get("pressure_qb_rating")),

        # RB stats
        "elusive_rating": safe_float(row.get("elusive_rating")),
        "yards_after_contact": safe_float(row.get("yards_after_contact")),
        "yaco_per_attempt": safe_float(row.get("yaco_per_attempt")),
        "breakaway_pct": safe_float(row.get("breakaway_pct")),
        "missed_tackles_forced": safe_float(row.get("missed_tackles_forced")),
        "yards": safe_float(row.get("yards")),
        "touchdowns": safe_float(row.get("touchdowns")),
        "yards_per_carry": safe_float(row.get("ypa")),

        # WR/TE stats
        "yards_per_route_run": safe_float(row.get("yards_per_route_run")),
        "drop_rate": safe_float(row.get("drop_rate")),
        "contested_catch_rate": safe_float(row.get("contested_catch_rate")),
        "yards_after_catch": safe_float(row.get("yards_after_catch")),
        "yards_after_catch_per_reception": safe_float(row.get("yards_after_catch_per_reception")),
        "targets": safe_float(row.get("targets")),
        "receptions": safe_float(row.get("receptions")),
        "rec_yards": safe_float(row.get("rec_yards")),
        "routes_run": safe_float(row.get("routes_run")),
        "avg_depth_of_target": safe_float(row.get("avg_depth_of_target")),
        "caught_percent": safe_float(row.get("caught_percent")),

        # Pass rush stats
        "pass_rushing_productivity": safe_float(row.get("pass_rushing_productivity")),
        "pass_rush_win_rate": safe_float(row.get("pass_rush_win_rate")),
        "pressures": safe_float(row.get("pressures")),
        "sacks": safe_float(row.get("sacks")),
        "hurries": safe_float(row.get("hurries")),
        "hits": safe_float(row.get("hits")),

        # Coverage stats
        "passer_rating_allowed": safe_float(row.get("passer_rating_allowed")),
        "yards_per_coverage_snap": safe_float(row.get("yards_per_coverage_snap")),
        "forced_incompletes": safe_float(row.get("forced_incompletes")),
        "forced_incompletion_rate": safe_float(row.get("forced_incompletion_rate")),
        "interceptions": safe_float(row.get("ints")),
        "pass_breakups": safe_float(row.get("pbus")),
        "missed_tackle_rate": safe_float(row.get("missed_tackle_rate")),

        # Blocking stats
        "pass_blocking_efficiency": safe_float(row.get("pass_blocking_efficiency")),
        "pressures_allowed": safe_float(row.get("pressures_allowed")),
        "sacks_allowed": safe_float(row.get("sacks_allowed")),
        "hurries_allowed": safe_float(row.get("hurries_allowed")),
        "run_block_percent": safe_float(row.get("run_block_percent")),

        # Tackling stats
        "tackles": safe_float(row.get("tackles")),
        "assists": safe_float(row.get("assists")),
        "stops": safe_float(row.get("stops")),
        "tackles_for_loss": safe_float(row.get("tackles_for_loss")),
        "missed_tackles": safe_float(row.get("missed_tackles")),

        # =======================================================================
        # HIGH-VALUE UNDERUTILIZED COLUMNS (Added Feb 2026)
        # These are the top predictive metrics we weren't using
        # =======================================================================

        # WR Dominance - How well QB performs when targeting this receiver
        "targeted_qb_rating": safe_float(row.get("targeted_qb_rating")),
        "man_targeted_qb_rating": safe_float(row.get("man_targeted_qb_rating")),
        "zone_targeted_qb_rating": safe_float(row.get("zone_targeted_qb_rating")),

        # True Pass Set (more accurate than general pass block - excludes scrambles)
        "true_pass_set_grades_pass_block": safe_float(row.get("true_pass_set_grades_pass_block")),
        "true_pass_set_pbe": safe_float(row.get("true_pass_set_pbe")),  # Pass block efficiency
        "true_pass_set_pressures_allowed": safe_float(row.get("true_pass_set_pressures_allowed")),
        "true_pass_set_sacks_allowed": safe_float(row.get("true_pass_set_sacks_allowed")),

        # Pressure Situation Performance (key NFL predictor)
        "pressure_grades_pass": safe_float(row.get("pressure_grades_pass")),
        "pressure_yards": safe_float(row.get("pressure_yards")),
        "pressure_first_downs": safe_float(row.get("pressure_first_downs")),
        "no_pressure_completion_pct": safe_float(row.get("no_pressure_completion_percent")),
        "no_pressure_qb_rating": safe_float(row.get("no_pressure_qb_rating")),

        # Blitz Situation Performance
        "blitz_completion_pct": safe_float(row.get("blitz_completion_pct")),
        "blitz_yards_per_attempt": safe_float(row.get("blitz_yards_per_attempt")),
        "blitz_grades_pass": safe_float(row.get("blitz_grades_pass")),
        "no_blitz_completion_pct": safe_float(row.get("no_blitz_completion_pct")),

        # Coverage Efficiency (per-snap is more predictive than per-game)
        "coverage_snaps_per_target": safe_float(row.get("coverage_snaps_per_target")),
        "coverage_snaps_per_reception": safe_float(row.get("coverage_snaps_per_reception")),

        # Man vs Zone Coverage Breakdown (critical for DBs)
        "man_grades_coverage_defense": safe_float(row.get("man_grades_coverage_defense")),
        "man_caught_percent": safe_float(row.get("man_caught_percent")),
        "man_yards_per_coverage_snap": safe_float(row.get("man_yards_per_coverage_snap")),
        "zone_grades_coverage_defense": safe_float(row.get("zone_grades_coverage_defense")),
        "zone_caught_percent": safe_float(row.get("zone_caught_percent")),
        "zone_yards_per_coverage_snap": safe_float(row.get("zone_yards_per_coverage_snap")),

        # Pass Rush by Side (technique specificity)
        "lhs_pass_rush_productivity": safe_float(row.get("lhs_prp")),
        "lhs_pass_rush_win_rate": safe_float(row.get("lhs_pass_rush_percent")),
        "rhs_pass_rush_productivity": safe_float(row.get("rhs_prp")),
        "rhs_pass_rush_win_rate": safe_float(row.get("rhs_pass_rush_percent")),

        # Ball Security / Turnovers
        "forced_fumbles": safe_float(row.get("forced_fumbles")),
        "fumble_recoveries": safe_float(row.get("fumble_recoveries")),
        "fumbles": safe_float(row.get("fumbles")),
        "grades_hands_drop": safe_float(row.get("grades_hands_drop")),
        "grades_hands_fumble": safe_float(row.get("grades_hands_fumble")),

        # Route Distribution (WR specialization)
        "route_rate": safe_float(row.get("route_rate")),
        "slot_rate": safe_float(row.get("slot_rate")),
        "wide_rate": safe_float(row.get("wide_rate")),
        "inline_rate": safe_float(row.get("inline_rate")),

        # Explosive Plays
        "explosive": safe_float(row.get("explosive")),
        "breakaway_yards": safe_float(row.get("breakaway_yards")),
        "breakaway_attempts": safe_float(row.get("breakaway_attempts")),
        "longest": safe_float(row.get("longest")),

        # Snap Count Distribution (usage/versatility)
        "snap_counts_offense": safe_float(row.get("snap_counts_offense")),
        "snap_counts_defense": safe_float(row.get("snap_counts_defense")),
        "snap_counts_coverage": safe_float(row.get("snap_counts_coverage")),
        "snap_counts_pass_rush": safe_float(row.get("snap_counts_pass_rush")),

        # Run Blocking Scheme Analysis (OL evaluation)
        "gap_grades_run_block": safe_float(row.get("gap_grades_run_block")),
        "zone_grades_run_block": safe_float(row.get("zone_grades_run_block")),

        # Penalties (discipline indicator)
        "penalties": safe_float(row.get("penalties")),
        "grades_offense_penalty": safe_float(row.get("grades_offense_penalty")),
        "grades_defense_penalty": safe_float(row.get("grades_defense_penalty")),
    }

    return stats


# =============================================================================
# Transfer Portal Functions
# =============================================================================

def get_portal_players(
    year: int = 2026,
    status: Optional[str] = None,
    position: Optional[str] = None,
    origin_school: Optional[str] = None,
    origin_conference: Optional[str] = None,
    min_stars: Optional[int] = None,
    limit: int = 50000,  # Default to all data
) -> pd.DataFrame:
    """Get transfer portal player data.

    Args:
        year: Portal year (2024, 2025, 2026)
        status: Filter by status (Committed, Entered, etc.)
        position: Filter by position
        origin_school: Filter by origin school
        origin_conference: Filter by origin conference
        min_stars: Minimum star rating
        limit: Max results

    Returns:
        DataFrame with portal players
    """
    df = _load_csv("on3_transfer_portal.csv")

    if df.empty:
        logger.warning("No portal data found")
        return pd.DataFrame()

    # Deduplicate - keep first occurrence, prioritize committed
    if "status" in df.columns:
        status_order = {"Committed": 0, "Entered": 1, "Withdrawn": 2, "Expected": 3}
        df["_status_order"] = df["status"].map(status_order).fillna(99)
        df = df.sort_values(["_status_order", "name"])
        df = df.drop(columns=["_status_order"])

    if "name" in df.columns and "from_school" in df.columns:
        df = df.drop_duplicates(subset=["name", "from_school"], keep="first")
    elif "name" in df.columns:
        df = df.drop_duplicates(subset=["name"], keep="first")

    # Standardize column names
    df = df.rename(columns={
        "nil_valuation": "nil_value_raw",
        "from_school": "origin_school",
        "to_school": "destination_school",
        "rating": "overall_rating",
    })

    # Enrich with calibrated NIL valuations from portal_nil_valuations.csv
    try:
        val_df = _load_csv("portal_nil_valuations.csv")
        if not val_df.empty and "name" in val_df.columns:
            val_df = val_df.rename(columns={
                "nil_value_predicted": "nil_value",
            })
            # Keep only relevant columns for merge
            val_cols = ["name"]
            if "nil_value" in val_df.columns:
                val_cols.append("nil_value")
            if "nil_tier" in val_df.columns:
                val_cols.append("nil_tier")
            if "pff_overall" in val_df.columns:
                val_cols.append("pff_overall")
            if "confidence" in val_df.columns:
                val_cols.append("confidence")

            val_merge = val_df[val_cols].drop_duplicates(subset=["name"], keep="first")

            # Normalize names for matching: strip periods, apostrophes, lowercase
            import re
            def _normalize_name(n):
                if not isinstance(n, str):
                    return ""
                return re.sub(r"[.\-']", "", n).lower().strip()

            df["_name_key"] = df["name"].apply(_normalize_name)
            val_merge["_name_key"] = val_merge["name"].apply(_normalize_name)

            # Drop the 'name' col from val_merge to avoid conflict, merge on normalized key
            val_merge_keyed = val_merge.drop(columns=["name"]).drop_duplicates(subset=["_name_key"], keep="first")
            df = df.merge(val_merge_keyed, on="_name_key", how="left")
            df = df.drop(columns=["_name_key"])

            # Use calibrated value, fall back to raw On3 value
            if "nil_value" in df.columns:
                raw_col = df["nil_value_raw"].fillna(0).astype(float) if "nil_value_raw" in df.columns else 0
                df["nil_value"] = df["nil_value"].fillna(raw_col)
            else:
                df["nil_value"] = df.get("nil_value_raw", 0)

            matched = df["nil_value"].notna().sum()
            logger.info(f"Enriched {matched}/{len(df)} portal players with calibrated valuations")
        else:
            df["nil_value"] = df.get("nil_value_raw", 0)
    except Exception as e:
        logger.warning(f"Failed to enrich portal data with calibrated valuations: {e}")
        df["nil_value"] = df.get("nil_value_raw", 0)

    # Add NIL tier if not already present
    if "nil_tier" not in df.columns and "nil_value" in df.columns:
        df["nil_tier"] = df["nil_value"].apply(lambda v: _get_nil_tier(v) if pd.notna(v) else "entry")

    # Normalize school names for clean display
    if "origin_school" in df.columns:
        df["origin_school"] = df["origin_school"].apply(normalize_school_name)
    if "destination_school" in df.columns:
        df["destination_school"] = df["destination_school"].apply(normalize_school_name)

    # Extract year from source column
    if "source" in df.columns:
        df["portal_year"] = df["source"].str.extract(r"(\d{4})")[0].fillna("2026").astype(int)

    # Merge headshots
    df = _merge_headshots(df)

    # Apply filters
    if year and "portal_year" in df.columns:
        df = df[df["portal_year"] == year]

    if status and "status" in df.columns:
        df = df[df["status"] == status]

    if position and "position" in df.columns:
        df = df[df["position"].str.upper() == position.upper()]

    if origin_school and "origin_school" in df.columns:
        df = df[df["origin_school"].str.contains(origin_school, case=False, na=False)]

    if origin_conference and "conference" in df.columns:
        df = df[df["conference"].str.contains(origin_conference, case=False, na=False)]

    if min_stars and "stars" in df.columns:
        df = df[df["stars"] >= min_stars]

    return df.head(limit)


def get_active_portal_players(
    position: Optional[str] = None,
    status: str = "available",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get active portal players as list of dicts for API.

    Args:
        position: Filter by position
        status: "available", "committed", or "all"
        limit: Max results

    Returns:
        List of player dicts
    """
    # Map status to filter
    status_filter = None
    if status == "available":
        status_filter = "Entered"
    elif status == "committed":
        status_filter = "Committed"

    df = get_portal_players(
        status=status_filter,
        position=position,
        limit=limit,
    )

    if df.empty:
        return []

    # Select columns for API
    output_cols = [
        "name", "position", "origin_school", "destination_school",
        "status", "stars", "nil_value", "headshot_url", "conference",
    ]
    available_cols = [c for c in output_cols if c in df.columns]
    result_df = df[available_cols].copy()

    return result_df.to_dict(orient="records")


def get_team_portal_rankings(year: int = 2026) -> pd.DataFrame:
    """Get team portal rankings."""
    df = _load_csv("on3_team_portal_rankings.csv")

    if df.empty:
        return pd.DataFrame()

    if year and "year" in df.columns:
        df = df[df["year"] == year]

    return df


# =============================================================================
# School Name Normalization
# =============================================================================

# FBS mascot suffixes to strip when normalizing school names
_MASCOT_SUFFIXES = [
    "Crimson Tide", "Fighting Irish", "Nittany Lions", "Yellow Jackets",
    "Sun Devils", "Golden Eagles", "Demon Deacons", "Blue Devils",
    "Red Wolves", "Golden Gophers", "Scarlet Knights", "Horned Frogs",
    "War Eagles", "Mean Green", "Tar Heels",
    "Longhorns", "Bulldogs", "Tigers", "Buckeyes", "Wolverines",
    "Sooners", "Gators", "Volunteers", "Seminoles", "Hurricanes",
    "Trojans", "Wildcats", "Ducks", "Bears", "Aggies", "Jayhawks",
    "Cyclones", "Bruins", "Huskies", "Beavers", "Cougars",
    "Razorbacks", "Commodores", "Rebels", "Mountaineers", "Cavaliers",
    "Cardinals", "Hokies", "Wolfpack", "Panthers", "Owls", "Mustangs",
    "Falcons", "Bobcats", "Rockets", "Redhawks", "Thundering Herd",
    "Miners", "Roadrunners", "Jaguars", "Hilltoppers", "Chanticleers",
    "Broncos", "Aztecs", "Rainbows", "Warriors", "Lobos", "Rams",
    "Cowboys", "Buffaloes", "Utes", "Hawkeyes", "Badgers", "Hoosiers",
    "Boilermakers", "Illini", "Cornhuskers", "Spartans", "Terrapins",
    "Knights", "Bearcats", "Bulls",
]

# Known school name mappings (lowercase → canonical)
_SCHOOL_NAME_MAP = {
    "miami hurricanes": "Miami",
    "miami (fl) hurricanes": "Miami",
    "miami (oh) redhawks": "Miami (OH)",
    "usc trojans": "USC",
    "ole miss rebels": "Ole Miss",
    "lsu tigers": "LSU",
    "ucf knights": "UCF",
    "smu mustangs": "SMU",
    "byu cougars": "BYU",
    "tcu horned frogs": "TCU",
    "usf bulls": "South Florida",
    "uab blazers": "UAB",
    "utsa roadrunners": "UTSA",
    "utep miners": "UTEP",
    "fiu panthers": "FIU",
    "unlv rebels": "UNLV",
}


def normalize_school_name(name) -> str:
    """Normalize school name by stripping mascot suffixes.

    Handles: 'Alabama Crimson Tide' → 'Alabama', 'Ole Miss Rebels' → 'Ole Miss'
    """
    if pd.isna(name) or not name:
        return ""
    name = str(name).strip()

    # Check explicit mapping first
    name_lower = name.lower()
    if name_lower in _SCHOOL_NAME_MAP:
        return _SCHOOL_NAME_MAP[name_lower]

    # Strip mascot suffixes (try longest first to handle multi-word mascots)
    for suffix in sorted(_MASCOT_SUFFIXES, key=len, reverse=True):
        if name.endswith(" " + suffix):
            stripped = name[: -(len(suffix) + 1)].strip()
            if stripped:
                return stripped

    return name


# =============================================================================
# Team Aggregation Functions
# =============================================================================

# Ideal FBS roster composition by position
IDEAL_ROSTER = {
    "QB": 3, "RB": 4, "WR": 6, "TE": 3,
    "OT": 4, "OG": 4, "C": 2, "OL": 10,
    "EDGE": 4, "DT": 4, "DL": 4, "LB": 4,
    "CB": 5, "S": 3, "K": 1, "P": 1,
}

# =============================================================================
# WAR v2 — Performance-First (must match war.ts)
# =============================================================================

# Base position WAR values (expected wins above replacement for elite player)
POSITION_BASE_WAR = {
    "QB": 3.0, "WR": 1.2, "RB": 0.9, "TE": 0.8,
    "OT": 1.0, "OG": 0.7, "C": 0.6,
    "EDGE": 1.5, "CB": 1.2, "S": 0.9, "LB": 1.0,
    "DT": 0.8, "DL": 0.8, "K": 0.4, "P": 0.3, "ATH": 0.8,
}

# Position scarcity multiplier (harder to find quality = higher value)
POSITION_SCARCITY = {
    "QB": 1.4, "EDGE": 1.3, "OT": 1.2, "CB": 1.2,
    "WR": 1.0, "RB": 0.8,
}

# Star adjustment REDUCED: secondary indicator, not primary driver
# Old range: 0.3 to 2.0 (6.7x swing). New: 0.85 to 1.15 (1.35x swing)
STAR_WAR_MULT = {5: 1.15, 4: 1.08, 3: 1.0, 2: 0.93, 1: 0.85, 0: 0.85}

# School tier WAR multipliers (competition level context)
SCHOOL_TIER_WAR = {
    "blue_blood": 1.3, "elite": 1.15, "power_strong": 1.0,
    "power_mid": 0.95, "power_low": 0.9, "g5_strong": 0.85,
    "g5_mid": 0.8, "fcs": 0.7,
}

# PFF estimate from star rating (fallback when no PFF data available)
STAR_PFF_ESTIMATE = {5: 82, 4: 72, 3: 62, 2: 55, 1: 48, 0: 48}

# FBS average starter grade (normalization baseline: grade/65 = multiplier)
FBS_AVG_STARTER_GRADE = 65.0


def calculate_position_performance_score(
    position: str,
    pff_stats: Optional[Dict] = None,
    stars: int = 2,
) -> Tuple[float, str]:
    """Calculate position-specific performance score from PFF stats.

    This is the PRIMARY differentiator in WAR v2.

    Returns:
        Tuple of (performance_multiplier, confidence_type) where confidence_type
        is "measured" if real PFF data, "projected" if estimated from stars.
    """
    pos = position.upper() if position else "ATH"

    if not pff_stats:
        # No PFF data: estimate from stars
        est_grade = STAR_PFF_ESTIMATE.get(stars, 55)
        return est_grade / FBS_AVG_STARTER_GRADE, "projected"

    def _g(key, default=0.0):
        val = pff_stats.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return float(val)

    if pos == "QB":
        pff_pass = _g("pff_passing", _g("pff_offense", _g("pff_overall", 60)))
        pff_rush = _g("pff_rushing", 60)
        comp_pct = _g("completion_pct", 58)
        btt_pct = _g("big_time_throw_pct", 3.5)
        twp_pct = _g("turnover_worthy_play_pct", 3.5)
        comp_score = min(comp_pct / 65.0, 1.5) * 65
        decision_bonus = max(0, (btt_pct - twp_pct)) * 3
        grade = (0.50 * pff_pass + 0.15 * pff_rush
                 + 0.20 * comp_score + 0.15 * min(80, 60 + decision_bonus))

    elif pos in ("WR", "TE"):
        pff_rec = _g("pff_receiving", _g("pff_offense", _g("pff_overall", 60)))
        yprr = _g("yards_per_route_run", 1.2)
        drop = _g("drop_rate", 8.0)
        yprr_score = min(yprr / 1.5, 1.5) * 65
        drop_score = max(40, 80 - drop * 3)
        grade = 0.50 * pff_rec + 0.30 * yprr_score + 0.20 * drop_score

    elif pos == "RB":
        pff_rush = _g("pff_rushing", _g("pff_offense", _g("pff_overall", 60)))
        elusive = _g("elusive_rating", 40)
        yaco = _g("yaco_per_attempt", 2.0)
        elusive_score = min(elusive / 50, 1.5) * 65
        yaco_score = min(yaco / 2.5, 1.5) * 65
        grade = 0.50 * pff_rush + 0.25 * elusive_score + 0.25 * yaco_score

    elif pos in ("EDGE", "DL", "DE"):
        pff_pr = _g("pff_pass_rush", _g("pff_defense", _g("pff_overall", 60)))
        prwr = _g("pass_rush_win_rate", 10)
        prp = _g("pass_rushing_productivity", 5)
        prwr_score = min(prwr / 12.0, 1.5) * 65
        prp_score = min(prp / 6.0, 1.5) * 65
        grade = 0.50 * pff_pr + 0.30 * prwr_score + 0.20 * prp_score

    elif pos == "DT":
        pff_def = _g("pff_run_defense", _g("pff_defense", _g("pff_overall", 60)))
        pff_pr = _g("pff_pass_rush", 55)
        grade = 0.50 * pff_def + 0.50 * pff_pr

    elif pos in ("CB", "S"):
        pff_cov = _g("pff_coverage", _g("pff_defense", _g("pff_overall", 60)))
        fi_rate = _g("forced_incompletion_rate", 8)
        pra = _g("passer_rating_allowed", 90)
        fi_score = min(fi_rate / 10.0, 1.5) * 65
        pra_score = max(40, 90 - (pra - 70) * 0.8)
        grade = 0.50 * pff_cov + 0.25 * fi_score + 0.25 * pra_score

    elif pos in ("OT", "OG", "C", "OL", "IOL"):
        pff_pb = _g("pff_pass_block", _g("pff_offense", _g("pff_overall", 60)))
        pff_rb = _g("pff_run_block", _g("pff_offense", _g("pff_overall", 60)))
        pbe = _g("pass_blocking_efficiency", 95)
        pbe_score = min(pbe / 96.0, 1.3) * 65
        grade = 0.40 * pff_pb + 0.35 * pff_rb + 0.25 * pbe_score

    elif pos == "LB":
        pff_def = _g("pff_defense", _g("pff_overall", 60))
        pff_cov = _g("pff_coverage", 55)
        pff_rd = _g("pff_run_defense", 55)
        grade = 0.40 * pff_def + 0.30 * pff_cov + 0.30 * pff_rd

    else:
        grade = _g("pff_overall", 60)

    return grade / FBS_AVG_STARTER_GRADE, "measured"


def calculate_player_war(
    position: str, stars=None, nil_value=0, school: str = "",
    pff_stats: Optional[Dict] = None,
) -> Dict:
    """Calculate WAR for a single player. Performance-first approach.

    Must match war.ts logic.

    Returns dict with: war, war_low, war_high, confidence, breakdown
    """
    pos = str(position).upper() if position else "ATH"
    stars_int = int(stars) if pd.notna(stars) and stars else 2

    base = POSITION_BASE_WAR.get(pos, 0.8)
    scarcity = POSITION_SCARCITY.get(pos, 1.0)
    star_adj = STAR_WAR_MULT.get(stars_int, 0.93)

    # School tier multiplier
    school_mult = 1.0
    school_tier = "unknown"
    if school:
        try:
            from ..models.school_tiers import get_school_tier
            tier_name, tier_info = get_school_tier(normalize_school_name(school))
            school_tier = tier_name
            school_mult = SCHOOL_TIER_WAR.get(tier_name, 0.9)
        except Exception:
            school_mult = 1.0

    # Performance multiplier (THE PRIMARY DIFFERENTIATOR)
    perf_mult, confidence_type = calculate_position_performance_score(
        pos, pff_stats, stars_int
    )

    # NIL market signal bonus (MINOR — avoid circularity with valuations)
    nil_bonus = 0
    nil_val = float(nil_value) if pd.notna(nil_value) and nil_value else 0
    if nil_val > 0:
        baseline_map = {"QB": 50000, "WR": 20000, "RB": 15000, "EDGE": 15000, "CB": 12000}
        baseline = baseline_map.get(pos, 10000)
        ratio = nil_val / baseline
        if ratio >= 10:
            nil_bonus = 0.15
        elif ratio >= 5:
            nil_bonus = 0.10
        elif ratio >= 2:
            nil_bonus = 0.06
        elif ratio >= 1:
            nil_bonus = 0.03

    raw_war = base * scarcity * perf_mult * school_mult * star_adj + nil_bonus
    war = round(max(0, raw_war), 2)

    # Confidence-based range
    if confidence_type == "measured":
        war_low = round(war * 0.85, 2)
        war_high = round(war * 1.15, 2)
        confidence = "high"
    else:
        war_low = round(war * 0.6, 2)
        war_high = round(war * 1.4, 2)
        confidence = "low"

    return {
        "war": war,
        "war_low": war_low,
        "war_high": war_high,
        "confidence": confidence,
        "breakdown": {
            "base_war": round(base, 2),
            "position_scarcity": round(scarcity, 2),
            "performance_multiplier": round(perf_mult, 2),
            "school_tier": school_tier,
            "school_multiplier": round(school_mult, 2),
            "star_adjustment": round(star_adj, 2),
            "nil_bonus": round(nil_bonus, 2),
            "confidence_type": confidence_type,
        },
    }


def calculate_player_war_score(
    position: str, stars=None, nil_value=0, school: str = "",
    pff_stats: Optional[Dict] = None,
) -> float:
    """Convenience wrapper returning just the WAR float (backward compat)."""
    result = calculate_player_war(position, stars, nil_value, school, pff_stats)
    return result["war"]


def get_team_roster_composition(school: str) -> dict:
    """Get current roster composition by position for a school.

    Returns dict of position → player count.
    """
    df = _load_csv("espn_rosters.csv")
    if df.empty:
        return {}

    # Find school column
    school_col = "team" if "team" in df.columns else "school" if "school" in df.columns else None
    if not school_col:
        return {}

    # Normalize and match
    school_norm = normalize_school_name(school).lower()
    df["_school_norm"] = df[school_col].apply(lambda x: normalize_school_name(x).lower())
    team_df = df[df["_school_norm"] == school_norm]

    if team_df.empty:
        # Try contains match
        team_df = df[df["_school_norm"].str.contains(school_norm, na=False)]

    if team_df.empty or "position" not in team_df.columns:
        return {}

    counts = team_df["position"].str.upper().value_counts().to_dict()
    return counts


def get_team_pff_summary(school: str) -> dict:
    """Get average PFF grades for a school's roster."""
    df = _load_csv("pff_player_grades.csv")
    if df.empty:
        return {"avg_overall": 0, "avg_offense": 0, "avg_defense": 0, "player_count": 0}

    # Find team column
    team_col = None
    for col in ["team", "school", "Team"]:
        if col in df.columns:
            team_col = col
            break
    if not team_col:
        return {"avg_overall": 0, "avg_offense": 0, "avg_defense": 0, "player_count": 0}

    school_norm = normalize_school_name(school).lower()
    df["_team_norm"] = df[team_col].apply(lambda x: normalize_school_name(str(x)).lower() if pd.notna(x) else "")
    team_df = df[df["_team_norm"] == school_norm]

    if team_df.empty:
        team_df = df[df["_team_norm"].str.contains(school_norm, na=False)]

    if team_df.empty:
        return {"avg_overall": 0, "avg_offense": 0, "avg_defense": 0, "player_count": 0}

    result = {"player_count": len(team_df)}
    for col, key in [("pff_overall", "avg_overall"), ("pff_offense", "avg_offense"), ("pff_defense", "avg_defense")]:
        if col in team_df.columns:
            result[key] = round(team_df[col].dropna().mean(), 1) if not team_df[col].dropna().empty else 0
        else:
            result[key] = 0

    return result


def get_team_cfbd_profile(school: str) -> dict:
    """Get CFBD team data: talent, SP+, records."""
    result = {"talent": 0, "sp_overall": 0, "sp_offense": 0, "sp_defense": 0,
              "wins": 0, "losses": 0, "conference": ""}

    school_norm = normalize_school_name(school).lower()

    # Team talent
    talent_df = _load_csv("cfbd_team_talent.csv")
    if not talent_df.empty and "school" in talent_df.columns:
        talent_df["_norm"] = talent_df["school"].apply(lambda x: str(x).lower().strip() if pd.notna(x) else "")
        match = talent_df[talent_df["_norm"] == school_norm]
        if not match.empty and "talent" in match.columns:
            result["talent"] = round(float(match.iloc[0]["talent"]), 1) if pd.notna(match.iloc[0].get("talent")) else 0

    # SP+ ratings
    sp_df = _load_csv("cfbd_sp_ratings.csv")
    if not sp_df.empty and "school" in sp_df.columns:
        sp_df["_norm"] = sp_df["school"].apply(lambda x: str(x).lower().strip() if pd.notna(x) else "")
        match = sp_df[sp_df["_norm"] == school_norm]
        if not match.empty:
            row = match.iloc[0]
            for col in ["sp_overall", "sp_offense", "sp_defense"]:
                if col in row and pd.notna(row[col]):
                    result[col] = round(float(row[col]), 1)
            if "conference" in row and pd.notna(row["conference"]):
                result["conference"] = str(row["conference"])

    # Team records
    records_df = _load_csv("cfbd_team_records.csv")
    if not records_df.empty and "school" in records_df.columns:
        records_df["_norm"] = records_df["school"].apply(lambda x: str(x).lower().strip() if pd.notna(x) else "")
        match = records_df[records_df["_norm"] == school_norm]
        if not match.empty:
            row = match.iloc[0]
            result["wins"] = int(row.get("total_wins", 0)) if pd.notna(row.get("total_wins")) else 0
            result["losses"] = int(row.get("total_losses", 0)) if pd.notna(row.get("total_losses")) else 0
            if not result["conference"] and "conference" in row and pd.notna(row["conference"]):
                result["conference"] = str(row["conference"])

    return result


def get_on3_team_portal_rankings() -> pd.DataFrame:
    """Load On3 team portal rankings for comparison/validation."""
    df = _load_csv("on3_team_portal_rankings.csv")
    if df.empty:
        return pd.DataFrame()
    # Normalize team names for matching
    if "team" in df.columns:
        df["team_normalized"] = df["team"].apply(normalize_school_name)
    return df


def get_roster_needs(school: str, incoming_players=None, outgoing_players=None) -> dict:
    """Calculate position-by-position roster needs for a school.

    Returns dict with needs per position and priority list.
    """
    roster = get_team_roster_composition(school)
    needs = {}
    priority_positions = []

    for pos, ideal in IDEAL_ROSTER.items():
        if pos == "OL":
            continue  # OL is aggregate of OT/OG/C

        current = roster.get(pos, 0)

        # Count incoming/outgoing at this position
        incoming_count = 0
        outgoing_count = 0
        if incoming_players:
            incoming_count = sum(1 for p in incoming_players
                                if str(p.get("position", "")).upper() == pos)
        if outgoing_players:
            outgoing_count = sum(1 for p in outgoing_players
                                if str(p.get("position", "")).upper() == pos)

        adjusted = current - outgoing_count + incoming_count
        net = incoming_count - outgoing_count
        deficit = ideal - adjusted

        if deficit >= 2:
            need_level = "critical"
        elif deficit >= 1:
            need_level = "moderate"
        elif deficit == 0:
            need_level = "low"
        else:
            need_level = "none"

        needs[pos] = {
            "current": current,
            "ideal": ideal,
            "incoming": incoming_count,
            "outgoing": outgoing_count,
            "net": net,
            "adjusted": adjusted,
            "need_level": need_level,
        }

        if need_level in ("critical", "moderate"):
            priority_positions.append(pos)

    return {
        "needs": needs,
        "priority_positions": priority_positions,
    }


# =============================================================================
# Team Logos
# =============================================================================

# Cache for team_id lookup (school name → ESPN team_id)
_team_id_cache: Optional[Dict[str, str]] = None


def _get_team_id_lookup() -> Dict[str, str]:
    """Build a school name → ESPN team_id lookup from espn_rosters.csv.

    Returns dict mapping normalized school name (lowercase) to team_id.
    """
    global _team_id_cache
    if _team_id_cache is not None:
        return _team_id_cache

    _team_id_cache = {}
    try:
        df = _load_csv("espn_rosters.csv")
        if df.empty:
            return _team_id_cache

        # The ESPN rosters have 'team' (with mascot) and 'team_id'
        team_col = "team" if "team" in df.columns else "school"
        if team_col in df.columns and "team_id" in df.columns:
            for _, row in df.drop_duplicates(subset=[team_col]).iterrows():
                raw_name = str(row.get(team_col, ""))
                team_id = row.get("team_id")
                if raw_name and pd.notna(team_id):
                    # Map both raw and normalized names
                    normalized = normalize_school_name(raw_name).lower()
                    _team_id_cache[normalized] = str(int(team_id)) if isinstance(team_id, float) else str(team_id)
                    _team_id_cache[raw_name.lower()] = _team_id_cache[normalized]
    except Exception as e:
        logger.warning(f"Failed to build team ID lookup: {e}")

    return _team_id_cache


def get_team_logo_url(school: str) -> Optional[str]:
    """Get ESPN team logo URL for a school.

    Uses ESPN CDN: https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png

    Args:
        school: School name (with or without mascot)

    Returns:
        Logo URL string or None
    """
    if not school:
        return None

    lookup = _get_team_id_lookup()
    normalized = normalize_school_name(school).lower()
    team_id = lookup.get(normalized) or lookup.get(school.lower())

    if team_id:
        return f"https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png"
    return None


def get_all_team_logos() -> Dict[str, str]:
    """Get all available team logo URLs.

    Returns:
        Dict mapping normalized school name to logo URL.
    """
    lookup = _get_team_id_lookup()
    logos = {}
    seen = set()
    for school_lower, team_id in lookup.items():
        if team_id not in seen:
            # Use the normalized form as key
            canonical = normalize_school_name(school_lower)
            logos[canonical] = f"https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png"
            seen.add(team_id)
    return logos


# =============================================================================
# Player Search
# =============================================================================

def search_players(
    query: str,
    data_type: str = "all",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Search for players by name across datasets.

    Args:
        query: Search query
        data_type: "nil", "portal", or "all"
        limit: Max results

    Returns:
        List of matching player dicts
    """
    results = []

    if data_type in ("nil", "all"):
        nil_df = get_nil_players(limit=500)
        if not nil_df.empty and "name" in nil_df.columns:
            matches = nil_df[nil_df["name"].str.contains(query, case=False, na=False)]
            matches = matches.copy()
            matches["data_source"] = "nil_rankings"
            results.append(matches)

    if data_type in ("portal", "all"):
        portal_df = get_portal_players(limit=500)
        if not portal_df.empty and "name" in portal_df.columns:
            matches = portal_df[portal_df["name"].str.contains(query, case=False, na=False)]
            matches = matches.copy()
            matches["data_source"] = "transfer_portal"
            results.append(matches)

    if results:
        combined = pd.concat(results, ignore_index=True)
        # Dedupe by name, keeping first
        combined = combined.drop_duplicates(subset=["name"], keep="first")
        return combined.head(limit).to_dict(orient="records")

    return []


# =============================================================================
# Statistics & Metadata
# =============================================================================

def get_database_stats() -> Dict[str, Any]:
    """Get real database statistics from loaded data."""
    stats = {
        "total_players": 0,
        "portal_players": 0,
        "nil_valuations": 0,
        "schools": 0,
        "last_updated": None,
    }

    # NIL valuations count
    df = _load_csv("portal_nil_valuations.csv")
    if not df.empty:
        stats["nil_valuations"] = len(df)
        stats["total_players"] = len(df["name"].unique()) if "name" in df.columns else len(df)
    else:
        df = _load_csv("on3_all_nil_rankings.csv")
        if not df.empty:
            stats["nil_valuations"] = len(df)
            stats["total_players"] = len(df["name"].unique()) if "name" in df.columns else len(df)

    # Portal count
    df = _load_csv("on3_transfer_portal.csv")
    if not df.empty:
        stats["portal_players"] = len(df)

    # Schools count
    df = _load_csv("on3_team_portal_rankings.csv")
    if not df.empty and "team" in df.columns:
        stats["schools"] = len(df["team"].unique())

    return stats


def get_positions() -> List[str]:
    """Get list of football positions."""
    return [
        "QB", "RB", "WR", "TE", "OT", "OG", "C", "IOL",
        "EDGE", "DT", "DL", "LB", "CB", "S", "K", "P", "ATH"
    ]


def get_conferences() -> List[str]:
    """Get list of conferences."""
    return [
        "SEC", "Big Ten", "Big 12", "ACC", "Pac-12",
        "Mountain West", "AAC", "Sun Belt", "MAC", "C-USA"
    ]


# =============================================================================
# Dual NIL Valuation (On3 + Portal IQ)
# =============================================================================

def get_player_dual_valuation(player_name: str) -> Optional[Dict[str, Any]]:
    """Get both On3 actual NIL value and Portal IQ predicted value with reasoning.

    Args:
        player_name: Player name to search for

    Returns:
        Dict with on3_value, portal_iq_value, value_breakdown, and reasoning
    """
    # Import CustomNILValuator lazily to avoid circular imports
    import sys
    import importlib.util
    from pathlib import Path

    # Get the path to custom_nil_valuator.py
    models_path = Path(__file__).parent.parent / "models" / "custom_nil_valuator.py"
    spec = importlib.util.spec_from_file_location("custom_nil_valuator", models_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    CustomNILValuator = module.CustomNILValuator

    # Find player in NIL data
    df = get_nil_players(limit=50000)
    if df.empty:
        return None

    player_name_lower = player_name.lower()
    mask = df["name"].str.lower() == player_name_lower
    matches = df[mask]

    if matches.empty:
        # Try contains match
        mask = df["name"].str.lower().str.contains(player_name_lower, na=False)
        matches = df[mask]
        if matches.empty:
            return None

    player_row = matches.iloc[0]

    # Get On3 actual value (if this player has one)
    nil_value = float(player_row.get("nil_value", 0)) if pd.notna(player_row.get("nil_value")) else 0
    is_predicted = player_row.get("is_predicted", True)

    on3_value = nil_value if not is_predicted else None

    # Calculate Portal IQ predicted value using our formula
    # Use calibration factor to align with market (On3 valuations)
    valuator = CustomNILValuator(calibration_factor=1.8)

    # Get PFF stats if available - these contain the REAL production numbers
    pff_stats = get_player_pff_stats(player_name)

    # Get player attributes
    position = str(player_row.get("position", "ATH"))
    school = str(player_row.get("school", ""))
    stars = int(player_row.get("stars", 0)) if pd.notna(player_row.get("stars")) else 0

    # Extract actual production stats from PFF data
    games_played = 0
    passing_yards = 0
    passing_tds = 0
    rushing_yards = 0
    rushing_tds = 0
    receiving_yards = 0
    receiving_tds = 0
    tackles = 0
    sacks = 0
    interceptions = 0
    pff_grade = None

    if pff_stats:
        games_played = int(pff_stats.get("games_played") or 12)

        # Use position-specific grades (more accurate than overall)
        if position == "QB":
            pff_grade = pff_stats.get("pff_passing") or pff_stats.get("pff_offense") or pff_stats.get("pff_overall")
        elif position in ("WR", "TE"):
            pff_grade = pff_stats.get("pff_receiving") or pff_stats.get("pff_offense") or pff_stats.get("pff_overall")
        elif position == "RB":
            pff_grade = pff_stats.get("pff_rushing") or pff_stats.get("pff_offense") or pff_stats.get("pff_overall")
        elif position in ("EDGE", "DT", "DL", "DE"):
            pff_grade = pff_stats.get("pff_pass_rush") or pff_stats.get("pff_defense") or pff_stats.get("pff_overall")
        elif position in ("CB", "S"):
            pff_grade = pff_stats.get("pff_coverage") or pff_stats.get("pff_defense") or pff_stats.get("pff_overall")
        elif position in ("OT", "OG", "C", "OL", "IOL"):
            pff_grade = pff_stats.get("pff_pass_block") or pff_stats.get("pff_offense") or pff_stats.get("pff_overall")
        else:
            pff_grade = pff_stats.get("pff_overall")

        # Get actual production stats
        receiving_yards = int(pff_stats.get("rec_yards") or 0)
        receiving_tds = int(pff_stats.get("touchdowns") or 0) if position in ("WR", "TE", "RB") else 0
        rushing_yards = int(pff_stats.get("yards") or 0) if position in ("RB", "QB") else 0
        rushing_tds = int(pff_stats.get("touchdowns") or 0) if position == "RB" else 0
        passing_yards = int(pff_stats.get("yards") or 0) if position == "QB" else 0
        passing_tds = int(pff_stats.get("touchdowns") or 0) if position == "QB" else 0
        sacks = float(pff_stats.get("sacks") or 0)
        interceptions = int(pff_stats.get("interceptions") or 0) if position in ("CB", "S", "LB") else 0
        tackles = int(pff_stats.get("tackles") or 0)

    # Determine if starter based on production/grade
    is_starter = False
    if pff_grade and pff_grade > 70:
        is_starter = True
    elif receiving_yards > 500 or rushing_yards > 400 or passing_yards > 1500:
        is_starter = True
    elif sacks > 3 or interceptions > 2:
        is_starter = True

    # Calculate Portal IQ valuation with FULL stats
    valuation = valuator.calculate_valuation(
        player_name=player_name,
        position=position,
        school=school,
        games_played=games_played,
        games_started=games_played if is_starter else 0,
        passing_yards=passing_yards,
        passing_tds=passing_tds,
        rushing_yards=rushing_yards,
        rushing_tds=rushing_tds,
        receiving_yards=receiving_yards,
        receiving_tds=receiving_tds,
        tackles=tackles,
        sacks=sacks,
        interceptions=interceptions,
        pff_grade=pff_grade,
        recruiting_stars=stars,
        is_starter=is_starter,
    )

    portal_iq_value = valuation.total_valuation

    # Build value breakdown for reasoning
    value_breakdown = {
        "position_base": valuation.factors.get("position_base", 0),
        "performance_multiplier": round(valuation.factors.get("performance_multiplier", 1.0), 2),
        "school_multiplier": round(valuation.factors.get("school_multiplier", 1.0), 2),
        "social_value": valuation.factors.get("social_value", 0),
        "potential_value": valuation.factors.get("potential_value", 0),
        "starter_bonus": round(valuation.factors.get("starter_bonus", 1.0), 2),
    }

    # Generate human-readable reasoning
    reasoning_parts = []

    # Position reasoning
    reasoning_parts.append(f"{position} position base value: ${value_breakdown['position_base']:,.0f}")

    # School reasoning
    school_mult = value_breakdown['school_multiplier']
    if school_mult >= 2.5:
        reasoning_parts.append(f"{school} is a blue-blood program ({school_mult:.1f}x multiplier)")
    elif school_mult >= 1.8:
        reasoning_parts.append(f"{school} is an elite program ({school_mult:.1f}x multiplier)")
    elif school_mult >= 1.3:
        reasoning_parts.append(f"{school} is a strong Power 4 program ({school_mult:.1f}x multiplier)")
    elif school_mult < 1.0:
        reasoning_parts.append(f"{school} has limited NIL market ({school_mult:.1f}x multiplier)")

    # Performance reasoning
    perf_mult = value_breakdown['performance_multiplier']
    if pff_grade:
        if pff_grade >= 90:
            reasoning_parts.append(f"Elite PFF grade ({pff_grade:.1f}) adds significant value")
        elif pff_grade >= 80:
            reasoning_parts.append(f"Strong PFF grade ({pff_grade:.1f}) boosts valuation")
        elif pff_grade >= 70:
            reasoning_parts.append(f"Solid PFF grade ({pff_grade:.1f}) supports value")
        else:
            reasoning_parts.append(f"PFF grade ({pff_grade:.1f}) limits upside")

    # Recruiting reasoning
    if stars >= 5:
        reasoning_parts.append(f"5-star recruit commands premium floor value")
    elif stars >= 4:
        reasoning_parts.append(f"4-star recruit has strong market appeal")

    # Compare values if we have both
    if on3_value and portal_iq_value:
        diff = portal_iq_value - on3_value
        diff_pct = (diff / on3_value) * 100 if on3_value > 0 else 0

        if abs(diff_pct) < 10:
            reasoning_parts.append(f"Portal IQ value closely matches On3 ({diff_pct:+.1f}%)")
        elif diff > 0:
            reasoning_parts.append(f"Portal IQ values {player_name} higher than On3 ({diff_pct:+.1f}%) based on performance metrics")
        else:
            reasoning_parts.append(f"On3 values {player_name} higher ({-diff_pct:.1f}%) - likely due to social media/brand factors not in our model")

    return {
        "player_name": player_name,
        "position": position,
        "school": school,
        "on3_value": on3_value,
        "portal_iq_value": portal_iq_value,
        "portal_iq_tier": valuation.valuation_tier,
        "confidence": valuation.confidence,
        "value_breakdown": value_breakdown,
        "reasoning": reasoning_parts,
        "has_on3_data": on3_value is not None,
    }


# =============================================================================
# Legacy DataLoader Class (for backwards compatibility)
# =============================================================================

class DataLoader:
    """Unified data loader for ML Engine."""

    def __init__(self):
        self.config = get_config()
        self.data_dir = self.config.data_dir
        self.cache_dir = self.config.cache_dir

    def load_csv(
        self,
        filename: str,
        subdir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a CSV file from S3 or local data directory.

        Args:
            filename: Name of the CSV file
            subdir: Optional subdirectory within data_dir

        Returns:
            DataFrame with the loaded data
        """
        if subdir:
            return _load_csv(filename, subdir=subdir)
        else:
            return _load_csv(filename, subdir="processed")

    def load_parquet(
        self,
        filename: str,
        subdir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a Parquet file from the data directory.

        Args:
            filename: Name of the Parquet file
            subdir: Optional subdirectory within data_dir

        Returns:
            DataFrame with the loaded data
        """
        if subdir:
            path = self.data_dir / subdir / filename
        else:
            path = self.data_dir / filename

        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()

        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return pd.DataFrame()

    def save_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        subdir: Optional[str] = None,
    ) -> bool:
        """
        Save a DataFrame to CSV.

        Args:
            df: DataFrame to save
            filename: Name of the output file
            subdir: Optional subdirectory within data_dir

        Returns:
            True if successful, False otherwise
        """
        if subdir:
            path = self.data_dir / subdir
        else:
            path = self.data_dir

        path.mkdir(parents=True, exist_ok=True)
        filepath = path / filename

        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} rows to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
            return False

    def save_parquet(
        self,
        df: pd.DataFrame,
        filename: str,
        subdir: Optional[str] = None,
    ) -> bool:
        """
        Save a DataFrame to Parquet.

        Args:
            df: DataFrame to save
            filename: Name of the output file
            subdir: Optional subdirectory within data_dir

        Returns:
            True if successful, False otherwise
        """
        if subdir:
            path = self.data_dir / subdir
        else:
            path = self.data_dir

        path.mkdir(parents=True, exist_ok=True)
        filepath = path / filename

        try:
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(df)} rows to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
            return False

    def load_processed_data(self, product: str) -> Dict[str, pd.DataFrame]:
        """
        Load all processed data for a product.

        Args:
            product: 'portal_iq' or 'cap_iq'

        Returns:
            Dictionary of DataFrames
        """
        processed_dir = self.data_dir / "processed" / product

        if not processed_dir.exists():
            logger.warning(f"Processed directory not found: {processed_dir}")
            return {}

        data = {}
        for file in processed_dir.glob("*.parquet"):
            name = file.stem
            data[name] = pd.read_parquet(file)
            logger.info(f"Loaded {name}: {len(data[name])} rows")

        return data

    def get_available_years(self, data_type: str) -> List[int]:
        """
        Get available years for a data type.

        Args:
            data_type: Type of data (e.g., 'cfb_stats', 'nfl_contracts')

        Returns:
            List of available years
        """
        cache_pattern = f"*{data_type}*.csv"
        files = list(self.cache_dir.glob(cache_pattern))

        years = set()
        for f in files:
            # Extract years from filename patterns like "cfb_stats_2020_2024_cache.csv"
            parts = f.stem.split("_")
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    years.add(int(part))

        return sorted(years)

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cached files.

        Args:
            pattern: Optional glob pattern to match specific files

        Returns:
            Number of files deleted
        """
        if pattern:
            files = list(self.cache_dir.glob(pattern))
        else:
            files = list(self.cache_dir.glob("*"))

        deleted = 0
        for f in files:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Could not delete {f}: {e}")

        logger.info(f"Cleared {deleted} cache files")
        return deleted
