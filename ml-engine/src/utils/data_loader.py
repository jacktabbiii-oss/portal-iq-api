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
    """Get NIL tier based on value."""
    if pd.isna(value) or value == 0:
        return "unknown"
    if value >= 1_000_000:
        return "mega"
    if value >= 500_000:
        return "premium"
    if value >= 200_000:
        return "established"
    if value >= 50_000:
        return "emerging"
    return "developing"


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
        "nil_valuation": "nil_value",
        "from_school": "origin_school",
        "to_school": "destination_school",
        "rating": "overall_rating",
    })

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
