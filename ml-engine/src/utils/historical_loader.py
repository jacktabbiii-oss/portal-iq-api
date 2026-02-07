"""
Historical Data Loader for Portal IQ

Loads NFL, historical college, and combine data from R2 storage.
This module extends the base s3_storage to support multi-year datasets.

Data Structure in R2:
    historical/
        nfl/
            2021_nfl_stats.csv
            2022_nfl_stats.csv
            ...
        college/
            2021_fbs_stats.csv
            2022_fcs_stats.csv
            ...
    combine/
        2020_combine.csv
        2021_combine.csv
        ...
"""

import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache

import pandas as pd

from .s3_storage import (
    load_csv_with_fallback,
    get_s3_client,
    is_s3_configured,
    R2DataLoadError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NFL Data Loading
# =============================================================================

def load_nfl_stats(season: int) -> pd.DataFrame:
    """Load NFL performance data for a single season.

    Args:
        season: Year (e.g., 2021, 2022, ..., 2025)

    Returns:
        DataFrame with NFL player stats
    """
    s3_key = f"historical/nfl/{season}_nfl_stats.csv"

    try:
        df = load_csv_with_fallback(s3_key, cache_hours=24)
        df["season"] = season
        df["league"] = "NFL"
        logger.info(f"Loaded NFL {season}: {len(df)} players")
        return df
    except R2DataLoadError:
        logger.warning(f"NFL data not found for season {season}")
        return pd.DataFrame()


def load_all_nfl_stats(start_year: int = 2021, end_year: int = 2025) -> pd.DataFrame:
    """Load all available NFL stats across seasons.

    Args:
        start_year: First season to load
        end_year: Last season to load

    Returns:
        Combined DataFrame with all NFL stats
    """
    all_data = []

    for year in range(start_year, end_year + 1):
        df = load_nfl_stats(year)
        if not df.empty:
            all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined)} total NFL records ({start_year}-{end_year})")
        return combined

    return pd.DataFrame()


# =============================================================================
# College Historical Data Loading
# =============================================================================

def load_college_stats(season: int, division: str = "fbs") -> pd.DataFrame:
    """Load college performance data for a single season.

    Args:
        season: Year (e.g., 2021, 2022, ..., 2025)
        division: "fbs" or "fcs"

    Returns:
        DataFrame with college player stats
    """
    division = division.lower()
    s3_key = f"historical/college/{season}_{division}_stats.csv"

    try:
        df = load_csv_with_fallback(s3_key, cache_hours=24)
        df["season"] = season
        df["division"] = division.upper()
        df["league"] = "NCAA"
        logger.info(f"Loaded {division.upper()} {season}: {len(df)} players")
        return df
    except R2DataLoadError:
        logger.warning(f"College data not found for {division.upper()} {season}")
        return pd.DataFrame()


def load_all_college_stats(
    start_year: int = 2021,
    end_year: int = 2025,
    include_fcs: bool = True
) -> pd.DataFrame:
    """Load all available college stats across seasons.

    Args:
        start_year: First season to load
        end_year: Last season to load
        include_fcs: Whether to include FCS data

    Returns:
        Combined DataFrame with all college stats
    """
    all_data = []

    for year in range(start_year, end_year + 1):
        # Always try FBS
        fbs_df = load_college_stats(year, "fbs")
        if not fbs_df.empty:
            all_data.append(fbs_df)

        # Optionally include FCS
        if include_fcs:
            fcs_df = load_college_stats(year, "fcs")
            if not fcs_df.empty:
                all_data.append(fcs_df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined)} total college records ({start_year}-{end_year})")
        return combined

    return pd.DataFrame()


# =============================================================================
# Combine Data Loading
# =============================================================================

def load_combine_data(year: Optional[int] = None) -> pd.DataFrame:
    """Load NFL combine/pro day data.

    Args:
        year: Specific year, or None for all years

    Returns:
        DataFrame with combine measurables
    """
    if year:
        s3_key = f"combine/{year}_combine.csv"
        try:
            df = load_csv_with_fallback(s3_key, cache_hours=168)  # Cache for a week
            df["combine_year"] = year
            logger.info(f"Loaded {year} combine: {len(df)} players")
            return df
        except R2DataLoadError:
            logger.warning(f"Combine data not found for {year}")
            return pd.DataFrame()

    # Load all years
    return load_all_combine_data()


def load_all_combine_data(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """Load all available combine data.

    Args:
        start_year: First year to load
        end_year: Last year to load

    Returns:
        Combined DataFrame with all combine data
    """
    all_data = []

    for year in range(start_year, end_year + 1):
        s3_key = f"combine/{year}_combine.csv"
        try:
            df = load_csv_with_fallback(s3_key, cache_hours=168)
            df["combine_year"] = year
            all_data.append(df)
        except R2DataLoadError:
            continue

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined)} total combine records ({start_year}-{end_year})")
        return combined

    return pd.DataFrame()


# =============================================================================
# Player Career Functions
# =============================================================================

def get_player_career_stats(
    player_name: str,
    include_nfl: bool = True,
    include_college: bool = True,
) -> Dict[str, Any]:
    """Get complete career stats for a player across college and NFL.

    Args:
        player_name: Player name to search
        include_nfl: Include NFL data
        include_college: Include college data

    Returns:
        Dict with career stats by season and league
    """
    career = {
        "player_name": player_name,
        "college_seasons": [],
        "nfl_seasons": [],
        "combine_data": None,
        "total_seasons": 0,
    }

    # Search college stats
    if include_college:
        college_df = load_all_college_stats()
        if not college_df.empty and "name" in college_df.columns:
            player_college = college_df[
                college_df["name"].str.contains(player_name, case=False, na=False)
            ]
            if not player_college.empty:
                career["college_seasons"] = player_college.to_dict(orient="records")

    # Search NFL stats
    if include_nfl:
        nfl_df = load_all_nfl_stats()
        if not nfl_df.empty and "name" in nfl_df.columns:
            player_nfl = nfl_df[
                nfl_df["name"].str.contains(player_name, case=False, na=False)
            ]
            if not player_nfl.empty:
                career["nfl_seasons"] = player_nfl.to_dict(orient="records")

    # Get combine data
    combine_df = load_all_combine_data()
    if not combine_df.empty and "name" in combine_df.columns:
        player_combine = combine_df[
            combine_df["name"].str.contains(player_name, case=False, na=False)
        ]
        if not player_combine.empty:
            career["combine_data"] = player_combine.iloc[0].to_dict()

    career["total_seasons"] = len(career["college_seasons"]) + len(career["nfl_seasons"])

    return career


def get_player_measurables(player_name: str) -> Dict[str, Any]:
    """Get combine/pro day measurables for a player.

    Args:
        player_name: Player name to search

    Returns:
        Dict with measurable data (height, weight, forty, etc.)
    """
    combine_df = load_all_combine_data()

    if combine_df.empty or "name" not in combine_df.columns:
        return {}

    # Try exact match first
    player_data = combine_df[combine_df["name"].str.lower() == player_name.lower()]

    # Fall back to contains match
    if player_data.empty:
        player_data = combine_df[
            combine_df["name"].str.contains(player_name, case=False, na=False)
        ]

    if player_data.empty:
        return {}

    # Return most recent combine data
    row = player_data.sort_values("combine_year", ascending=False).iloc[0]

    # Standard measurable columns
    measurable_cols = [
        "height", "weight", "forty", "vertical", "broad_jump",
        "bench", "three_cone", "shuttle", "arm_length", "hand_size",
        "position", "school", "combine_year"
    ]

    return {col: row.get(col) for col in measurable_cols if col in row.index}


# =============================================================================
# Available Data Discovery
# =============================================================================

def list_available_historical_data() -> Dict[str, List[int]]:
    """List what historical data is available in R2.

    Returns:
        Dict with available years by category
    """
    client = get_s3_client()
    available = {
        "nfl_seasons": [],
        "fbs_seasons": [],
        "fcs_seasons": [],
        "combine_years": [],
    }

    if not is_s3_configured():
        return available

    # Check NFL data
    nfl_files = client.list_files("historical/nfl/")
    for f in nfl_files:
        if f.endswith("_nfl_stats.csv"):
            try:
                year = int(f.split("/")[-1].split("_")[0])
                available["nfl_seasons"].append(year)
            except (ValueError, IndexError):
                continue

    # Check college data
    college_files = client.list_files("historical/college/")
    for f in college_files:
        try:
            parts = f.split("/")[-1].split("_")
            year = int(parts[0])
            division = parts[1].lower()
            if division == "fbs":
                available["fbs_seasons"].append(year)
            elif division == "fcs":
                available["fcs_seasons"].append(year)
        except (ValueError, IndexError):
            continue

    # Check combine data
    combine_files = client.list_files("combine/")
    for f in combine_files:
        if f.endswith("_combine.csv"):
            try:
                year = int(f.split("/")[-1].split("_")[0])
                available["combine_years"].append(year)
            except (ValueError, IndexError):
                continue

    # Sort all lists
    for key in available:
        available[key] = sorted(available[key])

    return available


# =============================================================================
# Position Group Mapping
# =============================================================================

POSITION_GROUPS = {
    "QB": ["QB"],
    "RB": ["RB", "FB", "HB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "OL": ["OT", "OG", "C", "IOL", "OL", "T", "G"],
    "EDGE": ["EDGE", "DE", "OLB", "RUSH"],
    "DL": ["DT", "NT", "DL", "IDL"],
    "LB": ["LB", "ILB", "MLB", "WILL", "MIKE", "SAM"],
    "CB": ["CB"],
    "S": ["S", "FS", "SS", "DB"],
    "K": ["K", "PK"],
    "P": ["P"],
}


def normalize_position(position: str) -> str:
    """Normalize a position to its position group.

    Args:
        position: Raw position string

    Returns:
        Normalized position group
    """
    pos = position.upper().strip()

    for group, positions in POSITION_GROUPS.items():
        if pos in positions:
            return group

    return pos


def get_position_group(position: str) -> str:
    """Alias for normalize_position."""
    return normalize_position(position)
