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
# PFF Detailed Stats Functions
# =============================================================================

def get_pff_player_stats(player_name: str, season: int = 2025) -> Optional[Dict[str, Any]]:
    """Get detailed PFF stats for a specific player including box score stats.

    Loads from pff_player_grades.csv which has 120+ columns including
    actual counting stats (yards, TDs, receptions, etc.) not just grades.

    Args:
        player_name: Player name to search for
        season: Season year (default 2025)

    Returns:
        Dict with all available stats, or None if player not found
    """
    try:
        df = _load_csv("pff_player_grades.csv")
        if df.empty:
            return None

        # Search by name (case-insensitive)
        name_lower = player_name.lower()
        mask = df["name"].str.lower() == name_lower if "name" in df.columns else pd.Series(dtype=bool)

        # Filter by season if available
        if "season" in df.columns and not mask.empty:
            season_mask = mask & (df["season"] == season)
            matches = df[season_mask]
            if matches.empty:
                # Try without season filter
                matches = df[mask]
        else:
            matches = df[mask]

        if matches.empty:
            # Try partial match
            mask = df["name"].str.lower().str.contains(name_lower, na=False) if "name" in df.columns else pd.Series(dtype=bool)
            matches = df[mask]

        if matches.empty:
            return None

        # Use the most recent / highest-graded entry
        if "pff_overall" in matches.columns:
            row = matches.sort_values("pff_overall", ascending=False).iloc[0]
        else:
            row = matches.iloc[0]

        # Convert to dict, replacing NaN with None
        stats = {}
        for col in row.index:
            val = row[col]
            if pd.isna(val):
                stats[col] = None
            elif isinstance(val, (int, float)):
                stats[col] = float(val)
            else:
                stats[col] = str(val)

        return stats

    except (R2NotConfiguredError, R2DataLoadError):
        logger.warning("Could not load PFF stats from R2")
        return None
    except Exception as e:
        logger.error(f"Error loading PFF stats for {player_name}: {e}")
        return None


# =============================================================================
# Portal IQ Valuation Calculator
# =============================================================================

def calculate_portal_iq_value(
    position: str,
    school: str,
    pff_overall: float = 0,
    stars: int = 0,
) -> Dict[str, Any]:
    """Calculate Portal IQ's proprietary NIL valuation.

    Uses position market value, school brand power, on-field performance,
    and recruiting profile to calculate a fair market NIL value.

    Returns dict with: value, tier, breakdown, reasoning
    """
    position = position.upper() if position else ""

    # Position base value (reflects market demand for each position)
    position_values = {
        "QB": 200000, "WR": 100000, "RB": 80000, "TE": 60000,
        "OT": 70000, "OG": 50000, "C": 50000, "OL": 60000,
        "EDGE": 80000, "DT": 60000, "DE": 80000, "DL": 65000,
        "LB": 60000, "CB": 80000, "S": 60000,
        "K": 15000, "P": 12000, "LS": 10000,
    }
    position_base = position_values.get(position, 40000)

    # School brand multiplier (reflects NIL market size)
    # Use startswith matching to handle "Penn State Nittany Lions" -> "Penn State"
    tier1_schools = [
        "Ohio State", "Alabama", "Georgia", "Michigan", "Texas", "USC",
        "LSU", "Oregon", "Clemson", "Notre Dame", "Penn State", "Tennessee",
        "Florida", "Oklahoma", "Miami", "Texas A&M",
    ]
    tier2_schools = [
        "Auburn", "Wisconsin", "Arkansas", "Iowa", "Ole Miss", "NC State",
        "Missouri", "Kentucky", "South Carolina", "Colorado", "Nebraska",
        "Michigan State", "UCLA", "Washington", "Arizona", "Utah",
        "Virginia Tech", "Louisville", "Pittsburgh", "Florida State",
        "Baylor", "Kansas State", "TCU", "Iowa State", "Illinois",
        "Maryland", "Minnesota", "Oregon State", "Cal", "Stanford",
        "West Virginia", "Syracuse", "Duke", "Wake Forest", "Georgia Tech",
        "North Carolina", "Boston College", "Virginia", "Vanderbilt",
        "Mississippi State", "Purdue", "Indiana", "Rutgers", "Northwestern",
    ]

    # Normalize school name (strip mascot suffixes)
    school_normalized = school.strip()

    def _school_matches(school_name: str, school_list: list) -> bool:
        """Check if school matches any in the list (handles mascot suffixes)."""
        for s in school_list:
            if school_name == s or school_name.startswith(s + " "):
                return True
        return False

    if _school_matches(school_normalized, tier1_schools):
        school_mult = 3.5
    elif _school_matches(school_normalized, tier2_schools):
        school_mult = 2.0
    else:
        school_mult = 0.8  # G5/FCS

    # Performance multiplier (PFF grade)
    if pff_overall >= 90:
        perf_mult = 3.0
    elif pff_overall >= 80:
        perf_mult = 2.2
    elif pff_overall >= 70:
        perf_mult = 1.5
    elif pff_overall >= 60:
        perf_mult = 1.0
    elif pff_overall > 0:
        perf_mult = 0.6
    else:
        # No PFF data - estimate from star rating
        perf_mult = {5: 1.5, 4: 1.2, 3: 0.9, 2: 0.6}.get(stars, 0.4)

    # Star/recruiting multiplier
    star_mult_values = {5: 4.0, 4: 2.0, 3: 1.3, 2: 0.7, 1: 0.3}
    star_mult = star_mult_values.get(stars, 0.2) if stars > 0 else 0.2

    # Core calculated value
    core_value = position_base * school_mult * perf_mult * star_mult

    # Social media value estimate (based on profile, not actual follower counts)
    if stars > 0:
        social_value = min(int(stars * 50000 * (school_mult / 3.5)), 500000)
    else:
        social_value = 5000

    # Potential premium (recruiting ceiling)
    potential_values = {5: 150000, 4: 75000, 3: 25000, 2: 10000}
    potential_value = potential_values.get(stars, 5000)

    # Final value
    total_value = int(core_value + social_value + potential_value)

    # Determine tier
    tier = _get_nil_tier(total_value)

    # Generate reasoning
    reasoning = []
    if position in ("QB", "WR", "CB", "EDGE", "DE", "RB"):
        reasoning.append(f"{position} is a premium NIL position with high market demand")
    if _school_matches(school_normalized, tier1_schools):
        reasoning.append(f"{school} is a Tier 1 NIL market with elite brand value")
    elif _school_matches(school_normalized, tier2_schools):
        reasoning.append(f"{school} is a strong Power conference program")
    if pff_overall >= 80:
        reasoning.append(f"Elite on-field performance (grade: {pff_overall:.1f}) commands premium valuation")
    elif pff_overall >= 70:
        reasoning.append(f"Strong on-field performance (grade: {pff_overall:.1f}) supports higher valuation")
    elif pff_overall >= 60:
        reasoning.append(f"Solid on-field production (grade: {pff_overall:.1f})")
    if stars >= 5:
        reasoning.append("5-star recruit with maximum recruiting premium and national profile")
    elif stars >= 4:
        reasoning.append("4-star recruit with strong recruiting pedigree")
    elif stars >= 3:
        reasoning.append("3-star recruit with development potential")

    return {
        "value": total_value,
        "tier": tier,
        "breakdown": {
            "position_base": position_base,
            "school_multiplier": round(school_mult, 2),
            "performance_multiplier": round(perf_mult, 2),
            "star_multiplier": round(star_mult, 2),
            "social_value": social_value,
            "potential_value": potential_value,
        },
        "reasoning": reasoning,
    }


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
    # Try current cycle first (has active + committed players)
    df = _load_csv("on3_transfer_portal_current.csv")
    if df.empty:
        # Fallback to historical data
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
