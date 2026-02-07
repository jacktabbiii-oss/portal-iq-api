"""
Player Similarity Engine for Portal IQ

Finds comparable players using position-specific feature vectors and cosine similarity.
Supports finding NFL player comps for draft prediction and college player comps for
transfer portal analysis.

Key Features:
- Position-specific feature sets (QB, RB, WR, etc.)
- Multi-source data integration (performance + measurables)
- Weighted similarity with recency bias
- NFL outcome tracking for predictive comps
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.historical_loader import (
    load_all_nfl_stats,
    load_all_college_stats,
    load_all_combine_data,
    get_player_measurables,
    normalize_position,
)
from ..utils.data_loader import get_nil_players, _load_csv

logger = logging.getLogger(__name__)


# =============================================================================
# Position-Specific Feature Sets
# =============================================================================

POSITION_FEATURES = {
    "QB": {
        "performance": [
            "passer_rating", "completion_pct", "yards_per_attempt",
            "big_time_throw_pct", "turnover_worthy_pct", "pressure_to_sack_pct",
            "adjusted_completion_pct", "avg_depth_of_target",
        ],
        "measurables": [
            "height", "weight", "forty", "vertical", "broad_jump",
        ],
        "weights": {"performance": 0.7, "measurables": 0.3},
    },
    "RB": {
        "performance": [
            "yards_per_carry", "elusive_rating", "yards_after_contact",
            "breakaway_pct", "forced_missed_tackles", "receiving_grade",
            "pass_block_grade",
        ],
        "measurables": [
            "height", "weight", "forty", "vertical", "bench", "broad_jump",
        ],
        "weights": {"performance": 0.6, "measurables": 0.4},
    },
    "WR": {
        "performance": [
            "yards_per_route_run", "drop_rate", "contested_catch_rate",
            "separation_score", "yards_after_catch_per_reception",
            "deep_target_rate", "slot_rate",
        ],
        "measurables": [
            "height", "weight", "forty", "vertical", "three_cone", "broad_jump",
        ],
        "weights": {"performance": 0.55, "measurables": 0.45},
    },
    "TE": {
        "performance": [
            "yards_per_route_run", "contested_catch_rate", "run_block_grade",
            "pass_block_grade", "targets_per_game", "drop_rate",
        ],
        "measurables": [
            "height", "weight", "forty", "vertical", "bench", "broad_jump",
        ],
        "weights": {"performance": 0.6, "measurables": 0.4},
    },
    "OL": {
        "performance": [
            "pass_block_efficiency", "pressures_allowed_per_snap",
            "run_block_grade", "true_pass_set_grade", "penalties",
        ],
        "measurables": [
            "height", "weight", "forty", "bench", "arm_length", "hand_size",
        ],
        "weights": {"performance": 0.5, "measurables": 0.5},
    },
    "EDGE": {
        "performance": [
            "pass_rush_productivity", "pass_rush_win_rate", "pressures_per_snap",
            "run_defense_grade", "tackles_per_snap", "sack_rate",
        ],
        "measurables": [
            "height", "weight", "forty", "vertical", "bench", "broad_jump",
        ],
        "weights": {"performance": 0.55, "measurables": 0.45},
    },
    "DL": {
        "performance": [
            "run_defense_grade", "pass_rush_productivity", "stops_per_snap",
            "pressures_per_snap", "tackles_for_loss_rate",
        ],
        "measurables": [
            "height", "weight", "forty", "bench", "arm_length",
        ],
        "weights": {"performance": 0.55, "measurables": 0.45},
    },
    "LB": {
        "performance": [
            "coverage_grade", "run_defense_grade", "pass_rush_grade",
            "tackles_per_snap", "passer_rating_allowed", "forced_incompletion_rate",
        ],
        "measurables": [
            "height", "weight", "forty", "vertical", "three_cone",
        ],
        "weights": {"performance": 0.6, "measurables": 0.4},
    },
    "CB": {
        "performance": [
            "coverage_grade", "passer_rating_allowed", "yards_per_coverage_snap",
            "forced_incompletion_rate", "catch_rate_allowed", "slot_coverage_grade",
        ],
        "measurables": [
            "height", "weight", "forty", "vertical", "three_cone", "broad_jump",
        ],
        "weights": {"performance": 0.5, "measurables": 0.5},
    },
    "S": {
        "performance": [
            "coverage_grade", "run_defense_grade", "passer_rating_allowed",
            "tackles_per_snap", "forced_incompletion_rate", "box_snap_rate",
        ],
        "measurables": [
            "height", "weight", "forty", "vertical", "broad_jump",
        ],
        "weights": {"performance": 0.55, "measurables": 0.45},
    },
}

# Default features for positions not explicitly defined
DEFAULT_FEATURES = {
    "performance": ["overall_grade", "snap_count"],
    "measurables": ["height", "weight", "forty"],
    "weights": {"performance": 0.6, "measurables": 0.4},
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PlayerComparison:
    """Represents a player comparison result."""
    name: str
    school_or_team: str
    position: str
    seasons: List[int]
    similarity_score: float  # 0-100
    league: str  # "NFL" or "NCAA"
    matching_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    nfl_outcome: Optional[Dict[str, Any]] = None
    headshot_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "school_or_team": self.school_or_team,
            "position": self.position,
            "seasons": self.seasons,
            "similarity": self.similarity_score,
            "league": self.league,
            "matching_stats": self.matching_stats,
            "nfl_outcome": self.nfl_outcome,
            "headshot_url": self.headshot_url,
        }


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_player_features(
    player_data: Dict[str, Any],
    position: str,
    combine_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """Extract feature vector for a player based on position.

    Args:
        player_data: Dict with player stats
        position: Player position (normalized)
        combine_data: Optional combine measurables

    Returns:
        Tuple of (feature_dict, feature_names)
    """
    position = normalize_position(position)
    config = POSITION_FEATURES.get(position, DEFAULT_FEATURES)

    features = {}
    feature_names = []

    # Extract performance features
    for feat in config["performance"]:
        value = player_data.get(feat)
        if value is not None and not pd.isna(value):
            features[feat] = float(value)
            feature_names.append(feat)

    # Extract measurable features
    measurables = combine_data or {}
    for feat in config["measurables"]:
        value = measurables.get(feat) or player_data.get(feat)
        if value is not None and not pd.isna(value):
            features[feat] = float(value)
            feature_names.append(feat)

    return features, feature_names


def build_feature_matrix(
    players: List[Dict[str, Any]],
    position: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build feature matrix for a list of players.

    Args:
        players: List of player dicts
        position: Position to use for feature selection

    Returns:
        Tuple of (feature_matrix, player_names, feature_names)
    """
    position = normalize_position(position)
    config = POSITION_FEATURES.get(position, DEFAULT_FEATURES)
    all_features = config["performance"] + config["measurables"]

    # First pass: collect all available features
    feature_values = []
    player_names = []
    valid_players = []

    for player in players:
        name = player.get("name", "Unknown")
        features, _ = extract_player_features(player, position)

        if features:  # Only include players with at least some features
            player_names.append(name)
            feature_values.append(features)
            valid_players.append(player)

    if not feature_values:
        return np.array([]), [], []

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(feature_values)

    # Keep only columns that have at least 50% non-null values
    min_coverage = 0.5
    valid_cols = df.columns[df.notna().mean() >= min_coverage].tolist()

    if not valid_cols:
        # Fall back to any columns with data
        valid_cols = df.columns[df.notna().any()].tolist()

    df = df[valid_cols]

    # Fill missing values with column median
    df = df.fillna(df.median())

    # If still empty, return empty
    if df.empty:
        return np.array([]), [], []

    # Standardize features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(df.values)

    return feature_matrix, player_names, valid_cols


# =============================================================================
# Similarity Calculation
# =============================================================================

def calculate_similarity(
    target_features: np.ndarray,
    comparison_features: np.ndarray,
) -> np.ndarray:
    """Calculate cosine similarity between target and comparison players.

    Args:
        target_features: Feature vector for target player (1, n_features)
        comparison_features: Feature matrix for comparison players (m, n_features)

    Returns:
        Array of similarity scores (m,)
    """
    if target_features.size == 0 or comparison_features.size == 0:
        return np.array([])

    # Reshape target if needed
    if target_features.ndim == 1:
        target_features = target_features.reshape(1, -1)

    similarities = cosine_similarity(target_features, comparison_features)
    return similarities[0]


def find_comparable_players(
    player_name: str,
    position: str,
    n_comparisons: int = 5,
    include_nfl: bool = True,
    include_college: bool = True,
    seasons: Optional[List[int]] = None,
) -> List[PlayerComparison]:
    """Find players most similar to the target player.

    Args:
        player_name: Target player name
        position: Target player position
        n_comparisons: Number of comparisons to return
        include_nfl: Include NFL players in comparisons
        include_college: Include college players in comparisons
        seasons: Optional list of seasons to consider

    Returns:
        List of PlayerComparison objects
    """
    position = normalize_position(position)
    comparisons = []

    # Get target player data
    target_data = _get_target_player_data(player_name)
    if not target_data:
        logger.warning(f"No data found for target player: {player_name}")
        return []

    # Get target player measurables
    target_measurables = get_player_measurables(player_name)

    # Extract target features
    target_features, target_feature_names = extract_player_features(
        target_data, position, target_measurables
    )

    if not target_features:
        logger.warning(f"Could not extract features for: {player_name}")
        return []

    # Load comparison data
    comparison_players = []

    if include_nfl:
        nfl_df = load_all_nfl_stats()
        if not nfl_df.empty:
            nfl_players = _filter_by_position(nfl_df, position)
            comparison_players.extend([
                {**row.to_dict(), "league": "NFL"}
                for _, row in nfl_players.iterrows()
            ])

    if include_college:
        college_df = load_all_college_stats()
        if not college_df.empty:
            college_players = _filter_by_position(college_df, position)
            comparison_players.extend([
                {**row.to_dict(), "league": "NCAA"}
                for _, row in college_players.iterrows()
            ])

    # Remove target player from comparisons
    comparison_players = [
        p for p in comparison_players
        if p.get("name", "").lower() != player_name.lower()
    ]

    if not comparison_players:
        logger.warning(f"No comparison players found for position: {position}")
        return []

    # Build feature matrix for comparisons
    feature_matrix, comp_names, feature_names = build_feature_matrix(
        comparison_players, position
    )

    if feature_matrix.size == 0:
        logger.warning("Could not build feature matrix")
        return []

    # Build target vector with same features
    target_vector = np.array([
        target_features.get(f, 0) for f in feature_names
    ]).reshape(1, -1)

    # Standardize target vector (using comparison stats)
    scaler = StandardScaler()
    scaler.fit(feature_matrix)
    target_vector = scaler.transform(target_vector)

    # Calculate similarities
    similarities = calculate_similarity(target_vector, feature_matrix)

    # Sort by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]

    # Build comparison results
    seen_names = set()
    for idx in sorted_indices:
        if len(comparisons) >= n_comparisons:
            break

        comp_name = comp_names[idx]

        # Skip duplicates (same player different seasons)
        if comp_name.lower() in seen_names:
            continue
        seen_names.add(comp_name.lower())

        # Find original player data
        comp_data = next(
            (p for p in comparison_players if p.get("name") == comp_name),
            {}
        )

        # Build matching stats comparison
        matching_stats = {}
        for feat in feature_names[:5]:  # Top 5 features
            target_val = target_features.get(feat)
            comp_val = comp_data.get(feat)
            if target_val is not None and comp_val is not None:
                matching_stats[feat] = {
                    "target": round(target_val, 2),
                    "comparison": round(comp_val, 2),
                }

        # Get NFL outcome if applicable
        nfl_outcome = None
        if comp_data.get("league") == "NFL":
            nfl_outcome = _get_nfl_outcome(comp_name, comp_data)

        comparison = PlayerComparison(
            name=comp_name,
            school_or_team=comp_data.get("school") or comp_data.get("team", "Unknown"),
            position=position,
            seasons=[comp_data.get("season", 0)],
            similarity_score=round(float(similarities[idx]) * 100, 1),
            league=comp_data.get("league", "Unknown"),
            matching_stats=matching_stats,
            nfl_outcome=nfl_outcome,
        )
        comparisons.append(comparison)

    return comparisons


# =============================================================================
# Helper Functions
# =============================================================================

def _get_target_player_data(player_name: str) -> Optional[Dict[str, Any]]:
    """Get data for the target player from available sources."""
    # Try NIL valuations first (current college players)
    nil_df = get_nil_players(limit=50000)
    if not nil_df.empty and "name" in nil_df.columns:
        match = nil_df[nil_df["name"].str.contains(player_name, case=False, na=False)]
        if not match.empty:
            return match.iloc[0].to_dict()

    # Try college stats
    college_df = load_all_college_stats()
    if not college_df.empty and "name" in college_df.columns:
        match = college_df[college_df["name"].str.contains(player_name, case=False, na=False)]
        if not match.empty:
            # Get most recent season
            match = match.sort_values("season", ascending=False)
            return match.iloc[0].to_dict()

    return None


def _filter_by_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Filter DataFrame to players of a specific position."""
    if "position" not in df.columns:
        return df

    position = normalize_position(position)

    # Get all positions that map to this group
    from ..utils.historical_loader import POSITION_GROUPS
    valid_positions = POSITION_GROUPS.get(position, [position])

    return df[df["position"].str.upper().isin([p.upper() for p in valid_positions])]


def _get_nfl_outcome(player_name: str, player_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get NFL career outcome for a player."""
    # Try to get draft data
    try:
        from .draft_projector import get_player_draft_outcome
        draft_outcome = get_player_draft_outcome(player_name)
        if draft_outcome:
            return {
                "team": draft_outcome.get("team") or player_data.get("team"),
                "seasons_played": player_data.get("seasons_played", 1),
                "draft_round": draft_outcome.get("round"),
                "draft_pick": draft_outcome.get("overall"),
                "draft_year": draft_outcome.get("year"),
                "contract_value": draft_outcome.get("contract_value"),
                "guaranteed": draft_outcome.get("guaranteed"),
                "career_highlights": player_data.get("career_highlights", []),
            }
    except Exception:
        pass

    # Fall back to basic info from the data we have
    return {
        "team": player_data.get("team"),
        "seasons_played": player_data.get("seasons_played", 1),
        "draft_round": player_data.get("draft_round"),
        "draft_pick": player_data.get("draft_pick"),
        "career_highlights": player_data.get("career_highlights", []),
    }


# =============================================================================
# API-Ready Functions
# =============================================================================

def get_player_comparisons(
    player_name: str,
    include_nfl: bool = True,
    include_college: bool = True,
    limit: int = 5,
) -> Dict[str, Any]:
    """Get player comparisons formatted for API response.

    Args:
        player_name: Player to find comparisons for
        include_nfl: Include NFL comparisons
        include_college: Include college comparisons
        limit: Max comparisons per category

    Returns:
        Dict with nfl_comparisons and college_comparisons lists
    """
    # Get player position
    target_data = _get_target_player_data(player_name)
    if not target_data:
        return {
            "player": player_name,
            "error": "Player not found",
            "nfl_comparisons": [],
            "college_comparisons": [],
        }

    position = target_data.get("position", "ATH")

    # Get comparisons
    nfl_comps = []
    college_comps = []

    if include_nfl:
        nfl_results = find_comparable_players(
            player_name, position, n_comparisons=limit,
            include_nfl=True, include_college=False
        )
        nfl_comps = [c.to_dict() for c in nfl_results]

    if include_college:
        college_results = find_comparable_players(
            player_name, position, n_comparisons=limit,
            include_nfl=False, include_college=True
        )
        college_comps = [c.to_dict() for c in college_results]

    return {
        "player": player_name,
        "position": position,
        "nfl_comparisons": nfl_comps,
        "college_comparisons": college_comps,
    }
