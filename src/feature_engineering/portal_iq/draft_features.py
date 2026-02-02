"""
NFL Draft Feature Engineering

Builds features for draft projection models.
"""

import logging
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DraftFeatureBuilder:
    """Builds features for NFL draft projections."""

    # Position-specific athletic thresholds (elite = 90th percentile)
    ATHLETIC_THRESHOLDS = {
        "QB": {"forty": 4.8, "vertical": 32, "broad": 110},
        "RB": {"forty": 4.5, "vertical": 36, "broad": 120},
        "WR": {"forty": 4.45, "vertical": 38, "broad": 125},
        "TE": {"forty": 4.65, "vertical": 34, "broad": 118},
        "OT": {"forty": 5.1, "vertical": 28, "broad": 105},
        "IOL": {"forty": 5.2, "vertical": 26, "broad": 100},
        "EDGE": {"forty": 4.7, "vertical": 35, "broad": 118},
        "DT": {"forty": 5.0, "vertical": 28, "broad": 105},
        "LB": {"forty": 4.65, "vertical": 35, "broad": 118},
        "CB": {"forty": 4.45, "vertical": 38, "broad": 125},
        "S": {"forty": 4.5, "vertical": 36, "broad": 120},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def build_production_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build college production features.

        Args:
            df: Player stats DataFrame

        Returns:
            DataFrame with production features
        """
        features = df.copy()

        # Per-game stats normalization
        games = features.get("games", 1).replace(0, 1)

        # Passing
        if "passing_yards" in features.columns:
            features["pass_ypg"] = features["passing_yards"] / games
            features["pass_td_rate"] = features["passing_tds"] / games
            features["int_rate"] = features["interceptions"] / games

        # Rushing
        if "rushing_yards" in features.columns:
            features["rush_ypg"] = features["rushing_yards"] / games
            features["ypc"] = (
                features["rushing_yards"]
                / features["rushing_attempts"].replace(0, 1)
            )

        # Receiving
        if "receiving_yards" in features.columns:
            features["rec_ypg"] = features["receiving_yards"] / games
            features["ypr"] = (
                features["receiving_yards"]
                / features["receptions"].replace(0, 1)
            )

        # Defensive
        if "tackles" in features.columns:
            features["tackles_pg"] = features["tackles"] / games
        if "sacks" in features.columns:
            features["sacks_pg"] = features["sacks"] / games
        if "interceptions_def" in features.columns:
            features["int_pg"] = features["interceptions_def"] / games

        # Career production weight (more recent = more important)
        if "year" in features.columns:
            max_year = features["year"].max()
            features["recency_weight"] = 1 + (features["year"] - max_year + 3) * 0.1

        return features

    def build_athletic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build athletic testing features.

        Args:
            df: Player DataFrame with combine/pro day data

        Returns:
            DataFrame with athletic features
        """
        features = df.copy()

        # Normalize athletic metrics to position
        if "position" in features.columns:
            for metric in ["forty", "vertical", "broad", "bench", "cone", "shuttle"]:
                if metric in features.columns:
                    features[f"{metric}_percentile"] = features.apply(
                        lambda r: self._calculate_percentile(
                            r[metric], r["position"], metric
                        ),
                        axis=1,
                    )

            # Composite athletic score
            features["athletic_score"] = self._calculate_athletic_score(features)

        # Size metrics
        if "height" in features.columns and "weight" in features.columns:
            features["bmi"] = (
                features["weight"] / (features["height"] ** 2) * 703
            )
            features["size_score"] = self._calculate_size_score(features)

        return features

    def build_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build age-related features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with age features
        """
        features = df.copy()

        if "birth_date" in features.columns:
            # Age at draft (assuming April draft)
            draft_date = pd.Timestamp(f"{features['draft_year'].iloc[0]}-04-25")
            features["draft_age"] = (
                (draft_date - pd.to_datetime(features["birth_date"])).dt.days / 365.25
            )
        elif "age" in features.columns:
            features["draft_age"] = features["age"]

        if "draft_age" in features.columns:
            # Age penalty (younger is better)
            features["age_score"] = 1 - ((features["draft_age"] - 21) / 5).clip(0, 1)

        return features

    def build_school_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build school/conference features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with school features
        """
        features = df.copy()

        if "school" in features.columns:
            features["school_tier"] = features["school"].apply(self._get_school_tier)
            features["school_draft_rate"] = features["school"].apply(
                self._get_historical_draft_rate
            )

        if "conference" in features.columns:
            features["conference_tier"] = features["conference"].apply(
                self._get_conference_tier
            )

        return features

    def build_projection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build composite draft projection features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with projection score
        """
        features = df.copy()

        # Weights for different factors
        weights = self.config.get("draft_weights", {
            "production": 0.30,
            "athletic": 0.25,
            "age": 0.15,
            "school": 0.15,
            "size": 0.15,
        })

        projection_score = pd.Series(0.0, index=features.index)

        # Production contribution
        if "pass_ypg" in features.columns:
            prod_score = (features["pass_ypg"] / 300).clip(0, 1)
        elif "rush_ypg" in features.columns:
            prod_score = (features["rush_ypg"] / 100).clip(0, 1)
        elif "rec_ypg" in features.columns:
            prod_score = (features["rec_ypg"] / 80).clip(0, 1)
        else:
            prod_score = 0.5

        projection_score += prod_score * weights.get("production", 0.3)

        # Athletic contribution
        if "athletic_score" in features.columns:
            projection_score += features["athletic_score"] * weights.get("athletic", 0.25)

        # Age contribution
        if "age_score" in features.columns:
            projection_score += features["age_score"] * weights.get("age", 0.15)

        # School contribution
        if "school_tier" in features.columns:
            school_score = features["school_tier"].map({
                "blue_blood": 1.0,
                "elite": 0.85,
                "power_brand": 0.7,
                "p4_mid": 0.5,
                "g5_strong": 0.4,
                "g5": 0.3,
            }).fillna(0.5)
            projection_score += school_score * weights.get("school", 0.15)

        # Size contribution
        if "size_score" in features.columns:
            projection_score += features["size_score"] * weights.get("size", 0.15)

        features["draft_projection_score"] = projection_score.clip(0, 1)

        # Convert to draft round estimate
        features["projected_round"] = self._score_to_round(
            features["draft_projection_score"]
        )

        return features

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all draft-related features.

        Args:
            df: Raw player DataFrame

        Returns:
            DataFrame with all features
        """
        features = df.copy()

        features = self.build_production_features(features)
        features = self.build_athletic_features(features)
        features = self.build_age_features(features)
        features = self.build_school_features(features)
        features = self.build_projection_features(features)

        return features

    def _calculate_percentile(
        self,
        value: float,
        position: str,
        metric: str,
    ) -> float:
        """Calculate percentile for athletic metric."""
        if pd.isna(value) or not position:
            return 0.5

        thresholds = self.ATHLETIC_THRESHOLDS.get(position.upper(), {})
        elite_value = thresholds.get(metric)

        if not elite_value:
            return 0.5

        # Lower is better for time-based metrics
        if metric in ["forty", "cone", "shuttle"]:
            if value <= elite_value:
                return 0.9 + (elite_value - value) * 0.5
            return max(0.1, 0.9 - (value - elite_value) * 0.5)
        else:
            # Higher is better
            if value >= elite_value:
                return 0.9 + (value - elite_value) * 0.01
            return max(0.1, 0.9 - (elite_value - value) * 0.02)

    def _calculate_athletic_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite athletic score."""
        percentile_cols = [c for c in df.columns if c.endswith("_percentile")]

        if not percentile_cols:
            return pd.Series(0.5, index=df.index)

        return df[percentile_cols].mean(axis=1).fillna(0.5)

    def _calculate_size_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate size score based on position requirements."""
        # Simplified - would use position-specific ideal sizes
        return pd.Series(0.5, index=df.index)

    def _get_school_tier(self, school: str) -> str:
        """Determine school tier for draft purposes."""
        school_lower = school.lower() if school else ""

        blue_bloods = ["alabama", "ohio state", "georgia", "clemson"]
        if any(bb in school_lower for bb in blue_bloods):
            return "blue_blood"

        return "p4_mid"

    def _get_historical_draft_rate(self, school: str) -> float:
        """Get historical draft pick rate for school."""
        # Would use actual historical data
        return 0.5

    def _get_conference_tier(self, conference: str) -> int:
        """Get conference tier for draft purposes."""
        conf_lower = conference.lower() if conference else ""

        if any(c in conf_lower for c in ["sec", "big ten"]):
            return 1
        if any(c in conf_lower for c in ["big 12", "acc"]):
            return 2

        return 3

    def _score_to_round(self, scores: pd.Series) -> pd.Series:
        """Convert projection score to draft round."""
        rounds = pd.cut(
            scores,
            bins=[0, 0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0],
            labels=[7, 6, 5, 4, 3, 2, 1],
        )
        return rounds.astype(float)
