"""
NIL Feature Engineering

Builds features for NIL valuation models.
"""

import logging
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class NILFeatureBuilder:
    """Builds features for NIL valuation."""

    # Position value multipliers for NIL
    POSITION_MULTIPLIERS = {
        "QB": 2.5,
        "WR": 1.5,
        "RB": 1.2,
        "TE": 1.0,
        "OL": 0.8,
        "DL": 0.9,
        "LB": 0.9,
        "DB": 1.1,
        "K": 0.5,
        "P": 0.4,
    }

    # School tier multipliers
    SCHOOL_TIER_MULTIPLIERS = {
        "blue_blood": 2.0,
        "elite": 1.5,
        "power_brand": 1.2,
        "p4_mid": 1.0,
        "g5_strong": 0.7,
        "g5": 0.5,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def build_production_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build production-based features.

        Args:
            df: Player stats DataFrame

        Returns:
            DataFrame with production features
        """
        features = df.copy()

        # Passing features (QBs)
        if "passing_yards" in features.columns:
            features["pass_yards_per_game"] = (
                features["passing_yards"] / features.get("games", 1)
            ).fillna(0)
            features["td_int_ratio"] = (
                features["passing_tds"] / features["interceptions"].replace(0, 1)
            ).fillna(0)
            features["qb_rating_normalized"] = (
                features.get("passer_rating", 0) / 158.3
            ).clip(0, 1)

        # Rushing features
        if "rushing_yards" in features.columns:
            features["rush_yards_per_game"] = (
                features["rushing_yards"] / features.get("games", 1)
            ).fillna(0)
            features["yards_per_carry"] = (
                features["rushing_yards"] / features["rushing_attempts"].replace(0, 1)
            ).fillna(0)

        # Receiving features
        if "receiving_yards" in features.columns:
            features["rec_yards_per_game"] = (
                features["receiving_yards"] / features.get("games", 1)
            ).fillna(0)
            features["yards_per_reception"] = (
                features["receiving_yards"] / features["receptions"].replace(0, 1)
            ).fillna(0)

        # Total production score
        features["production_score"] = self._calculate_production_score(features)

        return features

    def build_brand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build brand/marketability features.

        Args:
            df: Player DataFrame with social/brand data

        Returns:
            DataFrame with brand features
        """
        features = df.copy()

        # Social media presence (if available)
        social_cols = ["instagram_followers", "twitter_followers", "tiktok_followers"]
        available_social = [c for c in social_cols if c in features.columns]

        if available_social:
            features["total_followers"] = features[available_social].sum(axis=1)
            features["social_score"] = np.log1p(features["total_followers"]) / 20
        else:
            features["social_score"] = 0.0

        # Market size factor
        if "school" in features.columns:
            features["market_factor"] = features["school"].apply(
                self._get_market_factor
            )
        else:
            features["market_factor"] = 1.0

        return features

    def build_school_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build school-related features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with school features
        """
        features = df.copy()

        if "school" in features.columns:
            features["school_tier"] = features["school"].apply(self._get_school_tier)
            features["school_multiplier"] = features["school_tier"].map(
                self.SCHOOL_TIER_MULTIPLIERS
            ).fillna(0.5)

        if "conference" in features.columns:
            features["conference_tier"] = features["conference"].apply(
                self._get_conference_tier
            )

        return features

    def build_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build position-based features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with position features
        """
        features = df.copy()

        if "position" in features.columns:
            features["position_group"] = features["position"].apply(
                self._get_position_group
            )
            features["position_multiplier"] = features["position"].apply(
                lambda p: self.POSITION_MULTIPLIERS.get(p, 0.8)
            )

        return features

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all NIL-related features.

        Args:
            df: Raw player DataFrame

        Returns:
            DataFrame with all features
        """
        features = df.copy()

        features = self.build_production_features(features)
        features = self.build_brand_features(features)
        features = self.build_school_features(features)
        features = self.build_position_features(features)

        # Calculate composite NIL score
        features["nil_score"] = self._calculate_nil_score(features)

        return features

    def _calculate_production_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate normalized production score."""
        score = pd.Series(0.0, index=df.index)

        # Passing production
        if "pass_yards_per_game" in df.columns:
            score += df["pass_yards_per_game"] / 300 * 0.4
            score += df.get("td_int_ratio", 0) / 5 * 0.3

        # Rushing production
        if "rush_yards_per_game" in df.columns:
            score += df["rush_yards_per_game"] / 100 * 0.3

        # Receiving production
        if "rec_yards_per_game" in df.columns:
            score += df["rec_yards_per_game"] / 80 * 0.3

        return score.clip(0, 1)

    def _calculate_nil_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite NIL score."""
        weights = self.config.get("nil_weights", {
            "production": 0.35,
            "brand": 0.25,
            "school": 0.25,
            "position": 0.15,
        })

        score = pd.Series(0.0, index=df.index)

        if "production_score" in df.columns:
            score += df["production_score"] * weights.get("production", 0.35)

        if "social_score" in df.columns:
            score += df["social_score"] * weights.get("brand", 0.25)

        if "school_multiplier" in df.columns:
            score += (df["school_multiplier"] / 2) * weights.get("school", 0.25)

        if "position_multiplier" in df.columns:
            score += (df["position_multiplier"] / 2.5) * weights.get("position", 0.15)

        return score.clip(0, 1)

    def _get_school_tier(self, school: str) -> str:
        """Determine school tier."""
        school_lower = school.lower() if school else ""

        blue_bloods = ["alabama", "ohio state", "georgia", "texas", "usc", "oklahoma"]
        if any(bb in school_lower for bb in blue_bloods):
            return "blue_blood"

        elite = ["michigan", "penn state", "lsu", "florida", "oregon", "tennessee"]
        if any(e in school_lower for e in elite):
            return "elite"

        return "p4_mid"

    def _get_conference_tier(self, conference: str) -> int:
        """Get conference tier (1-5)."""
        conf_lower = conference.lower() if conference else ""

        if any(c in conf_lower for c in ["sec", "big ten"]):
            return 1
        if any(c in conf_lower for c in ["big 12", "acc"]):
            return 2
        if any(c in conf_lower for c in ["american", "mountain west"]):
            return 3

        return 4

    def _get_position_group(self, position: str) -> str:
        """Get position group."""
        pos = position.upper() if position else ""

        if pos in ["QB"]:
            return "QB"
        if pos in ["RB", "FB"]:
            return "RB"
        if pos in ["WR"]:
            return "WR"
        if pos in ["TE"]:
            return "TE"
        if pos in ["OT", "OG", "C", "OL"]:
            return "OL"
        if pos in ["DE", "DT", "DL"]:
            return "DL"
        if pos in ["LB", "ILB", "OLB"]:
            return "LB"
        if pos in ["CB", "S", "DB", "FS", "SS"]:
            return "DB"

        return "OTHER"

    def _get_market_factor(self, school: str) -> float:
        """Get market size factor for school."""
        large_markets = ["usc", "texas", "miami", "ucla"]
        school_lower = school.lower() if school else ""

        if any(m in school_lower for m in large_markets):
            return 1.3

        return 1.0
