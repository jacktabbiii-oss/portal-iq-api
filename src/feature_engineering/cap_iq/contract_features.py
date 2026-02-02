"""
NFL Contract Feature Engineering

Builds features for contract valuation models.
"""

import logging
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ContractFeatureBuilder:
    """Builds features for NFL contract analysis."""

    # Position value tiers
    POSITION_VALUES = {
        "QB": 1.00,
        "EDGE": 0.85,
        "WR": 0.80,
        "CB": 0.75,
        "OT": 0.75,
        "DT": 0.65,
        "S": 0.60,
        "LB": 0.55,
        "TE": 0.55,
        "IOL": 0.50,
        "RB": 0.45,
        "K": 0.20,
        "P": 0.15,
        "LS": 0.10,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.salary_cap = config.get("salary_cap", 255_000_000)

    def build_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build performance-based features.

        Args:
            df: Player stats DataFrame

        Returns:
            DataFrame with performance features
        """
        features = df.copy()

        # Games played rate
        if "games_played" in features.columns:
            features["availability_rate"] = (
                features["games_played"] / 17
            ).clip(0, 1)

        # Position-specific production metrics
        if "position" in features.columns:
            features["production_score"] = features.apply(
                self._calculate_position_production,
                axis=1,
            )

        # Pro Bowl / All-Pro history
        if "pro_bowls" in features.columns:
            features["accolades_score"] = (
                features["pro_bowls"] * 0.1
                + features.get("all_pro_first", 0) * 0.3
                + features.get("all_pro_second", 0) * 0.15
            ).clip(0, 1)

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

        if "age" in features.columns:
            # Peak years by position
            features["years_from_peak"] = features.apply(
                lambda r: self._years_from_peak(r.get("age", 26), r.get("position", "")),
                axis=1,
            )

            # Age curve factor
            features["age_factor"] = features.apply(
                lambda r: self._age_curve_factor(r.get("age", 26), r.get("position", "")),
                axis=1,
            )

        if "years_experience" in features.columns:
            features["experience_tier"] = pd.cut(
                features["years_experience"],
                bins=[0, 2, 4, 7, 10, 20],
                labels=["rookie", "young", "prime", "veteran", "elder"],
            )

        return features

    def build_contract_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build contract history features.

        Args:
            df: Player DataFrame with contract history

        Returns:
            DataFrame with contract features
        """
        features = df.copy()

        if "previous_aav" in features.columns:
            features["aav_growth"] = (
                features.get("current_aav", 0) / features["previous_aav"].replace(0, 1)
            ) - 1

        if "contract_year" in features.columns:
            features["is_contract_year"] = features["contract_year"] == features.get(
                "contract_length", 1
            )

        if "times_tagged" in features.columns:
            features["franchise_tag_risk"] = features["times_tagged"] >= 1

        return features

    def build_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build market context features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with market features
        """
        features = df.copy()

        if "position" in features.columns:
            # Position value in current market
            features["position_value"] = features["position"].map(
                self.POSITION_VALUES
            ).fillna(0.5)

            # Position scarcity (would be calculated from league data)
            features["position_scarcity"] = 0.5  # Placeholder

        # Cap percentage
        if "aav" in features.columns:
            features["cap_percentage"] = features["aav"] / self.salary_cap * 100

        return features

    def build_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build injury history features.

        Args:
            df: Player DataFrame with injury history

        Returns:
            DataFrame with injury features
        """
        features = df.copy()

        if "games_missed_injury" in features.columns:
            total_games = features.get("career_games_possible", 17 * 4)
            features["injury_rate"] = features["games_missed_injury"] / total_games

        if "major_injuries" in features.columns:
            features["durability_concern"] = features["major_injuries"] >= 2

        return features

    def build_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build contract value features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with value assessment
        """
        features = df.copy()

        # Calculate expected market value
        features["expected_aav"] = features.apply(
            self._calculate_expected_aav,
            axis=1,
        )

        if "aav" in features.columns:
            # Surplus value (positive = team-friendly)
            features["surplus_value"] = (
                features["expected_aav"] - features["aav"]
            )
            features["value_rating"] = (
                features["surplus_value"] / features["expected_aav"].replace(0, 1)
            ).clip(-0.5, 0.5)

        return features

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all contract-related features.

        Args:
            df: Raw player DataFrame

        Returns:
            DataFrame with all features
        """
        features = df.copy()

        features = self.build_performance_features(features)
        features = self.build_age_features(features)
        features = self.build_contract_history_features(features)
        features = self.build_market_features(features)
        features = self.build_injury_features(features)
        features = self.build_value_features(features)

        return features

    def _calculate_position_production(self, row: pd.Series) -> float:
        """Calculate normalized production score for position."""
        position = row.get("position", "").upper()

        if position == "QB":
            # QB production based on passer rating, yards, TDs
            passer_rating = row.get("passer_rating", 90)
            return (passer_rating / 158.3).clip(0, 1)

        elif position in ["RB"]:
            yards = row.get("rushing_yards", 0) + row.get("receiving_yards", 0)
            return (yards / 1500).clip(0, 1)

        elif position == "WR":
            yards = row.get("receiving_yards", 0)
            return (yards / 1200).clip(0, 1)

        elif position == "TE":
            yards = row.get("receiving_yards", 0)
            return (yards / 800).clip(0, 1)

        elif position in ["EDGE", "DE"]:
            sacks = row.get("sacks", 0)
            return (sacks / 12).clip(0, 1)

        elif position in ["DT", "DL"]:
            sacks = row.get("sacks", 0)
            tackles = row.get("tackles", 0)
            return ((sacks / 8) * 0.6 + (tackles / 50) * 0.4).clip(0, 1)

        elif position in ["CB", "S", "DB"]:
            ints = row.get("interceptions", 0)
            pbu = row.get("pass_breakups", 0)
            return ((ints / 5) * 0.5 + (pbu / 15) * 0.5).clip(0, 1)

        elif position in ["LB", "ILB", "OLB"]:
            tackles = row.get("tackles", 0)
            return (tackles / 100).clip(0, 1)

        return 0.5

    def _years_from_peak(self, age: int, position: str) -> int:
        """Calculate years from peak age for position."""
        peak_ages = {
            "QB": 30,
            "RB": 25,
            "WR": 27,
            "TE": 27,
            "OT": 28,
            "IOL": 28,
            "EDGE": 26,
            "DT": 27,
            "LB": 26,
            "CB": 26,
            "S": 27,
        }
        peak = peak_ages.get(position.upper(), 27)
        return abs(age - peak)

    def _age_curve_factor(self, age: int, position: str) -> float:
        """Calculate age curve adjustment factor."""
        years_from_peak = self._years_from_peak(age, position)

        # Decline rate varies by position
        decline_rates = {
            "QB": 0.02,
            "RB": 0.08,
            "WR": 0.04,
            "CB": 0.05,
        }
        rate = decline_rates.get(position.upper(), 0.04)

        return max(0.5, 1.0 - years_from_peak * rate)

    def _calculate_expected_aav(self, row: pd.Series) -> float:
        """Calculate expected market AAV."""
        position = row.get("position", "").upper()
        position_value = self.POSITION_VALUES.get(position, 0.5)

        production = row.get("production_score", 0.5)
        age_factor = row.get("age_factor", 1.0)
        accolades = row.get("accolades_score", 0)

        # Base calculation
        base_value = self.salary_cap * position_value * 0.08

        # Adjust for production and age
        adjusted = base_value * production * age_factor * (1 + accolades * 0.5)

        return adjusted
