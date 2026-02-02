"""
Shared Player Feature Engineering

Common features used across Portal IQ and Cap IQ.
"""

import logging
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PlayerFeatureBuilder:
    """Builds common player features for both products."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def normalize_position(self, position: str) -> str:
        """
        Normalize position to standard format.

        Args:
            position: Raw position string

        Returns:
            Normalized position
        """
        pos = position.upper().strip() if position else ""

        # Offensive positions
        if pos in ["QB", "QUARTERBACK"]:
            return "QB"
        if pos in ["RB", "HB", "FB", "RUNNING BACK", "HALFBACK", "FULLBACK"]:
            return "RB"
        if pos in ["WR", "WIDE RECEIVER", "FLANKER", "SPLIT END"]:
            return "WR"
        if pos in ["TE", "TIGHT END"]:
            return "TE"
        if pos in ["OT", "T", "LT", "RT", "TACKLE", "OFFENSIVE TACKLE"]:
            return "OT"
        if pos in ["OG", "G", "LG", "RG", "GUARD", "OFFENSIVE GUARD"]:
            return "IOL"
        if pos in ["C", "CENTER"]:
            return "IOL"
        if pos in ["OL", "OFFENSIVE LINE"]:
            return "OL"

        # Defensive positions
        if pos in ["EDGE", "OLB", "OUTSIDE LINEBACKER", "RUSH", "LEO"]:
            return "EDGE"
        if pos in ["DE", "DEFENSIVE END"]:
            return "EDGE"
        if pos in ["DT", "NT", "NOSE TACKLE", "DEFENSIVE TACKLE"]:
            return "DT"
        if pos in ["DL", "DEFENSIVE LINE"]:
            return "DL"
        if pos in ["LB", "ILB", "MLB", "LINEBACKER", "INSIDE LINEBACKER"]:
            return "LB"
        if pos in ["CB", "CORNERBACK"]:
            return "CB"
        if pos in ["S", "FS", "SS", "SAFETY", "FREE SAFETY", "STRONG SAFETY"]:
            return "S"
        if pos in ["DB", "DEFENSIVE BACK"]:
            return "DB"

        # Special teams
        if pos in ["K", "KICKER", "PK"]:
            return "K"
        if pos in ["P", "PUNTER"]:
            return "P"
        if pos in ["LS", "LONG SNAPPER"]:
            return "LS"

        return pos

    def build_biometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build physical/biometric features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with biometric features
        """
        features = df.copy()

        # BMI calculation
        if "height" in features.columns and "weight" in features.columns:
            # Assuming height in inches, weight in pounds
            features["bmi"] = (
                features["weight"] / (features["height"] ** 2) * 703
            )

        # Height in standard format
        if "height" in features.columns:
            features["height_feet"] = features["height"] // 12
            features["height_inches"] = features["height"] % 12

        # Arm length ratio
        if "arm_length" in features.columns and "height" in features.columns:
            features["arm_ratio"] = features["arm_length"] / features["height"]

        # Hand size
        if "hand_size" in features.columns:
            features["hand_score"] = (features["hand_size"] - 9) / 2  # 9-11 typical

        return features

    def build_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build experience-related features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with experience features
        """
        features = df.copy()

        if "years_experience" in features.columns:
            features["is_rookie"] = features["years_experience"] == 0
            features["is_veteran"] = features["years_experience"] >= 4

            # Experience tier
            features["experience_tier"] = pd.cut(
                features["years_experience"],
                bins=[-1, 0, 2, 4, 7, 20],
                labels=["rookie", "young", "prime", "veteran", "elder"],
            )

        if "starts" in features.columns and "games" in features.columns:
            features["start_rate"] = (
                features["starts"] / features["games"].replace(0, 1)
            )

        return features

    def build_durability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build injury/durability features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with durability features
        """
        features = df.copy()

        if "games_played" in features.columns:
            # Assuming 17-game seasons
            seasons = features.get("seasons", 1).replace(0, 1)
            max_games = seasons * 17
            features["availability"] = features["games_played"] / max_games

        if "games_missed" in features.columns:
            features["injury_rate"] = (
                features["games_missed"]
                / (features["games_played"] + features["games_missed"]).replace(0, 1)
            )

        if "injury_history" in features.columns:
            features["major_injury_flag"] = features["injury_history"].str.contains(
                "ACL|Achilles|torn|surgery", case=False, na=False
            )

        return features

    def build_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build performance consistency features.

        Args:
            df: Player DataFrame with multi-game/season data

        Returns:
            DataFrame with consistency features
        """
        features = df.copy()

        # Coefficient of variation for key stats
        stat_cols = [c for c in features.columns if "per_game" in c or "avg" in c]

        for col in stat_cols:
            std_col = col.replace("avg", "std").replace("per_game", "std")
            if std_col in features.columns:
                features[f"{col}_cv"] = (
                    features[std_col] / features[col].replace(0, 1)
                )

        return features

    def calculate_war_estimate(
        self,
        df: pd.DataFrame,
        position_col: str = "position",
    ) -> pd.Series:
        """
        Calculate estimated WAR (Wins Above Replacement).

        Args:
            df: Player stats DataFrame
            position_col: Column containing position

        Returns:
            Series with WAR estimates
        """
        war = pd.Series(0.0, index=df.index)

        # Simplified WAR calculation by position
        for idx, row in df.iterrows():
            pos = self.normalize_position(row.get(position_col, ""))

            if pos == "QB":
                # QB WAR based on passer rating and yards
                pr = row.get("passer_rating", 90)
                yards = row.get("passing_yards", 0)
                war.loc[idx] = (pr - 80) / 20 + yards / 5000

            elif pos == "RB":
                yards = row.get("rushing_yards", 0) + row.get("receiving_yards", 0)
                war.loc[idx] = yards / 1500

            elif pos == "WR":
                yards = row.get("receiving_yards", 0)
                war.loc[idx] = yards / 1000

            elif pos in ["EDGE", "DL"]:
                sacks = row.get("sacks", 0)
                war.loc[idx] = sacks / 10

            elif pos in ["CB", "S"]:
                ints = row.get("interceptions", 0)
                pbu = row.get("pass_breakups", 0)
                war.loc[idx] = ints * 0.3 + pbu * 0.1

            else:
                # Default based on approximate value if available
                war.loc[idx] = row.get("approximate_value", 5) / 10

        return war.clip(0, 10)

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all common player features.

        Args:
            df: Raw player DataFrame

        Returns:
            DataFrame with all features
        """
        features = df.copy()

        # Normalize positions
        if "position" in features.columns:
            features["position_normalized"] = features["position"].apply(
                self.normalize_position
            )

        features = self.build_biometric_features(features)
        features = self.build_experience_features(features)
        features = self.build_durability_features(features)
        features = self.build_consistency_features(features)

        # Add WAR estimate
        features["war_estimate"] = self.calculate_war_estimate(features)

        return features
