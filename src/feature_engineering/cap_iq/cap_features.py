"""
Salary Cap Feature Engineering

Builds features for cap analysis and optimization.
"""

import logging
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CapFeatureBuilder:
    """Builds features for salary cap analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.salary_cap = config.get("salary_cap", 255_000_000)
        self.cap_growth_rate = config.get("cap_growth_rate", 0.07)

    def build_team_cap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build team-level cap features.

        Args:
            df: Team cap DataFrame

        Returns:
            DataFrame with team cap features
        """
        features = df.copy()

        if "cap_spent" in features.columns:
            features["cap_utilization"] = features["cap_spent"] / self.salary_cap
            features["cap_space"] = self.salary_cap - features["cap_spent"]
            features["cap_space_pct"] = features["cap_space"] / self.salary_cap * 100

        if "dead_money" in features.columns:
            features["dead_money_pct"] = (
                features["dead_money"] / self.salary_cap * 100
            )
            features["effective_cap"] = self.salary_cap - features["dead_money"]

        if "top_51_cap" in features.columns:
            features["depth_spending"] = (
                features["cap_spent"] - features["top_51_cap"]
            )

        return features

    def build_position_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build position group spending features.

        Args:
            df: Team roster/cap DataFrame

        Returns:
            DataFrame with position spending features
        """
        features = df.copy()

        position_groups = {
            "offense": ["QB", "RB", "WR", "TE", "OT", "OG", "C"],
            "defense": ["EDGE", "DT", "LB", "CB", "S"],
            "special_teams": ["K", "P", "LS"],
        }

        for group, positions in position_groups.items():
            col_name = f"{group}_spending"
            if col_name in features.columns:
                features[f"{group}_cap_pct"] = (
                    features[col_name] / self.salary_cap * 100
                )

        # QB spending ratio (important indicator)
        if "qb_cap_hit" in features.columns:
            features["qb_cap_ratio"] = features["qb_cap_hit"] / self.salary_cap

        return features

    def build_contract_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build contract structure features.

        Args:
            df: Contract DataFrame

        Returns:
            DataFrame with structure features
        """
        features = df.copy()

        if "guaranteed" in features.columns and "total_value" in features.columns:
            features["gtd_percentage"] = (
                features["guaranteed"] / features["total_value"].replace(0, 1) * 100
            )

        if "signing_bonus" in features.columns and "total_value" in features.columns:
            features["bonus_percentage"] = (
                features["signing_bonus"] / features["total_value"].replace(0, 1) * 100
            )

        if "years" in features.columns:
            features["aav"] = (
                features["total_value"] / features["years"].replace(0, 1)
            )

        if "base_salary" in features.columns and "cap_hit" in features.columns:
            features["cap_vs_cash"] = features["cap_hit"] / features[
                "base_salary"
            ].replace(0, 1)

        return features

    def build_flexibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build cap flexibility features.

        Args:
            df: Team cap DataFrame

        Returns:
            DataFrame with flexibility features
        """
        features = df.copy()

        # Current year flexibility
        if "cap_space" in features.columns:
            features["flexibility_score"] = (
                features["cap_space"] / self.salary_cap
            ).clip(0, 0.3) / 0.3

        # Future flexibility (based on expiring contracts)
        if "expiring_cap" in features.columns:
            features["future_space"] = features["cap_space"] + features["expiring_cap"]
            features["future_flexibility"] = (
                features["future_space"] / self.salary_cap
            ).clip(0, 0.4) / 0.4

        # Dead money burden
        if "dead_money" in features.columns:
            features["dead_money_burden"] = 1 - (
                features["dead_money"] / (self.salary_cap * 0.1)
            ).clip(0, 1)

        return features

    def build_roster_construction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build roster construction analysis features.

        Args:
            df: Team roster DataFrame

        Returns:
            DataFrame with roster features
        """
        features = df.copy()

        # Veteran vs rookie balance
        if "veteran_contracts" in features.columns and "rookie_contracts" in features.columns:
            total = features["veteran_contracts"] + features["rookie_contracts"]
            features["rookie_ratio"] = features["rookie_contracts"] / total.replace(0, 1)

        # Contract year count (players in final year)
        if "contract_year_players" in features.columns:
            features["turnover_risk"] = features["contract_year_players"] / 53

        # Extension candidates
        if "extension_candidates" in features.columns:
            features["extension_pressure"] = features["extension_candidates"] / 10

        return features

    def build_projection_features(
        self,
        df: pd.DataFrame,
        years_ahead: int = 3,
    ) -> pd.DataFrame:
        """
        Build cap projection features.

        Args:
            df: Team cap DataFrame
            years_ahead: Years to project

        Returns:
            DataFrame with projection features
        """
        features = df.copy()

        # Project future cap
        for year in range(1, years_ahead + 1):
            projected_cap = self.salary_cap * (
                (1 + self.cap_growth_rate) ** year
            )
            features[f"projected_cap_y{year}"] = projected_cap

            if f"committed_y{year}" in features.columns:
                features[f"projected_space_y{year}"] = (
                    projected_cap - features[f"committed_y{year}"]
                )

        return features

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all cap-related features.

        Args:
            df: Raw DataFrame

        Returns:
            DataFrame with all features
        """
        features = df.copy()

        features = self.build_team_cap_features(features)
        features = self.build_position_spending_features(features)
        features = self.build_contract_structure_features(features)
        features = self.build_flexibility_features(features)
        features = self.build_roster_construction_features(features)
        features = self.build_projection_features(features)

        return features

    def calculate_cap_efficiency(
        self,
        team_df: pd.DataFrame,
        performance_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate cap efficiency metrics.

        Args:
            team_df: Team cap DataFrame
            performance_df: Team performance DataFrame

        Returns:
            DataFrame with efficiency metrics
        """
        # Merge team data
        merged = team_df.merge(performance_df, on="team", how="left")

        # Wins per cap dollar
        if "wins" in merged.columns and "cap_spent" in merged.columns:
            merged["wins_per_million"] = (
                merged["wins"] / (merged["cap_spent"] / 1_000_000)
            )

        # Surplus value total
        if "total_surplus" in merged.columns:
            merged["surplus_per_win"] = (
                merged["total_surplus"] / merged["wins"].replace(0, 1)
            )

        return merged
