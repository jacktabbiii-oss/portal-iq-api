"""
Surplus Value Model

Calculates contract surplus value (value over/under market).
"""

import logging
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SurplusValueModel:
    """Calculates surplus value for NFL contracts."""

    # Dollars per WAR (market rate)
    DOLLARS_PER_WAR = 3_000_000

    # Position WAR weights
    POSITION_WAR_FACTORS = {
        "QB": 1.5,
        "EDGE": 1.2,
        "CB": 1.1,
        "WR": 1.0,
        "OT": 1.0,
        "DT": 0.9,
        "S": 0.9,
        "LB": 0.85,
        "TE": 0.85,
        "IOL": 0.8,
        "RB": 0.7,
    }

    def __init__(
        self,
        salary_cap: int = 255_000_000,
        dollars_per_war: Optional[float] = None,
    ):
        """
        Initialize surplus value model.

        Args:
            salary_cap: Current NFL salary cap
            dollars_per_war: Market rate per WAR
        """
        self.salary_cap = salary_cap
        self.dollars_per_war = dollars_per_war or self.DOLLARS_PER_WAR

    def calculate_expected_value(
        self,
        war: float,
        position: str,
    ) -> float:
        """
        Calculate expected contract value based on WAR.

        Args:
            war: Wins Above Replacement
            position: Player position

        Returns:
            Expected contract value (AAV)
        """
        pos_factor = self.POSITION_WAR_FACTORS.get(position.upper(), 1.0)
        return war * self.dollars_per_war * pos_factor

    def calculate_surplus(
        self,
        cap_hit: float,
        war: float,
        position: str,
    ) -> float:
        """
        Calculate surplus value.

        Args:
            cap_hit: Player's cap hit
            war: Wins Above Replacement
            position: Player position

        Returns:
            Surplus value (positive = team-friendly)
        """
        expected = self.calculate_expected_value(war, position)
        return expected - cap_hit

    def calculate_surplus_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate surplus value for DataFrame.

        Args:
            df: Player DataFrame with cap_hit, war, position

        Returns:
            DataFrame with surplus values
        """
        result = df.copy()

        result["expected_value"] = result.apply(
            lambda r: self.calculate_expected_value(
                r.get("war", 0),
                r.get("position", ""),
            ),
            axis=1,
        )

        result["surplus_value"] = (
            result["expected_value"] - result.get("cap_hit", 0)
        )

        result["surplus_pct"] = (
            result["surplus_value"]
            / result["expected_value"].replace(0, 1)
            * 100
        )

        result["contract_grade"] = result["surplus_pct"].apply(
            self._grade_contract
        )

        return result

    def _grade_contract(self, surplus_pct: float) -> str:
        """Grade contract based on surplus percentage."""
        if surplus_pct >= 50:
            return "A+"
        elif surplus_pct >= 30:
            return "A"
        elif surplus_pct >= 15:
            return "B+"
        elif surplus_pct >= 0:
            return "B"
        elif surplus_pct >= -15:
            return "C"
        elif surplus_pct >= -30:
            return "D"
        else:
            return "F"

    def get_leaderboard(
        self,
        df: pd.DataFrame,
        metric: str = "surplus_value",
        top_n: int = 25,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Get surplus value leaderboard.

        Args:
            df: Player DataFrame
            metric: Sorting metric
            top_n: Number of players
            ascending: Sort order

        Returns:
            Leaderboard DataFrame
        """
        surplus_df = self.calculate_surplus_df(df)

        return surplus_df.nlargest(top_n, metric) if not ascending else \
               surplus_df.nsmallest(top_n, metric)

    def get_team_surplus(self, team_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate total team surplus value.

        Args:
            team_df: Team roster DataFrame

        Returns:
            Team surplus analysis
        """
        surplus_df = self.calculate_surplus_df(team_df)

        total_surplus = surplus_df["surplus_value"].sum()
        avg_surplus = surplus_df["surplus_value"].mean()

        best_contract = surplus_df.loc[surplus_df["surplus_value"].idxmax()]
        worst_contract = surplus_df.loc[surplus_df["surplus_value"].idxmin()]

        grade_counts = surplus_df["contract_grade"].value_counts().to_dict()

        return {
            "total_surplus": total_surplus,
            "avg_surplus": avg_surplus,
            "roster_size": len(team_df),
            "best_contract": {
                "player": best_contract.get("player_name", ""),
                "surplus": best_contract["surplus_value"],
                "grade": best_contract["contract_grade"],
            },
            "worst_contract": {
                "player": worst_contract.get("player_name", ""),
                "surplus": worst_contract["surplus_value"],
                "grade": worst_contract["contract_grade"],
            },
            "grade_distribution": grade_counts,
        }

    def get_position_surplus(
        self,
        df: pd.DataFrame,
        position: str,
    ) -> Dict[str, Any]:
        """
        Get surplus analysis for a position group.

        Args:
            df: Player DataFrame
            position: Position to analyze

        Returns:
            Position surplus analysis
        """
        pos_df = df[df["position"].str.upper() == position.upper()].copy()

        if pos_df.empty:
            return {"position": position, "players": 0}

        surplus_df = self.calculate_surplus_df(pos_df)

        return {
            "position": position,
            "players": len(pos_df),
            "total_surplus": surplus_df["surplus_value"].sum(),
            "avg_surplus": surplus_df["surplus_value"].mean(),
            "best_value": surplus_df["surplus_value"].max(),
            "worst_value": surplus_df["surplus_value"].min(),
        }

    def project_future_surplus(
        self,
        player_data: Dict[str, Any],
        years: int = 3,
        war_decline_rate: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Project future surplus value.

        Args:
            player_data: Player information
            years: Years to project
            war_decline_rate: Annual WAR decline rate

        Returns:
            List of yearly projections
        """
        projections = []

        current_war = player_data.get("war", 2.0)
        position = player_data.get("position", "")
        cap_hits = player_data.get("future_cap_hits", [])

        for year in range(years):
            # Project WAR with decline
            projected_war = current_war * ((1 - war_decline_rate) ** year)

            # Get cap hit for year (or use last known)
            cap_hit = cap_hits[year] if year < len(cap_hits) else cap_hits[-1] if cap_hits else 10_000_000

            surplus = self.calculate_surplus(cap_hit, projected_war, position)

            projections.append({
                "year": year + 1,
                "projected_war": round(projected_war, 2),
                "cap_hit": cap_hit,
                "surplus_value": round(surplus),
                "grade": self._grade_contract(
                    surplus / self.calculate_expected_value(projected_war, position) * 100
                    if projected_war > 0 else 0
                ),
            })

        return projections

    def estimate_war(
        self,
        player_data: Dict[str, Any],
    ) -> float:
        """
        Estimate WAR from player stats.

        Args:
            player_data: Player statistics

        Returns:
            Estimated WAR
        """
        position = player_data.get("position", "").upper()

        # Simplified WAR estimation
        if position == "QB":
            passer_rating = player_data.get("passer_rating", 90)
            games = player_data.get("games", 17)
            return ((passer_rating - 80) / 20) * (games / 17) * 3

        elif position == "RB":
            yards = player_data.get("rushing_yards", 0) + player_data.get("receiving_yards", 0)
            return yards / 500

        elif position == "WR":
            yards = player_data.get("receiving_yards", 0)
            return yards / 400

        elif position in ["EDGE", "DE"]:
            sacks = player_data.get("sacks", 0)
            return sacks / 4

        elif position in ["CB", "S"]:
            ints = player_data.get("interceptions", 0)
            pbu = player_data.get("pass_breakups", 0)
            return ints * 0.5 + pbu * 0.1

        else:
            return player_data.get("approximate_value", 5) / 5

    def analyze_contract_efficiency(
        self,
        contracts_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analyze overall contract efficiency.

        Args:
            contracts_df: DataFrame of contracts

        Returns:
            Efficiency analysis
        """
        surplus_df = self.calculate_surplus_df(contracts_df)

        positive_surplus = surplus_df[surplus_df["surplus_value"] > 0]
        negative_surplus = surplus_df[surplus_df["surplus_value"] < 0]

        return {
            "total_contracts": len(contracts_df),
            "positive_surplus_contracts": len(positive_surplus),
            "negative_surplus_contracts": len(negative_surplus),
            "total_positive_surplus": positive_surplus["surplus_value"].sum(),
            "total_negative_surplus": negative_surplus["surplus_value"].sum(),
            "net_surplus": surplus_df["surplus_value"].sum(),
            "avg_surplus": surplus_df["surplus_value"].mean(),
            "median_surplus": surplus_df["surplus_value"].median(),
        }
