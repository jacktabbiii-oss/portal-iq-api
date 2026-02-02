"""
Salary Cap Mathematics

Utility functions for NFL salary cap calculations.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ContractYear:
    """Single year of a contract."""
    year: int
    base_salary: float
    signing_bonus: float
    roster_bonus: float
    option_bonus: float
    incentives: float
    guaranteed: bool

    @property
    def cap_hit(self) -> float:
        """Calculate cap hit for this year."""
        return (
            self.base_salary
            + self.signing_bonus
            + self.roster_bonus
            + self.option_bonus
            + self.incentives
        )

    @property
    def cash(self) -> float:
        """Calculate cash paid this year."""
        return (
            self.base_salary
            + self.roster_bonus
            + self.option_bonus
            + self.incentives
        )


class CapMath:
    """NFL salary cap calculation utilities."""

    # 2025 salary cap
    SALARY_CAP_2025 = 255_000_000

    # Historical cap growth rate (average ~7% per year)
    CAP_GROWTH_RATE = 0.07

    # Minimum salary by years in league (2025)
    VETERAN_MINIMUMS = {
        0: 795_000,   # Rookie
        1: 915_000,
        2: 990_000,
        3: 1_065_000,
        4: 1_185_000,
        5: 1_260_000,
        6: 1_335_000,
        7: 1_425_000,  # 7+ years
    }

    @classmethod
    def project_cap(cls, year: int, base_year: int = 2025) -> float:
        """
        Project salary cap for a future year.

        Args:
            year: Target year
            base_year: Base year for projection

        Returns:
            Projected salary cap
        """
        if year <= base_year:
            return cls.SALARY_CAP_2025

        years_out = year - base_year
        return cls.SALARY_CAP_2025 * ((1 + cls.CAP_GROWTH_RATE) ** years_out)

    @classmethod
    def calculate_dead_money(
        cls,
        remaining_guarantees: float,
        remaining_prorated_bonus: float,
    ) -> float:
        """
        Calculate dead money from cutting a player.

        Args:
            remaining_guarantees: Unpaid guaranteed money
            remaining_prorated_bonus: Remaining prorated signing bonus

        Returns:
            Dead cap hit
        """
        return remaining_guarantees + remaining_prorated_bonus

    @classmethod
    def prorate_signing_bonus(
        cls,
        bonus: float,
        years: int,
    ) -> List[float]:
        """
        Prorate a signing bonus over contract years.

        Args:
            bonus: Total signing bonus
            years: Number of years (max 5)

        Returns:
            List of prorated amounts per year
        """
        prorate_years = min(years, 5)
        annual = bonus / prorate_years
        return [annual] * prorate_years

    @classmethod
    def calculate_aav(
        cls,
        total_value: float,
        years: int,
    ) -> float:
        """
        Calculate Average Annual Value.

        Args:
            total_value: Total contract value
            years: Contract length

        Returns:
            AAV
        """
        if years <= 0:
            return 0.0
        return total_value / years

    @classmethod
    def calculate_gtd_percentage(
        cls,
        guaranteed: float,
        total_value: float,
    ) -> float:
        """
        Calculate guaranteed percentage of contract.

        Args:
            guaranteed: Total guaranteed money
            total_value: Total contract value

        Returns:
            Guaranteed percentage (0-100)
        """
        if total_value <= 0:
            return 0.0
        return (guaranteed / total_value) * 100

    @classmethod
    def calculate_cap_percentage(
        cls,
        cap_hit: float,
        salary_cap: Optional[float] = None,
    ) -> float:
        """
        Calculate percentage of salary cap.

        Args:
            cap_hit: Player's cap hit
            salary_cap: Total cap (defaults to 2025 cap)

        Returns:
            Percentage of cap (0-100)
        """
        cap = salary_cap or cls.SALARY_CAP_2025
        return (cap_hit / cap) * 100

    @classmethod
    def calculate_surplus_value(
        cls,
        cap_hit: float,
        war: float,
        dollars_per_war: float = 3_000_000,
    ) -> float:
        """
        Calculate surplus value (value over replacement).

        Args:
            cap_hit: Player's cap hit
            war: Wins Above Replacement
            dollars_per_war: Dollars per WAR (market rate)

        Returns:
            Surplus value (positive = good deal)
        """
        expected_value = war * dollars_per_war
        return expected_value - cap_hit

    @classmethod
    def get_veteran_minimum(cls, years_in_league: int) -> float:
        """
        Get veteran minimum salary.

        Args:
            years_in_league: Years of NFL experience

        Returns:
            Minimum salary
        """
        if years_in_league >= 7:
            return cls.VETERAN_MINIMUMS[7]
        return cls.VETERAN_MINIMUMS.get(years_in_league, cls.VETERAN_MINIMUMS[0])

    @classmethod
    def calculate_cap_space(
        cls,
        cap_spent: float,
        salary_cap: Optional[float] = None,
        dead_money: float = 0,
    ) -> float:
        """
        Calculate available cap space.

        Args:
            cap_spent: Total cap commitments
            salary_cap: Total cap (defaults to 2025 cap)
            dead_money: Dead cap from cuts

        Returns:
            Available cap space
        """
        cap = salary_cap or cls.SALARY_CAP_2025
        return cap - cap_spent - dead_money

    @classmethod
    def estimate_market_value(
        cls,
        position: str,
        tier: str,
        age: int,
        position_values: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """
        Estimate market value range for a player.

        Args:
            position: Player position
            tier: Player tier (elite, above_average, average, below_average)
            age: Player age
            position_values: Position value multipliers

        Returns:
            Tuple of (min_value, max_value) as AAV
        """
        # Default position values if not provided
        if position_values is None:
            position_values = {
                "QB": 1.0,
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
            }

        # Tier multipliers
        tier_multipliers = {
            "elite": (0.08, 0.12),
            "above_average": (0.05, 0.08),
            "average": (0.02, 0.05),
            "below_average": (0.01, 0.02),
        }

        # Age adjustment (peak is 26-28)
        age_adj = 1.0
        if age < 26:
            age_adj = 0.85 + (age - 22) * 0.0375
        elif age > 28:
            age_adj = max(0.5, 1.0 - (age - 28) * 0.075)

        pos_value = position_values.get(position.upper(), 0.5)
        tier_range = tier_multipliers.get(tier.lower(), (0.02, 0.05))

        base_cap = cls.SALARY_CAP_2025
        min_pct, max_pct = tier_range

        min_value = base_cap * min_pct * pos_value * age_adj
        max_value = base_cap * max_pct * pos_value * age_adj

        return (min_value, max_value)

    @classmethod
    def format_money(cls, amount: float) -> str:
        """Format money value for display."""
        if amount >= 1_000_000:
            return f"${amount / 1_000_000:.2f}M"
        elif amount >= 1_000:
            return f"${amount / 1_000:.0f}K"
        else:
            return f"${amount:.0f}"
