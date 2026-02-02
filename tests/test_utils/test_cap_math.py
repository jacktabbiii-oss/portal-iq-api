"""
Tests for Cap Math Utilities

Tests salary cap calculation functions.
"""

import pytest


class TestCapMath:
    """Tests for CapMath utility class."""

    def test_project_cap_same_year(self):
        """Cap projection for current year."""
        from src.utils.cap_math import CapMath

        projected = CapMath.project_cap(2025, base_year=2025)
        assert projected == CapMath.SALARY_CAP_2025

    def test_project_cap_future_year(self):
        """Cap projection for future years."""
        from src.utils.cap_math import CapMath

        projected_2026 = CapMath.project_cap(2026, base_year=2025)
        expected = CapMath.SALARY_CAP_2025 * 1.07

        assert abs(projected_2026 - expected) < 1  # Within $1

    def test_prorate_signing_bonus(self):
        """Signing bonus proration works correctly."""
        from src.utils.cap_math import CapMath

        # 5-year proration
        prorated = CapMath.prorate_signing_bonus(25_000_000, 5)
        assert len(prorated) == 5
        assert all(p == 5_000_000 for p in prorated)

        # Max 5 years even for longer contracts
        prorated_long = CapMath.prorate_signing_bonus(25_000_000, 7)
        assert len(prorated_long) == 5

    def test_calculate_aav(self):
        """AAV calculation works correctly."""
        from src.utils.cap_math import CapMath

        aav = CapMath.calculate_aav(100_000_000, 4)
        assert aav == 25_000_000

        # Handle zero years
        aav_zero = CapMath.calculate_aav(100_000_000, 0)
        assert aav_zero == 0

    def test_calculate_gtd_percentage(self):
        """Guaranteed percentage calculation."""
        from src.utils.cap_math import CapMath

        pct = CapMath.calculate_gtd_percentage(50_000_000, 100_000_000)
        assert pct == 50.0

        pct_zero = CapMath.calculate_gtd_percentage(50_000_000, 0)
        assert pct_zero == 0.0

    def test_calculate_cap_percentage(self):
        """Cap percentage calculation."""
        from src.utils.cap_math import CapMath

        pct = CapMath.calculate_cap_percentage(25_500_000)
        assert pct == 10.0  # 25.5M / 255M = 10%

    def test_calculate_surplus_value(self):
        """Surplus value calculation."""
        from src.utils.cap_math import CapMath

        # Player worth $15M getting paid $10M
        surplus = CapMath.calculate_surplus_value(
            cap_hit=10_000_000,
            war=5.0,
            dollars_per_war=3_000_000,
        )

        assert surplus == 5_000_000  # $15M - $10M

    def test_get_veteran_minimum(self):
        """Veteran minimum lookup."""
        from src.utils.cap_math import CapMath

        assert CapMath.get_veteran_minimum(0) == 795_000  # Rookie
        assert CapMath.get_veteran_minimum(4) == 1_185_000
        assert CapMath.get_veteran_minimum(10) == 1_425_000  # 7+ years

    def test_calculate_cap_space(self):
        """Cap space calculation."""
        from src.utils.cap_math import CapMath

        space = CapMath.calculate_cap_space(
            cap_spent=240_000_000,
            dead_money=5_000_000,
        )

        expected = 255_000_000 - 240_000_000 - 5_000_000
        assert space == expected

    def test_estimate_market_value(self):
        """Market value estimation."""
        from src.utils.cap_math import CapMath

        min_val, max_val = CapMath.estimate_market_value(
            position="QB",
            tier="elite",
            age=27,
        )

        assert min_val > 0
        assert max_val > min_val
        assert max_val < 100_000_000

    def test_format_money(self):
        """Money formatting."""
        from src.utils.cap_math import CapMath

        assert CapMath.format_money(25_500_000) == "$25.50M"
        assert CapMath.format_money(500_000) == "$500K"
        assert CapMath.format_money(500) == "$500"

    def test_calculate_dead_money(self):
        """Dead money calculation."""
        from src.utils.cap_math import CapMath

        dead = CapMath.calculate_dead_money(
            remaining_guarantees=5_000_000,
            remaining_prorated_bonus=10_000_000,
        )

        assert dead == 15_000_000
