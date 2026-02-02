"""
Tests for Surplus Value Model

Tests surplus value calculations.
"""

import pytest
import pandas as pd


class TestSurplusValueModel:
    """Tests for SurplusValueModel."""

    def test_model_initialization(self):
        """Model initializes with default values."""
        from src.models.cap_iq.surplus_model import SurplusValueModel
        model = SurplusValueModel()

        assert model.salary_cap == 255_000_000
        assert model.dollars_per_war == 3_000_000

    def test_calculate_expected_value(self):
        """Expected value calculation works correctly."""
        from src.models.cap_iq.surplus_model import SurplusValueModel
        model = SurplusValueModel()

        # QB with 5 WAR
        expected = model.calculate_expected_value(5.0, "QB")
        assert expected == 5.0 * 3_000_000 * 1.5  # QB factor = 1.5

        # RB with 2 WAR
        expected = model.calculate_expected_value(2.0, "RB")
        assert expected == 2.0 * 3_000_000 * 0.7  # RB factor = 0.7

    def test_calculate_surplus_positive(self):
        """Positive surplus for team-friendly contract."""
        from src.models.cap_iq.surplus_model import SurplusValueModel
        model = SurplusValueModel()

        # Player worth $15M getting paid $10M
        surplus = model.calculate_surplus(
            cap_hit=10_000_000,
            war=5.0,
            position="WR",
        )

        expected_value = 5.0 * 3_000_000 * 1.0  # $15M
        assert surplus == expected_value - 10_000_000
        assert surplus > 0

    def test_calculate_surplus_negative(self):
        """Negative surplus for overpaid player."""
        from src.models.cap_iq.surplus_model import SurplusValueModel
        model = SurplusValueModel()

        # Player worth $6M getting paid $15M
        surplus = model.calculate_surplus(
            cap_hit=15_000_000,
            war=2.0,
            position="WR",
        )

        expected_value = 2.0 * 3_000_000 * 1.0  # $6M
        assert surplus == expected_value - 15_000_000
        assert surplus < 0

    def test_calculate_surplus_df(self, sample_nfl_contracts):
        """DataFrame surplus calculation works."""
        from src.models.cap_iq.surplus_model import SurplusValueModel
        model = SurplusValueModel()

        # Rename aav to cap_hit for the test
        sample_nfl_contracts["cap_hit"] = sample_nfl_contracts["aav"]

        result = model.calculate_surplus_df(sample_nfl_contracts)

        assert "expected_value" in result.columns
        assert "surplus_value" in result.columns
        assert "contract_grade" in result.columns
        assert len(result) == len(sample_nfl_contracts)

    def test_contract_grades(self):
        """Contract grading works correctly."""
        from src.models.cap_iq.surplus_model import SurplusValueModel
        model = SurplusValueModel()

        assert model._grade_contract(60) == "A+"
        assert model._grade_contract(35) == "A"
        assert model._grade_contract(20) == "B+"
        assert model._grade_contract(5) == "B"
        assert model._grade_contract(-10) == "C"
        assert model._grade_contract(-25) == "D"
        assert model._grade_contract(-50) == "F"

    def test_get_team_surplus(self, sample_nfl_contracts):
        """Team surplus analysis works."""
        from src.models.cap_iq.surplus_model import SurplusValueModel
        model = SurplusValueModel()

        sample_nfl_contracts["cap_hit"] = sample_nfl_contracts["aav"]

        result = model.get_team_surplus(sample_nfl_contracts)

        assert "total_surplus" in result
        assert "avg_surplus" in result
        assert "best_contract" in result
        assert "worst_contract" in result
        assert "roster_size" in result
        assert result["roster_size"] == len(sample_nfl_contracts)

    def test_estimate_war_qb(self):
        """WAR estimation for QB works."""
        from src.models.cap_iq.surplus_model import SurplusValueModel
        model = SurplusValueModel()

        player_data = {
            "position": "QB",
            "passer_rating": 100,
            "games": 17,
        }

        war = model.estimate_war(player_data)
        assert war > 0
        assert war < 10

    def test_estimate_war_rb(self):
        """WAR estimation for RB works."""
        from src.models.cap_iq.surplus_model import SurplusValueModel
        model = SurplusValueModel()

        player_data = {
            "position": "RB",
            "rushing_yards": 1000,
            "receiving_yards": 500,
        }

        war = model.estimate_war(player_data)
        assert war == 1500 / 500  # 3.0

    def test_project_future_surplus(self):
        """Future surplus projection works."""
        from src.models.cap_iq.surplus_model import SurplusValueModel
        model = SurplusValueModel()

        player_data = {
            "position": "WR",
            "war": 4.0,
            "future_cap_hits": [15_000_000, 18_000_000, 20_000_000],
        }

        projections = model.project_future_surplus(player_data, years=3)

        assert len(projections) == 3
        assert all("year" in p for p in projections)
        assert all("projected_war" in p for p in projections)
        assert all("surplus_value" in p for p in projections)

        # WAR should decline each year
        assert projections[0]["projected_war"] > projections[2]["projected_war"]
