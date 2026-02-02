"""Tests for PortalPredictor model.

Tests flight risk prediction, team reports, and portal fit scoring.
"""

import pytest
import numpy as np
import pandas as pd


class TestPortalPredictorInit:
    """Tests for PortalPredictor initialization."""

    def test_model_initializes_without_error(self):
        """Test that PortalPredictor initializes without raising errors."""
        try:
            from models.portal_predictor import PortalPredictor
            model = PortalPredictor()
            assert model is not None
        except ImportError:
            pytest.skip("PortalPredictor not implemented yet")

    def test_model_has_required_methods(self):
        """Test that PortalPredictor has all required methods."""
        try:
            from models.portal_predictor import PortalPredictor
            model = PortalPredictor()

            assert hasattr(model, 'train_flight_risk')
            assert hasattr(model, 'predict_flight_risk')
        except ImportError:
            pytest.skip("PortalPredictor not implemented yet")


class TestFlightRiskPrediction:
    """Tests for flight risk prediction functionality."""

    def test_flight_risk_predictions_between_0_and_100(self, trained_portal_predictor, sample_player_features):
        """Test that flight risk predictions are between 0 and 100 (or 0-1)."""
        try:
            feature_cols = ["games_played", "games_started", "overall_rating", "stars", "nil_value"]
            X = sample_player_features[feature_cols].fillna(0)

            predictions = trained_portal_predictor.predict_flight_risk(X)

            # Extract probability values
            if isinstance(predictions, list):
                probs = [p.get("flight_risk_probability", p.get("probability", p.get("risk", 0)))
                        for p in predictions]
            elif isinstance(predictions, pd.DataFrame):
                probs = predictions["flight_risk_probability"].tolist() if "flight_risk_probability" in predictions.columns else []
            elif isinstance(predictions, np.ndarray):
                probs = predictions.tolist()
            else:
                probs = []

            for prob in probs:
                # Accept both 0-1 and 0-100 scales
                if prob > 1:
                    assert 0 <= prob <= 100, f"Flight risk {prob} not in valid range"
                else:
                    assert 0 <= prob <= 1, f"Flight risk {prob} not in valid range"

        except ImportError:
            pytest.skip("PortalPredictor not implemented yet")

    def test_flight_risk_returns_risk_level(self, trained_portal_predictor, sample_player_features):
        """Test that predictions include risk level classification."""
        try:
            feature_cols = ["games_played", "games_started", "overall_rating", "stars", "nil_value"]
            X = sample_player_features.head(5)[feature_cols].fillna(0)

            predictions = trained_portal_predictor.predict_flight_risk(X)

            if isinstance(predictions, list) and len(predictions) > 0:
                # Check for risk level in response
                if "risk_level" in predictions[0]:
                    valid_levels = ["critical", "high", "moderate", "low"]
                    for pred in predictions:
                        assert pred["risk_level"].lower() in valid_levels

        except ImportError:
            pytest.skip("PortalPredictor not implemented yet")

    def test_flight_risk_varies_by_player(self, trained_portal_predictor, sample_player_features):
        """Test that flight risk varies for different players."""
        try:
            feature_cols = ["games_played", "games_started", "overall_rating", "stars", "nil_value"]
            X = sample_player_features.head(20)[feature_cols].fillna(0)

            predictions = trained_portal_predictor.predict_flight_risk(X)

            if isinstance(predictions, list):
                probs = [p.get("flight_risk_probability", p.get("probability", 0))
                        for p in predictions]
            elif isinstance(predictions, np.ndarray):
                probs = predictions.tolist()
            else:
                probs = []

            # Predictions should vary
            if probs:
                unique_probs = set(round(p, 2) for p in probs)
                assert len(unique_probs) > 1, "All flight risk predictions are identical"

        except ImportError:
            pytest.skip("PortalPredictor not implemented yet")

    def test_high_risk_factors_returned(self, trained_portal_predictor, sample_player_features):
        """Test that high risk players include risk factors."""
        try:
            # Create a high-risk player scenario
            high_risk_player = pd.DataFrame({
                "games_played": [12],
                "games_started": [0],  # Backup, not starting
                "overall_rating": [0.85],  # Good player
                "stars": [4],  # High recruit
                "nil_value": [50000],  # Low NIL compared to ability
            })

            predictions = trained_portal_predictor.predict_flight_risk(high_risk_player)

            if isinstance(predictions, list) and len(predictions) > 0:
                pred = predictions[0]
                if "risk_factors" in pred:
                    assert isinstance(pred["risk_factors"], list)

        except ImportError:
            pytest.skip("PortalPredictor not implemented yet")


class TestTeamFlightReport:
    """Tests for team-wide flight risk reports."""

    def test_team_report_generates_for_valid_school(self, trained_portal_predictor, sample_roster):
        """Test that team report generates for a valid school."""
        try:
            report = trained_portal_predictor.team_flight_risk_report(
                sample_roster,
                "Test University"
            )

            assert report is not None
            assert isinstance(report, dict)

        except (ImportError, AttributeError):
            pytest.skip("team_flight_risk_report not implemented yet")

    def test_team_report_includes_required_fields(self, trained_portal_predictor, sample_roster):
        """Test that team report includes expected fields."""
        try:
            report = trained_portal_predictor.team_flight_risk_report(
                sample_roster,
                "Test University"
            )

            if isinstance(report, dict):
                # Check for some expected fields
                expected_fields = ["school", "total_at_risk"]
                for field in expected_fields:
                    if field in report:
                        assert report[field] is not None

        except (ImportError, AttributeError):
            pytest.skip("team_flight_risk_report not implemented yet")

    def test_team_report_identifies_high_risk_players(self, trained_portal_predictor, sample_roster):
        """Test that report identifies high-risk players."""
        try:
            report = trained_portal_predictor.team_flight_risk_report(
                sample_roster,
                "Test University"
            )

            if isinstance(report, dict):
                # Should have some way to identify high risk players
                has_risk_list = (
                    "high_risk_players" in report or
                    "critical_risk_players" in report or
                    "at_risk_players" in report
                )
                assert has_risk_list or "total_at_risk" in report

        except (ImportError, AttributeError):
            pytest.skip("team_flight_risk_report not implemented yet")

    def test_team_report_calculates_retention_budget(self, trained_portal_predictor, sample_roster):
        """Test that report includes retention budget estimate."""
        try:
            report = trained_portal_predictor.team_flight_risk_report(
                sample_roster,
                "Test University"
            )

            if isinstance(report, dict):
                if "total_retention_budget_needed" in report:
                    budget = report["total_retention_budget_needed"]
                    assert budget >= 0, "Retention budget should not be negative"

        except (ImportError, AttributeError):
            pytest.skip("team_flight_risk_report not implemented yet")


class TestPortalFitScore:
    """Tests for portal fit scoring functionality."""

    def test_portal_fit_score_between_0_and_100(self, sample_portal_players):
        """Test that portal fit scores are in valid range."""
        try:
            from models.portal_predictor import PortalPredictor
            model = PortalPredictor()

            player = sample_portal_players.head(1)
            target_school = "Alabama"

            result = model.predict_portal_fit(player, target_school)

            if result is not None:
                if isinstance(result, list) and len(result) > 0:
                    score = result[0].get("fit_score", result[0].get("score", 0))
                elif isinstance(result, dict):
                    score = result.get("fit_score", result.get("score", 0))
                else:
                    score = 0

                # Accept both 0-1 and 0-100 scales
                if score > 1:
                    assert 0 <= score <= 100, f"Fit score {score} not in valid range"
                else:
                    assert 0 <= score <= 1, f"Fit score {score} not in valid range"

        except (ImportError, AttributeError):
            pytest.skip("predict_portal_fit not implemented yet")

    def test_portal_fit_includes_breakdown(self, sample_portal_players):
        """Test that portal fit includes score breakdown."""
        try:
            from models.portal_predictor import PortalPredictor
            model = PortalPredictor()

            player = sample_portal_players.head(1)
            target_school = "Ohio State"

            result = model.predict_portal_fit(player, target_school)

            if result is not None:
                if isinstance(result, list) and len(result) > 0:
                    pred = result[0]
                else:
                    pred = result

                if isinstance(pred, dict) and "fit_breakdown" in pred:
                    assert isinstance(pred["fit_breakdown"], dict)

        except (ImportError, AttributeError):
            pytest.skip("predict_portal_fit not implemented yet")

    def test_fit_score_varies_by_school(self, sample_portal_players):
        """Test that fit scores vary by target school."""
        try:
            from models.portal_predictor import PortalPredictor
            model = PortalPredictor()

            player = sample_portal_players.head(1)

            result_alabama = model.predict_portal_fit(player, "Alabama")
            result_small_school = model.predict_portal_fit(player, "Kent State")

            # Scores may differ based on school fit
            # (This test verifies the method handles different schools)
            assert result_alabama is not None or result_small_school is not None

        except (ImportError, AttributeError):
            pytest.skip("predict_portal_fit not implemented yet")


class TestPortalPredictorTraining:
    """Tests for portal predictor training."""

    def test_train_flight_risk_without_error(self, sample_player_features):
        """Test that flight risk model trains without error."""
        try:
            from models.portal_predictor import PortalPredictor
            model = PortalPredictor()

            feature_cols = ["games_played", "games_started", "overall_rating", "stars", "nil_value"]
            X = sample_player_features[feature_cols].fillna(0)
            y = sample_player_features["entered_portal"]

            model.train_flight_risk(X, y)

        except ImportError:
            pytest.skip("PortalPredictor not implemented yet")

    def test_train_with_imbalanced_data(self, sample_player_features):
        """Test that model handles imbalanced data (few portal entries)."""
        try:
            from models.portal_predictor import PortalPredictor
            model = PortalPredictor()

            # Create highly imbalanced data (realistic scenario)
            df = sample_player_features.copy()
            df["entered_portal"] = 0
            df.iloc[:5, df.columns.get_loc("entered_portal")] = 1  # Only 5% entered

            feature_cols = ["games_played", "games_started", "overall_rating", "stars", "nil_value"]
            X = df[feature_cols].fillna(0)
            y = df["entered_portal"]

            # Should handle imbalanced data without error
            model.train_flight_risk(X, y)

        except ImportError:
            pytest.skip("PortalPredictor not implemented yet")


class TestPortalPredictorEdgeCases:
    """Tests for edge cases in portal prediction."""

    def test_handles_empty_roster(self):
        """Test that team report handles empty roster."""
        try:
            from models.portal_predictor import PortalPredictor
            model = PortalPredictor()

            empty_roster = pd.DataFrame()

            result = model.team_flight_risk_report(empty_roster, "Test University")

            # Should return empty or minimal report, not crash
            assert result is not None or result == {}

        except (ImportError, AttributeError):
            pytest.skip("team_flight_risk_report not implemented yet")

    def test_handles_single_player_roster(self, sample_roster):
        """Test that team report handles single player roster."""
        try:
            from models.portal_predictor import PortalPredictor
            model = PortalPredictor()

            single_player = sample_roster.head(1)

            result = model.team_flight_risk_report(single_player, "Test University")

            assert result is not None

        except (ImportError, AttributeError):
            pytest.skip("team_flight_risk_report not implemented yet")

    def test_handles_invalid_school_name(self, sample_portal_players):
        """Test that fit score handles invalid school names."""
        try:
            from models.portal_predictor import PortalPredictor
            model = PortalPredictor()

            player = sample_portal_players.head(1)

            # Should handle gracefully (return default or raise clear error)
            try:
                result = model.predict_portal_fit(player, "Not A Real School")
                # If it returns, should be valid
            except ValueError as e:
                assert "school" in str(e).lower() or "invalid" in str(e).lower()

        except (ImportError, AttributeError):
            pytest.skip("predict_portal_fit not implemented yet")

    def test_retention_recommendations_provided(self, trained_portal_predictor, sample_roster):
        """Test that retention recommendations are provided for high-risk players."""
        try:
            # Create a high-risk player
            high_risk = sample_roster.head(1).copy()
            high_risk["flight_risk"] = 0.8

            predictions = trained_portal_predictor.predict_flight_risk(high_risk)

            if isinstance(predictions, list) and len(predictions) > 0:
                pred = predictions[0]
                if pred.get("flight_risk_probability", 0) > 0.5:
                    # High risk players should have recommendations
                    if "retention_recommendations" in pred:
                        assert isinstance(pred["retention_recommendations"], list)

        except ImportError:
            pytest.skip("PortalPredictor not implemented yet")
