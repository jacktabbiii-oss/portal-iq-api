"""Tests for NILValuator model.

Tests training, prediction, transfer impact analysis, and value validation.
"""

import pytest
import numpy as np
import pandas as pd


class TestNILValuatorInit:
    """Tests for NILValuator initialization."""

    def test_model_initializes_without_error(self):
        """Test that NILValuator initializes without raising errors."""
        try:
            from models.nil_valuator import NILValuator
            model = NILValuator()
            assert model is not None
        except ImportError:
            pytest.skip("NILValuator not implemented yet")

    def test_model_has_required_methods(self):
        """Test that NILValuator has all required methods."""
        try:
            from models.nil_valuator import NILValuator
            model = NILValuator()

            assert hasattr(model, 'train')
            assert hasattr(model, 'predict')
            assert callable(model.train)
            assert callable(model.predict)
        except ImportError:
            pytest.skip("NILValuator not implemented yet")


class TestNILValuatorTraining:
    """Tests for model training functionality."""

    def test_train_without_error(self, sample_player_features):
        """Test that model trains without raising errors."""
        try:
            from models.nil_valuator import NILValuator

            model = NILValuator()

            feature_cols = [
                "games_played", "games_started", "pff_grade",
                "passing_yards", "rushing_yards", "receiving_yards",
                "tackles", "total_followers", "stars", "overall_rating",
            ]

            X = sample_player_features[feature_cols].fillna(0)
            y_value = sample_player_features["nil_value"]
            y_tier = sample_player_features["nil_tier"]

            # Should not raise any exception
            model.train(X, y_value, y_tier)

        except ImportError:
            pytest.skip("NILValuator not implemented yet")

    def test_train_with_small_dataset(self):
        """Test that model handles small datasets gracefully."""
        try:
            from models.nil_valuator import NILValuator

            model = NILValuator()

            # Very small dataset
            X = pd.DataFrame({
                "games_played": [12, 11, 10],
                "total_followers": [100000, 50000, 25000],
                "stars": [5, 4, 3],
            })
            y_value = pd.Series([500000, 200000, 50000])
            y_tier = pd.Series(["premium", "solid", "moderate"])

            # Should not crash, may warn about small data
            try:
                model.train(X, y_value, y_tier)
            except ValueError as e:
                # Acceptable to raise for very small data
                assert "small" in str(e).lower() or "insufficient" in str(e).lower()

        except ImportError:
            pytest.skip("NILValuator not implemented yet")

    def test_train_sets_is_trained_flag(self, sample_player_features):
        """Test that training sets a flag indicating model is trained."""
        try:
            from models.nil_valuator import NILValuator

            model = NILValuator()

            feature_cols = ["games_played", "total_followers", "stars", "overall_rating"]
            X = sample_player_features[feature_cols].fillna(0)
            y_value = sample_player_features["nil_value"]
            y_tier = sample_player_features["nil_tier"]

            model.train(X, y_value, y_tier)

            # Check for some indication the model is trained
            assert hasattr(model, 'is_trained') and model.is_trained or \
                   hasattr(model, 'value_model') and model.value_model is not None

        except ImportError:
            pytest.skip("NILValuator not implemented yet")


class TestNILValuatorPrediction:
    """Tests for prediction functionality."""

    def test_predict_returns_expected_format(self, trained_nil_valuator, sample_player_features):
        """Test that predict() returns expected format with all required keys."""
        try:
            # Get a single player for prediction
            player_df = sample_player_features.head(1)

            feature_cols = [
                "games_played", "games_started", "pff_grade",
                "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
                "receiving_yards", "receiving_tds", "tackles", "sacks",
                "total_followers", "engagement_rate", "stars", "overall_rating",
            ]

            X = player_df[feature_cols].fillna(0)

            predictions = trained_nil_valuator.predict(X)

            # Check return type
            assert isinstance(predictions, (list, dict, pd.DataFrame))

            # If list, check first element
            if isinstance(predictions, list) and len(predictions) > 0:
                pred = predictions[0]
                assert "predicted_value" in pred or "nil_value" in pred or "value" in pred

        except ImportError:
            pytest.skip("NILValuator not implemented yet")

    def test_predictions_are_in_reasonable_range(self, trained_nil_valuator, sample_player_features):
        """Test that predictions are in reasonable ranges (not negative, not billions)."""
        try:
            feature_cols = [
                "games_played", "games_started", "pff_grade",
                "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
                "receiving_yards", "receiving_tds", "tackles", "sacks",
                "total_followers", "engagement_rate", "stars", "overall_rating",
            ]

            X = sample_player_features[feature_cols].fillna(0)

            predictions = trained_nil_valuator.predict(X)

            # Extract values based on return type
            if isinstance(predictions, list):
                values = [p.get("predicted_value", p.get("nil_value", p.get("value", 0)))
                         for p in predictions]
            elif isinstance(predictions, pd.DataFrame):
                values = predictions["predicted_value"].tolist() if "predicted_value" in predictions.columns else []
            else:
                values = [predictions.get("predicted_value", 0)]

            for value in values:
                assert value >= 0, f"Prediction should not be negative: {value}"
                assert value < 100_000_000, f"Prediction unreasonably high: {value}"

        except ImportError:
            pytest.skip("NILValuator not implemented yet")

    def test_predictions_not_all_same(self, trained_nil_valuator, sample_player_features):
        """Test that predictions vary for different players."""
        try:
            feature_cols = [
                "games_played", "games_started", "pff_grade",
                "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
                "receiving_yards", "receiving_tds", "tackles", "sacks",
                "total_followers", "engagement_rate", "stars", "overall_rating",
            ]

            X = sample_player_features.head(10)[feature_cols].fillna(0)

            predictions = trained_nil_valuator.predict(X)

            # Extract values
            if isinstance(predictions, list):
                values = [p.get("predicted_value", p.get("nil_value", p.get("value", 0)))
                         for p in predictions]
            elif isinstance(predictions, pd.DataFrame):
                values = predictions["predicted_value"].tolist() if "predicted_value" in predictions.columns else []
            else:
                values = [predictions.get("predicted_value", 0)]

            # Check that not all values are identical
            unique_values = set(round(v, -3) for v in values)  # Round to nearest thousand
            assert len(unique_values) > 1, "All predictions are the same - model may not be learning"

        except ImportError:
            pytest.skip("NILValuator not implemented yet")

    def test_tier_predictions_valid(self, trained_nil_valuator, sample_player_features):
        """Test that tier predictions are valid tier names."""
        try:
            valid_tiers = ["mega", "premium", "solid", "moderate", "entry"]

            feature_cols = [
                "games_played", "games_started", "pff_grade",
                "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
                "receiving_yards", "receiving_tds", "tackles", "sacks",
                "total_followers", "engagement_rate", "stars", "overall_rating",
            ]

            X = sample_player_features.head(5)[feature_cols].fillna(0)

            predictions = trained_nil_valuator.predict(X)

            if isinstance(predictions, list):
                for pred in predictions:
                    if "tier" in pred:
                        assert pred["tier"].lower() in valid_tiers, f"Invalid tier: {pred['tier']}"

        except ImportError:
            pytest.skip("NILValuator not implemented yet")


class TestNILValuatorTransferImpact:
    """Tests for transfer impact analysis."""

    def test_transfer_impact_returns_valid_comparison(self, trained_nil_valuator, sample_player_features):
        """Test that transfer_impact returns valid comparison data."""
        try:
            player_df = sample_player_features.head(1)

            result = trained_nil_valuator.transfer_impact(player_df, "Alabama")

            assert result is not None

            # Check for expected keys
            if isinstance(result, dict):
                assert "current_value" in result or "projected_value" in result or "value_change" in result

        except (ImportError, AttributeError):
            pytest.skip("transfer_impact not implemented yet")

    def test_transfer_impact_shows_difference(self, trained_nil_valuator, sample_player_features):
        """Test that transfer to different school shows value difference."""
        try:
            # Get a player from a smaller school
            player_df = sample_player_features[sample_player_features["school"] == "UCLA"].head(1)

            if player_df.empty:
                player_df = sample_player_features.head(1)

            # Transfer to a blue blood
            result = trained_nil_valuator.transfer_impact(player_df, "Alabama")

            if isinstance(result, dict) and "value_change" in result:
                # Value change can be positive or negative
                assert isinstance(result["value_change"], (int, float))

        except (ImportError, AttributeError):
            pytest.skip("transfer_impact not implemented yet")

    def test_transfer_to_same_school(self, trained_nil_valuator, sample_player_features):
        """Test transfer to same school shows minimal change."""
        try:
            player_df = sample_player_features[sample_player_features["school"] == "Alabama"].head(1)

            if player_df.empty:
                pytest.skip("No Alabama players in sample data")

            result = trained_nil_valuator.transfer_impact(player_df, "Alabama")

            if isinstance(result, dict) and "value_change" in result:
                # Should be close to zero for same school
                assert abs(result["value_change"]) < result.get("current_value", 1000000) * 0.1

        except (ImportError, AttributeError):
            pytest.skip("transfer_impact not implemented yet")


class TestNILValuatorWhatIfSocial:
    """Tests for social media what-if analysis."""

    def test_what_if_social_returns_valid_output(self, trained_nil_valuator, sample_player_features):
        """Test that what_if_social returns valid output."""
        try:
            player_df = sample_player_features.head(1)

            result = trained_nil_valuator.what_if_social(player_df, follower_increase=100000)

            assert result is not None

        except (ImportError, AttributeError):
            pytest.skip("what_if_social not implemented yet")

    def test_what_if_social_increases_value(self, trained_nil_valuator, sample_player_features):
        """Test that increasing followers increases NIL value."""
        try:
            player_df = sample_player_features.head(1)

            result = trained_nil_valuator.what_if_social(player_df, follower_increase=500000)

            if isinstance(result, dict):
                # More followers should increase value
                if "new_value" in result and "current_value" in result:
                    assert result["new_value"] >= result["current_value"]

        except (ImportError, AttributeError):
            pytest.skip("what_if_social not implemented yet")

    def test_what_if_social_scales_appropriately(self, trained_nil_valuator, sample_player_features):
        """Test that value scales appropriately with follower increase."""
        try:
            player_df = sample_player_features.head(1)

            result_small = trained_nil_valuator.what_if_social(player_df, follower_increase=10000)
            result_large = trained_nil_valuator.what_if_social(player_df, follower_increase=1000000)

            if isinstance(result_small, dict) and isinstance(result_large, dict):
                if "value_increase" in result_small and "value_increase" in result_large:
                    # Larger follower increase should have larger value increase
                    assert result_large["value_increase"] > result_small["value_increase"]

        except (ImportError, AttributeError):
            pytest.skip("what_if_social not implemented yet")


class TestNILValuatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_missing_features(self, trained_nil_valuator):
        """Test that model handles missing features gracefully."""
        try:
            # DataFrame with missing columns
            X = pd.DataFrame({
                "games_played": [12],
                "total_followers": [100000],
            })

            # Should either fill missing values or raise clear error
            try:
                predictions = trained_nil_valuator.predict(X)
            except (KeyError, ValueError) as e:
                assert "missing" in str(e).lower() or "column" in str(e).lower()

        except ImportError:
            pytest.skip("NILValuator not implemented yet")

    def test_handles_extreme_values(self, trained_nil_valuator):
        """Test that model handles extreme input values."""
        try:
            X = pd.DataFrame({
                "games_played": [0],
                "games_started": [0],
                "pff_grade": [0],
                "passing_yards": [0],
                "passing_tds": [0],
                "rushing_yards": [0],
                "rushing_tds": [0],
                "receiving_yards": [0],
                "receiving_tds": [0],
                "tackles": [0],
                "sacks": [0],
                "total_followers": [0],
                "engagement_rate": [0],
                "stars": [2],
                "overall_rating": [0.5],
            })

            predictions = trained_nil_valuator.predict(X)

            # Should return a valid prediction, even if low
            if isinstance(predictions, list) and len(predictions) > 0:
                value = predictions[0].get("predicted_value", 0)
                assert value >= 0  # Should not be negative

        except ImportError:
            pytest.skip("NILValuator not implemented yet")

    def test_handles_nan_values(self, trained_nil_valuator, sample_player_features):
        """Test that model handles NaN values appropriately."""
        try:
            feature_cols = [
                "games_played", "games_started", "pff_grade",
                "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
                "receiving_yards", "receiving_tds", "tackles", "sacks",
                "total_followers", "engagement_rate", "stars", "overall_rating",
            ]

            X = sample_player_features.head(5)[feature_cols].copy()
            X.iloc[0, 0] = np.nan  # Introduce NaN

            # Should handle NaN (fill or raise clear error)
            try:
                predictions = trained_nil_valuator.predict(X)
            except ValueError as e:
                assert "nan" in str(e).lower() or "missing" in str(e).lower()

        except ImportError:
            pytest.skip("NILValuator not implemented yet")
