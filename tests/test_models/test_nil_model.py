"""
Tests for NIL Valuation Model

Tests NIL value predictions and features.
"""

import pytest
import pandas as pd
import numpy as np


class TestNILValuationModel:
    """Tests for NILValuationModel."""

    def test_model_initialization(self):
        """Model initializes without errors."""
        from src.models.portal_iq.nil_model import NILValuationModel
        model = NILValuationModel()
        assert model is not None
        assert model.is_trained is False

    def test_rule_based_prediction(self, sample_player_data):
        """Rule-based prediction returns reasonable values."""
        from src.models.portal_iq.nil_model import NILValuationModel
        model = NILValuationModel()

        # Add required features
        sample_player_data["production_score"] = 0.7
        sample_player_data["social_score"] = 0.5
        sample_player_data["school_multiplier"] = 2.0
        sample_player_data["position_multiplier"] = 2.5
        sample_player_data["market_factor"] = 1.0

        predictions = model.predict(sample_player_data)

        assert len(predictions) == len(sample_player_data)
        assert all(predictions > 0)
        assert all(predictions < 10_000_000)

    def test_qb_valued_higher_than_rb(self, sample_player_data):
        """QB position valued higher than RB."""
        from src.models.portal_iq.nil_model import NILValuationModel
        model = NILValuationModel()

        # Create QB and RB with same stats
        qb_data = pd.DataFrame([{
            "position": "QB",
            "production_score": 0.5,
            "social_score": 0.5,
            "school_multiplier": 1.0,
            "position_multiplier": 2.5,
            "market_factor": 1.0,
        }])

        rb_data = pd.DataFrame([{
            "position": "RB",
            "production_score": 0.5,
            "social_score": 0.5,
            "school_multiplier": 1.0,
            "position_multiplier": 1.2,
            "market_factor": 1.0,
        }])

        qb_value = model.predict(qb_data).iloc[0]
        rb_value = model.predict(rb_data).iloc[0]

        assert qb_value > rb_value

    def test_blue_blood_school_premium(self):
        """Blue blood schools get NIL premium."""
        from src.models.portal_iq.nil_model import NILValuationModel
        model = NILValuationModel()

        blue_blood = pd.DataFrame([{
            "position": "WR",
            "production_score": 0.5,
            "social_score": 0.5,
            "school_multiplier": 2.0,  # Blue blood
            "position_multiplier": 1.5,
            "market_factor": 1.0,
        }])

        g5_school = pd.DataFrame([{
            "position": "WR",
            "production_score": 0.5,
            "social_score": 0.5,
            "school_multiplier": 0.5,  # G5
            "position_multiplier": 1.5,
            "market_factor": 1.0,
        }])

        bb_value = model.predict(blue_blood).iloc[0]
        g5_value = model.predict(g5_school).iloc[0]

        assert bb_value > g5_value

    def test_get_tier(self):
        """Tier assignment works correctly."""
        from src.models.portal_iq.nil_model import NILValuationModel
        model = NILValuationModel()

        assert model.get_tier(2_000_000) == "mega"
        assert model.get_tier(750_000) == "premium"
        assert model.get_tier(200_000) == "solid"
        assert model.get_tier(50_000) == "moderate"
        assert model.get_tier(10_000) == "entry"

    def test_predict_with_confidence(self, sample_player_data):
        """Prediction with confidence returns ranges."""
        from src.models.portal_iq.nil_model import NILValuationModel
        model = NILValuationModel()

        sample_player_data["production_score"] = 0.7
        sample_player_data["social_score"] = 0.5
        sample_player_data["school_multiplier"] = 1.5
        sample_player_data["position_multiplier"] = 1.5
        sample_player_data["market_factor"] = 1.0
        sample_player_data["nil_score"] = 0.6
        sample_player_data["conference_tier"] = 1

        predictions, ranges = model.predict_with_confidence(sample_player_data)

        assert len(predictions) == len(sample_player_data)
        assert len(ranges) == len(sample_player_data)
        assert all(ranges >= 0)
