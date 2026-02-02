"""
Tests for NIL Feature Engineering

Tests NIL feature building.
"""

import pytest
import pandas as pd
import numpy as np


class TestNILFeatureBuilder:
    """Tests for NILFeatureBuilder."""

    def test_initialization(self):
        """Builder initializes correctly."""
        from src.feature_engineering.portal_iq.nil_features import NILFeatureBuilder
        builder = NILFeatureBuilder()
        assert builder is not None

    def test_build_production_features_passing(self):
        """Production features work for passing stats."""
        from src.feature_engineering.portal_iq.nil_features import NILFeatureBuilder
        builder = NILFeatureBuilder()

        df = pd.DataFrame([{
            "passing_yards": 3000,
            "passing_tds": 25,
            "interceptions": 5,
            "games": 12,
            "passer_rating": 145.0,
        }])

        result = builder.build_production_features(df)

        assert "pass_yards_per_game" in result.columns
        assert result["pass_yards_per_game"].iloc[0] == 250.0
        assert "td_int_ratio" in result.columns
        assert result["td_int_ratio"].iloc[0] == 5.0

    def test_build_production_features_rushing(self):
        """Production features work for rushing stats."""
        from src.feature_engineering.portal_iq.nil_features import NILFeatureBuilder
        builder = NILFeatureBuilder()

        df = pd.DataFrame([{
            "rushing_yards": 1200,
            "rushing_attempts": 200,
            "games": 12,
        }])

        result = builder.build_production_features(df)

        assert "rush_yards_per_game" in result.columns
        assert result["rush_yards_per_game"].iloc[0] == 100.0
        assert "yards_per_carry" in result.columns
        assert result["yards_per_carry"].iloc[0] == 6.0

    def test_build_school_features(self):
        """School features work correctly."""
        from src.feature_engineering.portal_iq.nil_features import NILFeatureBuilder
        builder = NILFeatureBuilder()

        df = pd.DataFrame([
            {"school": "Alabama", "conference": "SEC"},
            {"school": "Michigan", "conference": "Big Ten"},
            {"school": "Boise State", "conference": "Mountain West"},
        ])

        result = builder.build_school_features(df)

        assert "school_tier" in result.columns
        assert result["school_tier"].iloc[0] == "blue_blood"
        assert "conference_tier" in result.columns
        assert result["conference_tier"].iloc[0] == 1

    def test_build_position_features(self):
        """Position features work correctly."""
        from src.feature_engineering.portal_iq.nil_features import NILFeatureBuilder
        builder = NILFeatureBuilder()

        df = pd.DataFrame([
            {"position": "QB"},
            {"position": "RB"},
            {"position": "OL"},
        ])

        result = builder.build_position_features(df)

        assert "position_multiplier" in result.columns
        assert result["position_multiplier"].iloc[0] == 2.5  # QB
        assert result["position_multiplier"].iloc[1] == 1.2  # RB

    def test_build_all_features(self, sample_player_data):
        """Full feature pipeline works."""
        from src.feature_engineering.portal_iq.nil_features import NILFeatureBuilder
        builder = NILFeatureBuilder()

        result = builder.build_all_features(sample_player_data)

        assert "production_score" in result.columns
        assert "nil_score" in result.columns
        assert len(result) == len(sample_player_data)

    def test_nil_score_bounded(self, sample_player_data):
        """NIL score is bounded 0-1."""
        from src.feature_engineering.portal_iq.nil_features import NILFeatureBuilder
        builder = NILFeatureBuilder()

        result = builder.build_all_features(sample_player_data)

        assert all(result["nil_score"] >= 0)
        assert all(result["nil_score"] <= 1)
