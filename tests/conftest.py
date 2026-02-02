"""
Pytest Configuration and Fixtures

Comprehensive fixtures for Portal IQ and Cap IQ ML Engine tests.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Sample Player Data Fixtures
# =============================================================================

@pytest.fixture
def sample_player_data():
    """Sample player data for testing."""
    return pd.DataFrame([
        {
            "player_id": "1",
            "player_name": "John Smith",
            "position": "QB",
            "school": "Alabama",
            "conference": "SEC",
            "passing_yards": 3500,
            "passing_tds": 30,
            "interceptions": 8,
            "games": 12,
            "passer_rating": 145.5,
        },
        {
            "player_id": "2",
            "player_name": "Mike Johnson",
            "position": "RB",
            "school": "Ohio State",
            "conference": "Big Ten",
            "rushing_yards": 1200,
            "rushing_tds": 12,
            "rushing_attempts": 200,
            "receiving_yards": 300,
            "games": 12,
        },
        {
            "player_id": "3",
            "player_name": "Chris Williams",
            "position": "WR",
            "school": "Georgia",
            "conference": "SEC",
            "receiving_yards": 1100,
            "receiving_tds": 10,
            "receptions": 70,
            "games": 12,
        },
    ])


@pytest.fixture
def sample_player_features() -> pd.DataFrame:
    """Create comprehensive sample player features for NIL valuation testing."""
    np.random.seed(42)

    players = []
    positions = ["QB", "RB", "WR", "TE", "OT", "EDGE", "DT", "LB", "CB", "S"]
    schools = ["Alabama", "Ohio State", "Georgia", "Texas", "Oregon", "UCLA", "Miami", "Tennessee"]
    tiers = ["blue_blood", "elite", "power_brand", "p4_mid"]

    for i in range(100):
        pos = positions[i % len(positions)]
        school = schools[i % len(schools)]

        # Base features
        player = {
            "player_id": f"player_{i}",
            "name": f"Test Player {i}",
            "player_name": f"Test Player {i}",
            "school": school,
            "position": pos,
            "class_year": np.random.choice(["Freshman", "Sophomore", "Junior", "Senior"]),
            "eligibility_remaining": np.random.randint(1, 5),

            # Performance features
            "games_played": np.random.randint(8, 14),
            "games_started": np.random.randint(0, 13),
            "pff_grade": np.random.uniform(55, 92),

            # Position-specific stats
            "passing_yards": np.random.randint(0, 4000) if pos == "QB" else 0,
            "passing_tds": np.random.randint(0, 40) if pos == "QB" else 0,
            "interceptions": np.random.randint(0, 15) if pos == "QB" else 0,
            "completion_pct": np.random.uniform(55, 72) if pos == "QB" else 0,
            "rushing_yards": np.random.randint(0, 1500) if pos in ["RB", "QB"] else 0,
            "rushing_tds": np.random.randint(0, 20) if pos in ["RB", "QB"] else 0,
            "receiving_yards": np.random.randint(0, 1200) if pos in ["WR", "TE", "RB"] else 0,
            "receiving_tds": np.random.randint(0, 15) if pos in ["WR", "TE", "RB"] else 0,
            "receptions": np.random.randint(0, 80) if pos in ["WR", "TE", "RB"] else 0,
            "tackles": np.random.randint(0, 120) if pos in ["LB", "S", "CB", "EDGE", "DT"] else 0,
            "sacks": np.random.uniform(0, 15) if pos in ["EDGE", "DT", "LB"] else 0,
            "interceptions_def": np.random.randint(0, 8) if pos in ["CB", "S", "LB"] else 0,
            "passes_defended": np.random.randint(0, 20) if pos in ["CB", "S"] else 0,

            # Social media
            "instagram_followers": np.random.randint(1000, 500000),
            "twitter_followers": np.random.randint(500, 200000),
            "tiktok_followers": np.random.randint(0, 1000000),
            "total_followers": 0,
            "engagement_rate": np.random.uniform(0.01, 0.08),

            # Recruiting
            "stars": np.random.choice([2, 3, 3, 3, 4, 4, 5], p=[0.05, 0.25, 0.25, 0.2, 0.15, 0.07, 0.03]),
            "national_rank": np.random.randint(1, 2000),
            "position_rank": np.random.randint(1, 100),

            # School features
            "school_tier": tiers[schools.index(school) % len(tiers)],
            "conference": np.random.choice(["SEC", "Big Ten", "Big 12", "ACC", "Pac-12"]),

            # Overall rating
            "overall_rating": np.random.uniform(0.65, 0.95),
            "is_starter": np.random.choice([True, False], p=[0.5, 0.5]),
        }

        player["total_followers"] = (
            player["instagram_followers"] +
            player["twitter_followers"] +
            player["tiktok_followers"]
        )

        # Target: NIL value (correlated with features)
        base_value = 20000
        position_mult = {"QB": 4.0, "WR": 1.8, "RB": 1.5, "EDGE": 2.0, "CB": 1.7}.get(pos, 1.0)
        rating_mult = 1 + (player["overall_rating"] - 0.75) * 5
        social_bonus = player["total_followers"] / 5
        school_mult = {"blue_blood": 2.5, "elite": 1.8, "power_brand": 1.4, "p4_mid": 1.0}.get(player["school_tier"], 1.0)

        player["nil_value"] = base_value * position_mult * rating_mult * school_mult + social_bonus
        player["nil_value"] = max(5000, min(5000000, player["nil_value"]))

        # Tier classification
        if player["nil_value"] >= 1000000:
            player["nil_tier"] = "mega"
        elif player["nil_value"] >= 500000:
            player["nil_tier"] = "premium"
        elif player["nil_value"] >= 100000:
            player["nil_tier"] = "solid"
        elif player["nil_value"] >= 25000:
            player["nil_tier"] = "moderate"
        else:
            player["nil_tier"] = "entry"

        # Flight risk (for portal prediction)
        player["entered_portal"] = np.random.binomial(1, 0.25)
        player["flight_risk"] = np.random.uniform(0.1, 0.8)

        players.append(player)

    return pd.DataFrame(players)


@pytest.fixture
def sample_roster() -> pd.DataFrame:
    """Create sample roster data for a team."""
    np.random.seed(42)

    positions = ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "WR",
                 "TE", "TE", "OT", "OT", "OG", "OG", "C", "OT", "OG",
                 "EDGE", "EDGE", "DT", "DT", "DT", "LB", "LB", "LB",
                 "CB", "CB", "CB", "S", "S", "K", "P"]

    roster = []
    for i, pos in enumerate(positions):
        rating = round(np.random.uniform(0.70, 0.92), 2)
        stars = 5 if rating > 0.88 else 4 if rating > 0.80 else 3

        nil_value = int(rating * 200000 * (1.5 if pos == "QB" else 1.0))
        flight_risk = round(np.random.uniform(0.15, 0.75), 2)

        roster.append({
            "player_id": f"roster_{i}",
            "name": f"Roster Player {i}",
            "player_name": f"Roster Player {i}",
            "position": pos,
            "class_year": np.random.choice(["Freshman", "Sophomore", "Junior", "Senior"]),
            "overall_rating": rating,
            "stars": stars,
            "nil_value": nil_value,
            "flight_risk": flight_risk,
            "games_played": np.random.randint(8, 13),
            "games_started": np.random.randint(0, 13),
            "is_starter": i < 22,
            "school": "Test University",
            "snap_share": np.random.uniform(0.2, 1.0),
            "years_remaining": np.random.randint(1, 5),
            "has_transferred": np.random.choice([True, False], p=[0.2, 0.8]),
        })

    return pd.DataFrame(roster)


@pytest.fixture
def sample_portal_players() -> pd.DataFrame:
    """Create sample portal player data."""
    np.random.seed(42)

    positions = ["QB", "WR", "WR", "RB", "EDGE", "CB", "LB", "OT", "S", "DT"]
    origins = ["UCLA", "Miami", "Oregon", "FSU", "Wisconsin", "Tennessee", "Penn State", "LSU"]

    portal = []
    for i, pos in enumerate(positions):
        portal.append({
            "player_id": f"portal_{i}",
            "name": f"Portal Player {i}",
            "player_name": f"Portal Player {i}",
            "position": pos,
            "origin_school": origins[i % len(origins)],
            "stars": np.random.choice([3, 4, 4, 5]),
            "overall_rating": round(np.random.uniform(0.75, 0.90), 2),
            "nil_value": np.random.randint(100000, 600000),
            "portal_entry_date": "2024-12-15",
            "years_remaining": np.random.randint(1, 4),
            "games_played": np.random.randint(20, 40),
            "career_starts": np.random.randint(10, 35),
        })

    return pd.DataFrame(portal)


@pytest.fixture
def sample_draft_features() -> pd.DataFrame:
    """Create sample draft prospect features."""
    np.random.seed(42)

    positions = ["QB", "WR", "WR", "RB", "EDGE", "EDGE", "CB", "CB", "OT", "DT", "LB", "S"]

    prospects = []
    for i, pos in enumerate(positions * 5):
        height = np.random.randint(68, 78) if pos != "OT" else np.random.randint(74, 80)
        weight = np.random.randint(180, 250) if pos not in ["OT", "DT"] else np.random.randint(280, 330)

        prospect = {
            "player_id": f"draft_{i}",
            "name": f"Draft Prospect {i}",
            "player_name": f"Draft Prospect {i}",
            "school": np.random.choice(["Georgia", "Ohio State", "Alabama", "Michigan", "Texas"]),
            "position": pos,

            # Production
            "games_played": np.random.randint(30, 50),
            "career_starts": np.random.randint(20, 45),
            "pff_grade": np.random.uniform(70, 95),

            # Measurables
            "height": height,
            "weight": weight,
            "forty_yard": round(np.random.uniform(4.3, 5.2), 2),
            "vertical": np.random.randint(28, 42),
            "broad_jump": np.random.randint(105, 135),
            "three_cone": round(np.random.uniform(6.5, 7.5), 2),

            # Recruiting background
            "stars": np.random.choice([3, 4, 4, 5]),
            "national_rank": np.random.randint(1, 500),

            # Draft context
            "class_year": "Senior",
            "draft_year": 2025,
        }

        # Calculate targets
        rating = prospect["pff_grade"] / 100
        athletic_score = (prospect["vertical"] - 28) / 14 * 0.3 + (5.2 - prospect["forty_yard"]) / 0.9 * 0.7

        base_prob = rating * 0.6 + athletic_score * 0.2 + (5 - prospect["stars"]) * -0.05
        prospect["draft_probability"] = min(0.99, max(0.05, base_prob))
        prospect["was_drafted"] = prospect["draft_probability"] > 0.4

        if prospect["was_drafted"]:
            if prospect["draft_probability"] > 0.85:
                prospect["draft_round"] = 1
            elif prospect["draft_probability"] > 0.7:
                prospect["draft_round"] = 2
            elif prospect["draft_probability"] > 0.55:
                prospect["draft_round"] = 3
            elif prospect["draft_probability"] > 0.45:
                prospect["draft_round"] = np.random.randint(4, 6)
            else:
                prospect["draft_round"] = np.random.randint(5, 8)

            prospect["draft_pick"] = (prospect["draft_round"] - 1) * 32 + np.random.randint(1, 33)
        else:
            prospect["draft_round"] = None
            prospect["draft_pick"] = None

        prospects.append(prospect)

    return pd.DataFrame(prospects)


@pytest.fixture
def sample_team_data() -> Dict[str, Any]:
    """Create sample team/school data."""
    return {
        "school": "Test University",
        "conference": "SEC",
        "tier": "elite",
        "nil_budget": 10000000,
        "recent_wins": [9, 10, 8, 11],
        "coaching_tenure": 4,
        "recruiting_rank": 12,
    }


# =============================================================================
# NFL/Cap IQ Fixtures
# =============================================================================

@pytest.fixture
def sample_nfl_contracts():
    """Sample NFL contract data for testing."""
    return pd.DataFrame([
        {
            "player_id": "nfl_1",
            "player_name": "Patrick Mahomes",
            "position": "QB",
            "team": "KC",
            "aav": 45_000_000,
            "total_value": 450_000_000,
            "guaranteed": 141_000_000,
            "years": 10,
            "age": 28,
            "war": 8.5,
        },
        {
            "player_id": "nfl_2",
            "player_name": "Nick Bosa",
            "position": "EDGE",
            "team": "SF",
            "aav": 34_000_000,
            "total_value": 170_000_000,
            "guaranteed": 122_500_000,
            "years": 5,
            "age": 26,
            "war": 6.2,
        },
        {
            "player_id": "nfl_3",
            "player_name": "Ja'Marr Chase",
            "position": "WR",
            "team": "CIN",
            "aav": 28_000_000,
            "total_value": 140_000_000,
            "guaranteed": 100_000_000,
            "years": 5,
            "age": 24,
            "war": 4.5,
        },
    ])


@pytest.fixture
def sample_team_cap():
    """Sample team cap data for testing."""
    return {
        "team": "KC",
        "year": 2025,
        "cap_limit": 255_000_000,
        "cap_spent": 240_000_000,
        "cap_space": 15_000_000,
        "dead_money": 5_000_000,
    }


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def trained_nil_valuator(sample_player_features):
    """Create a trained NILValuator model on sample data."""
    from models.nil_valuator import NILValuator

    model = NILValuator()

    # Prepare features
    feature_cols = [
        "games_played", "games_started", "pff_grade",
        "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds", "tackles", "sacks",
        "total_followers", "engagement_rate", "stars", "overall_rating",
    ]

    X = sample_player_features[feature_cols].fillna(0)
    y_value = sample_player_features["nil_value"]
    y_tier = sample_player_features["nil_tier"]

    model.train(X, y_value, y_tier)

    return model


@pytest.fixture
def trained_portal_predictor(sample_player_features):
    """Create a trained PortalPredictor model on sample data."""
    from models.portal_predictor import PortalPredictor

    model = PortalPredictor()

    feature_cols = [
        "games_played", "games_started", "overall_rating",
        "stars", "nil_value",
    ]

    X = sample_player_features[feature_cols].fillna(0)
    y = sample_player_features["entered_portal"]

    model.train_flight_risk(X, y)

    return model


@pytest.fixture
def trained_draft_projector(sample_draft_features):
    """Create a trained DraftProjector model on sample data."""
    from models.draft_projector import DraftProjector

    model = DraftProjector()

    feature_cols = [
        "games_played", "career_starts", "pff_grade",
        "height", "weight", "forty_yard", "vertical",
        "stars", "national_rank",
    ]

    X = sample_draft_features[feature_cols].fillna(0)

    y_drafted = sample_draft_features["was_drafted"].astype(int)
    y_round = sample_draft_features["draft_round"].fillna(7)

    model.train(X, y_drafted, y_round)

    return model


# =============================================================================
# API Fixtures
# =============================================================================

@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from api.app import app

    return TestClient(app)


@pytest.fixture
def api_key():
    """Get valid API key for testing."""
    return "dev-key-123"


@pytest.fixture
def invalid_api_key():
    """Get invalid API key for testing."""
    return "invalid-key-xyz"


@pytest.fixture
def sample_api_player_request():
    """Sample player request for API testing."""
    return {
        "player": {
            "name": "Test Player",
            "school": "Alabama",
            "position": "QB",
            "class_year": "Junior",
            "stats": {
                "games_played": 12,
                "passing_yards": 3200,
                "passing_tds": 28,
                "interceptions": 5,
            },
            "social_media": {
                "instagram_followers": 150000,
                "twitter_followers": 75000,
            },
            "recruiting": {
                "stars": 5,
                "national_rank": 25,
            },
            "overall_rating": 0.88,
        }
    }


# =============================================================================
# Data Collection Fixtures
# =============================================================================

@pytest.fixture
def mock_cfb_api_response():
    """Mock response from CFB API."""
    return {
        "data": [
            {
                "id": 1,
                "school": "Alabama",
                "conference": "SEC",
                "wins": 11,
                "losses": 2,
            },
            {
                "id": 2,
                "school": "Georgia",
                "conference": "SEC",
                "wins": 13,
                "losses": 1,
            },
        ]
    }


@pytest.fixture
def sample_api_player_data():
    """Sample player data as returned from external API."""
    return [
        {
            "id": 1,
            "name": "John Smith",
            "team": "Alabama",
            "position": "QB",
            "year": "Junior",
            "stats": {
                "passing_yards": 3200,
                "passing_tds": 28,
                "interceptions": 5,
            }
        },
        {
            "id": 2,
            "name": "Mike Johnson",
            "team": "Georgia",
            "position": "RB",
            "year": "Senior",
            "stats": {
                "rushing_yards": 1200,
                "rushing_tds": 14,
            }
        },
    ]


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    (data_dir / "cache").mkdir()
    return data_dir


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for test models."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def mock_config(temp_data_dir):
    """Mock configuration for testing."""
    return {
        "data_dir": str(temp_data_dir),
        "cache_hours": 24,
        "rate_limit_seconds": 0.1,
        "salary_cap": 255_000_000,
    }


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
