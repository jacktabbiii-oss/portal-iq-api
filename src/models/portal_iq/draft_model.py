"""
NFL Draft Projection Model

Projects draft position for college football players.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class DraftProjectionModel:
    """Projects NFL draft position for college players."""

    # Features for draft projection
    DRAFT_FEATURES = [
        "production_score",
        "athletic_score",
        "age_score",
        "school_tier",
        "size_score",
        "draft_projection_score",
    ]

    # Draft round thresholds (pick numbers)
    ROUND_THRESHOLDS = {
        1: 32,
        2: 64,
        3: 100,
        4: 140,
        5: 180,
        6: 220,
        7: 262,
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize draft projection model.

        Args:
            model_path: Path to saved model
        """
        self.model = None
        self.scaler = None
        self.is_trained = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """
        Train draft projection model.

        Args:
            X: Feature DataFrame
            y: Target draft pick numbers

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training")

        available_features = [f for f in self.DRAFT_FEATURES if f in X.columns]
        X_train = X[available_features].fillna(0)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True
        self._features_used = available_features

        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=5, scoring="neg_mean_absolute_error"
        )

        logger.info(f"Model trained. CV MAE = {-cv_scores.mean():.1f} picks")

        return {
            "mae_mean": -cv_scores.mean(),
            "mae_std": cv_scores.std(),
            "n_samples": len(y),
        }

    def predict_pick(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict draft pick number.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted pick numbers
        """
        if self.is_trained and SKLEARN_AVAILABLE:
            features = getattr(self, "_features_used", self.DRAFT_FEATURES)
            available = [f for f in features if f in X.columns]
            X_pred = X[available].fillna(0)
            X_scaled = self.scaler.transform(X_pred)
            picks = self.model.predict(X_scaled)
            return pd.Series(picks, index=X.index).clip(1, 300)

        return self._rule_based_prediction(X)

    def _rule_based_prediction(self, X: pd.DataFrame) -> pd.Series:
        """
        Rule-based draft prediction fallback.

        Args:
            X: Feature DataFrame

        Returns:
            Estimated pick numbers
        """
        picks = pd.Series(150.0, index=X.index)  # Default mid-round

        for idx, row in X.iterrows():
            pick = 150.0

            # Production adjustment
            prod = row.get("production_score", 0.5)
            pick -= prod * 100

            # Athletic adjustment
            athletic = row.get("athletic_score", 0.5)
            pick -= athletic * 50

            # Age adjustment (younger = better)
            age = row.get("age_score", 0.5)
            pick -= age * 30

            # School tier adjustment
            tier = row.get("school_tier", "p4_mid")
            tier_adj = {
                "blue_blood": -30,
                "elite": -20,
                "power_brand": -10,
                "p4_mid": 0,
                "g5_strong": 10,
                "g5": 20,
            }
            pick += tier_adj.get(tier, 0)

            picks.loc[idx] = pick

        return picks.clip(1, 300)

    def predict_round(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict draft round.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted rounds (1-7, or 8 for undrafted)
        """
        picks = self.predict_pick(X)

        rounds = pd.Series(8, index=X.index)  # Default undrafted
        for rnd, threshold in self.ROUND_THRESHOLDS.items():
            rounds = rounds.where(picks > threshold, rnd)

        return rounds

    def predict_range(
        self,
        X: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Predict draft pick range.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (low_pick, high_pick)
        """
        base_picks = self.predict_pick(X)

        # Uncertainty based on features available
        uncertainty = 20 + (1 - X.notna().mean(axis=1)) * 30

        low_picks = (base_picks - uncertainty).clip(1)
        high_picks = (base_picks + uncertainty).clip(1, 300)

        return low_picks, high_picks

    def get_draft_board(
        self,
        players: pd.DataFrame,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        Generate draft board ranking.

        Args:
            players: Player DataFrame
            top_n: Number of players to include

        Returns:
            Draft board DataFrame
        """
        board = players.copy()

        board["projected_pick"] = self.predict_pick(board)
        board["projected_round"] = self.predict_round(board)

        low, high = self.predict_range(board)
        board["pick_range_low"] = low
        board["pick_range_high"] = high

        board = board.sort_values("projected_pick")
        board["board_rank"] = range(1, len(board) + 1)

        return board.head(top_n)

    def compare_to_consensus(
        self,
        players: pd.DataFrame,
        consensus_col: str = "consensus_rank",
    ) -> pd.DataFrame:
        """
        Compare projections to consensus rankings.

        Args:
            players: Player DataFrame with consensus rankings
            consensus_col: Column name for consensus rank

        Returns:
            Comparison DataFrame
        """
        comparison = players.copy()

        comparison["model_rank"] = self.predict_pick(comparison).rank()
        comparison["rank_diff"] = (
            comparison["model_rank"] - comparison[consensus_col]
        )
        comparison["is_value"] = comparison["rank_diff"] < -10  # Model likes more
        comparison["is_overrated"] = comparison["rank_diff"] > 10  # Model dislikes

        return comparison

    def get_position_rankings(
        self,
        players: pd.DataFrame,
        position: str,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Get position-specific draft rankings.

        Args:
            players: Player DataFrame
            position: Position to filter
            top_n: Number of players

        Returns:
            Position rankings DataFrame
        """
        pos_players = players[
            players["position"].str.upper() == position.upper()
        ].copy()

        if pos_players.empty:
            return pd.DataFrame()

        pos_players["projected_pick"] = self.predict_pick(pos_players)
        pos_players["position_rank"] = pos_players["projected_pick"].rank()

        return pos_players.sort_values("projected_pick").head(top_n)

    def save(self, path: str) -> None:
        """Save model to file."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "features_used": getattr(self, "_features_used", self.DRAFT_FEATURES),
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load(self, path: str) -> None:
        """Load model from file."""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.is_trained = model_data["is_trained"]
            self._features_used = model_data.get("features_used", self.DRAFT_FEATURES)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
