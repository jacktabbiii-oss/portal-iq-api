"""
Transfer Portal Prediction Model

Predicts portal entry risk and destination matches.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class PortalPredictionModel:
    """Predicts transfer portal entry and destinations."""

    # Features for portal entry prediction
    PORTAL_FEATURES = [
        "snap_share",
        "is_starter",
        "new_coach",
        "years_remaining",
        "has_transferred",
        "depth_score",
        "portal_risk_score",
    ]

    # Risk thresholds
    RISK_LEVELS = {
        "high": 0.7,
        "medium": 0.4,
        "low": 0.0,
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize portal prediction model.

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
        Train portal entry prediction model.

        Args:
            X: Feature DataFrame
            y: Binary target (entered portal or not)

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training")

        available_features = [f for f in self.PORTAL_FEATURES if f in X.columns]
        X_train = X[available_features].fillna(0)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True
        self._features_used = available_features

        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=5, scoring="roc_auc"
        )

        logger.info(f"Model trained. CV AUC = {cv_scores.mean():.3f}")

        return {
            "auc_mean": cv_scores.mean(),
            "auc_std": cv_scores.std(),
            "n_samples": len(y),
        }

    def predict_probability(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict probability of entering portal.

        Args:
            X: Feature DataFrame

        Returns:
            Probabilities
        """
        if self.is_trained and SKLEARN_AVAILABLE:
            features = getattr(self, "_features_used", self.PORTAL_FEATURES)
            available = [f for f in features if f in X.columns]
            X_pred = X[available].fillna(0)
            X_scaled = self.scaler.transform(X_pred)
            probs = self.model.predict_proba(X_scaled)[:, 1]
            return pd.Series(probs, index=X.index)

        return self._rule_based_probability(X)

    def _rule_based_probability(self, X: pd.DataFrame) -> pd.Series:
        """
        Rule-based portal probability fallback.

        Args:
            X: Feature DataFrame

        Returns:
            Estimated probabilities
        """
        probs = pd.Series(0.15, index=X.index)  # Base rate ~15%

        for idx, row in X.iterrows():
            prob = 0.15

            # Low playing time increases risk
            snap_share = row.get("snap_share", 0.5)
            if snap_share < 0.3:
                prob += 0.25
            elif snap_share < 0.5:
                prob += 0.1

            # Non-starter risk
            if not row.get("is_starter", True):
                prob += 0.15

            # Coaching change risk
            if row.get("new_coach", False):
                prob += 0.2

            # More eligibility = more likely
            years = row.get("years_remaining", 2)
            prob += years * 0.05

            # Already transferred = less likely
            if row.get("has_transferred", False):
                prob -= 0.1

            probs.loc[idx] = prob

        return probs.clip(0.05, 0.95)

    def get_risk_level(self, probability: float) -> str:
        """
        Get risk level for probability.

        Args:
            probability: Portal entry probability

        Returns:
            Risk level string
        """
        for level, threshold in self.RISK_LEVELS.items():
            if probability >= threshold:
                return level
        return "low"

    def predict_destinations(
        self,
        player_data: Dict[str, Any],
        schools: pd.DataFrame,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Predict likely transfer destinations.

        Args:
            player_data: Player information
            schools: DataFrame of potential schools
            top_n: Number of destinations to return

        Returns:
            List of destination predictions
        """
        predictions = []

        position = player_data.get("position", "")
        current_tier = player_data.get("school_tier", "p4_mid")

        for idx, school in schools.iterrows():
            score = 0.5  # Base score

            # Position need
            if position in school.get("position_needs", []):
                score += 0.2

            # Tier compatibility
            school_tier = school.get("tier", "p4_mid")
            if self._is_tier_compatible(current_tier, school_tier):
                score += 0.15

            # Geographic proximity
            if school.get("region") == player_data.get("home_region"):
                score += 0.1

            # Playing time opportunity
            if school.get("depth_chart_opening", False):
                score += 0.15

            predictions.append({
                "school": school.get("school_name", ""),
                "conference": school.get("conference", ""),
                "match_score": round(score, 2),
                "reason": self._get_match_reason(score, position, school),
            })

        # Sort by score and return top N
        predictions.sort(key=lambda x: x["match_score"], reverse=True)
        return predictions[:top_n]

    def _is_tier_compatible(self, player_tier: str, school_tier: str) -> bool:
        """Check if school tier is compatible for transfer."""
        tier_order = ["blue_blood", "elite", "power_brand", "p4_mid", "g5_strong", "g5"]

        try:
            player_idx = tier_order.index(player_tier)
            school_idx = tier_order.index(school_tier)
            # Allow lateral or one tier up/down
            return abs(player_idx - school_idx) <= 1
        except ValueError:
            return True

    def _get_match_reason(
        self,
        score: float,
        position: str,
        school: pd.Series,
    ) -> str:
        """Generate match reason text."""
        reasons = []

        if position in school.get("position_needs", []):
            reasons.append(f"Need at {position}")

        if school.get("depth_chart_opening", False):
            reasons.append("Starting opportunity")

        if score >= 0.7:
            reasons.append("Strong fit")

        return "; ".join(reasons) if reasons else "Potential fit"

    def get_at_risk_players(
        self,
        roster: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Get players at risk of entering portal.

        Args:
            roster: Team roster DataFrame
            threshold: Probability threshold

        Returns:
            DataFrame of at-risk players
        """
        probs = self.predict_probability(roster)
        roster = roster.copy()
        roster["portal_probability"] = probs
        roster["risk_level"] = probs.apply(self.get_risk_level)

        at_risk = roster[roster["portal_probability"] >= threshold]
        return at_risk.sort_values("portal_probability", ascending=False)

    def save(self, path: str) -> None:
        """Save model to file."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "features_used": getattr(self, "_features_used", self.PORTAL_FEATURES),
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
            self._features_used = model_data.get("features_used", self.PORTAL_FEATURES)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
