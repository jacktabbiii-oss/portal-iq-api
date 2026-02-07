"""
NIL Valuation Model

Predicts NIL market value for college football players.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import elite traits bonus calculator (top 10% measurables get boost)
try:
    from models.elite_traits import calculate_elite_bonus, get_athletic_profile
    ELITE_TRAITS_AVAILABLE = True
except ImportError:
    try:
        from .elite_traits import calculate_elite_bonus, get_athletic_profile
        ELITE_TRAITS_AVAILABLE = True
    except ImportError:
        ELITE_TRAITS_AVAILABLE = False

logger = logging.getLogger(__name__)


class NILValuationModel:
    """Predicts NIL value for college football players."""

    # Feature columns used for prediction
    FEATURE_COLS = [
        "production_score",
        "social_score",
        "school_multiplier",
        "position_multiplier",
        "nil_score",
        "conference_tier",
        "market_factor",
    ]

    # NIL value tiers
    TIERS = {
        "mega": 1_000_000,
        "premium": 500_000,
        "solid": 100_000,
        "moderate": 25_000,
        "entry": 0,
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize NIL valuation model.

        Args:
            model_path: Path to saved model weights
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not installed. Using rule-based fallback.")

        self.model = None
        self.scaler = None
        self.is_trained = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "gradient_boosting",
    ) -> Dict[str, float]:
        """
        Train the NIL valuation model.

        Args:
            X: Feature DataFrame
            y: Target NIL values

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training")

        # Prepare features
        X_train = X[self.FEATURE_COLS].fillna(0)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Initialize model
        if model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )
        else:
            self.model = Ridge(alpha=1.0)

        # Train
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=5, scoring="r2"
        )

        logger.info(f"Model trained. CV R² = {cv_scores.mean():.3f}")

        return {
            "r2_mean": cv_scores.mean(),
            "r2_std": cv_scores.std(),
            "n_samples": len(y),
        }

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict NIL values.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted NIL values
        """
        if self.is_trained and SKLEARN_AVAILABLE:
            X_pred = X[self.FEATURE_COLS].fillna(0)
            X_scaled = self.scaler.transform(X_pred)
            predictions = self.model.predict(X_scaled)
            return pd.Series(predictions, index=X.index).clip(0)

        # Fallback to rule-based prediction
        return self._rule_based_predict(X)

    def _rule_based_predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Rule-based NIL prediction fallback.

        Args:
            X: Feature DataFrame

        Returns:
            Estimated NIL values
        """
        values = pd.Series(0.0, index=X.index)

        for idx, row in X.iterrows():
            base_value = 50_000  # Base NIL value

            # Position adjustment
            pos_mult = row.get("position_multiplier", 1.0)
            base_value *= pos_mult

            # School tier adjustment
            school_mult = row.get("school_multiplier", 1.0)
            base_value *= school_mult

            # Production adjustment
            prod_score = row.get("production_score", 0.5)
            base_value *= (1 + prod_score)

            # Social media adjustment
            social = row.get("social_score", 0)
            base_value *= (1 + social * 0.5)

            # Market factor
            market = row.get("market_factor", 1.0)
            base_value *= market

            # Apply elite traits bonus (top 10% measurables only)
            # Average athletes get 1.0 (no change), elite get up to 1.25 (+25%)
            if ELITE_TRAITS_AVAILABLE:
                position = row.get("position", "")
                elite_mult = calculate_elite_bonus(row.to_dict(), position)
                base_value *= elite_mult

            values.loc[idx] = base_value

        return values.clip(5_000, 5_000_000)

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Predict NIL values with confidence intervals.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (predictions, confidence_ranges)
        """
        predictions = self.predict(X)

        # Estimate confidence based on data completeness
        completeness = X[self.FEATURE_COLS].notna().mean(axis=1)
        confidence = completeness * 0.8 + 0.2  # 20-100% confidence

        # Create ranges
        ranges = predictions * (1 - confidence) * 0.5

        return predictions, ranges

    def get_tier(self, value: float) -> str:
        """
        Get NIL tier for a value.

        Args:
            value: Predicted NIL value

        Returns:
            Tier name
        """
        for tier, threshold in self.TIERS.items():
            if value >= threshold:
                return tier
        return "entry"

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model.

        Returns:
            Dict of feature -> importance
        """
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return {f: 1.0 / len(self.FEATURE_COLS) for f in self.FEATURE_COLS}

        return dict(zip(self.FEATURE_COLS, self.model.feature_importances_))

    def save(self, path: str) -> None:
        """Save model to file."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from file."""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.is_trained = model_data["is_trained"]
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")


def valuate_player(
    player_data: Dict[str, Any],
    model: Optional[NILValuationModel] = None,
) -> Dict[str, Any]:
    """
    Valuate a single player's NIL.

    Args:
        player_data: Player information dict
        model: Optional trained model

    Returns:
        Valuation result with elite traits info
    """
    if model is None:
        model = NILValuationModel()

    df = pd.DataFrame([player_data])
    value = model.predict(df).iloc[0]
    tier = model.get_tier(value)

    result = {
        "player_name": player_data.get("player_name", "Unknown"),
        "position": player_data.get("position", ""),
        "school": player_data.get("school", ""),
        "estimated_nil_value": round(value, -3),
        "nil_tier": tier,
        "confidence": "medium",
    }

    # Add elite athlete profile if available
    if ELITE_TRAITS_AVAILABLE:
        position = player_data.get("position", "")
        profile = get_athletic_profile(player_data, position)
        result["athletic_profile"] = {
            "tier": profile["tier"],
            "tier_label": profile["tier_label"],
            "elite_bonus": profile["elite_bonus"],
            "elite_traits": profile["elite_traits"],
        }

    return result
