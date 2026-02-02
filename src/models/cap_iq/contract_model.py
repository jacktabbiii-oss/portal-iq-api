"""
NFL Contract Valuation Model

Predicts contract values for NFL players.
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


class ContractValuationModel:
    """Predicts contract value (AAV) for NFL players."""

    # Position value multipliers
    POSITION_VALUES = {
        "QB": 1.00,
        "EDGE": 0.85,
        "WR": 0.80,
        "CB": 0.75,
        "OT": 0.75,
        "DT": 0.65,
        "S": 0.60,
        "LB": 0.55,
        "TE": 0.55,
        "IOL": 0.50,
        "RB": 0.45,
    }

    # Features for contract prediction
    CONTRACT_FEATURES = [
        "production_score",
        "age_factor",
        "position_value",
        "accolades_score",
        "availability_rate",
        "years_experience",
    ]

    def __init__(
        self,
        salary_cap: int = 255_000_000,
        model_path: Optional[str] = None,
    ):
        """
        Initialize contract valuation model.

        Args:
            salary_cap: Current NFL salary cap
            model_path: Path to saved model
        """
        self.salary_cap = salary_cap
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
        Train contract valuation model.

        Args:
            X: Feature DataFrame
            y: Target AAV values

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training")

        available_features = [f for f in self.CONTRACT_FEATURES if f in X.columns]
        X_train = X[available_features].fillna(0)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True
        self._features_used = available_features

        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=5, scoring="r2"
        )

        logger.info(f"Model trained. CV R² = {cv_scores.mean():.3f}")

        return {
            "r2_mean": cv_scores.mean(),
            "r2_std": cv_scores.std(),
            "n_samples": len(y),
        }

    def predict_aav(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict AAV (Average Annual Value).

        Args:
            X: Feature DataFrame

        Returns:
            Predicted AAV values
        """
        if self.is_trained and SKLEARN_AVAILABLE:
            features = getattr(self, "_features_used", self.CONTRACT_FEATURES)
            available = [f for f in features if f in X.columns]
            X_pred = X[available].fillna(0)
            X_scaled = self.scaler.transform(X_pred)
            aav = self.model.predict(X_scaled)
            return pd.Series(aav, index=X.index).clip(self._min_salary())

        return self._rule_based_aav(X)

    def _rule_based_aav(self, X: pd.DataFrame) -> pd.Series:
        """
        Rule-based AAV prediction fallback.

        Args:
            X: Feature DataFrame

        Returns:
            Estimated AAV values
        """
        aav = pd.Series(5_000_000, index=X.index)  # Default

        for idx, row in X.iterrows():
            value = 5_000_000

            # Position base
            pos = row.get("position", "").upper()
            pos_mult = self.POSITION_VALUES.get(pos, 0.5)
            value = self.salary_cap * pos_mult * 0.05  # ~5% of cap for avg player

            # Production adjustment
            prod = row.get("production_score", 0.5)
            value *= (0.5 + prod)

            # Age adjustment
            age_factor = row.get("age_factor", 1.0)
            value *= age_factor

            # Accolades
            accolades = row.get("accolades_score", 0)
            value *= (1 + accolades * 0.5)

            aav.loc[idx] = value

        return aav.clip(self._min_salary(), self.salary_cap * 0.20)

    def predict_contract_structure(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict full contract structure.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with contract details
        """
        aav = self.predict_aav(X)

        results = X.copy()
        results["predicted_aav"] = aav

        # Estimate years based on age
        results["predicted_years"] = results.apply(
            lambda r: self._estimate_years(r.get("age", 27), r.get("position", "")),
            axis=1,
        )

        # Total value
        results["predicted_total"] = results["predicted_aav"] * results["predicted_years"]

        # Guaranteed estimate (~50-70% for top contracts)
        gtd_pct = 0.5 + (results.get("production_score", 0.5) * 0.2)
        results["predicted_guaranteed"] = results["predicted_total"] * gtd_pct

        return results

    def predict_range(
        self,
        X: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Predict AAV range.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (low_aav, high_aav)
        """
        base_aav = self.predict_aav(X)

        # 15-20% uncertainty range
        low = base_aav * 0.85
        high = base_aav * 1.20

        return low, high

    def compare_to_market(
        self,
        player_data: Dict[str, Any],
        comparables: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compare prediction to market comparables.

        Args:
            player_data: Target player data
            comparables: DataFrame of comparable contracts

        Returns:
            Comparison analysis
        """
        player_df = pd.DataFrame([player_data])
        predicted_aav = self.predict_aav(player_df).iloc[0]

        # Filter comparables by position
        pos = player_data.get("position", "").upper()
        pos_comps = comparables[
            comparables["position"].str.upper() == pos
        ]

        if pos_comps.empty:
            return {
                "predicted_aav": predicted_aav,
                "market_avg": None,
                "market_rank": None,
            }

        market_avg = pos_comps["aav"].mean()
        market_rank = (pos_comps["aav"] < predicted_aav).sum() + 1

        return {
            "predicted_aav": predicted_aav,
            "market_avg": market_avg,
            "market_max": pos_comps["aav"].max(),
            "market_rank": market_rank,
            "total_at_position": len(pos_comps),
            "pct_above_avg": ((predicted_aav - market_avg) / market_avg * 100),
        }

    def _estimate_years(self, age: int, position: str) -> int:
        """Estimate contract length based on age and position."""
        if age >= 32:
            return 2
        elif age >= 29:
            return 3
        elif age >= 26:
            return 4
        else:
            return 5

    def _min_salary(self) -> float:
        """Get minimum salary (rookie minimum)."""
        return 795_000

    def get_position_market(
        self,
        contracts: pd.DataFrame,
        position: str,
    ) -> Dict[str, Any]:
        """
        Get market overview for a position.

        Args:
            contracts: Contract DataFrame
            position: Position to analyze

        Returns:
            Market analysis dict
        """
        pos_contracts = contracts[
            contracts["position"].str.upper() == position.upper()
        ]

        if pos_contracts.empty:
            return {"position": position, "contracts": 0}

        return {
            "position": position,
            "contracts": len(pos_contracts),
            "avg_aav": pos_contracts["aav"].mean(),
            "max_aav": pos_contracts["aav"].max(),
            "min_aav": pos_contracts["aav"].min(),
            "median_aav": pos_contracts["aav"].median(),
            "cap_pct_avg": (
                pos_contracts["aav"].mean() / self.salary_cap * 100
            ),
        }

    def save(self, path: str) -> None:
        """Save model to file."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "features_used": getattr(self, "_features_used", self.CONTRACT_FEATURES),
            "salary_cap": self.salary_cap,
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
            self._features_used = model_data.get("features_used", self.CONTRACT_FEATURES)
            self.salary_cap = model_data.get("salary_cap", 255_000_000)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
