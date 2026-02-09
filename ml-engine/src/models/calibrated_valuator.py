"""
Calibrated NIL Valuator - ML ensemble trained on real On3 market data.

Trains on ~330 real On3 NIL valuations to learn relative market rankings,
then applies a rank-based distribution correction to produce realistic values
for all 21K+ FBS players. Uses Ridge + RandomForest + LightGBM ensemble
with log-transformed target.

Key insight: The On3 training data only covers the top ~2% of FBS players
($148K-$5.4M). The model excels at RANKING players (Spearman ~0.65) but
can't extrapolate absolute values below $148K. We solve this by:
1. Training the model on On3 data to learn relative player rankings
2. Using the model's ranking to assign values from a realistic FBS distribution
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from .school_tiers import get_school_multiplier, get_school_tier

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Position market rank (1 = highest NIL market value)
POSITION_MARKET_RANK = {
    "QB": 1, "WR": 2, "EDGE": 3, "RB": 4, "OT": 5,
    "CB": 6, "S": 7, "LB": 8, "TE": 9, "DL": 10,
    "OL": 11, "IOL": 11, "OG": 11, "C": 11, "G": 11,
    "DE": 5, "DT": 10, "NT": 10, "ILB": 8, "OLB": 7,
    "DB": 6, "FB": 9, "ATH": 5, "CG": 11,
    "K": 12, "P": 13, "LS": 14,
}

# Position base floor (minimum reasonable value at a blue-blood school)
POSITION_FLOOR = {
    "QB": 8000, "WR": 4000, "EDGE": 4000, "RB": 3000, "OT": 3000,
    "CB": 3000, "S": 2500, "LB": 2500, "TE": 2500, "DL": 2000,
    "OL": 2000, "K": 1000, "P": 800, "LS": 500,
}

# Non-football positions to filter out of On3 data
BASKETBALL_POSITIONS = {"PG", "SG", "SF", "PF", "CG"}

# NIL tier thresholds (calibrated to On3 distribution)
TIER_THRESHOLDS = {
    "mega": 2_000_000,
    "premium": 500_000,
    "solid": 100_000,
    "moderate": 25_000,
    "entry": 0,
}

# FBS mascot suffixes to strip when normalizing school names
MASCOT_SUFFIXES = [
    "Longhorns", "Bulldogs", "Tigers", "Crimson Tide", "Buckeyes",
    "Wolverines", "Fighting Irish", "Sooners", "Gators", "Volunteers",
    "Seminoles", "Hurricanes", "Trojans", "Nittany Lions", "Wildcats",
    "Ducks", "Bears", "Aggies", "Jayhawks", "Cyclones", "Bruins",
    "Huskies", "Sun Devils", "Beavers", "Cougars", "Razorbacks",
    "Commodores", "Rebels", "War Eagles", "Mountaineers", "Cavaliers",
    "Tar Heels", "Yellow Jackets", "Cardinals", "Hokies", "Wolfpack",
    "Panthers", "Blue Devils", "Demon Deacons", "Owls", "Mustangs",
    "Falcons", "Golden Eagles", "Bobcats", "Rockets", "Redhawks",
    "Thundering Herd", "Miners", "Mean Green", "Roadrunners",
    "Jaguars", "Hilltoppers", "Red Wolves", "Chanticleers",
    "Broncos", "Aztecs", "Rainbows", "Warriors", "Lobos",
    "Rams", "Cowboys", "Buffaloes", "Utes", "Hawkeyes", "Badgers",
    "Golden Gophers", "Hoosiers", "Boilermakers", "Illini", "Cornhuskers",
    "Spartans", "Terrapins", "Scarlet Knights",
]

# Known school name mappings (On3 → canonical)
SCHOOL_NAME_MAP = {
    "miami hurricanes": "miami",
    "miami (fl) hurricanes": "miami",
    "miami (oh) redhawks": "miami (oh)",
    "usc trojans": "usc",
    "ole miss rebels": "ole miss",
    "lsu tigers": "lsu",
    "ucf knights": "ucf",
    "smu mustangs": "smu",
    "byu cougars": "byu",
    "tcu horned frogs": "tcu",
}

# Distribution parameters for rank-based value mapping
# value(rank) = C / (rank + 1)^b  (power law)
# Tuned so: rank 400 ≈ $80K, rank 2000 ≈ $20K, rank 10000 ≈ $5K, rank 21000 ≈ $2.5K
DIST_POWER = 0.86   # Power law exponent (gentler decay for realistic spread)
DIST_SCALE = 17_500_000  # Scale factor (C)
DIST_TOP_N = 350   # Number of top players to trust model predictions for


@dataclass
class CalibratedValuation:
    """Result of a calibrated NIL valuation."""
    nil_value: float
    nil_tier: str
    confidence: str
    position: str
    school: str
    features_used: Dict[str, float] = field(default_factory=dict)


class CalibratedNILValuator:
    """
    ML-calibrated NIL valuator trained on real On3 market data.

    Uses an ensemble of Ridge + RandomForest + LightGBM trained on
    log-transformed On3 valuations. For the top ~400 players (within
    training distribution), the model's absolute predictions are used.
    For all other players, the model's ranking is preserved but values
    are mapped to a realistic FBS distribution using a power law.
    """

    FEATURE_COLS = [
        "position_rank", "school_score", "school_multiplier",
        "stars", "national_rank_inv", "recruiting_rating",
        "log_followers", "class_year_num", "pff_overall",
        "has_pff", "has_followers", "has_stats",
    ]

    def __init__(self):
        self.models: Dict[str, object] = {}
        self.ensemble_weights: List[float] = []
        self.scaler: Optional[StandardScaler] = None
        self.follower_model: Optional[Ridge] = None
        self.is_trained: bool = False
        self.cv_metrics: Dict[str, float] = {}
        self.training_min_log: float = 0.0
        self.training_stats: Dict = {}

    # =========================================================================
    # School Name Normalization
    # =========================================================================

    def _normalize_school(self, school: str) -> str:
        """Normalize school name by stripping mascot suffixes."""
        if pd.isna(school):
            return "unknown"

        name = str(school).strip()
        name_lower = name.lower()

        # Check explicit mappings first
        if name_lower in SCHOOL_NAME_MAP:
            return SCHOOL_NAME_MAP[name_lower]

        # Strip mascot suffixes
        for suffix in MASCOT_SUFFIXES:
            if name_lower.endswith(suffix.lower()):
                stripped = name[:len(name) - len(suffix)].strip()
                if stripped:
                    return stripped

        return name

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    def _engineer_features(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        """Build feature matrix from raw player data."""
        features = pd.DataFrame(index=df.index)

        # Position rank (ordinal by market premium)
        pos_col = df.get("position", pd.Series("ATH", index=df.index))
        features["position_rank"] = pos_col.map(
            lambda p: POSITION_MARKET_RANK.get(str(p).upper().strip(), 8)
        )

        # School features via school_tiers.py
        school_col = df.get("school", pd.Series("Unknown", index=df.index))
        normalized_schools = school_col.map(self._normalize_school)

        scores = []
        mults = []
        for school in normalized_schools:
            try:
                tier_name, tier_info = get_school_tier(school)
                scores.append(tier_info.get("score", 30))
                mults.append(tier_info.get("multiplier", 0.8))
            except Exception:
                scores.append(30)
                mults.append(0.8)
        features["school_score"] = scores
        features["school_multiplier"] = mults

        # Recruiting stars
        stars_col = df.get("recruiting_stars", df.get("stars", pd.Series(dtype=float)))
        features["stars"] = stars_col.fillna(2).clip(0, 5).astype(float)

        # National rank (inverse log — closer to #1 = higher value)
        rank_col = df.get("national_rank", pd.Series(dtype=float))
        features["national_rank_inv"] = rank_col.apply(
            lambda x: 1.0 / np.log1p(x) if pd.notna(x) and x > 0 else 0.1
        )

        # Recruiting rating
        rating_col = df.get("recruiting_rating", pd.Series(dtype=float))
        features["recruiting_rating"] = rating_col.fillna(70).clip(0, 100).astype(float)

        # Social media followers (log-transformed)
        followers_col = df.get("followers", pd.Series(dtype=float))
        has_followers = followers_col.notna() & (followers_col > 0)
        features["has_followers"] = has_followers.astype(float)

        if training:
            features["log_followers"] = np.log1p(followers_col.fillna(0).clip(0))
        else:
            features["log_followers"] = np.log1p(followers_col.fillna(0).clip(0))
            if self.follower_model is not None:
                missing_mask = ~has_followers
                if missing_mask.any():
                    impute_feats = features.loc[missing_mask, [
                        "position_rank", "school_score", "stars", "class_year_num"
                    ]].fillna(0)
                    if len(impute_feats) > 0:
                        predicted_log_followers = self.follower_model.predict(impute_feats)
                        features.loc[missing_mask, "log_followers"] = predicted_log_followers

        # Class year
        class_col = df.get("class_year", pd.Series("", index=df.index)).fillna("").str.lower()
        year_map = {"freshman": 1, "sophomore": 2, "junior": 3, "senior": 4, "redshirt": 1.5}
        features["class_year_num"] = class_col.map(
            lambda y: next((v for k, v in year_map.items() if k in str(y)), 2.0)
        )

        # PFF overall grade
        pff_col = df.get("pff_overall", pd.Series(dtype=float))
        features["has_pff"] = (pff_col.notna() & (pff_col > 0)).astype(float)
        features["pff_overall"] = pff_col.fillna(60).clip(30, 99).astype(float)

        # Has stats indicator
        stat_cols = ["passing_yards", "rushing_yards", "receiving_yards", "tackles", "sacks"]
        has_any_stat = pd.Series(False, index=df.index)
        for col in stat_cols:
            if col in df.columns:
                has_any_stat = has_any_stat | (df[col].notna() & (df[col] > 0))
        features["has_stats"] = has_any_stat.astype(float)

        return features[self.FEATURE_COLS]

    # =========================================================================
    # Follower Imputation
    # =========================================================================

    def _train_follower_model(self, on3_df: pd.DataFrame, features: pd.DataFrame):
        """Train a small model to impute social followers for players without data."""
        followers = on3_df.get("followers", pd.Series(dtype=float))
        valid_mask = followers.notna() & (followers > 0)

        if valid_mask.sum() < 50:
            logger.warning("Not enough follower data for imputation model")
            return

        X = features.loc[valid_mask, ["position_rank", "school_score", "stars", "class_year_num"]].fillna(0)
        y = np.log1p(followers[valid_mask].values)

        self.follower_model = Ridge(alpha=10.0)
        self.follower_model.fit(X, y)
        logger.info(f"Trained follower imputation model on {len(X)} samples")

    # =========================================================================
    # Training
    # =========================================================================

    def train(self, on3_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the calibrated model on real On3 data.

        The model learns relative rankings from On3 data. Post-prediction,
        a distribution correction maps rankings to realistic FBS values.

        Args:
            on3_df: DataFrame with On3 NIL ranking data

        Returns:
            Dict of cross-validation metrics
        """
        # Filter to football only
        football_mask = ~on3_df["position"].isin(BASKETBALL_POSITIONS)
        df = on3_df[football_mask].copy()

        # Remove zero/null valuations
        df = df[df["nil_valuation"].notna() & (df["nil_valuation"] > 0)].copy()
        logger.info(f"Training on {len(df)} football players with real On3 values")

        # Normalize school names for tier lookup
        df["school"] = df["school"].map(self._normalize_school)

        # Target: log-transformed NIL value
        y = np.log1p(df["nil_valuation"].values)
        self.training_min_log = float(y.min())

        # Store training stats for diagnostics
        self.training_stats = {
            "n_samples": len(df),
            "value_min": float(df["nil_valuation"].min()),
            "value_max": float(df["nil_valuation"].max()),
            "value_median": float(df["nil_valuation"].median()),
            "value_mean": float(df["nil_valuation"].mean()),
        }

        # Engineer features
        X_df = self._engineer_features(df, training=True)

        # Train follower imputation model BEFORE scaling
        self._train_follower_model(df, X_df)

        X = X_df.fillna(0).values.astype(float)

        # Scale features (replace NaN from zero-variance columns with 0)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # --- Cross-Validation ---
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_preds = np.zeros(len(y))

        model_cv_scores = {"ridge": [], "rf": [], "lgbm": []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tr, y_tr)
            ridge_pred = ridge.predict(X_val)
            model_cv_scores["ridge"].append(_mape(y_val, ridge_pred))

            # Random Forest
            rf = RandomForestRegressor(
                n_estimators=200, max_depth=5, min_samples_leaf=10,
                random_state=42, n_jobs=-1
            )
            rf.fit(X_tr, y_tr)
            rf_pred = rf.predict(X_val)
            model_cv_scores["rf"].append(_mape(y_val, rf_pred))

            # LightGBM
            if HAS_LGBM:
                lgbm = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    reg_alpha=1.0, reg_lambda=1.0, min_child_samples=20,
                    random_state=42, verbose=-1,
                )
                lgbm.fit(X_tr, y_tr)
                lgbm_pred = lgbm.predict(X_val)
                model_cv_scores["lgbm"].append(_mape(y_val, lgbm_pred))
            else:
                lgbm_pred = ridge_pred
                model_cv_scores["lgbm"].append(model_cv_scores["ridge"][-1])

            cv_preds[val_idx] = 0.3 * ridge_pred + 0.3 * rf_pred + 0.4 * lgbm_pred

        # --- Optimize ensemble weights ---
        best_mape = float("inf")
        best_weights = [0.3, 0.3, 0.4]

        # Retrain all models on full data
        ridge_full = Ridge(alpha=1.0).fit(X_scaled, y)
        rf_full = RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        ).fit(X_scaled, y)

        if HAS_LGBM:
            lgbm_full = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                reg_alpha=1.0, reg_lambda=1.0, min_child_samples=20,
                random_state=42, verbose=-1,
            ).fit(X_scaled, y)
        else:
            lgbm_full = ridge_full

        # Grid search over weights
        for w1 in np.arange(0.1, 0.6, 0.1):
            for w2 in np.arange(0.1, 0.6, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < 0.05:
                    continue
                full_preds = (
                    w1 * ridge_full.predict(X_scaled)
                    + w2 * rf_full.predict(X_scaled)
                    + w3 * lgbm_full.predict(X_scaled)
                )
                mape = _mape(y, full_preds)
                if mape < best_mape:
                    best_mape = mape
                    best_weights = [float(w1), float(w2), float(w3)]

        self.ensemble_weights = best_weights
        self.models = {"ridge": ridge_full, "rf": rf_full, "lgbm": lgbm_full}

        # --- Compute final CV metrics ---
        y_actual_dollars = np.expm1(y)
        cv_pred_dollars = np.expm1(cv_preds)

        mape = float(np.median(np.abs(y_actual_dollars - cv_pred_dollars) / np.maximum(y_actual_dollars, 1)))
        mae = float(mean_absolute_error(y_actual_dollars, cv_pred_dollars))
        median_ae = float(np.median(np.abs(y_actual_dollars - cv_pred_dollars)))
        r2 = float(r2_score(y, cv_preds))
        spearman, _ = spearmanr(y_actual_dollars, cv_pred_dollars)

        self.cv_metrics = {
            "mape": mape,
            "mae": mae,
            "median_ae": median_ae,
            "r2": r2,
            "spearman": float(spearman),
            "n_samples": len(y),
            "model_mapes": {k: float(np.mean(v)) for k, v in model_cv_scores.items()},
            "ensemble_weights": best_weights,
        }

        self.is_trained = True
        logger.info(f"Calibrated model trained: MAPE={mape:.1%}, R²={r2:.3f}, Spearman={spearman:.3f}")
        return self.cv_metrics

    # =========================================================================
    # Prediction
    # =========================================================================

    def predict(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict calibrated NIL values for a DataFrame of players.

        Uses the model to rank all players, then applies a distribution
        correction so values follow a realistic FBS power-law distribution.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        result = players_df.copy()

        # Normalize school names
        school_col = result.get("school", result.get("team", pd.Series("Unknown", index=result.index)))
        result["school"] = school_col.map(self._normalize_school)

        # Engineer features
        X_df = self._engineer_features(result, training=False)
        X = X_df.fillna(0).values.astype(float)
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Ensemble prediction (used for ranking)
        w = self.ensemble_weights
        preds_log = (
            w[0] * self.models["ridge"].predict(X_scaled)
            + w[1] * self.models["rf"].predict(X_scaled)
            + w[2] * self.models["lgbm"].predict(X_scaled)
        )

        # Apply rank-based distribution correction
        preds_dollars = self._apply_distribution_correction(preds_log, X_df, result)

        # Apply position/school floors
        preds_dollars = self._apply_floors(preds_dollars, X_df, result)

        # Ensure non-negative and round
        preds_dollars = np.maximum(preds_dollars, 100)
        preds_dollars = np.round(preds_dollars / 100) * 100

        # Assign tiers and confidence
        result["nil_value"] = preds_dollars
        result["nil_tier"] = [self._assign_tier(v) for v in preds_dollars]
        result["confidence"] = [
            self._assess_confidence(row, X_df.iloc[i])
            for i, (_, row) in enumerate(result.iterrows())
        ]
        result["is_predicted"] = True

        return result

    def predict_single(self, **kwargs) -> CalibratedValuation:
        """Predict for a single player (useful for API endpoint)."""
        df = pd.DataFrame([kwargs])
        result = self.predict(df)
        row = result.iloc[0]
        return CalibratedValuation(
            nil_value=float(row["nil_value"]),
            nil_tier=str(row["nil_tier"]),
            confidence=str(row["confidence"]),
            position=str(kwargs.get("position", "")),
            school=str(kwargs.get("school", "")),
        )

    # =========================================================================
    # Distribution Correction
    # =========================================================================

    def _apply_distribution_correction(
        self, preds_log: np.ndarray, features: pd.DataFrame, players: pd.DataFrame
    ) -> np.ndarray:
        """
        Map model predictions to a realistic FBS value distribution.

        The model is trained on top ~330 players ($148K-$5.4M) and ranks
        players well (Spearman ~0.65), but can't produce values below ~$148K.

        Strategy:
        - Top DIST_TOP_N players by model score: use model prediction directly
          (these are within training distribution)
        - Players below top N: use model ranking to assign values from a
          power-law distribution: value = C / (rank+1)^b
        - Smooth blend in the transition zone around rank N
        """
        n = len(preds_log)
        preds_dollars = np.expm1(preds_log)

        # Rank by model prediction (descending: rank 0 = highest predicted)
        rank_order = np.argsort(-preds_log)
        ranks = np.empty(n, dtype=int)
        ranks[rank_order] = np.arange(n)

        result = preds_dollars.copy()

        # Power-law target distribution: value(rank) = C / (rank+1)^b
        top_n = min(DIST_TOP_N, n)
        blend_zone = min(200, top_n // 2)  # Smooth transition zone

        for i in range(n):
            rank = ranks[i]

            # Target value from power-law distribution
            target = DIST_SCALE / (rank + 1) ** DIST_POWER

            if rank < top_n - blend_zone:
                # Fully within training distribution: trust model
                result[i] = preds_dollars[i]
            elif rank < top_n:
                # Blend zone: smooth transition from model to distribution
                blend = (rank - (top_n - blend_zone)) / blend_zone  # 0 to 1
                result[i] = (1 - blend) * preds_dollars[i] + blend * target
            else:
                # Below training distribution: use power-law distribution
                # Modulate by feature quality (better features = higher within band)
                feature_mult = self._compute_feature_quality(features.iloc[i])
                result[i] = target * feature_mult

        return result

    def _compute_feature_quality(self, feature_row: pd.Series) -> float:
        """
        Compute a quality multiplier based on player features.

        Returns a value centered around 1.0 that adjusts the rank-based
        distribution value up or down based on player characteristics.
        """
        mult = 1.0

        # Stars adjustment (default 2 = neutral)
        stars = feature_row.get("stars", 2)
        if stars >= 4:
            mult *= 1.4
        elif stars >= 3:
            mult *= 1.15
        elif stars <= 1:
            mult *= 0.7

        # School quality
        school_mult = feature_row.get("school_multiplier", 0.8)
        if school_mult >= 2.5:
            mult *= 1.3
        elif school_mult >= 1.5:
            mult *= 1.15
        elif school_mult <= 0.6:
            mult *= 0.8

        # PFF performance
        if feature_row.get("has_pff", 0) > 0:
            pff = feature_row.get("pff_overall", 60)
            if pff >= 80:
                mult *= 1.3
            elif pff >= 70:
                mult *= 1.15
            elif pff < 55:
                mult *= 0.85

        # Has real social following
        if feature_row.get("has_followers", 0) > 0:
            log_followers = feature_row.get("log_followers", 0)
            if log_followers >= 12:  # ~160K+ followers
                mult *= 1.5
            elif log_followers >= 10:  # ~22K+ followers
                mult *= 1.2

        return mult

    # =========================================================================
    # Floors & Confidence
    # =========================================================================

    def _apply_floors(
        self, preds: np.ndarray, features: pd.DataFrame, players: pd.DataFrame
    ) -> np.ndarray:
        """Apply position/school-based minimum values."""
        result = preds.copy()

        for i in range(len(result)):
            pos = str(players.iloc[i].get("position", "ATH")).upper()
            pos_group = pos if pos in POSITION_FLOOR else "OL"
            floor = POSITION_FLOOR.get(pos_group, 1000)

            school_mult = features.iloc[i].get("school_multiplier", 0.8)
            adjusted_floor = floor * school_mult * 0.3  # 30% of max floor

            stars = features.iloc[i].get("stars", 2)
            if stars >= 5:
                adjusted_floor = max(adjusted_floor, 50000)
            elif stars >= 4:
                adjusted_floor = max(adjusted_floor, 15000)
            elif stars >= 3:
                adjusted_floor = max(adjusted_floor, 3000)

            result[i] = max(result[i], adjusted_floor)

        return result

    def _assess_confidence(self, player_row: pd.Series, feature_row: pd.Series) -> str:
        """Assess prediction confidence based on available data."""
        score = 0

        if feature_row.get("has_pff", 0) > 0:
            score += 2
        if feature_row.get("has_followers", 0) > 0:
            score += 2
        if feature_row.get("has_stats", 0) > 0:
            score += 1
        if feature_row.get("stars", 2) != 2:
            score += 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        return "low"

    def _assign_tier(self, value: float) -> str:
        """Assign NIL tier based on calibrated thresholds."""
        if value >= TIER_THRESHOLDS["mega"]:
            return "mega"
        elif value >= TIER_THRESHOLDS["premium"]:
            return "premium"
        elif value >= TIER_THRESHOLDS["solid"]:
            return "solid"
        elif value >= TIER_THRESHOLDS["moderate"]:
            return "moderate"
        return "entry"

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_calibration_report(self) -> str:
        """Generate a human-readable calibration report."""
        if not self.is_trained:
            return "Model not yet trained."

        lines = [
            "=" * 60,
            "NIL CALIBRATION REPORT",
            "=" * 60,
            "",
            f"Training samples: {self.cv_metrics['n_samples']} real On3 valuations",
            f"Value range: ${self.training_stats['value_min']:,.0f} - ${self.training_stats['value_max']:,.0f}",
            f"Median value: ${self.training_stats['value_median']:,.0f}",
            "",
            "--- Cross-Validation Metrics (5-fold, on On3 data) ---",
            f"  MAPE (Median):     {self.cv_metrics['mape']:.1%}",
            f"  MAE:               ${self.cv_metrics['mae']:,.0f}",
            f"  Median AE:         ${self.cv_metrics['median_ae']:,.0f}",
            f"  R²:                {self.cv_metrics['r2']:.3f}",
            f"  Spearman corr:     {self.cv_metrics['spearman']:.3f}",
            "",
            "--- Individual Model MAPE ---",
        ]
        for name, mape in self.cv_metrics.get("model_mapes", {}).items():
            lines.append(f"  {name:10}: {mape:.1%}")

        w = self.cv_metrics.get("ensemble_weights", [])
        if w:
            lines.append(f"\n  Ensemble weights: Ridge={w[0]:.1f}, RF={w[1]:.1f}, LGBM={w[2]:.1f}")

        lines.append("")
        lines.append(f"Distribution correction: power law (b={DIST_POWER}, top_n={DIST_TOP_N})")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Helper Functions
# =============================================================================

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Median Absolute Percentage Error in log space."""
    actual = np.expm1(y_true)
    predicted = np.expm1(y_pred)
    return float(np.median(np.abs(actual - predicted) / np.maximum(actual, 1)))
