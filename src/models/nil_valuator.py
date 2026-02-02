"""
NIL Valuation Model

Predicts NIL (Name, Image, Likeness) values for college athletes using
ensemble machine learning models with comprehensive prediction outputs.

Author: Elite Sports Solutions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
import logging
import warnings
import os

# ML imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    KFold, LeaveOneOut
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)
from sklearn.neighbors import NearestNeighbors
import joblib

# Optional imports with fallbacks
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not available. Install with: pip install shap")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Feature groups for SHAP analysis
FEATURE_GROUPS = {
    'on_field_performance': [
        'production_score', 'pass_yards_per_game', 'pass_tds_per_game',
        'rush_yards_per_game', 'rec_yards_per_game', 'tackles_per_game',
        'completion_pct', 'yards_per_attempt', 'yards_per_carry',
        'receptions_per_game', 'rec_tds_per_game', 'sacks_per_game',
    ],
    'social_media': [
        'total_social_following', 'log_total_following',
        'instagram_followers_log', 'tiktok_followers_log',
        'twitter_followers_log', 'youtube_followers_log',
        'estimated_social_value', 'has_significant_following',
        'social_platform_concentration',
    ],
    'school_brand': [
        'school_tier', 'conference_tier', 'team_wins', 'team_win_pct',
        'cfp_appearance', 'conference_championship',
        'school_market_size', 'football_state',
    ],
    'recruiting_pedigree': [
        'recruiting_stars', 'recruiting_composite',
        'recruiting_position_rank', 'recruiting_class_rank',
    ],
    'draft_projection': [
        'projected_draft_flag', 'projected_draft_round', 'draft_score',
    ],
    'player_profile': [
        'years_in_college', 'year_classification', 'remaining_eligibility',
        'is_starter', 'games_started_pct', 'age', 'qb_premium',
    ],
}

# School data for transfer analysis
SCHOOL_DATA = {
    'Alabama': {'tier': 6, 'conference_tier': 3, 'market_size': 260, 'wins': 11, 'football_state': 1},
    'Ohio State': {'tier': 6, 'conference_tier': 3, 'market_size': 2150, 'wins': 12, 'football_state': 1},
    'Georgia': {'tier': 5, 'conference_tier': 3, 'market_size': 450, 'wins': 13, 'football_state': 1},
    'Texas': {'tier': 6, 'conference_tier': 3, 'market_size': 2300, 'wins': 12, 'football_state': 1},
    'USC': {'tier': 6, 'conference_tier': 3, 'market_size': 13000, 'wins': 8, 'football_state': 1},
    'Michigan': {'tier': 6, 'conference_tier': 3, 'market_size': 4400, 'wins': 13, 'football_state': 1},
    'Oregon': {'tier': 5, 'conference_tier': 3, 'market_size': 380, 'wins': 12, 'football_state': 0},
    'Penn State': {'tier': 5, 'conference_tier': 3, 'market_size': 160, 'wins': 10, 'football_state': 1},
    'LSU': {'tier': 5, 'conference_tier': 3, 'market_size': 870, 'wins': 10, 'football_state': 1},
    'Florida': {'tier': 5, 'conference_tier': 3, 'market_size': 330, 'wins': 7, 'football_state': 1},
    'Tennessee': {'tier': 5, 'conference_tier': 3, 'market_size': 900, 'wins': 9, 'football_state': 1},
    'Clemson': {'tier': 5, 'conference_tier': 2, 'market_size': 920, 'wins': 9, 'football_state': 0},
    'Notre Dame': {'tier': 6, 'conference_tier': 2, 'market_size': 320, 'wins': 10, 'football_state': 0},
    'Oklahoma': {'tier': 6, 'conference_tier': 3, 'market_size': 1450, 'wins': 10, 'football_state': 1},
    'Texas A&M': {'tier': 5, 'conference_tier': 3, 'market_size': 275, 'wins': 8, 'football_state': 1},
    'Miami': {'tier': 5, 'conference_tier': 2, 'market_size': 6200, 'wins': 7, 'football_state': 1},
    'Florida State': {'tier': 5, 'conference_tier': 2, 'market_size': 390, 'wins': 13, 'football_state': 1},
    'Wisconsin': {'tier': 4, 'conference_tier': 3, 'market_size': 680, 'wins': 7, 'football_state': 0},
    'Iowa': {'tier': 4, 'conference_tier': 3, 'market_size': 175, 'wins': 8, 'football_state': 0},
    'UCLA': {'tier': 4, 'conference_tier': 3, 'market_size': 13000, 'wins': 8, 'football_state': 1},
    'Colorado': {'tier': 4, 'conference_tier': 2, 'market_size': 2900, 'wins': 9, 'football_state': 0},
    'Ole Miss': {'tier': 4, 'conference_tier': 3, 'market_size': 175, 'wins': 11, 'football_state': 0},
    'Missouri': {'tier': 4, 'conference_tier': 3, 'market_size': 180, 'wins': 11, 'football_state': 0},
    'UCF': {'tier': 3, 'conference_tier': 2, 'market_size': 2700, 'wins': 6, 'football_state': 1},
    'Cincinnati': {'tier': 3, 'conference_tier': 2, 'market_size': 2250, 'wins': 5, 'football_state': 1},
    'Boise State': {'tier': 2, 'conference_tier': 1, 'market_size': 780, 'wins': 8, 'football_state': 0},
    'Memphis': {'tier': 2, 'conference_tier': 1, 'market_size': 1350, 'wins': 10, 'football_state': 1},
}

# NIL tier thresholds
NIL_TIERS = {
    'mega': 1_000_000,
    'premium': 500_000,
    'solid': 100_000,
    'moderate': 25_000,
    'entry': 0,
}

TIER_ORDER = ['entry', 'moderate', 'solid', 'premium', 'mega']


class NILValuator:
    """
    NIL valuation model for predicting college athlete NIL values.

    Features:
    - Regression model for dollar value prediction
    - Classification model for tier prediction
    - Two-stage model combining both
    - Transfer impact analysis
    - Social media what-if scenarios
    - Position market reports
    """

    def __init__(
        self,
        model_dir: str = "models/nil_valuation",
        output_dir: str = "outputs/reports"
    ):
        """
        Initialize the NIL valuator.

        Args:
            model_dir: Directory to save trained models
            output_dir: Directory to save reports
        """
        self.model_dir = model_dir
        self.output_dir = output_dir

        # Models
        self.value_model = None
        self.tier_model = None
        self.tier_value_models = {}  # Two-stage: one regressor per tier
        self.scaler = None
        self.tier_encoder = None

        # Training data reference
        self.feature_names = []
        self.training_data = None
        self.training_stats = {}

        # SHAP explainer
        self.shap_explainer = None
        self.shap_values = None

        # Performance metrics
        self.metrics = {}

        # Configuration
        self.use_two_stage = False
        self.small_data_mode = False
        self.min_recommended_samples = 100

    def train(
        self,
        features_df: pd.DataFrame,
        target_col: str = 'nil_value',
        tier_col: str = 'nil_tier'
    ) -> Dict[str, Any]:
        """
        Train NIL valuation models.

        Args:
            features_df: DataFrame with features and targets
            target_col: Column name for NIL value (regression target)
            tier_col: Column name for NIL tier (classification target)

        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting NIL model training...")

        # Validate inputs
        df = features_df.copy()
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Handle small data
        n_samples = len(df)
        if n_samples < 50:
            self.small_data_mode = True
            logger.warning(f"Small dataset ({n_samples} samples). Using simplified models.")
            logger.warning(f"Recommended minimum: {self.min_recommended_samples} samples for reliable predictions.")

        # Prepare features
        X, y_value, y_tier, feature_names = self._prepare_data(df, target_col, tier_col)
        self.feature_names = feature_names
        self.training_data = df.copy()

        # Store training statistics
        self.training_stats = {
            'n_samples': n_samples,
            'feature_means': X.mean(axis=0),
            'feature_stds': X.std(axis=0),
            'value_mean': np.mean(y_value),
            'value_std': np.std(y_value),
        }

        # Split data (stratified by tier)
        X_train, X_test, y_val_train, y_val_test, y_tier_train, y_tier_test = \
            self._split_data(X, y_value, y_tier)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train models
        results = {}

        # 1. Regression model (value prediction)
        logger.info("Training regression models...")
        reg_results = self._train_regression_models(
            X_train_scaled, y_val_train, X_test_scaled, y_val_test
        )
        results['regression'] = reg_results

        # 2. Classification model (tier prediction)
        logger.info("Training classification models...")
        clf_results = self._train_classification_models(
            X_train_scaled, y_tier_train, X_test_scaled, y_tier_test
        )
        results['classification'] = clf_results

        # 3. Two-stage model
        logger.info("Training two-stage model...")
        two_stage_results = self._train_two_stage_model(
            X_train_scaled, y_val_train, y_tier_train,
            X_test_scaled, y_val_test, y_tier_test
        )
        results['two_stage'] = two_stage_results

        # Compare and select best approach
        single_mae = reg_results['best_metrics']['test_mae']
        two_stage_mae = two_stage_results['test_mae']

        if two_stage_mae < single_mae * 0.95:  # 5% improvement threshold
            self.use_two_stage = True
            logger.info(f"Two-stage model selected (MAE: ${two_stage_mae:,.0f} vs ${single_mae:,.0f})")
        else:
            self.use_two_stage = False
            logger.info(f"Single regression model selected (MAE: ${single_mae:,.0f})")

        results['selected_approach'] = 'two_stage' if self.use_two_stage else 'single_regression'

        # Calculate SHAP values
        if HAS_SHAP and not self.small_data_mode:
            logger.info("Calculating SHAP values...")
            self._calculate_shap_values(X_train_scaled, feature_names)

        # Save models
        self._save_models()

        # Store metrics
        self.metrics = results

        # Generate and save report
        report = self._generate_training_report(results)
        self._save_training_report(report)

        logger.info("Training complete!")
        return results

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        tier_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare features and targets for training."""

        # Identify feature columns (exclude metadata and targets)
        exclude_cols = [
            target_col, tier_col, 'nil_tier_encoded',
            'player_name_std', 'player_name', 'name', 'school_name',
            'position_group', 'position_raw'
        ]

        feature_cols = [c for c in df.columns if c not in exclude_cols
                       and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

        # Extract features
        X = df[feature_cols].values.astype(np.float32)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Target: NIL value (log transform for right-skewed distribution)
        y_value = df[target_col].values.astype(np.float64)
        y_value = np.clip(y_value, 1, None)  # Avoid log(0)
        y_value_log = np.log1p(y_value)

        # Target: NIL tier
        if tier_col in df.columns:
            y_tier = df[tier_col].values
        else:
            # Create tier from value
            y_tier = pd.cut(
                y_value,
                bins=[0, 25000, 100000, 500000, 1000000, np.inf],
                labels=['entry', 'moderate', 'solid', 'premium', 'mega']
            ).astype(str)

        # Encode tiers
        self.tier_encoder = LabelEncoder()
        y_tier_encoded = self.tier_encoder.fit_transform(y_tier)

        return X, y_value_log, y_tier_encoded, feature_cols

    def _split_data(
        self,
        X: np.ndarray,
        y_value: np.ndarray,
        y_tier: np.ndarray
    ) -> Tuple:
        """Split data with stratification by tier."""

        if self.small_data_mode:
            test_size = 0.3  # Larger test set for small data
        else:
            test_size = 0.2

        try:
            X_train, X_test, y_val_train, y_val_test, y_tier_train, y_tier_test = \
                train_test_split(
                    X, y_value, y_tier,
                    test_size=test_size,
                    stratify=y_tier,
                    random_state=42
                )
        except ValueError:
            # Stratification failed (too few samples in some classes)
            logger.warning("Stratification failed, using random split")
            X_train, X_test, y_val_train, y_val_test, y_tier_train, y_tier_test = \
                train_test_split(
                    X, y_value, y_tier,
                    test_size=test_size,
                    random_state=42
                )

        return X_train, X_test, y_val_train, y_val_test, y_tier_train, y_tier_test

    def _train_regression_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train and evaluate regression models."""

        results = {'models': {}}

        # Define models
        models = {
            'ridge': Ridge(alpha=1.0),
        }

        if not self.small_data_mode:
            models['random_forest'] = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            if HAS_XGBOOST:
                models['xgboost'] = xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbosity=0
                )
            if HAS_LIGHTGBM:
                models['lightgbm'] = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbose=-1
                )
        else:
            # Simpler models for small data
            models['random_forest'] = RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=42
            )

        # Cross-validation setup
        if self.small_data_mode and len(y_train) < 20:
            cv = LeaveOneOut()
            cv_name = "Leave-One-Out"
        else:
            n_splits = 3 if self.small_data_mode else 5
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_name = f"{n_splits}-Fold"

        # Train and evaluate each model
        best_model = None
        best_mae = float('inf')
        best_name = None

        for name, model in models.items():
            logger.info(f"  Training {name}...")

            # Cross-validation
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring='neg_mean_absolute_error'
                )
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as e:
                logger.warning(f"  CV failed for {name}: {e}")
                cv_mae = float('inf')
                cv_std = 0

            # Fit on full training set
            model.fit(X_train, y_train)

            # Test set predictions
            y_pred_log = model.predict(X_test)

            # Convert back from log scale
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_test)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # MAPE (handle zeros)
            mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100

            results['models'][name] = {
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'cv_method': cv_name,
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2,
                'test_mape': mape,
            }

            logger.info(f"    CV MAE: ${np.expm1(cv_mae):,.0f}, Test MAE: ${mae:,.0f}, R²: {r2:.3f}")

            # Track best model
            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_name = name

        # Store best model
        self.value_model = best_model
        results['best_model'] = best_name
        results['best_metrics'] = results['models'][best_name]

        return results

    def _train_classification_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train and evaluate classification models."""

        results = {'models': {}}

        # Define models
        models = {
            'logistic': LogisticRegression(
                max_iter=1000, random_state=42, multi_class='multinomial'
            ),
        }

        if not self.small_data_mode:
            models['random_forest'] = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            if HAS_XGBOOST:
                models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False,
                    eval_metric='mlogloss'
                )
        else:
            models['random_forest'] = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42
            )

        # Train and evaluate
        best_model = None
        best_f1 = 0
        best_name = None

        for name, model in models.items():
            logger.info(f"  Training {name}...")

            # Fit
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)

            precision, recall, f1_per_class, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0
            )

            results['models'][name] = {
                'accuracy': accuracy,
                'weighted_f1': f1,
                'confusion_matrix': cm.tolist(),
                'per_class_precision': precision.tolist(),
                'per_class_recall': recall.tolist(),
                'per_class_f1': f1_per_class.tolist(),
            }

            logger.info(f"    Accuracy: {accuracy:.3f}, Weighted F1: {f1:.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = name

        self.tier_model = best_model
        results['best_model'] = best_name
        results['best_metrics'] = results['models'][best_name]
        results['class_labels'] = self.tier_encoder.classes_.tolist()

        return results

    def _train_two_stage_model(
        self,
        X_train: np.ndarray,
        y_val_train: np.ndarray,
        y_tier_train: np.ndarray,
        X_test: np.ndarray,
        y_val_test: np.ndarray,
        y_tier_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train two-stage model: tier classification then value regression per tier."""

        results = {}

        # Stage 1: Use already-trained tier classifier
        tier_predictions_train = self.tier_model.predict(X_train)
        tier_predictions_test = self.tier_model.predict(X_test)

        # Stage 2: Train separate regressor for each tier
        self.tier_value_models = {}
        tier_labels = self.tier_encoder.classes_

        for tier_idx, tier_name in enumerate(tier_labels):
            # Get samples in this tier
            train_mask = tier_predictions_train == tier_idx
            if train_mask.sum() < 3:
                logger.warning(f"  Tier '{tier_name}': too few samples ({train_mask.sum()}), using global model")
                self.tier_value_models[tier_idx] = self.value_model
                continue

            X_tier = X_train[train_mask]
            y_tier = y_val_train[train_mask]

            # Train tier-specific regressor
            if self.small_data_mode or train_mask.sum() < 20:
                model = Ridge(alpha=1.0)
            else:
                model = RandomForestRegressor(
                    n_estimators=50, max_depth=8, random_state=42
                )

            model.fit(X_tier, y_tier)
            self.tier_value_models[tier_idx] = model
            logger.info(f"  Tier '{tier_name}': trained on {train_mask.sum()} samples")

        # Evaluate two-stage model on test set
        y_pred_log = np.zeros(len(X_test))
        for i, (x, tier_pred) in enumerate(zip(X_test, tier_predictions_test)):
            model = self.tier_value_models.get(tier_pred, self.value_model)
            y_pred_log[i] = model.predict(x.reshape(1, -1))[0]

        # Convert back from log
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_val_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100

        results['test_rmse'] = rmse
        results['test_mae'] = mae
        results['test_r2'] = r2
        results['test_mape'] = mape
        results['n_tier_models'] = len(self.tier_value_models)

        logger.info(f"  Two-stage: MAE ${mae:,.0f}, R² {r2:.3f}")

        return results

    def _calculate_shap_values(
        self,
        X_train: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Calculate SHAP values for model interpretability."""

        if not HAS_SHAP:
            return

        try:
            # Use TreeExplainer for tree-based models, KernelExplainer otherwise
            if hasattr(self.value_model, 'feature_importances_'):
                self.shap_explainer = shap.TreeExplainer(self.value_model)
            else:
                # Sample background data for KernelExplainer
                background = X_train[:min(100, len(X_train))]
                self.shap_explainer = shap.KernelExplainer(
                    self.value_model.predict, background
                )

            # Calculate SHAP values on sample
            sample_size = min(200, len(X_train))
            X_sample = X_train[:sample_size]
            self.shap_values = self.shap_explainer.shap_values(X_sample)

            logger.info(f"  SHAP values calculated for {sample_size} samples")

        except Exception as e:
            logger.warning(f"  SHAP calculation failed: {e}")
            self.shap_explainer = None
            self.shap_values = None

    def _save_models(self) -> None:
        """Save trained models to disk."""

        os.makedirs(self.model_dir, exist_ok=True)

        # Save value model
        joblib.dump(self.value_model, os.path.join(self.model_dir, 'value_model.joblib'))

        # Save tier model
        joblib.dump(self.tier_model, os.path.join(self.model_dir, 'tier_model.joblib'))

        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))

        # Save tier encoder
        joblib.dump(self.tier_encoder, os.path.join(self.model_dir, 'tier_encoder.joblib'))

        # Save tier-specific models if using two-stage
        if self.use_two_stage:
            joblib.dump(
                self.tier_value_models,
                os.path.join(self.model_dir, 'tier_value_models.joblib')
            )

        # Save metrics
        with open(os.path.join(self.model_dir, 'metrics.json'), 'w') as f:
            # Convert numpy types for JSON serialization
            metrics_json = self._convert_to_json_serializable(self.metrics)
            json.dump(metrics_json, f, indent=2)

        # Save feature names
        with open(os.path.join(self.model_dir, 'feature_names.json'), 'w') as f:
            json.dump(self.feature_names, f)

        # Save config
        config = {
            'use_two_stage': self.use_two_stage,
            'small_data_mode': self.small_data_mode,
            'n_features': len(self.feature_names),
            'trained_at': datetime.now().isoformat(),
        }
        with open(os.path.join(self.model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Models saved to {self.model_dir}")

    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    def _generate_training_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive training report."""

        lines = []
        lines.append("=" * 70)
        lines.append("NIL VALUATION MODEL TRAINING REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # Data summary
        lines.append("DATA SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total samples: {self.training_stats['n_samples']}")
        lines.append(f"Features: {len(self.feature_names)}")
        lines.append(f"Small data mode: {'Yes' if self.small_data_mode else 'No'}")
        if self.small_data_mode:
            lines.append(f"⚠️  WARNING: Dataset is small. Recommended minimum: {self.min_recommended_samples} samples")
        lines.append("")

        # Regression results
        lines.append("REGRESSION MODEL RESULTS (Value Prediction)")
        lines.append("-" * 40)
        for name, metrics in results['regression']['models'].items():
            lines.append(f"\n{name.upper()}:")
            lines.append(f"  Cross-validation ({metrics['cv_method']}):")
            lines.append(f"    MAE: ${np.expm1(metrics['cv_mae']):,.0f} (±${np.expm1(metrics['cv_std']):,.0f})")
            lines.append(f"  Test Set:")
            lines.append(f"    RMSE: ${metrics['test_rmse']:,.0f}")
            lines.append(f"    MAE: ${metrics['test_mae']:,.0f}")
            lines.append(f"    R²: {metrics['test_r2']:.3f}")
            lines.append(f"    MAPE: {metrics['test_mape']:.1f}%")

        lines.append(f"\n✓ Best regression model: {results['regression']['best_model'].upper()}")
        lines.append("")

        # Classification results
        lines.append("CLASSIFICATION MODEL RESULTS (Tier Prediction)")
        lines.append("-" * 40)
        for name, metrics in results['classification']['models'].items():
            lines.append(f"\n{name.upper()}:")
            lines.append(f"  Accuracy: {metrics['accuracy']:.3f}")
            lines.append(f"  Weighted F1: {metrics['weighted_f1']:.3f}")
            lines.append(f"  Per-class F1: {[f'{f:.2f}' for f in metrics['per_class_f1']]}")

        lines.append(f"\n✓ Best classification model: {results['classification']['best_model'].upper()}")
        lines.append(f"  Class labels: {results['classification']['class_labels']}")
        lines.append("")

        # Two-stage results
        lines.append("TWO-STAGE MODEL RESULTS")
        lines.append("-" * 40)
        two_stage = results['two_stage']
        lines.append(f"  RMSE: ${two_stage['test_rmse']:,.0f}")
        lines.append(f"  MAE: ${two_stage['test_mae']:,.0f}")
        lines.append(f"  R²: {two_stage['test_r2']:.3f}")
        lines.append(f"  MAPE: {two_stage['test_mape']:.1f}%")
        lines.append(f"  Tier-specific models: {two_stage['n_tier_models']}")
        lines.append("")

        # Final selection
        lines.append("FINAL MODEL SELECTION")
        lines.append("-" * 40)
        lines.append(f"✓ Selected approach: {results['selected_approach'].upper()}")
        if results['selected_approach'] == 'two_stage':
            lines.append("  Reason: Two-stage model showed >5% improvement in MAE")
        else:
            lines.append("  Reason: Single regression model performed similarly or better")
        lines.append("")

        # Feature importance
        if hasattr(self.value_model, 'feature_importances_'):
            lines.append("TOP 15 FEATURE IMPORTANCES")
            lines.append("-" * 40)
            importances = self.value_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            for i, idx in enumerate(indices):
                lines.append(f"  {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _save_training_report(self, report: str) -> None:
        """Save training report to file."""

        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, 'nil_model_training_report.txt')

        with open(report_path, 'w') as f:
            f.write(report)

        print(report)
        logger.info(f"Training report saved to {report_path}")

    def load_models(self) -> None:
        """Load previously trained models from disk."""

        self.value_model = joblib.load(os.path.join(self.model_dir, 'value_model.joblib'))
        self.tier_model = joblib.load(os.path.join(self.model_dir, 'tier_model.joblib'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
        self.tier_encoder = joblib.load(os.path.join(self.model_dir, 'tier_encoder.joblib'))

        with open(os.path.join(self.model_dir, 'feature_names.json'), 'r') as f:
            self.feature_names = json.load(f)

        with open(os.path.join(self.model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
            self.use_two_stage = config.get('use_two_stage', False)
            self.small_data_mode = config.get('small_data_mode', False)

        if self.use_two_stage:
            self.tier_value_models = joblib.load(
                os.path.join(self.model_dir, 'tier_value_models.joblib')
            )

        logger.info("Models loaded successfully")

    def predict(self, player_features: Union[pd.Series, pd.DataFrame, Dict]) -> Dict[str, Any]:
        """
        Predict NIL value for a player.

        Args:
            player_features: Player feature data (Series, single-row DataFrame, or dict)

        Returns:
            Comprehensive prediction dictionary
        """
        if self.value_model is None:
            raise ValueError("Model not trained. Call train() or load_models() first.")

        # Convert to array
        X = self._prepare_prediction_input(player_features)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get predictions
        if self.use_two_stage:
            # Two-stage prediction
            tier_pred = self.tier_model.predict(X_scaled)[0]
            model = self.tier_value_models.get(tier_pred, self.value_model)
            value_pred_log = model.predict(X_scaled)[0]
        else:
            # Single model prediction
            value_pred_log = self.value_model.predict(X_scaled)[0]

        # Convert from log scale
        predicted_value = np.expm1(value_pred_log)
        predicted_value = max(0, predicted_value)

        # Tier prediction and probabilities
        tier_pred_idx = self.tier_model.predict(X_scaled)[0]
        tier_probs = self.tier_model.predict_proba(X_scaled)[0]
        predicted_tier = self.tier_encoder.inverse_transform([tier_pred_idx])[0]

        tier_probabilities = {
            self.tier_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(tier_probs)
        }

        # Prediction range (using training error as confidence interval)
        cv_error = self.metrics.get('regression', {}).get('best_metrics', {}).get('test_mae', predicted_value * 0.2)
        pred_range = {
            'low': max(0, predicted_value - 1.5 * cv_error),
            'mid': predicted_value,
            'high': predicted_value + 1.5 * cv_error,
        }

        # Confidence assessment
        confidence = self._assess_prediction_confidence(X_scaled)

        # Value breakdown by feature group
        value_breakdown = self._calculate_value_breakdown(X_scaled, predicted_value)

        # Comparable players
        comparable_players = self._find_comparable_players(X_scaled)

        # SHAP explanation
        shap_explanation = self._get_shap_explanation(X_scaled, predicted_value)

        return {
            'predicted_nil_value': round(predicted_value, 2),
            'predicted_nil_range': {k: round(v, 2) for k, v in pred_range.items()},
            'predicted_tier': predicted_tier,
            'tier_probabilities': tier_probabilities,
            'confidence': confidence,
            'value_breakdown': value_breakdown,
            'comparable_players': comparable_players,
            'shap_explanation': shap_explanation,
        }

    def _prepare_prediction_input(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict]
    ) -> np.ndarray:
        """Convert player features to model input array."""

        if isinstance(player_features, dict):
            player_features = pd.Series(player_features)
        elif isinstance(player_features, pd.DataFrame):
            if len(player_features) != 1:
                raise ValueError("DataFrame must have exactly 1 row")
            player_features = player_features.iloc[0]

        # Extract features in correct order
        X = np.zeros((1, len(self.feature_names)))
        for i, feat in enumerate(self.feature_names):
            if feat in player_features.index:
                X[0, i] = player_features[feat]

        return X.astype(np.float32)

    def _assess_prediction_confidence(self, X_scaled: np.ndarray) -> str:
        """Assess confidence based on distance from training distribution."""

        if not hasattr(self, 'training_stats') or 'feature_means' not in self.training_stats:
            return 'medium'

        # Calculate Mahalanobis-like distance from training mean
        diff = X_scaled[0] - 0  # Scaled data has mean ~0
        distance = np.sqrt(np.sum(diff ** 2))

        # Thresholds (scaled features have std ~1)
        n_features = len(self.feature_names)
        expected_distance = np.sqrt(n_features)  # Expected for standard normal

        if distance < expected_distance * 1.5:
            return 'high'
        elif distance < expected_distance * 2.5:
            return 'medium'
        else:
            return 'low'

    def _calculate_value_breakdown(
        self,
        X_scaled: np.ndarray,
        predicted_value: float
    ) -> Dict[str, float]:
        """Calculate value attribution by feature group."""

        breakdown = {}

        # Use feature importances if available
        if hasattr(self.value_model, 'feature_importances_'):
            importances = self.value_model.feature_importances_

            for group_name, group_features in FEATURE_GROUPS.items():
                group_importance = 0
                for feat in group_features:
                    if feat in self.feature_names:
                        idx = self.feature_names.index(feat)
                        group_importance += importances[idx]
                breakdown[group_name] = group_importance

            # Normalize to sum to 1
            total = sum(breakdown.values())
            if total > 0:
                breakdown = {k: round(v / total, 3) for k, v in breakdown.items()}
        else:
            # Default breakdown
            breakdown = {
                'on_field_performance': 0.30,
                'social_media': 0.25,
                'school_brand': 0.20,
                'recruiting_pedigree': 0.15,
                'draft_projection': 0.10,
            }

        return breakdown

    def _find_comparable_players(
        self,
        X_scaled: np.ndarray,
        n_comparables: int = 3
    ) -> List[Dict[str, Any]]:
        """Find most similar players from training data."""

        if self.training_data is None:
            return []

        try:
            # Prepare training features
            X_train = self.training_data[self.feature_names].values.astype(np.float32)
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_train_scaled = self.scaler.transform(X_train)

            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=min(n_comparables, len(X_train)), metric='euclidean')
            nn.fit(X_train_scaled)
            distances, indices = nn.kneighbors(X_scaled)

            # Build comparable list
            comparables = []
            for dist, idx in zip(distances[0], indices[0]):
                row = self.training_data.iloc[idx]

                name = row.get('player_name_std', row.get('player_name', row.get('name', f'Player {idx}')))
                school = row.get('school_name', row.get('school', 'Unknown'))
                nil_value = row.get('nil_value', 0)

                # Similarity score (inverse of distance, normalized)
                similarity = max(0, 1 - dist / 10)

                comparables.append({
                    'name': str(name),
                    'school': str(school),
                    'nil_value': float(nil_value),
                    'similarity_score': round(similarity, 3),
                })

            return comparables

        except Exception as e:
            logger.warning(f"Error finding comparables: {e}")
            return []

    def _get_shap_explanation(
        self,
        X_scaled: np.ndarray,
        predicted_value: float
    ) -> Dict[str, float]:
        """Get SHAP-based explanation for prediction."""

        if not HAS_SHAP or self.shap_explainer is None:
            # Fallback: use feature importances * feature values
            if hasattr(self.value_model, 'feature_importances_'):
                importances = self.value_model.feature_importances_
                contributions = importances * np.abs(X_scaled[0])

                # Get top 5
                indices = np.argsort(contributions)[::-1][:5]
                explanation = {}
                for idx in indices:
                    feat_name = self.feature_names[idx]
                    # Estimate dollar impact (rough approximation)
                    impact = contributions[idx] * predicted_value * 0.5
                    if X_scaled[0, idx] < 0:
                        impact = -impact
                    explanation[feat_name] = round(impact, 0)
                return explanation
            return {}

        try:
            # Calculate SHAP values for this prediction
            shap_vals = self.shap_explainer.shap_values(X_scaled)

            # Get top 5 by absolute value
            indices = np.argsort(np.abs(shap_vals[0]))[::-1][:5]

            explanation = {}
            for idx in indices:
                feat_name = self.feature_names[idx]
                # Convert from log scale impact to dollar impact
                log_impact = shap_vals[0][idx]
                dollar_impact = np.expm1(log_impact) * np.sign(log_impact) * predicted_value * 0.1
                explanation[feat_name] = round(dollar_impact, 0)

            return explanation

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {}

    def transfer_impact(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        new_school: str
    ) -> Dict[str, Any]:
        """
        Analyze impact of transferring to a new school.

        Args:
            player_features: Current player features
            new_school: Target school name

        Returns:
            Dictionary with current and projected values
        """
        # Get current prediction
        current_pred = self.predict(player_features)
        current_value = current_pred['predicted_nil_value']

        # Get new school data
        school_data = SCHOOL_DATA.get(new_school)
        if school_data is None:
            # Use defaults for unknown school
            school_data = {'tier': 3, 'conference_tier': 2, 'market_size': 500, 'wins': 7, 'football_state': 0}
            logger.warning(f"School '{new_school}' not found, using default values")

        # Create modified features
        if isinstance(player_features, dict):
            modified = player_features.copy()
        elif isinstance(player_features, pd.Series):
            modified = player_features.to_dict()
        else:
            modified = player_features.iloc[0].to_dict()

        # Swap school-related features
        modified['school_tier'] = school_data['tier']
        modified['conference_tier'] = school_data['conference_tier']
        modified['school_market_size'] = school_data['market_size']
        modified['team_wins'] = school_data['wins']
        modified['football_state'] = school_data['football_state']

        # Update interaction features if present
        if 'school_x_production' in modified and 'production_score' in modified:
            modified['school_x_production'] = school_data['tier'] * modified['production_score']
        if 'market_x_production' in modified and 'production_score' in modified:
            modified['market_x_production'] = np.log1p(school_data['market_size']) * modified['production_score']

        # Get new prediction
        new_pred = self.predict(modified)
        new_value = new_pred['predicted_nil_value']

        return {
            'current_school_value': round(current_value, 2),
            'projected_value_at_new_school': round(new_value, 2),
            'value_change': round(new_value - current_value, 2),
            'pct_change': round((new_value - current_value) / current_value * 100, 2) if current_value > 0 else 0,
            'new_school': new_school,
            'current_tier': current_pred['predicted_tier'],
            'projected_tier': new_pred['predicted_tier'],
        }

    def what_if_social(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        new_follower_counts: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Analyze impact of social media growth on NIL value.

        Args:
            player_features: Current player features
            new_follower_counts: Dict with platform -> follower count
                e.g., {'instagram': 500000, 'tiktok': 1000000}

        Returns:
            Dictionary with current and projected values
        """
        # Get current prediction
        current_pred = self.predict(player_features)
        current_value = current_pred['predicted_nil_value']

        # Create modified features
        if isinstance(player_features, dict):
            modified = player_features.copy()
        elif isinstance(player_features, pd.Series):
            modified = player_features.to_dict()
        else:
            modified = player_features.iloc[0].to_dict()

        # Swap social media features
        platform_mapping = {
            'instagram': 'instagram_followers',
            'tiktok': 'tiktok_followers',
            'twitter': 'twitter_followers',
            'youtube': 'youtube_followers',
        }

        total_new = 0
        for platform, count in new_follower_counts.items():
            col = platform_mapping.get(platform.lower())
            if col:
                modified[col] = count
                modified[f'{col}_log'] = np.log1p(count)
                total_new += count

        # Update derived features
        modified['total_social_following'] = total_new
        modified['log_total_following'] = np.log1p(total_new)
        modified['has_significant_following'] = 1 if total_new > 50000 else 0

        # Calculate concentration (max platform / total)
        max_platform = max(new_follower_counts.values()) if new_follower_counts else 0
        modified['social_platform_concentration'] = max_platform / total_new if total_new > 0 else 1

        # Estimate social value (CPM-based)
        cpm_rates = {'instagram': 10, 'tiktok': 5, 'twitter': 3, 'youtube': 15}
        est_value = 0
        for platform, count in new_follower_counts.items():
            cpm = cpm_rates.get(platform.lower(), 5)
            est_value += count * 0.03 * 4 * 12 * cpm / 1000  # 3% engagement, 4 posts/month
        modified['estimated_social_value'] = est_value

        # Update interaction features
        if 'social_x_production' in modified and 'production_score' in modified:
            modified['social_x_production'] = np.log1p(total_new) * modified['production_score']

        # Get new prediction
        new_pred = self.predict(modified)
        new_value = new_pred['predicted_nil_value']

        return {
            'current_value': round(current_value, 2),
            'projected_value': round(new_value, 2),
            'value_change': round(new_value - current_value, 2),
            'pct_change': round((new_value - current_value) / current_value * 100, 2) if current_value > 0 else 0,
            'new_follower_counts': new_follower_counts,
            'current_tier': current_pred['predicted_tier'],
            'projected_tier': new_pred['predicted_tier'],
        }

    def generate_position_market_report(
        self,
        features_df: pd.DataFrame,
        position_group: str
    ) -> Dict[str, Any]:
        """
        Generate market report for a specific position.

        Args:
            features_df: DataFrame with player features and NIL values
            position_group: Position group to analyze (QB, RB, WR, etc.)

        Returns:
            Dictionary with market analysis
        """
        # Filter to position
        if 'position_group' in features_df.columns:
            pos_df = features_df[features_df['position_group'] == position_group].copy()
        else:
            pos_df = features_df.copy()
            logger.warning("No position_group column, using all data")

        if len(pos_df) == 0:
            return {'error': f'No players found for position {position_group}'}

        # Value distribution
        nil_col = 'nil_value' if 'nil_value' in pos_df.columns else None
        if nil_col is None:
            return {'error': 'No nil_value column found'}

        values = pos_df[nil_col].dropna()

        distribution = {
            'count': len(values),
            'mean': round(values.mean(), 2),
            'median': round(values.median(), 2),
            'std': round(values.std(), 2),
            'min': round(values.min(), 2),
            'max': round(values.max(), 2),
            'percentiles': {
                '25th': round(values.quantile(0.25), 2),
                '50th': round(values.quantile(0.50), 2),
                '75th': round(values.quantile(0.75), 2),
                '90th': round(values.quantile(0.90), 2),
                '95th': round(values.quantile(0.95), 2),
            }
        }

        # Top 10 most valuable
        name_col = None
        for col in ['player_name_std', 'player_name', 'name']:
            if col in pos_df.columns:
                name_col = col
                break

        school_col = None
        for col in ['school_name', 'school', 'team']:
            if col in pos_df.columns:
                school_col = col
                break

        top_10 = pos_df.nlargest(10, nil_col)
        top_players = []
        for _, row in top_10.iterrows():
            player = {
                'name': str(row[name_col]) if name_col else 'Unknown',
                'school': str(row[school_col]) if school_col else 'Unknown',
                'nil_value': round(row[nil_col], 2),
            }
            if 'production_score' in row:
                player['production_score'] = round(row['production_score'], 1)
            top_players.append(player)

        # Average by school tier
        if 'school_tier' in pos_df.columns:
            tier_avg = pos_df.groupby('school_tier')[nil_col].mean().round(2).to_dict()
            tier_mapping = {6: 'blue_blood', 5: 'elite', 4: 'power_brand', 3: 'p4_mid', 2: 'g5_strong', 1: 'g5'}
            tier_avg_named = {tier_mapping.get(k, str(k)): v for k, v in tier_avg.items()}
        else:
            tier_avg_named = {}

        # Average by conference tier
        if 'conference_tier' in pos_df.columns:
            conf_avg = pos_df.groupby('conference_tier')[nil_col].mean().round(2).to_dict()
            conf_mapping = {3: 'SEC/Big Ten', 2: 'Big 12/ACC', 1: 'G5'}
            conf_avg_named = {conf_mapping.get(k, str(k)): v for k, v in conf_avg.items()}
        else:
            conf_avg_named = {}

        # Most undervalued (high production, low NIL)
        if 'production_score' in pos_df.columns:
            pos_df['value_efficiency'] = pos_df['production_score'] / (pos_df[nil_col] / 100000 + 1)
            undervalued = pos_df.nlargest(5, 'value_efficiency')
            undervalued_players = []
            for _, row in undervalued.iterrows():
                undervalued_players.append({
                    'name': str(row[name_col]) if name_col else 'Unknown',
                    'school': str(row[school_col]) if school_col else 'Unknown',
                    'nil_value': round(row[nil_col], 2),
                    'production_score': round(row['production_score'], 1),
                    'value_efficiency': round(row['value_efficiency'], 2),
                })
        else:
            undervalued_players = []

        report = {
            'position_group': position_group,
            'generated_at': datetime.now().isoformat(),
            'distribution': distribution,
            'top_10_players': top_players,
            'average_by_school_tier': tier_avg_named,
            'average_by_conference': conf_avg_named,
            'most_undervalued': undervalued_players,
        }

        # Save report
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(
            self.output_dir,
            f'nil_market_report_{position_group.lower()}.json'
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Also save text version
        text_report = self._format_market_report_text(report)
        text_path = os.path.join(
            self.output_dir,
            f'nil_market_report_{position_group.lower()}.txt'
        )
        with open(text_path, 'w') as f:
            f.write(text_report)

        logger.info(f"Position market report saved to {report_path}")

        return report

    def _format_market_report_text(self, report: Dict[str, Any]) -> str:
        """Format market report as text."""

        lines = []
        lines.append("=" * 60)
        lines.append(f"NIL MARKET REPORT: {report['position_group']}")
        lines.append(f"Generated: {report['generated_at']}")
        lines.append("=" * 60)
        lines.append("")

        # Distribution
        dist = report['distribution']
        lines.append("VALUE DISTRIBUTION")
        lines.append("-" * 40)
        lines.append(f"Players: {dist['count']}")
        lines.append(f"Mean: ${dist['mean']:,.0f}")
        lines.append(f"Median: ${dist['median']:,.0f}")
        lines.append(f"Std Dev: ${dist['std']:,.0f}")
        lines.append(f"Range: ${dist['min']:,.0f} - ${dist['max']:,.0f}")
        lines.append(f"90th percentile: ${dist['percentiles']['90th']:,.0f}")
        lines.append("")

        # Top players
        lines.append("TOP 10 MOST VALUABLE")
        lines.append("-" * 40)
        for i, player in enumerate(report['top_10_players'], 1):
            prod = f" (Prod: {player.get('production_score', 'N/A')})" if 'production_score' in player else ""
            lines.append(f"{i}. {player['name']} ({player['school']}): ${player['nil_value']:,.0f}{prod}")
        lines.append("")

        # By tier
        if report['average_by_school_tier']:
            lines.append("AVERAGE BY SCHOOL TIER")
            lines.append("-" * 40)
            for tier, avg in sorted(report['average_by_school_tier'].items(), key=lambda x: -x[1]):
                lines.append(f"  {tier}: ${avg:,.0f}")
            lines.append("")

        # By conference
        if report['average_by_conference']:
            lines.append("AVERAGE BY CONFERENCE")
            lines.append("-" * 40)
            for conf, avg in sorted(report['average_by_conference'].items(), key=lambda x: -x[1]):
                lines.append(f"  {conf}: ${avg:,.0f}")
            lines.append("")

        # Undervalued
        if report['most_undervalued']:
            lines.append("MOST UNDERVALUED (High Production, Low NIL)")
            lines.append("-" * 40)
            for player in report['most_undervalued']:
                lines.append(f"  {player['name']} ({player['school']})")
                lines.append(f"    NIL: ${player['nil_value']:,.0f}, Production: {player['production_score']}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    print("NIL Valuator - Standalone Mode")
    print("=" * 50)

    # Create sample data
    print("\nGenerating sample training data...")

    np.random.seed(42)
    n_samples = 100

    # Generate synthetic features
    data = {
        'player_name_std': [f'player_{i}' for i in range(n_samples)],
        'school_name': np.random.choice(['Alabama', 'Ohio State', 'Georgia', 'Texas', 'USC', 'Michigan'], n_samples),
        'position_group': np.random.choice(['QB', 'WR', 'RB', 'TE', 'EDGE', 'CB'], n_samples),
        'production_score': np.random.uniform(20, 95, n_samples),
        'school_tier': np.random.choice([6, 5, 4, 3, 2, 1], n_samples, p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05]),
        'conference_tier': np.random.choice([3, 2, 1], n_samples, p=[0.5, 0.3, 0.2]),
        'school_market_size': np.random.uniform(100, 5000, n_samples),
        'recruiting_stars': np.random.choice([5, 4, 3, 2, 0], n_samples, p=[0.05, 0.15, 0.4, 0.3, 0.1]),
        'recruiting_composite': np.random.uniform(0.7, 1.0, n_samples),
        'total_social_following': np.random.exponential(100000, n_samples),
        'log_total_following': None,  # Will calculate
        'instagram_followers_log': np.random.uniform(8, 14, n_samples),
        'tiktok_followers_log': np.random.uniform(7, 15, n_samples),
        'twitter_followers_log': np.random.uniform(7, 13, n_samples),
        'has_significant_following': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'projected_draft_flag': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'projected_draft_round': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], n_samples),
        'team_wins': np.random.randint(4, 14, n_samples),
        'years_in_college': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'qb_premium': (np.array([1 if p == 'QB' else 0 for p in np.random.choice(['QB', 'WR', 'RB', 'TE', 'EDGE', 'CB'], n_samples)])),
    }

    data['log_total_following'] = np.log1p(data['total_social_following'])

    # Generate NIL values based on features (with noise)
    nil_values = (
        data['production_score'] * 5000 +
        data['school_tier'] * 50000 +
        data['log_total_following'] * 20000 +
        data['recruiting_stars'] * 30000 +
        data['projected_draft_flag'] * 200000 +
        np.random.normal(0, 50000, n_samples)
    )
    nil_values = np.clip(nil_values, 5000, 5000000)
    data['nil_value'] = nil_values

    # Create tier
    def assign_tier(val):
        if val >= 1000000: return 'mega'
        elif val >= 500000: return 'premium'
        elif val >= 100000: return 'solid'
        elif val >= 25000: return 'moderate'
        else: return 'entry'

    data['nil_tier'] = [assign_tier(v) for v in nil_values]

    # Add interaction features
    data['school_x_production'] = np.array(data['school_tier']) * np.array(data['production_score'])
    data['social_x_production'] = np.array(data['log_total_following']) * np.array(data['production_score'])

    df = pd.DataFrame(data)

    print(f"Generated {len(df)} sample players")
    print(f"NIL Value range: ${df['nil_value'].min():,.0f} - ${df['nil_value'].max():,.0f}")
    print(f"Tier distribution: {df['nil_tier'].value_counts().to_dict()}")

    # Initialize and train
    valuator = NILValuator(
        model_dir="models/nil_valuation",
        output_dir="outputs/reports"
    )

    try:
        print("\n" + "=" * 50)
        print("TRAINING")
        print("=" * 50)

        results = valuator.train(df, target_col='nil_value', tier_col='nil_tier')

        # Test prediction
        print("\n" + "=" * 50)
        print("SAMPLE PREDICTION")
        print("=" * 50)

        test_player = df.iloc[0]
        prediction = valuator.predict(test_player)

        print(f"\nPlayer: {test_player['player_name_std']}")
        print(f"Actual NIL: ${test_player['nil_value']:,.0f}")
        print(f"Predicted NIL: ${prediction['predicted_nil_value']:,.0f}")
        print(f"Predicted Tier: {prediction['predicted_tier']}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"\nValue Breakdown: {prediction['value_breakdown']}")
        print(f"\nTop SHAP factors: {prediction['shap_explanation']}")

        # Test transfer impact
        print("\n" + "=" * 50)
        print("TRANSFER IMPACT ANALYSIS")
        print("=" * 50)

        transfer = valuator.transfer_impact(test_player, 'Texas')
        print(f"\nTransfer from {test_player['school_name']} to Texas:")
        print(f"  Current: ${transfer['current_school_value']:,.0f}")
        print(f"  Projected: ${transfer['projected_value_at_new_school']:,.0f}")
        print(f"  Change: ${transfer['value_change']:,.0f} ({transfer['pct_change']:+.1f}%)")

        # Test social media what-if
        print("\n" + "=" * 50)
        print("SOCIAL MEDIA WHAT-IF")
        print("=" * 50)

        social_impact = valuator.what_if_social(test_player, {
            'instagram': 1000000,
            'tiktok': 2000000,
            'twitter': 500000
        })
        print(f"\nIf player grows to 3.5M total followers:")
        print(f"  Current: ${social_impact['current_value']:,.0f}")
        print(f"  Projected: ${social_impact['projected_value']:,.0f}")
        print(f"  Change: ${social_impact['value_change']:,.0f} ({social_impact['pct_change']:+.1f}%)")

        # Generate position report
        print("\n" + "=" * 50)
        print("POSITION MARKET REPORT")
        print("=" * 50)

        report = valuator.generate_position_market_report(df, 'QB')
        print(f"\nQB Market Report:")
        print(f"  Players: {report['distribution']['count']}")
        print(f"  Mean NIL: ${report['distribution']['mean']:,.0f}")
        print(f"  Median NIL: ${report['distribution']['median']:,.0f}")

        print("\n✓ NIL Valuator test complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
