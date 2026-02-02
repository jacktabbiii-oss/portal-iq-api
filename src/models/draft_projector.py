"""
NFL Draft Projection Model

Predicts whether college players will be drafted, their projected round/pick,
and estimated career earnings. Supports mock draft generation and stock tracking.

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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.neighbors import NearestNeighbors
import joblib

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Rookie wage scale by round and pick range (4-year total value, signing bonus)
# Based on 2024 rookie contracts
ROOKIE_WAGE_SCALE = {
    1: {  # Round 1
        (1, 5): {'total': 38_000_000, 'signing': 25_000_000, 'fifth_year': True},
        (6, 10): {'total': 26_000_000, 'signing': 16_000_000, 'fifth_year': True},
        (11, 16): {'total': 18_000_000, 'signing': 10_000_000, 'fifth_year': True},
        (17, 22): {'total': 14_000_000, 'signing': 7_500_000, 'fifth_year': True},
        (23, 32): {'total': 11_500_000, 'signing': 5_500_000, 'fifth_year': True},
    },
    2: {  # Round 2
        (33, 40): {'total': 8_500_000, 'signing': 3_800_000, 'fifth_year': False},
        (41, 50): {'total': 6_800_000, 'signing': 2_800_000, 'fifth_year': False},
        (51, 64): {'total': 5_500_000, 'signing': 2_000_000, 'fifth_year': False},
    },
    3: {  # Round 3
        (65, 80): {'total': 5_000_000, 'signing': 1_200_000, 'fifth_year': False},
        (81, 100): {'total': 4_500_000, 'signing': 900_000, 'fifth_year': False},
    },
    4: {  # Round 4
        (101, 120): {'total': 4_200_000, 'signing': 700_000, 'fifth_year': False},
        (121, 140): {'total': 4_000_000, 'signing': 550_000, 'fifth_year': False},
    },
    5: {  # Round 5
        (141, 160): {'total': 3_900_000, 'signing': 400_000, 'fifth_year': False},
        (161, 180): {'total': 3_800_000, 'signing': 300_000, 'fifth_year': False},
    },
    6: {  # Round 6
        (181, 210): {'total': 3_750_000, 'signing': 200_000, 'fifth_year': False},
    },
    7: {  # Round 7
        (211, 260): {'total': 3_700_000, 'signing': 125_000, 'fifth_year': False},
    },
}

# Average career earnings by draft position (8-year estimate including 2nd contract)
CAREER_EARNINGS_BY_PICK = {
    (1, 5): 180_000_000,      # Top 5 picks
    (6, 10): 120_000_000,     # Top 10
    (11, 20): 75_000_000,     # Rest of 1st round
    (21, 32): 55_000_000,     # Late 1st
    (33, 50): 35_000_000,     # Early 2nd
    (51, 75): 25_000_000,     # 2nd-3rd round
    (76, 100): 18_000_000,    # 3rd round
    (101, 150): 12_000_000,   # 4th-5th round
    (151, 200): 8_000_000,    # 5th-6th round
    (201, 260): 5_000_000,    # 7th round
}

# Position-specific career earnings multipliers
POSITION_EARNINGS_MULTIPLIER = {
    'QB': 2.5,    # QBs get massive 2nd contracts
    'EDGE': 1.5,  # Pass rushers valued
    'WR': 1.3,
    'CB': 1.2,
    'OL': 1.2,
    'DL': 1.1,
    'LB': 1.0,
    'S': 0.9,
    'TE': 0.9,
    'RB': 0.6,    # RB contracts declining
}

# Draft stock factor descriptions
STOCK_FACTOR_DESCRIPTIONS = {
    # Positive factors
    'elite_production': "Elite on-field production",
    'power_conference': "Power 4 conference competition",
    'young_age': "Young for draft class",
    'good_measurables': "Above-average athletic testing",
    'high_recruit': "Highly-recruited out of high school",
    'early_declare': "Early declaration shows confidence",
    'rising_production': "Production trending upward",
    'team_success': "Played on winning teams",
    'multiple_years': "Multiple years of college tape",
    'combine_standout': "Standout combine performance",

    # Negative factors
    'below_avg_measurables': "Below-average measurables for position",
    'one_year_starter': "Only one year of starting experience",
    'older_age': "Older than typical for draft class",
    'injury_history': "Injury concerns",
    'declining_production': "Production declined this year",
    'weak_competition': "Played against weaker competition",
    'off_field_concerns': "Character/off-field concerns noted",
    'scheme_dependent': "Production may be scheme-dependent",
    'raw_technique': "Needs technique refinement",
}

# Pick to round mapping
def pick_to_round(pick: int) -> int:
    """Convert overall pick to round number."""
    if pick <= 32:
        return 1
    elif pick <= 64:
        return 2
    elif pick <= 100:
        return 3
    elif pick <= 135:
        return 4
    elif pick <= 176:
        return 5
    elif pick <= 220:
        return 6
    else:
        return 7


def round_to_pick_range(round_num: int) -> Tuple[int, int]:
    """Get pick range for a round."""
    ranges = {
        1: (1, 32),
        2: (33, 64),
        3: (65, 100),
        4: (101, 135),
        5: (136, 176),
        6: (177, 220),
        7: (221, 260),
    }
    return ranges.get(round_num, (221, 260))


class DraftProjector:
    """
    NFL draft projection model.

    Features:
    - Draft classification (will be drafted?)
    - Round/pick prediction
    - Rookie contract estimation
    - Career earnings projection
    - Mock draft generation
    - Draft stock tracking
    """

    def __init__(
        self,
        model_dir: str = "models/draft_projection",
        output_dir: str = "outputs/reports"
    ):
        """
        Initialize the draft projector.

        Args:
            model_dir: Directory to save trained models
            output_dir: Directory to save reports
        """
        self.model_dir = model_dir
        self.output_dir = output_dir

        # Models
        self.drafted_model = None
        self.round_model = None
        self.scaler = None

        # Training data
        self.feature_names = []
        self.training_data = None

        # SHAP
        self.shap_explainer = None

        # Metrics
        self.metrics = {}

    def train(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train draft projection models.

        Args:
            features_df: DataFrame with features and targets
                - was_drafted: binary (0/1)
                - draft_round: 1-7 (0 for undrafted)
                - draft_pick: overall pick number

        Returns:
            Dictionary with training results
        """
        logger.info("Training draft projection models...")

        df = features_df.copy()

        # Validate targets
        if 'was_drafted' not in df.columns:
            raise ValueError("'was_drafted' target column required")

        # Prepare data
        X, y_drafted, y_round, y_pick, feature_names = self._prepare_data(df)
        self.feature_names = feature_names
        self.training_data = df.copy()

        # Split data
        X_train, X_test, y_draft_train, y_draft_test, y_round_train, y_round_test, y_pick_train, y_pick_test = \
            self._split_data(X, y_drafted, y_round, y_pick)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        # 1. Train draft classification model
        logger.info("Training draft classification model...")
        clf_results = self._train_draft_classifier(
            X_train_scaled, y_draft_train, X_test_scaled, y_draft_test
        )
        results['classification'] = clf_results

        # 2. Train round/pick prediction model (only on drafted players)
        logger.info("Training round prediction model...")
        drafted_train_mask = y_draft_train == 1
        drafted_test_mask = y_draft_test == 1

        if drafted_train_mask.sum() >= 10:
            round_results = self._train_round_predictor(
                X_train_scaled[drafted_train_mask], y_pick_train[drafted_train_mask],
                X_test_scaled[drafted_test_mask], y_pick_test[drafted_test_mask]
            )
            results['round_prediction'] = round_results
        else:
            logger.warning("Not enough drafted players to train round model")
            results['round_prediction'] = {'error': 'Insufficient drafted samples'}

        # Calculate SHAP values
        if HAS_SHAP:
            logger.info("Calculating SHAP values...")
            self._calculate_shap_values(X_train_scaled, feature_names)

        # Save models
        self._save_models()
        self.metrics = results

        # Generate report
        report = self._generate_training_report(results)
        self._save_training_report(report)

        logger.info("Draft projection training complete!")
        return results

    def _prepare_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare features and targets."""

        exclude_cols = [
            'was_drafted', 'draft_round', 'draft_pick', 'draft_value',
            'player_name_std', 'player_name', 'player_name_original',
            'name', 'school_name', 'position_group', 'position_raw', 'draft_year'
        ]

        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

        X = df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)

        y_drafted = df['was_drafted'].values.astype(int)

        if 'draft_round' in df.columns:
            y_round = df['draft_round'].fillna(0).values.astype(int)
        else:
            y_round = np.zeros(len(df), dtype=int)

        if 'draft_pick' in df.columns:
            y_pick = df['draft_pick'].fillna(0).values.astype(float)
        else:
            y_pick = np.zeros(len(df), dtype=float)

        return X, y_drafted, y_round, y_pick, feature_cols

    def _split_data(
        self,
        X: np.ndarray,
        y_drafted: np.ndarray,
        y_round: np.ndarray,
        y_pick: np.ndarray
    ) -> Tuple:
        """Split data for training and testing."""

        # Stratify by drafted status
        try:
            (X_train, X_test, y_draft_train, y_draft_test,
             y_round_train, y_round_test, y_pick_train, y_pick_test) = train_test_split(
                X, y_drafted, y_round, y_pick,
                test_size=0.2, stratify=y_drafted, random_state=42
            )
        except ValueError:
            # Stratification failed
            (X_train, X_test, y_draft_train, y_draft_test,
             y_round_train, y_round_test, y_pick_train, y_pick_test) = train_test_split(
                X, y_drafted, y_round, y_pick,
                test_size=0.2, random_state=42
            )

        return (X_train, X_test, y_draft_train, y_draft_test,
                y_round_train, y_round_test, y_pick_train, y_pick_test)

    def _train_draft_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train draft classification model."""

        results = {'models': {}}

        models = {
            'logistic': LogisticRegression(
                max_iter=1000, random_state=42, class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42,
                class_weight='balanced', n_jobs=-1
            ),
        }

        if HAS_XGBOOST:
            scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                scale_pos_weight=scale_pos, random_state=42,
                n_jobs=-1, verbosity=0, use_label_encoder=False,
                eval_metric='logloss'
            )

        best_model = None
        best_auc = 0
        best_name = None

        for name, model in models.items():
            logger.info(f"  Training {name}...")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            auc_roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results['models'][name] = {
                'auc_roc': auc_roc,
                'accuracy': accuracy,
                'f1': f1,
            }

            logger.info(f"    AUC: {auc_roc:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

            if auc_roc > best_auc:
                best_auc = auc_roc
                best_model = model
                best_name = name

        self.drafted_model = best_model
        results['best_model'] = best_name
        results['best_metrics'] = results['models'][best_name]

        return results

    def _train_round_predictor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,  # This is pick number
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train round/pick prediction model."""

        results = {'models': {}}

        models = {
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
        }

        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0
            )

        best_model = None
        best_mae = float('inf')
        best_name = None

        for name, model in models.items():
            logger.info(f"  Training {name}...")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Clip predictions to valid range
            y_pred = np.clip(y_pred, 1, 260)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0

            # Calculate round-level accuracy
            y_round_true = np.array([pick_to_round(int(p)) for p in y_test])
            y_round_pred = np.array([pick_to_round(int(p)) for p in y_pred])

            round_accuracy = (y_round_true == y_round_pred).mean()
            within_1_round = (np.abs(y_round_true - y_round_pred) <= 1).mean()
            mean_round_error = np.mean(np.abs(y_round_true - y_round_pred))

            results['models'][name] = {
                'pick_mae': mae,
                'pick_rmse': rmse,
                'pick_r2': r2,
                'round_accuracy': round_accuracy,
                'within_1_round_accuracy': within_1_round,
                'mean_round_error': mean_round_error,
            }

            logger.info(f"    Pick MAE: {mae:.1f}, Round Acc: {round_accuracy:.3f}, Within 1 Round: {within_1_round:.3f}")

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_name = name

        self.round_model = best_model
        results['best_model'] = best_name
        results['best_metrics'] = results['models'][best_name]

        return results

    def _calculate_shap_values(
        self,
        X_train: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Calculate SHAP values for draft model."""

        try:
            if hasattr(self.drafted_model, 'feature_importances_'):
                self.shap_explainer = shap.TreeExplainer(self.drafted_model)
            else:
                background = X_train[:min(100, len(X_train))]
                self.shap_explainer = shap.KernelExplainer(
                    self.drafted_model.predict_proba, background
                )
        except Exception as e:
            logger.warning(f"SHAP explainer creation failed: {e}")
            self.shap_explainer = None

    def _save_models(self) -> None:
        """Save trained models to disk."""

        os.makedirs(self.model_dir, exist_ok=True)

        joblib.dump(self.drafted_model, os.path.join(self.model_dir, 'drafted_model.joblib'))

        if self.round_model is not None:
            joblib.dump(self.round_model, os.path.join(self.model_dir, 'round_model.joblib'))

        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))

        with open(os.path.join(self.model_dir, 'feature_names.json'), 'w') as f:
            json.dump(self.feature_names, f)

        with open(os.path.join(self.model_dir, 'metrics.json'), 'w') as f:
            json.dump(self._convert_to_json(self.metrics), f, indent=2)

        logger.info(f"Models saved to {self.model_dir}")

    def _generate_training_report(self, results: Dict[str, Any]) -> str:
        """Generate training report."""

        lines = []
        lines.append("=" * 70)
        lines.append("DRAFT PROJECTION MODEL TRAINING REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # Classification results
        lines.append("DRAFT CLASSIFICATION (Drafted vs Undrafted)")
        lines.append("-" * 40)
        for name, metrics in results['classification']['models'].items():
            marker = "✓" if name == results['classification']['best_model'] else " "
            lines.append(f"\n{marker} {name.upper()}:")
            lines.append(f"    AUC-ROC: {metrics['auc_roc']:.3f}")
            lines.append(f"    Accuracy: {metrics['accuracy']:.3f}")
            lines.append(f"    F1 Score: {metrics['f1']:.3f}")

        lines.append(f"\n✓ Best classifier: {results['classification']['best_model'].upper()}")
        lines.append("")

        # Round prediction results
        if 'error' not in results.get('round_prediction', {}):
            lines.append("ROUND/PICK PREDICTION (Drafted Players Only)")
            lines.append("-" * 40)
            for name, metrics in results['round_prediction']['models'].items():
                marker = "✓" if name == results['round_prediction']['best_model'] else " "
                lines.append(f"\n{marker} {name.upper()}:")
                lines.append(f"    Pick MAE: {metrics['pick_mae']:.1f}")
                lines.append(f"    Pick RMSE: {metrics['pick_rmse']:.1f}")
                lines.append(f"    Round Accuracy: {metrics['round_accuracy']:.3f}")
                lines.append(f"    Within 1 Round: {metrics['within_1_round_accuracy']:.3f}")
                lines.append(f"    Mean Round Error: {metrics['mean_round_error']:.2f}")

            lines.append(f"\n✓ Best round predictor: {results['round_prediction']['best_model'].upper()}")
        else:
            lines.append("ROUND PREDICTION: Skipped (insufficient drafted samples)")

        lines.append("")

        # Feature importance
        if hasattr(self.drafted_model, 'feature_importances_'):
            lines.append("TOP 15 FEATURE IMPORTANCES (Draft Classification)")
            lines.append("-" * 40)
            importances = self.drafted_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            for i, idx in enumerate(indices):
                lines.append(f"  {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _save_training_report(self, report: str) -> None:
        """Save training report."""

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'draft_projection_training_report.txt')

        with open(path, 'w') as f:
            f.write(report)

        print(report)
        logger.info(f"Report saved to {path}")

    def predict(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict]
    ) -> Dict[str, Any]:
        """
        Predict draft outcome for a player.

        Args:
            player_features: Player feature data

        Returns:
            Comprehensive draft prediction dictionary
        """
        if self.drafted_model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare input
        X = self._prepare_prediction_input(player_features)
        X_scaled = self.scaler.transform(X)

        # Get player info
        player_name = self._get_player_name(player_features)
        position = self._get_position(player_features)

        # Draft probability
        draft_prob = self.drafted_model.predict_proba(X_scaled)[0, 1]
        will_be_drafted = draft_prob >= 0.5

        # Round/pick prediction
        if will_be_drafted and self.round_model is not None:
            predicted_pick = self.round_model.predict(X_scaled)[0]
            predicted_pick = int(np.clip(predicted_pick, 1, 260))
            projected_round = pick_to_round(predicted_pick)

            # Calculate pick range (early/mid/late within predicted round)
            round_start, round_end = round_to_pick_range(projected_round)
            round_span = round_end - round_start
            pick_range = {
                'early': max(round_start, int(predicted_pick - round_span * 0.3)),
                'mid': predicted_pick,
                'late': min(round_end, int(predicted_pick + round_span * 0.3)),
            }
        else:
            predicted_pick = 0
            projected_round = 0
            pick_range = {'early': 0, 'mid': 0, 'late': 0}

        # Get comparable draft picks
        comparables = self._find_comparable_picks(X_scaled, player_features)

        # Calculate rookie contract
        rookie_contract = self._calculate_rookie_contract(projected_round, predicted_pick)

        # Project career earnings
        career_earnings = self._project_career_earnings(projected_round, predicted_pick, position)

        # Get draft stock factors
        stock_factors = self._analyze_draft_stock_factors(player_features, X_scaled)

        # SHAP explanation
        shap_explanation = self._get_shap_explanation(X_scaled)

        return {
            'player_name': player_name,
            'will_be_drafted': will_be_drafted,
            'draft_probability': round(draft_prob, 3),
            'projected_round': projected_round,
            'projected_pick_range': pick_range,
            'comparable_draft_picks': comparables,
            'projected_rookie_contract': rookie_contract,
            'projected_career_earnings_8yr': career_earnings,
            'draft_stock_factors': stock_factors,
            'shap_explanation': shap_explanation,
        }

    def _prepare_prediction_input(
        self,
        features: Union[pd.Series, pd.DataFrame, Dict]
    ) -> np.ndarray:
        """Convert features to model input array."""

        if isinstance(features, dict):
            features = pd.Series(features)
        elif isinstance(features, pd.DataFrame):
            if len(features) != 1:
                raise ValueError("DataFrame must have exactly 1 row")
            features = features.iloc[0]

        X = np.zeros((1, len(self.feature_names)))
        for i, feat in enumerate(self.feature_names):
            if feat in features.index:
                val = features[feat]
                X[0, i] = float(val) if pd.notna(val) else 0.0

        return X.astype(np.float32)

    def _get_player_name(self, features: Union[pd.Series, pd.DataFrame, Dict]) -> str:
        """Extract player name."""

        if isinstance(features, dict):
            features = pd.Series(features)
        elif isinstance(features, pd.DataFrame):
            features = features.iloc[0]

        for col in ['player_name_std', 'player_name', 'player_name_original', 'name']:
            if col in features.index and pd.notna(features[col]):
                return str(features[col])
        return 'Unknown'

    def _get_position(self, features: Union[pd.Series, pd.DataFrame, Dict]) -> str:
        """Extract position."""

        if isinstance(features, dict):
            features = pd.Series(features)
        elif isinstance(features, pd.DataFrame):
            features = features.iloc[0]

        for col in ['position_group', 'position']:
            if col in features.index and pd.notna(features[col]):
                return str(features[col])
        return 'ATH'

    def _find_comparable_picks(
        self,
        X_scaled: np.ndarray,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        n_comparables: int = 3
    ) -> List[Dict[str, Any]]:
        """Find similar historical draft picks."""

        if self.training_data is None or len(self.training_data) == 0:
            return []

        try:
            # Only look at drafted players
            drafted_data = self.training_data[self.training_data['was_drafted'] == 1].copy()
            if len(drafted_data) == 0:
                return []

            # Prepare training features
            X_train = drafted_data[self.feature_names].values.astype(np.float32)
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_train_scaled = self.scaler.transform(X_train)

            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=min(n_comparables, len(X_train)), metric='euclidean')
            nn.fit(X_train_scaled)
            distances, indices = nn.kneighbors(X_scaled)

            # Build comparable list
            comparables = []
            for dist, idx in zip(distances[0], indices[0]):
                row = drafted_data.iloc[idx]

                name = row.get('player_name_std', row.get('player_name', f'Player {idx}'))
                school = row.get('school_name', 'Unknown')
                year = row.get('draft_year', 0)
                pick = row.get('draft_pick', 0)

                similarity = max(0, 1 - dist / 10)

                comparables.append({
                    'name': str(name),
                    'college': str(school),
                    'year': int(year) if pd.notna(year) else 0,
                    'pick': int(pick) if pd.notna(pick) else 0,
                    'similarity': round(similarity, 3),
                })

            return comparables

        except Exception as e:
            logger.warning(f"Error finding comparables: {e}")
            return []

    def _calculate_rookie_contract(self, round_num: int, pick: int) -> Dict[str, Any]:
        """Calculate projected rookie contract."""

        if round_num == 0 or pick == 0:
            return {
                'total_value': 0,
                'signing_bonus': 0,
                'years': 0,
                'fifth_year_option_eligible': False,
            }

        # Find applicable wage scale
        round_scale = ROOKIE_WAGE_SCALE.get(round_num, ROOKIE_WAGE_SCALE[7])

        for (low, high), contract in round_scale.items():
            if low <= pick <= high:
                return {
                    'total_value': contract['total'],
                    'signing_bonus': contract['signing'],
                    'years': 4,
                    'fifth_year_option_eligible': contract['fifth_year'],
                }

        # Default to round 7
        return {
            'total_value': 3_700_000,
            'signing_bonus': 125_000,
            'years': 4,
            'fifth_year_option_eligible': False,
        }

    def _project_career_earnings(
        self,
        round_num: int,
        pick: int,
        position: str
    ) -> int:
        """Project 8-year career earnings."""

        if round_num == 0 or pick == 0:
            return 0

        # Base earnings by pick
        base_earnings = 0
        for (low, high), earnings in CAREER_EARNINGS_BY_PICK.items():
            if low <= pick <= high:
                base_earnings = earnings
                break

        if base_earnings == 0:
            base_earnings = 5_000_000  # Default for late picks

        # Apply position multiplier
        multiplier = POSITION_EARNINGS_MULTIPLIER.get(position, 1.0)
        adjusted_earnings = int(base_earnings * multiplier)

        return adjusted_earnings

    def _analyze_draft_stock_factors(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        X_scaled: np.ndarray
    ) -> Dict[str, List[str]]:
        """Analyze factors helping and hurting draft stock."""

        if isinstance(player_features, pd.DataFrame):
            player_features = player_features.iloc[0]
        if isinstance(player_features, dict):
            player_features = pd.Series(player_features)

        helps = []
        hurts = []

        # Production
        production = player_features.get('production_score', 50)
        if production >= 80:
            helps.append(STOCK_FACTOR_DESCRIPTIONS['elite_production'])
        elif production >= 65:
            helps.append("Strong on-field production")
        elif production < 40:
            hurts.append("Below-average production")

        # Conference
        conf_tier = player_features.get('conference_tier', 2)
        if conf_tier >= 3:
            helps.append(STOCK_FACTOR_DESCRIPTIONS['power_conference'])
        elif conf_tier == 1:
            hurts.append(STOCK_FACTOR_DESCRIPTIONS['weak_competition'])

        # Age
        age = player_features.get('draft_age', 22)
        if age <= 21:
            helps.append(STOCK_FACTOR_DESCRIPTIONS['young_age'])
        elif age >= 24:
            hurts.append(STOCK_FACTOR_DESCRIPTIONS['older_age'])

        # Measurables
        ras_score = player_features.get('ras_score', 50)
        if ras_score >= 75:
            helps.append(STOCK_FACTOR_DESCRIPTIONS['good_measurables'])
        elif ras_score < 40:
            hurts.append(STOCK_FACTOR_DESCRIPTIONS['below_avg_measurables'])

        # Recruiting
        stars = player_features.get('recruiting_stars', 3)
        if stars >= 5:
            helps.append(STOCK_FACTOR_DESCRIPTIONS['high_recruit'])
        elif stars >= 4:
            helps.append("Strong recruiting pedigree")

        # Experience
        years = player_features.get('years_in_college', 3)
        if years <= 3:
            helps.append(STOCK_FACTOR_DESCRIPTIONS['early_declare'])
        if player_features.get('career_starts', 0) < 15:
            hurts.append(STOCK_FACTOR_DESCRIPTIONS['one_year_starter'])
        elif player_features.get('career_starts', 0) >= 30:
            helps.append(STOCK_FACTOR_DESCRIPTIONS['multiple_years'])

        # Production trend
        trend = player_features.get('production_trend', 0)
        if trend > 10:
            helps.append(STOCK_FACTOR_DESCRIPTIONS['rising_production'])
        elif trend < -10:
            hurts.append(STOCK_FACTOR_DESCRIPTIONS['declining_production'])

        # Team success
        team_wins = player_features.get('team_wins', 6)
        if team_wins >= 10:
            helps.append(STOCK_FACTOR_DESCRIPTIONS['team_success'])

        return {
            'helps': helps[:5],  # Top 5
            'hurts': hurts[:5],
        }

    def _get_shap_explanation(self, X_scaled: np.ndarray) -> Dict[str, float]:
        """Get SHAP-based explanation."""

        if not HAS_SHAP or self.shap_explainer is None:
            # Fallback to feature importance
            if hasattr(self.drafted_model, 'feature_importances_'):
                importances = self.drafted_model.feature_importances_
                contributions = importances * X_scaled[0]

                indices = np.argsort(np.abs(contributions))[::-1][:5]
                explanation = {}
                for idx in indices:
                    feat_name = self.feature_names[idx]
                    explanation[feat_name] = round(contributions[idx], 4)
                return explanation
            return {}

        try:
            shap_vals = self.shap_explainer.shap_values(X_scaled)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            indices = np.argsort(np.abs(shap_vals[0]))[::-1][:5]
            explanation = {}
            for idx in indices:
                feat_name = self.feature_names[idx]
                explanation[feat_name] = round(shap_vals[0][idx], 4)
            return explanation

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {}

    def project_nil_from_draft_value(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict]
    ) -> Dict[str, Any]:
        """
        Project NIL value based on draft potential and career earnings.

        Logic: Agents/families use projected NFL earnings as leverage in NIL negotiations.

        Args:
            player_features: Player feature data

        Returns:
            Dictionary with NIL recommendations based on draft value
        """
        # Get draft projection
        draft_pred = self.predict(player_features)

        player_name = draft_pred['player_name']
        career_earnings = draft_pred['projected_career_earnings_8yr']
        projected_round = draft_pred['projected_round']
        draft_prob = draft_pred['draft_probability']

        # Calculate recommended NIL based on draft value
        # Logic: NIL should be ~2-4% of projected career earnings for top prospects
        # Lower percentages for lower draft prospects

        if projected_round == 1:
            nil_pct = 0.03  # 3% of career earnings
            nil_tier = 'mega'
        elif projected_round == 2:
            nil_pct = 0.025  # 2.5%
            nil_tier = 'premium'
        elif projected_round <= 4:
            nil_pct = 0.02  # 2%
            nil_tier = 'solid'
        elif projected_round <= 6:
            nil_pct = 0.015  # 1.5%
            nil_tier = 'moderate'
        else:
            nil_pct = 0.01  # 1%
            nil_tier = 'entry'

        # Adjust for draft probability
        if draft_prob < 0.5:
            nil_pct *= 0.5

        recommended_nil = int(career_earnings * nil_pct)

        # Calculate negotiation rationale
        if projected_round == 1:
            rationale = f"Projects as Round {projected_round} pick worth ~${career_earnings/1e6:.0f}M in career earnings. NIL should reflect top-tier status."
        elif projected_round <= 3:
            rationale = f"Projects as Day 2 pick (Round {projected_round}) with ~${career_earnings/1e6:.0f}M career earnings potential. Strong NIL warranted."
        elif projected_round <= 5:
            rationale = f"Projects as mid-round pick (Round {projected_round}). Moderate NIL investment appropriate."
        elif draft_prob > 0.5:
            rationale = f"Projects as late-round pick (Round {projected_round}). Entry-level NIL appropriate."
        else:
            rationale = "Draft projection uncertain. NIL should be based primarily on current college production."

        return {
            'player_name': player_name,
            'projected_round': projected_round,
            'draft_probability': draft_prob,
            'projected_career_earnings': career_earnings,
            'recommended_nil_value': recommended_nil,
            'recommended_nil_tier': nil_tier,
            'nil_range': {
                'minimum': int(recommended_nil * 0.6),
                'recommended': recommended_nil,
                'maximum': int(recommended_nil * 1.5),
            },
            'negotiation_rationale': rationale,
        }

    def generate_mock_draft(
        self,
        all_player_features: pd.DataFrame,
        n_rounds: int = 7
    ) -> pd.DataFrame:
        """
        Generate a mock draft board.

        Args:
            all_player_features: DataFrame of all draft-eligible players
            n_rounds: Number of rounds to simulate (default 7)

        Returns:
            DataFrame with mock draft rankings
        """
        logger.info(f"Generating mock draft for {len(all_player_features)} players...")

        results = []

        for idx, row in all_player_features.iterrows():
            try:
                pred = self.predict(row)

                results.append({
                    'player_name': pred['player_name'],
                    'school': row.get('school_name', row.get('school', 'Unknown')),
                    'position': row.get('position_group', row.get('position', 'ATH')),
                    'draft_probability': pred['draft_probability'],
                    'projected_round': pred['projected_round'],
                    'projected_pick': pred['projected_pick_range']['mid'],
                    'production_score': row.get('production_score', 50),
                    'ras_score': row.get('ras_score', 50),
                    'draft_value': self._calculate_draft_value(
                        pred['draft_probability'],
                        pred['projected_pick_range']['mid']
                    ),
                })
            except Exception as e:
                logger.warning(f"Failed to predict for player {idx}: {e}")

        # Create DataFrame
        mock_df = pd.DataFrame(results)

        if len(mock_df) == 0:
            return mock_df

        # Sort by draft value (higher = better prospect)
        mock_df = mock_df.sort_values('draft_value', ascending=False).reset_index(drop=True)

        # Assign mock draft rank and pick
        mock_df['rank'] = range(1, len(mock_df) + 1)

        # Filter to draftable players
        max_picks = n_rounds * 32
        draftable = mock_df[mock_df['draft_probability'] >= 0.3].head(max_picks).copy()
        draftable['mock_pick'] = range(1, len(draftable) + 1)
        draftable['mock_round'] = draftable['mock_pick'].apply(pick_to_round)

        # Save
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, 'model_mock_draft.csv')
        draftable.to_csv(report_path, index=False)
        logger.info(f"Mock draft saved to {report_path}")

        # Also save full big board
        board_path = os.path.join(self.output_dir, 'model_big_board.csv')
        mock_df.to_csv(board_path, index=False)
        logger.info(f"Big board saved to {board_path}")

        return draftable

    def _calculate_draft_value(self, draft_prob: float, projected_pick: int) -> float:
        """Calculate draft value score for ranking."""

        if draft_prob < 0.3:
            return draft_prob * 50

        # Higher value for earlier picks
        if projected_pick <= 0:
            pick_value = 0
        elif projected_pick <= 10:
            pick_value = 100 - projected_pick
        elif projected_pick <= 32:
            pick_value = 80 - (projected_pick - 10) * 1.5
        elif projected_pick <= 100:
            pick_value = 50 - (projected_pick - 32) * 0.5
        else:
            pick_value = max(10, 30 - (projected_pick - 100) * 0.2)

        return draft_prob * pick_value

    def draft_stock_tracker(
        self,
        player_name: str,
        season_stats_by_week: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Track draft stock over time based on weekly/monthly performance.

        Args:
            player_name: Player name
            season_stats_by_week: List of stat dictionaries by week
                Each dict should have 'week' and cumulative stats

        Returns:
            List of projection snapshots over time
        """
        if self.drafted_model is None:
            raise ValueError("Model not trained. Call train() first.")

        snapshots = []

        for week_stats in season_stats_by_week:
            week_num = week_stats.get('week', len(snapshots) + 1)

            try:
                # Get prediction for this week's cumulative stats
                pred = self.predict(week_stats)

                snapshot = {
                    'week': week_num,
                    'date': week_stats.get('date', f'Week {week_num}'),
                    'draft_probability': pred['draft_probability'],
                    'projected_round': pred['projected_round'],
                    'projected_pick': pred['projected_pick_range']['mid'],
                    'production_score': week_stats.get('production_score', 50),
                    'key_events': week_stats.get('events', []),
                }

                # Compare to previous
                if snapshots:
                    prev = snapshots[-1]
                    round_change = prev['projected_round'] - snapshot['projected_round']

                    if round_change > 0:
                        snapshot['trend'] = 'rising'
                        snapshot['change_note'] = f"Stock up (was Round {prev['projected_round']})"
                    elif round_change < 0:
                        snapshot['trend'] = 'falling'
                        snapshot['change_note'] = f"Stock down (was Round {prev['projected_round']})"
                    else:
                        snapshot['trend'] = 'stable'
                        snapshot['change_note'] = "Stock steady"
                else:
                    snapshot['trend'] = 'initial'
                    snapshot['change_note'] = "Initial projection"

                snapshots.append(snapshot)

            except Exception as e:
                logger.warning(f"Failed to project for week {week_num}: {e}")
                snapshots.append({
                    'week': week_num,
                    'error': str(e),
                })

        # Save tracker
        os.makedirs(self.output_dir, exist_ok=True)
        safe_name = player_name.lower().replace(' ', '_')[:30]
        tracker_path = os.path.join(self.output_dir, f'{safe_name}_draft_stock_tracker.json')
        with open(tracker_path, 'w') as f:
            json.dump({
                'player_name': player_name,
                'snapshots': snapshots,
            }, f, indent=2)

        return snapshots

    def load_models(self) -> None:
        """Load previously trained models."""

        self.drafted_model = joblib.load(
            os.path.join(self.model_dir, 'drafted_model.joblib')
        )
        self.scaler = joblib.load(
            os.path.join(self.model_dir, 'scaler.joblib')
        )

        round_path = os.path.join(self.model_dir, 'round_model.joblib')
        if os.path.exists(round_path):
            self.round_model = joblib.load(round_path)

        with open(os.path.join(self.model_dir, 'feature_names.json'), 'r') as f:
            self.feature_names = json.load(f)

        logger.info("Models loaded successfully")

    def _convert_to_json(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Draft Projector - Standalone Mode")
    print("=" * 50)

    # Generate sample data
    print("\nGenerating sample draft data...")
    np.random.seed(42)
    n_samples = 300

    # Generate features
    data = {
        'player_name_std': [f'player_{i}' for i in range(n_samples)],
        'school_name': np.random.choice(
            ['Alabama', 'Ohio State', 'Georgia', 'Texas', 'USC', 'Michigan', 'Oregon', 'UCF', 'Boise State'],
            n_samples
        ),
        'position_group': np.random.choice(
            ['QB', 'WR', 'RB', 'TE', 'OL', 'EDGE', 'DL', 'LB', 'CB', 'S'],
            n_samples
        ),
        'production_score': np.random.uniform(20, 95, n_samples),
        'school_tier': np.random.choice([6, 5, 4, 3, 2, 1], n_samples, p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05]),
        'conference_tier': np.random.choice([3, 2, 1], n_samples, p=[0.4, 0.35, 0.25]),
        'recruiting_stars': np.random.choice([5, 4, 3, 2, 0], n_samples, p=[0.05, 0.15, 0.4, 0.3, 0.1]),
        'recruiting_composite': np.random.uniform(0.7, 1.0, n_samples),
        'draft_age': np.random.uniform(20.5, 24.5, n_samples),
        'years_in_college': np.random.choice([3, 4, 5], n_samples, p=[0.4, 0.4, 0.2]),
        'career_starts': np.random.randint(10, 50, n_samples),
        'production_trend': np.random.uniform(-20, 30, n_samples),
        'ras_score': np.random.uniform(30, 95, n_samples),
        'forty_actual': np.random.uniform(4.3, 5.2, n_samples),
        'height_inches': np.random.uniform(68, 78, n_samples),
        'weight_lbs': np.random.uniform(180, 320, n_samples),
        'team_wins': np.random.randint(4, 14, n_samples),
    }

    # Generate draft outcomes based on features
    draft_score = (
        data['production_score'] * 0.4 +
        data['school_tier'] * 8 +
        data['recruiting_stars'] * 10 +
        data['ras_score'] * 0.2 +
        (100 - data['draft_age'] * 3.5) +
        np.random.normal(0, 10, n_samples)
    )

    data['was_drafted'] = (draft_score > 70).astype(int)

    # Generate pick for drafted players
    picks = []
    for i, drafted in enumerate(data['was_drafted']):
        if drafted:
            # Higher score = earlier pick
            base_pick = max(1, 261 - int(draft_score[i] * 2.5))
            pick = max(1, min(260, base_pick + np.random.randint(-20, 20)))
            picks.append(pick)
        else:
            picks.append(0)

    data['draft_pick'] = picks
    data['draft_round'] = [pick_to_round(p) if p > 0 else 0 for p in picks]

    df = pd.DataFrame(data)

    print(f"Generated {len(df)} players")
    print(f"Drafted: {df['was_drafted'].sum()} ({df['was_drafted'].mean():.1%})")

    # Initialize and train
    projector = DraftProjector(
        model_dir="models/draft_projection",
        output_dir="outputs/reports"
    )

    try:
        print("\n" + "=" * 50)
        print("TRAINING")
        print("=" * 50)

        results = projector.train(df)

        # Test prediction
        print("\n" + "=" * 50)
        print("SAMPLE PREDICTION")
        print("=" * 50)

        test_player = df[df['was_drafted'] == 1].iloc[0]
        prediction = projector.predict(test_player)

        print(f"\nPlayer: {prediction['player_name']}")
        print(f"Will be drafted: {prediction['will_be_drafted']} ({prediction['draft_probability']:.0%})")
        print(f"Projected Round: {prediction['projected_round']}")
        print(f"Projected Pick Range: {prediction['projected_pick_range']}")
        print(f"\nRookie Contract: ${prediction['projected_rookie_contract']['total_value']:,}")
        print(f"Career Earnings (8yr): ${prediction['projected_career_earnings_8yr']:,}")
        print(f"\nStock Helps: {prediction['draft_stock_factors']['helps'][:3]}")
        print(f"Stock Hurts: {prediction['draft_stock_factors']['hurts'][:3]}")

        # Test NIL from draft
        print("\n" + "=" * 50)
        print("NIL FROM DRAFT VALUE")
        print("=" * 50)

        nil_rec = projector.project_nil_from_draft_value(test_player)
        print(f"\nRecommended NIL: ${nil_rec['recommended_nil_value']:,}")
        print(f"NIL Tier: {nil_rec['recommended_nil_tier']}")
        print(f"Rationale: {nil_rec['negotiation_rationale']}")

        # Generate mock draft
        print("\n" + "=" * 50)
        print("MOCK DRAFT")
        print("=" * 50)

        mock = projector.generate_mock_draft(df, n_rounds=3)
        print(f"\nMock draft generated: {len(mock)} picks")
        print("\nTop 10 picks:")
        print(mock[['mock_pick', 'mock_round', 'player_name', 'position', 'projected_round']].head(10).to_string())

        print("\n✓ Draft Projector test complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
