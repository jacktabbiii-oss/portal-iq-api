"""
Transfer Portal Prediction Model

Predicts transfer portal flight risk and portal fit for college athletes.
Helps teams with retention strategy and portal acquisition planning.

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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.neighbors import NearestNeighbors
import joblib

# Handle imbalanced data
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    warnings.warn("imbalanced-learn not available. Install with: pip install imbalanced-learn")

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

# Risk factor descriptions
RISK_FACTOR_DESCRIPTIONS = {
    'coaching_change': "Head coach was replaced this offseason",
    'coordinator_change': "Offensive or defensive coordinator was replaced",
    'snap_pct': "Current playing time percentage",
    'snap_trend': "Playing time trend compared to last season",
    'depth_chart_position': "Position on the depth chart (1=starter, 2=backup, 3+=buried)",
    'production_vs_star_avg': "Performance vs. expectation based on recruiting ranking",
    'team_wins': "Team's win total this season",
    'team_win_trend': "Team's win trend compared to last season",
    'nil_vs_team_median': "NIL value compared to team median",
    'nil_could_increase_elsewhere': "Could earn more NIL at a higher-tier school",
    'incoming_recruits_at_position': "Highly-rated recruits signed at player's position",
    'position_depth': "Number of players at same position competing for playing time",
    'distance_from_home': "Distance from hometown to current school",
    'years_at_school': "Years spent at current school",
    'is_grad_transfer_eligible': "Eligible to transfer as graduate with immediate eligibility",
    'remaining_eligibility': "Years of eligibility remaining",
    'school_tier': "Current school's prestige tier",
    'conference_tier': "Current conference's competitiveness",
    'team_nil_tier': "School's NIL budget tier",
    'position_portal_rate': "Historical transfer rate for this position",
    'school_portal_rate': "Historical transfer rate for this school",
    'breakout_flag': "Player significantly outperformed expectations this season",
    'production_score': "Overall production rating (0-100)",
    'recruiting_stars': "Original recruiting star rating",
}

# Retention recommendation templates
RETENTION_RECOMMENDATIONS = {
    'nil_below_median': "Increase NIL package — player is below team median",
    'nil_could_increase': "Proactively match potential NIL offers from higher-tier programs",
    'coaching_change': "Ensure role clarity with new coaching staff",
    'snap_pct_low': "Discuss path to increased playing time or starting role",
    'snap_trend_negative': "Address declining playing time in one-on-one meeting",
    'incoming_competition': "Reassure player about their role despite new recruits at position",
    'depth_buried': "Be transparent about role — consider facilitating transfer if mutually beneficial",
    'production_high': "Player's production merits starter-level playing time and recognition",
    'far_from_home': "Emphasize team culture and support system; facilitate family visits",
    'team_losing': "Communicate improvement plans and player's importance to rebuild",
    'breakout_performer': "Capitalize on breakout season with enhanced NIL and leadership role",
    'grad_eligible': "Offer graduate assistant or other post-eligibility opportunities if applicable",
}

# School data for fit analysis
SCHOOL_DATA = {
    'Alabama': {'tier': 6, 'conference_tier': 3, 'market_size': 260, 'wins': 11, 'nil_tier': 5, 'style': 'pro'},
    'Ohio State': {'tier': 6, 'conference_tier': 3, 'market_size': 2150, 'wins': 12, 'nil_tier': 5, 'style': 'spread'},
    'Georgia': {'tier': 5, 'conference_tier': 3, 'market_size': 450, 'wins': 13, 'nil_tier': 5, 'style': 'pro'},
    'Texas': {'tier': 6, 'conference_tier': 3, 'market_size': 2300, 'wins': 12, 'nil_tier': 5, 'style': 'spread'},
    'USC': {'tier': 6, 'conference_tier': 3, 'market_size': 13000, 'wins': 8, 'nil_tier': 5, 'style': 'spread'},
    'Michigan': {'tier': 6, 'conference_tier': 3, 'market_size': 4400, 'wins': 13, 'nil_tier': 5, 'style': 'pro'},
    'Oregon': {'tier': 5, 'conference_tier': 3, 'market_size': 380, 'wins': 12, 'nil_tier': 4, 'style': 'spread'},
    'Penn State': {'tier': 5, 'conference_tier': 3, 'market_size': 160, 'wins': 10, 'nil_tier': 4, 'style': 'pro'},
    'LSU': {'tier': 5, 'conference_tier': 3, 'market_size': 870, 'wins': 10, 'nil_tier': 5, 'style': 'spread'},
    'Florida': {'tier': 5, 'conference_tier': 3, 'market_size': 330, 'wins': 7, 'nil_tier': 4, 'style': 'spread'},
    'Tennessee': {'tier': 5, 'conference_tier': 3, 'market_size': 900, 'wins': 9, 'nil_tier': 4, 'style': 'spread'},
    'Clemson': {'tier': 5, 'conference_tier': 2, 'market_size': 920, 'wins': 9, 'nil_tier': 4, 'style': 'spread'},
    'Notre Dame': {'tier': 6, 'conference_tier': 2, 'market_size': 320, 'wins': 10, 'nil_tier': 4, 'style': 'pro'},
    'Oklahoma': {'tier': 6, 'conference_tier': 3, 'market_size': 1450, 'wins': 10, 'nil_tier': 5, 'style': 'spread'},
    'Texas A&M': {'tier': 5, 'conference_tier': 3, 'market_size': 275, 'wins': 8, 'nil_tier': 5, 'style': 'pro'},
    'Miami': {'tier': 5, 'conference_tier': 2, 'market_size': 6200, 'wins': 7, 'nil_tier': 4, 'style': 'spread'},
    'Florida State': {'tier': 5, 'conference_tier': 2, 'market_size': 390, 'wins': 13, 'nil_tier': 4, 'style': 'pro'},
    'Wisconsin': {'tier': 4, 'conference_tier': 3, 'market_size': 680, 'wins': 7, 'nil_tier': 3, 'style': 'pro'},
    'Iowa': {'tier': 4, 'conference_tier': 3, 'market_size': 175, 'wins': 8, 'nil_tier': 3, 'style': 'pro'},
    'UCLA': {'tier': 4, 'conference_tier': 3, 'market_size': 13000, 'wins': 8, 'nil_tier': 3, 'style': 'spread'},
    'Colorado': {'tier': 4, 'conference_tier': 2, 'market_size': 2900, 'wins': 9, 'nil_tier': 4, 'style': 'spread'},
    'Ole Miss': {'tier': 4, 'conference_tier': 3, 'market_size': 175, 'wins': 11, 'nil_tier': 4, 'style': 'spread'},
    'Missouri': {'tier': 4, 'conference_tier': 3, 'market_size': 180, 'wins': 11, 'nil_tier': 3, 'style': 'spread'},
    'UCF': {'tier': 3, 'conference_tier': 2, 'market_size': 2700, 'wins': 6, 'nil_tier': 2, 'style': 'spread'},
    'Cincinnati': {'tier': 3, 'conference_tier': 2, 'market_size': 2250, 'wins': 5, 'nil_tier': 2, 'style': 'spread'},
    'Boise State': {'tier': 2, 'conference_tier': 1, 'market_size': 780, 'wins': 8, 'nil_tier': 1, 'style': 'spread'},
    'Memphis': {'tier': 2, 'conference_tier': 1, 'market_size': 1350, 'wins': 10, 'nil_tier': 1, 'style': 'spread'},
}

# NIL budget estimates by tier (annual)
NIL_BUDGET_BY_TIER = {
    5: 20_000_000,  # Blue bloods
    4: 12_000_000,  # Elite
    3: 6_000_000,   # Power brand
    2: 2_000_000,   # P4 mid
    1: 500_000,     # G5 strong
    0: 100_000,     # G5
}

# Position value for roster impact calculation
POSITION_WIN_IMPACT = {
    'QB': 2.5,    # Starting QB worth ~2.5 wins
    'EDGE': 1.2,  # Elite pass rusher
    'WR': 0.8,
    'OL': 0.7,
    'CB': 0.7,
    'RB': 0.5,
    'TE': 0.5,
    'LB': 0.6,
    'DL': 0.6,
    'S': 0.5,
}


class PortalPredictor:
    """
    Transfer portal prediction model for flight risk and fit analysis.

    Features:
    - Flight risk prediction (will player enter portal?)
    - Team-wide flight risk reports
    - Portal fit prediction (how well does player fit school?)
    - Portal target ranking for schools
    - Destination ranking for portal players
    """

    def __init__(
        self,
        model_dir: str = "models/portal_prediction",
        output_dir: str = "outputs/reports"
    ):
        """
        Initialize the portal predictor.

        Args:
            model_dir: Directory to save trained models
            output_dir: Directory to save reports
        """
        self.model_dir = model_dir
        self.output_dir = output_dir

        # Models
        self.flight_risk_model = None
        self.fit_model = None
        self.scaler_flight = None
        self.scaler_fit = None

        # Training data reference
        self.flight_risk_features = []
        self.fit_features = []
        self.training_data_flight = None
        self.training_data_fit = None

        # SHAP
        self.shap_explainer_flight = None

        # Metrics
        self.flight_risk_metrics = {}
        self.fit_metrics = {}

    # =========================================================================
    # FLIGHT RISK MODEL
    # =========================================================================

    def train_flight_risk(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train flight risk prediction model.

        Args:
            features_df: DataFrame with player features and entered_portal target

        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Training flight risk model...")

        df = features_df.copy()

        # Validate
        if 'entered_portal' not in df.columns:
            raise ValueError("'entered_portal' target column required")

        # Prepare data
        X, y, feature_names = self._prepare_flight_risk_data(df)
        self.flight_risk_features = feature_names
        self.training_data_flight = df.copy()

        # Temporal split if season column exists
        if 'season' in df.columns:
            X_train, X_test, y_train, y_test = self._temporal_split(df, X, y)
            split_method = "temporal"
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            split_method = "stratified"

        logger.info(f"Split method: {split_method}")
        logger.info(f"Training: {len(y_train)} samples, Test: {len(y_test)} samples")
        logger.info(f"Class balance - Train: {y_train.mean():.2%} positive, Test: {y_test.mean():.2%} positive")

        # Scale features
        self.scaler_flight = StandardScaler()
        X_train_scaled = self.scaler_flight.fit_transform(X_train)
        X_test_scaled = self.scaler_flight.transform(X_test)

        # Handle class imbalance
        if HAS_IMBLEARN and y_train.mean() < 0.3:
            logger.info("Applying SMOTE for class imbalance...")
            try:
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
                logger.info(f"After SMOTE: {len(y_train_resampled)} samples, {y_train_resampled.mean():.2%} positive")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}, using original data with class_weight")
                X_train_resampled, y_train_resampled = X_train_scaled, y_train
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train

        # Train models
        results = self._train_flight_risk_models(
            X_train_resampled, y_train_resampled,
            X_test_scaled, y_test
        )

        # Calculate SHAP values
        if HAS_SHAP:
            logger.info("Calculating SHAP values...")
            self._calculate_flight_risk_shap(X_train_scaled, feature_names)

        # Save
        self._save_flight_risk_models()
        self.flight_risk_metrics = results

        # Generate report
        report = self._generate_flight_risk_report(results, split_method)
        self._save_flight_risk_report(report)

        logger.info("Flight risk model training complete!")
        return results

    def _prepare_flight_risk_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for flight risk model."""

        # Exclude metadata and target
        exclude_cols = [
            'entered_portal', 'player_name_std', 'player_name', 'name',
            'school_name', 'school', 'position_group', 'position_raw', 'season'
        ]

        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

        X = df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)

        y = df['entered_portal'].values.astype(int)

        return X, y, feature_cols

    def _temporal_split(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split by time - train on earlier years, test on most recent."""

        max_season = df['season'].max()
        train_mask = df['season'] < max_season
        test_mask = df['season'] == max_season

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        return X_train, X_test, y_train, y_test

    def _train_flight_risk_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train and evaluate flight risk classification models."""

        results = {'models': {}}

        # Define models
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
            # Calculate scale_pos_weight for imbalance
            scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                scale_pos_weight=scale_pos, random_state=42,
                n_jobs=-1, verbosity=0, use_label_encoder=False,
                eval_metric='logloss'
            )

        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                class_weight='balanced', random_state=42,
                n_jobs=-1, verbose=-1
            )

        # Train and evaluate
        best_model = None
        best_recall = 0  # Focus on recall for positive class
        best_name = None

        for name, model in models.items():
            logger.info(f"  Training {name}...")

            # Fit
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Metrics
            auc_roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Precision-Recall AUC
            if len(np.unique(y_test)) > 1:
                prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = auc(rec_curve, prec_curve)
            else:
                pr_auc = 0

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            results['models'][name] = {
                'auc_roc': auc_roc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pr_auc': pr_auc,
                'confusion_matrix': cm.tolist(),
            }

            logger.info(f"    AUC-ROC: {auc_roc:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, F1: {f1:.3f}")

            # Track best by recall (we want to catch transfers)
            if recall > best_recall or (recall == best_recall and f1 > results['models'].get(best_name, {}).get('f1', 0)):
                best_recall = recall
                best_model = model
                best_name = name

        self.flight_risk_model = best_model
        results['best_model'] = best_name
        results['best_metrics'] = results['models'][best_name]

        return results

    def _calculate_flight_risk_shap(
        self,
        X_train: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Calculate SHAP values for flight risk model."""

        try:
            if hasattr(self.flight_risk_model, 'feature_importances_'):
                self.shap_explainer_flight = shap.TreeExplainer(self.flight_risk_model)
            else:
                background = X_train[:min(100, len(X_train))]
                self.shap_explainer_flight = shap.KernelExplainer(
                    self.flight_risk_model.predict_proba, background
                )
        except Exception as e:
            logger.warning(f"SHAP explainer creation failed: {e}")
            self.shap_explainer_flight = None

    def _save_flight_risk_models(self) -> None:
        """Save flight risk models to disk."""

        os.makedirs(self.model_dir, exist_ok=True)

        joblib.dump(
            self.flight_risk_model,
            os.path.join(self.model_dir, 'flight_risk_model.joblib')
        )
        joblib.dump(
            self.scaler_flight,
            os.path.join(self.model_dir, 'scaler_flight.joblib')
        )

        with open(os.path.join(self.model_dir, 'flight_risk_features.json'), 'w') as f:
            json.dump(self.flight_risk_features, f)

        with open(os.path.join(self.model_dir, 'flight_risk_metrics.json'), 'w') as f:
            json.dump(self._convert_to_json(self.flight_risk_metrics), f, indent=2)

        logger.info(f"Flight risk models saved to {self.model_dir}")

    def _generate_flight_risk_report(self, results: Dict[str, Any], split_method: str) -> str:
        """Generate flight risk training report."""

        lines = []
        lines.append("=" * 70)
        lines.append("FLIGHT RISK MODEL TRAINING REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Split method: {split_method}")
        lines.append(f"Features: {len(self.flight_risk_features)}")
        lines.append("")

        lines.append("MODEL RESULTS")
        lines.append("-" * 40)

        for name, metrics in results['models'].items():
            marker = "✓" if name == results['best_model'] else " "
            lines.append(f"\n{marker} {name.upper()}:")
            lines.append(f"    AUC-ROC: {metrics['auc_roc']:.3f}")
            lines.append(f"    Precision: {metrics['precision']:.3f}")
            lines.append(f"    Recall: {metrics['recall']:.3f}")
            lines.append(f"    F1 Score: {metrics['f1']:.3f}")
            lines.append(f"    PR-AUC: {metrics['pr_auc']:.3f}")
            lines.append(f"    Confusion Matrix: {metrics['confusion_matrix']}")

        lines.append(f"\n✓ Best model selected: {results['best_model'].upper()}")
        lines.append("  Selection criteria: Highest recall (catching transfers is priority)")
        lines.append("")

        # Feature importance
        if hasattr(self.flight_risk_model, 'feature_importances_'):
            lines.append("TOP 15 FEATURE IMPORTANCES")
            lines.append("-" * 40)
            importances = self.flight_risk_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            for i, idx in enumerate(indices):
                lines.append(f"  {i+1}. {self.flight_risk_features[idx]}: {importances[idx]:.4f}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _save_flight_risk_report(self, report: str) -> None:
        """Save flight risk training report."""

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'flight_risk_training_report.txt')

        with open(path, 'w') as f:
            f.write(report)

        print(report)
        logger.info(f"Report saved to {path}")

    def predict_flight_risk(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict]
    ) -> Dict[str, Any]:
        """
        Predict flight risk for a single player.

        Args:
            player_features: Player feature data

        Returns:
            Dictionary with flight risk prediction and analysis
        """
        if self.flight_risk_model is None:
            raise ValueError("Model not trained. Call train_flight_risk() first.")

        # Prepare input
        X = self._prepare_prediction_input(player_features, self.flight_risk_features)
        X_scaled = self.scaler_flight.transform(X)

        # Get prediction
        prob = self.flight_risk_model.predict_proba(X_scaled)[0, 1]
        flight_risk_score = prob * 100

        # Determine risk level
        if flight_risk_score >= 60:
            risk_level = 'high'
        elif flight_risk_score >= 30:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Get player name
        player_name = self._get_player_name(player_features)

        # Get top risk factors
        top_risk_factors = self._get_top_risk_factors(X_scaled, player_features)

        # Get retention recommendations
        retention_recommendations = self._get_retention_recommendations(
            X_scaled, player_features, top_risk_factors
        )

        return {
            'player_name': player_name,
            'flight_risk_score': round(flight_risk_score, 1),
            'risk_level': risk_level,
            'top_risk_factors': top_risk_factors,
            'retention_recommendations': retention_recommendations,
        }

    def _prepare_prediction_input(
        self,
        features: Union[pd.Series, pd.DataFrame, Dict],
        feature_names: List[str]
    ) -> np.ndarray:
        """Convert features to model input array."""

        if isinstance(features, dict):
            features = pd.Series(features)
        elif isinstance(features, pd.DataFrame):
            if len(features) != 1:
                raise ValueError("DataFrame must have exactly 1 row")
            features = features.iloc[0]

        X = np.zeros((1, len(feature_names)))
        for i, feat in enumerate(feature_names):
            if feat in features.index:
                val = features[feat]
                X[0, i] = float(val) if pd.notna(val) else 0.0

        return X.astype(np.float32)

    def _get_player_name(self, features: Union[pd.Series, pd.DataFrame, Dict]) -> str:
        """Extract player name from features."""

        if isinstance(features, dict):
            features = pd.Series(features)
        elif isinstance(features, pd.DataFrame):
            features = features.iloc[0]

        for col in ['player_name_std', 'player_name', 'name']:
            if col in features.index and pd.notna(features[col]):
                return str(features[col])

        return 'Unknown'

    def _get_top_risk_factors(
        self,
        X_scaled: np.ndarray,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        n_factors: int = 5
    ) -> List[Dict[str, str]]:
        """Get top factors contributing to flight risk."""

        factors = []

        # Get feature importances or SHAP values
        if HAS_SHAP and self.shap_explainer_flight is not None:
            try:
                shap_vals = self.shap_explainer_flight.shap_values(X_scaled)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # Class 1 (transfer)
                contributions = shap_vals[0]
            except:
                contributions = None
        else:
            contributions = None

        if contributions is None and hasattr(self.flight_risk_model, 'feature_importances_'):
            # Use feature importance * feature value as proxy
            importances = self.flight_risk_model.feature_importances_
            contributions = importances * X_scaled[0]

        if contributions is None:
            return []

        # Get top positive contributors (pushing toward transfer)
        positive_mask = contributions > 0
        if positive_mask.any():
            positive_indices = np.where(positive_mask)[0]
            positive_contributions = contributions[positive_mask]
            sorted_idx = np.argsort(positive_contributions)[::-1][:n_factors]

            for idx in sorted_idx:
                feat_idx = positive_indices[idx]
                feat_name = self.flight_risk_features[feat_idx]
                contrib = contributions[feat_idx]

                # Determine impact level
                if abs(contrib) > 0.3:
                    impact = 'high'
                elif abs(contrib) > 0.1:
                    impact = 'medium'
                else:
                    impact = 'low'

                # Get description
                description = RISK_FACTOR_DESCRIPTIONS.get(
                    feat_name,
                    f"Feature '{feat_name}' is contributing to transfer risk"
                )

                # Add context from actual feature value
                if isinstance(player_features, pd.DataFrame):
                    player_features = player_features.iloc[0]
                if isinstance(player_features, dict):
                    player_features = pd.Series(player_features)

                if feat_name in player_features.index:
                    val = player_features[feat_name]
                    description = self._add_value_context(feat_name, val, description)

                factors.append({
                    'factor': feat_name,
                    'impact': impact,
                    'description': description,
                })

        return factors

    def _add_value_context(self, feature: str, value: Any, description: str) -> str:
        """Add specific value context to description."""

        if pd.isna(value):
            return description

        value = float(value)

        if feature == 'snap_pct':
            return f"Playing time is {value*100:.0f}% of snaps"
        elif feature == 'snap_trend':
            if value < 0:
                return f"Playing time decreased {abs(value)*100:.0f}% from last season"
            else:
                return f"Playing time increased {value*100:.0f}% from last season"
        elif feature == 'coaching_change' and value == 1:
            return "Head coach was replaced this offseason"
        elif feature == 'nil_vs_team_median':
            if value < 0:
                return f"NIL value is ${abs(value):,.0f} below team median"
            else:
                return f"NIL value is ${value:,.0f} above team median"
        elif feature == 'depth_chart_position':
            if value == 1:
                return "Player is the starter"
            elif value == 2:
                return "Player is primary backup"
            else:
                return f"Player is #{int(value)} on depth chart"
        elif feature == 'team_wins':
            return f"Team won {int(value)} games this season"
        elif feature == 'incoming_recruits_at_position':
            return f"{int(value)} highly-rated recruits signed at player's position"

        return description

    def _get_retention_recommendations(
        self,
        X_scaled: np.ndarray,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        risk_factors: List[Dict]
    ) -> List[str]:
        """Generate actionable retention recommendations."""

        recommendations = []

        if isinstance(player_features, pd.DataFrame):
            player_features = player_features.iloc[0]
        if isinstance(player_features, dict):
            player_features = pd.Series(player_features)

        # Check each potential issue
        if 'nil_vs_team_median' in player_features.index:
            val = player_features['nil_vs_team_median']
            if pd.notna(val) and val < 0:
                recommendations.append(RETENTION_RECOMMENDATIONS['nil_below_median'])

        if 'nil_could_increase_elsewhere' in player_features.index:
            if player_features['nil_could_increase_elsewhere'] == 1:
                recommendations.append(RETENTION_RECOMMENDATIONS['nil_could_increase'])

        if 'coaching_change' in player_features.index:
            if player_features['coaching_change'] == 1:
                recommendations.append(RETENTION_RECOMMENDATIONS['coaching_change'])

        if 'snap_pct' in player_features.index:
            val = player_features['snap_pct']
            if pd.notna(val) and val < 0.3:
                recommendations.append(RETENTION_RECOMMENDATIONS['snap_pct_low'])

        if 'snap_trend' in player_features.index:
            val = player_features['snap_trend']
            if pd.notna(val) and val < -0.15:
                recommendations.append(RETENTION_RECOMMENDATIONS['snap_trend_negative'])

        if 'incoming_recruits_at_position' in player_features.index:
            val = player_features['incoming_recruits_at_position']
            if pd.notna(val) and val >= 2:
                recommendations.append(RETENTION_RECOMMENDATIONS['incoming_competition'])

        if 'depth_chart_position' in player_features.index:
            val = player_features['depth_chart_position']
            if pd.notna(val) and val >= 3:
                recommendations.append(RETENTION_RECOMMENDATIONS['depth_buried'])

        if 'production_score' in player_features.index:
            val = player_features['production_score']
            if pd.notna(val) and val > 70:
                recommendations.append(RETENTION_RECOMMENDATIONS['production_high'])

        if 'breakout_flag' in player_features.index:
            if player_features['breakout_flag'] == 1:
                recommendations.append(RETENTION_RECOMMENDATIONS['breakout_performer'])

        if 'is_grad_transfer_eligible' in player_features.index:
            if player_features['is_grad_transfer_eligible'] == 1:
                recommendations.append(RETENTION_RECOMMENDATIONS['grad_eligible'])

        if 'team_wins' in player_features.index:
            val = player_features['team_wins']
            if pd.notna(val) and val < 5:
                recommendations.append(RETENTION_RECOMMENDATIONS['team_losing'])

        # Limit to top 5 most relevant
        return recommendations[:5]

    def team_flight_risk_report(
        self,
        school: str,
        roster_features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate flight risk report for an entire team roster.

        Args:
            school: School name
            roster_features_df: DataFrame with features for all roster players

        Returns:
            DataFrame sorted by flight risk
        """
        if self.flight_risk_model is None:
            raise ValueError("Model not trained. Call train_flight_risk() first.")

        results = []

        for idx, row in roster_features_df.iterrows():
            try:
                pred = self.predict_flight_risk(row)

                player_data = {
                    'player_name': pred['player_name'],
                    'position': row.get('position_group', row.get('position', 'UNK')),
                    'flight_risk_score': pred['flight_risk_score'],
                    'risk_level': pred['risk_level'],
                    'production_score': row.get('production_score', 0),
                    'estimated_nil_value': row.get('estimated_nil_value', row.get('nil_value', 0)),
                }

                # Add top risk factor
                if pred['top_risk_factors']:
                    player_data['top_risk_factor'] = pred['top_risk_factors'][0]['factor']
                else:
                    player_data['top_risk_factor'] = 'None identified'

                results.append(player_data)

            except Exception as e:
                logger.warning(f"Failed to predict for player {idx}: {e}")

        # Create DataFrame
        report_df = pd.DataFrame(results)

        if len(report_df) == 0:
            return report_df

        # Sort by flight risk score
        report_df = report_df.sort_values('flight_risk_score', ascending=False)

        # Add critical retention flag
        report_df['critical_retention'] = (
            (report_df['flight_risk_score'] >= 50) &
            (report_df['production_score'] >= 60)
        )

        # Estimate roster impact (wins lost if player leaves)
        report_df['est_wins_impact'] = report_df.apply(
            lambda row: self._estimate_win_impact(row['position'], row['production_score']),
            axis=1
        )

        # Save report
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, f'{school.lower().replace(" ", "_")}_flight_risk_report.csv')
        report_df.to_csv(report_path, index=False)
        logger.info(f"Team flight risk report saved to {report_path}")

        # Also generate summary text
        summary = self._generate_team_risk_summary(school, report_df)
        summary_path = os.path.join(self.output_dir, f'{school.lower().replace(" ", "_")}_flight_risk_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)

        return report_df

    def _estimate_win_impact(self, position: str, production_score: float) -> float:
        """Estimate wins lost if player leaves."""

        base_impact = POSITION_WIN_IMPACT.get(position, 0.5)

        # Scale by production (elite players worth more)
        production_multiplier = production_score / 50  # 1.0 for average, 2.0 for elite

        return round(base_impact * production_multiplier, 2)

    def _generate_team_risk_summary(self, school: str, report_df: pd.DataFrame) -> str:
        """Generate text summary of team flight risk."""

        lines = []
        lines.append("=" * 60)
        lines.append(f"FLIGHT RISK SUMMARY: {school.upper()}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")

        # Overall stats
        lines.append("ROSTER OVERVIEW")
        lines.append("-" * 40)
        lines.append(f"Total players analyzed: {len(report_df)}")
        lines.append(f"High risk (>60): {(report_df['risk_level'] == 'high').sum()}")
        lines.append(f"Medium risk (30-60): {(report_df['risk_level'] == 'medium').sum()}")
        lines.append(f"Low risk (<30): {(report_df['risk_level'] == 'low').sum()}")
        lines.append(f"Average flight risk score: {report_df['flight_risk_score'].mean():.1f}")
        lines.append("")

        # Critical retention targets
        critical = report_df[report_df['critical_retention']]
        if len(critical) > 0:
            lines.append("⚠️  CRITICAL RETENTION TARGETS")
            lines.append("-" * 40)
            lines.append("(High flight risk + High production)")
            for _, row in critical.iterrows():
                lines.append(f"  • {row['player_name']} ({row['position']})")
                lines.append(f"    Risk: {row['flight_risk_score']:.0f}, Production: {row['production_score']:.0f}")
                lines.append(f"    Top factor: {row['top_risk_factor']}")
                lines.append(f"    Est. wins impact if leaves: {row['est_wins_impact']:.1f}")
            lines.append("")

        # Total potential wins at risk
        high_risk = report_df[report_df['flight_risk_score'] >= 50]
        total_wins_at_risk = high_risk['est_wins_impact'].sum()
        lines.append(f"TOTAL WINS AT RISK (high-risk players): {total_wins_at_risk:.1f}")
        lines.append("")

        # Top 10 highest risk
        lines.append("TOP 10 HIGHEST FLIGHT RISK")
        lines.append("-" * 40)
        for i, (_, row) in enumerate(report_df.head(10).iterrows(), 1):
            lines.append(f"{i}. {row['player_name']} ({row['position']}): {row['flight_risk_score']:.0f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    # =========================================================================
    # PORTAL FIT MODEL
    # =========================================================================

    def train_portal_fit(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train portal fit prediction model.

        Args:
            features_df: DataFrame with fit features and transfer_success target

        Returns:
            Dictionary with training results
        """
        logger.info("Training portal fit model...")

        df = features_df.copy()

        # Target: transfer success (improvement in production)
        if 'transfer_success' not in df.columns:
            # Create proxy from production improvement if available
            if 'production_improvement' in df.columns:
                df['transfer_success'] = df['production_improvement']
            else:
                raise ValueError("'transfer_success' or 'production_improvement' target required")

        # Prepare data
        X, y, feature_names = self._prepare_fit_data(df)
        self.fit_features = feature_names
        self.training_data_fit = df.copy()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale
        self.scaler_fit = StandardScaler()
        X_train_scaled = self.scaler_fit.fit_transform(X_train)
        X_test_scaled = self.scaler_fit.transform(X_test)

        # Train models
        results = self._train_fit_models(X_train_scaled, y_train, X_test_scaled, y_test)

        # Save
        self._save_fit_models()
        self.fit_metrics = results

        logger.info("Portal fit model training complete!")
        return results

    def _prepare_fit_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for fit model."""

        exclude_cols = [
            'transfer_success', 'production_improvement',
            'player_name', 'player_name_std', 'target_school', 'origin_school'
        ]

        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

        X = df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)

        y = df['transfer_success'].values.astype(np.float32)

        return X, y, feature_cols

    def _train_fit_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train fit regression models."""

        results = {'models': {}}

        models = {
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
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

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results['models'][name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
            }

            logger.info(f"    RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_name = name

        self.fit_model = best_model
        results['best_model'] = best_name
        results['best_metrics'] = results['models'][best_name]

        return results

    def _save_fit_models(self) -> None:
        """Save fit models to disk."""

        os.makedirs(self.model_dir, exist_ok=True)

        joblib.dump(
            self.fit_model,
            os.path.join(self.model_dir, 'fit_model.joblib')
        )
        joblib.dump(
            self.scaler_fit,
            os.path.join(self.model_dir, 'scaler_fit.joblib')
        )

        with open(os.path.join(self.model_dir, 'fit_features.json'), 'w') as f:
            json.dump(self.fit_features, f)

    def predict_portal_fit(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        target_school_features: Union[pd.Series, pd.DataFrame, Dict]
    ) -> Dict[str, Any]:
        """
        Predict how well a portal player fits a target school.

        Args:
            player_features: Portal player's features
            target_school_features: Target school's features

        Returns:
            Dictionary with fit score and analysis
        """
        # Combine player and school features
        combined = self._combine_fit_features(player_features, target_school_features)

        # Get fit score
        if self.fit_model is not None and self.scaler_fit is not None:
            X = self._prepare_prediction_input(combined, self.fit_features)
            X_scaled = self.scaler_fit.transform(X)
            fit_score = float(self.fit_model.predict(X_scaled)[0])
            fit_score = np.clip(fit_score, 0, 100)
        else:
            # Calculate heuristic fit score
            fit_score = self._calculate_heuristic_fit(player_features, target_school_features)

        # Get specific fit factors
        fit_factors = self._get_fit_factors(player_features, target_school_features)

        # Find comparable past transfers
        comparables = self._find_comparable_transfers(player_features, target_school_features)

        return {
            'fit_score': round(fit_score, 1),
            'fit_level': 'excellent' if fit_score >= 80 else ('good' if fit_score >= 60 else ('moderate' if fit_score >= 40 else 'poor')),
            'fit_factors': fit_factors,
            'comparable_transfers': comparables,
        }

    def _combine_fit_features(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        school_features: Union[pd.Series, pd.DataFrame, Dict]
    ) -> pd.Series:
        """Combine player and school features for fit prediction."""

        if isinstance(player_features, pd.DataFrame):
            player_features = player_features.iloc[0]
        if isinstance(player_features, dict):
            player_features = pd.Series(player_features)

        if isinstance(school_features, pd.DataFrame):
            school_features = school_features.iloc[0]
        if isinstance(school_features, dict):
            school_features = pd.Series(school_features)

        # Combine with prefixes
        combined = {}
        for k, v in player_features.items():
            combined[f'player_{k}'] = v
        for k, v in school_features.items():
            combined[f'school_{k}'] = v

        return pd.Series(combined)

    def _calculate_heuristic_fit(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        target_school_features: Union[pd.Series, pd.DataFrame, Dict]
    ) -> float:
        """Calculate fit score using heuristics when model not available."""

        if isinstance(player_features, pd.DataFrame):
            player_features = player_features.iloc[0]
        if isinstance(player_features, dict):
            player_features = pd.Series(player_features)

        if isinstance(target_school_features, pd.DataFrame):
            target_school_features = target_school_features.iloc[0]
        if isinstance(target_school_features, dict):
            target_school_features = pd.Series(target_school_features)

        score = 50  # Base score

        # School tier match (same tier = best)
        player_tier = player_features.get('school_tier', 3)
        target_tier = target_school_features.get('tier', target_school_features.get('school_tier', 3))

        tier_diff = target_tier - player_tier
        if tier_diff == 0:
            score += 15
        elif tier_diff == 1:
            score += 5
        elif tier_diff == -1:
            score += 10
        elif abs(tier_diff) >= 2:
            score -= 10

        # Production score
        production = player_features.get('production_score', 50)
        if production > 70:
            score += 15
        elif production > 50:
            score += 5

        # NIL budget fit
        player_nil = player_features.get('nil_value', player_features.get('estimated_nil_value', 100000))
        school_nil_tier = target_school_features.get('nil_tier', 3)
        school_budget = NIL_BUDGET_BY_TIER.get(school_nil_tier, 2000000)

        if player_nil <= school_budget * 0.1:
            score += 10
        elif player_nil <= school_budget * 0.15:
            score += 5
        elif player_nil > school_budget * 0.2:
            score -= 10

        return np.clip(score, 0, 100)

    def _get_fit_factors(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        target_school_features: Union[pd.Series, pd.DataFrame, Dict]
    ) -> Dict[str, Dict[str, Any]]:
        """Get detailed fit factors."""

        if isinstance(player_features, pd.DataFrame):
            player_features = player_features.iloc[0]
        if isinstance(player_features, dict):
            player_features = pd.Series(player_features)

        if isinstance(target_school_features, pd.DataFrame):
            target_school_features = target_school_features.iloc[0]
        if isinstance(target_school_features, dict):
            target_school_features = pd.Series(target_school_features)

        factors = {}

        # Positional need
        factors['positional_need'] = {
            'score': target_school_features.get('positional_need_score', 50),
            'description': "School's need at player's position"
        }

        # Tier match
        player_tier = player_features.get('school_tier', 3)
        target_tier = target_school_features.get('tier', target_school_features.get('school_tier', 3))
        tier_diff = target_tier - player_tier

        if tier_diff == 0:
            tier_desc = "Same tier - good fit"
            tier_score = 80
        elif tier_diff > 0:
            tier_desc = f"Moving up {tier_diff} tier(s) - ambitious"
            tier_score = max(40, 80 - tier_diff * 15)
        else:
            tier_desc = f"Moving down {abs(tier_diff)} tier(s) - safe choice"
            tier_score = max(50, 80 - abs(tier_diff) * 10)

        factors['tier_match'] = {'score': tier_score, 'description': tier_desc}

        # NIL budget fit
        player_nil = player_features.get('nil_value', player_features.get('estimated_nil_value', 100000))
        school_nil_tier = target_school_features.get('nil_tier', 3)
        school_budget = NIL_BUDGET_BY_TIER.get(school_nil_tier, 2000000)

        if player_nil <= school_budget * 0.08:
            nil_score = 90
            nil_desc = "Easily affordable"
        elif player_nil <= school_budget * 0.12:
            nil_score = 70
            nil_desc = "Affordable"
        elif player_nil <= school_budget * 0.18:
            nil_score = 50
            nil_desc = "Stretch but possible"
        else:
            nil_score = 30
            nil_desc = "May exceed budget"

        factors['nil_budget_fit'] = {'score': nil_score, 'description': nil_desc}

        # Production upgrade
        player_production = player_features.get('production_score', 50)
        factors['production_level'] = {
            'score': player_production,
            'description': f"Production score: {player_production:.0f}/100"
        }

        return factors

    def _find_comparable_transfers(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        target_school_features: Union[pd.Series, pd.DataFrame, Dict],
        n_comparables: int = 3
    ) -> List[Dict[str, Any]]:
        """Find similar past transfers."""

        if self.training_data_fit is None or len(self.training_data_fit) == 0:
            return []

        # For now, return placeholder
        return [
            {
                'name': 'Similar Transfer 1',
                'origin': 'School A',
                'destination': 'School B',
                'outcome': 'Successful',
                'similarity': 0.85
            }
        ]

    def rank_portal_targets(
        self,
        school: str,
        school_roster: pd.DataFrame,
        available_portal_players: pd.DataFrame,
        position_filter: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Rank available portal players by fit for a school.

        Args:
            school: Target school name
            school_roster: Current roster features
            available_portal_players: DataFrame of available portal players
            position_filter: Optional position group to filter
            top_n: Number of top recommendations to return

        Returns:
            DataFrame with ranked recommendations
        """
        logger.info(f"Ranking portal targets for {school}...")

        # Get school data
        school_data = SCHOOL_DATA.get(school, {
            'tier': 3, 'conference_tier': 2, 'market_size': 500,
            'wins': 7, 'nil_tier': 2, 'style': 'spread'
        })
        school_features = pd.Series(school_data)

        # Calculate roster needs
        roster_needs = self._calculate_roster_needs(school_roster)
        school_features['roster_needs'] = roster_needs

        # Filter by position if specified
        if position_filter:
            players = available_portal_players[
                available_portal_players['position_group'] == position_filter
            ].copy()
        else:
            players = available_portal_players.copy()

        if len(players) == 0:
            return pd.DataFrame()

        # Score each player
        results = []
        for idx, player in players.iterrows():
            try:
                # Get fit prediction
                fit_result = self.predict_portal_fit(player, school_features)

                # Adjust for positional need
                position = player.get('position_group', 'ATH')
                need_bonus = roster_needs.get(position, 0) * 10

                adjusted_score = fit_result['fit_score'] + need_bonus

                # Estimate NIL cost
                base_nil = player.get('nil_value', player.get('estimated_nil_value', 100000))
                # Premium for moving up tiers
                player_tier = player.get('school_tier', 3)
                tier_diff = school_data['tier'] - player_tier
                nil_multiplier = 1.0 + max(0, tier_diff) * 0.15
                estimated_nil_cost = base_nil * nil_multiplier

                results.append({
                    'player_name': player.get('player_name_std', player.get('name', f'Player {idx}')),
                    'position': position,
                    'origin_school': player.get('school_name', player.get('origin_school', 'Unknown')),
                    'fit_score': fit_result['fit_score'],
                    'adjusted_score': adjusted_score,
                    'positional_need': roster_needs.get(position, 0),
                    'production_score': player.get('production_score', 50),
                    'estimated_nil_cost': estimated_nil_cost,
                    'fit_level': fit_result['fit_level'],
                    'reasoning': self._generate_recommendation_reasoning(
                        player, school_features, fit_result
                    ),
                })

            except Exception as e:
                logger.warning(f"Failed to score player {idx}: {e}")

        # Create DataFrame and sort
        rankings = pd.DataFrame(results)
        if len(rankings) == 0:
            return rankings

        rankings = rankings.sort_values('adjusted_score', ascending=False).head(top_n)

        # Save report
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(
            self.output_dir,
            f'{school.lower().replace(" ", "_")}_portal_targets.csv'
        )
        rankings.to_csv(report_path, index=False)
        logger.info(f"Portal targets report saved to {report_path}")

        return rankings

    def _calculate_roster_needs(self, roster: pd.DataFrame) -> Dict[str, float]:
        """Calculate positional needs based on roster composition."""

        # Ideal roster composition
        ideal = {
            'QB': 3, 'RB': 4, 'WR': 8, 'TE': 3, 'OL': 12,
            'EDGE': 6, 'DL': 8, 'LB': 6, 'CB': 6, 'S': 4
        }

        needs = {}

        if 'position_group' in roster.columns:
            current = roster['position_group'].value_counts().to_dict()
        else:
            current = {}

        for pos, ideal_count in ideal.items():
            actual = current.get(pos, 0)
            need = (ideal_count - actual) / ideal_count
            needs[pos] = max(0, need)

        return needs

    def _generate_recommendation_reasoning(
        self,
        player: pd.Series,
        school_features: pd.Series,
        fit_result: Dict
    ) -> str:
        """Generate human-readable recommendation reasoning."""

        reasons = []

        # Production
        production = player.get('production_score', 50)
        if production >= 75:
            reasons.append("Elite producer")
        elif production >= 60:
            reasons.append("Strong producer")

        # Fit level
        if fit_result['fit_level'] == 'excellent':
            reasons.append("excellent scheme/tier fit")
        elif fit_result['fit_level'] == 'good':
            reasons.append("good program fit")

        # Positional need
        if 'positional_need' in fit_result['fit_factors']:
            need_score = fit_result['fit_factors']['positional_need']['score']
            if need_score >= 70:
                reasons.append("fills critical need")

        return "; ".join(reasons) if reasons else "Solid option"

    def rank_destinations(
        self,
        portal_player_features: Union[pd.Series, pd.DataFrame, Dict],
        schools_list: Optional[List[str]] = None,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Rank best destination schools for a portal player.

        Args:
            portal_player_features: Portal player's features
            schools_list: List of schools to consider (default: all known)
            top_n: Number of top destinations to return

        Returns:
            DataFrame with ranked destinations
        """
        if isinstance(portal_player_features, pd.DataFrame):
            portal_player_features = portal_player_features.iloc[0]
        if isinstance(portal_player_features, dict):
            portal_player_features = pd.Series(portal_player_features)

        player_name = portal_player_features.get(
            'player_name_std',
            portal_player_features.get('name', 'Unknown')
        )

        logger.info(f"Ranking destinations for {player_name}...")

        # Default to all known schools
        if schools_list is None:
            schools_list = list(SCHOOL_DATA.keys())

        results = []

        for school in schools_list:
            school_data = SCHOOL_DATA.get(school)
            if school_data is None:
                continue

            school_features = pd.Series(school_data)

            try:
                # Get fit score
                fit_result = self.predict_portal_fit(portal_player_features, school_features)

                # Additional factors
                # NIL potential
                player_production = portal_player_features.get('production_score', 50)
                nil_tier = school_data['nil_tier']
                nil_potential = NIL_BUDGET_BY_TIER.get(nil_tier, 1000000) * (player_production / 100) * 0.1

                # Playing time likelihood
                player_tier = portal_player_features.get('school_tier', 3)
                tier_diff = school_data['tier'] - player_tier
                if tier_diff <= 0:
                    playing_time_score = 80 + player_production * 0.2
                elif tier_diff == 1:
                    playing_time_score = 60 + player_production * 0.3
                else:
                    playing_time_score = 40 + player_production * 0.4
                playing_time_score = min(100, playing_time_score)

                # Team competitiveness
                competitiveness = school_data['wins'] * 7 + school_data['tier'] * 5

                # Geography (placeholder - would use actual distance)
                geography_score = 50

                # Overall score (weighted)
                overall_score = (
                    fit_result['fit_score'] * 0.30 +
                    (nil_potential / 500000) * 0.20 +
                    playing_time_score * 0.25 +
                    competitiveness * 0.15 +
                    geography_score * 0.10
                )

                results.append({
                    'school': school,
                    'overall_score': overall_score,
                    'fit_score': fit_result['fit_score'],
                    'nil_potential': nil_potential,
                    'playing_time_likelihood': playing_time_score,
                    'team_competitiveness': competitiveness,
                    'school_tier': school_data['tier'],
                    'conference_tier': school_data['conference_tier'],
                    'reasoning': self._generate_destination_reasoning(
                        school_data, fit_result, playing_time_score, nil_potential
                    ),
                })

            except Exception as e:
                logger.warning(f"Failed to evaluate {school}: {e}")

        # Create and sort DataFrame
        rankings = pd.DataFrame(results)
        if len(rankings) == 0:
            return rankings

        rankings = rankings.sort_values('overall_score', ascending=False).head(top_n)

        # Save report
        os.makedirs(self.output_dir, exist_ok=True)
        safe_name = str(player_name).lower().replace(' ', '_')[:30]
        report_path = os.path.join(
            self.output_dir,
            f'{safe_name}_destination_rankings.csv'
        )
        rankings.to_csv(report_path, index=False)
        logger.info(f"Destination rankings saved to {report_path}")

        return rankings

    def _generate_destination_reasoning(
        self,
        school_data: Dict,
        fit_result: Dict,
        playing_time_score: float,
        nil_potential: float
    ) -> str:
        """Generate reasoning for destination recommendation."""

        reasons = []

        if fit_result['fit_score'] >= 70:
            reasons.append("Strong fit")
        if nil_potential >= 500000:
            reasons.append(f"High NIL potential (${nil_potential:,.0f})")
        if playing_time_score >= 75:
            reasons.append("Good playing time outlook")
        if school_data['wins'] >= 10:
            reasons.append("Winning program")
        if school_data['tier'] >= 5:
            reasons.append("Elite program")

        return "; ".join(reasons) if reasons else "Solid option"

    def load_models(self) -> None:
        """Load previously trained models."""

        self.flight_risk_model = joblib.load(
            os.path.join(self.model_dir, 'flight_risk_model.joblib')
        )
        self.scaler_flight = joblib.load(
            os.path.join(self.model_dir, 'scaler_flight.joblib')
        )

        with open(os.path.join(self.model_dir, 'flight_risk_features.json'), 'r') as f:
            self.flight_risk_features = json.load(f)

        # Load fit model if exists
        fit_path = os.path.join(self.model_dir, 'fit_model.joblib')
        if os.path.exists(fit_path):
            self.fit_model = joblib.load(fit_path)
            self.scaler_fit = joblib.load(
                os.path.join(self.model_dir, 'scaler_fit.joblib')
            )
            with open(os.path.join(self.model_dir, 'fit_features.json'), 'r') as f:
                self.fit_features = json.load(f)

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

    print("Portal Predictor - Standalone Mode")
    print("=" * 50)

    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    n_samples = 200

    # Generate flight risk training data
    flight_data = {
        'player_name_std': [f'player_{i}' for i in range(n_samples)],
        'school_name': np.random.choice(['Alabama', 'Ohio State', 'Georgia', 'Texas', 'Oregon', 'UCLA'], n_samples),
        'position_group': np.random.choice(['QB', 'WR', 'RB', 'CB', 'EDGE', 'LB'], n_samples),
        'season': np.random.choice([2022, 2023, 2024], n_samples, p=[0.3, 0.4, 0.3]),
        'snap_pct': np.random.uniform(0.1, 1.0, n_samples),
        'snap_trend': np.random.uniform(-0.3, 0.3, n_samples),
        'is_starter': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'depth_chart_position': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.3, 0.25, 0.15]),
        'production_score': np.random.uniform(20, 90, n_samples),
        'production_vs_star_avg': np.random.uniform(-30, 30, n_samples),
        'recruiting_stars': np.random.choice([2, 3, 4, 5], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        'team_wins': np.random.randint(3, 13, n_samples),
        'team_win_trend': np.random.uniform(-4, 4, n_samples),
        'coaching_change': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'school_tier': np.random.choice([6, 5, 4, 3, 2], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
        'conference_tier': np.random.choice([3, 2, 1], n_samples, p=[0.4, 0.35, 0.25]),
        'team_nil_tier': np.random.choice([5, 4, 3, 2, 1], n_samples),
        'nil_vs_team_median': np.random.uniform(-100000, 100000, n_samples),
        'nil_could_increase_elsewhere': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'incoming_recruits_at_position': np.random.randint(0, 4, n_samples),
        'position_depth': np.random.randint(3, 10, n_samples),
        'distance_from_home': np.random.uniform(50, 2000, n_samples),
        'years_at_school': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'is_grad_transfer_eligible': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'remaining_eligibility': np.random.choice([1, 2, 3, 4], n_samples),
        'breakout_flag': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'estimated_nil_value': np.random.exponential(150000, n_samples),
    }

    # Generate target (entered_portal) based on features
    entered_portal = (
        (flight_data['snap_pct'] < 0.4).astype(int) * 0.3 +
        (flight_data['coaching_change'] == 1).astype(int) * 0.25 +
        (flight_data['nil_vs_team_median'] < 0).astype(int) * 0.15 +
        (flight_data['incoming_recruits_at_position'] >= 2).astype(int) * 0.15 +
        np.random.uniform(0, 0.3, n_samples)
    )
    flight_data['entered_portal'] = (entered_portal > 0.5).astype(int)

    flight_df = pd.DataFrame(flight_data)

    print(f"Generated {len(flight_df)} player-seasons")
    print(f"Transfer rate: {flight_df['entered_portal'].mean():.1%}")

    # Initialize predictor
    predictor = PortalPredictor(
        model_dir="models/portal_prediction",
        output_dir="outputs/reports"
    )

    try:
        # Train flight risk model
        print("\n" + "=" * 50)
        print("TRAINING FLIGHT RISK MODEL")
        print("=" * 50)

        results = predictor.train_flight_risk(flight_df)

        # Test prediction
        print("\n" + "=" * 50)
        print("SAMPLE FLIGHT RISK PREDICTION")
        print("=" * 50)

        test_player = flight_df.iloc[0]
        prediction = predictor.predict_flight_risk(test_player)

        print(f"\nPlayer: {prediction['player_name']}")
        print(f"Flight Risk Score: {prediction['flight_risk_score']:.0f}/100")
        print(f"Risk Level: {prediction['risk_level'].upper()}")
        print(f"\nTop Risk Factors:")
        for factor in prediction['top_risk_factors'][:3]:
            print(f"  - {factor['factor']}: {factor['description']}")
        print(f"\nRetention Recommendations:")
        for rec in prediction['retention_recommendations'][:3]:
            print(f"  - {rec}")

        # Test team report
        print("\n" + "=" * 50)
        print("TEAM FLIGHT RISK REPORT")
        print("=" * 50)

        alabama_roster = flight_df[flight_df['school_name'] == 'Alabama'].head(20)
        if len(alabama_roster) > 0:
            team_report = predictor.team_flight_risk_report('Alabama', alabama_roster)
            print(f"\nAlabama roster analyzed: {len(team_report)} players")
            print(f"High risk players: {(team_report['risk_level'] == 'high').sum()}")
            print(f"Critical retention targets: {team_report['critical_retention'].sum()}")

        # Test destination ranking
        print("\n" + "=" * 50)
        print("DESTINATION RANKING")
        print("=" * 50)

        destinations = predictor.rank_destinations(test_player, top_n=5)
        print(f"\nTop 5 destinations for {prediction['player_name']}:")
        for _, row in destinations.iterrows():
            print(f"  {row['school']}: Score {row['overall_score']:.1f}")
            print(f"    Fit: {row['fit_score']:.0f}, NIL: ${row['nil_potential']:,.0f}")

        print("\n✓ Portal Predictor test complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
