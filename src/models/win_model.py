"""
Win Impact Model

Predicts team wins based on roster composition and estimates each player's
contribution to winning. Enables scenario analysis for roster decisions.

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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Position group weights for win contribution
# Based on historical analysis of position value in college football
POSITION_WIN_WEIGHTS = {
    'QB': 0.25,      # QBs have outsized impact
    'EDGE': 0.12,    # Pass rush is crucial
    'OL': 0.12,      # Offensive line (combined)
    'CB': 0.08,      # Cornerbacks
    'WR': 0.08,      # Wide receivers (combined)
    'RB': 0.06,      # Running backs
    'DL': 0.08,      # Interior defensive line
    'LB': 0.07,      # Linebackers
    'S': 0.05,       # Safeties
    'TE': 0.04,      # Tight ends
    'ST': 0.03,      # Special teams
    'ATH': 0.02,     # Athletes
}

# Replacement level production by position (percentile)
# A "replacement level" player produces at this percentile within their position
REPLACEMENT_LEVEL = {
    'QB': 30,   # Replacement QB is 30th percentile
    'RB': 35,
    'WR': 35,
    'TE': 40,
    'OL': 40,
    'EDGE': 35,
    'DL': 40,
    'LB': 40,
    'CB': 35,
    'S': 40,
    'ST': 50,
    'ATH': 35,
}

# Conference championship caliber roster template
# Production score thresholds by position (0-100)
CHAMPIONSHIP_ROSTER_TEMPLATE = {
    'QB': {'count': 1, 'min_production': 80, 'avg_production': 85},
    'RB': {'count': 2, 'min_production': 60, 'avg_production': 70},
    'WR': {'count': 3, 'min_production': 60, 'avg_production': 70},
    'TE': {'count': 1, 'min_production': 55, 'avg_production': 65},
    'OL': {'count': 5, 'min_production': 60, 'avg_production': 70},
    'EDGE': {'count': 2, 'min_production': 65, 'avg_production': 75},
    'DL': {'count': 3, 'min_production': 55, 'avg_production': 65},
    'LB': {'count': 3, 'min_production': 55, 'avg_production': 65},
    'CB': {'count': 2, 'min_production': 60, 'avg_production': 70},
    'S': {'count': 2, 'min_production': 55, 'avg_production': 65},
}

# Position win impact coefficients (wins per 10-point production increase)
POSITION_WIN_IMPACT_COEF = {
    'QB': 0.8,      # Elite QB vs average = ~2-3 wins
    'EDGE': 0.4,    # Elite pass rusher = ~1-1.5 wins
    'OL': 0.25,     # Per lineman
    'CB': 0.3,      # Shutdown corner = ~1 win
    'WR': 0.2,      # Per receiver
    'RB': 0.15,     # RB impact declining
    'DL': 0.2,
    'LB': 0.18,
    'S': 0.15,
    'TE': 0.12,
    'ST': 0.08,
    'ATH': 0.1,
}

# Conference strength factors
CONFERENCE_STRENGTH = {
    'SEC': 1.2,
    'Big Ten': 1.15,
    'Big 12': 1.0,
    'ACC': 0.95,
    'Pac-12': 0.95,
    'Mountain West': 0.8,
    'American': 0.8,
    'Sun Belt': 0.75,
    'MAC': 0.7,
    'Conference USA': 0.7,
    'Independent': 1.0,
}


class WinImpactModel:
    """
    Win impact model for college football teams.

    Features:
    - Team win prediction based on roster composition
    - Player win contribution estimation (college WAR)
    - Scenario analysis for roster changes
    - Roster gap analysis for strategic planning
    """

    def __init__(
        self,
        model_dir: str = "models/win_impact",
        output_dir: str = "outputs/reports"
    ):
        """
        Initialize the win impact model.

        Args:
            model_dir: Directory to save trained models
            output_dir: Directory to save reports
        """
        self.model_dir = model_dir
        self.output_dir = output_dir

        # Models
        self.team_win_model = None
        self.scaler = None

        # Training data
        self.feature_names = []
        self.training_data = None
        self.position_baselines = {}  # Position-level production baselines

        # Metrics
        self.metrics = {}

    def train(
        self,
        team_data: pd.DataFrame,
        player_stats: pd.DataFrame,
        recruiting_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Train win prediction models.

        Args:
            team_data: Team-level data with wins, conference, etc.
            player_stats: Player statistics with production scores
            recruiting_data: Optional recruiting class data

        Returns:
            Dictionary with training results
        """
        logger.info("Training win impact models...")

        # Calculate position baselines from player stats
        self._calculate_position_baselines(player_stats)

        # Build team-level features
        team_features = self._build_team_features(team_data, player_stats, recruiting_data)
        self.training_data = team_features.copy()

        # Prepare for model training
        X, y, feature_names = self._prepare_team_data(team_features)
        self.feature_names = feature_names

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train team win model
        results = self._train_team_win_model(
            X_train_scaled, y_train, X_test_scaled, y_test
        )

        # Save models
        self._save_models()

        # Generate report
        report = self._generate_training_report(results)
        self._save_report(report)

        self.metrics = results
        logger.info("Win impact model training complete!")

        return results

    def _calculate_position_baselines(self, player_stats: pd.DataFrame) -> None:
        """Calculate baseline production by position for WAR calculation."""

        if 'position_group' not in player_stats.columns:
            logger.warning("No position_group column, using defaults")
            self.position_baselines = {pos: 40 for pos in POSITION_WIN_WEIGHTS.keys()}
            return

        if 'production_score' not in player_stats.columns:
            logger.warning("No production_score column, using defaults")
            self.position_baselines = {pos: 40 for pos in POSITION_WIN_WEIGHTS.keys()}
            return

        # Calculate percentiles by position
        for pos in POSITION_WIN_WEIGHTS.keys():
            pos_data = player_stats[player_stats['position_group'] == pos]['production_score']
            if len(pos_data) > 0:
                replacement_pct = REPLACEMENT_LEVEL.get(pos, 35)
                self.position_baselines[pos] = pos_data.quantile(replacement_pct / 100)
            else:
                self.position_baselines[pos] = 40  # Default

        logger.info(f"Position baselines calculated: {self.position_baselines}")

    def _build_team_features(
        self,
        team_data: pd.DataFrame,
        player_stats: pd.DataFrame,
        recruiting_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Build team-level features from player and recruiting data."""

        teams = []

        # Get unique team-season combinations
        if 'team' in team_data.columns:
            team_col = 'team'
        elif 'school' in team_data.columns:
            team_col = 'school'
        else:
            team_col = team_data.columns[0]

        if 'season' in team_data.columns:
            unique_teams = team_data[[team_col, 'season']].drop_duplicates()
        else:
            unique_teams = team_data[[team_col]].drop_duplicates()
            unique_teams['season'] = datetime.now().year

        for _, row in unique_teams.iterrows():
            team_name = row[team_col]
            season = row['season'] if 'season' in row else datetime.now().year

            team_features = {'team': team_name, 'season': season}

            # Get team wins (target)
            team_row = team_data[
                (team_data[team_col] == team_name) &
                (team_data.get('season', season) == season)
            ] if 'season' in team_data.columns else team_data[team_data[team_col] == team_name]

            if len(team_row) > 0:
                team_features['wins'] = team_row['wins'].iloc[0] if 'wins' in team_row.columns else 6
                team_features['prev_wins'] = team_row['prev_wins'].iloc[0] if 'prev_wins' in team_row.columns else 6
                team_features['conference'] = team_row['conference'].iloc[0] if 'conference' in team_row.columns else 'Unknown'
            else:
                team_features['wins'] = 6
                team_features['prev_wins'] = 6
                team_features['conference'] = 'Unknown'

            # Calculate roster features from player stats
            if 'school' in player_stats.columns or 'team' in player_stats.columns:
                player_team_col = 'school' if 'school' in player_stats.columns else 'team'
                roster = player_stats[player_stats[player_team_col] == team_name]

                if 'season' in player_stats.columns:
                    roster = roster[roster['season'] == season]
            else:
                roster = pd.DataFrame()

            # Calculate roster composition features
            roster_features = self._calculate_roster_features(roster)
            team_features.update(roster_features)

            # Add recruiting features if available
            if recruiting_data is not None:
                recruit_features = self._calculate_recruiting_features(
                    recruiting_data, team_name, season
                )
                team_features.update(recruit_features)
            else:
                team_features['recruiting_composite'] = 0.85
                team_features['recruiting_rank'] = 50

            # Conference strength
            team_features['conference_strength'] = CONFERENCE_STRENGTH.get(
                team_features.get('conference', 'Unknown'), 0.85
            )

            # Win trajectory
            team_features['win_trajectory'] = team_features['prev_wins'] - 6  # vs average

            teams.append(team_features)

        return pd.DataFrame(teams)

    def _calculate_roster_features(self, roster: pd.DataFrame) -> Dict[str, float]:
        """Calculate roster composition features."""

        features = {}

        if len(roster) == 0:
            # Return defaults
            return {
                'total_production': 2500,
                'qb_production': 50,
                'offense_production': 1200,
                'defense_production': 1000,
                'avg_production': 50,
                'elite_player_count': 2,
                'starter_avg_production': 60,
                'depth_score': 50,
                'position_balance': 0.8,
                'returning_production_pct': 0.6,
            }

        # Overall production metrics
        if 'production_score' in roster.columns:
            features['total_production'] = roster['production_score'].sum()
            features['avg_production'] = roster['production_score'].mean()
            features['elite_player_count'] = (roster['production_score'] >= 75).sum()

            # Top 22 starters (approximate)
            starters = roster.nlargest(22, 'production_score')
            features['starter_avg_production'] = starters['production_score'].mean()

            # Depth: average of non-starters
            non_starters = roster[~roster.index.isin(starters.index)]
            features['depth_score'] = non_starters['production_score'].mean() if len(non_starters) > 0 else 30
        else:
            features['total_production'] = 2500
            features['avg_production'] = 50
            features['elite_player_count'] = 2
            features['starter_avg_production'] = 60
            features['depth_score'] = 40

        # Position group production
        if 'position_group' in roster.columns and 'production_score' in roster.columns:
            # QB production
            qb = roster[roster['position_group'] == 'QB']
            features['qb_production'] = qb['production_score'].max() if len(qb) > 0 else 40

            # Offense production
            offense_pos = ['QB', 'RB', 'WR', 'TE', 'OL']
            offense = roster[roster['position_group'].isin(offense_pos)]
            features['offense_production'] = offense['production_score'].sum() if len(offense) > 0 else 1200

            # Defense production
            defense_pos = ['EDGE', 'DL', 'LB', 'CB', 'S']
            defense = roster[roster['position_group'].isin(defense_pos)]
            features['defense_production'] = defense['production_score'].sum() if len(defense) > 0 else 1000

            # Position balance (how evenly distributed is talent)
            pos_totals = roster.groupby('position_group')['production_score'].sum()
            if len(pos_totals) > 0:
                features['position_balance'] = 1 - (pos_totals.std() / pos_totals.mean()) if pos_totals.mean() > 0 else 0.5
            else:
                features['position_balance'] = 0.5
        else:
            features['qb_production'] = 50
            features['offense_production'] = 1200
            features['defense_production'] = 1000
            features['position_balance'] = 0.5

        # Returning production estimate
        if 'year_classification' in roster.columns:
            returning = roster[~roster['year_classification'].isin([4, 5, 'SR', 'GR'])]
            total_prod = roster['production_score'].sum() if 'production_score' in roster.columns else 1
            returning_prod = returning['production_score'].sum() if 'production_score' in returning.columns else 0
            features['returning_production_pct'] = returning_prod / total_prod if total_prod > 0 else 0.6
        else:
            features['returning_production_pct'] = 0.6

        return features

    def _calculate_recruiting_features(
        self,
        recruiting_data: pd.DataFrame,
        team: str,
        season: int
    ) -> Dict[str, float]:
        """Calculate recruiting class features."""

        # Filter to team's recruiting class
        if 'committed_school' in recruiting_data.columns:
            team_recruits = recruiting_data[recruiting_data['committed_school'] == team]
        elif 'school' in recruiting_data.columns:
            team_recruits = recruiting_data[recruiting_data['school'] == team]
        else:
            return {'recruiting_composite': 0.85, 'recruiting_rank': 50}

        if len(team_recruits) == 0:
            return {'recruiting_composite': 0.85, 'recruiting_rank': 50}

        # Average composite rating
        if 'rating' in team_recruits.columns:
            composite = team_recruits['rating'].mean()
        elif 'composite' in team_recruits.columns:
            composite = team_recruits['composite'].mean()
        else:
            composite = 0.85

        # Class rank (would need full recruiting data)
        features = {
            'recruiting_composite': composite,
            'recruiting_rank': 50,  # Would calculate from full data
            'five_stars': (team_recruits.get('stars', 0) == 5).sum() if 'stars' in team_recruits.columns else 0,
            'four_stars': (team_recruits.get('stars', 0) == 4).sum() if 'stars' in team_recruits.columns else 0,
        }

        return features

    def _prepare_team_data(
        self,
        team_features: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare team-level data for model training."""

        # Select feature columns
        exclude_cols = ['team', 'season', 'wins', 'conference']
        feature_cols = [
            c for c in team_features.columns
            if c not in exclude_cols and team_features[c].dtype in ['int64', 'float64']
        ]

        X = team_features[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)

        y = team_features['wins'].values.astype(np.float32)

        return X, y, feature_cols

    def _train_team_win_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train team win prediction model."""

        results = {'models': {}}

        models = {
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            ),
        }

        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0
            )

        best_model = None
        best_mae = float('inf')
        best_name = None

        for name, model in models.items():
            logger.info(f"  Training {name}...")

            # Cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()

            # Fit and evaluate on test
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Within 1 win accuracy
            within_1 = (np.abs(y_test - y_pred) <= 1).mean()
            within_2 = (np.abs(y_test - y_pred) <= 2).mean()

            results['models'][name] = {
                'cv_mae': cv_mae,
                'test_mae': mae,
                'test_rmse': rmse,
                'test_r2': r2,
                'within_1_win': within_1,
                'within_2_wins': within_2,
            }

            logger.info(f"    MAE: {mae:.2f} wins, Within 1: {within_1:.1%}, Within 2: {within_2:.1%}")

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_name = name

        self.team_win_model = best_model
        results['best_model'] = best_name
        results['best_metrics'] = results['models'][best_name]

        return results

    def _save_models(self) -> None:
        """Save trained models."""

        os.makedirs(self.model_dir, exist_ok=True)

        joblib.dump(self.team_win_model, os.path.join(self.model_dir, 'team_win_model.joblib'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))

        with open(os.path.join(self.model_dir, 'feature_names.json'), 'w') as f:
            json.dump(self.feature_names, f)

        with open(os.path.join(self.model_dir, 'position_baselines.json'), 'w') as f:
            json.dump({k: float(v) for k, v in self.position_baselines.items()}, f)

        with open(os.path.join(self.model_dir, 'metrics.json'), 'w') as f:
            json.dump(self._convert_to_json(self.metrics), f, indent=2)

        logger.info(f"Models saved to {self.model_dir}")

    def _generate_training_report(self, results: Dict[str, Any]) -> str:
        """Generate training report."""

        lines = []
        lines.append("=" * 70)
        lines.append("WIN IMPACT MODEL TRAINING REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        lines.append("TEAM WIN PREDICTION MODEL")
        lines.append("-" * 40)

        for name, metrics in results['models'].items():
            marker = "✓" if name == results['best_model'] else " "
            lines.append(f"\n{marker} {name.upper()}:")
            lines.append(f"    CV MAE: {metrics['cv_mae']:.2f} wins")
            lines.append(f"    Test MAE: {metrics['test_mae']:.2f} wins")
            lines.append(f"    Test RMSE: {metrics['test_rmse']:.2f} wins")
            lines.append(f"    Within 1 win: {metrics['within_1_win']:.1%}")
            lines.append(f"    Within 2 wins: {metrics['within_2_wins']:.1%}")

        lines.append(f"\n✓ Best model: {results['best_model'].upper()}")
        lines.append("")

        # Feature importance
        if hasattr(self.team_win_model, 'feature_importances_'):
            lines.append("TOP 10 FEATURE IMPORTANCES")
            lines.append("-" * 40)
            importances = self.team_win_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            for i, idx in enumerate(indices):
                lines.append(f"  {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")

        lines.append("")
        lines.append("PLAYER WIN CONTRIBUTION METHODOLOGY")
        lines.append("-" * 40)
        lines.append("College WAR (Wins Above Replacement) calculated as:")
        lines.append("  WAR = (Player Production - Replacement Level) * Position Weight * Team Context")
        lines.append("")
        lines.append("Position baselines (replacement level):")
        for pos, baseline in self.position_baselines.items():
            lines.append(f"  {pos}: {baseline:.1f}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _save_report(self, report: str) -> None:
        """Save training report."""

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'win_model_training_report.txt')

        with open(path, 'w') as f:
            f.write(report)

        print(report)
        logger.info(f"Report saved to {path}")

    def predict_team_wins(
        self,
        school: str,
        roster_features: pd.DataFrame,
        incoming_players: Optional[pd.DataFrame] = None,
        outgoing_players: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Predict team wins for next season.

        Args:
            school: School name
            roster_features: Current roster with player features
            incoming_players: Players joining (transfers, recruits)
            outgoing_players: Players leaving (transfers out, draft, graduation)

        Returns:
            Win projection with breakdown
        """
        if self.team_win_model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Calculate current roster features
        current_roster = roster_features.copy()

        # Apply roster changes
        if outgoing_players is not None and len(outgoing_players) > 0:
            # Remove outgoing players
            outgoing_names = set(
                outgoing_players.get('player_name_std', outgoing_players.get('name', [])).tolist()
            )
            if 'player_name_std' in current_roster.columns:
                current_roster = current_roster[~current_roster['player_name_std'].isin(outgoing_names)]
            elif 'name' in current_roster.columns:
                current_roster = current_roster[~current_roster['name'].isin(outgoing_names)]

        if incoming_players is not None and len(incoming_players) > 0:
            # Add incoming players
            current_roster = pd.concat([current_roster, incoming_players], ignore_index=True)

        # Calculate team-level features
        team_features = self._calculate_roster_features(current_roster)

        # Add required features with defaults
        team_features['conference_strength'] = 1.0
        team_features['win_trajectory'] = 0
        team_features['prev_wins'] = 6
        team_features['recruiting_composite'] = 0.85
        team_features['recruiting_rank'] = 50

        # Prepare for prediction
        X = np.zeros((1, len(self.feature_names)))
        for i, feat in enumerate(self.feature_names):
            if feat in team_features:
                X[0, i] = team_features[feat]

        X_scaled = self.scaler.transform(X)

        # Get prediction
        predicted_wins = self.team_win_model.predict(X_scaled)[0]
        predicted_wins = np.clip(predicted_wins, 0, 15)

        # Calculate impact of roster changes
        incoming_impact = []
        if incoming_players is not None:
            for _, player in incoming_players.iterrows():
                impact = self._calculate_player_win_impact(player)
                incoming_impact.append({
                    'player': player.get('player_name_std', player.get('name', 'Unknown')),
                    'position': player.get('position_group', 'ATH'),
                    'win_impact': impact,
                })

        outgoing_impact = []
        if outgoing_players is not None:
            for _, player in outgoing_players.iterrows():
                impact = self._calculate_player_win_impact(player)
                outgoing_impact.append({
                    'player': player.get('player_name_std', player.get('name', 'Unknown')),
                    'position': player.get('position_group', 'ATH'),
                    'win_impact': -impact,  # Negative because leaving
                })

        # Confidence interval (based on model error)
        mae = self.metrics.get('best_metrics', {}).get('test_mae', 1.5)
        confidence_interval = {
            'low': max(0, predicted_wins - 1.5 * mae),
            'mid': predicted_wins,
            'high': min(15, predicted_wins + 1.5 * mae),
        }

        return {
            'school': school,
            'projected_wins': round(predicted_wins, 1),
            'confidence_interval': {k: round(v, 1) for k, v in confidence_interval.items()},
            'incoming_player_impact': incoming_impact,
            'outgoing_player_impact': outgoing_impact,
            'net_roster_impact': sum(p['win_impact'] for p in incoming_impact) + sum(p['win_impact'] for p in outgoing_impact),
            'roster_size': len(current_roster),
        }

    def _calculate_player_win_impact(self, player: pd.Series) -> float:
        """Calculate a single player's win contribution."""

        position = player.get('position_group', 'ATH')
        production = player.get('production_score', 50)

        # Get baseline for position
        baseline = self.position_baselines.get(position, 40)

        # Production above replacement
        par = max(0, production - baseline)

        # Position weight
        weight = POSITION_WIN_WEIGHTS.get(position, 0.05)

        # Win impact coefficient
        coef = POSITION_WIN_IMPACT_COEF.get(position, 0.15)

        # Calculate WAR (wins above replacement)
        war = (par / 10) * coef * (weight / 0.08)  # Normalized to 0.08 base weight

        return round(war, 3)

    def player_win_value(
        self,
        player_features: Union[pd.Series, pd.DataFrame, Dict],
        team_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calculate player's win contribution.

        Args:
            player_features: Player's feature data
            team_context: Optional team context (roster, wins, etc.)

        Returns:
            Win value analysis for the player
        """
        if isinstance(player_features, pd.DataFrame):
            player_features = player_features.iloc[0]
        if isinstance(player_features, dict):
            player_features = pd.Series(player_features)

        player_name = player_features.get(
            'player_name_std',
            player_features.get('name', 'Unknown')
        )

        position = player_features.get('position_group', 'ATH')
        production = player_features.get('production_score', 50)
        nil_value = player_features.get('nil_value', player_features.get('estimated_nil_value', 0))

        # Calculate WAR
        war = self._calculate_player_win_impact(player_features)

        # Calculate NIL per win
        nil_per_win = nil_value / war if war > 0 else float('inf')

        # Position baseline comparison
        baseline = self.position_baselines.get(position, 40)
        production_vs_replacement = production - baseline

        # Rank among teammates (if team context provided)
        team_rank = None
        if team_context and 'roster' in team_context:
            roster = team_context['roster']
            pos_players = roster[roster['position_group'] == position] if 'position_group' in roster.columns else roster
            if 'production_score' in pos_players.columns:
                better = (pos_players['production_score'] > production).sum()
                team_rank = better + 1

        return {
            'player_name': str(player_name),
            'position': position,
            'production_score': round(production, 1),
            'war': round(war, 3),  # Wins Above Replacement
            'production_vs_replacement': round(production_vs_replacement, 1),
            'nil_value': nil_value,
            'nil_per_win': round(nil_per_win, 0) if nil_per_win != float('inf') else 'N/A',
            'nil_efficiency_rating': self._rate_nil_efficiency(nil_per_win),
            'team_position_rank': team_rank,
            'interpretation': self._interpret_war(war, position),
        }

    def _rate_nil_efficiency(self, nil_per_win: float) -> str:
        """Rate NIL efficiency (cost per win)."""

        if nil_per_win == float('inf'):
            return 'N/A'
        elif nil_per_win < 200_000:
            return 'Excellent'
        elif nil_per_win < 400_000:
            return 'Good'
        elif nil_per_win < 700_000:
            return 'Fair'
        else:
            return 'Poor'

    def _interpret_war(self, war: float, position: str) -> str:
        """Provide interpretation of WAR value."""

        if war >= 1.5:
            return "Elite, program-changing player"
        elif war >= 1.0:
            return "All-Conference caliber contributor"
        elif war >= 0.5:
            return "Quality starter"
        elif war >= 0.2:
            return "Solid contributor"
        elif war >= 0:
            return "Replacement-level contributor"
        else:
            return "Below replacement level"

    def scenario_analysis(
        self,
        school: str,
        baseline_roster: pd.DataFrame,
        changes_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze win impact of roster changes.

        Args:
            school: School name
            baseline_roster: Current roster
            changes_list: List of changes like [{"action": "add", "player": X}, {"action": "remove", "player": Y}]

        Returns:
            Scenario analysis with win projections
        """
        # Get baseline projection
        baseline_pred = self.predict_team_wins(school, baseline_roster)
        baseline_wins = baseline_pred['projected_wins']

        scenarios = []

        # Analyze each change
        for change in changes_list:
            action = change.get('action', 'add')
            player = change.get('player')

            if player is None:
                continue

            if isinstance(player, dict):
                player = pd.Series(player)

            player_name = player.get('player_name_std', player.get('name', 'Unknown'))
            position = player.get('position_group', 'ATH')

            # Calculate win impact
            win_impact = self._calculate_player_win_impact(player)
            if action == 'remove':
                win_impact = -win_impact

            scenarios.append({
                'action': action,
                'player': str(player_name),
                'position': position,
                'production_score': player.get('production_score', 50),
                'win_impact': round(win_impact, 2),
                'projected_wins_after': round(baseline_wins + win_impact, 1),
            })

        # Calculate cumulative impact
        total_impact = sum(s['win_impact'] for s in scenarios)
        projected_wins = baseline_wins + total_impact

        # Sort scenarios by absolute impact
        scenarios.sort(key=lambda x: abs(x['win_impact']), reverse=True)

        return {
            'school': school,
            'baseline_wins': baseline_wins,
            'projected_wins_after_changes': round(projected_wins, 1),
            'net_win_impact': round(total_impact, 2),
            'scenarios': scenarios,
            'summary': self._generate_scenario_summary(scenarios, baseline_wins, projected_wins),
        }

    def _generate_scenario_summary(
        self,
        scenarios: List[Dict],
        baseline: float,
        projected: float
    ) -> str:
        """Generate human-readable scenario summary."""

        adds = [s for s in scenarios if s['action'] == 'add']
        removes = [s for s in scenarios if s['action'] == 'remove']

        parts = []

        if adds:
            add_names = ', '.join([s['player'] for s in adds[:3]])
            add_impact = sum(s['win_impact'] for s in adds)
            parts.append(f"Adding {add_names} projects +{add_impact:.1f} wins")

        if removes:
            remove_names = ', '.join([s['player'] for s in removes[:3]])
            remove_impact = sum(s['win_impact'] for s in removes)
            parts.append(f"Losing {remove_names} projects {remove_impact:.1f} wins")

        net = projected - baseline
        direction = "increase" if net > 0 else "decrease"

        summary = ". ".join(parts) + f". Net: {abs(net):.1f} win {direction} ({baseline:.1f} → {projected:.1f})."

        return summary

    def roster_gap_analysis(
        self,
        school: str,
        current_roster: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare roster to championship caliber and identify gaps.

        Args:
            school: School name
            current_roster: Current roster with production scores

        Returns:
            Gap analysis with prioritized recommendations
        """
        gaps = []

        for position, template in CHAMPIONSHIP_ROSTER_TEMPLATE.items():
            # Get current players at position
            if 'position_group' in current_roster.columns:
                pos_players = current_roster[current_roster['position_group'] == position]
            else:
                pos_players = pd.DataFrame()

            # Current count
            current_count = len(pos_players)
            needed_count = template['count']

            # Current production
            if 'production_score' in pos_players.columns and len(pos_players) > 0:
                top_players = pos_players.nlargest(template['count'], 'production_score')
                current_avg = top_players['production_score'].mean()
                current_max = pos_players['production_score'].max()
            else:
                current_avg = 40
                current_max = 50

            # Gaps
            count_gap = max(0, needed_count - current_count)
            production_gap = template['avg_production'] - current_avg

            # Calculate potential win impact of filling gap
            if production_gap > 0:
                coef = POSITION_WIN_IMPACT_COEF.get(position, 0.15)
                win_impact_potential = (production_gap / 10) * coef * template['count']
            else:
                win_impact_potential = 0

            gaps.append({
                'position': position,
                'current_count': current_count,
                'needed_count': needed_count,
                'count_gap': count_gap,
                'current_avg_production': round(current_avg, 1),
                'target_avg_production': template['avg_production'],
                'production_gap': round(production_gap, 1),
                'best_player_production': round(current_max, 1),
                'win_impact_potential': round(win_impact_potential, 2),
                'priority': self._calculate_gap_priority(count_gap, production_gap, position),
            })

        # Sort by priority/win impact
        gaps.sort(key=lambda x: x['win_impact_potential'], reverse=True)

        # Generate recommendations
        recommendations = self._generate_gap_recommendations(gaps)

        # Calculate overall roster grade
        total_gap = sum(g['production_gap'] * g['needed_count'] for g in gaps if g['production_gap'] > 0)
        roster_grade = self._calculate_roster_grade(total_gap)

        # Save report
        self._save_gap_analysis_report(school, gaps, recommendations, roster_grade)

        return {
            'school': school,
            'roster_grade': roster_grade,
            'position_gaps': gaps,
            'recommendations': recommendations,
            'total_win_upside': round(sum(g['win_impact_potential'] for g in gaps), 1),
        }

    def _calculate_gap_priority(self, count_gap: int, production_gap: float, position: str) -> str:
        """Calculate priority level for filling a gap."""

        # High-impact positions
        high_impact = ['QB', 'EDGE', 'OL', 'CB']

        score = 0
        if count_gap > 0:
            score += 30
        if production_gap > 20:
            score += 30
        elif production_gap > 10:
            score += 15
        if position in high_impact:
            score += 20

        if score >= 50:
            return 'Critical'
        elif score >= 30:
            return 'High'
        elif score >= 15:
            return 'Medium'
        else:
            return 'Low'

    def _generate_gap_recommendations(self, gaps: List[Dict]) -> List[str]:
        """Generate prioritized recommendations."""

        recommendations = []

        for gap in gaps[:5]:  # Top 5 priorities
            position = gap['position']
            win_impact = gap['win_impact_potential']
            production_gap = gap['production_gap']

            if win_impact >= 0.8:
                rec = f"CRITICAL: Adding an elite {position} projects +{win_impact:.1f} wins. "
                rec += f"Current avg production ({gap['current_avg_production']}) is {abs(production_gap):.0f} pts below championship level."
            elif win_impact >= 0.4:
                rec = f"HIGH: Upgrading {position} room projects +{win_impact:.1f} wins. "
                rec += f"Target {gap['target_avg_production']} avg production."
            elif win_impact >= 0.2:
                rec = f"MEDIUM: {position} improvement projects +{win_impact:.1f} wins."
            else:
                continue  # Skip low-impact

            recommendations.append(rec)

        return recommendations

    def _calculate_roster_grade(self, total_gap: float) -> str:
        """Calculate overall roster grade."""

        if total_gap < 50:
            return 'A'  # Championship caliber
        elif total_gap < 100:
            return 'B+'
        elif total_gap < 150:
            return 'B'
        elif total_gap < 200:
            return 'C+'
        elif total_gap < 300:
            return 'C'
        else:
            return 'D'

    def _save_gap_analysis_report(
        self,
        school: str,
        gaps: List[Dict],
        recommendations: List[str],
        grade: str
    ) -> None:
        """Save gap analysis report."""

        os.makedirs(self.output_dir, exist_ok=True)

        # Save JSON
        report_data = {
            'school': school,
            'generated_at': datetime.now().isoformat(),
            'roster_grade': grade,
            'position_gaps': gaps,
            'recommendations': recommendations,
        }

        json_path = os.path.join(self.output_dir, f'{school.lower().replace(" ", "_")}_roster_gap_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        # Save text report
        lines = []
        lines.append("=" * 60)
        lines.append(f"ROSTER GAP ANALYSIS: {school.upper()}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append(f"\nOVERALL ROSTER GRADE: {grade}")
        lines.append("")

        lines.append("POSITION-BY-POSITION ANALYSIS")
        lines.append("-" * 40)

        for gap in gaps:
            lines.append(f"\n{gap['position']}:")
            lines.append(f"  Current: {gap['current_count']} players, {gap['current_avg_production']:.0f} avg production")
            lines.append(f"  Target:  {gap['needed_count']} players, {gap['target_avg_production']:.0f} avg production")
            lines.append(f"  Gap: {gap['production_gap']:.0f} pts")
            lines.append(f"  Win Impact Potential: +{gap['win_impact_potential']:.2f}")
            lines.append(f"  Priority: {gap['priority']}")

        lines.append("")
        lines.append("PRIORITIZED RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"\n{i}. {rec}")

        lines.append("")
        lines.append("=" * 60)

        text_path = os.path.join(self.output_dir, f'{school.lower().replace(" ", "_")}_roster_gap_analysis.txt')
        with open(text_path, 'w') as f:
            f.write("\n".join(lines))

        logger.info(f"Gap analysis saved to {json_path}")

    def load_models(self) -> None:
        """Load previously trained models."""

        self.team_win_model = joblib.load(os.path.join(self.model_dir, 'team_win_model.joblib'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))

        with open(os.path.join(self.model_dir, 'feature_names.json'), 'r') as f:
            self.feature_names = json.load(f)

        with open(os.path.join(self.model_dir, 'position_baselines.json'), 'r') as f:
            self.position_baselines = json.load(f)

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

    print("Win Impact Model - Standalone Mode")
    print("=" * 50)

    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)

    # Team data
    teams = ['Alabama', 'Ohio State', 'Georgia', 'Texas', 'Oregon', 'Michigan',
             'Florida', 'LSU', 'Clemson', 'USC', 'Oklahoma', 'Penn State',
             'Tennessee', 'Miami', 'Notre Dame', 'Wisconsin', 'Iowa', 'UCLA',
             'UCF', 'Boise State']

    team_data = pd.DataFrame({
        'team': teams * 3,  # Multiple seasons
        'season': [2022]*20 + [2023]*20 + [2024]*20,
        'wins': np.random.randint(4, 14, 60),
        'prev_wins': np.random.randint(4, 14, 60),
        'conference': np.random.choice(['SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12', 'American'], 60),
    })

    # Player stats
    n_players = 500
    player_stats = pd.DataFrame({
        'player_name_std': [f'player_{i}' for i in range(n_players)],
        'school': np.random.choice(teams, n_players),
        'season': np.random.choice([2022, 2023, 2024], n_players),
        'position_group': np.random.choice(['QB', 'RB', 'WR', 'TE', 'OL', 'EDGE', 'DL', 'LB', 'CB', 'S'], n_players),
        'production_score': np.random.uniform(20, 95, n_players),
        'year_classification': np.random.choice([1, 2, 3, 4, 5], n_players),
        'nil_value': np.random.exponential(150000, n_players),
    })

    # Recruiting data
    recruiting_data = pd.DataFrame({
        'name': [f'recruit_{i}' for i in range(200)],
        'committed_school': np.random.choice(teams, 200),
        'stars': np.random.choice([3, 4, 5], 200, p=[0.6, 0.3, 0.1]),
        'rating': np.random.uniform(0.8, 1.0, 200),
    })

    print(f"Generated {len(team_data)} team-seasons, {len(player_stats)} players")

    # Initialize and train
    model = WinImpactModel(
        model_dir="models/win_impact",
        output_dir="outputs/reports"
    )

    try:
        print("\n" + "=" * 50)
        print("TRAINING")
        print("=" * 50)

        results = model.train(team_data, player_stats, recruiting_data)

        # Test team win prediction
        print("\n" + "=" * 50)
        print("TEAM WIN PREDICTION")
        print("=" * 50)

        alabama_roster = player_stats[(player_stats['school'] == 'Alabama') & (player_stats['season'] == 2024)]

        # Create incoming player
        incoming = pd.DataFrame([{
            'player_name_std': 'elite_transfer_qb',
            'position_group': 'QB',
            'production_score': 85,
        }])

        # Create outgoing player
        outgoing = pd.DataFrame([{
            'player_name_std': alabama_roster.iloc[0]['player_name_std'],
            'position_group': alabama_roster.iloc[0]['position_group'],
            'production_score': alabama_roster.iloc[0]['production_score'],
        }])

        prediction = model.predict_team_wins('Alabama', alabama_roster, incoming, outgoing)

        print(f"\nAlabama win projection: {prediction['projected_wins']}")
        print(f"Confidence interval: {prediction['confidence_interval']}")
        print(f"Net roster impact: {prediction['net_roster_impact']:.2f} wins")

        # Test player win value
        print("\n" + "=" * 50)
        print("PLAYER WIN VALUE")
        print("=" * 50)

        test_player = player_stats[player_stats['production_score'] > 80].iloc[0]
        win_value = model.player_win_value(test_player)

        print(f"\nPlayer: {win_value['player_name']}")
        print(f"Position: {win_value['position']}")
        print(f"Production: {win_value['production_score']}")
        print(f"WAR: {win_value['war']}")
        print(f"NIL: ${win_value['nil_value']:,.0f}")
        print(f"NIL per Win: ${win_value['nil_per_win']:,.0f}" if win_value['nil_per_win'] != 'N/A' else "NIL per Win: N/A")
        print(f"Interpretation: {win_value['interpretation']}")

        # Test scenario analysis
        print("\n" + "=" * 50)
        print("SCENARIO ANALYSIS")
        print("=" * 50)

        changes = [
            {'action': 'add', 'player': {'player_name_std': 'portal_qb', 'position_group': 'QB', 'production_score': 82}},
            {'action': 'add', 'player': {'player_name_std': 'portal_edge', 'position_group': 'EDGE', 'production_score': 75}},
            {'action': 'remove', 'player': {'player_name_std': 'old_lb', 'position_group': 'LB', 'production_score': 55}},
        ]

        scenario = model.scenario_analysis('Alabama', alabama_roster, changes)

        print(f"\nBaseline wins: {scenario['baseline_wins']}")
        print(f"Projected after changes: {scenario['projected_wins_after_changes']}")
        print(f"Net impact: {scenario['net_win_impact']:+.2f} wins")
        print(f"\nSummary: {scenario['summary']}")

        # Test roster gap analysis
        print("\n" + "=" * 50)
        print("ROSTER GAP ANALYSIS")
        print("=" * 50)

        gap_analysis = model.roster_gap_analysis('Alabama', alabama_roster)

        print(f"\nRoster Grade: {gap_analysis['roster_grade']}")
        print(f"Total Win Upside: +{gap_analysis['total_win_upside']:.1f}")
        print("\nTop Recommendations:")
        for rec in gap_analysis['recommendations'][:3]:
            print(f"  • {rec}")

        print("\n✓ Win Impact Model test complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
