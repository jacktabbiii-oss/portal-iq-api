"""
Transfer Portal Feature Engineering

Builds features for portal entry prediction and destination matching.
"""

import logging
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PortalFeatureBuilder:
    """Builds features for transfer portal analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def build_playing_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build playing time features.

        Args:
            df: Player stats DataFrame

        Returns:
            DataFrame with playing time features
        """
        features = df.copy()

        if "snap_count" in features.columns and "team_total_snaps" in features.columns:
            features["snap_share"] = (
                features["snap_count"] / features["team_total_snaps"].replace(0, 1)
            )
        else:
            features["snap_share"] = 0.5

        if "games_played" in features.columns and "games_available" in features.columns:
            features["games_played_pct"] = (
                features["games_played"] / features["games_available"].replace(0, 1)
            )
        else:
            features["games_played_pct"] = 1.0

        # Playing time trend (if multi-year data)
        if "year" in features.columns:
            features["playing_time_trend"] = features.groupby("player_id")[
                "snap_share"
            ].diff().fillna(0)

        return features

    def build_depth_chart_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build depth chart position features.

        Args:
            df: Player DataFrame with depth chart info

        Returns:
            DataFrame with depth chart features
        """
        features = df.copy()

        if "depth_chart_position" in features.columns:
            features["is_starter"] = features["depth_chart_position"] == 1
            features["is_backup"] = features["depth_chart_position"] == 2
            features["depth_score"] = 1 / features["depth_chart_position"].replace(0, 1)

        if "players_ahead" in features.columns:
            features["blocked_score"] = features["players_ahead"] / 3

        return features

    def build_eligibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build eligibility features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with eligibility features
        """
        features = df.copy()

        if "eligibility_year" in features.columns:
            features["years_remaining"] = 4 - features["eligibility_year"]
            features["is_senior"] = features["eligibility_year"] >= 4
            features["is_grad_transfer"] = features.get("is_graduate", False)

        if "redshirt" in features.columns:
            features["used_redshirt"] = features["redshirt"]

        return features

    def build_coaching_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build coaching change features.

        Args:
            df: Player DataFrame with coaching info

        Returns:
            DataFrame with coaching features
        """
        features = df.copy()

        if "coaching_change" in features.columns:
            features["new_coach"] = features["coaching_change"]

        if "coach_tenure" in features.columns:
            features["coach_stability"] = np.log1p(features["coach_tenure"]) / 3

        if "position_coach_change" in features.columns:
            features["pos_coach_changed"] = features["position_coach_change"]

        return features

    def build_transfer_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build transfer history features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with transfer history features
        """
        features = df.copy()

        if "previous_transfers" in features.columns:
            features["has_transferred"] = features["previous_transfers"] > 0
            features["transfer_count"] = features["previous_transfers"]

        return features

    def build_portal_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build composite portal risk features.

        Args:
            df: Player DataFrame

        Returns:
            DataFrame with portal risk score
        """
        features = df.copy()

        # Calculate individual risk factors
        risk_score = pd.Series(0.0, index=features.index)

        # Low playing time = higher risk
        if "snap_share" in features.columns:
            risk_score += (1 - features["snap_share"]) * 0.3

        # Non-starter = higher risk
        if "is_starter" in features.columns:
            risk_score += (~features["is_starter"]).astype(float) * 0.2

        # Coaching change = higher risk
        if "new_coach" in features.columns:
            risk_score += features["new_coach"].astype(float) * 0.2

        # More eligibility = more likely to transfer
        if "years_remaining" in features.columns:
            risk_score += (features["years_remaining"] / 4) * 0.15

        # Already transferred = less likely again (NCAA rules)
        if "has_transferred" in features.columns:
            risk_score -= features["has_transferred"].astype(float) * 0.15

        features["portal_risk_score"] = risk_score.clip(0, 1)

        return features

    def build_destination_features(
        self,
        player_df: pd.DataFrame,
        school_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build features for destination school matching.

        Args:
            player_df: Player DataFrame
            school_df: School DataFrame with needs

        Returns:
            DataFrame with destination match scores
        """
        # This would typically be more complex with actual school needs data
        features = player_df.copy()

        if "position" in features.columns and "school_needs" in school_df.columns:
            # Match players to school position needs
            pass

        return features

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all portal-related features.

        Args:
            df: Raw player DataFrame

        Returns:
            DataFrame with all features
        """
        features = df.copy()

        features = self.build_playing_time_features(features)
        features = self.build_depth_chart_features(features)
        features = self.build_eligibility_features(features)
        features = self.build_coaching_features(features)
        features = self.build_transfer_history_features(features)
        features = self.build_portal_risk_features(features)

        return features
