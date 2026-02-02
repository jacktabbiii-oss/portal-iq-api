"""
Transfer Portal Feature Engineering Module

Transforms raw data into features for:
1. Flight Risk Prediction - Will a player enter the portal?
2. Portal Fit Scoring - How well does a portal player fit a target school?

Author: Elite Sports Solutions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Position group mappings
POSITION_TO_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB",
    "WR": "WR", "SLOT": "WR", "TE": "TE",
    "OT": "OL", "OG": "OL", "C": "OL", "OL": "OL", "T": "OL", "G": "OL",
    "DE": "EDGE", "OLB": "EDGE", "EDGE": "EDGE", "RUSH": "EDGE",
    "DT": "DL", "NT": "DL", "DL": "DL", "DI": "DL",
    "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "CB", "DB": "CB", "NB": "CB",
    "S": "S", "FS": "S", "SS": "S", "SAF": "S",
    "K": "ST", "P": "ST", "LS": "ST",
    "ATH": "ATH",
}

# School tier encoding
SCHOOL_TIER_ENCODING = {
    "blue_blood": 6, "elite": 5, "power_brand": 4,
    "p4_mid": 3, "g5_strong": 2, "g5": 1, "fcs": 0,
}

# Conference tier encoding
CONFERENCE_TIER_ENCODING = {
    "SEC": 3, "Big Ten": 3, "BIG TEN": 3, "B1G": 3,
    "Big 12": 2, "BIG 12": 2, "ACC": 2, "Pac-12": 2,
    "Mountain West": 1, "MWC": 1, "American": 1, "AAC": 1,
    "Sun Belt": 1, "MAC": 1, "Conference USA": 1, "CUSA": 1,
    "Independent": 2, "FCS": 0,
}

# School tiers (subset - full list in nil_features.py)
SCHOOL_TIERS = {
    "Alabama": "blue_blood", "Ohio State": "blue_blood", "Notre Dame": "blue_blood",
    "USC": "blue_blood", "Oklahoma": "blue_blood", "Texas": "blue_blood",
    "Michigan": "blue_blood", "Nebraska": "blue_blood",
    "Georgia": "elite", "Clemson": "elite", "LSU": "elite", "Florida": "elite",
    "Penn State": "elite", "Florida State": "elite", "Miami": "elite",
    "Tennessee": "elite", "Auburn": "elite", "Oregon": "elite", "Texas A&M": "elite",
    "Wisconsin": "power_brand", "Iowa": "power_brand", "Michigan State": "power_brand",
    "UCLA": "power_brand", "Washington": "power_brand", "Arkansas": "power_brand",
    "Ole Miss": "power_brand", "South Carolina": "power_brand", "Kentucky": "power_brand",
    "Missouri": "power_brand", "Oklahoma State": "power_brand", "TCU": "power_brand",
    "Baylor": "power_brand", "Utah": "power_brand", "Arizona State": "power_brand",
    "Colorado": "power_brand", "Stanford": "power_brand",
    "Minnesota": "p4_mid", "Illinois": "p4_mid", "Purdue": "p4_mid", "Indiana": "p4_mid",
    "Northwestern": "p4_mid", "Maryland": "p4_mid", "Rutgers": "p4_mid",
    "Duke": "p4_mid", "North Carolina": "p4_mid", "Virginia": "p4_mid",
    "Wake Forest": "p4_mid", "Syracuse": "p4_mid", "Boston College": "p4_mid",
    "Vanderbilt": "p4_mid", "Kansas": "p4_mid", "Kansas State": "p4_mid",
    "Iowa State": "p4_mid", "Texas Tech": "p4_mid", "Houston": "p4_mid",
    "Cincinnati": "p4_mid", "UCF": "p4_mid", "BYU": "p4_mid",
    "Boise State": "g5_strong", "Memphis": "g5_strong", "SMU": "g5_strong",
    "Tulane": "g5_strong", "Appalachian State": "g5_strong", "Liberty": "g5_strong",
    "James Madison": "g5_strong", "San Diego State": "g5_strong",
}

# NIL tier thresholds (collective budgets)
SCHOOL_NIL_TIERS = {
    "blue_blood": 5, "elite": 4, "power_brand": 3,
    "p4_mid": 2, "g5_strong": 1, "g5": 0,
}

# Average production by star rating and position (from historical data)
# Format: {position_group: {stars: avg_production_score}}
STAR_AVG_PRODUCTION = {
    "QB": {5: 75, 4: 55, 3: 40, 2: 25, 0: 20},
    "RB": {5: 70, 4: 50, 3: 38, 2: 25, 0: 20},
    "WR": {5: 65, 4: 48, 3: 35, 2: 25, 0: 20},
    "TE": {5: 60, 4: 45, 3: 35, 2: 25, 0: 20},
    "OL": {5: 70, 4: 55, 3: 42, 2: 30, 0: 25},
    "EDGE": {5: 68, 4: 50, 3: 38, 2: 25, 0: 20},
    "DL": {5: 65, 4: 48, 3: 38, 2: 28, 0: 22},
    "LB": {5: 65, 4: 50, 3: 40, 2: 28, 0: 22},
    "CB": {5: 62, 4: 48, 3: 38, 2: 28, 0: 22},
    "S": {5: 60, 4: 45, 3: 38, 2: 28, 0: 22},
    "ATH": {5: 60, 4: 45, 3: 35, 2: 25, 0: 20},
    "ST": {5: 50, 4: 40, 3: 35, 2: 30, 0: 25},
}

# Historical portal entry rates by position (approximate)
POSITION_PORTAL_RATES = {
    "QB": 0.25,  # QBs transfer frequently for playing time
    "RB": 0.18,
    "WR": 0.20,
    "TE": 0.15,
    "OL": 0.12,  # OL transfer less frequently
    "EDGE": 0.16,
    "DL": 0.14,
    "LB": 0.15,
    "CB": 0.18,
    "S": 0.16,
    "ATH": 0.22,
    "ST": 0.10,
}

# School city coordinates for distance calculations (lat, lon)
SCHOOL_COORDINATES = {
    "Alabama": (33.2098, -87.5692), "Auburn": (32.6099, -85.4808),
    "Georgia": (33.9480, -83.3773), "Florida": (29.6436, -82.3549),
    "Tennessee": (35.9544, -83.9295), "Kentucky": (38.0306, -84.5039),
    "LSU": (30.4113, -91.1835), "Texas A&M": (30.6187, -96.3365),
    "Arkansas": (36.0675, -94.1749), "Ole Miss": (34.3647, -89.5186),
    "Mississippi State": (33.4552, -88.7901), "Missouri": (38.9404, -92.3277),
    "South Carolina": (34.0007, -81.0348), "Vanderbilt": (36.1447, -86.8027),
    "Ohio State": (40.0067, -83.0305), "Michigan": (42.2650, -83.7483),
    "Penn State": (40.7982, -77.8599), "Michigan State": (42.7251, -84.4791),
    "Wisconsin": (43.0766, -89.4125), "Iowa": (41.6611, -91.5302),
    "Minnesota": (44.9740, -93.2277), "Nebraska": (40.8202, -96.7005),
    "Northwestern": (42.0565, -87.6753), "Illinois": (40.1020, -88.2272),
    "Purdue": (40.4237, -86.9212), "Indiana": (39.1681, -86.5230),
    "Maryland": (38.9869, -76.9426), "Rutgers": (40.5018, -74.4479),
    "Texas": (30.2849, -97.7341), "Oklahoma": (35.2058, -97.4457),
    "USC": (34.0224, -118.2851), "UCLA": (34.0689, -118.4452),
    "Oregon": (44.0448, -123.0726), "Washington": (47.6553, -122.3035),
    "Notre Dame": (41.7052, -86.2350), "Clemson": (34.6834, -82.8374),
    "Florida State": (30.4383, -84.2807), "Miami": (25.7152, -80.2789),
    "Virginia Tech": (37.2296, -80.4139), "North Carolina": (35.9049, -79.0469),
    "NC State": (35.7872, -78.6705), "Duke": (36.0014, -78.9382),
    "Pittsburgh": (40.4443, -79.9608), "Louisville": (38.2146, -85.7589),
    "Boston College": (42.3355, -71.1685), "Syracuse": (43.0481, -76.1474),
    "Wake Forest": (36.1346, -80.2776), "Virginia": (38.0336, -78.5080),
    "Georgia Tech": (33.7756, -84.3963), "Colorado": (40.0076, -105.2659),
    "Arizona State": (33.4255, -111.9400), "Arizona": (32.2319, -110.9501),
    "Utah": (40.7649, -111.8421), "Stanford": (37.4346, -122.1609),
    "Cal": (37.8716, -122.2727), "Oregon State": (44.5646, -123.2620),
    "Washington State": (46.7298, -117.1817), "TCU": (32.7096, -97.3628),
    "Baylor": (31.5493, -97.1189), "Texas Tech": (33.5843, -101.8783),
    "Oklahoma State": (36.1264, -97.0716), "Kansas State": (39.2014, -96.5847),
    "Kansas": (38.9543, -95.2558), "Iowa State": (42.0140, -93.6358),
    "West Virginia": (39.6480, -79.9564), "Cincinnati": (39.1329, -84.5150),
    "UCF": (28.6024, -81.2001), "Houston": (29.7199, -95.3422),
    "BYU": (40.2518, -111.6493), "Boise State": (43.6036, -116.1974),
    "Memphis": (35.1174, -89.9711), "SMU": (32.8412, -96.7852),
    "Tulane": (29.9400, -90.1222), "San Diego State": (32.7757, -117.0719),
    "Fresno State": (36.8136, -119.7485), "Nevada": (39.5461, -119.8173),
    "UNLV": (36.1086, -115.1439), "Hawaii": (21.2969, -157.8171),
    "Air Force": (38.9983, -104.8614), "Army": (41.3915, -73.9565),
    "Navy": (38.9869, -76.4853),
}

# State coordinates (for hometown distance estimates)
STATE_COORDINATES = {
    "AL": (32.806671, -86.791130), "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221), "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371), "DE": (39.318523, -75.507141),
    "FL": (27.766279, -81.686783), "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337), "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137), "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526), "KS": (38.526600, -96.726486),
    "KY": (37.668140, -84.670067), "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927), "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106), "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192), "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368), "MT": (46.921925, -110.454353),
    "NE": (41.125370, -98.268082), "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896), "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482), "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419), "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915), "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938), "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.511780), "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828), "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461), "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686), "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494), "WV": (38.491226, -80.954453),
    "WI": (44.268543, -89.616508), "WY": (42.755966, -107.302490),
    "DC": (38.897438, -77.026817),
}


class PortalFeatureEngineer:
    """
    Feature engineering for transfer portal prediction models.

    Two main feature sets:
    1. Flight Risk Features - Predict if a player will enter the portal
    2. Portal Fit Features - Score how well a portal entrant fits a target school
    """

    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize the portal feature engineer.

        Args:
            output_dir: Directory to save processed features
        """
        self.output_dir = output_dir
        self.imputation_log = []
        self.flight_risk_features = []
        self.portal_fit_features = []

    # =========================================================================
    # FLIGHT RISK FEATURES
    # =========================================================================

    def build_flight_risk_features(
        self,
        player_data: pd.DataFrame,
        team_data: pd.DataFrame,
        nil_data: pd.DataFrame,
        recruiting_data: pd.DataFrame,
        portal_history: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build features predicting whether a player will enter the transfer portal.

        Args:
            player_data: Player stats and info
            team_data: Team stats, coaching changes
            nil_data: NIL valuations
            recruiting_data: Recruiting rankings
            portal_history: Historical portal entries

        Returns:
            DataFrame with flight risk features, one row per player-season
        """
        logger.info("Building flight risk features...")
        self.imputation_log = []

        # Start with player data
        df = player_data.copy()
        df = self._standardize_columns(df)
        logger.info(f"Starting with {len(df)} player-seasons")

        # Merge data sources
        df = self._merge_for_flight_risk(df, team_data, nil_data, recruiting_data, portal_history)

        # Build feature groups
        df = self._build_playing_time_features(df, team_data)
        df = self._build_performance_vs_expectation_features(df, recruiting_data)
        df = self._build_team_context_features(df, team_data)
        df = self._build_positional_context_features(df, player_data, recruiting_data)
        df = self._build_personal_geographic_features(df)
        df = self._build_nil_context_features(df, nil_data)
        df = self._build_portal_pattern_features(df, portal_history)
        df = self._build_flight_risk_target(df, portal_history)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Select final features
        df = self._select_flight_risk_features(df)

        # Save
        self._save_flight_risk_features(df)

        logger.info(f"Flight risk features complete: {len(df)} rows, {len(self.flight_risk_features)} features")
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and create standard identifiers."""

        # Create player name standard
        name_cols = ['name', 'player_name', 'player', 'athlete_name']
        for col in name_cols:
            if col in df.columns:
                df['player_name_std'] = (
                    df[col].astype(str).str.lower()
                    .str.replace(r'[^a-z\s]', '', regex=True)
                    .str.strip().str.replace(r'\s+', ' ', regex=True)
                )
                break
        if 'player_name_std' not in df.columns:
            df['player_name_std'] = df.index.astype(str)

        # Standardize school name
        school_cols = ['school', 'team', 'college', 'university']
        for col in school_cols:
            if col in df.columns:
                df['school_name'] = df[col].astype(str).str.strip()
                break
        if 'school_name' not in df.columns:
            df['school_name'] = 'Unknown'

        # Standardize position
        pos_cols = ['position', 'pos', 'player_position']
        for col in pos_cols:
            if col in df.columns:
                df['position_raw'] = df[col].astype(str).str.upper().str.strip()
                df['position_group'] = df['position_raw'].map(POSITION_TO_GROUP).fillna('ATH')
                break
        if 'position_group' not in df.columns:
            df['position_raw'] = 'ATH'
            df['position_group'] = 'ATH'

        # Standardize season/year
        if 'season' in df.columns:
            df['season'] = df['season'].astype(int)
        elif 'year' in df.columns and df['year'].dtype in ['int64', 'float64']:
            df['season'] = df['year'].astype(int)
        else:
            df['season'] = datetime.now().year

        return df

    def _merge_for_flight_risk(
        self,
        df: pd.DataFrame,
        team_data: pd.DataFrame,
        nil_data: pd.DataFrame,
        recruiting_data: pd.DataFrame,
        portal_history: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all data sources for flight risk analysis."""

        # Merge team data
        if not team_data.empty:
            team_data = team_data.copy()
            if 'team' in team_data.columns:
                team_data['school_name'] = team_data['team'].astype(str).str.strip()
            team_cols = ['school_name', 'season', 'wins', 'losses', 'conference',
                        'coaching_change', 'head_coach', 'prev_wins']
            team_cols = [c for c in team_cols if c in team_data.columns]
            if 'season' in team_data.columns:
                df = df.merge(team_data[team_cols], on=['school_name', 'season'],
                             how='left', suffixes=('', '_team'))
            else:
                df = df.merge(team_data[team_cols], on='school_name',
                             how='left', suffixes=('', '_team'))

        # Merge NIL data
        if not nil_data.empty:
            nil_data = nil_data.copy()
            nil_data = self._standardize_nil_data(nil_data)
            nil_cols = ['player_name_std', 'nil_value', 'nil_tier']
            nil_cols = [c for c in nil_cols if c in nil_data.columns]
            df = df.merge(nil_data[nil_cols], on='player_name_std',
                         how='left', suffixes=('', '_nil'))

        # Merge recruiting data
        if not recruiting_data.empty:
            recruiting_data = recruiting_data.copy()
            recruiting_data = self._standardize_recruiting_data(recruiting_data)
            recruit_cols = ['player_name_std', 'stars', 'rating', 'national_rank']
            recruit_cols = [c for c in recruit_cols if c in recruiting_data.columns]
            df = df.merge(recruiting_data[recruit_cols], on='player_name_std',
                         how='left', suffixes=('', '_recruit'))

        return df

    def _standardize_nil_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize NIL data columns."""
        name_cols = ['name', 'player_name', 'player']
        for col in name_cols:
            if col in df.columns:
                df['player_name_std'] = (
                    df[col].astype(str).str.lower()
                    .str.replace(r'[^a-z\s]', '', regex=True).str.strip()
                )
                break
        return df

    def _standardize_recruiting_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize recruiting data columns."""
        name_cols = ['name', 'player_name', 'player', 'athlete_name']
        for col in name_cols:
            if col in df.columns:
                df['player_name_std'] = (
                    df[col].astype(str).str.lower()
                    .str.replace(r'[^a-z\s]', '', regex=True).str.strip()
                )
                break
        return df

    def _build_playing_time_features(self, df: pd.DataFrame, team_data: pd.DataFrame) -> pd.DataFrame:
        """Build playing time features."""

        # Snap percentage estimate (games started / team games)
        if 'games_started' in df.columns and 'games' in df.columns:
            df['snap_pct'] = df['games_started'] / df['games'].clip(lower=1)
        elif 'games_played' in df.columns and 'games' in df.columns:
            df['snap_pct'] = df['games_played'] / df['games'].clip(lower=1) * 0.7
        else:
            df['snap_pct'] = 0.5
            self._log_imputation("snap_pct defaulted to 0.5")
        df['snap_pct'] = df['snap_pct'].clip(0, 1)

        # Snap trend (year over year change)
        if 'prev_snap_pct' in df.columns:
            df['snap_trend'] = df['snap_pct'] - df['prev_snap_pct']
        else:
            # Calculate from historical data if available
            df['snap_trend'] = 0
            self._log_imputation("snap_trend defaulted to 0")

        # Is starter
        df['is_starter'] = (df['snap_pct'] >= 0.6).astype(int)

        # Depth chart position estimate (1=starter, 2=backup, 3+=buried)
        def estimate_depth(snap_pct):
            if snap_pct >= 0.6:
                return 1
            elif snap_pct >= 0.3:
                return 2
            else:
                return 3

        df['depth_chart_position'] = df['snap_pct'].apply(estimate_depth)

        # Career starts
        if 'career_games_started' in df.columns:
            df['career_starts'] = df['career_games_started']
        elif 'games_started' in df.columns:
            df['career_starts'] = df['games_started']  # Current season only
        else:
            df['career_starts'] = 0
            self._log_imputation("career_starts defaulted to 0")

        # Games played percentage
        if 'career_games_played' in df.columns and 'career_games_available' in df.columns:
            df['games_played_pct'] = df['career_games_played'] / df['career_games_available'].clip(lower=1)
        elif 'games_played' in df.columns:
            df['games_played_pct'] = df['games_played'] / 13  # Assume 13-game season
        else:
            df['games_played_pct'] = 0.5
            self._log_imputation("games_played_pct defaulted to 0.5")
        df['games_played_pct'] = df['games_played_pct'].clip(0, 1)

        return df

    def _build_performance_vs_expectation_features(
        self, df: pd.DataFrame, recruiting_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Build performance vs expectation features."""

        # Recruiting stars
        if 'stars' in df.columns:
            df['recruiting_stars'] = df['stars'].fillna(0).astype(int).clip(0, 5)
        else:
            df['recruiting_stars'] = 0
            self._log_imputation("recruiting_stars defaulted to 0")

        # Calculate production score if not present
        if 'production_score' not in df.columns:
            df['production_score'] = 50  # Default median
            self._log_imputation("production_score defaulted to 50")

        # Production vs star average
        def calc_production_vs_expected(row):
            pos = row.get('position_group', 'ATH')
            stars = int(row.get('recruiting_stars', 0))
            production = row.get('production_score', 50)

            expected = STAR_AVG_PRODUCTION.get(pos, {}).get(stars, 40)
            return production - expected

        df['production_vs_star_avg'] = df.apply(calc_production_vs_expected, axis=1)

        # Production trend (year over year)
        if 'prev_production_score' in df.columns:
            df['production_trend'] = df['production_score'] - df['prev_production_score']
        else:
            df['production_trend'] = 0
            self._log_imputation("production_trend defaulted to 0")

        # Breakout flag (significantly outperformed expectations)
        df['breakout_flag'] = (df['production_vs_star_avg'] > 20).astype(int)

        return df

    def _build_team_context_features(self, df: pd.DataFrame, team_data: pd.DataFrame) -> pd.DataFrame:
        """Build team context features."""

        # Team wins
        if 'wins' in df.columns:
            df['team_wins'] = df['wins'].fillna(6)
        else:
            df['team_wins'] = 6
            self._log_imputation("team_wins defaulted to 6")

        # Team win trend
        if 'wins' in df.columns and 'prev_wins' in df.columns:
            df['team_win_trend'] = df['wins'] - df['prev_wins']
        else:
            df['team_win_trend'] = 0
            self._log_imputation("team_win_trend defaulted to 0")

        # Coaching change flag
        if 'coaching_change' in df.columns:
            df['coaching_change'] = df['coaching_change'].fillna(0).astype(int)
        else:
            df['coaching_change'] = 0
            self._log_imputation("coaching_change defaulted to 0")

        # Coordinator change (if available)
        if 'coordinator_change' in df.columns:
            df['coordinator_change'] = df['coordinator_change'].fillna(0).astype(int)
        else:
            df['coordinator_change'] = 0

        # Team NIL tier
        df['school_tier_name'] = df['school_name'].map(SCHOOL_TIERS).fillna('g5')
        df['team_nil_tier'] = df['school_tier_name'].map(SCHOOL_NIL_TIERS).fillna(0)

        # Conference tier
        if 'conference' in df.columns:
            df['conference_tier'] = df['conference'].map(CONFERENCE_TIER_ENCODING).fillna(1)
        else:
            df['conference_tier'] = df['school_tier_name'].apply(
                lambda x: 3 if x in ['blue_blood', 'elite'] else (2 if x in ['power_brand', 'p4_mid'] else 1)
            )
            self._log_imputation("conference_tier inferred from school tier")

        # School tier (numeric)
        df['school_tier'] = df['school_tier_name'].map(SCHOOL_TIER_ENCODING).fillna(1)

        return df

    def _build_positional_context_features(
        self,
        df: pd.DataFrame,
        player_data: pd.DataFrame,
        recruiting_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Build positional context features."""

        # Position depth (players at same position on roster)
        position_counts = df.groupby(['school_name', 'season', 'position_group']).size()
        df['position_depth'] = df.apply(
            lambda row: position_counts.get(
                (row['school_name'], row['season'], row['position_group']), 5
            ), axis=1
        )

        # Incoming recruits at position
        if not recruiting_data.empty and 'committed_school' in recruiting_data.columns:
            # Count high-rated incoming recruits by position
            recruiting_data = recruiting_data.copy()
            if 'position' in recruiting_data.columns:
                recruiting_data['position_group'] = recruiting_data['position'].str.upper().map(
                    POSITION_TO_GROUP
                ).fillna('ATH')
            if 'stars' in recruiting_data.columns:
                high_recruits = recruiting_data[recruiting_data['stars'] >= 4]
                incoming_counts = high_recruits.groupby(
                    ['committed_school', 'position_group']
                ).size().to_dict()

                df['incoming_recruits_at_position'] = df.apply(
                    lambda row: incoming_counts.get(
                        (row['school_name'], row['position_group']), 0
                    ), axis=1
                )
            else:
                df['incoming_recruits_at_position'] = 0
        else:
            df['incoming_recruits_at_position'] = 0
            self._log_imputation("incoming_recruits_at_position defaulted to 0")

        # Position group encoded (for model)
        position_encoding = {
            'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'OL': 5,
            'EDGE': 6, 'DL': 7, 'LB': 8, 'CB': 9, 'S': 10,
            'ATH': 11, 'ST': 12
        }
        df['position_group_encoded'] = df['position_group'].map(position_encoding).fillna(11)

        return df

    def _build_personal_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build personal and geographic features."""

        # Distance from home (if hometown data available)
        if 'hometown_state' in df.columns or 'home_state' in df.columns:
            state_col = 'hometown_state' if 'hometown_state' in df.columns else 'home_state'

            def calc_distance(row):
                school = row['school_name']
                state = row[state_col]

                school_coords = SCHOOL_COORDINATES.get(school)
                state_coords = STATE_COORDINATES.get(state)

                if school_coords and state_coords:
                    return self._haversine_distance(school_coords, state_coords)
                return 500  # Default 500 miles

            df['distance_from_home'] = df.apply(calc_distance, axis=1)
        else:
            df['distance_from_home'] = 500
            self._log_imputation("distance_from_home defaulted to 500 miles")

        # Years at school
        if 'years_at_school' in df.columns:
            df['years_at_school'] = df['years_at_school'].clip(1, 6)
        elif 'year_in_school' in df.columns:
            year_map = {'FR': 1, 'SO': 2, 'JR': 3, 'SR': 4, 'GR': 5}
            df['years_at_school'] = df['year_in_school'].map(year_map).fillna(2)
        else:
            df['years_at_school'] = 2
            self._log_imputation("years_at_school defaulted to 2")

        # Grad transfer eligible
        if 'is_graduate' in df.columns:
            df['is_grad_transfer_eligible'] = df['is_graduate'].astype(int)
        else:
            df['is_grad_transfer_eligible'] = (df['years_at_school'] >= 4).astype(int)
            self._log_imputation("is_grad_transfer_eligible inferred from years")

        # Remaining eligibility
        df['remaining_eligibility'] = (5 - df['years_at_school']).clip(0, 4)

        return df

    def _build_nil_context_features(self, df: pd.DataFrame, nil_data: pd.DataFrame) -> pd.DataFrame:
        """Build NIL context features."""

        # Estimated NIL value
        if 'nil_value' in df.columns:
            df['estimated_nil_value'] = df['nil_value'].fillna(0)
        else:
            df['estimated_nil_value'] = 0
            self._log_imputation("estimated_nil_value defaulted to 0")

        # NIL vs team median
        team_median_nil = df.groupby('school_name')['estimated_nil_value'].transform('median')
        df['nil_vs_team_median'] = df['estimated_nil_value'] - team_median_nil
        df['nil_vs_team_median'] = df['nil_vs_team_median'].fillna(0)

        # NIL vs position average (nationally)
        position_avg_nil = df.groupby('position_group')['estimated_nil_value'].transform('mean')
        df['nil_vs_position_avg'] = df['estimated_nil_value'] - position_avg_nil
        df['nil_vs_position_avg'] = df['nil_vs_position_avg'].fillna(0)

        # NIL could increase elsewhere
        # If player is at lower-tier school with good production, higher tier might pay more
        df['nil_could_increase_elsewhere'] = (
            (df['school_tier'] <= 3) &
            (df['production_score'] > 60) &
            (df['estimated_nil_value'] < 100000)
        ).astype(int)

        return df

    def _build_portal_pattern_features(
        self, df: pd.DataFrame, portal_history: pd.DataFrame
    ) -> pd.DataFrame:
        """Build historical portal pattern features."""

        # Position portal rate
        df['position_portal_rate'] = df['position_group'].map(POSITION_PORTAL_RATES).fillna(0.15)

        # School portal rate (from history if available)
        if not portal_history.empty and 'origin_school' in portal_history.columns:
            # Calculate actual rates from history
            portal_history = portal_history.copy()
            school_portal_counts = portal_history.groupby('origin_school').size()
            # Estimate total roster as ~85 players, ~3 years of data
            school_portal_rates = (school_portal_counts / 255).clip(0, 0.5).to_dict()
            df['school_portal_rate'] = df['school_name'].map(school_portal_rates).fillna(0.15)
        else:
            # Estimate from school tier (lower tiers have higher rates)
            df['school_portal_rate'] = 0.20 - (df['school_tier'] * 0.02)
            self._log_imputation("school_portal_rate estimated from school tier")

        # Conference portal rate
        if 'conference' in df.columns and not portal_history.empty:
            # Would calculate from history
            df['conference_portal_rate'] = 0.15
        else:
            df['conference_portal_rate'] = 0.15
            self._log_imputation("conference_portal_rate defaulted to 0.15")

        return df

    def _build_flight_risk_target(self, df: pd.DataFrame, portal_history: pd.DataFrame) -> pd.DataFrame:
        """Build the target variable: entered_portal."""

        df['entered_portal'] = 0  # Default to not entered

        if not portal_history.empty:
            portal_history = portal_history.copy()

            # Standardize names in portal history
            name_cols = ['name', 'player_name', 'player']
            for col in name_cols:
                if col in portal_history.columns:
                    portal_history['player_name_std'] = (
                        portal_history[col].astype(str).str.lower()
                        .str.replace(r'[^a-z\s]', '', regex=True).str.strip()
                    )
                    break

            if 'player_name_std' in portal_history.columns:
                # Get set of portal entrants by season
                if 'season' in portal_history.columns or 'entry_year' in portal_history.columns:
                    season_col = 'season' if 'season' in portal_history.columns else 'entry_year'
                    portal_entries = set(
                        zip(portal_history['player_name_std'], portal_history[season_col])
                    )
                    df['entered_portal'] = df.apply(
                        lambda row: 1 if (row['player_name_std'], row['season']) in portal_entries else 0,
                        axis=1
                    )
                else:
                    portal_names = set(portal_history['player_name_std'])
                    df['entered_portal'] = df['player_name_std'].isin(portal_names).astype(int)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate imputation."""

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = ['entered_portal']
        numeric_cols = [c for c in numeric_cols if c not in target_cols]

        for col in numeric_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                # Use position-group median for player stats
                if col in ['production_score', 'snap_pct', 'career_starts']:
                    group_median = df.groupby('position_group')[col].transform('median')
                    df[col] = df[col].fillna(group_median)

                # Fill remaining with overall median
                remaining = df[col].isna().sum()
                if remaining > 0:
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df[col] = df[col].fillna(median_val)
                    self._log_imputation(f"{col}: {remaining} values filled with median ({median_val:.2f})")

        return df

    def _select_flight_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order final flight risk features."""

        self.flight_risk_features = [
            # Playing time
            'snap_pct', 'snap_trend', 'is_starter', 'depth_chart_position',
            'career_starts', 'games_played_pct',

            # Performance vs expectation
            'recruiting_stars', 'production_vs_star_avg', 'production_trend', 'breakout_flag',

            # Team context
            'team_wins', 'team_win_trend', 'coaching_change', 'coordinator_change',
            'team_nil_tier', 'conference_tier', 'school_tier',

            # Positional context
            'position_depth', 'incoming_recruits_at_position', 'position_group_encoded',

            # Personal/geographic
            'distance_from_home', 'years_at_school', 'is_grad_transfer_eligible',
            'remaining_eligibility',

            # NIL context
            'estimated_nil_value', 'nil_vs_team_median', 'nil_vs_position_avg',
            'nil_could_increase_elsewhere',

            # Portal patterns
            'position_portal_rate', 'school_portal_rate', 'conference_portal_rate',
        ]

        # Filter to existing columns
        self.flight_risk_features = [c for c in self.flight_risk_features if c in df.columns]

        # Metadata columns
        metadata_cols = ['player_name_std', 'school_name', 'position_group', 'season']
        metadata_cols = [c for c in metadata_cols if c in df.columns]

        # Target
        target_cols = ['entered_portal']

        final_cols = metadata_cols + self.flight_risk_features + target_cols
        return df[final_cols]

    def _save_flight_risk_features(self, df: pd.DataFrame) -> None:
        """Save flight risk features to CSV."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'portal_flight_risk_features.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Flight risk features saved to {output_path}")

    # =========================================================================
    # PORTAL FIT FEATURES
    # =========================================================================

    def build_portal_fit_features(
        self,
        portal_player: pd.Series,
        target_school: str,
        school_roster: pd.DataFrame,
        team_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Score how well a portal player fits a specific target school.

        Args:
            portal_player: Series with portal player's data
            target_school: Name of the target school
            school_roster: Current roster of the target school
            team_data: Team statistics and info

        Returns:
            Single-row DataFrame with fit features
        """
        features = {}

        # Get player info
        player_position = portal_player.get('position_group', 'ATH')
        player_production = portal_player.get('production_score', 50)
        player_nil_value = portal_player.get('nil_value', 0)
        player_hometown_state = portal_player.get('hometown_state', None)
        origin_school = portal_player.get('school_name', 'Unknown')

        # Get target school info
        school_tier_name = SCHOOL_TIERS.get(target_school, 'g5')
        school_tier = SCHOOL_TIER_ENCODING.get(school_tier_name, 1)
        school_nil_tier = SCHOOL_NIL_TIERS.get(school_tier_name, 0)

        # Origin school info
        origin_tier_name = SCHOOL_TIERS.get(origin_school, 'g5')
        origin_tier = SCHOOL_TIER_ENCODING.get(origin_tier_name, 1)

        # 1. Positional need score
        features['positional_need_score'] = self._calculate_positional_need(
            player_position, school_roster, team_data, target_school
        )

        # 2. Production upgrade
        features['production_upgrade'] = self._calculate_production_upgrade(
            player_position, player_production, school_roster
        )

        # 3. School tier match
        tier_diff = school_tier - origin_tier
        features['school_tier_match'] = self._score_tier_match(tier_diff)

        # 4. Conference level match
        target_conf_tier = self._get_conference_tier(target_school, team_data)
        origin_conf_tier = self._get_conference_tier(origin_school, team_data)
        conf_diff = target_conf_tier - origin_conf_tier
        features['conference_level_match'] = self._score_tier_match(conf_diff)

        # 5. Scheme fit estimate (basic proxy)
        features['scheme_fit_estimate'] = self._estimate_scheme_fit(
            portal_player, target_school, team_data
        )

        # 6. Geographic proximity
        features['geographic_proximity'] = self._calculate_geographic_score(
            player_hometown_state, target_school
        )

        # 7. NIL budget fit
        features['nil_budget_fit'] = self._score_nil_budget_fit(
            player_nil_value, school_nil_tier
        )

        # 8. Academic fit (placeholder - would need academic data)
        features['academic_fit'] = 50  # Neutral default

        # 9. Returning production need
        features['returning_production_need'] = self._calculate_returning_production_need(
            player_position, school_roster, team_data, target_school
        )

        # 10. Team win trajectory
        features['team_win_trajectory'] = self._calculate_win_trajectory(
            target_school, team_data
        )

        # Overall fit score (weighted average)
        features['overall_fit_score'] = (
            features['positional_need_score'] * 0.25 +
            features['production_upgrade'] * 0.20 +
            features['school_tier_match'] * 0.10 +
            features['scheme_fit_estimate'] * 0.10 +
            features['geographic_proximity'] * 0.05 +
            features['nil_budget_fit'] * 0.15 +
            features['returning_production_need'] * 0.10 +
            features['team_win_trajectory'] * 0.05
        )

        # Create DataFrame
        result = pd.DataFrame([features])
        result['player_name'] = portal_player.get('player_name_std', 'unknown')
        result['target_school'] = target_school
        result['position_group'] = player_position

        # Save feature names
        self.portal_fit_features = list(features.keys())

        return result

    def build_portal_fit_batch(
        self,
        portal_players: pd.DataFrame,
        target_schools: List[str],
        school_rosters: Dict[str, pd.DataFrame],
        team_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build fit features for multiple portal players against multiple schools.

        Args:
            portal_players: DataFrame of portal players
            target_schools: List of target school names
            school_rosters: Dict mapping school name to roster DataFrame
            team_data: Team statistics

        Returns:
            DataFrame with fit features for all player-school combinations
        """
        results = []

        for idx, player in portal_players.iterrows():
            for school in target_schools:
                roster = school_rosters.get(school, pd.DataFrame())
                fit_df = self.build_portal_fit_features(player, school, roster, team_data)
                results.append(fit_df)

        if results:
            combined = pd.concat(results, ignore_index=True)
            self._save_portal_fit_features(combined)
            return combined
        else:
            return pd.DataFrame()

    def _calculate_positional_need(
        self,
        position: str,
        roster: pd.DataFrame,
        team_data: pd.DataFrame,
        school: str
    ) -> float:
        """Calculate positional need score (0-100)."""

        # Ideal roster sizes by position
        ideal_counts = {
            'QB': 3, 'RB': 4, 'WR': 8, 'TE': 3, 'OL': 12,
            'EDGE': 6, 'DL': 8, 'LB': 6, 'CB': 6, 'S': 4,
            'ATH': 2, 'ST': 3
        }

        ideal = ideal_counts.get(position, 5)

        if roster.empty:
            return 75  # High need if no roster data

        # Count current players at position
        if 'position_group' in roster.columns:
            current = (roster['position_group'] == position).sum()
        elif 'position' in roster.columns:
            roster_positions = roster['position'].str.upper().map(POSITION_TO_GROUP)
            current = (roster_positions == position).sum()
        else:
            current = ideal  # Assume full if no data

        # Calculate need (more need if under ideal)
        if current < ideal * 0.5:
            return 90  # Critical need
        elif current < ideal * 0.75:
            return 75  # High need
        elif current < ideal:
            return 60  # Moderate need
        elif current < ideal * 1.25:
            return 40  # Low need
        else:
            return 20  # Overloaded

    def _calculate_production_upgrade(
        self,
        position: str,
        player_production: float,
        roster: pd.DataFrame
    ) -> float:
        """Calculate how much of an upgrade this player would be."""

        if roster.empty:
            return 50  # Neutral if no data

        # Get current starters at position
        if 'position_group' in roster.columns:
            position_players = roster[roster['position_group'] == position]
        else:
            return 50

        if position_players.empty:
            return 80  # Big upgrade if no one at position

        if 'production_score' in position_players.columns:
            best_current = position_players['production_score'].max()
            upgrade = player_production - best_current

            # Scale to 0-100 (upgrade of +30 = 100, -30 = 0)
            return np.clip(50 + (upgrade * 1.67), 0, 100)
        else:
            return 50

    def _score_tier_match(self, tier_diff: int) -> float:
        """Score tier match (0-100). Same tier = best, moving up 1 = good, etc."""

        if tier_diff == 0:
            return 90  # Same tier - good fit
        elif tier_diff == 1:
            return 70  # Moving up 1 tier - realistic reach
        elif tier_diff == -1:
            return 80  # Moving down 1 tier - safe choice
        elif tier_diff == 2:
            return 40  # Big jump up - risky
        elif tier_diff == -2:
            return 60  # Big step down
        else:
            return max(20, 90 - abs(tier_diff) * 20)

    def _get_conference_tier(self, school: str, team_data: pd.DataFrame) -> int:
        """Get conference tier for a school."""

        if not team_data.empty:
            school_data = team_data[team_data['team'] == school] if 'team' in team_data.columns else pd.DataFrame()
            if not school_data.empty and 'conference' in school_data.columns:
                conf = school_data['conference'].iloc[0]
                return CONFERENCE_TIER_ENCODING.get(conf, 1)

        # Infer from school tier
        tier_name = SCHOOL_TIERS.get(school, 'g5')
        if tier_name in ['blue_blood', 'elite']:
            return 3
        elif tier_name in ['power_brand', 'p4_mid']:
            return 2
        else:
            return 1

    def _estimate_scheme_fit(
        self,
        player: pd.Series,
        target_school: str,
        team_data: pd.DataFrame
    ) -> float:
        """Estimate scheme fit (simplified proxy)."""

        position = player.get('position_group', 'ATH')

        # For QBs, check passing vs rushing balance
        if position == 'QB':
            pass_yards = player.get('passing_yards', 0)
            rush_yards = player.get('rushing_yards', 0)

            # Mobile QB (>400 rush yards) fits spread systems better
            is_mobile = rush_yards > 400

            # Simplified: assume blue bloods/elite run more pro-style
            school_tier = SCHOOL_TIERS.get(target_school, 'g5')
            prefers_pocket = school_tier in ['blue_blood', 'elite']

            if is_mobile and not prefers_pocket:
                return 80
            elif not is_mobile and prefers_pocket:
                return 80
            else:
                return 50

        # For other positions, return neutral
        return 60

    def _calculate_geographic_score(self, hometown_state: str, target_school: str) -> float:
        """Score geographic proximity (0-100, higher = closer)."""

        school_coords = SCHOOL_COORDINATES.get(target_school)
        state_coords = STATE_COORDINATES.get(hometown_state)

        if not school_coords or not state_coords:
            return 50  # Neutral if unknown

        distance = self._haversine_distance(school_coords, state_coords)

        # Score: <100mi = 100, 100-300 = 80, 300-500 = 60, 500-1000 = 40, >1000 = 20
        if distance < 100:
            return 100
        elif distance < 300:
            return 80
        elif distance < 500:
            return 60
        elif distance < 1000:
            return 40
        else:
            return 20

    def _score_nil_budget_fit(self, player_nil_value: float, school_nil_tier: int) -> float:
        """Score whether school can afford the player."""

        # Estimate school NIL budget by tier
        tier_budgets = {5: 20_000_000, 4: 12_000_000, 3: 6_000_000, 2: 2_000_000, 1: 500_000, 0: 100_000}
        school_budget = tier_budgets.get(school_nil_tier, 1_000_000)

        # Can school afford this player? (player shouldn't be >10% of budget)
        max_affordable = school_budget * 0.10

        if player_nil_value <= max_affordable * 0.5:
            return 90  # Easily affordable
        elif player_nil_value <= max_affordable:
            return 70  # Affordable
        elif player_nil_value <= max_affordable * 1.5:
            return 50  # Stretch
        elif player_nil_value <= max_affordable * 2:
            return 30  # Difficult
        else:
            return 10  # Probably can't afford

    def _calculate_returning_production_need(
        self,
        position: str,
        roster: pd.DataFrame,
        team_data: pd.DataFrame,
        school: str
    ) -> float:
        """Calculate how much production the team lost at this position."""

        if roster.empty:
            return 70  # Assume moderate need

        # Check for departures (seniors, declared for draft, portal exits)
        if 'year_in_school' in roster.columns:
            position_players = roster[roster.get('position_group', roster.get('position', '')) == position]
            if position_players.empty:
                return 80

            # Count departing production
            departing = position_players[position_players['year_in_school'].isin(['SR', 'GR', '4', '5'])]
            if 'production_score' in departing.columns:
                lost_production = departing['production_score'].sum()
                total_production = position_players['production_score'].sum()

                if total_production > 0:
                    pct_lost = lost_production / total_production
                    return min(100, pct_lost * 100 + 30)

        return 50  # Default moderate need

    def _calculate_win_trajectory(self, school: str, team_data: pd.DataFrame) -> float:
        """Calculate team's win trajectory (0-100, higher = improving)."""

        if team_data.empty:
            return 50

        school_data = team_data[team_data['team'] == school] if 'team' in team_data.columns else pd.DataFrame()

        if school_data.empty:
            return 50

        if 'wins' in school_data.columns and 'prev_wins' in school_data.columns:
            wins = school_data['wins'].iloc[-1] if len(school_data) > 0 else 6
            prev_wins = school_data['prev_wins'].iloc[-1] if len(school_data) > 0 else 6
            trend = wins - prev_wins

            # +4 or more = 100, -4 or less = 0
            return np.clip(50 + (trend * 12.5), 0, 100)

        elif 'wins' in school_data.columns:
            wins = school_data['wins'].iloc[-1] if len(school_data) > 0 else 6
            # More wins = better trajectory (assuming)
            return np.clip(wins * 8, 20, 90)

        return 50

    def _save_portal_fit_features(self, df: pd.DataFrame) -> None:
        """Save portal fit features to CSV."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'portal_fit_features.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Portal fit features saved to {output_path}")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates in miles."""

        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Earth's radius in miles
        r = 3956

        return c * r

    def _log_imputation(self, message: str) -> None:
        """Log an imputation decision."""
        self.imputation_log.append(message)
        logger.debug(f"Imputation: {message}")

    def get_flight_risk_feature_names(self) -> List[str]:
        """Get list of flight risk feature names."""
        return self.flight_risk_features.copy()

    def get_portal_fit_feature_names(self) -> List[str]:
        """Get list of portal fit feature names."""
        return self.portal_fit_features.copy()

    def get_flight_risk_feature_groups(self) -> Dict[str, List[str]]:
        """Get flight risk features grouped by category for SHAP analysis."""
        return {
            'playing_time': [
                'snap_pct', 'snap_trend', 'is_starter', 'depth_chart_position',
                'career_starts', 'games_played_pct',
            ],
            'performance_expectation': [
                'recruiting_stars', 'production_vs_star_avg', 'production_trend', 'breakout_flag',
            ],
            'team_context': [
                'team_wins', 'team_win_trend', 'coaching_change', 'coordinator_change',
                'team_nil_tier', 'conference_tier', 'school_tier',
            ],
            'positional_context': [
                'position_depth', 'incoming_recruits_at_position', 'position_group_encoded',
            ],
            'personal_geographic': [
                'distance_from_home', 'years_at_school', 'is_grad_transfer_eligible',
                'remaining_eligibility',
            ],
            'nil_context': [
                'estimated_nil_value', 'nil_vs_team_median', 'nil_vs_position_avg',
                'nil_could_increase_elsewhere',
            ],
            'portal_patterns': [
                'position_portal_rate', 'school_portal_rate', 'conference_portal_rate',
            ],
        }

    def get_portal_fit_feature_groups(self) -> Dict[str, List[str]]:
        """Get portal fit features grouped by category for SHAP analysis."""
        return {
            'need_fit': [
                'positional_need_score', 'returning_production_need',
            ],
            'player_quality': [
                'production_upgrade',
            ],
            'tier_match': [
                'school_tier_match', 'conference_level_match',
            ],
            'scheme_geography': [
                'scheme_fit_estimate', 'geographic_proximity',
            ],
            'resources': [
                'nil_budget_fit', 'academic_fit',
            ],
            'team_trajectory': [
                'team_win_trajectory',
            ],
        }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Portal Feature Engineer - Standalone Mode")
    print("=" * 50)

    # Create sample data for testing
    print("\nCreating sample test data...")

    # Sample player data (multiple seasons)
    player_data = pd.DataFrame({
        'name': ['Player A', 'Player B', 'Player C', 'Player D', 'Player E'],
        'position': ['QB', 'WR', 'RB', 'CB', 'LB'],
        'school': ['Alabama', 'Ohio State', 'Georgia', 'Texas', 'Oregon'],
        'season': [2024, 2024, 2024, 2024, 2024],
        'games': [13, 12, 11, 13, 12],
        'games_started': [13, 8, 5, 13, 10],
        'passing_yards': [3800, 0, 0, 0, 0],
        'rushing_yards': [400, 50, 900, 10, 5],
        'receptions': [0, 65, 25, 0, 0],
        'receiving_yards': [0, 950, 200, 0, 0],
        'tackles': [0, 0, 0, 55, 85],
        'production_score': [78, 62, 55, 70, 72],
        'hometown_state': ['TX', 'OH', 'GA', 'FL', 'CA'],
        'years_at_school': [3, 2, 2, 4, 3],
    })

    # Sample team data
    team_data = pd.DataFrame({
        'team': ['Alabama', 'Ohio State', 'Georgia', 'Texas', 'Oregon', 'USC'],
        'season': [2024, 2024, 2024, 2024, 2024, 2024],
        'wins': [11, 12, 13, 10, 11, 8],
        'prev_wins': [12, 11, 14, 8, 10, 7],
        'conference': ['SEC', 'Big Ten', 'SEC', 'SEC', 'Big Ten', 'Big Ten'],
        'coaching_change': [0, 0, 0, 0, 0, 1],
    })

    # Sample NIL data
    nil_data = pd.DataFrame({
        'name': ['Player A', 'Player B', 'Player C', 'Player D', 'Player E'],
        'nil_value': [2500000, 500000, 300000, 400000, 350000],
    })

    # Sample recruiting data
    recruiting_data = pd.DataFrame({
        'name': ['Player A', 'Player B', 'Player C', 'Player D', 'Player E'],
        'stars': [5, 4, 4, 4, 3],
        'rating': [0.9995, 0.9400, 0.9200, 0.9100, 0.8800],
        'national_rank': [3, 45, 78, 95, 250],
    })

    # Sample portal history
    portal_history = pd.DataFrame({
        'name': ['Portal Player 1', 'Portal Player 2', 'Player C'],
        'origin_school': ['USC', 'Michigan', 'Georgia'],
        'season': [2024, 2024, 2024],
        'position': ['QB', 'WR', 'RB'],
    })

    # Initialize engineer
    engineer = PortalFeatureEngineer(output_dir="data/processed")

    try:
        # Test flight risk features
        print("\n1. Building flight risk features...")
        flight_risk_df = engineer.build_flight_risk_features(
            player_data=player_data,
            team_data=team_data,
            nil_data=nil_data,
            recruiting_data=recruiting_data,
            portal_history=portal_history
        )

        print(f"   ✓ Flight risk features: {flight_risk_df.shape}")
        print(f"   Features: {len(engineer.get_flight_risk_feature_names())}")
        print(f"\n   Sample output:")
        print(flight_risk_df[['player_name_std', 'school_name', 'snap_pct',
                             'production_vs_star_avg', 'entered_portal']].to_string())

        # Test portal fit features
        print("\n2. Building portal fit features...")

        # Create a portal player
        portal_player = pd.Series({
            'player_name_std': 'player c',
            'position_group': 'RB',
            'production_score': 55,
            'nil_value': 300000,
            'hometown_state': 'GA',
            'school_name': 'Georgia',
            'passing_yards': 0,
            'rushing_yards': 900,
        })

        # USC roster (simplified)
        usc_roster = pd.DataFrame({
            'position_group': ['QB', 'QB', 'RB', 'RB', 'WR', 'WR', 'WR'],
            'production_score': [70, 40, 45, 35, 60, 55, 45],
            'year_in_school': ['SR', 'SO', 'JR', 'FR', 'SR', 'JR', 'SO'],
        })

        fit_df = engineer.build_portal_fit_features(
            portal_player=portal_player,
            target_school='USC',
            school_roster=usc_roster,
            team_data=team_data
        )

        print(f"   ✓ Portal fit features: {fit_df.shape}")
        print(f"\n   Fit scores for Player C → USC:")
        for col in engineer.get_portal_fit_feature_names():
            if col in fit_df.columns:
                print(f"   - {col}: {fit_df[col].iloc[0]:.1f}")

        print("\n" + "=" * 50)
        print("Feature groups for SHAP analysis:")

        print("\nFlight Risk Groups:")
        for group, features in engineer.get_flight_risk_feature_groups().items():
            print(f"  {group}: {len(features)} features")

        print("\nPortal Fit Groups:")
        for group, features in engineer.get_portal_fit_feature_groups().items():
            print(f"  {group}: {len(features)} features")

        print("\n✓ Portal feature engineering complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
