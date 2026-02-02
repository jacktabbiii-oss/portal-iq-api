"""
NFL Draft Feature Engineering Module

Transforms raw data into features for predicting where college players will be drafted.
Includes production metrics, combine measurables, school context, and draft dynamics.

Author: Elite Sports Solutions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

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

# School tiers
SCHOOL_TIER_ENCODING = {
    "blue_blood": 6, "elite": 5, "power_brand": 4,
    "p4_mid": 3, "g5_strong": 2, "g5": 1, "fcs": 0,
}

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
    "Baylor": "power_brand", "Utah": "power_brand", "Colorado": "power_brand",
    "Minnesota": "p4_mid", "Illinois": "p4_mid", "Purdue": "p4_mid", "Indiana": "p4_mid",
    "Northwestern": "p4_mid", "Maryland": "p4_mid", "Rutgers": "p4_mid",
    "Duke": "p4_mid", "North Carolina": "p4_mid", "Virginia": "p4_mid",
    "Vanderbilt": "p4_mid", "Kansas": "p4_mid", "Kansas State": "p4_mid",
    "Cincinnati": "p4_mid", "UCF": "p4_mid", "Houston": "p4_mid", "BYU": "p4_mid",
    "Boise State": "g5_strong", "Memphis": "g5_strong", "SMU": "g5_strong",
    "Tulane": "g5_strong", "Appalachian State": "g5_strong", "Liberty": "g5_strong",
}

# Conference tiers
CONFERENCE_TIER_ENCODING = {
    "SEC": 3, "Big Ten": 3, "BIG TEN": 3, "B1G": 3,
    "Big 12": 2, "BIG 12": 2, "ACC": 2, "Pac-12": 2,
    "Mountain West": 1, "MWC": 1, "American": 1, "AAC": 1,
    "Sun Belt": 1, "MAC": 1, "Conference USA": 1,
    "Independent": 2, "FCS": 0,
}

# Jimmy Johnson Draft Value Chart (pick -> value)
JIMMY_JOHNSON_CHART = {
    1: 3000, 2: 2600, 3: 2200, 4: 1800, 5: 1700,
    6: 1600, 7: 1500, 8: 1400, 9: 1350, 10: 1300,
    11: 1250, 12: 1200, 13: 1150, 14: 1100, 15: 1050,
    16: 1000, 17: 950, 18: 900, 19: 875, 20: 850,
    21: 800, 22: 780, 23: 760, 24: 740, 25: 720,
    26: 700, 27: 680, 28: 660, 29: 640, 30: 620,
    31: 600, 32: 590, 33: 580, 34: 560, 35: 550,
    36: 540, 37: 530, 38: 520, 39: 510, 40: 500,
    41: 490, 42: 480, 43: 470, 44: 460, 45: 450,
    46: 440, 47: 430, 48: 420, 49: 410, 50: 400,
    51: 390, 52: 380, 53: 370, 54: 360, 55: 350,
    56: 340, 57: 330, 58: 320, 59: 310, 60: 300,
    61: 292, 62: 284, 63: 276, 64: 270, 65: 265,
    66: 260, 67: 255, 68: 250, 69: 245, 70: 240,
    71: 235, 72: 230, 73: 225, 74: 220, 75: 215,
    76: 210, 77: 205, 78: 200, 79: 195, 80: 190,
    81: 185, 82: 180, 83: 175, 84: 170, 85: 165,
    86: 160, 87: 155, 88: 150, 89: 145, 90: 140,
    91: 136, 92: 132, 93: 128, 94: 124, 95: 120,
    96: 116, 97: 112, 98: 108, 99: 104, 100: 100,
}

# Extend chart to pick 260 with decay
for pick in range(101, 261):
    JIMMY_JOHNSON_CHART[pick] = max(1, int(100 * (0.97 ** (pick - 100))))

# Historical draft position scarcity (avg picks in rounds 1-3 per position)
POSITION_SCARCITY = {
    "QB": 4.5,    # Few QBs drafted early
    "RB": 5.0,    # RBs have declined in early-round value
    "WR": 12.0,   # Many WRs go early
    "TE": 4.0,    # Few TEs
    "OL": 15.0,   # OL is valued
    "EDGE": 14.0, # Pass rushers are premium
    "DL": 10.0,   # Interior DL
    "LB": 8.0,    # Linebackers
    "CB": 12.0,   # Corners are valued
    "S": 6.0,     # Safeties less so
    "ATH": 2.0,   # Athletes/tweeners
    "ST": 0.5,    # Specialists rarely early
}

# Average combine metrics by position (for imputation)
POSITION_COMBINE_AVERAGES = {
    "QB": {"forty": 4.85, "vertical": 31.0, "bench": 18, "broad": 112, "cone": 7.10, "shuttle": 4.30, "height": 74.5, "weight": 220},
    "RB": {"forty": 4.52, "vertical": 34.5, "bench": 20, "broad": 120, "cone": 6.95, "shuttle": 4.20, "height": 70.0, "weight": 210},
    "WR": {"forty": 4.48, "vertical": 36.0, "bench": 14, "broad": 122, "cone": 6.90, "shuttle": 4.15, "height": 73.0, "weight": 200},
    "TE": {"forty": 4.72, "vertical": 33.0, "bench": 22, "broad": 117, "cone": 7.05, "shuttle": 4.35, "height": 77.0, "weight": 250},
    "OL": {"forty": 5.25, "vertical": 26.0, "bench": 26, "broad": 102, "cone": 7.70, "shuttle": 4.75, "height": 77.0, "weight": 310},
    "EDGE": {"forty": 4.70, "vertical": 34.0, "bench": 23, "broad": 118, "cone": 7.00, "shuttle": 4.25, "height": 75.5, "weight": 255},
    "DL": {"forty": 5.05, "vertical": 29.0, "bench": 27, "broad": 108, "cone": 7.40, "shuttle": 4.55, "height": 75.0, "weight": 295},
    "LB": {"forty": 4.68, "vertical": 34.0, "bench": 23, "broad": 118, "cone": 7.00, "shuttle": 4.22, "height": 73.5, "weight": 240},
    "CB": {"forty": 4.45, "vertical": 37.0, "bench": 14, "broad": 125, "cone": 6.85, "shuttle": 4.10, "height": 71.0, "weight": 190},
    "S": {"forty": 4.52, "vertical": 36.0, "bench": 16, "broad": 122, "cone": 6.90, "shuttle": 4.15, "height": 72.0, "weight": 205},
    "ATH": {"forty": 4.55, "vertical": 35.0, "bench": 18, "broad": 120, "cone": 6.95, "shuttle": 4.20, "height": 73.0, "weight": 210},
    "ST": {"forty": 5.00, "vertical": 28.0, "bench": 15, "broad": 105, "cone": 7.20, "shuttle": 4.50, "height": 73.0, "weight": 200},
}

# Spread offense schools (inflated passing stats)
SPREAD_OFFENSE_SCHOOLS = {
    "Texas Tech", "Washington State", "Air Force", "Navy", "Oklahoma",
    "Ohio State", "Oregon", "Ole Miss", "USC", "UCF", "Houston",
    "Miami", "Louisville", "Arizona", "Baylor", "Oklahoma State",
    "TCU", "Memphis", "SMU", "Tulane", "Liberty", "Coastal Carolina",
}

# Average production by star rating (for development score calculation)
STAR_AVG_DRAFT_PRODUCTION = {
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


class DraftFeatureEngineer:
    """
    Feature engineering for NFL draft prediction models.

    Builds features from college stats, combine measurables, recruiting data,
    and draft context to predict draft position.
    """

    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize the draft feature engineer.

        Args:
            output_dir: Directory to save processed features
        """
        self.output_dir = output_dir
        self.imputation_log = []
        self.feature_names = []

    def build_features(
        self,
        player_stats: pd.DataFrame,
        recruiting_data: pd.DataFrame,
        combine_data: pd.DataFrame,
        draft_history: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build features for predicting NFL draft position.

        Args:
            player_stats: College player statistics (career and per-season)
            recruiting_data: Recruiting rankings and ratings
            combine_data: NFL combine measurements
            draft_history: Historical draft results for targets

        Returns:
            DataFrame with draft prediction features
        """
        logger.info("Building draft prediction features...")
        self.imputation_log = []

        # Start with player stats as base
        df = player_stats.copy()
        df = self._standardize_columns(df)
        logger.info(f"Starting with {len(df)} players")

        # Merge data sources
        df = self._merge_data_sources(df, recruiting_data, combine_data, draft_history)

        # Build feature groups
        df = self._build_production_features(df)
        df = self._build_measurable_features(df, combine_data)
        df = self._build_school_context_features(df)
        df = self._build_recruiting_features(df)
        df = self._build_draft_context_features(df, draft_history)
        df = self._build_target_variables(df, draft_history)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Select final features
        df = self._select_final_features(df)

        # Save
        self._save_features(df)

        logger.info(f"Draft features complete: {len(df)} players, {len(self.feature_names)} features")
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and create identifiers."""

        # Player name
        name_cols = ['name', 'player_name', 'player', 'athlete_name']
        for col in name_cols:
            if col in df.columns:
                df['player_name_std'] = (
                    df[col].astype(str).str.lower()
                    .str.replace(r'[^a-z\s]', '', regex=True)
                    .str.strip().str.replace(r'\s+', ' ', regex=True)
                )
                df['player_name_original'] = df[col]
                break
        if 'player_name_std' not in df.columns:
            df['player_name_std'] = df.index.astype(str)
            df['player_name_original'] = df.index.astype(str)

        # School
        school_cols = ['school', 'team', 'college', 'university']
        for col in school_cols:
            if col in df.columns:
                df['school_name'] = df[col].astype(str).str.strip()
                break
        if 'school_name' not in df.columns:
            df['school_name'] = 'Unknown'

        # Position
        pos_cols = ['position', 'pos', 'player_position']
        for col in pos_cols:
            if col in df.columns:
                df['position_raw'] = df[col].astype(str).str.upper().str.strip()
                df['position_group'] = df['position_raw'].map(POSITION_TO_GROUP).fillna('ATH')
                break
        if 'position_group' not in df.columns:
            df['position_raw'] = 'ATH'
            df['position_group'] = 'ATH'

        # Draft year (if applicable)
        if 'draft_year' in df.columns:
            df['draft_year'] = df['draft_year'].astype(int)
        elif 'season' in df.columns:
            df['draft_year'] = df['season'].astype(int) + 1  # Draft follows season
        else:
            df['draft_year'] = datetime.now().year

        return df

    def _merge_data_sources(
        self,
        df: pd.DataFrame,
        recruiting_data: pd.DataFrame,
        combine_data: pd.DataFrame,
        draft_history: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all data sources."""

        # Merge recruiting data
        if not recruiting_data.empty:
            recruiting_data = recruiting_data.copy()
            recruiting_data = self._standardize_name(recruiting_data)
            recruit_cols = ['player_name_std', 'stars', 'rating', 'national_rank',
                           'position_rank', 'state', 'committed_school']
            recruit_cols = [c for c in recruit_cols if c in recruiting_data.columns]
            df = df.merge(recruiting_data[recruit_cols], on='player_name_std',
                         how='left', suffixes=('', '_recruit'))
            self._log_imputation(f"Recruiting data merged: {len(recruiting_data)} records")

        # Merge combine data
        if not combine_data.empty:
            combine_data = combine_data.copy()
            combine_data = self._standardize_name(combine_data)
            combine_cols = ['player_name_std', 'forty', 'vertical', 'bench',
                           'broad_jump', 'cone', 'shuttle', 'height', 'weight',
                           'arm_length', 'hand_size']
            combine_cols = [c for c in combine_cols if c in combine_data.columns]
            # Rename broad_jump to broad if present
            if 'broad_jump' in combine_data.columns:
                combine_data['broad'] = combine_data['broad_jump']
                combine_cols = [c.replace('broad_jump', 'broad') for c in combine_cols]
            df = df.merge(combine_data[combine_cols], on='player_name_std',
                         how='left', suffixes=('', '_combine'))
            self._log_imputation(f"Combine data merged: {len(combine_data)} records")

        return df

    def _standardize_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize name column for merging."""
        name_cols = ['name', 'player_name', 'player', 'athlete_name']
        for col in name_cols:
            if col in df.columns:
                df['player_name_std'] = (
                    df[col].astype(str).str.lower()
                    .str.replace(r'[^a-z\s]', '', regex=True)
                    .str.strip().str.replace(r'\s+', ' ', regex=True)
                )
                break
        return df

    def _build_production_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build position-specific production features."""

        # Initialize production columns
        df = self._init_production_columns(df)

        # Calculate position-specific stats
        df = self._calculate_qb_production(df)
        df = self._calculate_rb_production(df)
        df = self._calculate_wr_te_production(df)
        df = self._calculate_defensive_production(df)
        df = self._calculate_ol_production(df)

        # Universal production features
        df = self._calculate_career_totals(df)
        df = self._calculate_production_trend(df)
        df = self._calculate_breakout_age(df)
        df = self._calculate_production_score(df)

        return df

    def _init_production_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initialize all production columns with NaN."""

        prod_cols = [
            # QB
            'pass_yards_pg_final', 'pass_tds_pg_final', 'comp_pct_final',
            'ypa_final', 'td_int_ratio', 'qb_rating_final',
            'pass_yards_pg_career', 'pass_tds_pg_career',
            # RB
            'rush_yards_pg_final', 'rush_tds_pg_final', 'ypc_final',
            'rush_yards_pg_career', 'total_touches_pg',
            # WR/TE
            'rec_yards_pg_final', 'rec_tds_pg_final', 'rec_pg_final',
            'ypr_final', 'rec_yards_pg_career',
            # Defense
            'tackles_pg_final', 'tfl_pg_final', 'sacks_pg_final',
            'ints_career', 'pbu_career',
            'tackles_pg_career', 'tfl_pg_career',
            # Universal
            'career_games', 'career_starts', 'games_started_pct',
            'production_trend', 'breakout_year', 'breakout_age',
            'production_score',
        ]

        for col in prod_cols:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def _calculate_qb_production(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate QB-specific production features."""

        qb_mask = df['position_group'] == 'QB'
        if qb_mask.sum() == 0:
            return df

        # Final season per-game stats
        if 'passing_yards' in df.columns and 'games' in df.columns:
            df.loc[qb_mask, 'pass_yards_pg_final'] = (
                df.loc[qb_mask, 'passing_yards'] / df.loc[qb_mask, 'games'].clip(lower=1)
            )
        if 'passing_tds' in df.columns and 'games' in df.columns:
            df.loc[qb_mask, 'pass_tds_pg_final'] = (
                df.loc[qb_mask, 'passing_tds'] / df.loc[qb_mask, 'games'].clip(lower=1)
            )
        if 'completions' in df.columns and 'pass_attempts' in df.columns:
            df.loc[qb_mask, 'comp_pct_final'] = (
                df.loc[qb_mask, 'completions'] / df.loc[qb_mask, 'pass_attempts'].clip(lower=1) * 100
            )
            df.loc[qb_mask, 'ypa_final'] = (
                df.loc[qb_mask, 'passing_yards'] / df.loc[qb_mask, 'pass_attempts'].clip(lower=1)
            )
        if 'passing_tds' in df.columns and 'interceptions_thrown' in df.columns:
            df.loc[qb_mask, 'td_int_ratio'] = (
                df.loc[qb_mask, 'passing_tds'] /
                df.loc[qb_mask, 'interceptions_thrown'].replace(0, 0.5).clip(lower=0.5)
            )

        # Passer rating (simplified NCAA formula)
        if all(c in df.columns for c in ['completions', 'pass_attempts', 'passing_yards', 'passing_tds', 'interceptions_thrown']):
            comp_pct = df.loc[qb_mask, 'completions'] / df.loc[qb_mask, 'pass_attempts'].clip(lower=1)
            ypa = df.loc[qb_mask, 'passing_yards'] / df.loc[qb_mask, 'pass_attempts'].clip(lower=1)
            td_pct = df.loc[qb_mask, 'passing_tds'] / df.loc[qb_mask, 'pass_attempts'].clip(lower=1)
            int_pct = df.loc[qb_mask, 'interceptions_thrown'] / df.loc[qb_mask, 'pass_attempts'].clip(lower=1)

            # NCAA passer rating formula
            df.loc[qb_mask, 'qb_rating_final'] = (
                (comp_pct * 100 * 8.4) + (ypa * 100) + (td_pct * 100 * 330) - (int_pct * 100 * 200)
            ) / 10

        # Career averages
        if 'career_passing_yards' in df.columns and 'career_games' in df.columns:
            df.loc[qb_mask, 'pass_yards_pg_career'] = (
                df.loc[qb_mask, 'career_passing_yards'] / df.loc[qb_mask, 'career_games'].clip(lower=1)
            )
        if 'career_passing_tds' in df.columns and 'career_games' in df.columns:
            df.loc[qb_mask, 'pass_tds_pg_career'] = (
                df.loc[qb_mask, 'career_passing_tds'] / df.loc[qb_mask, 'career_games'].clip(lower=1)
            )

        return df

    def _calculate_rb_production(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RB-specific production features."""

        rb_mask = df['position_group'] == 'RB'
        if rb_mask.sum() == 0:
            return df

        # Final season
        if 'rushing_yards' in df.columns and 'games' in df.columns:
            df.loc[rb_mask, 'rush_yards_pg_final'] = (
                df.loc[rb_mask, 'rushing_yards'] / df.loc[rb_mask, 'games'].clip(lower=1)
            )
        if 'rushing_tds' in df.columns and 'games' in df.columns:
            df.loc[rb_mask, 'rush_tds_pg_final'] = (
                df.loc[rb_mask, 'rushing_tds'] / df.loc[rb_mask, 'games'].clip(lower=1)
            )
        if 'rushing_yards' in df.columns and 'rush_attempts' in df.columns:
            df.loc[rb_mask, 'ypc_final'] = (
                df.loc[rb_mask, 'rushing_yards'] / df.loc[rb_mask, 'rush_attempts'].clip(lower=1)
            )

        # Total touches per game
        rush_att = df['rush_attempts'] if 'rush_attempts' in df.columns else 0
        receptions = df['receptions'] if 'receptions' in df.columns else 0
        games = df['games'] if 'games' in df.columns else 1
        df.loc[rb_mask, 'total_touches_pg'] = (rush_att + receptions).loc[rb_mask] / games.loc[rb_mask].clip(lower=1)

        # Career
        if 'career_rushing_yards' in df.columns and 'career_games' in df.columns:
            df.loc[rb_mask, 'rush_yards_pg_career'] = (
                df.loc[rb_mask, 'career_rushing_yards'] / df.loc[rb_mask, 'career_games'].clip(lower=1)
            )

        return df

    def _calculate_wr_te_production(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate WR/TE-specific production features."""

        wr_te_mask = df['position_group'].isin(['WR', 'TE'])
        if wr_te_mask.sum() == 0:
            return df

        # Final season
        if 'receiving_yards' in df.columns and 'games' in df.columns:
            df.loc[wr_te_mask, 'rec_yards_pg_final'] = (
                df.loc[wr_te_mask, 'receiving_yards'] / df.loc[wr_te_mask, 'games'].clip(lower=1)
            )
        if 'receiving_tds' in df.columns and 'games' in df.columns:
            df.loc[wr_te_mask, 'rec_tds_pg_final'] = (
                df.loc[wr_te_mask, 'receiving_tds'] / df.loc[wr_te_mask, 'games'].clip(lower=1)
            )
        if 'receptions' in df.columns and 'games' in df.columns:
            df.loc[wr_te_mask, 'rec_pg_final'] = (
                df.loc[wr_te_mask, 'receptions'] / df.loc[wr_te_mask, 'games'].clip(lower=1)
            )
        if 'receiving_yards' in df.columns and 'receptions' in df.columns:
            df.loc[wr_te_mask, 'ypr_final'] = (
                df.loc[wr_te_mask, 'receiving_yards'] / df.loc[wr_te_mask, 'receptions'].clip(lower=1)
            )

        # Career
        if 'career_receiving_yards' in df.columns and 'career_games' in df.columns:
            df.loc[wr_te_mask, 'rec_yards_pg_career'] = (
                df.loc[wr_te_mask, 'career_receiving_yards'] / df.loc[wr_te_mask, 'career_games'].clip(lower=1)
            )

        return df

    def _calculate_defensive_production(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate defensive position production features."""

        def_mask = df['position_group'].isin(['EDGE', 'DL', 'LB', 'CB', 'S'])
        if def_mask.sum() == 0:
            return df

        # Final season
        if 'tackles' in df.columns and 'games' in df.columns:
            df.loc[def_mask, 'tackles_pg_final'] = (
                df.loc[def_mask, 'tackles'] / df.loc[def_mask, 'games'].clip(lower=1)
            )
        if 'tackles_for_loss' in df.columns and 'games' in df.columns:
            df.loc[def_mask, 'tfl_pg_final'] = (
                df.loc[def_mask, 'tackles_for_loss'] / df.loc[def_mask, 'games'].clip(lower=1)
            )
        if 'sacks' in df.columns and 'games' in df.columns:
            # Sacks mainly for EDGE/DL/LB
            pass_rush_mask = df['position_group'].isin(['EDGE', 'DL', 'LB'])
            df.loc[pass_rush_mask, 'sacks_pg_final'] = (
                df.loc[pass_rush_mask, 'sacks'] / df.loc[pass_rush_mask, 'games'].clip(lower=1)
            )

        # Career counting stats
        if 'career_interceptions' in df.columns:
            df.loc[def_mask, 'ints_career'] = df.loc[def_mask, 'career_interceptions']
        elif 'interceptions' in df.columns:
            df.loc[def_mask, 'ints_career'] = df.loc[def_mask, 'interceptions']

        if 'career_pass_breakups' in df.columns:
            df.loc[def_mask, 'pbu_career'] = df.loc[def_mask, 'career_pass_breakups']
        elif 'pass_breakups' in df.columns:
            df.loc[def_mask, 'pbu_career'] = df.loc[def_mask, 'pass_breakups']

        # Career averages
        if 'career_tackles' in df.columns and 'career_games' in df.columns:
            df.loc[def_mask, 'tackles_pg_career'] = (
                df.loc[def_mask, 'career_tackles'] / df.loc[def_mask, 'career_games'].clip(lower=1)
            )
        if 'career_tfl' in df.columns and 'career_games' in df.columns:
            df.loc[def_mask, 'tfl_pg_career'] = (
                df.loc[def_mask, 'career_tfl'] / df.loc[def_mask, 'career_games'].clip(lower=1)
            )

        return df

    def _calculate_ol_production(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate OL-specific features (limited stats available)."""

        ol_mask = df['position_group'] == 'OL'
        if ol_mask.sum() == 0:
            return df

        # OL production is mainly games started
        if 'games_started' in df.columns:
            df.loc[ol_mask, 'ol_starts'] = df.loc[ol_mask, 'games_started']

        # Sacks allowed (if available)
        if 'sacks_allowed' in df.columns:
            df.loc[ol_mask, 'sacks_allowed_pg'] = (
                df.loc[ol_mask, 'sacks_allowed'] / df.loc[ol_mask, 'games'].clip(lower=1)
            )

        return df

    def _calculate_career_totals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate career total features."""

        # Career games
        if 'career_games' in df.columns:
            df['career_games'] = df['career_games'].fillna(df['games'] if 'games' in df.columns else 13)
        elif 'games' in df.columns:
            # Estimate career games from years
            years = df['years_in_college'] if 'years_in_college' in df.columns else 3
            df['career_games'] = df['games'] * (years / 1.5)  # Rough estimate
        else:
            df['career_games'] = 40
            self._log_imputation("career_games estimated at 40")

        # Career starts
        if 'career_starts' in df.columns:
            pass  # Already have it
        elif 'career_games_started' in df.columns:
            df['career_starts'] = df['career_games_started']
        elif 'games_started' in df.columns:
            years = df['years_in_college'] if 'years_in_college' in df.columns else 3
            df['career_starts'] = df['games_started'] * (years / 2)
        else:
            df['career_starts'] = 20
            self._log_imputation("career_starts estimated at 20")

        # Games started percentage
        df['games_started_pct'] = df['career_starts'] / df['career_games'].clip(lower=1)
        df['games_started_pct'] = df['games_started_pct'].clip(0, 1)

        return df

    def _calculate_production_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate production trend (improving vs declining)."""

        # If we have multi-season data
        if 'final_season_production' in df.columns and 'prev_season_production' in df.columns:
            df['production_trend'] = df['final_season_production'] - df['prev_season_production']
        elif 'production_score' in df.columns:
            # Use production score as proxy, assume positive trend for high producers
            df['production_trend'] = (df['production_score'] - 50) * 0.3
        else:
            df['production_trend'] = 0
            self._log_imputation("production_trend defaulted to 0")

        return df

    def _calculate_breakout_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout age - when player first became a starter."""

        # Breakout year (what year did they first start)
        if 'first_start_year' in df.columns:
            df['breakout_year'] = df['first_start_year']
        elif 'years_as_starter' in df.columns:
            years_in_college = df['years_in_college'] if 'years_in_college' in df.columns else 3
            df['breakout_year'] = years_in_college - df['years_as_starter'] + 1
        else:
            # Estimate: starters by junior year on average
            df['breakout_year'] = 2
            self._log_imputation("breakout_year estimated at year 2")

        # Breakout age (assuming 18 as freshman)
        if 'age_at_breakout' in df.columns:
            df['breakout_age'] = df['age_at_breakout']
        else:
            df['breakout_age'] = 18 + df['breakout_year']

        return df

    def _calculate_production_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized production score (0-100) within position group."""

        if 'production_score' in df.columns and df['production_score'].notna().any():
            return df  # Already calculated

        df['production_score'] = 50  # Default

        position_key_stats = {
            'QB': ['pass_yards_pg_final', 'pass_tds_pg_final', 'qb_rating_final'],
            'RB': ['rush_yards_pg_final', 'ypc_final', 'total_touches_pg'],
            'WR': ['rec_yards_pg_final', 'rec_pg_final', 'rec_tds_pg_final'],
            'TE': ['rec_yards_pg_final', 'rec_pg_final'],
            'EDGE': ['sacks_pg_final', 'tfl_pg_final', 'tackles_pg_final'],
            'DL': ['tfl_pg_final', 'tackles_pg_final'],
            'LB': ['tackles_pg_final', 'tfl_pg_final'],
            'CB': ['ints_career', 'pbu_career', 'tackles_pg_final'],
            'S': ['tackles_pg_final', 'ints_career'],
            'OL': ['career_starts', 'games_started_pct'],
        }

        for pos, stats in position_key_stats.items():
            mask = df['position_group'] == pos
            if mask.sum() == 0:
                continue

            available = [s for s in stats if s in df.columns]
            if not available:
                continue

            percentiles = []
            for stat in available:
                vals = df.loc[mask, stat]
                if vals.notna().sum() > 0:
                    pct = vals.rank(pct=True, na_option='keep') * 100
                    percentiles.append(pct)

            if percentiles:
                avg_pct = pd.concat(percentiles, axis=1).mean(axis=1)
                df.loc[mask, 'production_score'] = avg_pct.fillna(50)

        return df

    def _build_measurable_features(self, df: pd.DataFrame, combine_data: pd.DataFrame) -> pd.DataFrame:
        """Build physical measurable features."""

        # Height (convert to inches if needed)
        if 'height' in df.columns:
            df['height_inches'] = df['height'].apply(self._convert_height_to_inches)
        else:
            df['height_inches'] = df['position_group'].map(
                lambda p: POSITION_COMBINE_AVERAGES.get(p, {}).get('height', 73)
            )
            df['height_estimated'] = 1
            self._log_imputation("height filled with position averages")

        # Weight
        if 'weight' in df.columns:
            df['weight_lbs'] = df['weight']
        else:
            df['weight_lbs'] = df['position_group'].map(
                lambda p: POSITION_COMBINE_AVERAGES.get(p, {}).get('weight', 220)
            )
            df['weight_estimated'] = 1
            self._log_imputation("weight filled with position averages")

        # BMI
        df['bmi'] = (df['weight_lbs'] * 703) / (df['height_inches'] ** 2)

        # Combine metrics
        combine_metrics = ['forty', 'vertical', 'bench', 'broad', 'cone', 'shuttle']
        for metric in combine_metrics:
            if metric in df.columns:
                df[f'{metric}_actual'] = df[metric]
                df[f'{metric}_estimated'] = 0
            else:
                # Fill with position average
                df[f'{metric}_actual'] = df['position_group'].map(
                    lambda p: POSITION_COMBINE_AVERAGES.get(p, {}).get(metric, np.nan)
                )
                df[f'{metric}_estimated'] = 1

        # Flag if all combine metrics are estimated
        estimated_cols = [f'{m}_estimated' for m in combine_metrics]
        df['combine_data_available'] = (
            df[[c for c in estimated_cols if c in df.columns]].sum(axis=1) < len(combine_metrics)
        ).astype(int)

        # Position-specific athletic scores
        df = self._calculate_athletic_scores(df)

        return df

    def _convert_height_to_inches(self, height) -> float:
        """Convert height to inches (handles 6-2, 6'2", 74 formats)."""
        if pd.isna(height):
            return np.nan

        height_str = str(height)

        # Already in inches
        if height_str.isdigit() and int(height_str) > 60:
            return float(height_str)

        # Format: 6-2 or 6'2" or 6'2
        import re
        match = re.match(r"(\d+)['\-](\d+)", height_str)
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            return feet * 12 + inches

        return np.nan

    def _calculate_athletic_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position-specific athletic scores."""

        forty = df['forty_actual'] if 'forty_actual' in df.columns else 4.6
        weight = df['weight_lbs'] if 'weight_lbs' in df.columns else 220

        # Speed Score (RB) = (weight * 200) / (40_time ^ 4)
        rb_mask = df['position_group'] == 'RB'
        df.loc[rb_mask, 'speed_score'] = (weight * 200) / (forty ** 4)

        # Height-Adjusted Speed Score (WR) = (speed_score * height / 73)
        wr_mask = df['position_group'] == 'WR'
        height = df['height_inches'] if 'height_inches' in df.columns else 73
        df.loc[wr_mask, 'height_adj_speed_score'] = (weight * 200) / (forty ** 4) * (height / 73)

        # Burst Score (explosiveness) = vertical + (broad / 12)
        vertical = df['vertical_actual'] if 'vertical_actual' in df.columns else 33
        broad = df['broad_actual'] if 'broad_actual' in df.columns else 115
        df['burst_score'] = vertical + (broad / 12)

        # Agility Score (inverse of cone + shuttle)
        cone = df['cone_actual'] if 'cone_actual' in df.columns else 7.1
        shuttle = df['shuttle_actual'] if 'shuttle_actual' in df.columns else 4.3
        df['agility_score'] = 100 / (cone + shuttle)

        # Relative Athletic Score (RAS) - simplified
        # Compare each metric to position average, normalize
        df['ras_score'] = 50  # Default
        for pos in df['position_group'].unique():
            mask = df['position_group'] == pos
            if mask.sum() == 0:
                continue

            pos_avgs = POSITION_COMBINE_AVERAGES.get(pos, {})
            if not pos_avgs:
                continue

            scores = []
            if 'forty_actual' in df.columns:
                # Lower 40 is better
                forty_score = (pos_avgs.get('forty', 4.6) - df.loc[mask, 'forty_actual']) * 20 + 50
                scores.append(forty_score)
            if 'vertical_actual' in df.columns:
                vert_score = (df.loc[mask, 'vertical_actual'] - pos_avgs.get('vertical', 33)) * 2 + 50
                scores.append(vert_score)
            if 'broad_actual' in df.columns:
                broad_score = (df.loc[mask, 'broad_actual'] - pos_avgs.get('broad', 115)) * 0.5 + 50
                scores.append(broad_score)

            if scores:
                df.loc[mask, 'ras_score'] = pd.concat(scores, axis=1).mean(axis=1).clip(0, 100)

        return df

    def _build_school_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build school and context features."""

        # School tier
        df['school_tier_name'] = df['school_name'].map(SCHOOL_TIERS).fillna('g5')
        df['school_tier'] = df['school_tier_name'].map(SCHOOL_TIER_ENCODING).fillna(1)

        # Conference tier
        if 'conference' in df.columns:
            df['conference_tier'] = df['conference'].map(CONFERENCE_TIER_ENCODING).fillna(1)
        else:
            # Infer from school tier
            df['conference_tier'] = df['school_tier'].apply(
                lambda x: 3 if x >= 5 else (2 if x >= 3 else 1)
            )
            self._log_imputation("conference_tier inferred from school tier")

        # Competition level (strength of schedule proxy)
        # Higher school tier = played tougher competition
        df['competition_level'] = (df['school_tier'] + df['conference_tier']) / 2

        # Production environment (spread vs pro-style)
        df['spread_offense'] = df['school_name'].isin(SPREAD_OFFENSE_SCHOOLS).astype(int)

        # Team talent composite (proxy from school tier)
        # In reality would use recruiting class rankings
        df['team_talent_composite'] = df['school_tier'] * 15 + 10  # Scale 10-100

        return df

    def _build_recruiting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build recruiting profile features."""

        # Stars
        if 'stars' in df.columns:
            df['recruiting_stars'] = df['stars'].fillna(0).astype(int).clip(0, 5)
        else:
            df['recruiting_stars'] = 0
            self._log_imputation("recruiting_stars defaulted to 0")

        # Composite rating
        if 'rating' in df.columns:
            df['recruiting_composite'] = df['rating'].clip(0, 1)
        else:
            # Estimate from stars
            star_to_comp = {0: 0.0, 2: 0.75, 3: 0.82, 4: 0.92, 5: 0.98}
            df['recruiting_composite'] = df['recruiting_stars'].map(star_to_comp).fillna(0)
            self._log_imputation("recruiting_composite estimated from stars")

        # Position rank
        if 'position_rank' in df.columns:
            df['recruiting_position_rank'] = df['position_rank'].fillna(999)
        else:
            df['recruiting_position_rank'] = 999

        # Overall rank
        if 'national_rank' in df.columns:
            df['recruiting_national_rank'] = df['national_rank'].fillna(9999)
        elif 'overall_rank' in df.columns:
            df['recruiting_national_rank'] = df['overall_rank'].fillna(9999)
        else:
            df['recruiting_national_rank'] = 9999

        # Development score: production relative to recruiting
        def calc_development(row):
            pos = row.get('position_group', 'ATH')
            stars = int(row.get('recruiting_stars', 0))
            production = row.get('production_score', 50)

            expected = STAR_AVG_DRAFT_PRODUCTION.get(pos, {}).get(stars, 40)
            return production - expected

        df['development_score'] = df.apply(calc_development, axis=1)

        return df

    def _build_draft_context_features(self, df: pd.DataFrame, draft_history: pd.DataFrame) -> pd.DataFrame:
        """Build draft context features."""

        # Position scarcity (how often position is drafted early)
        df['position_scarcity'] = df['position_group'].map(POSITION_SCARCITY).fillna(5)

        # Draft class depth (would need current year's class data)
        # For now, use average
        df['draft_class_depth'] = 50  # Neutral
        self._log_imputation("draft_class_depth set to neutral (50)")

        # Age on draft day
        if 'birth_date' in df.columns:
            df['draft_age'] = df.apply(
                lambda row: self._calculate_draft_age(row['birth_date'], row['draft_year']),
                axis=1
            )
        elif 'age' in df.columns:
            df['draft_age'] = df['age']
        else:
            # Estimate from years in college
            years = df['years_in_college'] if 'years_in_college' in df.columns else 3
            df['draft_age'] = 18 + years + 0.5  # Draft is in April
            self._log_imputation("draft_age estimated from years in college")

        # Years in college
        if 'years_in_college' in df.columns:
            df['years_in_college'] = df['years_in_college'].clip(1, 6)
        elif 'year' in df.columns:
            year_map = {'FR': 1, 'SO': 2, 'JR': 3, 'SR': 4, 'GR': 5, 'RS': 5}
            df['years_in_college'] = df['year'].astype(str).str.upper().map(year_map).fillna(3)
        else:
            df['years_in_college'] = 3
            self._log_imputation("years_in_college defaulted to 3")

        # Early declare flag (3 years or less = early)
        df['early_declare'] = (df['years_in_college'] <= 3).astype(int)

        # Position group encoded
        position_encoding = {
            'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'OL': 5,
            'EDGE': 6, 'DL': 7, 'LB': 8, 'CB': 9, 'S': 10,
            'ATH': 11, 'ST': 12
        }
        df['position_group_encoded'] = df['position_group'].map(position_encoding).fillna(11)

        return df

    def _calculate_draft_age(self, birth_date, draft_year) -> float:
        """Calculate age on draft day (April 25)."""
        try:
            if pd.isna(birth_date) or pd.isna(draft_year):
                return np.nan
            birth = pd.to_datetime(birth_date)
            draft_day = pd.Timestamp(year=int(draft_year), month=4, day=25)
            age = (draft_day - birth).days / 365.25
            return round(age, 2)
        except:
            return np.nan

    def _build_target_variables(self, df: pd.DataFrame, draft_history: pd.DataFrame) -> pd.DataFrame:
        """Build target variables from draft history."""

        # Initialize targets
        df['was_drafted'] = 0
        df['draft_round'] = 0
        df['draft_pick'] = 0
        df['draft_value'] = 0

        if draft_history.empty:
            self._log_imputation("No draft history - targets are all 0")
            return df

        # Standardize draft history names
        draft_history = draft_history.copy()
        draft_history = self._standardize_name(draft_history)

        if 'player_name_std' not in draft_history.columns:
            return df

        # Create lookup from draft history
        draft_lookup = {}
        for _, row in draft_history.iterrows():
            name = row['player_name_std']
            draft_lookup[name] = {
                'was_drafted': 1,
                'draft_round': row.get('round', row.get('draft_round', 0)),
                'draft_pick': row.get('pick', row.get('overall_pick', row.get('draft_pick', 0))),
            }

        # Apply to df
        for idx, row in df.iterrows():
            name = row['player_name_std']
            if name in draft_lookup:
                df.loc[idx, 'was_drafted'] = 1
                df.loc[idx, 'draft_round'] = draft_lookup[name]['draft_round']
                df.loc[idx, 'draft_pick'] = draft_lookup[name]['draft_pick']

        # Calculate draft value using Jimmy Johnson chart
        df['draft_value'] = df['draft_pick'].apply(
            lambda p: JIMMY_JOHNSON_CHART.get(int(p), 0) if p > 0 else 0
        )

        # Log results
        drafted_count = (df['was_drafted'] == 1).sum()
        self._log_imputation(f"Draft targets set: {drafted_count} drafted players matched")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with position-aware imputation."""

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = ['was_drafted', 'draft_round', 'draft_pick', 'draft_value']
        numeric_cols = [c for c in numeric_cols if c not in target_cols]

        for col in numeric_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                # Use position-group median
                group_median = df.groupby('position_group')[col].transform('median')
                df[col] = df[col].fillna(group_median)

                remaining = df[col].isna().sum()
                if remaining > 0:
                    overall_median = df[col].median()
                    if pd.isna(overall_median):
                        overall_median = 0
                    df[col] = df[col].fillna(overall_median)

        return df

    def _select_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order final features."""

        self.feature_names = [
            # Production - Final Season
            'pass_yards_pg_final', 'pass_tds_pg_final', 'comp_pct_final',
            'ypa_final', 'td_int_ratio', 'qb_rating_final',
            'rush_yards_pg_final', 'rush_tds_pg_final', 'ypc_final', 'total_touches_pg',
            'rec_yards_pg_final', 'rec_tds_pg_final', 'rec_pg_final', 'ypr_final',
            'tackles_pg_final', 'tfl_pg_final', 'sacks_pg_final',

            # Production - Career
            'pass_yards_pg_career', 'pass_tds_pg_career',
            'rush_yards_pg_career', 'rec_yards_pg_career',
            'tackles_pg_career', 'tfl_pg_career',
            'ints_career', 'pbu_career',
            'career_games', 'career_starts', 'games_started_pct',
            'production_trend', 'breakout_year', 'breakout_age',
            'production_score',

            # Measurables
            'height_inches', 'weight_lbs', 'bmi',
            'forty_actual', 'vertical_actual', 'bench_actual',
            'broad_actual', 'cone_actual', 'shuttle_actual',
            'combine_data_available',
            'speed_score', 'height_adj_speed_score', 'burst_score',
            'agility_score', 'ras_score',

            # School context
            'school_tier', 'conference_tier', 'competition_level',
            'spread_offense', 'team_talent_composite',

            # Recruiting
            'recruiting_stars', 'recruiting_composite',
            'recruiting_position_rank', 'recruiting_national_rank',
            'development_score',

            # Draft context
            'position_scarcity', 'draft_class_depth',
            'draft_age', 'years_in_college', 'early_declare',
            'position_group_encoded',
        ]

        # Filter to existing columns
        self.feature_names = [c for c in self.feature_names if c in df.columns]

        # Metadata
        metadata_cols = ['player_name_std', 'player_name_original', 'school_name',
                        'position_group', 'draft_year']
        metadata_cols = [c for c in metadata_cols if c in df.columns]

        # Targets
        target_cols = ['was_drafted', 'draft_round', 'draft_pick', 'draft_value']
        target_cols = [c for c in target_cols if c in df.columns]

        final_cols = metadata_cols + self.feature_names + target_cols
        return df[final_cols]

    def _save_features(self, df: pd.DataFrame) -> None:
        """Save features to CSV."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'draft_features_ready.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Draft features saved to {output_path}")

        # Save imputation log
        log_path = os.path.join(self.output_dir, 'draft_feature_imputation_log.txt')
        with open(log_path, 'w') as f:
            f.write("Draft Feature Engineering Imputation Log\n")
            f.write("=" * 50 + "\n\n")
            for entry in self.imputation_log:
                f.write(f"- {entry}\n")

    def _log_imputation(self, message: str) -> None:
        """Log an imputation decision."""
        self.imputation_log.append(message)
        logger.debug(f"Imputation: {message}")

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_names.copy()

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by category for SHAP analysis."""
        return {
            'production_final_season': [
                'pass_yards_pg_final', 'pass_tds_pg_final', 'comp_pct_final',
                'ypa_final', 'td_int_ratio', 'qb_rating_final',
                'rush_yards_pg_final', 'rush_tds_pg_final', 'ypc_final', 'total_touches_pg',
                'rec_yards_pg_final', 'rec_tds_pg_final', 'rec_pg_final', 'ypr_final',
                'tackles_pg_final', 'tfl_pg_final', 'sacks_pg_final',
            ],
            'production_career': [
                'pass_yards_pg_career', 'pass_tds_pg_career',
                'rush_yards_pg_career', 'rec_yards_pg_career',
                'tackles_pg_career', 'tfl_pg_career',
                'ints_career', 'pbu_career',
                'career_games', 'career_starts', 'games_started_pct',
                'production_trend', 'breakout_year', 'breakout_age',
                'production_score',
            ],
            'measurables': [
                'height_inches', 'weight_lbs', 'bmi',
                'forty_actual', 'vertical_actual', 'bench_actual',
                'broad_actual', 'cone_actual', 'shuttle_actual',
                'combine_data_available',
            ],
            'athletic_scores': [
                'speed_score', 'height_adj_speed_score', 'burst_score',
                'agility_score', 'ras_score',
            ],
            'school_context': [
                'school_tier', 'conference_tier', 'competition_level',
                'spread_offense', 'team_talent_composite',
            ],
            'recruiting': [
                'recruiting_stars', 'recruiting_composite',
                'recruiting_position_rank', 'recruiting_national_rank',
                'development_score',
            ],
            'draft_context': [
                'position_scarcity', 'draft_class_depth',
                'draft_age', 'years_in_college', 'early_declare',
                'position_group_encoded',
            ],
        }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Draft Feature Engineer - Standalone Mode")
    print("=" * 50)

    # Create sample data
    print("\nCreating sample test data...")

    # Sample player stats
    player_stats = pd.DataFrame({
        'name': ['Caleb Williams', 'Marvin Harrison Jr', 'Brock Bowers',
                 'Rome Odunze', 'Jared Verse', 'Quinyon Mitchell'],
        'position': ['QB', 'WR', 'TE', 'WR', 'EDGE', 'CB'],
        'school': ['USC', 'Ohio State', 'Georgia', 'Washington', 'Florida State', 'Toledo'],
        'games': [12, 12, 13, 14, 13, 13],
        'games_started': [12, 12, 13, 14, 13, 13],
        'passing_yards': [3633, 0, 0, 0, 0, 0],
        'passing_tds': [30, 0, 0, 0, 0, 0],
        'completions': [280, 0, 0, 0, 0, 0],
        'pass_attempts': [399, 0, 0, 0, 0, 0],
        'interceptions_thrown': [5, 0, 0, 0, 0, 0],
        'rushing_yards': [11, 0, 0, 0, 0, 0],
        'receptions': [0, 67, 56, 92, 0, 0],
        'receiving_yards': [0, 1211, 792, 1640, 0, 0],
        'receiving_tds': [0, 14, 6, 13, 0, 0],
        'tackles': [0, 0, 0, 0, 45, 38],
        'tackles_for_loss': [0, 0, 0, 0, 17, 3],
        'sacks': [0, 0, 0, 0, 11.0, 0],
        'interceptions': [0, 0, 0, 0, 0, 4],
        'years_in_college': [3, 3, 3, 4, 4, 5],
        'draft_year': [2024, 2024, 2024, 2024, 2024, 2024],
    })

    # Sample recruiting data
    recruiting_data = pd.DataFrame({
        'name': ['Caleb Williams', 'Marvin Harrison Jr', 'Brock Bowers',
                 'Rome Odunze', 'Jared Verse', 'Quinyon Mitchell'],
        'stars': [5, 4, 4, 4, 3, 3],
        'rating': [0.9998, 0.9650, 0.9450, 0.9200, 0.8700, 0.8500],
        'national_rank': [2, 97, 38, 60, 450, 650],
        'position_rank': [1, 12, 4, 8, 35, 45],
    })

    # Sample combine data
    combine_data = pd.DataFrame({
        'name': ['Caleb Williams', 'Marvin Harrison Jr', 'Brock Bowers',
                 'Rome Odunze', 'Jared Verse', 'Quinyon Mitchell'],
        'forty': [4.59, 4.38, 4.49, 4.45, 4.68, 4.33],
        'vertical': [33.5, 36.5, 36.0, 38.5, 33.0, 41.5],
        'bench': [17, 17, 23, 15, 25, 14],
        'broad_jump': [117, 123, 127, 125, 121, 134],
        'cone': [7.12, 6.78, 7.05, 6.88, 7.21, 6.68],
        'shuttle': [4.38, 4.01, 4.23, 4.18, 4.37, 4.02],
        'height': ['6-1', '6-4', '6-4', '6-3', '6-4', '6-0'],
        'weight': [214, 209, 243, 215, 247, 193],
    })

    # Sample draft history (for targets)
    draft_history = pd.DataFrame({
        'name': ['Caleb Williams', 'Marvin Harrison Jr', 'Brock Bowers',
                 'Rome Odunze', 'Jared Verse', 'Quinyon Mitchell'],
        'round': [1, 1, 1, 1, 1, 1],
        'pick': [1, 4, 13, 9, 6, 22],
        'team': ['Bears', 'Cardinals', 'Raiders', 'Bears', 'Rams', 'Eagles'],
    })

    # Initialize and run
    engineer = DraftFeatureEngineer(output_dir="data/processed")

    try:
        features_df = engineer.build_features(
            player_stats=player_stats,
            recruiting_data=recruiting_data,
            combine_data=combine_data,
            draft_history=draft_history
        )

        print(f"\n✓ Draft features complete!")
        print(f"  - Output shape: {features_df.shape}")
        print(f"  - Features: {len(engineer.get_feature_names())}")

        print("\nFeature groups for SHAP analysis:")
        for group, features in engineer.get_feature_importance_groups().items():
            print(f"  - {group}: {len(features)} features")

        print("\nSample output (targets):")
        print(features_df[['player_name_std', 'position_group', 'production_score',
                          'ras_score', 'was_drafted', 'draft_round', 'draft_pick',
                          'draft_value']].to_string())

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
