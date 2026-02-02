"""
NIL Feature Engineering Module

Transforms raw collected data into model-ready features for NIL valuation.
Handles position-aware performance metrics, school branding, social media,
recruiting profiles, and draft projections.

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
    # Offense
    "QB": "QB",
    "RB": "RB", "FB": "RB", "HB": "RB",
    "WR": "WR", "SLOT": "WR",
    "TE": "TE",
    "OT": "OL", "OG": "OL", "C": "OL", "OL": "OL", "T": "OL", "G": "OL",
    # Defense
    "DE": "EDGE", "OLB": "EDGE", "EDGE": "EDGE", "RUSH": "EDGE",
    "DT": "DL", "NT": "DL", "DL": "DL", "DI": "DL",
    "ILB": "LB", "MLB": "LB", "LB": "LB", "MIKE": "LB", "WILL": "LB",
    "CB": "CB", "DB": "CB", "NB": "CB",
    "S": "S", "FS": "S", "SS": "S", "SAF": "S",
    # Special teams
    "K": "ST", "P": "ST", "LS": "ST", "PK": "ST",
    # Athletes
    "ATH": "ATH", "ATHLETE": "ATH",
}

# School tier encoding (higher = more prestigious)
SCHOOL_TIER_ENCODING = {
    "blue_blood": 6,
    "elite": 5,
    "power_brand": 4,
    "p4_mid": 3,
    "g5_strong": 2,
    "g5": 1,
    "fcs": 0,
    "unknown": 1,
}

# Conference tier encoding
CONFERENCE_TIER_ENCODING = {
    "SEC": 3, "Big Ten": 3, "BIG TEN": 3, "B1G": 3,
    "Big 12": 2, "BIG 12": 2, "ACC": 2,
    "Pac-12": 2, "PAC-12": 2, "PAC 12": 2,  # Legacy
    "Mountain West": 1, "MWC": 1,
    "American": 1, "AAC": 1,
    "Sun Belt": 1, "SBC": 1,
    "MAC": 1, "Conference USA": 1, "CUSA": 1,
    "Independent": 2,  # Notre Dame, etc.
    "FCS": 0,
    "unknown": 1,
}

# Top football recruiting states
FOOTBALL_STATES = {
    "Texas", "TX", "Florida", "FL", "Georgia", "GA", "Alabama", "AL",
    "Ohio", "OH", "California", "CA", "Louisiana", "LA", "Tennessee", "TN",
    "Oklahoma", "OK", "Michigan", "MI", "Pennsylvania", "PA"
}

# School to tier mapping (130 FBS schools)
SCHOOL_TIERS = {
    # Blue Bloods (6)
    "Alabama": "blue_blood", "Ohio State": "blue_blood", "Notre Dame": "blue_blood",
    "USC": "blue_blood", "Oklahoma": "blue_blood", "Texas": "blue_blood",
    "Michigan": "blue_blood", "Nebraska": "blue_blood",

    # Elite (5)
    "Georgia": "elite", "Clemson": "elite", "LSU": "elite", "Florida": "elite",
    "Penn State": "elite", "Florida State": "elite", "Miami": "elite",
    "Tennessee": "elite", "Auburn": "elite", "Oregon": "elite", "Texas A&M": "elite",

    # Power Brand (4)
    "Wisconsin": "power_brand", "Iowa": "power_brand", "Michigan State": "power_brand",
    "UCLA": "power_brand", "Washington": "power_brand", "Arkansas": "power_brand",
    "Ole Miss": "power_brand", "Mississippi State": "power_brand", "South Carolina": "power_brand",
    "Kentucky": "power_brand", "Missouri": "power_brand", "NC State": "power_brand",
    "Virginia Tech": "power_brand", "Louisville": "power_brand", "Pittsburgh": "power_brand",
    "West Virginia": "power_brand", "Oklahoma State": "power_brand", "TCU": "power_brand",
    "Baylor": "power_brand", "Utah": "power_brand", "Arizona State": "power_brand",
    "Colorado": "power_brand", "Stanford": "power_brand", "Cal": "power_brand",
    "Arizona": "power_brand", "Oregon State": "power_brand", "Washington State": "power_brand",

    # P4 Mid-tier (3)
    "Minnesota": "p4_mid", "Illinois": "p4_mid", "Purdue": "p4_mid", "Indiana": "p4_mid",
    "Northwestern": "p4_mid", "Maryland": "p4_mid", "Rutgers": "p4_mid",
    "Duke": "p4_mid", "North Carolina": "p4_mid", "Virginia": "p4_mid", "Wake Forest": "p4_mid",
    "Syracuse": "p4_mid", "Boston College": "p4_mid", "Georgia Tech": "p4_mid",
    "Vanderbilt": "p4_mid", "Kansas": "p4_mid", "Kansas State": "p4_mid",
    "Iowa State": "p4_mid", "Texas Tech": "p4_mid", "Houston": "p4_mid",
    "Cincinnati": "p4_mid", "UCF": "p4_mid", "BYU": "p4_mid",

    # G5 Strong (2)
    "Boise State": "g5_strong", "Memphis": "g5_strong", "SMU": "g5_strong",
    "Tulane": "g5_strong", "App State": "g5_strong", "Appalachian State": "g5_strong",
    "Coastal Carolina": "g5_strong", "Liberty": "g5_strong", "James Madison": "g5_strong",
    "San Diego State": "g5_strong", "Fresno State": "g5_strong", "Air Force": "g5_strong",
    "Army": "g5_strong", "Navy": "g5_strong", "Toledo": "g5_strong",
    "Western Kentucky": "g5_strong", "Marshall": "g5_strong", "Louisiana": "g5_strong",
    "Troy": "g5_strong", "Georgia Southern": "g5_strong", "Jacksonville State": "g5_strong",
    "Sam Houston": "g5_strong", "UTSA": "g5_strong",

    # G5 (1)
    "North Texas": "g5", "Rice": "g5", "Tulsa": "g5", "Temple": "g5",
    "East Carolina": "g5", "Charlotte": "g5", "FAU": "g5", "FIU": "g5",
    "UAB": "g5", "South Alabama": "g5", "Southern Miss": "g5",
    "Arkansas State": "g5", "Texas State": "g5", "Louisiana Tech": "g5",
    "Middle Tennessee": "g5", "UTEP": "g5", "New Mexico": "g5",
    "New Mexico State": "g5", "Nevada": "g5", "UNLV": "g5",
    "Hawaii": "g5", "San Jose State": "g5", "Utah State": "g5",
    "Wyoming": "g5", "Colorado State": "g5",
    "Northern Illinois": "g5", "Central Michigan": "g5", "Western Michigan": "g5",
    "Eastern Michigan": "g5", "Ball State": "g5", "Bowling Green": "g5",
    "Akron": "g5", "Kent State": "g5", "Ohio": "g5", "Buffalo": "g5",
    "UMass": "g5", "Connecticut": "g5", "UConn": "g5",
    "Georgia State": "g5", "Old Dominion": "g5", "Kennesaw State": "g5",
}

# Metro population by school (approximate, in thousands)
SCHOOL_METRO_POPULATION = {
    # Major metros (>3M)
    "USC": 13000, "UCLA": 13000,  # Los Angeles
    "Texas": 2300, "Texas Tech": 320,  # Austin, Lubbock
    "Texas A&M": 275,  # College Station
    "Houston": 7100,  # Houston
    "Miami": 6200, "FIU": 6200, "FAU": 6200,  # South Florida
    "Ohio State": 2150,  # Columbus
    "Michigan": 4400,  # Detroit metro (Ann Arbor nearby)
    "Penn State": 160,  # State College (small but draws from Philly/Pittsburgh)
    "Georgia": 450, "Georgia Tech": 6100,  # Athens / Atlanta
    "Notre Dame": 320,  # South Bend
    "Stanford": 4700, "Cal": 4700, "San Jose State": 4700,  # Bay Area
    "Washington": 4000,  # Seattle
    "Arizona State": 4900, "Arizona": 1050,  # Phoenix, Tucson
    "Colorado": 2900,  # Denver
    "Oregon": 380,  # Eugene
    "Florida": 330, "Florida State": 390,  # Gainesville, Tallahassee
    "UCF": 2700,  # Orlando
    "South Florida": 3200,  # Tampa
    "Georgia State": 6100,  # Atlanta
    "Temple": 6200, "Rutgers": 20000,  # Philadelphia / NYC metro
    "Maryland": 6300,  # DC metro
    "Pittsburgh": 2400,
    "Boston College": 4900,  # Boston
    "Syracuse": 660,
    "Louisville": 1400,
    "Cincinnati": 2250,
    "Kentucky": 520,  # Lexington
    "Tennessee": 900,  # Knoxville
    "Vanderbilt": 2000,  # Nashville
    "Alabama": 260,  # Tuscaloosa
    "Auburn": 175,  # Auburn
    "LSU": 870,  # Baton Rouge
    "Ole Miss": 175,  # Oxford
    "Mississippi State": 60,  # Starkville
    "Arkansas": 560,  # Fayetteville
    "Missouri": 180,  # Columbia
    "Oklahoma": 1450, "Oklahoma State": 160,  # Norman (OKC), Stillwater
    "TCU": 7600, "SMU": 7600, "North Texas": 7600,  # Dallas-Fort Worth
    "Baylor": 280,  # Waco
    "Texas State": 2300,  # San Marcos (Austin metro)
    "UTSA": 2600,  # San Antonio
    "Rice": 7100,  # Houston
    "Kansas": 2500, "Kansas State": 190,  # KC metro, Manhattan
    "Iowa": 175, "Iowa State": 700,  # Iowa City, Ames (Des Moines nearby)
    "Nebraska": 1000,  # Lincoln/Omaha
    "Minnesota": 3700,  # Minneapolis
    "Wisconsin": 680,  # Madison
    "Illinois": 240,  # Champaign
    "Northwestern": 9600,  # Chicago
    "Purdue": 230,  # West Lafayette
    "Indiana": 175,  # Bloomington
    "Michigan State": 540,  # Lansing
    "NC State": 1400, "Duke": 1400, "North Carolina": 1400,  # Raleigh-Durham
    "Wake Forest": 680,  # Winston-Salem
    "Clemson": 920,  # Greenville-Spartanburg
    "South Carolina": 880,  # Columbia
    "Virginia": 240, "Virginia Tech": 180,  # Charlottesville, Blacksburg
    "West Virginia": 140,  # Morgantown
    "BYU": 650,  # Provo/SLC
    "Utah": 1250,  # Salt Lake City
    "Boise State": 780,  # Boise
    "San Diego State": 3300,  # San Diego
    "Fresno State": 1100,  # Fresno
    "UNLV": 2300, "Nevada": 470,  # Las Vegas, Reno
    "Hawaii": 1000,  # Honolulu
    "Air Force": 750,  # Colorado Springs
    "Army": 20000,  # NYC metro (West Point)
    "Navy": 6300,  # DC metro (Annapolis)
    "Memphis": 1350,
    "Tulane": 1300,  # New Orleans
    "SMU": 7600,  # Dallas
    "Tulsa": 1000,
    "East Carolina": 180,  # Greenville NC
    "Marshall": 365,  # Huntington
    "Appalachian State": 100,  # Boone
    "Coastal Carolina": 480,  # Myrtle Beach
    "Liberty": 265,  # Lynchburg
    "James Madison": 55,  # Harrisonburg
    "Old Dominion": 1850,  # Norfolk
    "Charlotte": 2700,
    "Troy": 35,
    "South Alabama": 430,  # Mobile
    "Louisiana": 490,  # Lafayette
    "Louisiana Tech": 270,  # Ruston
    "Southern Miss": 150,  # Hattiesburg
    "UAB": 1150,  # Birmingham
    "Middle Tennessee": 2000,  # Nashville metro
    "Western Kentucky": 180,  # Bowling Green
    "Toledo": 650,
    "Bowling Green": 165,
    "Kent State": 2100,  # Cleveland metro
    "Akron": 700,
    "Ohio": 310,  # Athens
    "Buffalo": 1150,
    "Ball State": 120,  # Muncie
    "Central Michigan": 45,  # Mt. Pleasant
    "Eastern Michigan": 4400,  # Detroit metro
    "Western Michigan": 340,  # Kalamazoo
    "Northern Illinois": 9600,  # Chicago metro
    "UMass": 700,  # Springfield
    "UConn": 1200,  # Hartford
    "Connecticut": 1200,
}

# School state mapping for football state flag
SCHOOL_STATE = {
    "Alabama": "AL", "Auburn": "AL", "UAB": "AL", "South Alabama": "AL", "Troy": "AL",
    "Jacksonville State": "AL",
    "Arizona": "AZ", "Arizona State": "AZ",
    "Arkansas": "AR", "Arkansas State": "AR",
    "USC": "CA", "UCLA": "CA", "Stanford": "CA", "Cal": "CA", "San Diego State": "CA",
    "Fresno State": "CA", "San Jose State": "CA",
    "Colorado": "CO", "Colorado State": "CO", "Air Force": "CO",
    "UConn": "CT", "Connecticut": "CT",
    "Florida": "FL", "Florida State": "FL", "Miami": "FL", "UCF": "FL", "USF": "FL",
    "FAU": "FL", "FIU": "FL",
    "Georgia": "GA", "Georgia Tech": "GA", "Georgia State": "GA", "Georgia Southern": "GA",
    "Kennesaw State": "GA",
    "Hawaii": "HI",
    "Boise State": "ID",
    "Illinois": "IL", "Northwestern": "IL", "Northern Illinois": "IL",
    "Notre Dame": "IN", "Purdue": "IN", "Indiana": "IN", "Ball State": "IN",
    "Iowa": "IA", "Iowa State": "IA",
    "Kansas": "KS", "Kansas State": "KS",
    "Kentucky": "KY", "Louisville": "KY", "Western Kentucky": "KY",
    "LSU": "LA", "Louisiana": "LA", "Louisiana Tech": "LA", "Tulane": "LA",
    "Maryland": "MD", "Navy": "MD",
    "Boston College": "MA", "UMass": "MA",
    "Michigan": "MI", "Michigan State": "MI", "Central Michigan": "MI",
    "Eastern Michigan": "MI", "Western Michigan": "MI",
    "Minnesota": "MN",
    "Ole Miss": "MS", "Mississippi State": "MS", "Southern Miss": "MS",
    "Missouri": "MO",
    "Nebraska": "NE",
    "UNLV": "NV", "Nevada": "NV",
    "New Mexico": "NM", "New Mexico State": "NM",
    "Rutgers": "NJ",
    "Syracuse": "NY", "Army": "NY", "Buffalo": "NY",
    "North Carolina": "NC", "NC State": "NC", "Duke": "NC", "Wake Forest": "NC",
    "East Carolina": "NC", "Charlotte": "NC", "Appalachian State": "NC",
    "Ohio State": "OH", "Cincinnati": "OH", "Toledo": "OH", "Bowling Green": "OH",
    "Kent State": "OH", "Akron": "OH", "Ohio": "OH", "Miami (OH)": "OH",
    "Oklahoma": "OK", "Oklahoma State": "OK", "Tulsa": "OK",
    "Oregon": "OR", "Oregon State": "OR",
    "Penn State": "PA", "Pittsburgh": "PA", "Temple": "PA",
    "South Carolina": "SC", "Clemson": "SC", "Coastal Carolina": "SC",
    "Tennessee": "TN", "Vanderbilt": "TN", "Memphis": "TN", "Middle Tennessee": "TN",
    "Texas": "TX", "Texas A&M": "TX", "Texas Tech": "TX", "TCU": "TX", "Baylor": "TX",
    "Houston": "TX", "SMU": "TX", "North Texas": "TX", "UTSA": "TX", "UTEP": "TX",
    "Texas State": "TX", "Rice": "TX",
    "Utah": "UT", "BYU": "UT", "Utah State": "UT",
    "Virginia": "VA", "Virginia Tech": "VA", "Liberty": "VA", "James Madison": "VA",
    "Old Dominion": "VA",
    "Washington": "WA", "Washington State": "WA",
    "West Virginia": "WV", "Marshall": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

# Year classification mapping
YEAR_ENCODING = {
    "FR": 1, "Freshman": 1, "freshman": 1, "1": 1,
    "RS FR": 1.5, "Redshirt Freshman": 1.5,
    "SO": 2, "Sophomore": 2, "sophomore": 2, "2": 2,
    "RS SO": 2.5, "Redshirt Sophomore": 2.5,
    "JR": 3, "Junior": 3, "junior": 3, "3": 3,
    "RS JR": 3.5, "Redshirt Junior": 3.5,
    "SR": 4, "Senior": 4, "senior": 4, "4": 4,
    "RS SR": 4.5, "Redshirt Senior": 4.5,
    "GR": 5, "Graduate": 5, "5th Year": 5, "5": 5,
    "6th Year": 6, "6": 6,
}

# NIL tier thresholds (annual value)
NIL_TIERS = {
    "mega": 1_000_000,
    "premium": 500_000,
    "solid": 100_000,
    "moderate": 25_000,
    "entry": 0,
}

# Social media CPM rates for value estimation
DEFAULT_CPM_RATES = {
    "instagram": 10.0,
    "tiktok": 5.0,
    "twitter": 3.0,
    "youtube": 15.0,
}


class NILFeatureEngineer:
    """
    Feature engineering for NIL valuation models.

    Transforms raw collected data into model-ready features including:
    - Position-aware performance metrics
    - School and brand features
    - Recruiting profile features
    - Social media features
    - Draft projection features
    - Interaction features
    """

    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize the feature engineer.

        Args:
            output_dir: Directory to save processed features
        """
        self.output_dir = output_dir
        self.imputation_log = []  # Track all imputation decisions
        self.feature_names = []

    def build_features(
        self,
        player_data: pd.DataFrame,
        nil_data: pd.DataFrame,
        recruiting_data: pd.DataFrame,
        social_data: pd.DataFrame,
        team_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build model-ready features from raw collected data.

        Args:
            player_data: Player stats and info (from CFBStatsCollector)
            nil_data: NIL valuations (from CFBNILCollector)
            recruiting_data: Recruiting rankings (from CFBRecruitingCollector)
            social_data: Social media profiles (from CFBNILCollector)
            team_data: Team stats and info (from CFBStatsCollector)

        Returns:
            DataFrame with engineered features ready for modeling
        """
        logger.info("Starting NIL feature engineering...")
        self.imputation_log = []

        # Start with player data as base
        df = player_data.copy()
        logger.info(f"Starting with {len(df)} players")

        # Merge in other data sources
        df = self._merge_data_sources(df, nil_data, recruiting_data, social_data, team_data)
        logger.info(f"After merging: {len(df)} players")

        # Build feature groups
        df = self._build_position_features(df)
        df = self._build_performance_features(df)
        df = self._build_school_features(df, team_data)
        df = self._build_recruiting_features(df)
        df = self._build_social_features(df)
        df = self._build_draft_features(df)
        df = self._build_interaction_features(df)
        df = self._build_target_variables(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Drop players with too many missing features
        initial_count = len(df)
        df = self._drop_incomplete_rows(df)
        logger.info(f"Dropped {initial_count - len(df)} players with >50% missing features")

        # Select final feature columns
        df = self._select_final_features(df)

        # Save to file
        self._save_features(df)

        # Log imputation summary
        self._log_imputation_summary()

        logger.info(f"Feature engineering complete. Final dataset: {len(df)} players, {len(self.feature_names)} features")

        return df

    def _merge_data_sources(
        self,
        df: pd.DataFrame,
        nil_data: pd.DataFrame,
        recruiting_data: pd.DataFrame,
        social_data: pd.DataFrame,
        team_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all data sources into a single DataFrame."""

        # Standardize name columns for merging
        df = self._standardize_name_column(df)
        nil_data = self._standardize_name_column(nil_data)
        recruiting_data = self._standardize_name_column(recruiting_data)
        social_data = self._standardize_name_column(social_data)

        # Merge NIL data (left join to keep all players)
        if not nil_data.empty:
            nil_cols = [c for c in nil_data.columns if c not in df.columns or c == 'player_name_std']
            df = df.merge(
                nil_data[nil_cols],
                on='player_name_std',
                how='left',
                suffixes=('', '_nil')
            )
            self._log_imputation(f"NIL data merged: {nil_data['player_name_std'].nunique()} unique players matched")

        # Merge recruiting data
        if not recruiting_data.empty:
            recruit_cols = [c for c in recruiting_data.columns if c not in df.columns or c == 'player_name_std']
            df = df.merge(
                recruiting_data[recruit_cols],
                on='player_name_std',
                how='left',
                suffixes=('', '_recruit')
            )
            self._log_imputation(f"Recruiting data merged: {recruiting_data['player_name_std'].nunique()} unique players matched")

        # Merge social data
        if not social_data.empty:
            social_cols = [c for c in social_data.columns if c not in df.columns or c == 'player_name_std']
            df = df.merge(
                social_data[social_cols],
                on='player_name_std',
                how='left',
                suffixes=('', '_social')
            )
            self._log_imputation(f"Social data merged: {social_data['player_name_std'].nunique()} unique players matched")

        return df

    def _standardize_name_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create standardized name column for matching."""
        if df.empty:
            df['player_name_std'] = []
            return df

        # Find the name column
        name_cols = ['name', 'player_name', 'player', 'athlete_name', 'full_name']
        name_col = None
        for col in name_cols:
            if col in df.columns:
                name_col = col
                break

        if name_col is None:
            df['player_name_std'] = ''
            return df

        # Standardize: lowercase, remove special chars, strip whitespace
        df['player_name_std'] = (
            df[name_col]
            .astype(str)
            .str.lower()
            .str.replace(r'[^a-z\s]', '', regex=True)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )

        return df

    def _build_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position group mappings."""

        # Find position column
        pos_col = None
        for col in ['position', 'pos', 'player_position']:
            if col in df.columns:
                pos_col = col
                break

        if pos_col:
            df['position_raw'] = df[pos_col].astype(str).str.upper().str.strip()
            df['position_group'] = df['position_raw'].map(POSITION_TO_GROUP)

            # Fill unmapped positions
            unmapped = df['position_group'].isna().sum()
            if unmapped > 0:
                df['position_group'] = df['position_group'].fillna('ATH')
                self._log_imputation(f"Position group: {unmapped} unmapped positions set to 'ATH'")
        else:
            df['position_raw'] = 'UNKNOWN'
            df['position_group'] = 'ATH'
            self._log_imputation("No position column found, all players set to 'ATH'")

        return df

    def _build_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build position-aware performance features."""

        # Initialize all performance columns with NaN
        perf_cols = [
            # QB
            'pass_yards_per_game', 'pass_tds_per_game', 'completion_pct',
            'yards_per_attempt', 'td_int_ratio', 'rush_yards_per_game_qb',
            'total_tds_qb', 'passer_rating',
            # RB
            'rush_yards_per_game', 'rush_tds_per_game', 'yards_per_carry',
            'receptions_per_game_rb', 'rec_yards_per_game_rb', 'total_tds_rb',
            'scrimmage_yards_per_game',
            # WR/TE
            'receptions_per_game', 'rec_yards_per_game', 'rec_tds_per_game',
            'yards_per_reception', 'target_share_est',
            # Defense
            'tackles_per_game', 'tfl_per_game', 'sacks_per_game',
            'interceptions', 'pass_breakups',
            # Universal
            'production_score'
        ]

        for col in perf_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Calculate QB stats
        qb_mask = df['position_group'] == 'QB'
        df.loc[qb_mask, 'pass_yards_per_game'] = self._safe_divide(
            df.loc[qb_mask, 'passing_yards'] if 'passing_yards' in df.columns else 0,
            df.loc[qb_mask, 'games'] if 'games' in df.columns else 1
        )
        df.loc[qb_mask, 'pass_tds_per_game'] = self._safe_divide(
            df.loc[qb_mask, 'passing_tds'] if 'passing_tds' in df.columns else 0,
            df.loc[qb_mask, 'games'] if 'games' in df.columns else 1
        )
        if 'completions' in df.columns and 'pass_attempts' in df.columns:
            df.loc[qb_mask, 'completion_pct'] = self._safe_divide(
                df.loc[qb_mask, 'completions'],
                df.loc[qb_mask, 'pass_attempts']
            ) * 100
            df.loc[qb_mask, 'yards_per_attempt'] = self._safe_divide(
                df.loc[qb_mask, 'passing_yards'] if 'passing_yards' in df.columns else 0,
                df.loc[qb_mask, 'pass_attempts']
            )
        if 'passing_tds' in df.columns and 'interceptions_thrown' in df.columns:
            df.loc[qb_mask, 'td_int_ratio'] = self._safe_divide(
                df.loc[qb_mask, 'passing_tds'],
                df.loc[qb_mask, 'interceptions_thrown'].replace(0, 0.5)  # Avoid division by zero
            )
        if 'rushing_yards' in df.columns:
            df.loc[qb_mask, 'rush_yards_per_game_qb'] = self._safe_divide(
                df.loc[qb_mask, 'rushing_yards'],
                df.loc[qb_mask, 'games'] if 'games' in df.columns else 1
            )

        # Calculate RB stats
        rb_mask = df['position_group'] == 'RB'
        if 'rushing_yards' in df.columns:
            df.loc[rb_mask, 'rush_yards_per_game'] = self._safe_divide(
                df.loc[rb_mask, 'rushing_yards'],
                df.loc[rb_mask, 'games'] if 'games' in df.columns else 1
            )
        if 'rushing_tds' in df.columns:
            df.loc[rb_mask, 'rush_tds_per_game'] = self._safe_divide(
                df.loc[rb_mask, 'rushing_tds'],
                df.loc[rb_mask, 'games'] if 'games' in df.columns else 1
            )
        if 'rushing_yards' in df.columns and 'rush_attempts' in df.columns:
            df.loc[rb_mask, 'yards_per_carry'] = self._safe_divide(
                df.loc[rb_mask, 'rushing_yards'],
                df.loc[rb_mask, 'rush_attempts']
            )
        if 'receptions' in df.columns:
            df.loc[rb_mask, 'receptions_per_game_rb'] = self._safe_divide(
                df.loc[rb_mask, 'receptions'],
                df.loc[rb_mask, 'games'] if 'games' in df.columns else 1
            )
        if 'receiving_yards' in df.columns:
            df.loc[rb_mask, 'rec_yards_per_game_rb'] = self._safe_divide(
                df.loc[rb_mask, 'receiving_yards'],
                df.loc[rb_mask, 'games'] if 'games' in df.columns else 1
            )

        # Calculate scrimmage yards for RBs
        rush_yds = df['rushing_yards'] if 'rushing_yards' in df.columns else 0
        rec_yds = df['receiving_yards'] if 'receiving_yards' in df.columns else 0
        games = df['games'] if 'games' in df.columns else 1
        df.loc[rb_mask, 'scrimmage_yards_per_game'] = self._safe_divide(
            (rush_yds + rec_yds).loc[rb_mask], games.loc[rb_mask]
        )

        # Calculate WR/TE stats
        wr_te_mask = df['position_group'].isin(['WR', 'TE'])
        if 'receptions' in df.columns:
            df.loc[wr_te_mask, 'receptions_per_game'] = self._safe_divide(
                df.loc[wr_te_mask, 'receptions'],
                df.loc[wr_te_mask, 'games'] if 'games' in df.columns else 1
            )
        if 'receiving_yards' in df.columns:
            df.loc[wr_te_mask, 'rec_yards_per_game'] = self._safe_divide(
                df.loc[wr_te_mask, 'receiving_yards'],
                df.loc[wr_te_mask, 'games'] if 'games' in df.columns else 1
            )
        if 'receiving_tds' in df.columns:
            df.loc[wr_te_mask, 'rec_tds_per_game'] = self._safe_divide(
                df.loc[wr_te_mask, 'receiving_tds'],
                df.loc[wr_te_mask, 'games'] if 'games' in df.columns else 1
            )
        if 'receiving_yards' in df.columns and 'receptions' in df.columns:
            df.loc[wr_te_mask, 'yards_per_reception'] = self._safe_divide(
                df.loc[wr_te_mask, 'receiving_yards'],
                df.loc[wr_te_mask, 'receptions']
            )

        # Calculate defensive stats
        def_mask = df['position_group'].isin(['DL', 'EDGE', 'LB', 'CB', 'S'])
        if 'tackles' in df.columns:
            df.loc[def_mask, 'tackles_per_game'] = self._safe_divide(
                df.loc[def_mask, 'tackles'],
                df.loc[def_mask, 'games'] if 'games' in df.columns else 1
            )
        if 'tackles_for_loss' in df.columns:
            df.loc[def_mask, 'tfl_per_game'] = self._safe_divide(
                df.loc[def_mask, 'tackles_for_loss'],
                df.loc[def_mask, 'games'] if 'games' in df.columns else 1
            )
        if 'sacks' in df.columns:
            edge_lb_mask = df['position_group'].isin(['EDGE', 'LB', 'DL'])
            df.loc[edge_lb_mask, 'sacks_per_game'] = self._safe_divide(
                df.loc[edge_lb_mask, 'sacks'],
                df.loc[edge_lb_mask, 'games'] if 'games' in df.columns else 1
            )

        # Copy defensive counting stats
        if 'interceptions' in df.columns:
            df.loc[def_mask, 'interceptions'] = df.loc[def_mask, 'interceptions']
        if 'pass_breakups' in df.columns:
            df.loc[def_mask, 'pass_breakups'] = df.loc[def_mask, 'pass_breakups']

        # Calculate production score (0-100 percentile within position group)
        df = self._calculate_production_score(df)

        return df

    def _calculate_production_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized production score within each position group."""

        df['production_score'] = 50.0  # Default to median

        # Define key stats for each position group
        position_key_stats = {
            'QB': ['pass_yards_per_game', 'pass_tds_per_game', 'completion_pct'],
            'RB': ['rush_yards_per_game', 'yards_per_carry', 'scrimmage_yards_per_game'],
            'WR': ['rec_yards_per_game', 'receptions_per_game', 'rec_tds_per_game'],
            'TE': ['rec_yards_per_game', 'receptions_per_game'],
            'EDGE': ['sacks_per_game', 'tfl_per_game', 'tackles_per_game'],
            'DL': ['tfl_per_game', 'tackles_per_game'],
            'LB': ['tackles_per_game', 'tfl_per_game'],
            'CB': ['interceptions', 'pass_breakups', 'tackles_per_game'],
            'S': ['tackles_per_game', 'interceptions'],
        }

        for pos_group, key_stats in position_key_stats.items():
            mask = df['position_group'] == pos_group
            if mask.sum() == 0:
                continue

            # Get available stats for this position
            available_stats = [s for s in key_stats if s in df.columns]
            if not available_stats:
                continue

            # Calculate percentile for each stat and average
            percentiles = []
            for stat in available_stats:
                stat_values = df.loc[mask, stat]
                if stat_values.notna().sum() > 0:
                    pct = stat_values.rank(pct=True, na_option='keep') * 100
                    percentiles.append(pct)

            if percentiles:
                avg_percentile = pd.concat(percentiles, axis=1).mean(axis=1)
                df.loc[mask, 'production_score'] = avg_percentile.fillna(50)

        return df

    def _build_school_features(self, df: pd.DataFrame, team_data: pd.DataFrame) -> pd.DataFrame:
        """Build school and brand features."""

        # Find school column
        school_col = None
        for col in ['school', 'team', 'college', 'university']:
            if col in df.columns:
                school_col = col
                break

        if school_col:
            df['school_name'] = df[school_col].astype(str).str.strip()
        else:
            df['school_name'] = 'Unknown'
            self._log_imputation("No school column found, set to 'Unknown'")

        # School tier encoding
        df['school_tier'] = df['school_name'].map(SCHOOL_TIERS).map(SCHOOL_TIER_ENCODING)
        unmapped_schools = df['school_tier'].isna().sum()
        if unmapped_schools > 0:
            df['school_tier'] = df['school_tier'].fillna(1)  # Default to G5
            self._log_imputation(f"School tier: {unmapped_schools} unmapped schools defaulted to G5 (1)")

        # Conference tier (try to get from team_data or infer from school)
        if 'conference' in df.columns:
            df['conference_tier'] = df['conference'].map(CONFERENCE_TIER_ENCODING)
        else:
            # Infer from school tier
            df['conference_tier'] = df['school_tier'].apply(
                lambda x: 3 if x >= 5 else (2 if x >= 3 else 1)
            )
            self._log_imputation("Conference inferred from school tier")
        df['conference_tier'] = df['conference_tier'].fillna(1)

        # Team wins (from team_data if available)
        if not team_data.empty and 'wins' in team_data.columns:
            team_wins = team_data.groupby('team')['wins'].last().to_dict()
            df['team_wins'] = df['school_name'].map(team_wins)
        if 'team_wins' not in df.columns or df['team_wins'].isna().all():
            df['team_wins'] = 6  # Default to average
            self._log_imputation("Team wins not available, defaulted to 6")

        # Team win percentage (career)
        if 'career_wins' in df.columns and 'career_games' in df.columns:
            df['team_win_pct'] = self._safe_divide(df['career_wins'], df['career_games'])
        else:
            df['team_win_pct'] = 0.5  # Default to 50%
            self._log_imputation("Team win pct not available, defaulted to 0.5")

        # CFP appearance (binary)
        # Create set of CFP teams by year - simplified list of recent CFP teams
        cfp_teams = {
            'Alabama', 'Georgia', 'Ohio State', 'Michigan', 'Clemson', 'LSU',
            'Oklahoma', 'Notre Dame', 'Oregon', 'TCU', 'Cincinnati', 'Florida State',
            'Texas', 'Washington', 'Penn State', 'Tennessee'
        }
        df['cfp_appearance'] = df['school_name'].isin(cfp_teams).astype(int)

        # Conference championship (binary) - simplified
        conf_champ_teams = cfp_teams | {'USC', 'Utah', 'Tulane', 'Boise State', 'UCF'}
        df['conference_championship'] = df['school_name'].isin(conf_champ_teams).astype(int)

        # School market size
        df['school_market_size'] = df['school_name'].map(SCHOOL_METRO_POPULATION)
        median_market = 1000  # Default median market size
        unmapped_markets = df['school_market_size'].isna().sum()
        if unmapped_markets > 0:
            df['school_market_size'] = df['school_market_size'].fillna(median_market)
            self._log_imputation(f"Market size: {unmapped_markets} schools defaulted to {median_market}k")

        # Football state flag
        df['school_state'] = df['school_name'].map(SCHOOL_STATE)
        df['football_state'] = df['school_state'].isin(FOOTBALL_STATES).astype(int)
        df['football_state'] = df['football_state'].fillna(0)

        return df

    def _build_recruiting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build recruiting and profile features."""

        # Recruiting stars (2-5, or 0 if unrated)
        if 'stars' in df.columns:
            df['recruiting_stars'] = df['stars'].fillna(0).astype(int).clip(0, 5)
        elif 'rating' in df.columns:
            # Infer stars from rating
            df['recruiting_stars'] = pd.cut(
                df['rating'],
                bins=[0, 0.7, 0.8, 0.88, 0.95, 1.0],
                labels=[0, 2, 3, 4, 5]
            ).astype(float).fillna(0).astype(int)
        else:
            df['recruiting_stars'] = 0
            self._log_imputation("No recruiting stars data, defaulted to 0 (unrated)")

        # Recruiting composite (0-1)
        if 'rating' in df.columns:
            df['recruiting_composite'] = df['rating'].clip(0, 1)
        elif 'composite' in df.columns:
            df['recruiting_composite'] = df['composite'].clip(0, 1)
        else:
            # Estimate from stars
            star_to_composite = {0: 0.0, 2: 0.75, 3: 0.82, 4: 0.92, 5: 0.98}
            df['recruiting_composite'] = df['recruiting_stars'].map(star_to_composite)
            self._log_imputation("Recruiting composite estimated from stars")
        df['recruiting_composite'] = df['recruiting_composite'].fillna(0)

        # Position and class rankings
        if 'position_rank' in df.columns:
            df['recruiting_position_rank'] = df['position_rank'].fillna(999)
        else:
            df['recruiting_position_rank'] = 999
            self._log_imputation("Position rank not available, defaulted to 999")

        if 'national_rank' in df.columns:
            df['recruiting_class_rank'] = df['national_rank'].fillna(9999)
        elif 'overall_rank' in df.columns:
            df['recruiting_class_rank'] = df['overall_rank'].fillna(9999)
        else:
            df['recruiting_class_rank'] = 9999
            self._log_imputation("Class rank not available, defaulted to 9999")

        # Years in college
        if 'years_in_college' in df.columns:
            df['years_in_college'] = df['years_in_college'].clip(1, 6)
        elif 'year' in df.columns:
            df['years_in_college'] = df['year'].map(YEAR_ENCODING).fillna(3)
        else:
            df['years_in_college'] = 3  # Default to junior
            self._log_imputation("Years in college not available, defaulted to 3")

        # Year classification (numeric)
        if 'year' in df.columns:
            df['year_classification'] = df['year'].map(YEAR_ENCODING)
        else:
            df['year_classification'] = 3
        df['year_classification'] = df['year_classification'].fillna(3)

        # Remaining eligibility
        df['remaining_eligibility'] = 5 - df['years_in_college']
        df['remaining_eligibility'] = df['remaining_eligibility'].clip(0, 4)

        # Is starter (binary)
        if 'is_starter' in df.columns:
            df['is_starter'] = df['is_starter'].astype(int)
        elif 'games_started' in df.columns and 'games' in df.columns:
            df['is_starter'] = (df['games_started'] > df['games'] * 0.5).astype(int)
        else:
            df['is_starter'] = (df['production_score'] > 50).astype(int)
            self._log_imputation("Starter status inferred from production score")

        # Games started percentage
        if 'games_started' in df.columns and 'games_available' in df.columns:
            df['games_started_pct'] = self._safe_divide(df['games_started'], df['games_available'])
        elif 'games_started' in df.columns and 'games' in df.columns:
            # Estimate games available as games * (1 + missed_games_est)
            df['games_started_pct'] = self._safe_divide(df['games_started'], df['games'] * 1.1)
        else:
            df['games_started_pct'] = df['is_starter'] * 0.8
            self._log_imputation("Games started pct estimated from starter status")
        df['games_started_pct'] = df['games_started_pct'].clip(0, 1)

        # Age (if available)
        if 'age' in df.columns:
            df['age'] = df['age'].fillna(21)  # Default to 21
        elif 'birth_date' in df.columns:
            today = datetime.now()
            df['age'] = df['birth_date'].apply(
                lambda x: (today - pd.to_datetime(x)).days / 365.25 if pd.notna(x) else 21
            )
        else:
            # Estimate from years in college (assume 18 as freshman)
            df['age'] = 18 + df['years_in_college']
            self._log_imputation("Age estimated from years in college")

        return df

    def _build_social_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build social media features."""

        # Platform followers (fill with 0 if missing)
        social_platforms = ['instagram', 'tiktok', 'twitter', 'youtube']

        for platform in social_platforms:
            col = f'{platform}_followers'
            if col in df.columns:
                df[col] = df[col].fillna(0)
            else:
                df[col] = 0

        # Total social following
        df['total_social_following'] = (
            df['instagram_followers'] + df['tiktok_followers'] +
            df['twitter_followers'] + df['youtube_followers']
        )

        # Log transform (social follows power law)
        df['log_total_following'] = np.log1p(df['total_social_following'])
        df['instagram_followers_log'] = np.log1p(df['instagram_followers'])
        df['tiktok_followers_log'] = np.log1p(df['tiktok_followers'])
        df['twitter_followers_log'] = np.log1p(df['twitter_followers'])
        df['youtube_followers_log'] = np.log1p(df['youtube_followers'])

        # Estimated social value
        if 'estimated_social_value' in df.columns:
            df['estimated_social_value'] = df['estimated_social_value'].fillna(0)
        else:
            # Calculate using CPM formula
            df['estimated_social_value'] = self._estimate_social_value(df)
            self._log_imputation("Social value estimated using CPM formula")

        # Has significant following (>50K)
        df['has_significant_following'] = (df['total_social_following'] > 50000).astype(int)

        # Social platform concentration (% on largest platform)
        max_following = df[['instagram_followers', 'tiktok_followers',
                           'twitter_followers', 'youtube_followers']].max(axis=1)
        df['social_platform_concentration'] = self._safe_divide(
            max_following, df['total_social_following']
        )
        df['social_platform_concentration'] = df['social_platform_concentration'].fillna(1.0)

        return df

    def _estimate_social_value(self, df: pd.DataFrame) -> pd.Series:
        """Estimate annual social media value using CPM formula."""

        # Annual Value = followers * engagement_rate * posts_per_month * 12 * CPM / 1000
        # Assume 3% engagement, 4 posts/month for each platform
        engagement_rate = 0.03
        posts_per_month = 4

        value = 0
        for platform, cpm in DEFAULT_CPM_RATES.items():
            col = f'{platform}_followers'
            if col in df.columns:
                platform_value = (
                    df[col] * engagement_rate * posts_per_month * 12 * cpm / 1000
                )
                value = value + platform_value

        return value

    def _build_draft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build NFL draft projection features."""

        # Simple heuristic for draft projection
        # Based on: production + school tier + position value

        # Position draft value multiplier (some positions draft higher)
        position_draft_value = {
            'QB': 1.5, 'EDGE': 1.3, 'OL': 1.2, 'CB': 1.2,
            'WR': 1.1, 'DL': 1.1, 'LB': 1.0, 'S': 1.0,
            'RB': 0.9, 'TE': 0.9, 'ATH': 0.8, 'ST': 0.3,
        }

        df['position_draft_value'] = df['position_group'].map(position_draft_value).fillna(1.0)

        # Draft score (0-100)
        # 40% production, 30% school tier, 20% position value, 10% recruiting
        df['draft_score'] = (
            0.40 * df['production_score'] +
            0.30 * (df['school_tier'] / 6 * 100) +
            0.20 * (df['position_draft_value'] * 50) +
            0.10 * (df['recruiting_composite'] * 100)
        )

        # Projected draft flag (top ~300 players get drafted each year)
        # Use threshold of 60+ draft score
        df['projected_draft_flag'] = (df['draft_score'] >= 60).astype(int)

        # Projected draft round (heuristic)
        def project_round(score):
            if score >= 85:
                return 1
            elif score >= 75:
                return 2
            elif score >= 68:
                return 3
            elif score >= 62:
                return 4
            elif score >= 58:
                return 5
            elif score >= 55:
                return 6
            elif score >= 52:
                return 7
            else:
                return 0  # Undrafted projection

        df['projected_draft_round'] = df['draft_score'].apply(project_round)

        return df

    def _build_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build interaction features."""

        # School x production (great player at great school = premium)
        df['school_x_production'] = df['school_tier'] * df['production_score']

        # Social x production
        df['social_x_production'] = df['log_total_following'] * df['production_score']

        # QB premium flag
        df['qb_premium'] = (df['position_group'] == 'QB').astype(int)

        # Additional useful interactions
        df['market_x_production'] = np.log1p(df['school_market_size']) * df['production_score']
        df['recruit_x_production'] = df['recruiting_composite'] * df['production_score']

        return df

    def _build_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build target variables for modeling."""

        # NIL value (regression target)
        if 'nil_value' in df.columns:
            df['nil_value'] = df['nil_value'].fillna(0)
        elif 'annual_nil_value' in df.columns:
            df['nil_value'] = df['annual_nil_value'].fillna(0)
        elif 'valuation' in df.columns:
            df['nil_value'] = df['valuation'].fillna(0)
        else:
            # No NIL value - will need to be provided separately or estimated
            df['nil_value'] = np.nan
            self._log_imputation("NIL value not available - target variable is NaN")

        # NIL tier (classification target)
        def classify_nil_tier(value):
            if pd.isna(value):
                return np.nan
            if value >= NIL_TIERS['mega']:
                return 'mega'
            elif value >= NIL_TIERS['premium']:
                return 'premium'
            elif value >= NIL_TIERS['solid']:
                return 'solid'
            elif value >= NIL_TIERS['moderate']:
                return 'moderate'
            else:
                return 'entry'

        df['nil_tier'] = df['nil_value'].apply(classify_nil_tier)

        # Numeric encoding for nil_tier
        tier_encoding = {'entry': 0, 'moderate': 1, 'solid': 2, 'premium': 3, 'mega': 4}
        df['nil_tier_encoded'] = df['nil_tier'].map(tier_encoding)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with position-group aware imputation."""

        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Exclude target variables from imputation
        target_cols = ['nil_value', 'nil_tier', 'nil_tier_encoded']
        numeric_cols = [c for c in numeric_cols if c not in target_cols]
        categorical_cols = [c for c in categorical_cols if c not in target_cols]

        # Numeric: fill with position-group median
        for col in numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                # Try position-group median first
                group_medians = df.groupby('position_group')[col].transform('median')
                df[col] = df[col].fillna(group_medians)

                # If still missing (entire position group is NaN), use overall median
                remaining_missing = df[col].isna().sum()
                if remaining_missing > 0:
                    overall_median = df[col].median()
                    if pd.isna(overall_median):
                        overall_median = 0
                    df[col] = df[col].fillna(overall_median)
                    self._log_imputation(f"{col}: {remaining_missing} values filled with overall median ({overall_median:.2f})")
                else:
                    self._log_imputation(f"{col}: {missing_count} values filled with position-group median")

        # Categorical: fill with mode or 'unknown'
        for col in categorical_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
                    self._log_imputation(f"{col}: {missing_count} values filled with mode ('{mode_value[0]}')")
                else:
                    df[col] = df[col].fillna('unknown')
                    self._log_imputation(f"{col}: {missing_count} values filled with 'unknown'")

        return df

    def _drop_incomplete_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with >50% missing features."""

        # Get feature columns (exclude metadata and targets)
        exclude_cols = ['player_name_std', 'player_name', 'name', 'school_name',
                       'nil_value', 'nil_tier', 'nil_tier_encoded', 'position_raw',
                       'school_state']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Calculate missing percentage per row
        missing_pct = df[feature_cols].isna().sum(axis=1) / len(feature_cols)

        # Drop rows with >50% missing
        mask = missing_pct <= 0.5
        dropped_count = (~mask).sum()

        if dropped_count > 0:
            self._log_imputation(f"Dropped {dropped_count} rows with >50% missing features")

        return df[mask].reset_index(drop=True)

    def _select_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order final feature columns."""

        # Define feature column order
        self.feature_names = [
            # Performance features
            'production_score',
            'pass_yards_per_game', 'pass_tds_per_game', 'completion_pct',
            'yards_per_attempt', 'td_int_ratio', 'passer_rating',
            'rush_yards_per_game', 'rush_tds_per_game', 'yards_per_carry',
            'scrimmage_yards_per_game',
            'receptions_per_game', 'rec_yards_per_game', 'rec_tds_per_game',
            'yards_per_reception', 'target_share_est',
            'tackles_per_game', 'tfl_per_game', 'sacks_per_game',
            'interceptions', 'pass_breakups',

            # School features
            'school_tier', 'conference_tier', 'team_wins', 'team_win_pct',
            'cfp_appearance', 'conference_championship',
            'school_market_size', 'football_state',

            # Recruiting features
            'recruiting_stars', 'recruiting_composite',
            'recruiting_position_rank', 'recruiting_class_rank',
            'years_in_college', 'year_classification', 'remaining_eligibility',
            'is_starter', 'games_started_pct', 'age',

            # Social features
            'total_social_following', 'log_total_following',
            'instagram_followers_log', 'tiktok_followers_log',
            'twitter_followers_log', 'youtube_followers_log',
            'estimated_social_value', 'has_significant_following',
            'social_platform_concentration',

            # Draft features
            'projected_draft_flag', 'projected_draft_round', 'draft_score',

            # Interaction features
            'school_x_production', 'social_x_production', 'qb_premium',
            'market_x_production', 'recruit_x_production',
        ]

        # Filter to only columns that exist
        self.feature_names = [c for c in self.feature_names if c in df.columns]

        # Metadata columns to keep
        metadata_cols = ['player_name_std', 'position_group', 'school_name']
        metadata_cols = [c for c in metadata_cols if c in df.columns]

        # Target columns
        target_cols = ['nil_value', 'nil_tier', 'nil_tier_encoded']
        target_cols = [c for c in target_cols if c in df.columns]

        # Select and order columns
        final_cols = metadata_cols + self.feature_names + target_cols
        df = df[final_cols]

        return df

    def _save_features(self, df: pd.DataFrame) -> None:
        """Save engineered features to CSV."""
        import os

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'nil_features_ready.csv')

        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")

        # Also save imputation log
        log_path = os.path.join(self.output_dir, 'nil_feature_imputation_log.txt')
        with open(log_path, 'w') as f:
            f.write("NIL Feature Engineering Imputation Log\n")
            f.write("=" * 50 + "\n\n")
            for entry in self.imputation_log:
                f.write(f"- {entry}\n")
        logger.info(f"Imputation log saved to {log_path}")

    def _log_imputation(self, message: str) -> None:
        """Log an imputation decision."""
        self.imputation_log.append(message)
        logger.debug(f"Imputation: {message}")

    def _log_imputation_summary(self) -> None:
        """Print summary of all imputation decisions."""
        logger.info(f"\nImputation Summary ({len(self.imputation_log)} decisions):")
        for entry in self.imputation_log:
            logger.info(f"  - {entry}")

    @staticmethod
    def _safe_divide(numerator, denominator) -> pd.Series:
        """Safe division handling zeros and NaN."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result = result.replace([np.inf, -np.inf], np.nan)
        return result

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names (excluding targets).

        Returns:
            List of feature column names
        """
        return self.feature_names.copy()

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get features grouped by category for SHAP analysis.

        Returns:
            Dictionary mapping category names to feature lists
        """
        return {
            'performance': [
                'production_score',
                'pass_yards_per_game', 'pass_tds_per_game', 'completion_pct',
                'yards_per_attempt', 'td_int_ratio', 'passer_rating',
                'rush_yards_per_game', 'rush_tds_per_game', 'yards_per_carry',
                'scrimmage_yards_per_game',
                'receptions_per_game', 'rec_yards_per_game', 'rec_tds_per_game',
                'yards_per_reception', 'target_share_est',
                'tackles_per_game', 'tfl_per_game', 'sacks_per_game',
                'interceptions', 'pass_breakups',
            ],
            'school_brand': [
                'school_tier', 'conference_tier', 'team_wins', 'team_win_pct',
                'cfp_appearance', 'conference_championship',
                'school_market_size', 'football_state',
            ],
            'recruiting_profile': [
                'recruiting_stars', 'recruiting_composite',
                'recruiting_position_rank', 'recruiting_class_rank',
                'years_in_college', 'year_classification', 'remaining_eligibility',
                'is_starter', 'games_started_pct', 'age',
            ],
            'social_media': [
                'total_social_following', 'log_total_following',
                'instagram_followers_log', 'tiktok_followers_log',
                'twitter_followers_log', 'youtube_followers_log',
                'estimated_social_value', 'has_significant_following',
                'social_platform_concentration',
            ],
            'draft_projection': [
                'projected_draft_flag', 'projected_draft_round', 'draft_score',
            ],
            'interactions': [
                'school_x_production', 'social_x_production', 'qb_premium',
                'market_x_production', 'recruit_x_production',
            ],
        }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    print("NIL Feature Engineer - Standalone Mode")
    print("=" * 50)

    # Create sample data for testing
    print("\nCreating sample test data...")

    # Sample player data
    player_data = pd.DataFrame({
        'name': ['Caleb Williams', 'Travis Hunter', 'Quinn Ewers', 'Marvin Harrison Jr'],
        'position': ['QB', 'WR', 'QB', 'WR'],
        'school': ['USC', 'Colorado', 'Texas', 'Ohio State'],
        'games': [13, 12, 11, 13],
        'passing_yards': [4500, 0, 3800, 0],
        'passing_tds': [42, 0, 32, 0],
        'completions': [320, 0, 280, 0],
        'pass_attempts': [450, 0, 400, 0],
        'interceptions_thrown': [5, 0, 8, 0],
        'rushing_yards': [350, 100, 200, 50],
        'receptions': [0, 95, 0, 110],
        'receiving_yards': [0, 1450, 0, 1650],
        'receiving_tds': [0, 12, 0, 15],
        'year': ['SR', 'JR', 'JR', 'JR'],
    })

    # Sample NIL data
    nil_data = pd.DataFrame({
        'name': ['Caleb Williams', 'Travis Hunter', 'Quinn Ewers', 'Marvin Harrison Jr'],
        'nil_value': [4500000, 3800000, 2500000, 1800000],
    })

    # Sample recruiting data
    recruiting_data = pd.DataFrame({
        'name': ['Caleb Williams', 'Travis Hunter', 'Quinn Ewers', 'Marvin Harrison Jr'],
        'stars': [5, 5, 5, 5],
        'rating': [0.9998, 1.0000, 0.9995, 0.9990],
        'national_rank': [2, 1, 1, 4],
        'position_rank': [1, 1, 1, 1],
    })

    # Sample social data
    social_data = pd.DataFrame({
        'name': ['Caleb Williams', 'Travis Hunter', 'Quinn Ewers', 'Marvin Harrison Jr'],
        'instagram_followers': [850000, 1200000, 450000, 380000],
        'tiktok_followers': [500000, 2500000, 200000, 150000],
        'twitter_followers': [320000, 680000, 280000, 220000],
        'youtube_followers': [50000, 180000, 30000, 25000],
    })

    # Sample team data
    team_data = pd.DataFrame({
        'team': ['USC', 'Colorado', 'Texas', 'Ohio State'],
        'wins': [10, 9, 12, 11],
        'conference': ['Big Ten', 'Big 12', 'SEC', 'Big Ten'],
    })

    # Initialize and run feature engineering
    engineer = NILFeatureEngineer(output_dir="data/processed")

    try:
        features_df = engineer.build_features(
            player_data=player_data,
            nil_data=nil_data,
            recruiting_data=recruiting_data,
            social_data=social_data,
            team_data=team_data
        )

        print(f"\n✓ Feature engineering complete!")
        print(f"  - Output shape: {features_df.shape}")
        print(f"  - Features: {len(engineer.get_feature_names())}")

        print("\nFeature groups for SHAP analysis:")
        for group, features in engineer.get_feature_importance_groups().items():
            print(f"  - {group}: {len(features)} features")

        print("\nSample output:")
        print(features_df[['player_name_std', 'position_group', 'production_score',
                          'school_tier', 'nil_value', 'nil_tier']].to_string())

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
