"""
NFL Draft History Collector

Collects draft picks, combine data, and links to college production and NFL outcomes.
Used to train and validate draft projection models.

Data Sources:
- nfl_data_py: Draft picks, combine data, NFL player stats
- CFB stats: College production data
- Recruiting data: Star ratings and rankings
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd
import numpy as np
from dotenv import load_dotenv

try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DraftHistoryCollector:
    """
    Collects NFL draft history and links to college/NFL production.

    Creates unified datasets for draft model training and validation:
    - College stats + recruiting + combine + draft result + NFL production

    Features:
    - Draft pick collection via nfl_data_py
    - Fuzzy matching to college stats
    - Combine data integration
    - Rookie wage scale contract values
    - NFL production tracking (draft hit rate)
    """

    RATE_LIMIT_SECONDS = 0.5
    CACHE_HOURS = 24
    FUZZY_MATCH_THRESHOLD = 0.85

    # NFL Rookie Wage Scale (2024 values, approximate)
    # Format: {round: {pick_in_round: (total_4yr_value, signing_bonus)}}
    # These are approximations - actual values vary slightly year to year
    ROOKIE_WAGE_SCALE = {
        1: {
            1: (41_000_000, 27_000_000),
            2: (37_000_000, 24_000_000),
            3: (34_000_000, 22_000_000),
            4: (32_000_000, 20_500_000),
            5: (30_000_000, 19_000_000),
            10: (22_000_000, 13_000_000),
            15: (17_500_000, 10_000_000),
            20: (14_500_000, 7_500_000),
            25: (12_500_000, 6_000_000),
            32: (10_500_000, 4_800_000),
        },
        2: {
            1: (9_500_000, 4_200_000),
            10: (7_800_000, 2_800_000),
            20: (6_500_000, 1_900_000),
            32: (5_800_000, 1_400_000),
        },
        3: {
            1: (5_500_000, 1_200_000),
            15: (5_000_000, 950_000),
            32: (4_600_000, 750_000),
        },
        4: {
            1: (4_400_000, 650_000),
            15: (4_200_000, 550_000),
            32: (4_000_000, 450_000),
        },
        5: {
            1: (3_900_000, 400_000),
            15: (3_800_000, 350_000),
            32: (3_700_000, 300_000),
        },
        6: {
            1: (3_600_000, 250_000),
            15: (3_550_000, 220_000),
            32: (3_500_000, 200_000),
        },
        7: {
            1: (3_450_000, 175_000),
            15: (3_420_000, 160_000),
            32: (3_400_000, 150_000),
        },
    }

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the draft history collector.

        Args:
            data_dir: Base directory for data storage
        """
        load_dotenv()

        if not NFL_DATA_AVAILABLE:
            logger.warning("nfl_data_py not installed. Some features unavailable.")

        # Find data directory
        if data_dir is None:
            current = Path(__file__).parent
            while current.parent != current:
                if (current / "config.yaml").exists():
                    data_dir = str(current / "data")
                    break
                current = current.parent
            else:
                data_dir = "data"

        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Caches for loaded data
        self._college_stats_cache: Optional[pd.DataFrame] = None
        self._recruiting_cache: Optional[pd.DataFrame] = None

        logger.info(f"DraftHistoryCollector initialized. Data dir: {self.data_dir}")

    def _rate_limit(self) -> None:
        """Apply rate limiting between API calls."""
        time.sleep(self.RATE_LIMIT_SECONDS)

    def _get_cache_path(self, name: str, start_year: int, end_year: int) -> Path:
        """Get cache file path for a dataset."""
        return self.cache_dir / f"nfl_{name}_{start_year}_{end_year}_cache.csv"

    def _get_cache_meta_path(self, name: str, start_year: int, end_year: int) -> Path:
        """Get cache metadata file path."""
        return self.cache_dir / f"nfl_{name}_{start_year}_{end_year}_cache_meta.txt"

    def _is_cache_valid(self, name: str, start_year: int, end_year: int) -> bool:
        """Check if cached data exists and is less than 24 hours old."""
        cache_path = self._get_cache_path(name, start_year, end_year)
        meta_path = self._get_cache_meta_path(name, start_year, end_year)

        if not cache_path.exists() or not meta_path.exists():
            return False

        try:
            with open(meta_path, "r") as f:
                cache_time = datetime.fromisoformat(f.read().strip())

            age = datetime.now() - cache_time
            if age < timedelta(hours=self.CACHE_HOURS):
                logger.info(f"Valid cache found for {name} ({age.total_seconds() / 3600:.1f} hours old)")
                return True
            return False
        except Exception as e:
            logger.warning(f"Error reading cache metadata: {e}")
            return False

    def _load_cache(self, name: str, start_year: int, end_year: int) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        if self._is_cache_valid(name, start_year, end_year):
            try:
                cache_path = self._get_cache_path(name, start_year, end_year)
                df = pd.read_csv(cache_path)
                logger.info(f"Loaded {len(df)} rows from cache: {cache_path.name}")
                return df
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        return None

    def _save_cache(self, df: pd.DataFrame, name: str, start_year: int, end_year: int) -> None:
        """Save data to cache with timestamp."""
        try:
            cache_path = self._get_cache_path(name, start_year, end_year)
            meta_path = self._get_cache_meta_path(name, start_year, end_year)

            df.to_csv(cache_path, index=False)
            with open(meta_path, "w") as f:
                f.write(datetime.now().isoformat())

            logger.info(f"Saved {len(df)} rows to cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")

    def collect_draft_picks(
        self,
        start_year: int = 2018,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Collect NFL draft picks via nfl_data_py.

        Args:
            start_year: First draft year (default: 2018)
            end_year: Last draft year (default: 2025)

        Returns:
            DataFrame with draft picks including:
            - player name, college, position, round, pick, team, age
        """
        # Check cache first
        cached = self._load_cache("draft_picks", start_year, end_year)
        if cached is not None:
            return cached

        if not NFL_DATA_AVAILABLE:
            logger.error("nfl_data_py required for draft picks")
            return pd.DataFrame()

        logger.info(f"Collecting draft picks for {start_year}-{end_year}")

        try:
            years = list(range(start_year, end_year + 1))
            self._rate_limit()
            draft_df = nfl.import_draft_picks(years)

            if draft_df.empty:
                logger.warning("No draft data returned")
                return pd.DataFrame()

            # Standardize column names
            column_mapping = {
                "pfr_player_name": "player_name",
                "player_name": "player_name",
                "pfr_name": "player_name",
                "college": "college",
                "school": "college",
                "position": "position",
                "pos": "position",
                "round": "round",
                "pick": "pick",
                "overall": "overall_pick",
                "team": "team",
                "age": "age_at_draft",
                "season": "draft_year",
                "year": "draft_year",
            }

            # Rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in draft_df.columns and new_col not in draft_df.columns:
                    draft_df = draft_df.rename(columns={old_col: new_col})

            # Calculate overall pick if not present
            if "overall_pick" not in draft_df.columns and "round" in draft_df.columns and "pick" in draft_df.columns:
                draft_df["overall_pick"] = (draft_df["round"] - 1) * 32 + draft_df["pick"]

            # Save to cache
            self._save_cache(draft_df, "draft_picks", start_year, end_year)

            # Save to raw
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = self.raw_dir / f"nfl_draft_picks_{start_year}_{end_year}_{timestamp}.csv"
            draft_df.to_csv(raw_path, index=False)

            logger.info(f"Collected {len(draft_df)} draft picks")
            return draft_df

        except Exception as e:
            logger.error(f"Error collecting draft picks: {e}")
            return pd.DataFrame()

    def collect_combine_data(
        self,
        start_year: int = 2018,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Collect NFL combine data via nfl_data_py.

        Args:
            start_year: First combine year
            end_year: Last combine year

        Returns:
            DataFrame with combine measurements:
            - 40-yard, bench, vertical, broad jump, 3-cone, shuttle
            - height, weight, arm length, hand size
        """
        # Check cache first
        cached = self._load_cache("combine", start_year, end_year)
        if cached is not None:
            return cached

        if not NFL_DATA_AVAILABLE:
            logger.error("nfl_data_py required for combine data")
            return pd.DataFrame()

        logger.info(f"Collecting combine data for {start_year}-{end_year}")

        try:
            years = list(range(start_year, end_year + 1))
            self._rate_limit()
            combine_df = nfl.import_combine_data(years)

            if combine_df.empty:
                logger.warning("No combine data returned")
                return pd.DataFrame()

            # Standardize column names
            column_mapping = {
                "player_name": "player_name",
                "pfr_player_name": "player_name",
                "school": "college",
                "college": "college",
                "pos": "position",
                "position": "position",
                "ht": "height",
                "wt": "weight",
                "forty": "forty_yard",
                "vertical": "vertical_jump",
                "bench": "bench_press",
                "broad_jump": "broad_jump",
                "cone": "three_cone",
                "shuttle": "shuttle",
                "arm_length": "arm_length",
                "hand_size": "hand_size",
            }

            for old_col, new_col in column_mapping.items():
                if old_col in combine_df.columns and new_col not in combine_df.columns:
                    combine_df = combine_df.rename(columns={old_col: new_col})

            # Save to cache
            self._save_cache(combine_df, "combine", start_year, end_year)

            logger.info(f"Collected {len(combine_df)} combine records")
            return combine_df

        except Exception as e:
            logger.error(f"Error collecting combine data: {e}")
            return pd.DataFrame()

    def collect_nfl_player_stats(
        self,
        start_year: int = 2018,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Collect NFL player stats for measuring draft hit rate.

        Gets first 1-3 years of NFL production for drafted players.

        Args:
            start_year: First season
            end_year: Last season

        Returns:
            DataFrame with NFL player stats
        """
        # Check cache first
        cached = self._load_cache("nfl_stats", start_year, end_year)
        if cached is not None:
            return cached

        if not NFL_DATA_AVAILABLE:
            logger.error("nfl_data_py required for NFL stats")
            return pd.DataFrame()

        logger.info(f"Collecting NFL player stats for {start_year}-{end_year}")

        try:
            years = list(range(start_year, end_year + 1))
            self._rate_limit()
            stats_df = nfl.import_seasonal_data(years)

            if stats_df.empty:
                logger.warning("No NFL stats returned")
                return pd.DataFrame()

            # Save to cache
            self._save_cache(stats_df, "nfl_stats", start_year, end_year)

            logger.info(f"Collected {len(stats_df)} NFL player-season records")
            return stats_df

        except Exception as e:
            logger.error(f"Error collecting NFL stats: {e}")
            return pd.DataFrame()

    def get_rookie_contract_value(
        self,
        round_num: int,
        pick_in_round: int,
    ) -> Tuple[float, float]:
        """
        Get estimated rookie contract value from wage scale.

        Args:
            round_num: Draft round (1-7)
            pick_in_round: Pick number within round (1-32)

        Returns:
            Tuple of (total_4yr_value, signing_bonus)
        """
        if round_num not in self.ROOKIE_WAGE_SCALE:
            # Default to minimum values for undrafted
            return (3_400_000, 100_000)

        round_scale = self.ROOKIE_WAGE_SCALE[round_num]

        # Find closest pick in the scale
        available_picks = sorted(round_scale.keys())
        closest_pick = min(available_picks, key=lambda x: abs(x - pick_in_round))

        # Interpolate if not exact match
        if pick_in_round == closest_pick:
            return round_scale[closest_pick]

        # Linear interpolation between available points
        lower_picks = [p for p in available_picks if p <= pick_in_round]
        upper_picks = [p for p in available_picks if p >= pick_in_round]

        if not lower_picks:
            return round_scale[available_picks[0]]
        if not upper_picks:
            return round_scale[available_picks[-1]]

        lower_pick = max(lower_picks)
        upper_pick = min(upper_picks)

        if lower_pick == upper_pick:
            return round_scale[lower_pick]

        # Interpolate
        lower_val = round_scale[lower_pick]
        upper_val = round_scale[upper_pick]

        ratio = (pick_in_round - lower_pick) / (upper_pick - lower_pick)

        total = lower_val[0] + (upper_val[0] - lower_val[0]) * ratio
        bonus = lower_val[1] + (upper_val[1] - lower_val[1]) * ratio

        return (int(total), int(bonus))

    def _load_college_stats(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Load college stats for matching."""
        if self._college_stats_cache is not None:
            return self._college_stats_cache

        # Try loading from cache
        cache_path = self.cache_dir / f"cfb_player_stats_{start_year}_{end_year}_cache.csv"
        if cache_path.exists():
            try:
                self._college_stats_cache = pd.read_csv(cache_path)
                logger.info(f"Loaded {len(self._college_stats_cache)} college stats from cache")
                return self._college_stats_cache
            except Exception as e:
                logger.debug(f"Could not load college stats cache: {e}")

        # Try to collect fresh
        try:
            from .cfb_stats import CFBStatsCollector
            collector = CFBStatsCollector(str(self.data_dir))
            self._college_stats_cache = collector.collect_player_stats(start_year, end_year)
            return self._college_stats_cache
        except Exception as e:
            logger.warning(f"Could not load college stats: {e}")
            return pd.DataFrame()

    def _load_recruiting_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Load recruiting data for matching."""
        if self._recruiting_cache is not None:
            return self._recruiting_cache

        cache_path = self.cache_dir / f"cfb_recruiting_players_{start_year}_{end_year}_cache.csv"
        if cache_path.exists():
            try:
                self._recruiting_cache = pd.read_csv(cache_path)
                logger.info(f"Loaded {len(self._recruiting_cache)} recruiting records from cache")
                return self._recruiting_cache
            except Exception as e:
                logger.debug(f"Could not load recruiting cache: {e}")

        try:
            from .cfb_recruiting import CFBRecruitingCollector
            collector = CFBRecruitingCollector(str(self.data_dir))
            self._recruiting_cache = collector.collect_player_recruiting_rankings(start_year, end_year)
            return self._recruiting_cache
        except Exception as e:
            logger.warning(f"Could not load recruiting data: {e}")
            return pd.DataFrame()

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for matching."""
        if not name:
            return ""
        name = name.lower().strip()
        # Remove suffixes
        for suffix in [" jr", " jr.", " sr", " sr.", " ii", " iii", " iv", " v"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        # Remove punctuation
        name = "".join(c for c in name if c.isalnum() or c.isspace())
        return " ".join(name.split())

    def _fuzzy_match(
        self,
        name: str,
        college: str,
        candidates: pd.DataFrame,
        name_col: str = "player_name",
        school_col: str = "team",
        threshold: float = None,
    ) -> Optional[pd.Series]:
        """
        Fuzzy match a player to candidates by name and college.

        Args:
            name: Player name to match
            college: College name to match
            candidates: DataFrame of candidates
            name_col: Name column in candidates
            school_col: School column in candidates
            threshold: Match threshold

        Returns:
            Matched row or None
        """
        if threshold is None:
            threshold = self.FUZZY_MATCH_THRESHOLD

        if candidates.empty:
            return None

        name_norm = self._normalize_name(name)
        college_norm = self._normalize_name(college) if college else ""

        best_match = None
        best_score = 0.0

        for idx, row in candidates.iterrows():
            cand_name = self._normalize_name(str(row.get(name_col, "")))
            cand_school = self._normalize_name(str(row.get(school_col, "")))

            # Name similarity
            name_score = SequenceMatcher(None, name_norm, cand_name).ratio()

            # School similarity bonus
            school_bonus = 0.0
            if college_norm and cand_school:
                school_score = SequenceMatcher(None, college_norm, cand_school).ratio()
                if school_score > 0.8:
                    school_bonus = 0.1

            total_score = name_score + school_bonus

            if total_score > best_score:
                best_score = total_score
                best_match = row

        if best_score >= threshold and best_match is not None:
            return best_match

        return None

    def match_draft_to_college_stats(
        self,
        draft_df: pd.DataFrame,
        college_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Match drafted players to their college stats.

        Gets final 1-2 seasons of college production.

        Args:
            draft_df: Draft picks DataFrame
            college_stats: College stats DataFrame

        Returns:
            Draft DataFrame with college stats columns added
        """
        logger.info("Matching draft picks to college stats...")

        if college_stats.empty:
            logger.warning("No college stats available for matching")
            return draft_df

        # Determine name column
        name_col = "player_name" if "player_name" in college_stats.columns else "player"
        school_col = "team" if "team" in college_stats.columns else "school"

        matched_data = []
        matched_count = 0

        for idx, draft_row in draft_df.iterrows():
            player_name = draft_row.get("player_name", "")
            college = draft_row.get("college", "")
            draft_year = draft_row.get("draft_year", 2024)

            result = draft_row.to_dict()

            # Filter college stats to years before draft
            pre_draft_stats = college_stats[
                college_stats.get("season", 0) < draft_year
            ] if "season" in college_stats.columns else college_stats

            # Match player
            match = self._fuzzy_match(
                player_name, college, pre_draft_stats,
                name_col=name_col, school_col=school_col
            )

            if match is not None:
                # Get final season stats
                player_stats = pre_draft_stats[
                    pre_draft_stats[name_col].apply(
                        lambda x: SequenceMatcher(None, self._normalize_name(str(x)),
                                                  self._normalize_name(player_name)).ratio() > 0.85
                    )
                ]

                if not player_stats.empty:
                    # Get last 2 seasons
                    if "season" in player_stats.columns:
                        player_stats = player_stats.sort_values("season", ascending=False)

                    final_season = player_stats.iloc[0]

                    # Add college stats with prefix
                    for col in final_season.index:
                        if col not in [name_col, school_col, "season"]:
                            result[f"college_{col}"] = final_season[col]

                    result["college_seasons_matched"] = min(len(player_stats), 2)
                    matched_count += 1

            matched_data.append(result)

        logger.info(f"Matched {matched_count}/{len(draft_df)} draft picks to college stats")
        return pd.DataFrame(matched_data)

    def match_draft_to_recruiting(
        self,
        draft_df: pd.DataFrame,
        recruiting_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Match drafted players to their recruiting rankings.

        Args:
            draft_df: Draft picks DataFrame
            recruiting_df: Recruiting DataFrame

        Returns:
            Draft DataFrame with recruiting columns added
        """
        logger.info("Matching draft picks to recruiting data...")

        if recruiting_df.empty:
            logger.warning("No recruiting data available")
            return draft_df

        name_col = "name" if "name" in recruiting_df.columns else "player_name"

        matched_data = []
        matched_count = 0

        for idx, draft_row in draft_df.iterrows():
            player_name = draft_row.get("player_name", "")
            college = draft_row.get("college", "")

            result = draft_row.to_dict()

            # Match to recruiting
            match = self._fuzzy_match(
                player_name, college, recruiting_df,
                name_col=name_col, school_col="school"
            )

            if match is not None:
                result["recruiting_stars"] = match.get("stars")
                result["recruiting_rating"] = match.get("rating")
                result["recruiting_rank"] = match.get("national_ranking") or match.get("ranking")
                matched_count += 1

            matched_data.append(result)

        logger.info(f"Matched {matched_count}/{len(draft_df)} draft picks to recruiting")
        return pd.DataFrame(matched_data)

    def match_draft_to_combine(
        self,
        draft_df: pd.DataFrame,
        combine_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Match drafted players to their combine data.

        Args:
            draft_df: Draft picks DataFrame
            combine_df: Combine DataFrame

        Returns:
            Draft DataFrame with combine columns added
        """
        logger.info("Matching draft picks to combine data...")

        if combine_df.empty:
            logger.warning("No combine data available")
            return draft_df

        name_col = "player_name" if "player_name" in combine_df.columns else "pfr_player_name"
        school_col = "college" if "college" in combine_df.columns else "school"

        matched_data = []
        matched_count = 0

        combine_cols = [
            "forty_yard", "forty", "bench_press", "bench", "vertical_jump", "vertical",
            "broad_jump", "three_cone", "cone", "shuttle", "height", "ht",
            "weight", "wt", "arm_length", "hand_size"
        ]

        for idx, draft_row in draft_df.iterrows():
            player_name = draft_row.get("player_name", "")
            college = draft_row.get("college", "")

            result = draft_row.to_dict()

            match = self._fuzzy_match(
                player_name, college, combine_df,
                name_col=name_col, school_col=school_col
            )

            if match is not None:
                for col in combine_cols:
                    if col in match.index and pd.notna(match[col]):
                        # Standardize column names
                        std_col = col.replace("forty", "forty_yard").replace("ht", "height").replace("wt", "weight").replace("cone", "three_cone")
                        result[f"combine_{std_col}"] = match[col]
                matched_count += 1

            matched_data.append(result)

        logger.info(f"Matched {matched_count}/{len(draft_df)} draft picks to combine")
        return pd.DataFrame(matched_data)

    def add_nfl_production(
        self,
        draft_df: pd.DataFrame,
        nfl_stats: pd.DataFrame,
        max_seasons: int = 3,
    ) -> pd.DataFrame:
        """
        Add NFL production stats for drafted players.

        Tracks first 1-3 seasons to measure draft hit rate.

        Args:
            draft_df: Draft DataFrame
            nfl_stats: NFL player stats DataFrame
            max_seasons: Max seasons to track

        Returns:
            Draft DataFrame with NFL production columns
        """
        logger.info("Adding NFL production data...")

        if nfl_stats.empty:
            logger.warning("No NFL stats available")
            return draft_df

        name_col = "player_name" if "player_name" in nfl_stats.columns else "player"

        matched_data = []
        matched_count = 0

        for idx, draft_row in draft_df.iterrows():
            player_name = draft_row.get("player_name", "")
            draft_year = draft_row.get("draft_year", 2024)

            result = draft_row.to_dict()

            # Find player's NFL stats
            player_nfl = nfl_stats[
                nfl_stats[name_col].apply(
                    lambda x: SequenceMatcher(None, self._normalize_name(str(x)),
                                              self._normalize_name(player_name)).ratio() > 0.85
                )
            ]

            if not player_nfl.empty:
                # Filter to first N seasons after draft
                if "season" in player_nfl.columns:
                    player_nfl = player_nfl[
                        (player_nfl["season"] >= draft_year) &
                        (player_nfl["season"] < draft_year + max_seasons)
                    ].sort_values("season")

                if not player_nfl.empty:
                    # Aggregate stats
                    result["nfl_seasons_played"] = len(player_nfl)
                    result["nfl_games_played"] = player_nfl.get("games", player_nfl.get("g", pd.Series([0]))).sum()
                    result["nfl_games_started"] = player_nfl.get("games_started", player_nfl.get("gs", pd.Series([0]))).sum()

                    # Position-specific stats
                    if "passing_yards" in player_nfl.columns:
                        result["nfl_pass_yards"] = player_nfl["passing_yards"].sum()
                        result["nfl_pass_tds"] = player_nfl.get("passing_tds", pd.Series([0])).sum()
                    if "rushing_yards" in player_nfl.columns:
                        result["nfl_rush_yards"] = player_nfl["rushing_yards"].sum()
                        result["nfl_rush_tds"] = player_nfl.get("rushing_tds", pd.Series([0])).sum()
                    if "receiving_yards" in player_nfl.columns:
                        result["nfl_rec_yards"] = player_nfl["receiving_yards"].sum()
                        result["nfl_rec_tds"] = player_nfl.get("receiving_tds", pd.Series([0])).sum()
                    if "sacks" in player_nfl.columns:
                        result["nfl_sacks"] = player_nfl["sacks"].sum()
                    if "interceptions" in player_nfl.columns:
                        result["nfl_interceptions"] = player_nfl["interceptions"].sum()

                    matched_count += 1

            matched_data.append(result)

        logger.info(f"Matched {matched_count}/{len(draft_df)} draft picks to NFL stats")
        return pd.DataFrame(matched_data)

    def add_contract_values(self, draft_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rookie contract values to draft DataFrame.

        Args:
            draft_df: Draft DataFrame

        Returns:
            Draft DataFrame with contract columns added
        """
        logger.info("Adding rookie contract values...")

        draft_df = draft_df.copy()

        contract_values = []
        signing_bonuses = []

        for idx, row in draft_df.iterrows():
            round_num = int(row.get("round", 7))
            pick = int(row.get("pick", 32))

            total, bonus = self.get_rookie_contract_value(round_num, pick)
            contract_values.append(total)
            signing_bonuses.append(bonus)

        draft_df["rookie_contract_4yr"] = contract_values
        draft_df["rookie_signing_bonus"] = signing_bonuses

        return draft_df

    def calculate_draft_hit_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate draft hit rate metrics.

        Determines if a pick was a "hit" based on NFL production.

        Args:
            df: Unified draft DataFrame

        Returns:
            DataFrame with hit rate columns
        """
        df = df.copy()

        def classify_hit(row):
            """Classify if a draft pick was a hit."""
            games = row.get("nfl_games_played", 0) or 0
            starts = row.get("nfl_games_started", 0) or 0
            round_num = row.get("round", 7)

            # Hit criteria varies by round
            if round_num <= 2:
                # Early picks should be starters
                if starts >= 32:  # Started 2+ seasons
                    return "hit"
                elif games >= 32:
                    return "contributor"
                else:
                    return "bust"
            elif round_num <= 4:
                # Mid-round picks should contribute
                if starts >= 16:
                    return "hit"
                elif games >= 24:
                    return "contributor"
                else:
                    return "miss"
            else:
                # Late round picks any contribution is value
                if starts >= 8:
                    return "hit"
                elif games >= 16:
                    return "contributor"
                else:
                    return "miss"

        df["draft_outcome"] = df.apply(classify_hit, axis=1)

        return df

    def build_draft_history_dataset(
        self,
        draft_start: int = 2018,
        draft_end: int = 2024,
        college_start: int = 2015,
        college_end: int = 2024,
        recruiting_start: int = 2012,
        recruiting_end: int = 2024,
    ) -> pd.DataFrame:
        """
        Build unified draft history dataset.

        Combines:
        - Draft picks
        - College stats (final 1-2 seasons)
        - Recruiting rankings
        - Combine measurements
        - Rookie contract values
        - NFL production (first 1-3 years)
        - Draft hit rate classification

        Args:
            draft_start: First draft year
            draft_end: Last draft year
            college_start: First college stats year
            college_end: Last college stats year
            recruiting_start: First recruiting year
            recruiting_end: Last recruiting year

        Returns:
            Unified DataFrame saved to data/processed/draft_history_dataset.csv
        """
        logger.info("=" * 60)
        logger.info(f"Building draft history dataset: {draft_start}-{draft_end}")
        logger.info("=" * 60)

        # Collect draft picks
        logger.info("\n[1/6] Collecting draft picks...")
        draft_df = self.collect_draft_picks(draft_start, draft_end)
        if draft_df.empty:
            logger.error("No draft data available")
            return pd.DataFrame()
        logger.info(f"  {len(draft_df)} draft picks")

        # Collect combine data
        logger.info("\n[2/6] Collecting combine data...")
        combine_df = self.collect_combine_data(draft_start, draft_end)
        logger.info(f"  {len(combine_df)} combine records")

        # Load college stats
        logger.info("\n[3/6] Loading college stats...")
        college_stats = self._load_college_stats(college_start, college_end)
        logger.info(f"  {len(college_stats)} college stat records")

        # Load recruiting data
        logger.info("\n[4/6] Loading recruiting data...")
        recruiting_df = self._load_recruiting_data(recruiting_start, recruiting_end)
        logger.info(f"  {len(recruiting_df)} recruiting records")

        # Collect NFL stats
        logger.info("\n[5/6] Collecting NFL stats...")
        nfl_stats = self.collect_nfl_player_stats(draft_start, min(draft_end + 3, 2025))
        logger.info(f"  {len(nfl_stats)} NFL stat records")

        # Build unified dataset
        logger.info("\n[6/6] Building unified dataset...")

        # Match college stats
        unified_df = self.match_draft_to_college_stats(draft_df, college_stats)

        # Match recruiting
        unified_df = self.match_draft_to_recruiting(unified_df, recruiting_df)

        # Match combine
        unified_df = self.match_draft_to_combine(unified_df, combine_df)

        # Add contract values
        unified_df = self.add_contract_values(unified_df)

        # Add NFL production
        unified_df = self.add_nfl_production(unified_df, nfl_stats)

        # Calculate hit rate
        unified_df = self.calculate_draft_hit_rate(unified_df)

        # Save to processed
        output_path = self.processed_dir / "draft_history_dataset.csv"
        unified_df.to_csv(output_path, index=False)
        logger.info(f"\nSaved unified dataset to {output_path}")
        logger.info(f"  {len(unified_df)} rows, {len(unified_df.columns)} columns")

        # Print summary
        self._print_summary(unified_df)

        return unified_df

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print dataset summary."""
        print("\n" + "=" * 60)
        print("DRAFT HISTORY DATASET SUMMARY")
        print("=" * 60)

        print(f"\nTotal draft picks: {len(df)}")

        if "draft_year" in df.columns:
            print("\nBy draft year:")
            for year in sorted(df["draft_year"].unique()):
                count = len(df[df["draft_year"] == year])
                print(f"  {year}: {count} picks")

        if "round" in df.columns:
            print("\nBy round:")
            for rnd in sorted(df["round"].unique()):
                count = len(df[df["round"] == rnd])
                print(f"  Round {rnd}: {count} picks")

        if "draft_outcome" in df.columns:
            print("\nDraft outcomes:")
            outcomes = df["draft_outcome"].value_counts()
            for outcome, count in outcomes.items():
                pct = count / len(df) * 100
                print(f"  {outcome}: {count} ({pct:.1f}%)")

        # Data completeness
        print("\nData completeness:")
        completeness_cols = [
            ("college_", "College stats"),
            ("recruiting_", "Recruiting data"),
            ("combine_", "Combine data"),
            ("nfl_", "NFL stats"),
        ]

        for prefix, label in completeness_cols:
            cols = [c for c in df.columns if c.startswith(prefix)]
            if cols:
                non_null = df[cols].notna().any(axis=1).sum()
                pct = non_null / len(df) * 100
                print(f"  {label}: {non_null}/{len(df)} ({pct:.1f}%)")


if __name__ == "__main__":
    print("NFL Draft History Collector")
    print("-" * 50)

    collector = DraftHistoryCollector()

    # Build full dataset
    dataset = collector.build_draft_history_dataset(
        draft_start=2018,
        draft_end=2024,
    )

    if not dataset.empty:
        # Show sample
        print("\n" + "=" * 60)
        print("SAMPLE DATA (Top 10 picks from 2024)")
        print("=" * 60)

        sample_cols = [
            "player_name", "college", "position", "round", "pick",
            "recruiting_stars", "combine_forty_yard", "rookie_contract_4yr",
            "nfl_games_played", "draft_outcome"
        ]
        available_cols = [c for c in sample_cols if c in dataset.columns]

        if "draft_year" in dataset.columns:
            sample = dataset[dataset["draft_year"] == 2024].head(10)[available_cols]
        else:
            sample = dataset.head(10)[available_cols]

        print(sample.to_string(index=False))

        # Show hit rate by round
        if "draft_outcome" in dataset.columns and "round" in dataset.columns:
            print("\n" + "=" * 60)
            print("HIT RATE BY ROUND")
            print("=" * 60)

            for rnd in range(1, 8):
                rnd_df = dataset[dataset["round"] == rnd]
                if not rnd_df.empty:
                    hits = len(rnd_df[rnd_df["draft_outcome"] == "hit"])
                    total = len(rnd_df)
                    print(f"  Round {rnd}: {hits}/{total} hits ({hits/total*100:.1f}%)")
