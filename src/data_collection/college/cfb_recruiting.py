"""
College Football Recruiting Data Collector

Collects recruiting rankings and builds recruit-to-player performance datasets.
Links recruiting data to college production for model training.

Data Sources:
- CFBD API for recruiting rankings (2018-2025)
- Links to player stats to analyze recruit performance
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
from dotenv import load_dotenv

try:
    import cfbd
    from cfbd.rest import ApiException
    CFBD_AVAILABLE = True
except ImportError:
    CFBD_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CFBRecruitingCollector:
    """
    Collects college football recruiting data and links to player performance.

    Features:
    - Individual player recruiting rankings (stars, rating, composite ranking)
    - Team recruiting class rankings
    - Fuzzy matching recruits to college players
    - Performance analysis by recruiting rating
    - Production over expectation calculations
    """

    RATE_LIMIT_SECONDS = 0.5
    CACHE_HOURS = 24
    FUZZY_MATCH_THRESHOLD = 0.85

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the recruiting data collector.

        Args:
            data_dir: Base directory for data storage
        """
        load_dotenv()

        if not CFBD_AVAILABLE:
            logger.error("cfbd package not installed. Run: pip install cfbd")
            raise ImportError("cfbd package required but not installed")

        # Configure CFBD API
        self.api_key = os.getenv("CFBD_API_KEY")
        if not self.api_key:
            logger.warning("CFBD_API_KEY not found in environment. API calls will fail.")

        self.configuration = cfbd.Configuration()
        self.configuration.api_key["Authorization"] = self.api_key
        self.configuration.api_key_prefix["Authorization"] = "Bearer"

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
        self.cache_dir = self.data_dir / "cache"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for player stats (loaded on demand)
        self._player_stats_cache: Optional[pd.DataFrame] = None

        logger.info(f"CFBRecruitingCollector initialized. Data dir: {self.data_dir}")

    def _rate_limit(self) -> None:
        """Apply rate limiting between API calls."""
        time.sleep(self.RATE_LIMIT_SECONDS)

    def _get_cache_path(self, name: str, start_year: int, end_year: int) -> Path:
        """Get cache file path for a dataset."""
        return self.cache_dir / f"cfb_{name}_{start_year}_{end_year}_cache.csv"

    def _get_cache_meta_path(self, name: str, start_year: int, end_year: int) -> Path:
        """Get cache metadata file path."""
        return self.cache_dir / f"cfb_{name}_{start_year}_{end_year}_cache_meta.txt"

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
            else:
                logger.info(f"Cache expired for {name} ({age.total_seconds() / 3600:.1f} hours old)")
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

    def collect_player_recruiting_rankings(
        self,
        start_year: int = 2018,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Collect individual player recruiting rankings.

        Goes back to 2018 to capture recruits who played through 2025.

        Args:
            start_year: First recruiting class year (default: 2018)
            end_year: Last recruiting class year (default: 2025)

        Returns:
            DataFrame with player recruiting data:
            - name, school committed to, position
            - stars (1-5), rating (0-1 scale), composite ranking
            - state, city, height, weight
        """
        # Check cache first
        cached = self._load_cache("recruiting_players", start_year, end_year)
        if cached is not None:
            return cached

        logger.info(f"Collecting player recruiting rankings for {start_year}-{end_year}")

        all_recruits = []
        recruiting_api = cfbd.RecruitingApi(cfbd.ApiClient(self.configuration))

        try:
            for year in range(start_year, end_year + 1):
                logger.info(f"Fetching recruits for {year}...")

                try:
                    self._rate_limit()
                    recruits = recruiting_api.get_recruiting_players(year=year)

                    for recruit in recruits:
                        recruit_dict = {
                            "recruit_id": getattr(recruit, "id", None),
                            "recruiting_year": year,
                            "name": getattr(recruit, "name", None),
                            "school": getattr(recruit, "committed_to", None),
                            "position": getattr(recruit, "position", None),
                            "stars": getattr(recruit, "stars", None),
                            "rating": getattr(recruit, "rating", None),
                            "national_ranking": getattr(recruit, "ranking", None),
                            "state_ranking": getattr(recruit, "state_rank", None),
                            "position_ranking": getattr(recruit, "position_rank", None),
                            "state": getattr(recruit, "state_province", None),
                            "city": getattr(recruit, "city", None),
                            "height": getattr(recruit, "height", None),
                            "weight": getattr(recruit, "weight", None),
                        }
                        all_recruits.append(recruit_dict)

                    logger.info(f"  {year}: {len(recruits)} recruits")

                except ApiException as e:
                    logger.error(f"API error for recruiting {year}: {e}")
                except Exception as e:
                    logger.error(f"Error fetching recruiting {year}: {e}")

        except Exception as e:
            logger.error(f"Unexpected error collecting recruiting data: {e}")

        if not all_recruits:
            logger.warning("No recruiting data collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_recruits)

        # Save to cache
        self._save_cache(df, "recruiting_players", start_year, end_year)

        # Save to raw
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = self.raw_dir / f"recruiting_players_{start_year}_{end_year}_{timestamp}.csv"
        df.to_csv(raw_path, index=False)

        logger.info(f"Collected {len(df)} total recruit records")
        return df

    def collect_team_recruiting_rankings(
        self,
        start_year: int = 2018,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Collect team recruiting class rankings per year.

        Args:
            start_year: First recruiting class year
            end_year: Last recruiting class year

        Returns:
            DataFrame with team recruiting data:
            - school, year, overall rank
            - total commits, average rating, points
        """
        # Check cache first
        cached = self._load_cache("recruiting_teams", start_year, end_year)
        if cached is not None:
            return cached

        logger.info(f"Collecting team recruiting rankings for {start_year}-{end_year}")

        all_teams = []
        recruiting_api = cfbd.RecruitingApi(cfbd.ApiClient(self.configuration))

        try:
            for year in range(start_year, end_year + 1):
                logger.info(f"Fetching team recruiting for {year}...")

                try:
                    self._rate_limit()
                    teams = recruiting_api.get_recruiting_teams(year=year)

                    for team in teams:
                        team_dict = {
                            "school": getattr(team, "team", None),
                            "recruiting_year": year,
                            "overall_rank": getattr(team, "rank", None),
                            "total_commits": getattr(team, "commits", None),
                            "five_stars": getattr(team, "five_stars", 0) or 0,
                            "four_stars": getattr(team, "four_stars", 0) or 0,
                            "three_stars": getattr(team, "three_stars", 0) or 0,
                            "average_rating": getattr(team, "average_rating", None),
                            "points": getattr(team, "points", None),
                        }
                        all_teams.append(team_dict)

                    logger.info(f"  {year}: {len(teams)} teams")

                except ApiException as e:
                    logger.error(f"API error for team recruiting {year}: {e}")
                except Exception as e:
                    logger.error(f"Error fetching team recruiting {year}: {e}")

        except Exception as e:
            logger.error(f"Unexpected error collecting team recruiting: {e}")

        if not all_teams:
            logger.warning("No team recruiting data collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_teams)

        # Save to cache
        self._save_cache(df, "recruiting_teams", start_year, end_year)

        # Save to raw
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = self.raw_dir / f"recruiting_teams_{start_year}_{end_year}_{timestamp}.csv"
        df.to_csv(raw_path, index=False)

        logger.info(f"Collected {len(df)} total team recruiting records")
        return df

    def match_recruit_to_player(
        self,
        recruit_name: str,
        school: str,
        position: str,
        player_stats: pd.DataFrame,
        threshold: float = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fuzzy match a recruit to their college player stats record.

        Uses difflib.SequenceMatcher with a threshold of 0.85 by default.

        Args:
            recruit_name: Recruit's name
            school: School they committed to
            position: Their position
            player_stats: DataFrame of college player stats
            threshold: Match threshold (default: 0.85)

        Returns:
            Matched player record dict or None if no match found
        """
        if threshold is None:
            threshold = self.FUZZY_MATCH_THRESHOLD

        if player_stats.empty:
            return None

        # Normalize recruit name
        recruit_name_norm = self._normalize_name(recruit_name)

        # Filter to same school first (most restrictive)
        school_matches = player_stats[
            player_stats["team"].str.lower() == school.lower()
        ] if "team" in player_stats.columns else player_stats

        if school_matches.empty:
            return None

        # Find best name match
        best_match = None
        best_score = 0.0

        name_col = "player_name" if "player_name" in school_matches.columns else "player"

        for idx, row in school_matches.iterrows():
            player_name = row.get(name_col, "")
            if not player_name:
                continue

            player_name_norm = self._normalize_name(str(player_name))
            score = SequenceMatcher(None, recruit_name_norm, player_name_norm).ratio()

            # Bonus for position match
            if position and "position" in row:
                if self._positions_match(position, str(row.get("position", ""))):
                    score += 0.05

            if score > best_score:
                best_score = score
                best_match = row

        if best_score >= threshold and best_match is not None:
            logger.debug(f"Matched '{recruit_name}' to '{best_match.get(name_col)}' (score: {best_score:.3f})")
            return best_match.to_dict()

        return None

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for matching."""
        if not name:
            return ""
        # Lowercase, remove punctuation, extra spaces
        name = name.lower().strip()
        # Remove common suffixes
        for suffix in [" jr", " jr.", " sr", " sr.", " ii", " iii", " iv"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        # Remove punctuation
        name = "".join(c for c in name if c.isalnum() or c.isspace())
        return " ".join(name.split())

    def _positions_match(self, pos1: str, pos2: str) -> bool:
        """Check if two positions are equivalent or similar."""
        pos1 = pos1.upper() if pos1 else ""
        pos2 = pos2.upper() if pos2 else ""

        # Direct match
        if pos1 == pos2:
            return True

        # Position groups
        groups = {
            "QB": ["QB"],
            "RB": ["RB", "HB", "FB", "APB"],
            "WR": ["WR", "SLOT"],
            "TE": ["TE"],
            "OL": ["OT", "OG", "C", "OL", "T", "G", "IOL"],
            "DL": ["DT", "DE", "DL", "NT", "SDE", "WDE"],
            "LB": ["LB", "ILB", "OLB", "MLB"],
            "DB": ["CB", "S", "DB", "FS", "SS", "ATH"],
            "EDGE": ["EDGE", "DE", "OLB", "RUSH"],
        }

        for group_positions in groups.values():
            if pos1 in group_positions and pos2 in group_positions:
                return True

        return False

    def load_player_stats(
        self,
        start_year: int = 2018,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """Load player stats for matching."""
        if self._player_stats_cache is not None:
            return self._player_stats_cache

        # Try to load from cache
        cache_path = self.cache_dir / f"cfb_player_stats_{start_year}_{end_year}_cache.csv"
        if cache_path.exists():
            try:
                self._player_stats_cache = pd.read_csv(cache_path)
                logger.info(f"Loaded {len(self._player_stats_cache)} player stats from cache")
                return self._player_stats_cache
            except Exception as e:
                logger.warning(f"Error loading player stats cache: {e}")

        # If not available, collect fresh (requires CFBStatsCollector)
        try:
            from .cfb_stats import CFBStatsCollector
            collector = CFBStatsCollector(str(self.data_dir))
            self._player_stats_cache = collector.collect_player_stats(start_year, end_year)
            return self._player_stats_cache
        except Exception as e:
            logger.warning(f"Could not load player stats: {e}")
            return pd.DataFrame()

    def build_recruiting_performance_dataset(
        self,
        recruiting_start: int = 2018,
        recruiting_end: int = 2022,
        stats_start: int = 2018,
        stats_end: int = 2025,
    ) -> pd.DataFrame:
        """
        Build a dataset linking recruits to their college performance.

        Creates a merged dataset with recruiting info + college production,
        enabling analysis of whether 5-stars outperform 3-stars.
        Calculates "production over expectation" (actual stats vs average for star rating).

        Args:
            recruiting_start: First recruiting class to include
            recruiting_end: Last recruiting class to include (allow time for college career)
            stats_start: First season to search for stats
            stats_end: Last season to search for stats

        Returns:
            DataFrame with:
            - Recruiting info (stars, rating, ranking)
            - College stats (passing/rushing/receiving yards, TDs, etc.)
            - Production over expectation metrics
        """
        logger.info("Building recruiting performance dataset...")

        # Load recruiting data
        recruits = self.collect_player_recruiting_rankings(recruiting_start, recruiting_end)
        if recruits.empty:
            logger.error("No recruiting data available")
            return pd.DataFrame()

        # Load player stats
        player_stats = self.load_player_stats(stats_start, stats_end)
        if player_stats.empty:
            logger.warning("No player stats available for matching")
            return recruits

        # Match recruits to players
        matched_data = []
        matched_count = 0
        unmatched_count = 0

        logger.info(f"Matching {len(recruits)} recruits to player stats...")

        for idx, recruit in recruits.iterrows():
            recruit_dict = recruit.to_dict()

            match = self.match_recruit_to_player(
                recruit_name=recruit["name"],
                school=recruit.get("school", ""),
                position=recruit.get("position", ""),
                player_stats=player_stats,
            )

            if match:
                # Merge recruiting info with stats
                recruit_dict.update({
                    f"stats_{k}": v for k, v in match.items()
                    if k not in ["name", "player_name", "player"]
                })
                recruit_dict["matched"] = True
                matched_count += 1
            else:
                recruit_dict["matched"] = False
                unmatched_count += 1

            matched_data.append(recruit_dict)

            # Progress logging
            if (idx + 1) % 500 == 0:
                logger.info(f"  Processed {idx + 1}/{len(recruits)} recruits...")

        logger.info(f"Matched: {matched_count}, Unmatched: {unmatched_count}")

        df = pd.DataFrame(matched_data)

        # Calculate production over expectation
        df = self._calculate_production_over_expectation(df)

        # Save to raw
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = self.raw_dir / f"recruiting_performance_{recruiting_start}_{recruiting_end}_{timestamp}.csv"
        df.to_csv(raw_path, index=False)
        logger.info(f"Saved recruiting performance dataset to {raw_path}")

        return df

    def _calculate_production_over_expectation(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate production over expectation based on star rating.

        Compares actual stats vs average for players with same star rating.

        Args:
            df: DataFrame with recruiting and stats data

        Returns:
            DataFrame with production over expectation columns
        """
        if df.empty or "stars" not in df.columns:
            return df

        df = df.copy()

        # Define stat columns to analyze
        stat_cols = [
            col for col in df.columns
            if col.startswith("stats_") and any(
                metric in col.lower()
                for metric in ["yard", "yds", "td", "reception", "completion", "attempt", "rec", "rush", "pass"]
            )
        ]

        if not stat_cols:
            logger.warning("No stat columns found for production analysis")
            return df

        # Calculate averages by star rating for matched players only
        matched_df = df[df["matched"] == True]
        if matched_df.empty:
            return df

        star_groups = matched_df.groupby("stars")

        for col in stat_cols:
            if col in df.columns:
                try:
                    # Calculate mean for each star group
                    star_means = matched_df.groupby("stars")[col].mean()

                    # Map means back to all rows
                    df[f"expected_{col}"] = df["stars"].map(star_means)

                    # Calculate production over expectation
                    poe_col = col.replace("stats_", "poe_")
                    df[poe_col] = df[col] - df[f"expected_{col}"]
                except Exception as e:
                    logger.debug(f"Could not calculate POE for {col}: {e}")

        # Overall production score (normalized)
        try:
            df["production_score"] = 0.0

            # Add contributions from different stat types
            for col in stat_cols:
                if "passing" in col.lower() and "yds" in col.lower():
                    df["production_score"] += df[col].fillna(0) / 1000
                elif "rushing" in col.lower() and "yds" in col.lower():
                    df["production_score"] += df[col].fillna(0) / 500
                elif "receiving" in col.lower() and "yds" in col.lower():
                    df["production_score"] += df[col].fillna(0) / 500
                elif "td" in col.lower():
                    df["production_score"] += df[col].fillna(0) / 5

            # Calculate POE for production score
            star_prod_means = matched_df.groupby("stars")["production_score"].mean()
            df["expected_production"] = df["stars"].map(star_prod_means)
            df["production_over_expectation"] = df["production_score"] - df["expected_production"]

        except Exception as e:
            logger.warning(f"Error calculating production score: {e}")

        return df

    def get_star_rating_analysis(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Analyze performance by star rating.

        Args:
            df: Recruiting performance DataFrame (or load if None)

        Returns:
            Summary DataFrame by star rating
        """
        if df is None:
            df = self.build_recruiting_performance_dataset()

        if df.empty:
            return pd.DataFrame()

        # Group by stars and calculate metrics
        summary = df.groupby("stars").agg({
            "name": "count",
            "matched": "sum",
            "rating": "mean",
        }).rename(columns={
            "name": "total_recruits",
            "matched": "matched_to_stats",
            "rating": "avg_rating",
        })

        # Add production metrics if available
        if "production_score" in df.columns:
            matched_only = df[df["matched"] == True]
            prod_summary = matched_only.groupby("stars")["production_score"].agg(["mean", "median", "std"])
            prod_summary.columns = ["avg_production", "median_production", "std_production"]
            summary = summary.join(prod_summary)

        if "production_over_expectation" in df.columns:
            matched_only = df[df["matched"] == True]
            poe_summary = matched_only.groupby("stars")["production_over_expectation"].mean()
            summary["avg_poe"] = poe_summary

        summary["match_rate"] = summary["matched_to_stats"] / summary["total_recruits"]

        return summary.reset_index()

    def get_top_performers_by_star(
        self,
        df: pd.DataFrame,
        stars: int,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Get top performing players for a given star rating."""
        if df.empty or "stars" not in df.columns:
            return pd.DataFrame()

        star_df = df[(df["stars"] == stars) & (df["matched"] == True)]
        if star_df.empty:
            return pd.DataFrame()

        if "production_score" in star_df.columns:
            return star_df.nlargest(top_n, "production_score")

        return star_df.head(top_n)

    def get_biggest_outperformers(
        self,
        df: pd.DataFrame,
        top_n: int = 25,
    ) -> pd.DataFrame:
        """Get players who most exceeded expectations for their star rating."""
        if df.empty or "production_over_expectation" not in df.columns:
            return pd.DataFrame()

        matched = df[df["matched"] == True]
        return matched.nlargest(top_n, "production_over_expectation")

    def get_biggest_underperformers(
        self,
        df: pd.DataFrame,
        top_n: int = 25,
    ) -> pd.DataFrame:
        """Get players who most underperformed relative to their star rating."""
        if df.empty or "production_over_expectation" not in df.columns:
            return pd.DataFrame()

        matched = df[df["matched"] == True]
        return matched.nsmallest(top_n, "production_over_expectation")


if __name__ == "__main__":
    print("College Football Recruiting Collector")
    print("-" * 50)

    collector = CFBRecruitingCollector()

    # Collect recruiting data
    print("\n[1/3] Collecting player recruiting rankings...")
    players = collector.collect_player_recruiting_rankings(2018, 2024)
    print(f"  Collected {len(players)} recruits")

    print("\n[2/3] Collecting team recruiting rankings...")
    teams = collector.collect_team_recruiting_rankings(2018, 2024)
    print(f"  Collected {len(teams)} team-years")

    print("\n[3/3] Building recruiting performance dataset...")
    performance = collector.build_recruiting_performance_dataset(
        recruiting_start=2018,
        recruiting_end=2022,
        stats_start=2018,
        stats_end=2024,
    )
    print(f"  Built dataset with {len(performance)} records")

    # Print star rating analysis
    print("\n" + "=" * 60)
    print("STAR RATING ANALYSIS")
    print("=" * 60)

    analysis = collector.get_star_rating_analysis(performance)
    if not analysis.empty:
        print(analysis.to_string(index=False))
    else:
        print("No analysis available")

    # Print sample data
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)

    if not players.empty:
        print("\nTop 10 Recruits (by rating):")
        top_recruits = players.nlargest(10, "rating")[["name", "school", "position", "stars", "rating", "national_ranking"]]
        print(top_recruits.to_string(index=False))

    if not performance.empty and "production_over_expectation" in performance.columns:
        print("\nBiggest Outperformers (vs star rating expectation):")
        outperformers = collector.get_biggest_outperformers(performance, 5)
        if not outperformers.empty:
            cols = ["name", "school", "stars", "production_score", "production_over_expectation"]
            cols = [c for c in cols if c in outperformers.columns]
            print(outperformers[cols].to_string(index=False))
