"""
College Football Stats Collector

Collects college football data using the CFBD (College Football Data) API.
Includes player stats, team data, game results, and recruiting information.
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

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


class CFBStatsCollector:
    """
    Collects college football statistics from the CFBD API.

    Features:
    - Automatic caching with 24-hour expiration
    - Rate limiting to respect API limits
    - Comprehensive error handling
    - Support for multi-year data collection
    """

    RATE_LIMIT_SECONDS = 0.5
    CACHE_HOURS = 24

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the CFB stats collector.

        Args:
            data_dir: Base directory for data storage. If None, attempts to find
                     project root by looking for config.yaml
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

        logger.info(f"CFBStatsCollector initialized. Data dir: {self.data_dir}")

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

    def collect_player_stats(
        self,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """
        Collect player season statistics for passing, rushing, and receiving.

        Args:
            start_year: First season to collect
            end_year: Last season to collect (inclusive)

        Returns:
            DataFrame with player stats including:
            - player name, team, conference, season, games played
            - passing: attempts, completions, yards, TDs, INTs, rating
            - rushing: attempts, yards, TDs, yards per carry
            - receiving: receptions, yards, TDs, yards per reception
        """
        # Check cache first
        cached = self._load_cache("player_stats", start_year, end_year)
        if cached is not None:
            return cached

        logger.info(f"Collecting player stats for {start_year}-{end_year}")

        all_stats = []
        players_api = cfbd.PlayersApi(cfbd.ApiClient(self.configuration))
        stat_categories = ["passing", "rushing", "receiving"]

        try:
            for year in range(start_year, end_year + 1):
                logger.info(f"Fetching player stats for {year}...")

                for category in stat_categories:
                    try:
                        self._rate_limit()
                        stats = players_api.get_player_season_stats(
                            year=year,
                            category=category,
                        )

                        for player_stat in stats:
                            stat_dict = {
                                "player_id": getattr(player_stat, "player_id", None),
                                "player_name": getattr(player_stat, "player", None),
                                "team": getattr(player_stat, "team", None),
                                "conference": getattr(player_stat, "conference", None),
                                "season": year,
                                "category": category,
                            }

                            # Parse stat type and value
                            if hasattr(player_stat, "stat_type") and hasattr(player_stat, "stat"):
                                stat_dict["stat_type"] = player_stat.stat_type
                                stat_dict["stat_value"] = player_stat.stat

                            all_stats.append(stat_dict)

                        logger.info(f"  {year} {category}: {len(stats)} entries")

                    except ApiException as e:
                        logger.error(f"API error for {year} {category}: {e}")
                    except Exception as e:
                        logger.error(f"Error fetching {year} {category}: {e}")

        except Exception as e:
            logger.error(f"Unexpected error collecting player stats: {e}")

        if not all_stats:
            logger.warning("No player stats collected")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_stats)

        # Pivot to get one row per player with stats as columns
        if "stat_type" in df.columns and not df.empty:
            try:
                # Create a unique stat column name with category prefix
                df["stat_column"] = df["category"] + "_" + df["stat_type"].astype(str)

                # Pivot to wide format
                pivot_df = df.pivot_table(
                    index=["player_id", "player_name", "team", "conference", "season"],
                    columns="stat_column",
                    values="stat_value",
                    aggfunc="first"
                ).reset_index()

                df = pivot_df
                logger.info(f"Pivoted stats to {len(df.columns)} columns")
            except Exception as e:
                logger.warning(f"Could not pivot stats, keeping long format: {e}")

        # Save to cache
        self._save_cache(df, "player_stats", start_year, end_year)

        logger.info(f"Collected {len(df)} total player-season records")
        return df

    def collect_player_info(
        self,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """
        Collect player roster information for all teams.

        Args:
            start_year: First season to collect
            end_year: Last season to collect (inclusive)

        Returns:
            DataFrame with player info including:
            - player name, team, position, height, weight
            - year (FR/SO/JR/SR), hometown, jersey number
        """
        # Check cache first
        cached = self._load_cache("player_info", start_year, end_year)
        if cached is not None:
            return cached

        logger.info(f"Collecting player info for {start_year}-{end_year}")

        all_players = []
        players_api = cfbd.PlayersApi(cfbd.ApiClient(self.configuration))
        teams_api = cfbd.TeamsApi(cfbd.ApiClient(self.configuration))

        try:
            for year in range(start_year, end_year + 1):
                logger.info(f"Fetching rosters for {year}...")

                # Get list of FBS teams
                try:
                    self._rate_limit()
                    teams = teams_api.get_fbs_teams(year=year)
                    team_names = [t.school for t in teams]
                    logger.info(f"  Found {len(team_names)} FBS teams")
                except ApiException as e:
                    logger.error(f"Error getting teams for {year}: {e}")
                    continue

                year_count = 0
                for team in team_names:
                    try:
                        self._rate_limit()
                        roster = players_api.get_roster(year=year, team=team)

                        for player in roster:
                            # Format name
                            first = getattr(player, "first_name", "") or ""
                            last = getattr(player, "last_name", "") or ""
                            full_name = f"{first} {last}".strip()

                            # Format hometown
                            city = getattr(player, "home_city", "") or ""
                            state = getattr(player, "home_state", "") or ""
                            hometown = f"{city}, {state}".strip(", ") if city or state else ""

                            player_dict = {
                                "player_id": getattr(player, "id", None),
                                "player_name": full_name,
                                "team": team,
                                "season": year,
                                "position": getattr(player, "position", None),
                                "height": getattr(player, "height", None),
                                "weight": getattr(player, "weight", None),
                                "year": getattr(player, "year", None),
                                "hometown": hometown,
                                "home_city": city,
                                "home_state": state,
                                "jersey_number": getattr(player, "jersey", None),
                            }
                            all_players.append(player_dict)
                            year_count += 1

                    except ApiException as e:
                        if e.status != 404:
                            logger.debug(f"Error fetching roster for {team} {year}: {e}")
                    except Exception as e:
                        logger.debug(f"Error processing roster for {team} {year}: {e}")

                logger.info(f"  Collected {year_count} players for {year}")

        except Exception as e:
            logger.error(f"Unexpected error collecting player info: {e}")

        if not all_players:
            logger.warning("No player info collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_players)

        # Save to cache
        self._save_cache(df, "player_info", start_year, end_year)

        logger.info(f"Collected {len(df)} total player records")
        return df

    def collect_team_data(
        self,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """
        Collect team records, ratings, and recruiting rankings.

        Args:
            start_year: First season to collect
            end_year: Last season to collect (inclusive)

        Returns:
            DataFrame with team data including:
            - team name, conference, season
            - wins, losses, conference wins/losses
            - SP+ ratings (overall, offense, defense)
            - talent composite rankings
            - recruiting class rankings
        """
        # Check cache first
        cached = self._load_cache("team_data", start_year, end_year)
        if cached is not None:
            return cached

        logger.info(f"Collecting team data for {start_year}-{end_year}")

        all_teams = []
        games_api = cfbd.GamesApi(cfbd.ApiClient(self.configuration))
        ratings_api = cfbd.RatingsApi(cfbd.ApiClient(self.configuration))
        recruiting_api = cfbd.RecruitingApi(cfbd.ApiClient(self.configuration))
        teams_api = cfbd.TeamsApi(cfbd.ApiClient(self.configuration))

        try:
            for year in range(start_year, end_year + 1):
                logger.info(f"Fetching team data for {year}...")
                team_records = {}

                # Get team records (wins/losses)
                try:
                    self._rate_limit()
                    records = games_api.get_team_records(year=year)
                    for record in records:
                        team_name = getattr(record, "team", None)
                        if team_name:
                            total = getattr(record, "total", None)
                            conf = getattr(record, "conference_games", None)
                            team_records[team_name] = {
                                "team": team_name,
                                "conference": getattr(record, "conference", None),
                                "season": year,
                                "total_wins": getattr(total, "wins", 0) if total else 0,
                                "total_losses": getattr(total, "losses", 0) if total else 0,
                                "conf_wins": getattr(conf, "wins", 0) if conf else 0,
                                "conf_losses": getattr(conf, "losses", 0) if conf else 0,
                            }
                    logger.info(f"  Records: {len(records)} teams")
                except ApiException as e:
                    logger.error(f"Error fetching team records for {year}: {e}")

                # Get SP+ ratings
                try:
                    self._rate_limit()
                    sp_ratings = ratings_api.get_sp_ratings(year=year)
                    for rating in sp_ratings:
                        team_name = getattr(rating, "team", None)
                        if team_name:
                            if team_name not in team_records:
                                team_records[team_name] = {"team": team_name, "season": year}
                            team_records[team_name]["sp_overall"] = getattr(rating, "rating", None)
                            offense = getattr(rating, "offense", None)
                            defense = getattr(rating, "defense", None)
                            team_records[team_name]["sp_offense"] = getattr(offense, "rating", None) if offense else None
                            team_records[team_name]["sp_defense"] = getattr(defense, "rating", None) if defense else None
                    logger.info(f"  SP+ ratings: {len(sp_ratings)} teams")
                except ApiException as e:
                    logger.debug(f"SP+ ratings not available for {year}: {e}")

                # Get talent composite
                try:
                    self._rate_limit()
                    talent = teams_api.get_talent(year=year)
                    for t in talent:
                        team_name = getattr(t, "school", None)
                        if team_name:
                            if team_name not in team_records:
                                team_records[team_name] = {"team": team_name, "season": year}
                            team_records[team_name]["talent_composite"] = getattr(t, "talent", None)
                    logger.info(f"  Talent composite: {len(talent)} teams")
                except ApiException as e:
                    logger.debug(f"Talent composite not available for {year}: {e}")

                # Get recruiting rankings
                try:
                    self._rate_limit()
                    recruiting = recruiting_api.get_recruiting_teams(year=year)
                    for r in recruiting:
                        team_name = getattr(r, "team", None)
                        if team_name:
                            if team_name not in team_records:
                                team_records[team_name] = {"team": team_name, "season": year}
                            team_records[team_name]["recruiting_rank"] = getattr(r, "rank", None)
                            team_records[team_name]["recruiting_points"] = getattr(r, "points", None)
                    logger.info(f"  Recruiting: {len(recruiting)} teams")
                except ApiException as e:
                    logger.debug(f"Recruiting data not available for {year}: {e}")

                all_teams.extend(team_records.values())

        except Exception as e:
            logger.error(f"Unexpected error collecting team data: {e}")

        if not all_teams:
            logger.warning("No team data collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_teams)

        # Save to cache
        self._save_cache(df, "team_data", start_year, end_year)

        logger.info(f"Collected {len(df)} total team-season records")
        return df

    def collect_game_results(
        self,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """
        Collect game-by-game results for all teams.

        Args:
            start_year: First season to collect
            end_year: Last season to collect (inclusive)

        Returns:
            DataFrame with game results including:
            - season, week, home team, away team
            - home score, away score, winner
            - conference game flag, neutral site flag
            - venue information
        """
        # Check cache first
        cached = self._load_cache("game_results", start_year, end_year)
        if cached is not None:
            return cached

        logger.info(f"Collecting game results for {start_year}-{end_year}")

        all_games = []
        games_api = cfbd.GamesApi(cfbd.ApiClient(self.configuration))

        try:
            for year in range(start_year, end_year + 1):
                logger.info(f"Fetching games for {year}...")

                for season_type in ["regular", "postseason"]:
                    try:
                        self._rate_limit()
                        games = games_api.get_games(
                            year=year,
                            season_type=season_type,
                            division="fbs"
                        )

                        for game in games:
                            home_pts = getattr(game, "home_points", None)
                            away_pts = getattr(game, "away_points", None)

                            # Determine winner
                            winner = None
                            point_diff = None
                            if home_pts is not None and away_pts is not None:
                                if home_pts > away_pts:
                                    winner = getattr(game, "home_team", None)
                                elif away_pts > home_pts:
                                    winner = getattr(game, "away_team", None)
                                else:
                                    winner = "TIE"
                                point_diff = abs(home_pts - away_pts)

                            game_dict = {
                                "game_id": getattr(game, "id", None),
                                "season": year,
                                "week": getattr(game, "week", None),
                                "season_type": season_type,
                                "start_date": getattr(game, "start_date", None),
                                "neutral_site": getattr(game, "neutral_site", False),
                                "conference_game": getattr(game, "conference_game", False),
                                "home_team": getattr(game, "home_team", None),
                                "home_conference": getattr(game, "home_conference", None),
                                "home_points": home_pts,
                                "away_team": getattr(game, "away_team", None),
                                "away_conference": getattr(game, "away_conference", None),
                                "away_points": away_pts,
                                "winner": winner,
                                "point_differential": point_diff,
                                "venue": getattr(game, "venue", None),
                            }

                            all_games.append(game_dict)

                        logger.info(f"  {year} {season_type}: {len(games)} games")

                    except ApiException as e:
                        logger.error(f"Error fetching games for {year} {season_type}: {e}")

        except Exception as e:
            logger.error(f"Unexpected error collecting game results: {e}")

        if not all_games:
            logger.warning("No game results collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_games)

        # Save to cache
        self._save_cache(df, "game_results", start_year, end_year)

        logger.info(f"Collected {len(df)} total game records")
        return df

    def collect_all(
        self,
        start_year: int = 2020,
        end_year: int = 2025,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all collection methods and save results.

        Args:
            start_year: First season to collect (default: 2020)
            end_year: Last season to collect (default: 2025)

        Returns:
            Dictionary with all collected DataFrames:
            - player_stats
            - player_info
            - team_data
            - game_results
        """
        logger.info("=" * 60)
        logger.info(f"Starting full data collection: {start_year}-{end_year}")
        logger.info("=" * 60)

        results = {}
        warnings = []
        start_time = datetime.now()

        # Collect player stats
        logger.info("\n[1/4] Collecting player stats...")
        try:
            results["player_stats"] = self.collect_player_stats(start_year, end_year)
            if results["player_stats"].empty:
                warnings.append("player_stats: No data collected")
        except Exception as e:
            logger.error(f"Failed to collect player stats: {e}")
            results["player_stats"] = pd.DataFrame()
            warnings.append(f"player_stats: Collection failed - {e}")

        # Collect player info
        logger.info("\n[2/4] Collecting player info...")
        try:
            results["player_info"] = self.collect_player_info(start_year, end_year)
            if results["player_info"].empty:
                warnings.append("player_info: No data collected")
        except Exception as e:
            logger.error(f"Failed to collect player info: {e}")
            results["player_info"] = pd.DataFrame()
            warnings.append(f"player_info: Collection failed - {e}")

        # Collect team data
        logger.info("\n[3/4] Collecting team data...")
        try:
            results["team_data"] = self.collect_team_data(start_year, end_year)
            if results["team_data"].empty:
                warnings.append("team_data: No data collected")
        except Exception as e:
            logger.error(f"Failed to collect team data: {e}")
            results["team_data"] = pd.DataFrame()
            warnings.append(f"team_data: Collection failed - {e}")

        # Collect game results
        logger.info("\n[4/4] Collecting game results...")
        try:
            results["game_results"] = self.collect_game_results(start_year, end_year)
            if results["game_results"].empty:
                warnings.append("game_results: No data collected")
        except Exception as e:
            logger.error(f"Failed to collect game results: {e}")
            results["game_results"] = pd.DataFrame()
            warnings.append(f"game_results: Collection failed - {e}")

        # Save to raw directory with descriptive filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, df in results.items():
            if not df.empty:
                raw_path = self.raw_dir / f"cfb_{name}_{start_year}_{end_year}_{timestamp}.csv"
                df.to_csv(raw_path, index=False)
                logger.info(f"Saved {name} to {raw_path}")

        # Print summary
        elapsed = datetime.now() - start_time
        self._print_summary(results, start_year, end_year, warnings, elapsed)

        return results

    def _print_summary(
        self,
        results: Dict[str, pd.DataFrame],
        start_year: int,
        end_year: int,
        warnings: List[str],
        elapsed: timedelta,
    ) -> None:
        """Print collection summary."""
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Date Range: {start_year} - {end_year}")
        print(f"Collection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed: {elapsed.total_seconds():.1f} seconds")
        print("-" * 60)
        print("Rows Collected:")

        total_rows = 0
        for name, df in results.items():
            rows = len(df)
            total_rows += rows
            cols = len(df.columns) if not df.empty else 0
            print(f"  {name:20s}: {rows:>8,} rows, {cols:>3} columns")

        print("-" * 60)
        print(f"  {'TOTAL':20s}: {total_rows:>8,} rows")

        if warnings:
            print("-" * 60)
            print("WARNINGS:")
            for warning in warnings:
                print(f"  ⚠ {warning}")

        print("=" * 60)

        # Log locations
        print(f"\nData saved to:")
        print(f"  Cache: {self.cache_dir}")
        print(f"  Raw:   {self.raw_dir}")


if __name__ == "__main__":
    # Run standalone collection
    print("College Football Data Collector")
    print("-" * 40)

    collector = CFBStatsCollector()

    # Collect all data for recent years
    data = collector.collect_all(start_year=2020, end_year=2024)

    # Show sample of each dataset
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)

    for name, df in data.items():
        if not df.empty:
            print(f"\n{name.upper()} (first 3 rows):")
            print(df.head(3).to_string())
        else:
            print(f"\n{name.upper()}: No data")
