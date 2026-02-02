"""
NFL Player Stats Collector

Collects player statistics using nfl_data_py.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd

try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False

logger = logging.getLogger(__name__)


class NFLPlayerStatsCollector:
    """Collects NFL player statistics."""

    CACHE_HOURS = 24

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the NFL player stats collector."""
        if not NFL_DATA_AVAILABLE:
            logger.warning("nfl_data_py not installed. Some features unavailable.")

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
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, name: str) -> Path:
        return self.cache_dir / f"nfl_{name}_cache.csv"

    def _is_cache_valid(self, name: str) -> bool:
        cache_path = self._get_cache_path(name)
        meta_path = self.cache_dir / f"nfl_{name}_cache_meta.txt"
        if not cache_path.exists() or not meta_path.exists():
            return False
        try:
            with open(meta_path) as f:
                cache_time = datetime.fromisoformat(f.read().strip())
            return datetime.now() - cache_time < timedelta(hours=self.CACHE_HOURS)
        except Exception:
            return False

    def _load_cache(self, name: str) -> Optional[pd.DataFrame]:
        if self._is_cache_valid(name):
            try:
                return pd.read_csv(self._get_cache_path(name))
            except Exception:
                pass
        return None

    def _save_cache(self, df: pd.DataFrame, name: str) -> None:
        try:
            df.to_csv(self._get_cache_path(name), index=False)
            with open(self.cache_dir / f"nfl_{name}_cache_meta.txt", "w") as f:
                f.write(datetime.now().isoformat())
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def collect_weekly_stats(
        self,
        years: List[int],
        stat_type: str = "offense",
    ) -> pd.DataFrame:
        """
        Collect weekly player stats.

        Args:
            years: List of seasons
            stat_type: 'offense' or 'defense'

        Returns:
            DataFrame with weekly stats
        """
        if not NFL_DATA_AVAILABLE:
            logger.error("nfl_data_py required for weekly stats")
            return pd.DataFrame()

        cache_name = f"weekly_{stat_type}_{min(years)}_{max(years)}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        logger.info(f"Collecting weekly {stat_type} stats for {years}")

        try:
            df = nfl.import_weekly_data(years)
            self._save_cache(df, cache_name)
            return df
        except Exception as e:
            logger.error(f"Error collecting weekly stats: {e}")
            return pd.DataFrame()

    def collect_seasonal_stats(self, years: List[int]) -> pd.DataFrame:
        """
        Collect seasonal player stats.

        Args:
            years: List of seasons

        Returns:
            DataFrame with seasonal stats
        """
        if not NFL_DATA_AVAILABLE:
            return pd.DataFrame()

        cache_name = f"seasonal_{min(years)}_{max(years)}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        logger.info(f"Collecting seasonal stats for {years}")

        try:
            df = nfl.import_seasonal_data(years)
            self._save_cache(df, cache_name)
            return df
        except Exception as e:
            logger.error(f"Error collecting seasonal stats: {e}")
            return pd.DataFrame()

    def collect_rosters(self, years: List[int]) -> pd.DataFrame:
        """
        Collect team rosters.

        Args:
            years: List of seasons

        Returns:
            DataFrame with roster data
        """
        if not NFL_DATA_AVAILABLE:
            return pd.DataFrame()

        cache_name = f"rosters_{min(years)}_{max(years)}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        logger.info(f"Collecting rosters for {years}")

        try:
            df = nfl.import_rosters(years)
            self._save_cache(df, cache_name)
            return df
        except Exception as e:
            logger.error(f"Error collecting rosters: {e}")
            return pd.DataFrame()

    def collect_draft_picks(self, years: List[int]) -> pd.DataFrame:
        """
        Collect NFL draft picks.

        Args:
            years: List of draft years

        Returns:
            DataFrame with draft picks
        """
        if not NFL_DATA_AVAILABLE:
            return pd.DataFrame()

        cache_name = f"draft_{min(years)}_{max(years)}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        logger.info(f"Collecting draft data for {years}")

        try:
            df = nfl.import_draft_picks(years)
            self._save_cache(df, cache_name)
            return df
        except Exception as e:
            logger.error(f"Error collecting draft data: {e}")
            return pd.DataFrame()

    def collect_combine_data(self, years: List[int]) -> pd.DataFrame:
        """
        Collect NFL combine data.

        Args:
            years: List of combine years

        Returns:
            DataFrame with combine results
        """
        if not NFL_DATA_AVAILABLE:
            return pd.DataFrame()

        cache_name = f"combine_{min(years)}_{max(years)}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        logger.info(f"Collecting combine data for {years}")

        try:
            df = nfl.import_combine_data(years)
            self._save_cache(df, cache_name)
            return df
        except Exception as e:
            logger.error(f"Error collecting combine data: {e}")
            return pd.DataFrame()

    def get_player_stats(
        self,
        player_name: str,
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Get stats for a specific player."""
        if years is None:
            years = list(range(2020, 2025))

        df = self.collect_seasonal_stats(years)
        if df.empty:
            return df

        return df[df["player_name"].str.contains(player_name, case=False, na=False)]


if __name__ == "__main__":
    collector = NFLPlayerStatsCollector()
    print("NFL stats collector initialized")
    print(f"nfl_data_py available: {NFL_DATA_AVAILABLE}")
