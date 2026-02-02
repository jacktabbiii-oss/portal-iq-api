"""
NFL Salary Cap Data Collector

Collects salary cap data for all NFL teams.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class CapDataCollector:
    """Collects NFL salary cap data."""

    CACHE_HOURS = 24

    # 2025 salary cap
    SALARY_CAP_2025 = 255_000_000

    NFL_TEAMS = [
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
        "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
    ]

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the cap data collector."""
        load_dotenv()

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

    def collect_team_cap_data(self, year: int = 2025) -> pd.DataFrame:
        """
        Collect cap situation for all teams.

        Args:
            year: Season year

        Returns:
            DataFrame with team cap data
        """
        cache_name = f"team_caps_{year}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        logger.info(f"Collecting team cap data for {year}")

        # TODO: Integrate with real data source (Spotrac, OTC, etc.)

        columns = [
            "team",
            "year",
            "cap_limit",
            "cap_spent",
            "cap_space",
            "dead_money",
            "top_51_cap",
        ]

        df = pd.DataFrame(columns=columns)

        if not df.empty:
            self._save_cache(df, cache_name)

        return df

    def collect_cap_history(
        self,
        start_year: int = 2015,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Collect historical cap data.

        Args:
            start_year: First year
            end_year: Last year

        Returns:
            DataFrame with historical cap data
        """
        cache_name = f"cap_history_{start_year}_{end_year}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        logger.info(f"Collecting cap history for {start_year}-{end_year}")

        # Historical salary caps
        salary_caps = {
            2015: 143_280_000,
            2016: 155_270_000,
            2017: 167_000_000,
            2018: 177_200_000,
            2019: 188_200_000,
            2020: 198_200_000,
            2021: 182_500_000,
            2022: 208_200_000,
            2023: 224_800_000,
            2024: 255_400_000,
            2025: 255_000_000,
        }

        all_data = []
        for year in range(start_year, end_year + 1):
            cap = salary_caps.get(year, 255_000_000)
            for team in self.NFL_TEAMS:
                all_data.append({
                    "team": team,
                    "year": year,
                    "cap_limit": cap,
                })

        df = pd.DataFrame(all_data)

        if not df.empty:
            self._save_cache(df, cache_name)

        return df

    def get_team_cap(self, team: str, year: int = 2025) -> Dict:
        """Get cap details for a specific team."""
        df = self.collect_team_cap_data(year)
        if df.empty:
            return {
                "team": team,
                "year": year,
                "cap_limit": self.SALARY_CAP_2025,
                "cap_spent": 0,
                "cap_space": self.SALARY_CAP_2025,
            }

        team_data = df[df["team"] == team]
        if team_data.empty:
            return {
                "team": team,
                "year": year,
                "cap_limit": self.SALARY_CAP_2025,
                "cap_spent": 0,
                "cap_space": self.SALARY_CAP_2025,
            }

        return team_data.iloc[0].to_dict()

    def get_league_cap_overview(self, year: int = 2025) -> Dict:
        """Get cap overview for entire league."""
        df = self.collect_team_cap_data(year)

        if df.empty:
            return {
                "year": year,
                "cap_limit": self.SALARY_CAP_2025,
                "teams": 32,
                "total_spent": 0,
                "avg_spent": 0,
            }

        return {
            "year": year,
            "cap_limit": self.SALARY_CAP_2025,
            "teams": len(df),
            "total_spent": df["cap_spent"].sum(),
            "avg_spent": df["cap_spent"].mean(),
            "most_space": df.loc[df["cap_space"].idxmax(), "team"],
            "least_space": df.loc[df["cap_space"].idxmin(), "team"],
        }


if __name__ == "__main__":
    collector = CapDataCollector()
    print("Cap data collector initialized")
    print(f"2025 cap: ${collector.SALARY_CAP_2025:,}")
