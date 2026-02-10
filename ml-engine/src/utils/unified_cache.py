"""
Unified Player Data Cache

Loads unified_players.csv ONCE into memory on first access.
All endpoints read from this DataFrame instead of loading separate CSVs.

Thread-safe singleton with TTL-based refresh (1 hour default).
Falls back to portal_nil_valuations.csv if unified table not yet generated.
"""

import logging
import time
import threading
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from .s3_storage import load_csv_with_fallback

logger = logging.getLogger(__name__)

# Configuration
CACHE_TTL_SECONDS = 3600  # 1 hour
UNIFIED_S3_KEY = "processed/unified_players.csv"
FALLBACK_S3_KEY = "processed/portal_nil_valuations.csv"

# PFF columns for extraction
PFF_STAT_COLUMNS = [
    "pff_overall", "pff_offense", "pff_defense",
    "pff_passing", "pff_rushing", "pff_receiving",
    "pff_pass_block", "pff_run_block", "pff_pass_rush",
    "pff_coverage", "pff_run_defense", "pff_tackling",
    "games_played", "snap_counts_offense", "snap_counts_defense",
    "completion_pct", "passer_rating", "big_time_throw_pct",
    "turnover_worthy_play_pct", "avg_depth_of_target",
    "pressure_grades_pass",
    "elusive_rating", "yaco_per_attempt", "breakaway_pct",
    "breakaway_yards", "missed_tackles_forced",
    "yards_per_route_run", "drop_rate", "contested_catch_rate",
    "targeted_qb_rating", "targets", "receptions", "rec_yards",
    "pass_rush_win_rate", "pass_rushing_productivity",
    "pressures", "sacks", "hurries",
    "forced_incompletion_rate", "passer_rating_allowed",
    "yards_per_coverage_snap",
    "man_grades_coverage_defense", "zone_grades_coverage_defense",
    "pass_blocking_efficiency", "pressures_allowed",
    "true_pass_set_pbe",
    "tackles", "missed_tackle_rate",
    "yards", "touchdowns",
]


class UnifiedPlayerCache:
    """Thread-safe singleton cache for unified player data."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._df = None
                    cls._instance._loaded_at = 0
                    cls._instance._is_unified = False
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._df is not None and (time.time() - self._loaded_at < CACHE_TTL_SECONDS)

    @property
    def is_unified(self) -> bool:
        """Whether the full unified table is loaded (vs fallback)."""
        return self._is_unified

    @property
    def df(self) -> pd.DataFrame:
        """Get the cached DataFrame, loading if needed."""
        if not self.is_loaded:
            self._load()
        return self._df

    def _load(self):
        """Load unified_players.csv from R2 into memory."""
        with self._lock:
            # Double-check after acquiring lock
            if self.is_loaded:
                return

            # Try unified table first
            try:
                logger.info("Loading unified_players.csv into memory...")
                df = load_csv_with_fallback(UNIFIED_S3_KEY, None, 1)
                if df is not None and not df.empty and len(df.columns) > 20:
                    self._df = df
                    self._is_unified = True
                    self._loaded_at = time.time()
                    logger.info(
                        f"Unified cache loaded: {len(df)} players, "
                        f"{len(df.columns)} columns"
                    )
                    self._ensure_indexes()
                    return
            except Exception as e:
                logger.warning(f"Could not load unified_players.csv: {e}")

            # Fallback to legacy portal_nil_valuations.csv
            try:
                logger.info("Falling back to portal_nil_valuations.csv...")
                df = load_csv_with_fallback(FALLBACK_S3_KEY, None, 1)
                if df is not None and not df.empty:
                    self._df = df
                    self._is_unified = False
                    self._loaded_at = time.time()
                    # Add missing columns for compatibility
                    if "name_normalized" not in df.columns and "name" in df.columns:
                        df["name_normalized"] = df["name"].str.lower().str.strip()
                    if "school_normalized" not in df.columns and "school" in df.columns:
                        df["school_normalized"] = df["school"].str.lower().str.strip()
                    if "in_portal" not in df.columns:
                        df["in_portal"] = False
                    logger.info(
                        f"Legacy cache loaded: {len(df)} players, "
                        f"{len(df.columns)} columns (unified=False)"
                    )
                    return
            except Exception as e:
                logger.error(f"Could not load fallback data: {e}")

            # Last resort: empty DataFrame
            self._df = pd.DataFrame()
            self._loaded_at = time.time()
            logger.error("No player data available!")

    def _ensure_indexes(self):
        """Ensure normalized columns exist for fast lookups."""
        if self._df is None or self._df.empty:
            return
        if "name_normalized" not in self._df.columns and "name" in self._df.columns:
            self._df["name_normalized"] = self._df["name"].str.lower().str.strip()
        if "school_normalized" not in self._df.columns and "school" in self._df.columns:
            self._df["school_normalized"] = self._df["school"].str.lower().str.strip()

    def get_player(self, name: str) -> Optional[pd.Series]:
        """Get a single player by name."""
        if self.df.empty:
            return None

        name_norm = name.lower().strip()

        # Exact match on normalized name
        if "name_normalized" in self.df.columns:
            mask = self.df["name_normalized"] == name_norm
            matches = self.df[mask]
            if not matches.empty:
                return matches.iloc[0]

        # Exact match on name column
        if "name" in self.df.columns:
            mask = self.df["name"].str.lower() == name_norm
            matches = self.df[mask]
            if not matches.empty:
                return matches.iloc[0]

            # Contains fallback
            mask = self.df["name"].str.lower().str.contains(name_norm, na=False)
            matches = self.df[mask]
            if not matches.empty:
                # Prefer exact length match
                matches = matches.copy()
                matches["_len_diff"] = (matches["name"].str.len() - len(name)).abs()
                return matches.sort_values("_len_diff").iloc[0]

        return None

    def get_players_by_school(self, school: str) -> pd.DataFrame:
        """Get all players at a school."""
        if self.df.empty:
            return pd.DataFrame()

        school_norm = school.lower().strip()
        if "school_normalized" in self.df.columns:
            return self.df[self.df["school_normalized"] == school_norm].copy()
        if "school" in self.df.columns:
            return self.df[self.df["school"].str.lower().str.strip() == school_norm].copy()
        return pd.DataFrame()

    def get_nil_leaderboard(
        self,
        position: Optional[str] = None,
        school: Optional[str] = None,
        conference: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """Get NIL leaderboard with filters. Replaces get_nil_players()."""
        if self.df.empty:
            return pd.DataFrame()

        df = self.df.copy()

        if position:
            pos_upper = position.upper()
            if "position" in df.columns:
                df = df[df["position"].str.upper() == pos_upper]

        if school:
            school_lower = school.lower()
            if "school" in df.columns:
                df = df[df["school"].str.lower().str.contains(school_lower, na=False)]

        if conference and "conference" in df.columns:
            conf_lower = conference.lower()
            df = df[df["conference"].str.lower().str.contains(conf_lower, na=False)]

        if search:
            search_lower = search.lower()
            if "name" in df.columns:
                df = df[df["name"].str.lower().str.contains(search_lower, na=False)]

        if "nil_value" in df.columns:
            df = df.sort_values("nil_value", ascending=False)

        return df.head(limit)

    def get_portal_players(
        self,
        status: Optional[str] = None,
        position: Optional[str] = None,
        origin_school: Optional[str] = None,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """Get portal players. Replaces get_portal_players()."""
        if self.df.empty:
            return pd.DataFrame()

        if "in_portal" not in self.df.columns:
            return pd.DataFrame()

        df = self.df[self.df["in_portal"] == True].copy()

        if status and "portal_status" in df.columns:
            df = df[df["portal_status"] == status]

        if position and "position" in df.columns:
            df = df[df["position"].str.upper() == position.upper()]

        if origin_school and "origin_school" in df.columns:
            df = df[df["origin_school"].str.lower().str.contains(
                origin_school.lower(), na=False
            )]

        if "nil_value" in df.columns:
            df = df.sort_values("nil_value", ascending=False)

        return df.head(limit)

    def get_player_pff_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get PFF stats for a player. Replaces get_player_pff_stats()."""
        player = self.get_player(name)
        if player is None:
            return None

        # Check if player has PFF data
        pff_overall = player.get("pff_overall")
        if pd.isna(pff_overall) or pff_overall == 0:
            return None

        stats: Dict[str, Any] = {
            "name": player.get("name", name),
            "position": str(player.get("position", "")),
            "team": str(player.get("school", "")),
            "games_played": int(player.get("games_played", 0)) if pd.notna(player.get("games_played")) else None,
        }

        # Extract all PFF columns
        for col in PFF_STAT_COLUMNS:
            val = player.get(col)
            if pd.notna(val):
                try:
                    stats[col] = float(val)
                except (ValueError, TypeError):
                    pass

        return stats

    def get_player_war(self, name: str) -> Optional[Dict[str, Any]]:
        """Get pre-computed WAR for a player."""
        player = self.get_player(name)
        if player is None:
            return None

        war = player.get("war")
        if pd.isna(war):
            return None

        return {
            "war": float(war),
            "war_low": float(player.get("war_low", 0)),
            "war_high": float(player.get("war_high", 0)),
            "confidence": str(player.get("war_confidence", "low")),
        }

    def search_players(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Search players by name. Replaces search_players()."""
        if self.df.empty or not query:
            return pd.DataFrame()

        query_lower = query.lower().strip()
        if "name" in self.df.columns:
            mask = self.df["name"].str.lower().str.contains(query_lower, na=False)
            results = self.df[mask].copy()
            if "nil_value" in results.columns:
                results = results.sort_values("nil_value", ascending=False)
            return results.head(limit)

        return pd.DataFrame()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get aggregate database statistics. Replaces get_database_stats()."""
        if self.df.empty:
            return {"total_players": 0}

        stats = {
            "total_players": len(self.df),
            "nil_valuations": int(self.df["nil_value"].notna().sum()) if "nil_value" in self.df.columns else 0,
        }

        if "in_portal" in self.df.columns:
            stats["portal_entries"] = int(self.df["in_portal"].sum())

        if "pff_overall" in self.df.columns:
            stats["pff_records"] = int(self.df["pff_overall"].notna().sum())

        if "nil_value" in self.df.columns:
            stats["avg_nil_value"] = float(self.df["nil_value"].mean()) if self.df["nil_value"].notna().any() else 0
            stats["total_nil_value"] = float(self.df["nil_value"].sum())

        return stats

    def invalidate(self):
        """Force cache reload on next access."""
        with self._lock:
            self._loaded_at = 0
            logger.info("Unified cache invalidated")


def get_unified_cache() -> UnifiedPlayerCache:
    """Get the singleton unified player cache."""
    return UnifiedPlayerCache()
