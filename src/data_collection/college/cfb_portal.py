"""
Transfer Portal Data Collector

Collects transfer portal entries, enriches with player stats, and builds
outcome datasets for portal fit prediction models.

DATA SOURCES:
=============
If CFBD API doesn't have complete portal data, manually source from:

1. On3 Transfer Portal Rankings
   - https://www.on3.com/transfer-portal/rankings/football/
   - Best for: Player ratings, NIL valuations, commitment tracking

2. 247Sports Transfer Portal
   - https://247sports.com/Season/2025-Football/TransferPortal/
   - Best for: Original recruiting rankings, commitment dates

3. ESPN Transfer Portal Tracker
   - https://www.espn.com/college-football/story/_/id/transfer-portal-tracker
   - Best for: Commitment status, destination schools

4. Rivals Transfer Portal
   - https://n.rivals.com/transfer_portal
   - Best for: Player ratings, position rankings

TEMPLATE CSV:
=============
If API data is incomplete, create data/raw/portal_entries.csv with columns:
- player_name, original_school, original_conference, new_school, new_conference
- position, season, stars, recruiting_ranking
- transfer_direction (up/lateral/down based on school tier)
- prev_team_wins, prev_snap_pct, prev_starter_flag
- reason_category (playing_time/coaching_change/nil/scheme_fit/personal/unknown)
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import pandas as pd
import yaml
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


class CFBPortalCollector:
    """
    Collects transfer portal data and builds enriched datasets.

    Features:
    - Portal entry collection from CFBD API
    - CSV template for manual data entry when API incomplete
    - Enrichment with player stats and recruiting data
    - School tier classification and transfer direction
    - Transfer outcome analysis (success metrics)
    """

    RATE_LIMIT_SECONDS = 0.5
    CACHE_HOURS = 24

    # Default school tiers (loaded from config if available)
    DEFAULT_SCHOOL_TIERS = {
        "blue_blood": [
            "Alabama", "Ohio State", "Georgia", "Clemson",
            "Oklahoma", "Notre Dame", "Texas", "USC",
        ],
        "elite": [
            "Michigan", "Penn State", "LSU", "Florida",
            "Oregon", "Tennessee", "Miami", "Texas A&M",
        ],
        "power_brand": [
            "Auburn", "Florida State", "Wisconsin", "UCLA",
            "Nebraska", "Arkansas", "South Carolina", "Ole Miss",
        ],
        "p4_mid": [
            "Iowa", "Michigan State", "NC State", "Virginia Tech",
            "Louisville", "Pittsburgh", "West Virginia", "Arizona State",
        ],
        "g5_strong": [
            "Boise State", "Memphis", "UCF", "SMU",
            "Tulane", "Liberty", "James Madison", "Appalachian State",
        ],
    }

    TIER_RANK = {
        "blue_blood": 1,
        "elite": 2,
        "power_brand": 3,
        "p4_mid": 4,
        "g5_strong": 5,
        "g5": 6,
        "fcs": 7,
        "unknown": 8,
    }

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the portal data collector.

        Args:
            data_dir: Base directory for data storage
        """
        load_dotenv()

        if not CFBD_AVAILABLE:
            logger.warning("cfbd package not installed. Will use CSV data only.")

        # Configure CFBD API if available
        if CFBD_AVAILABLE:
            self.api_key = os.getenv("CFBD_API_KEY", "")
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

        # Load school tiers from config
        self.school_tiers = self._load_school_tiers()

        # Cache for loaded data
        self._player_stats_cache: Optional[pd.DataFrame] = None
        self._recruiting_cache: Optional[pd.DataFrame] = None
        self._coaching_changes_cache: Optional[pd.DataFrame] = None

        logger.info(f"CFBPortalCollector initialized. Data dir: {self.data_dir}")

    def _load_school_tiers(self) -> Dict[str, List[str]]:
        """Load school tiers from config.yaml if available."""
        config_path = self.data_dir.parent / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                if config and "school_tiers" in config:
                    return config["school_tiers"]
            except Exception as e:
                logger.warning(f"Could not load school tiers from config: {e}")

        return self.DEFAULT_SCHOOL_TIERS

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

    def collect_portal_entries(
        self,
        start_year: int = 2021,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Collect transfer portal entries for given years.

        Uses CFBD API if available, otherwise loads from CSV template.

        Args:
            start_year: First year to collect (default: 2021, when portal expanded)
            end_year: Last year to collect (inclusive)

        Returns:
            DataFrame with portal entries including:
            - player_name, original_school, new_school, position
            - star_rating, year, entry_date, commitment_date
        """
        # Check cache first
        cached = self._load_cache("portal_entries", start_year, end_year)
        if cached is not None:
            return cached

        logger.info(f"Collecting portal entries for {start_year}-{end_year}")

        all_entries = []

        # Try CFBD API first
        if CFBD_AVAILABLE and self.api_key:
            players_api = cfbd.PlayersApi(cfbd.ApiClient(self.configuration))

            for year in range(start_year, end_year + 1):
                try:
                    self._rate_limit()
                    transfers = players_api.get_transfer_portal(year=year)

                    for t in transfers:
                        entry = {
                            "player_id": getattr(t, "player_id", None),
                            "player_name": f"{getattr(t, 'first_name', '')} {getattr(t, 'last_name', '')}".strip(),
                            "position": getattr(t, "position", None),
                            "original_school": getattr(t, "origin", None),
                            "new_school": getattr(t, "destination", None),
                            "transfer_date": getattr(t, "transfer_date", None),
                            "stars": getattr(t, "stars", None),
                            "rating": getattr(t, "rating", None),
                            "eligibility": getattr(t, "eligibility", None),
                            "season": year,
                        }
                        all_entries.append(entry)

                    logger.info(f"  {year}: {len(transfers)} portal entries from API")

                except ApiException as e:
                    logger.error(f"API error for {year}: {e}")
                except Exception as e:
                    logger.error(f"Error for {year}: {e}")

        # Check for manual CSV data to supplement
        csv_path = self.raw_dir / "portal_entries.csv"
        if csv_path.exists():
            try:
                csv_data = pd.read_csv(csv_path)
                csv_data = csv_data[
                    (csv_data["season"] >= start_year) &
                    (csv_data["season"] <= end_year)
                ]
                logger.info(f"Loaded {len(csv_data)} entries from manual CSV")

                # Merge with API data
                for _, row in csv_data.iterrows():
                    all_entries.append(row.to_dict())

            except Exception as e:
                logger.warning(f"Error loading manual CSV: {e}")

        if not all_entries:
            logger.warning("No portal data collected. Creating template CSV.")
            self._create_template_csv()
            return pd.DataFrame()

        df = pd.DataFrame(all_entries)

        # Remove duplicates (prefer API data)
        if "player_name" in df.columns and "season" in df.columns:
            df = df.drop_duplicates(subset=["player_name", "season"], keep="first")

        # Save to cache
        self._save_cache(df, "portal_entries", start_year, end_year)

        # Save to raw
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = self.raw_dir / f"portal_entries_{start_year}_{end_year}_{timestamp}.csv"
        df.to_csv(raw_path, index=False)

        logger.info(f"Collected {len(df)} total portal entries")
        return df

    def _create_template_csv(self) -> None:
        """Create a template CSV for manual portal data entry."""
        template_path = self.raw_dir / "portal_entries_TEMPLATE.csv"

        columns = [
            "player_name",
            "original_school",
            "original_conference",
            "new_school",
            "new_conference",
            "position",
            "season",
            "stars",
            "recruiting_ranking",
            "transfer_direction",  # up/lateral/down
            "prev_team_wins",
            "prev_snap_pct",
            "prev_starter_flag",
            "reason_category",  # playing_time/coaching_change/nil/scheme_fit/personal/unknown
            "entry_date",
            "commitment_date",
        ]

        template_df = pd.DataFrame(columns=columns)
        template_df.to_csv(template_path, index=False)

        logger.info(f"Created template CSV at: {template_path}")
        logger.info("Fill this template with data from On3, 247Sports, ESPN, or Rivals")

    def get_school_tier(self, school: str) -> str:
        """
        Get the tier for a school.

        Args:
            school: School name

        Returns:
            Tier string (blue_blood, elite, power_brand, p4_mid, g5_strong, g5, fcs)
        """
        if not school:
            return "unknown"

        school_lower = school.lower()

        for tier, schools in self.school_tiers.items():
            if any(s.lower() == school_lower for s in schools):
                return tier

        # Default classification based on keywords
        if any(conf in school_lower for conf in ["sec", "big ten", "big 12", "acc", "pac"]):
            return "p4_mid"

        return "g5"

    def calculate_transfer_direction(
        self,
        origin_school: str,
        destination_school: str,
    ) -> str:
        """
        Calculate transfer direction based on school tiers.

        Args:
            origin_school: School transferred from
            destination_school: School transferred to

        Returns:
            "up", "lateral", or "down"
        """
        if not origin_school or not destination_school:
            return "unknown"

        origin_tier = self.get_school_tier(origin_school)
        dest_tier = self.get_school_tier(destination_school)

        origin_rank = self.TIER_RANK.get(origin_tier, 8)
        dest_rank = self.TIER_RANK.get(dest_tier, 8)

        if dest_rank < origin_rank:
            return "up"
        elif dest_rank > origin_rank:
            return "down"
        else:
            return "lateral"

    def enrich_portal_data(
        self,
        portal_df: pd.DataFrame = None,
        start_year: int = 2021,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Enrich portal entries with player stats, recruiting data, and school info.

        Joins with:
        - Player stats from previous school
        - Original recruiting ranking
        - School tier classifications
        - Transfer direction
        - Coaching change flags

        Args:
            portal_df: Portal entries DataFrame (loads if None)
            start_year: First year
            end_year: Last year

        Returns:
            Enriched DataFrame with additional columns
        """
        logger.info("Enriching portal data...")

        if portal_df is None:
            portal_df = self.collect_portal_entries(start_year, end_year)

        if portal_df.empty:
            logger.warning("No portal data to enrich")
            return portal_df

        enriched = portal_df.copy()

        # Add school tiers
        enriched["origin_tier"] = enriched["original_school"].apply(self.get_school_tier)
        enriched["destination_tier"] = enriched["new_school"].apply(self.get_school_tier)

        # Calculate transfer direction
        enriched["transfer_direction"] = enriched.apply(
            lambda r: self.calculate_transfer_direction(
                r.get("original_school"),
                r.get("new_school")
            ),
            axis=1,
        )

        # Load and join player stats
        player_stats = self._load_player_stats(start_year - 1, end_year)
        if not player_stats.empty:
            enriched = self._join_player_stats(enriched, player_stats)

        # Load and join recruiting data
        recruiting = self._load_recruiting_data(start_year - 5, end_year)
        if not recruiting.empty:
            enriched = self._join_recruiting_data(enriched, recruiting)

        # Add coaching change flags
        coaching_changes = self._load_coaching_changes(start_year, end_year)
        if not coaching_changes.empty:
            enriched = self._join_coaching_changes(enriched, coaching_changes)

        # Save enriched data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = self.raw_dir / f"portal_enriched_{start_year}_{end_year}_{timestamp}.csv"
        enriched.to_csv(raw_path, index=False)
        logger.info(f"Saved enriched portal data to {raw_path}")

        return enriched

    def _load_player_stats(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Load player stats for joining."""
        if self._player_stats_cache is not None:
            return self._player_stats_cache

        cache_path = self.cache_dir / f"cfb_player_stats_{start_year}_{end_year}_cache.csv"
        if cache_path.exists():
            try:
                self._player_stats_cache = pd.read_csv(cache_path)
                return self._player_stats_cache
            except Exception as e:
                logger.debug(f"Could not load player stats: {e}")

        try:
            from .cfb_stats import CFBStatsCollector
            collector = CFBStatsCollector(str(self.data_dir))
            self._player_stats_cache = collector.collect_player_stats(start_year, end_year)
            return self._player_stats_cache
        except Exception as e:
            logger.warning(f"Could not collect player stats: {e}")
            return pd.DataFrame()

    def _load_recruiting_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Load recruiting data for joining."""
        if self._recruiting_cache is not None:
            return self._recruiting_cache

        cache_path = self.cache_dir / f"cfb_recruiting_players_{start_year}_{end_year}_cache.csv"
        if cache_path.exists():
            try:
                self._recruiting_cache = pd.read_csv(cache_path)
                return self._recruiting_cache
            except Exception as e:
                logger.debug(f"Could not load recruiting data: {e}")

        try:
            from .cfb_recruiting import CFBRecruitingCollector
            collector = CFBRecruitingCollector(str(self.data_dir))
            self._recruiting_cache = collector.collect_player_recruiting_rankings(start_year, end_year)
            return self._recruiting_cache
        except Exception as e:
            logger.warning(f"Could not collect recruiting data: {e}")
            return pd.DataFrame()

    def _load_coaching_changes(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Load coaching changes data."""
        # Check for manual CSV
        csv_path = self.raw_dir / "coaching_changes.csv"
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path)
            except Exception:
                pass

        # Return empty - coaching changes typically need manual entry
        return pd.DataFrame()

    def _join_player_stats(
        self,
        portal_df: pd.DataFrame,
        stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join portal entries with player stats from previous school."""
        if stats_df.empty:
            return portal_df

        # Get name column
        name_col = "player_name" if "player_name" in stats_df.columns else "player"

        # Merge on player name and original school
        merged = portal_df.merge(
            stats_df,
            left_on=["player_name", "original_school"],
            right_on=[name_col, "team"],
            how="left",
            suffixes=("", "_prev"),
        )

        # Rename stat columns to indicate they're from previous school
        stat_cols = [c for c in merged.columns if c.endswith("_prev")]
        rename_map = {c: f"prev_{c.replace('_prev', '')}" for c in stat_cols}
        merged = merged.rename(columns=rename_map)

        return merged

    def _join_recruiting_data(
        self,
        portal_df: pd.DataFrame,
        recruiting_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join portal entries with original recruiting data."""
        if recruiting_df.empty:
            return portal_df

        # Merge on player name
        recruiting_cols = ["name", "stars", "rating", "national_ranking", "recruiting_year"]
        available_cols = [c for c in recruiting_cols if c in recruiting_df.columns]

        if not available_cols:
            return portal_df

        merged = portal_df.merge(
            recruiting_df[available_cols],
            left_on="player_name",
            right_on="name",
            how="left",
            suffixes=("", "_recruit"),
        )

        # Rename to indicate recruiting data
        if "stars_recruit" in merged.columns:
            merged["original_stars"] = merged["stars_recruit"]
        if "rating_recruit" in merged.columns:
            merged["original_rating"] = merged["rating_recruit"]

        return merged

    def _join_coaching_changes(
        self,
        portal_df: pd.DataFrame,
        coaching_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join portal entries with coaching change data."""
        if coaching_df.empty:
            portal_df["coaching_change"] = False
            return portal_df

        # Check if origin school had coaching change that year
        portal_df = portal_df.merge(
            coaching_df[["school", "season", "coaching_change"]],
            left_on=["original_school", "season"],
            right_on=["school", "season"],
            how="left",
        )

        portal_df["coaching_change"] = portal_df["coaching_change"].fillna(False)

        return portal_df

    def build_portal_outcomes(
        self,
        enriched_df: pd.DataFrame = None,
        start_year: int = 2021,
        end_year: int = 2024,
    ) -> pd.DataFrame:
        """
        Build portal outcome dataset comparing stats before/after transfer.

        For players who transferred and played a season, compares their
        production at the new school vs old school.

        Args:
            enriched_df: Enriched portal DataFrame (loads if None)
            start_year: First year
            end_year: Last year (needs year after for outcomes)

        Returns:
            DataFrame with:
            - Pre-transfer stats
            - Post-transfer stats
            - Transfer success metrics
        """
        logger.info("Building portal outcomes dataset...")

        if enriched_df is None:
            enriched_df = self.enrich_portal_data(start_year=start_year, end_year=end_year)

        if enriched_df.empty:
            logger.warning("No enriched portal data available")
            return pd.DataFrame()

        # Filter to committed transfers
        committed = enriched_df[enriched_df["new_school"].notna()].copy()
        if committed.empty:
            logger.warning("No committed transfers found")
            return pd.DataFrame()

        # Load stats for post-transfer years
        post_stats = self._load_player_stats(start_year, end_year + 1)

        if post_stats.empty:
            logger.warning("No post-transfer stats available")
            return committed

        # Join post-transfer stats
        name_col = "player_name" if "player_name" in post_stats.columns else "player"

        outcomes = committed.merge(
            post_stats,
            left_on=["player_name", "new_school"],
            right_on=[name_col, "team"],
            how="left",
            suffixes=("", "_post"),
        )

        # Calculate success metrics
        outcomes = self._calculate_transfer_success(outcomes)

        # Save outcomes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = self.raw_dir / f"portal_outcomes_{start_year}_{end_year}_{timestamp}.csv"
        outcomes.to_csv(raw_path, index=False)
        logger.info(f"Saved portal outcomes to {raw_path}")

        return outcomes

    def _calculate_transfer_success(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate transfer success metrics.

        Compares pre and post transfer stats to determine:
        - improved: Player's production increased
        - maintained: Production stayed similar
        - declined: Production decreased
        """
        df = df.copy()

        # Define stat pairs to compare (pre vs post)
        stat_pairs = [
            ("prev_passing_YDS", "passing_YDS_post"),
            ("prev_rushing_YDS", "rushing_YDS_post"),
            ("prev_receiving_YDS", "receiving_YDS_post"),
        ]

        # Calculate production change for available stats
        df["production_change"] = 0.0
        df["has_post_stats"] = False

        for pre_col, post_col in stat_pairs:
            if pre_col in df.columns and post_col in df.columns:
                pre_val = df[pre_col].fillna(0)
                post_val = df[post_col].fillna(0)

                # Mark as having post stats if any post values exist
                df.loc[post_val > 0, "has_post_stats"] = True

                # Calculate change
                change = post_val - pre_val
                df["production_change"] += change

        # Categorize success
        def categorize_success(row):
            if not row["has_post_stats"]:
                return "no_data"
            change = row["production_change"]
            if change > 100:
                return "improved"
            elif change < -100:
                return "declined"
            else:
                return "maintained"

        df["transfer_success"] = df.apply(categorize_success, axis=1)

        return df

    def get_portal_summary(
        self,
        df: pd.DataFrame = None,
        year: int = None,
    ) -> Dict[str, Any]:
        """
        Get summary statistics for portal activity.

        Args:
            df: Portal DataFrame (loads if None)
            year: Specific year to analyze (all years if None)

        Returns:
            Dictionary with summary statistics
        """
        if df is None:
            df = self.collect_portal_entries()

        if df.empty:
            return {}

        if year:
            df = df[df["season"] == year]

        committed = df[df["new_school"].notna()]

        return {
            "total_entries": len(df),
            "committed": len(committed),
            "uncommitted": len(df) - len(committed),
            "commitment_rate": len(committed) / len(df) if len(df) > 0 else 0,
            "by_position": df["position"].value_counts().to_dict(),
            "by_direction": df["transfer_direction"].value_counts().to_dict() if "transfer_direction" in df.columns else {},
            "top_origins": df["original_school"].value_counts().head(10).to_dict(),
            "top_destinations": committed["new_school"].value_counts().head(10).to_dict() if len(committed) > 0 else {},
            "avg_stars": df["stars"].mean() if "stars" in df.columns else None,
        }


if __name__ == "__main__":
    print("College Football Transfer Portal Collector")
    print("-" * 50)

    collector = CFBPortalCollector()

    # Collect portal data
    print("\n[1/3] Collecting portal entries...")
    portal = collector.collect_portal_entries(2021, 2024)
    print(f"  Collected {len(portal)} portal entries")

    # Enrich data
    print("\n[2/3] Enriching portal data...")
    enriched = collector.enrich_portal_data(portal)
    print(f"  Enriched {len(enriched)} entries")

    # Build outcomes
    print("\n[3/3] Building portal outcomes...")
    outcomes = collector.build_portal_outcomes(enriched)
    print(f"  Built outcomes for {len(outcomes)} transfers")

    # Print summary
    print("\n" + "=" * 60)
    print("PORTAL SUMMARY")
    print("=" * 60)

    summary = collector.get_portal_summary(enriched)
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in list(value.items())[:5]:
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    # Transfer direction breakdown
    if "transfer_direction" in enriched.columns:
        print("\n" + "=" * 60)
        print("TRANSFER DIRECTION")
        print("=" * 60)
        print(enriched["transfer_direction"].value_counts().to_string())

    # Success metrics
    if "transfer_success" in outcomes.columns:
        print("\n" + "=" * 60)
        print("TRANSFER SUCCESS")
        print("=" * 60)
        print(outcomes["transfer_success"].value_counts().to_string())
