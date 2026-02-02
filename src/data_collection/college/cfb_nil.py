"""
College Football NIL (Name, Image, Likeness) Data Collector

Collects and manages NIL valuation data for college football players.
Since NIL data isn't available via API, this module provides:
- Structured CSV templates for manual data entry
- Web scraping attempts for On3 NIL rankings
- Social media value estimation
- Data merging and enrichment

DATA SOURCES (Manual Collection Required):
==========================================

1. On3 NIL 100 Rankings
   - https://www.on3.com/nil/rankings/player/nil-100/
   - Best for: Comprehensive NIL valuations, ranking
   - Updates: Weekly during season

2. On3 NIL Valuation Tool
   - https://www.on3.com/nil/
   - Best for: Individual player valuations
   - Includes social media analysis

3. Opendorse NIL Data
   - https://opendorse.com/nil/
   - Best for: Deal structure, brand partnerships

4. INFLCR / Altius Sports Partners
   - Industry reports on collective spending
   - Best for: Collective budget estimates

5. Social Media Platforms
   - Instagram, TikTok, Twitter profile pages
   - For follower counts and engagement

NIL VALUATION METHODOLOGY:
==========================
NIL valuations are estimates based on:
- Social media following and engagement
- On-field performance and visibility
- School/market size
- Position (QBs and skill players command premiums)
- Marketability factors

CPM rates and multipliers should be calibrated with real data when available.
"""

import os
import logging
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import pandas as pd
import yaml
from dotenv import load_dotenv

# Optional import for web scraping
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CFBNILCollector:
    """
    Collects and manages NIL valuation data for college football players.

    Features:
    - CSV templates for manual data entry
    - On3 NIL rankings scraping (with fallback)
    - Social media value estimation
    - Data merging and enrichment
    - Sample data generation for testing
    """

    CACHE_HOURS = 24

    # NIL tier thresholds
    NIL_TIERS = {
        "mega": 1_000_000,      # $1M+
        "premium": 500_000,     # $500K-$1M
        "solid": 100_000,       # $100K-$500K
        "moderate": 25_000,     # $25K-$100K
        "entry": 0,             # Under $25K
    }

    # Default CPM rates for social media valuation
    # These are rough estimates - calibrate with real data
    DEFAULT_CPM_RATES = {
        "instagram": 10.0,   # $ per 1000 impressions
        "tiktok": 5.0,
        "twitter": 3.0,
        "youtube": 15.0,
    }

    # Default posts per month by platform
    DEFAULT_POSTS_PER_MONTH = {
        "instagram": 4,
        "tiktok": 8,
        "twitter": 10,
        "youtube": 2,
    }

    # Engagement rate assumptions (followers who see content)
    ENGAGEMENT_RATES = {
        "instagram": 0.01,    # 1% of followers
        "tiktok": 0.005,      # 0.5% of followers
        "twitter": 0.003,     # 0.3% of followers
        "youtube": 0.02,      # 2% of subscribers
    }

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the NIL data collector.

        Args:
            data_dir: Base directory for data storage
        """
        load_dotenv()

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

        # Load config for school/conference tiers
        self.config = self._load_config()

        logger.info(f"CFBNILCollector initialized. Data dir: {self.data_dir}")

        # Create templates if they don't exist
        self._ensure_templates_exist()

    def _load_config(self) -> Dict:
        """Load configuration from config.yaml."""
        config_path = self.data_dir.parent / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        return {}

    def _ensure_templates_exist(self) -> None:
        """Create CSV templates if they don't exist."""
        self._create_nil_valuations_template()
        self._create_social_media_template()
        self._create_collective_budgets_template()

    def _create_nil_valuations_template(self) -> None:
        """Create the NIL valuations CSV template."""
        template_path = self.raw_dir / "nil_valuations.csv"

        if template_path.exists():
            return

        columns = [
            "player_name",
            "school",
            "position",
            "conference",
            "season",
            "estimated_annual_nil_value",
            "nil_tier",                    # mega/premium/solid/moderate/entry
            "on3_nil_valuation",           # On3's estimate if available
            "primary_nil_source",          # collective/brand/social/combination
            "number_of_known_deals",
            "collective_deal_flag",        # True if has collective deal
            "brand_deal_flag",             # True if has brand endorsement
            "social_media_deal_flag",      # True if monetizing social
            "is_sample_data",              # True if this is sample/test data
            "notes",
        ]

        template_df = pd.DataFrame(columns=columns)
        template_df.to_csv(template_path, index=False)
        logger.info(f"Created NIL valuations template: {template_path}")

    def _create_social_media_template(self) -> None:
        """Create the social media profiles CSV template."""
        template_path = self.raw_dir / "social_media_profiles.csv"

        if template_path.exists():
            return

        columns = [
            "player_name",
            "school",
            "position",
            "instagram_followers",
            "instagram_engagement_rate",
            "tiktok_followers",
            "tiktok_avg_views",
            "twitter_followers",
            "youtube_subscribers",
            "total_social_following",
            "verified_flag",
            "measurement_date",
            "is_sample_data",
        ]

        template_df = pd.DataFrame(columns=columns)
        template_df.to_csv(template_path, index=False)
        logger.info(f"Created social media template: {template_path}")

    def _create_collective_budgets_template(self) -> None:
        """Create the NIL collective budgets CSV template."""
        template_path = self.raw_dir / "nil_collective_budgets.csv"

        if template_path.exists():
            return

        columns = [
            "school",
            "conference",
            "estimated_annual_budget",
            "estimated_roster_spots_funded",
            "estimated_avg_deal_value",
            "top_deal_value",
            "collective_name",
            "source",
            "season",
            "is_sample_data",
        ]

        template_df = pd.DataFrame(columns=columns)
        template_df.to_csv(template_path, index=False)
        logger.info(f"Created collective budgets template: {template_path}")

    def scrape_on3_nil_rankings(self) -> pd.DataFrame:
        """
        Attempt to scrape On3 NIL 100 rankings.

        Note: On3 uses JavaScript rendering, so this will likely fail.
        In that case, it provides instructions for manual data collection.

        Returns:
            DataFrame with NIL rankings (or empty if scraping fails)
        """
        if not SCRAPING_AVAILABLE:
            logger.warning("requests and beautifulsoup4 not installed. Cannot scrape.")
            self._print_manual_instructions()
            return pd.DataFrame()

        url = "https://www.on3.com/nil/rankings/player/nil-100/"
        logger.info(f"Attempting to scrape On3 NIL 100 from: {url}")

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Look for player data - On3's structure varies
            # This is a best-effort attempt that may not work due to JS rendering
            players = []

            # Try various selectors that might contain player data
            player_elements = soup.find_all("div", class_=lambda x: x and "player" in x.lower()) or \
                              soup.find_all("tr", class_=lambda x: x and "ranking" in x.lower()) or \
                              soup.find_all("li", class_=lambda x: x and "nil" in x.lower())

            if not player_elements:
                logger.warning("No player elements found - likely JavaScript-rendered content")
                self._print_manual_instructions()
                return pd.DataFrame()

            for element in player_elements:
                try:
                    # Extract player info (selector patterns vary)
                    name = element.find(class_=lambda x: x and "name" in x.lower())
                    school = element.find(class_=lambda x: x and "school" in x.lower())
                    position = element.find(class_=lambda x: x and "position" in x.lower())
                    valuation = element.find(class_=lambda x: x and "valuation" in x.lower())

                    if name:
                        players.append({
                            "player_name": name.get_text(strip=True) if name else None,
                            "school": school.get_text(strip=True) if school else None,
                            "position": position.get_text(strip=True) if position else None,
                            "on3_nil_valuation": self._parse_valuation(
                                valuation.get_text(strip=True) if valuation else None
                            ),
                            "source": "on3_scrape",
                            "scrape_date": datetime.now().isoformat(),
                        })
                except Exception as e:
                    logger.debug(f"Error parsing player element: {e}")
                    continue

            if players:
                df = pd.DataFrame(players)
                # Save scraped data
                output_path = self.raw_dir / f"on3_nil_100_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"Scraped {len(df)} players, saved to {output_path}")
                return df
            else:
                logger.warning("Scraping succeeded but no data extracted")
                self._print_manual_instructions()
                return pd.DataFrame()

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            self._print_manual_instructions()
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            self._print_manual_instructions()
            return pd.DataFrame()

    def _print_manual_instructions(self) -> None:
        """Print instructions for manual data collection."""
        print("\n" + "=" * 70)
        print("MANUAL DATA COLLECTION REQUIRED")
        print("=" * 70)
        print("""
On3 NIL 100 requires JavaScript rendering and cannot be scraped directly.

Please manually collect the data:

1. Visit: https://www.on3.com/nil/rankings/player/nil-100/

2. For each player, record:
   - Rank
   - Player name
   - School
   - Position
   - NIL Valuation (On3's estimate)

3. Save to: data/raw/on3_nil_100.csv

   With columns:
   rank,player_name,school,position,on3_nil_valuation

4. Alternative: Use browser developer tools to copy the data table,
   or use a browser automation tool like Selenium/Playwright.

Other data sources to cross-reference:
- https://247sports.com/ (recruiting rankings)
- https://opendorse.com/nil/ (deal data)
- Player social media profiles (follower counts)
""")
        print("=" * 70 + "\n")

    def _parse_valuation(self, value_str: str) -> Optional[float]:
        """Parse a valuation string like '$1.2M' to a float."""
        if not value_str:
            return None

        try:
            # Remove $ and whitespace
            value_str = value_str.replace("$", "").replace(",", "").strip()

            # Handle M for millions, K for thousands
            if "M" in value_str.upper():
                return float(value_str.upper().replace("M", "")) * 1_000_000
            elif "K" in value_str.upper():
                return float(value_str.upper().replace("K", "")) * 1_000
            else:
                return float(value_str)
        except Exception:
            return None

    def estimate_social_value(
        self,
        instagram_followers: int = 0,
        tiktok_followers: int = 0,
        twitter_followers: int = 0,
        youtube_subscribers: int = 0,
        instagram_posts_per_month: int = None,
        tiktok_posts_per_month: int = None,
        twitter_posts_per_month: int = None,
        youtube_videos_per_month: int = None,
        cpm_rates: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Estimate annual social media monetization value.

        METHODOLOGY:
        ============
        For each platform:
        Annual Value = followers * engagement_rate * posts_per_month * 12 * CPM / 1000

        These are rough estimates based on industry averages.
        CPM rates vary significantly based on:
        - Sport/niche
        - Engagement quality
        - Brand safety
        - Advertiser demand

        For accurate valuations, calibrate CPM rates with real deal data.

        Args:
            instagram_followers: Instagram follower count
            tiktok_followers: TikTok follower count
            twitter_followers: Twitter follower count
            youtube_subscribers: YouTube subscriber count
            instagram_posts_per_month: Custom posts/month (default: 4)
            tiktok_posts_per_month: Custom posts/month (default: 8)
            twitter_posts_per_month: Custom posts/month (default: 10)
            youtube_videos_per_month: Custom videos/month (default: 2)
            cpm_rates: Custom CPM rates dict

        Returns:
            Dict with per-platform and total annual estimated earnings
        """
        # Use defaults if not specified
        cpm = cpm_rates or self.DEFAULT_CPM_RATES
        posts = {
            "instagram": instagram_posts_per_month or self.DEFAULT_POSTS_PER_MONTH["instagram"],
            "tiktok": tiktok_posts_per_month or self.DEFAULT_POSTS_PER_MONTH["tiktok"],
            "twitter": twitter_posts_per_month or self.DEFAULT_POSTS_PER_MONTH["twitter"],
            "youtube": youtube_videos_per_month or self.DEFAULT_POSTS_PER_MONTH["youtube"],
        }

        results = {}

        # Instagram value
        ig_monthly = (
            instagram_followers
            * self.ENGAGEMENT_RATES["instagram"]
            * posts["instagram"]
            * cpm.get("instagram", 10.0)
            / 1000
        )
        results["instagram_annual"] = ig_monthly * 12

        # TikTok value
        tt_monthly = (
            tiktok_followers
            * self.ENGAGEMENT_RATES["tiktok"]
            * posts["tiktok"]
            * cpm.get("tiktok", 5.0)
            / 1000
        )
        results["tiktok_annual"] = tt_monthly * 12

        # Twitter value
        tw_monthly = (
            twitter_followers
            * self.ENGAGEMENT_RATES["twitter"]
            * posts["twitter"]
            * cpm.get("twitter", 3.0)
            / 1000
        )
        results["twitter_annual"] = tw_monthly * 12

        # YouTube value
        yt_monthly = (
            youtube_subscribers
            * self.ENGAGEMENT_RATES["youtube"]
            * posts["youtube"]
            * cpm.get("youtube", 15.0)
            / 1000
        )
        results["youtube_annual"] = yt_monthly * 12

        # Total
        results["total_annual"] = sum([
            results["instagram_annual"],
            results["tiktok_annual"],
            results["twitter_annual"],
            results["youtube_annual"],
        ])

        results["total_followers"] = (
            instagram_followers + tiktok_followers + twitter_followers + youtube_subscribers
        )

        # Add methodology note
        results["note"] = (
            "Estimates based on industry average CPM rates. "
            "Actual values vary significantly. Calibrate with real deal data."
        )

        return results

    def load_nil_valuations(self) -> pd.DataFrame:
        """Load NIL valuations from CSV."""
        path = self.raw_dir / "nil_valuations.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Error loading NIL valuations: {e}")
        return pd.DataFrame()

    def load_social_media_profiles(self) -> pd.DataFrame:
        """Load social media profiles from CSV."""
        path = self.raw_dir / "social_media_profiles.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Error loading social media profiles: {e}")
        return pd.DataFrame()

    def load_collective_budgets(self) -> pd.DataFrame:
        """Load collective budgets from CSV."""
        path = self.raw_dir / "nil_collective_budgets.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Error loading collective budgets: {e}")
        return pd.DataFrame()

    def get_nil_tier(self, value: float) -> str:
        """
        Get NIL tier for a given value.

        Args:
            value: Estimated annual NIL value

        Returns:
            Tier string: mega/premium/solid/moderate/entry
        """
        if value >= self.NIL_TIERS["mega"]:
            return "mega"
        elif value >= self.NIL_TIERS["premium"]:
            return "premium"
        elif value >= self.NIL_TIERS["solid"]:
            return "solid"
        elif value >= self.NIL_TIERS["moderate"]:
            return "moderate"
        else:
            return "entry"

    def get_school_tier(self, school: str) -> str:
        """Get school tier from config."""
        school_tiers = self.config.get("school_tiers", {})

        for tier, schools in school_tiers.items():
            if school in schools:
                return tier

        return "unknown"

    def get_conference_tier(self, conference: str) -> int:
        """Get conference tier from config."""
        conf_tiers = self.config.get("conference_tiers", {})

        for tier_num, (tier_name, conferences) in enumerate(conf_tiers.items(), 1):
            if conference in conferences:
                return tier_num

        return 5  # Default to lowest tier

    def build_nil_dataset(self) -> pd.DataFrame:
        """
        Build the master NIL analysis dataset.

        Merges:
        - NIL valuations
        - Social media profiles (with estimated social value)
        - Player stats (if available)
        - Recruiting data (if available)
        - School/conference tiers

        Returns:
            Merged DataFrame saved to data/processed/nil_master_dataset.csv
        """
        logger.info("Building NIL master dataset...")

        # Load base data
        nil_df = self.load_nil_valuations()
        social_df = self.load_social_media_profiles()

        if nil_df.empty and social_df.empty:
            logger.warning("No NIL or social media data available")
            logger.info("Run populate_sample_data() to generate test data")
            return pd.DataFrame()

        # Start with NIL valuations as base
        if not nil_df.empty:
            master_df = nil_df.copy()
        else:
            master_df = pd.DataFrame()

        # Merge social media data
        if not social_df.empty:
            # Calculate social value for each profile
            social_values = []
            for _, row in social_df.iterrows():
                value_estimate = self.estimate_social_value(
                    instagram_followers=int(row.get("instagram_followers", 0) or 0),
                    tiktok_followers=int(row.get("tiktok_followers", 0) or 0),
                    twitter_followers=int(row.get("twitter_followers", 0) or 0),
                    youtube_subscribers=int(row.get("youtube_subscribers", 0) or 0),
                )
                social_values.append({
                    "player_name": row["player_name"],
                    "estimated_social_value": value_estimate["total_annual"],
                    "total_social_following": value_estimate["total_followers"],
                })

            social_value_df = pd.DataFrame(social_values)

            if not master_df.empty:
                master_df = master_df.merge(
                    social_df,
                    on=["player_name", "school", "position"],
                    how="left",
                    suffixes=("", "_social"),
                )
                master_df = master_df.merge(
                    social_value_df,
                    on="player_name",
                    how="left",
                    suffixes=("", "_calc"),
                )
            else:
                master_df = social_df.merge(social_value_df, on="player_name")

        # Add school tier
        if "school" in master_df.columns:
            master_df["school_tier"] = master_df["school"].apply(self.get_school_tier)

        # Add conference tier
        if "conference" in master_df.columns:
            master_df["conference_tier"] = master_df["conference"].apply(self.get_conference_tier)

        # Try to load and merge recruiting data
        try:
            recruiting_path = self.cache_dir / "cfb_recruiting_players_2018_2025_cache.csv"
            if recruiting_path.exists():
                recruiting_df = pd.read_csv(recruiting_path)
                if not recruiting_df.empty and "name" in recruiting_df.columns:
                    recruiting_df = recruiting_df[["name", "stars", "rating", "national_ranking"]].rename(
                        columns={
                            "name": "player_name",
                            "stars": "recruiting_stars",
                            "rating": "recruiting_rating",
                            "national_ranking": "recruiting_rank",
                        }
                    )
                    master_df = master_df.merge(
                        recruiting_df,
                        on="player_name",
                        how="left",
                    )
                    logger.info("Merged recruiting data")
        except Exception as e:
            logger.debug(f"Could not merge recruiting data: {e}")

        # Try to load and merge player stats
        try:
            stats_path = self.cache_dir / "cfb_player_stats_2020_2025_cache.csv"
            if stats_path.exists():
                stats_df = pd.read_csv(stats_path)
                name_col = "player_name" if "player_name" in stats_df.columns else "player"
                if not stats_df.empty and name_col in stats_df.columns:
                    # Get latest season stats
                    latest_stats = stats_df.sort_values("season", ascending=False).groupby(name_col).first().reset_index()
                    if name_col != "player_name":
                        latest_stats = latest_stats.rename(columns={name_col: "player_name"})
                    master_df = master_df.merge(
                        latest_stats,
                        on="player_name",
                        how="left",
                        suffixes=("", "_stats"),
                    )
                    logger.info("Merged player stats")
        except Exception as e:
            logger.debug(f"Could not merge player stats: {e}")

        # Save master dataset
        if not master_df.empty:
            output_path = self.processed_dir / "nil_master_dataset.csv"
            master_df.to_csv(output_path, index=False)
            logger.info(f"Saved NIL master dataset to {output_path} ({len(master_df)} rows)")

        return master_df

    def populate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """
        Populate templates with realistic sample data for testing.

        Creates 50 sample players across different tiers and positions
        with realistic social media follower counts.

        All sample data is clearly marked with is_sample_data=True.

        Returns:
            Dict of created DataFrames
        """
        logger.info("Generating sample NIL data...")
        random.seed(42)  # For reproducibility

        # Sample data parameters
        sample_players = self._generate_sample_players(50)

        # Create NIL valuations
        nil_data = []
        for player in sample_players:
            nil_value = player["nil_value"]
            nil_data.append({
                "player_name": player["name"],
                "school": player["school"],
                "position": player["position"],
                "conference": player["conference"],
                "season": 2025,
                "estimated_annual_nil_value": nil_value,
                "nil_tier": self.get_nil_tier(nil_value),
                "on3_nil_valuation": nil_value * random.uniform(0.9, 1.1),
                "primary_nil_source": random.choice(["collective", "brand", "social", "combination"]),
                "number_of_known_deals": random.randint(1, 10) if nil_value > 100000 else random.randint(0, 3),
                "collective_deal_flag": nil_value > 50000,
                "brand_deal_flag": nil_value > 200000,
                "social_media_deal_flag": player["social"]["total"] > 100000,
                "is_sample_data": True,
                "notes": "SAMPLE DATA - for testing only",
            })

        nil_df = pd.DataFrame(nil_data)
        nil_path = self.raw_dir / "nil_valuations.csv"
        nil_df.to_csv(nil_path, index=False)
        logger.info(f"Created {len(nil_df)} sample NIL valuations")

        # Create social media profiles
        social_data = []
        for player in sample_players:
            social = player["social"]
            social_data.append({
                "player_name": player["name"],
                "school": player["school"],
                "position": player["position"],
                "instagram_followers": social["instagram"],
                "instagram_engagement_rate": random.uniform(0.02, 0.08),
                "tiktok_followers": social["tiktok"],
                "tiktok_avg_views": int(social["tiktok"] * random.uniform(0.05, 0.15)),
                "twitter_followers": social["twitter"],
                "youtube_subscribers": social["youtube"],
                "total_social_following": social["total"],
                "verified_flag": social["total"] > 500000,
                "measurement_date": datetime.now().strftime("%Y-%m-%d"),
                "is_sample_data": True,
            })

        social_df = pd.DataFrame(social_data)
        social_path = self.raw_dir / "social_media_profiles.csv"
        social_df.to_csv(social_path, index=False)
        logger.info(f"Created {len(social_df)} sample social media profiles")

        # Create collective budgets
        collective_data = self._generate_sample_collectives()
        collective_df = pd.DataFrame(collective_data)
        collective_path = self.raw_dir / "nil_collective_budgets.csv"
        collective_df.to_csv(collective_path, index=False)
        logger.info(f"Created {len(collective_df)} sample collective budgets")

        return {
            "nil_valuations": nil_df,
            "social_media_profiles": social_df,
            "nil_collective_budgets": collective_df,
        }

    def _generate_sample_players(self, count: int) -> List[Dict]:
        """Generate sample player data."""
        # School data by tier
        schools = {
            "blue_blood": [
                ("Alabama", "SEC"), ("Ohio State", "Big Ten"), ("Georgia", "SEC"),
                ("Texas", "SEC"), ("USC", "Big Ten"),
            ],
            "elite": [
                ("Michigan", "Big Ten"), ("Penn State", "Big Ten"), ("LSU", "SEC"),
                ("Oregon", "Big Ten"), ("Tennessee", "SEC"),
            ],
            "power_brand": [
                ("Auburn", "SEC"), ("Florida State", "ACC"), ("Wisconsin", "Big Ten"),
                ("UCLA", "Big Ten"), ("Miami", "ACC"),
            ],
            "p4_mid": [
                ("Iowa", "Big Ten"), ("Michigan State", "Big Ten"), ("NC State", "ACC"),
                ("Virginia Tech", "ACC"), ("Louisville", "ACC"),
            ],
            "g5_strong": [
                ("Boise State", "Mountain West"), ("Memphis", "AAC"), ("UCF", "Big 12"),
                ("SMU", "ACC"), ("Tulane", "AAC"),
            ],
        }

        positions = ["QB", "RB", "WR", "WR", "WR", "TE", "OL", "EDGE", "DL", "LB", "CB", "S"]

        # NIL value ranges by tier and position
        nil_ranges = {
            ("blue_blood", "QB"): (800_000, 2_500_000),
            ("blue_blood", "WR"): (200_000, 800_000),
            ("blue_blood", "RB"): (150_000, 500_000),
            ("elite", "QB"): (400_000, 1_200_000),
            ("elite", "WR"): (100_000, 400_000),
            ("p4_mid", "QB"): (100_000, 400_000),
            ("g5_strong", "QB"): (30_000, 150_000),
        }

        # Social media ranges by tier
        social_ranges = {
            "blue_blood": {
                "instagram": (200_000, 2_000_000),
                "tiktok": (100_000, 1_500_000),
                "twitter": (50_000, 500_000),
                "youtube": (10_000, 200_000),
            },
            "elite": {
                "instagram": (50_000, 500_000),
                "tiktok": (30_000, 400_000),
                "twitter": (20_000, 150_000),
                "youtube": (5_000, 50_000),
            },
            "power_brand": {
                "instagram": (20_000, 200_000),
                "tiktok": (10_000, 150_000),
                "twitter": (10_000, 75_000),
                "youtube": (1_000, 20_000),
            },
            "p4_mid": {
                "instagram": (5_000, 75_000),
                "tiktok": (3_000, 50_000),
                "twitter": (2_000, 30_000),
                "youtube": (500, 10_000),
            },
            "g5_strong": {
                "instagram": (2_000, 30_000),
                "tiktok": (1_000, 20_000),
                "twitter": (1_000, 15_000),
                "youtube": (100, 5_000),
            },
        }

        first_names = [
            "Jayden", "Caleb", "Travis", "Jalen", "Bryce", "Quinn", "Carson", "Arch",
            "Nico", "Dylan", "Keenan", "DeVonta", "Garrett", "Brock", "Cameron",
            "Drake", "Marvin", "Rome", "Tetairoa", "Miller", "Luther", "Cade",
        ]
        last_names = [
            "Williams", "Smith", "Johnson", "Brown", "Davis", "Wilson", "Thompson",
            "Martinez", "Anderson", "Taylor", "Thomas", "Jackson", "White", "Harris",
            "Moore", "Young", "Allen", "King", "Wright", "Scott", "Green", "Adams",
        ]

        players = []
        used_names = set()

        for i in range(count):
            # Distribute across tiers
            tier_weights = [0.15, 0.20, 0.25, 0.25, 0.15]  # Distribution across tiers
            tier = random.choices(list(schools.keys()), weights=tier_weights)[0]

            school, conference = random.choice(schools[tier])
            position = random.choice(positions)

            # Generate unique name
            while True:
                name = f"{random.choice(first_names)} {random.choice(last_names)}"
                if name not in used_names:
                    used_names.add(name)
                    break

            # Get NIL value
            nil_key = (tier, position)
            if nil_key in nil_ranges:
                min_val, max_val = nil_ranges[nil_key]
            else:
                # Default range based on tier
                base = {"blue_blood": 150_000, "elite": 75_000, "power_brand": 40_000,
                        "p4_mid": 20_000, "g5_strong": 10_000}[tier]
                min_val, max_val = base * 0.5, base * 2.5

            nil_value = int(random.uniform(min_val, max_val))

            # Get social media followers
            social_range = social_ranges.get(tier, social_ranges["p4_mid"])
            social = {
                "instagram": random.randint(*social_range["instagram"]),
                "tiktok": random.randint(*social_range["tiktok"]),
                "twitter": random.randint(*social_range["twitter"]),
                "youtube": random.randint(*social_range["youtube"]),
            }
            social["total"] = sum(social.values())

            players.append({
                "name": name,
                "school": school,
                "conference": conference,
                "position": position,
                "tier": tier,
                "nil_value": nil_value,
                "social": social,
            })

        return players

    def _generate_sample_collectives(self) -> List[Dict]:
        """Generate sample collective budget data."""
        collectives = [
            # Blue bloods
            {"school": "Alabama", "conference": "SEC", "budget": 20_000_000, "spots": 85, "collective": "Yea Alabama", "top": 3_000_000},
            {"school": "Ohio State", "conference": "Big Ten", "budget": 18_000_000, "spots": 80, "collective": "The Foundation", "top": 2_500_000},
            {"school": "Georgia", "conference": "SEC", "budget": 15_000_000, "spots": 75, "collective": "Champions Circle", "top": 2_000_000},
            {"school": "Texas", "conference": "SEC", "budget": 20_000_000, "spots": 85, "collective": "Horns with Heart", "top": 3_000_000},
            {"school": "USC", "conference": "Big Ten", "budget": 12_000_000, "spots": 70, "collective": "Student Body Right", "top": 2_000_000},
            # Elite
            {"school": "Michigan", "conference": "Big Ten", "budget": 10_000_000, "spots": 65, "collective": "Champions Circle", "top": 1_500_000},
            {"school": "Penn State", "conference": "Big Ten", "budget": 8_000_000, "spots": 60, "collective": "We Are", "top": 1_200_000},
            {"school": "Oregon", "conference": "Big Ten", "budget": 12_000_000, "spots": 70, "collective": "Division Street", "top": 1_800_000},
            {"school": "Tennessee", "conference": "SEC", "budget": 10_000_000, "spots": 65, "collective": "Spyre Sports", "top": 1_500_000},
            # Power brands
            {"school": "Miami", "conference": "ACC", "budget": 6_000_000, "spots": 50, "collective": "Canes Connection", "top": 1_000_000},
            {"school": "Florida State", "conference": "ACC", "budget": 5_000_000, "spots": 45, "collective": "Warpath", "top": 800_000},
        ]

        data = []
        for c in collectives:
            data.append({
                "school": c["school"],
                "conference": c["conference"],
                "estimated_annual_budget": c["budget"],
                "estimated_roster_spots_funded": c["spots"],
                "estimated_avg_deal_value": int(c["budget"] / c["spots"]),
                "top_deal_value": c["top"],
                "collective_name": c["collective"],
                "source": "Industry estimates (SAMPLE)",
                "season": 2025,
                "is_sample_data": True,
            })

        return data


if __name__ == "__main__":
    print("College Football NIL Data Collector")
    print("-" * 50)

    collector = CFBNILCollector()

    # Generate sample data
    print("\n[1/4] Generating sample data for testing...")
    samples = collector.populate_sample_data()
    for name, df in samples.items():
        print(f"  {name}: {len(df)} rows")

    # Try scraping On3 (will likely print manual instructions)
    print("\n[2/4] Attempting to scrape On3 NIL 100...")
    on3_data = collector.scrape_on3_nil_rankings()
    if not on3_data.empty:
        print(f"  Scraped {len(on3_data)} players")
    else:
        print("  Scraping failed - see instructions above")

    # Estimate social value example
    print("\n[3/4] Social media value estimation example...")
    example = collector.estimate_social_value(
        instagram_followers=500_000,
        tiktok_followers=300_000,
        twitter_followers=100_000,
        youtube_subscribers=50_000,
    )
    print(f"  Instagram annual: ${example['instagram_annual']:,.0f}")
    print(f"  TikTok annual: ${example['tiktok_annual']:,.0f}")
    print(f"  Twitter annual: ${example['twitter_annual']:,.0f}")
    print(f"  YouTube annual: ${example['youtube_annual']:,.0f}")
    print(f"  TOTAL annual: ${example['total_annual']:,.0f}")

    # Build master dataset
    print("\n[4/4] Building NIL master dataset...")
    master = collector.build_nil_dataset()
    if not master.empty:
        print(f"  Master dataset: {len(master)} rows, {len(master.columns)} columns")

        # Show tier distribution
        print("\n" + "=" * 50)
        print("NIL TIER DISTRIBUTION")
        print("=" * 50)
        if "nil_tier" in master.columns:
            tier_dist = master["nil_tier"].value_counts()
            for tier, count in tier_dist.items():
                avg_val = master[master["nil_tier"] == tier]["estimated_annual_nil_value"].mean()
                print(f"  {tier:12s}: {count:3d} players, avg ${avg_val:,.0f}")

        # Show top players
        print("\n" + "=" * 50)
        print("TOP 10 NIL VALUATIONS (SAMPLE DATA)")
        print("=" * 50)
        top_10 = master.nlargest(10, "estimated_annual_nil_value")[
            ["player_name", "school", "position", "estimated_annual_nil_value", "nil_tier"]
        ]
        print(top_10.to_string(index=False))
