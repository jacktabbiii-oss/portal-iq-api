"""
Weekly Stats & NIL Update Script

This script runs weekly to:
1. Pull latest player stats from CFBD API
2. Update NIL valuations based on new performance data
3. Optionally refresh recruiting rankings
4. Enrich with ESPN data (headshots, ESPN IDs)

Usage:
    python scripts/weekly_stats_update.py

Schedule with cron/Task Scheduler for weekly execution (recommended: Sunday night).
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(project_root / "logs" / "weekly_stats_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def fetch_cfbd_player_stats(season: int = 2024) -> pd.DataFrame:
    """
    Fetch player season stats from CFBD API.
    """
    try:
        import cfbd
        from cfbd.rest import ApiException

        api_key = os.getenv("CFBD_API_KEY")
        if not api_key:
            logger.error("CFBD_API_KEY not set")
            return pd.DataFrame()

        configuration = cfbd.Configuration()
        configuration.api_key['Authorization'] = api_key
        configuration.api_key_prefix['Authorization'] = 'Bearer'

        api_instance = cfbd.PlayersApi(cfbd.ApiClient(configuration))

        logger.info(f"Fetching player stats for {season} season...")

        try:
            stats = api_instance.get_player_season_stats(year=season)

            if stats:
                records = []
                for stat in stats:
                    records.append({
                        'player': stat.player,
                        'player_id': stat.player_id,
                        'team': stat.team,
                        'conference': stat.conference,
                        'position': getattr(stat, 'position', None),
                        'category': stat.category,
                        'stat_type': stat.stat_type,
                        'stat': stat.stat,
                        'season': season,
                    })

                df = pd.DataFrame(records)
                logger.info(f"Fetched {len(df)} stat records")
                return df

        except ApiException as e:
            logger.error(f"CFBD API error: {e}")

    except ImportError:
        logger.error("cfbd package not installed: pip install cfbd")

    return pd.DataFrame()


def fetch_cfbd_recruiting(year: int = 2025) -> pd.DataFrame:
    """
    Fetch recruiting rankings from CFBD API.
    """
    try:
        import cfbd

        api_key = os.getenv("CFBD_API_KEY")
        if not api_key:
            return pd.DataFrame()

        configuration = cfbd.Configuration()
        configuration.api_key['Authorization'] = api_key
        configuration.api_key_prefix['Authorization'] = 'Bearer'

        api_instance = cfbd.RecruitingApi(cfbd.ApiClient(configuration))

        logger.info(f"Fetching {year} recruiting rankings...")

        try:
            recruits = api_instance.get_recruiting_players(year=year)

            if recruits:
                records = []
                for r in recruits:
                    records.append({
                        'name': r.name,
                        'school': r.committed_to,
                        'position': r.position,
                        'stars': r.stars,
                        'rating': r.rating,
                        'ranking': r.ranking,
                        'state_rank': getattr(r, 'state_rank', None),
                        'position_rank': getattr(r, 'position_rank', None),
                        'year': year,
                    })

                df = pd.DataFrame(records)
                logger.info(f"Fetched {len(df)} recruiting records")
                return df

        except Exception as e:
            logger.warning(f"Recruiting fetch error: {e}")

    except ImportError:
        pass

    return pd.DataFrame()


def pivot_stats_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot stats from long format (one row per stat) to wide format (one row per player).
    """
    if df.empty:
        return df

    # Create stat key
    df['stat_key'] = df['category'] + '_' + df['stat_type']

    # Pivot
    pivot_df = df.pivot_table(
        index=['player', 'player_id', 'team', 'conference', 'position'],
        columns='stat_key',
        values='stat',
        aggfunc='first'
    ).reset_index()

    logger.info(f"Pivoted to {len(pivot_df)} unique players")
    return pivot_df


def run_nil_valuations(stats_df: pd.DataFrame, recruiting_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run NIL valuator on the stats data.
    """
    try:
        from models.custom_nil_valuator import CustomNILValuator

        # Map column names
        column_mapping = {
            'player': 'player_name',
            'team': 'school',
            'passing_YDS': 'passing_yards',
            'passing_TD': 'passing_tds',
            'rushing_YDS': 'rushing_yards',
            'rushing_TD': 'rushing_tds',
            'rushing_CAR': 'rushing_carries',
            'receiving_YDS': 'receiving_yards',
            'receiving_TD': 'receiving_tds',
            'receiving_REC': 'receptions',
            'defensive_TOT': 'tackles',
            'defensive_SACKS': 'sacks',
            'defensive_TFL': 'tackles_for_loss',
            'interceptions_INT': 'interceptions',
        }

        df = stats_df.copy()
        rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_cols)

        # Fill NaN with 0 for numeric stats
        stat_cols = ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
                     'receiving_yards', 'receiving_tds', 'tackles', 'sacks', 'interceptions']
        for col in stat_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Merge recruiting data if available
        if not recruiting_df.empty:
            recruiting_df['name_lower'] = recruiting_df['name'].str.lower().str.strip()
            df['name_lower'] = df['player_name'].str.lower().str.strip()

            merged = df.merge(
                recruiting_df[['name_lower', 'stars', 'rating', 'ranking']].rename(columns={
                    'stars': 'recruiting_stars',
                    'rating': 'recruiting_rating',
                    'ranking': 'national_rank'
                }),
                on='name_lower',
                how='left'
            )
            merged['recruiting_stars'] = merged['recruiting_stars'].fillna(0).astype(int)
            df = merged.drop(columns=['name_lower'], errors='ignore')

        # Run valuator
        valuator = CustomNILValuator(calibration_factor=1.0)
        result_df = valuator.valuate_dataframe(df)
        result_df = result_df.sort_values('custom_nil_value', ascending=False)

        logger.info(f"Generated valuations for {len(result_df)} players")
        return result_df

    except ImportError as e:
        logger.error(f"Could not import CustomNILValuator: {e}")
        return pd.DataFrame()


def enrich_with_espn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich player data with ESPN headshots and IDs.
    """
    try:
        from data_collection.college.espn_fetcher import ESPNFetcher

        fetcher = ESPNFetcher(data_dir=str(project_root / "data"))

        # Check if we have cached ESPN data
        espn_file = project_root / "data" / "raw" / "espn_rosters.csv"
        if not espn_file.exists():
            logger.info("ESPN roster cache not found, fetching (this may take a few minutes)...")
            fetcher.fetch_all_rosters(save=True)

        # Enrich the data
        enriched_df = fetcher.enrich_players_with_espn(df)
        return enriched_df

    except ImportError as e:
        logger.warning(f"Could not import ESPNFetcher: {e}")
        return df
    except Exception as e:
        logger.warning(f"ESPN enrichment failed: {e}")
        return df


def main():
    """Main weekly update function."""
    print("=" * 60)
    print("WEEKLY STATS & NIL UPDATE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    # Determine current season
    current_year = datetime.now().year
    # If we're in Jan-Jul, use previous year's season
    season = current_year if datetime.now().month > 7 else current_year - 1

    # 1. Fetch player stats
    print(f"\n[1/5] Fetching {season} player stats from CFBD...")
    stats_df = fetch_cfbd_player_stats(season)

    if stats_df.empty:
        # Try loading from cache
        cache_file = raw_dir / "player_season_stats.csv"
        if cache_file.exists():
            stats_df = pd.read_csv(cache_file)
            stats_df = stats_df[stats_df['season'] == season]
            logger.info(f"Loaded {len(stats_df)} cached stats for {season}")
        else:
            logger.error("No stats data available")
            return

    # Save raw stats
    stats_file = raw_dir / f"player_season_stats_{season}.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"  Saved {len(stats_df)} stats to {stats_file}")

    # 2. Fetch recruiting data
    print(f"\n[2/5] Fetching recruiting rankings...")
    recruiting_df = pd.DataFrame()

    # Load existing recruiting data
    recruiting_file = raw_dir / "recruiting_rankings.csv"
    if recruiting_file.exists():
        recruiting_df = pd.read_csv(recruiting_file)
        logger.info(f"Loaded {len(recruiting_df)} existing recruiting records")

    # Optionally fetch new recruiting data
    new_recruiting = fetch_cfbd_recruiting(current_year)
    if not new_recruiting.empty:
        recruiting_df = pd.concat([recruiting_df, new_recruiting], ignore_index=True)
        recruiting_df = recruiting_df.drop_duplicates(subset=['name', 'year'], keep='last')
        recruiting_df.to_csv(recruiting_file, index=False)
        print(f"  Updated recruiting data: {len(recruiting_df)} total records")

    # 3. Pivot stats and run valuations
    print(f"\n[3/5] Running NIL valuations...")
    wide_stats = pivot_stats_to_wide(stats_df)

    if wide_stats.empty:
        logger.error("No stats to valuate")
        return

    valuations_df = run_nil_valuations(wide_stats, recruiting_df)

    if valuations_df.empty:
        logger.error("Valuation failed")
        return

    # 4. Enrich with ESPN data (headshots, ESPN IDs)
    print(f"\n[4/5] Enriching with ESPN data...")
    valuations_df = enrich_with_espn(valuations_df)

    espn_cols = ['espn_id', 'espn_headshot_url', 'espn_profile_url']
    espn_matched = valuations_df['espn_id'].notna().sum() if 'espn_id' in valuations_df.columns else 0
    print(f"  ESPN data added for {espn_matched} players")

    # 5. Save results
    print(f"\n[5/5] Saving results...")

    # Select output columns
    output_cols = [
        'player_name', 'position', 'school', 'conference',
        'custom_nil_value', 'nil_tier', 'valuation_confidence',
        'performance_value', 'market_value', 'potential_value',
        'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
        'receiving_yards', 'receiving_tds', 'tackles', 'sacks', 'interceptions',
        'recruiting_stars', 'national_rank',
        # ESPN data
        'espn_id', 'espn_headshot_url', 'espn_profile_url'
    ]
    available_cols = [c for c in output_cols if c in valuations_df.columns]
    output_df = valuations_df[available_cols].copy()

    # Save to processed directory
    output_path = processed_dir / f"nil_valuations_{season}.csv"
    output_df.to_csv(output_path, index=False)
    print(f"  Saved valuations to {output_path}")

    # Also save as "latest" for API
    latest_path = processed_dir / "nil_valuations_latest.csv"
    output_df.to_csv(latest_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Season: {season}")
    print(f"Total players valuated: {len(output_df)}")
    print(f"Total NIL value: ${output_df['custom_nil_value'].sum():,.0f}")
    print(f"Average value: ${output_df['custom_nil_value'].mean():,.0f}")

    print("\nTier distribution:")
    tier_counts = output_df['nil_tier'].value_counts()
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count:,} players")

    print(f"\nTop 10 valuations:")
    for i, row in output_df.head(10).iterrows():
        name = row['player_name'][:25]
        val = row['custom_nil_value']
        pos = row['position']
        school = row['school'][:15] if 'school' in row else ''
        print(f"  {i+1:2}. ${val:>10,.0f} | {name:25} | {pos:4} | {school}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
