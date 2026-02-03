"""
Daily Transfer Portal Update Script

This script runs daily to update transfer portal data from multiple sources:
1. On3 portal page (scraped)
2. CFBD API /portal endpoint (if available)

Usage:
    python scripts/daily_portal_update.py

Schedule with cron/Task Scheduler for daily execution.
"""

import asyncio
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
        logging.FileHandler(project_root / "logs" / "daily_portal_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def fetch_cfbd_portal(year: int = 2025) -> pd.DataFrame:
    """
    Fetch transfer portal data from CFBD API.

    Note: CFBD portal endpoint may have limited data or require specific access.
    """
    try:
        import cfbd
        from cfbd.rest import ApiException

        api_key = os.getenv("CFBD_API_KEY")
        if not api_key:
            logger.warning("CFBD_API_KEY not set, skipping CFBD portal fetch")
            return pd.DataFrame()

        configuration = cfbd.Configuration()
        configuration.api_key['Authorization'] = api_key
        configuration.api_key_prefix['Authorization'] = 'Bearer'

        api_instance = cfbd.PlayersApi(cfbd.ApiClient(configuration))

        try:
            # Try to get transfer portal data
            portal_data = api_instance.get_transfer_portal(year=year)

            if portal_data:
                records = []
                for player in portal_data:
                    records.append({
                        'name': getattr(player, 'first_name', '') + ' ' + getattr(player, 'last_name', ''),
                        'position': getattr(player, 'position', None),
                        'previous_school': getattr(player, 'origin', None),
                        'new_school': getattr(player, 'destination', None),
                        'transfer_status': 'Committed' if getattr(player, 'destination', None) else 'In Portal',
                        'stars': getattr(player, 'stars', None),
                        'rating': getattr(player, 'rating', None),
                        'season': year,
                        'source': 'cfbd'
                    })

                df = pd.DataFrame(records)
                logger.info(f"CFBD: Fetched {len(df)} portal entries")
                return df

        except ApiException as e:
            logger.warning(f"CFBD portal endpoint not available or error: {e}")
        except Exception as e:
            logger.warning(f"CFBD fetch error: {e}")

    except ImportError:
        logger.warning("cfbd package not installed, skipping CFBD portal fetch")

    return pd.DataFrame()


async def scrape_on3_portal(year: int = 2025, pages: int = 10) -> pd.DataFrame:
    """
    Scrape transfer portal from On3.
    """
    try:
        from data_collection.college.on3_scraper import On3Scraper, TransferPortalEntry
        from dataclasses import asdict

        async with On3Scraper(headless=True) as scraper:
            entries = await scraper.scrape_transfer_portal(year=year, pages=pages)

            if entries:
                df = pd.DataFrame([asdict(e) for e in entries])
                df['source'] = 'on3'
                logger.info(f"On3: Scraped {len(df)} portal entries")
                return df

    except Exception as e:
        logger.error(f"On3 scraping error: {e}")

    return pd.DataFrame()


def merge_portal_data(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new portal data with existing data, handling updates and new entries.
    """
    if existing_df.empty:
        return new_df

    if new_df.empty:
        return existing_df

    # Create a unique key for each player
    def make_key(row):
        name = str(row.get('name', '')).lower().strip()
        prev = str(row.get('previous_school', '')).lower().strip()
        return f"{name}|{prev}"

    existing_df['_key'] = existing_df.apply(make_key, axis=1)
    new_df['_key'] = new_df.apply(make_key, axis=1)

    # Get existing keys
    existing_keys = set(existing_df['_key'])

    # Find new entries
    new_entries = new_df[~new_df['_key'].isin(existing_keys)]

    # Update existing entries (prefer new data for destination/status)
    for _, new_row in new_df.iterrows():
        key = new_row['_key']
        if key in existing_keys:
            # Update if new_school changed (player committed)
            mask = existing_df['_key'] == key
            if new_row.get('new_school') and not existing_df.loc[mask, 'new_school'].values[0]:
                existing_df.loc[mask, 'new_school'] = new_row['new_school']
                existing_df.loc[mask, 'transfer_status'] = new_row.get('transfer_status', 'Committed')
                logger.info(f"Updated: {new_row['name']} committed to {new_row['new_school']}")

    # Combine
    merged = pd.concat([existing_df, new_entries], ignore_index=True)
    merged = merged.drop(columns=['_key'], errors='ignore')

    # Remove duplicates (keep latest)
    merged = merged.drop_duplicates(subset=['name', 'previous_school'], keep='last')

    logger.info(f"Merged: {len(new_entries)} new entries, total {len(merged)}")
    return merged


def generate_portal_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the portal data."""
    summary = {
        'total_entries': len(df),
        'still_in_portal': len(df[df['transfer_status'] == 'In Portal']),
        'committed': len(df[df['transfer_status'].isin(['Committed', 'Enrolled'])]),
        'by_position': df['position'].value_counts().to_dict() if 'position' in df.columns else {},
        'top_destinations': df['new_school'].value_counts().head(10).to_dict() if 'new_school' in df.columns else {},
        'top_origins': df['previous_school'].value_counts().head(10).to_dict() if 'previous_school' in df.columns else {},
        'updated_at': datetime.now().isoformat(),
    }
    return summary


async def main():
    """Main daily update function."""
    print("=" * 60)
    print("DAILY TRANSFER PORTAL UPDATE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Ensure logs directory exists
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    # Load existing portal data
    portal_file = processed_dir / "transfer_portal.csv"
    if portal_file.exists():
        existing_df = pd.read_csv(portal_file)
        logger.info(f"Loaded {len(existing_df)} existing portal entries")
    else:
        existing_df = pd.DataFrame()
        logger.info("No existing portal data found")

    year = 2025

    # Fetch from CFBD
    print("\n[1/3] Fetching from CFBD API...")
    cfbd_df = fetch_cfbd_portal(year)

    # Scrape from On3
    print("\n[2/3] Scraping from On3...")
    on3_df = await scrape_on3_portal(year, pages=5)

    # Merge all data
    print("\n[3/3] Merging data...")
    combined_new = pd.concat([cfbd_df, on3_df], ignore_index=True)

    if combined_new.empty:
        logger.warning("No new data fetched from any source")
        print("\nNo new data available. Keeping existing data.")
        return

    # Merge with existing
    final_df = merge_portal_data(existing_df, combined_new)

    # Save updated data
    final_df.to_csv(portal_file, index=False)
    logger.info(f"Saved {len(final_df)} entries to {portal_file}")

    # Generate and save summary
    summary = generate_portal_summary(final_df)
    summary_file = processed_dir / "portal_summary.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PORTAL SUMMARY")
    print("=" * 60)
    print(f"Total entries: {summary['total_entries']}")
    print(f"Still in portal: {summary['still_in_portal']}")
    print(f"Committed: {summary['committed']}")

    if summary['top_destinations']:
        print("\nTop destinations:")
        for school, count in list(summary['top_destinations'].items())[:5]:
            print(f"  {school}: {count}")

    print(f"\nData saved to: {portal_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())
