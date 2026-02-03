"""
Portal IQ Data Update Runner

Unified script to run all data updates. Can be scheduled or run manually.

Usage:
    python scripts/run_updates.py              # Run appropriate updates based on schedule
    python scripts/run_updates.py --daily      # Force run daily updates
    python scripts/run_updates.py --weekly     # Force run weekly updates
    python scripts/run_updates.py --all        # Run all updates
"""

import asyncio
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def should_run_weekly() -> bool:
    """Check if it's time for weekly update (Sunday)."""
    return datetime.now().weekday() == 6  # Sunday


async def run_daily():
    """Run daily portal update."""
    print("\n" + "=" * 60)
    print("RUNNING DAILY PORTAL UPDATE")
    print("=" * 60)

    from scripts.daily_portal_update import main as daily_main
    await daily_main()


def run_weekly():
    """Run weekly stats update."""
    print("\n" + "=" * 60)
    print("RUNNING WEEKLY STATS UPDATE")
    print("=" * 60)

    from scripts.weekly_stats_update import main as weekly_main
    weekly_main()


async def run_all():
    """Run all updates."""
    await run_daily()
    run_weekly()


async def run_scheduled():
    """Run updates based on schedule."""
    print(f"Scheduled update check at {datetime.now()}")

    # Always run daily
    await run_daily()

    # Run weekly on Sundays
    if should_run_weekly():
        run_weekly()
    else:
        print("\nSkipping weekly update (not Sunday)")


def main():
    parser = argparse.ArgumentParser(description="Portal IQ Data Update Runner")
    parser.add_argument('--daily', action='store_true', help='Run daily portal update')
    parser.add_argument('--weekly', action='store_true', help='Run weekly stats update')
    parser.add_argument('--all', action='store_true', help='Run all updates')
    args = parser.parse_args()

    # Ensure logs directory exists
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    if args.daily:
        asyncio.run(run_daily())
    elif args.weekly:
        run_weekly()
    elif args.all:
        asyncio.run(run_all())
    else:
        # Default: run based on schedule
        asyncio.run(run_scheduled())

    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
