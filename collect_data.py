#!/usr/bin/env python3
"""
Portal IQ Data Collection Script
Fetches real data from CFBD API using direct HTTP requests.

Usage:
    python collect_data.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Loaded .env file")
except ImportError:
    print("Installing python-dotenv...")
    os.system(f"{sys.executable} -m pip install python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

import requests
import pandas as pd

# Check for CFBD API key
CFBD_API_KEY = os.getenv("CFBD_API_KEY")
if not CFBD_API_KEY or CFBD_API_KEY == "YOUR_CFBD_KEY_HERE":
    print("[ERROR] CFBD_API_KEY not set in .env file")
    print("  Get your free key at: https://collegefootballdata.com/key")
    sys.exit(1)

print(f"[OK] CFBD API key found (starts with: {CFBD_API_KEY[:8]}...)")

# API Configuration
CFBD_BASE_URL = "https://api.collegefootballdata.com"
CFBD_HEADERS = {
    "Authorization": f"Bearer {CFBD_API_KEY}",
    "Accept": "application/json"
}

# Create data directories
for dir_name in ["data/cache", "data/raw", "data/processed"]:
    Path(dir_name).mkdir(parents=True, exist_ok=True)
print("[OK] Created data directories")

def cfbd_get(endpoint: str, params: dict = None) -> list:
    """Make a GET request to CFBD API."""
    url = f"{CFBD_BASE_URL}{endpoint}"
    try:
        response = requests.get(url, headers=CFBD_HEADERS, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"    API Error: {e}")
        return []

def save_data(data: list, filename: str, subdir: str = "raw"):
    """Save data to CSV and JSON."""
    if not data:
        return

    df = pd.DataFrame(data)
    csv_path = Path(f"data/{subdir}/{filename}.csv")
    json_path = Path(f"data/{subdir}/{filename}.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"    Saved {len(data)} records to {csv_path}")

print("\n" + "="*60)
print("Starting CFBD Data Collection")
print("="*60)

# ============================================================
# 1. Player Stats
# ============================================================
print("\n[1/6] Collecting Player Season Stats...")
all_player_stats = []
for year in range(2022, 2026):
    print(f"    Fetching {year}...")
    stats = cfbd_get("/stats/player/season", {"year": year})
    if stats:
        for s in stats:
            s["season"] = year
        all_player_stats.extend(stats)
        print(f"    Got {len(stats)} player stat records for {year}")

save_data(all_player_stats, "player_season_stats")

# ============================================================
# 2. Team Rosters
# ============================================================
print("\n[2/6] Collecting Team Rosters...")
all_rosters = []
# Get list of FBS teams first
teams = cfbd_get("/teams/fbs", {"year": 2024})
team_names = [t.get("school") for t in teams if t.get("school")]
print(f"    Found {len(team_names)} FBS teams")

for year in [2024, 2025]:
    print(f"    Fetching rosters for {year}...")
    for team in team_names[:50]:  # Limit to 50 teams to avoid rate limits
        roster = cfbd_get("/roster", {"year": year, "team": team})
        if roster:
            for player in roster:
                player["team"] = team
                player["season"] = year
            all_rosters.extend(roster)

print(f"    Total roster entries: {len(all_rosters)}")
save_data(all_rosters, "team_rosters")

# ============================================================
# 3. Recruiting Rankings
# ============================================================
print("\n[3/6] Collecting Recruiting Rankings...")
all_recruits = []
for year in range(2020, 2026):
    print(f"    Fetching {year} recruiting class...")
    recruits = cfbd_get("/recruiting/players", {"year": year})
    if recruits:
        all_recruits.extend(recruits)
        print(f"    Got {len(recruits)} recruits for {year}")

save_data(all_recruits, "recruiting_rankings")

# ============================================================
# 4. Transfer Portal
# ============================================================
print("\n[4/6] Collecting Transfer Portal Data...")
all_transfers = []
for year in range(2022, 2026):
    print(f"    Fetching {year} transfers...")
    transfers = cfbd_get("/player/portal", {"year": year})
    if transfers:
        all_transfers.extend(transfers)
        print(f"    Got {len(transfers)} transfers for {year}")

save_data(all_transfers, "transfer_portal")

# ============================================================
# 5. Team Talent Rankings
# ============================================================
print("\n[5/6] Collecting Team Talent Rankings...")
all_talent = []
for year in range(2020, 2026):
    talent = cfbd_get("/talent", {"year": year})
    if talent:
        for t in talent:
            t["year"] = year
        all_talent.extend(talent)

save_data(all_talent, "team_talent")
print(f"    Got {len(all_talent)} team talent records")

# ============================================================
# 6. Draft Picks
# ============================================================
print("\n[6/6] Collecting NFL Draft Data...")
all_draft = []
for year in range(2020, 2025):
    picks = cfbd_get("/draft/picks", {"year": year})
    if picks:
        all_draft.extend(picks)
        print(f"    Got {len(picks)} draft picks for {year}")

save_data(all_draft, "nfl_draft_picks")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("Data Collection Complete!")
print("="*60)

# Create summary
summary = {
    "collection_date": datetime.now().isoformat(),
    "datasets": {
        "player_stats": len(all_player_stats),
        "rosters": len(all_rosters),
        "recruits": len(all_recruits),
        "transfers": len(all_transfers),
        "team_talent": len(all_talent),
        "draft_picks": len(all_draft)
    }
}

with open("data/collection_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nData collected:")
for name, count in summary["datasets"].items():
    print(f"  - {name}: {count:,} records")

print("\nFiles saved to:")
print("  - data/raw/    (CSV and JSON files)")
print("  - data/collection_summary.json")

print("\nNext steps:")
print("  1. Start API locally: python -m uvicorn src.api.app:app --reload")
print("  2. Or deploy to Railway with CFBD_API_KEY env var")
print("  3. Your frontend will now fetch real data!")
