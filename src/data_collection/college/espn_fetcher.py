"""
ESPN College Football Data Fetcher

Uses ESPN's unofficial JSON API endpoints to fetch player and team data.
No authentication required - these are public endpoints used by ESPN's website.

Features:
- Team rosters with ESPN player IDs
- Player headshot image URLs
- Team logos and info
- Current depth charts (when available)

Usage:
    from src.data_collection.college.espn_fetcher import ESPNFetcher

    fetcher = ESPNFetcher()
    teams = fetcher.get_all_teams()
    roster = fetcher.get_team_roster("georgia")
    player = fetcher.get_player_details(4426354)
"""

import logging
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ESPNPlayer:
    """ESPN player data."""
    espn_id: int
    name: str
    first_name: str
    last_name: str
    position: str
    jersey: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[int] = None
    class_year: Optional[str] = None
    team: Optional[str] = None
    team_id: Optional[int] = None
    headshot_url: Optional[str] = None
    profile_url: Optional[str] = None


@dataclass
class ESPNTeam:
    """ESPN team data."""
    espn_id: int
    name: str
    abbreviation: str
    display_name: str
    location: str
    conference: Optional[str] = None
    logo_url: Optional[str] = None
    color: Optional[str] = None
    alternate_color: Optional[str] = None


class ESPNFetcher:
    """
    Fetches college football data from ESPN's unofficial API.

    These endpoints are used by ESPN's website and are publicly accessible.
    """

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

    # Rate limiting
    MIN_DELAY = 0.5  # seconds between requests

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the ESPN fetcher.

        Args:
            data_dir: Directory for caching data
        """
        if data_dir is None:
            current = Path(__file__).parent
            while current.parent != current:
                if (current / "data").exists():
                    data_dir = str(current / "data")
                    break
                current = current.parent
            else:
                data_dir = "data"

        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache" / "espn"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        self._last_request = 0

        logger.info(f"ESPNFetcher initialized. Cache dir: {self.cache_dir}")

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.MIN_DELAY:
            time.sleep(self.MIN_DELAY - elapsed)
        self._last_request = time.time()

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a GET request with rate limiting."""
        self._rate_limit()

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"ESPN API error: {e}")
            return None

    def get_all_teams(self, limit: int = 500) -> List[ESPNTeam]:
        """
        Get all FBS college football teams.

        Args:
            limit: Maximum number of teams to fetch

        Returns:
            List of ESPNTeam objects
        """
        logger.info("Fetching all ESPN teams...")

        url = f"{self.BASE_URL}/teams"
        data = self._get(url, params={"limit": limit})

        if not data or "sports" not in data:
            logger.error("Failed to fetch teams")
            return []

        teams = []

        try:
            for sport in data.get("sports", []):
                for league in sport.get("leagues", []):
                    for team_data in league.get("teams", []):
                        team = team_data.get("team", {})

                        logo_url = None
                        logos = team.get("logos", [])
                        if logos:
                            logo_url = logos[0].get("href")

                        teams.append(ESPNTeam(
                            espn_id=int(team.get("id", 0)),
                            name=team.get("name", ""),
                            abbreviation=team.get("abbreviation", ""),
                            display_name=team.get("displayName", ""),
                            location=team.get("location", ""),
                            conference=league.get("name"),
                            logo_url=logo_url,
                            color=team.get("color"),
                            alternate_color=team.get("alternateColor"),
                        ))
        except Exception as e:
            logger.error(f"Error parsing teams: {e}")

        logger.info(f"Found {len(teams)} teams")
        return teams

    def get_team_by_name(self, name: str) -> Optional[ESPNTeam]:
        """
        Find a team by name (fuzzy match).

        Args:
            name: Team name to search for

        Returns:
            ESPNTeam if found, None otherwise
        """
        teams = self.get_all_teams()
        name_lower = name.lower()

        for team in teams:
            if (name_lower in team.name.lower() or
                name_lower in team.display_name.lower() or
                name_lower == team.abbreviation.lower()):
                return team

        return None

    def get_team_roster(self, team_id: int) -> List[ESPNPlayer]:
        """
        Get full roster for a team.

        Args:
            team_id: ESPN team ID

        Returns:
            List of ESPNPlayer objects
        """
        logger.info(f"Fetching roster for team {team_id}...")

        url = f"{self.BASE_URL}/teams/{team_id}/roster"
        data = self._get(url)

        if not data:
            logger.error(f"Failed to fetch roster for team {team_id}")
            return []

        players = []
        team_name = data.get("team", {}).get("displayName", "Unknown")

        try:
            for group in data.get("athletes", []):
                position_name = group.get("position", "Unknown")

                for athlete in group.get("items", []):
                    headshot_url = None
                    if athlete.get("headshot"):
                        headshot_url = athlete["headshot"].get("href")

                    players.append(ESPNPlayer(
                        espn_id=int(athlete.get("id", 0)),
                        name=athlete.get("displayName", ""),
                        first_name=athlete.get("firstName", ""),
                        last_name=athlete.get("lastName", ""),
                        position=athlete.get("position", {}).get("abbreviation", position_name),
                        jersey=athlete.get("jersey"),
                        height=athlete.get("displayHeight"),
                        weight=athlete.get("weight"),
                        class_year=athlete.get("experience", {}).get("displayValue"),
                        team=team_name,
                        team_id=team_id,
                        headshot_url=headshot_url,
                        profile_url=f"https://www.espn.com/college-football/player/_/id/{athlete.get('id')}",
                    ))
        except Exception as e:
            logger.error(f"Error parsing roster: {e}")

        logger.info(f"Found {len(players)} players on roster")
        return players

    def get_player_details(self, player_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed info for a specific player.

        Args:
            player_id: ESPN player ID

        Returns:
            Dict with player details or None
        """
        url = f"{self.BASE_URL}/athletes/{player_id}"
        data = self._get(url)

        if not data or "athlete" not in data:
            return None

        athlete = data["athlete"]

        # Extract stats if available
        stats = {}
        for stat_group in athlete.get("statistics", []):
            category = stat_group.get("type", {}).get("abbreviation", "")
            for stat in stat_group.get("stats", []):
                key = f"{category}_{stat.get('name', '')}".lower().replace(" ", "_")
                stats[key] = stat.get("value")

        return {
            "espn_id": player_id,
            "name": athlete.get("displayName"),
            "first_name": athlete.get("firstName"),
            "last_name": athlete.get("lastName"),
            "position": athlete.get("position", {}).get("abbreviation"),
            "jersey": athlete.get("jersey"),
            "height": athlete.get("displayHeight"),
            "weight": athlete.get("weight"),
            "birthplace": athlete.get("birthPlace", {}).get("city"),
            "class_year": athlete.get("experience", {}).get("displayValue"),
            "team": athlete.get("team", {}).get("displayName"),
            "headshot_url": athlete.get("headshot", {}).get("href"),
            "stats": stats,
        }

    def fetch_all_rosters(self, save: bool = True) -> pd.DataFrame:
        """
        Fetch rosters for all FBS teams.

        Args:
            save: Whether to save to CSV

        Returns:
            DataFrame with all players
        """
        logger.info("Fetching all FBS rosters from ESPN...")

        teams = self.get_all_teams()
        all_players = []

        for i, team in enumerate(teams):
            logger.info(f"[{i+1}/{len(teams)}] Fetching {team.display_name}...")

            try:
                roster = self.get_team_roster(team.espn_id)
                all_players.extend(roster)
            except Exception as e:
                logger.warning(f"Failed to fetch {team.display_name}: {e}")
                continue

        logger.info(f"Total players fetched: {len(all_players)}")

        # Convert to DataFrame
        df = pd.DataFrame([asdict(p) for p in all_players])

        if save and not df.empty:
            output_path = self.data_dir / "raw" / "espn_rosters.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved to {output_path}")

        return df

    def fetch_teams_data(self, save: bool = True) -> pd.DataFrame:
        """
        Fetch all team data and save to CSV.

        Args:
            save: Whether to save to CSV

        Returns:
            DataFrame with all teams
        """
        teams = self.get_all_teams()
        df = pd.DataFrame([asdict(t) for t in teams])

        if save and not df.empty:
            output_path = self.data_dir / "raw" / "espn_teams.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} teams to {output_path}")

        return df

    def enrich_players_with_espn(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich a player DataFrame with ESPN IDs and headshots.

        Matches by name and team using fuzzy matching.

        Args:
            players_df: DataFrame with 'player_name' and 'school' columns

        Returns:
            DataFrame with added ESPN columns
        """
        logger.info("Enriching player data with ESPN info...")

        # Load or fetch ESPN rosters
        espn_file = self.data_dir / "raw" / "espn_rosters.csv"
        if espn_file.exists():
            espn_df = pd.read_csv(espn_file)
            logger.info(f"Loaded {len(espn_df)} ESPN players from cache")
        else:
            espn_df = self.fetch_all_rosters(save=True)

        if espn_df.empty:
            logger.warning("No ESPN data available")
            return players_df

        # Create lookup keys
        espn_df['lookup_key'] = (
            espn_df['name'].str.lower().str.strip() + '|' +
            espn_df['team'].str.lower().str.strip()
        )

        # Try to match
        result_df = players_df.copy()
        result_df['espn_id'] = None
        result_df['espn_headshot_url'] = None
        result_df['espn_profile_url'] = None

        name_col = 'player_name' if 'player_name' in result_df.columns else 'name'
        school_col = 'school' if 'school' in result_df.columns else 'team'

        if name_col not in result_df.columns or school_col not in result_df.columns:
            logger.warning(f"Missing required columns: need '{name_col}' and '{school_col}'")
            return players_df

        result_df['lookup_key'] = (
            result_df[name_col].str.lower().str.strip() + '|' +
            result_df[school_col].str.lower().str.strip()
        )

        # Exact match first
        espn_lookup = espn_df.set_index('lookup_key')[['espn_id', 'headshot_url', 'profile_url']].to_dict('index')

        matched = 0
        for idx, row in result_df.iterrows():
            key = row['lookup_key']
            if key in espn_lookup:
                result_df.at[idx, 'espn_id'] = espn_lookup[key]['espn_id']
                result_df.at[idx, 'espn_headshot_url'] = espn_lookup[key]['headshot_url']
                result_df.at[idx, 'espn_profile_url'] = espn_lookup[key]['profile_url']
                matched += 1

        logger.info(f"Matched {matched}/{len(result_df)} players with ESPN data")

        # Clean up
        result_df = result_df.drop(columns=['lookup_key'], errors='ignore')

        return result_df


def main():
    """Example usage."""
    print("ESPN College Football Data Fetcher")
    print("=" * 50)

    fetcher = ESPNFetcher()

    # Fetch teams
    print("\n[1/3] Fetching teams...")
    teams_df = fetcher.fetch_teams_data(save=True)
    print(f"Found {len(teams_df)} teams")

    # Show sample teams
    if not teams_df.empty:
        print("\nSample teams:")
        for _, team in teams_df.head(5).iterrows():
            print(f"  {team['display_name']} ({team['abbreviation']}) - ID: {team['espn_id']}")

    # Fetch a sample roster
    print("\n[2/3] Fetching Georgia roster as example...")
    georgia = fetcher.get_team_by_name("Georgia")
    if georgia:
        roster = fetcher.get_team_roster(georgia.espn_id)
        print(f"Found {len(roster)} players")

        if roster:
            print("\nSample players:")
            for p in roster[:5]:
                headshot = "✓" if p.headshot_url else "✗"
                print(f"  {p.name} ({p.position}) - ESPN ID: {p.espn_id} [headshot: {headshot}]")

    print("\n[3/3] Done!")
    print("\nTo fetch all rosters, run: fetcher.fetch_all_rosters()")


if __name__ == "__main__":
    main()
