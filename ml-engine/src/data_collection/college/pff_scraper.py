"""
PFF Scraper - Player Grades Extraction

Scrapes PFF college player grades including:
- Overall grades (offense/defense)
- Position-specific grades (pass block, run block, coverage, etc.)
- Snap counts
- Advanced metrics (pressures, hurries, missed tackles)

Requires PFF Premium account. Authentication via:
1. Interactive browser login (first time)
2. Cookie persistence (subsequent runs)
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Rotate user agents to avoid detection (Updated Feb 2026)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# Modern browser headers
BROWSER_HEADERS = {
    "sec-ch-ua": '"Chromium";v="131", "Google Chrome";v="131", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
}


@dataclass
class PFFPlayerGrade:
    """PFF player grade data structure."""
    # Identity
    player_name: str
    team: str
    position: str
    jersey_number: Optional[int] = None

    # Overall Grades (0-100)
    pff_overall: float = 0.0
    pff_offense: Optional[float] = None
    pff_defense: Optional[float] = None

    # Offensive Grades
    pff_pass_block: Optional[float] = None
    pff_run_block: Optional[float] = None
    pff_receiving: Optional[float] = None
    pff_rushing: Optional[float] = None
    pff_passing: Optional[float] = None

    # Defensive Grades
    pff_pass_rush: Optional[float] = None
    pff_run_defense: Optional[float] = None
    pff_coverage: Optional[float] = None
    pff_tackling: Optional[float] = None

    # Snap Counts
    total_snaps: int = 0
    pass_snaps: Optional[int] = None
    run_snaps: Optional[int] = None

    # Advanced Metrics
    pressures: Optional[int] = None
    pressures_allowed: Optional[int] = None
    hurries: Optional[int] = None
    hits: Optional[int] = None
    missed_tackles: Optional[int] = None
    targets_allowed: Optional[int] = None
    completions_allowed: Optional[int] = None
    yards_allowed: Optional[int] = None
    tds_allowed: Optional[int] = None
    ints: Optional[int] = None
    pbus: Optional[int] = None

    # Metadata
    season: int = 2024
    scraped_at: str = ""

    def __post_init__(self):
        if not self.scraped_at:
            self.scraped_at = datetime.utcnow().isoformat()


class PFFScraper:
    """Scrapes PFF using direct HTTP requests with saved cookies."""

    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent / "data"

        self.data_dir = data_dir
        self.cookies_path = data_dir / "cache" / "pff_cookies.json"
        self.cookies_path.parent.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self._rotate_user_agent()
        self._load_cookies()
        self.request_count = 0
        self.is_logged_in = False

    def _rotate_user_agent(self):
        """Rotate user agent and headers to avoid detection."""
        ua = random.choice(USER_AGENTS)
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        headers.update(BROWSER_HEADERS)
        self.session.headers.update(headers)

    def _load_cookies(self):
        """Load cookies from file."""
        if self.cookies_path.exists():
            try:
                with open(self.cookies_path, 'r') as f:
                    cookies = json.load(f)
                for cookie in cookies:
                    self.session.cookies.set(
                        cookie.get('name', ''),
                        cookie.get('value', ''),
                        domain=cookie.get('domain', '.pff.com'),
                    )
                print(f"Loaded {len(cookies)} PFF cookies")
                self.is_logged_in = True
            except Exception as e:
                print(f"Error loading cookies: {e}")
        else:
            print(f"No PFF cookies found at {self.cookies_path}")
            print("Run interactive login first to save cookies.")

    def _save_cookies(self):
        """Save current cookies to file."""
        cookies = []
        for cookie in self.session.cookies:
            cookies.append({
                'name': cookie.name,
                'value': cookie.value,
                'domain': cookie.domain,
                'path': cookie.path,
            })
        with open(self.cookies_path, 'w') as f:
            json.dump(cookies, f, indent=2)
        print(f"Saved {len(cookies)} cookies to {self.cookies_path}")

    def login_interactive(self):
        """
        Open browser for interactive login.
        User logs in manually, then cookies are captured.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("Playwright not installed. Install with: pip install playwright && playwright install")
            return False

        print("\n" + "="*60)
        print("PFF Interactive Login")
        print("="*60)
        print("1. A browser window will open to PFF login page")
        print("2. Log in with your PFF account")
        print("3. Once logged in, close the browser window")
        print("="*60 + "\n")

        with sync_playwright() as p:
            # Use Edge with user's existing profile if available
            edge_path = os.path.expanduser("~") + "/AppData/Local/Microsoft/Edge/User Data"

            try:
                browser = p.chromium.launch(
                    headless=False,
                    channel="msedge",
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except Exception:
                browser = p.chromium.launch(headless=False)

            context = browser.new_context()
            page = context.new_page()

            # Navigate to PFF login
            page.goto("https://www.pff.com/login")

            print("Waiting for login... (close browser when done)")
            print("Looking for URL change to indicate successful login...")

            # Wait for user to complete login (URL changes from /login)
            try:
                page.wait_for_url(lambda url: "/login" not in url, timeout=300000)  # 5 min timeout
                print("Login detected!")

                # Give a moment for cookies to settle
                time.sleep(2)

                # Capture cookies
                cookies = context.cookies()

                # Save cookies for requests session
                pff_cookies = []
                for cookie in cookies:
                    if 'pff.com' in cookie.get('domain', ''):
                        pff_cookies.append({
                            'name': cookie['name'],
                            'value': cookie['value'],
                            'domain': cookie['domain'],
                            'path': cookie.get('path', '/'),
                        })
                        # Also add to requests session
                        self.session.cookies.set(
                            cookie['name'],
                            cookie['value'],
                            domain=cookie['domain'],
                        )

                with open(self.cookies_path, 'w') as f:
                    json.dump(pff_cookies, f, indent=2)

                print(f"Saved {len(pff_cookies)} PFF cookies")
                self.is_logged_in = True

            except Exception as e:
                print(f"Login timeout or error: {e}")
                return False
            finally:
                browser.close()

        return True

    def fetch_page(self, url: str, max_retries: int = 3) -> dict:
        """Fetch a page and extract JSON data."""
        self.request_count += 1

        # Rotate user agent periodically
        if self.request_count % 10 == 0:
            self._rotate_user_agent()

        # Add referer
        self.session.headers["Referer"] = "https://www.pff.com/"

        for attempt in range(max_retries):
            try:
                # Random delay between requests
                if attempt > 0 or self.request_count > 1:
                    delay = random.uniform(3, 6)
                    time.sleep(delay)

                response = self.session.get(url, timeout=30)

                if response.status_code == 429:
                    print(f"  Rate limited, waiting...")
                    time.sleep(random.uniform(15, 30))
                    continue

                if response.status_code == 403:
                    print(f"  Forbidden - may need to re-login")
                    self._rotate_user_agent()
                    time.sleep(random.uniform(5, 10))
                    continue

                if response.status_code == 401:
                    print(f"  Unauthorized - session expired, need to re-login")
                    self.is_logged_in = False
                    return {}

                if response.status_code == 404:
                    return {}

                response.raise_for_status()

                # Try to extract __NEXT_DATA__ JSON (if PFF uses Next.js)
                match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', response.text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))

                # Try other common patterns
                # Pattern 2: window.__INITIAL_STATE__
                match = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.+?});', response.text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))

                # Pattern 3: data-page attribute (React/Inertia)
                match = re.search(r'data-page="([^"]+)"', response.text)
                if match:
                    import html
                    return json.loads(html.unescape(match.group(1)))

                # If no JSON found, return raw HTML for DOM parsing
                return {"_html": response.text, "_url": url}

            except requests.exceptions.Timeout:
                print(f"  Timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                print(f"  Request error: {e}")
            except json.JSONDecodeError as e:
                print(f"  JSON parse error: {e}")
            except Exception as e:
                print(f"  Error fetching {url}: {e}")

        return {}

    def extract_player_grades(self, data: dict, team: str) -> List[PFFPlayerGrade]:
        """Extract player grades from page data."""
        grades = []

        # If we got raw HTML, try DOM parsing
        if "_html" in data:
            return self._extract_grades_from_html(data["_html"], team)

        # Try various JSON paths (PFF's structure may differ)
        props = data.get('props', {}).get('pageProps', {})

        # Common PFF data paths
        player_list = (
            props.get('players', []) or
            props.get('grades', []) or
            props.get('roster', []) or
            props.get('data', {}).get('players', []) or
            []
        )

        for player in player_list:
            try:
                grade = self._parse_player_grade(player, team)
                if grade:
                    grades.append(grade)
            except Exception as e:
                continue

        return grades

    def _parse_player_grade(self, player: dict, team: str) -> Optional[PFFPlayerGrade]:
        """Parse a single player's grade data."""
        name = player.get('name') or player.get('playerName') or player.get('full_name')
        if not name:
            return None

        position = player.get('position') or player.get('pos') or ''

        # Extract grades - PFF typically uses 'grade' or 'grades' object
        grades_data = player.get('grades', {}) or player.get('grade', {}) or player

        grade = PFFPlayerGrade(
            player_name=name,
            team=team,
            position=position,
            jersey_number=player.get('jersey') or player.get('number'),

            # Overall grades
            pff_overall=self._safe_float(grades_data.get('overall') or grades_data.get('pff_grade') or player.get('overall_grade')),
            pff_offense=self._safe_float(grades_data.get('offense') or grades_data.get('offense_grade')),
            pff_defense=self._safe_float(grades_data.get('defense') or grades_data.get('defense_grade')),

            # Offensive grades
            pff_pass_block=self._safe_float(grades_data.get('pass_block') or grades_data.get('passBlock')),
            pff_run_block=self._safe_float(grades_data.get('run_block') or grades_data.get('runBlock')),
            pff_receiving=self._safe_float(grades_data.get('receiving') or grades_data.get('recv')),
            pff_rushing=self._safe_float(grades_data.get('rushing') or grades_data.get('rush')),
            pff_passing=self._safe_float(grades_data.get('passing') or grades_data.get('pass')),

            # Defensive grades
            pff_pass_rush=self._safe_float(grades_data.get('pass_rush') or grades_data.get('passRush')),
            pff_run_defense=self._safe_float(grades_data.get('run_defense') or grades_data.get('runDefense')),
            pff_coverage=self._safe_float(grades_data.get('coverage') or grades_data.get('cov')),
            pff_tackling=self._safe_float(grades_data.get('tackling') or grades_data.get('tackle')),

            # Snap counts
            total_snaps=self._safe_int(player.get('snaps') or player.get('total_snaps') or player.get('snap_count')),
            pass_snaps=self._safe_int(player.get('pass_snaps')),
            run_snaps=self._safe_int(player.get('run_snaps')),

            # Advanced metrics
            pressures=self._safe_int(player.get('pressures') or player.get('pressure')),
            pressures_allowed=self._safe_int(player.get('pressures_allowed')),
            hurries=self._safe_int(player.get('hurries') or player.get('hurry')),
            hits=self._safe_int(player.get('hits') or player.get('qb_hits')),
            missed_tackles=self._safe_int(player.get('missed_tackles') or player.get('missedTackles')),
            targets_allowed=self._safe_int(player.get('targets') or player.get('targets_allowed')),
            completions_allowed=self._safe_int(player.get('completions_allowed') or player.get('receptions_allowed')),
            yards_allowed=self._safe_int(player.get('yards_allowed')),
            tds_allowed=self._safe_int(player.get('tds_allowed') or player.get('touchdowns_allowed')),
            ints=self._safe_int(player.get('interceptions') or player.get('ints')),
            pbus=self._safe_int(player.get('pass_breakups') or player.get('pbus') or player.get('pbu')),

            season=2024,
        )

        return grade

    def _extract_grades_from_html(self, html: str, team: str) -> List[PFFPlayerGrade]:
        """Fallback: Extract grades from HTML using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("BeautifulSoup not installed for HTML parsing")
            return []

        grades = []
        soup = BeautifulSoup(html, 'html.parser')

        # Look for player grade rows/cards (PFF-specific selectors)
        # These selectors will need to be adjusted based on actual PFF page structure
        player_rows = soup.select('tr[data-player], .player-card, .grade-row, [class*="PlayerRow"]')

        for row in player_rows:
            try:
                name_el = row.select_one('.player-name, [class*="name"], td:first-child')
                grade_el = row.select_one('.overall-grade, [class*="grade"], .pff-grade')
                pos_el = row.select_one('.position, [class*="pos"]')

                if name_el:
                    name = name_el.get_text(strip=True)
                    overall = self._safe_float(grade_el.get_text(strip=True)) if grade_el else 0.0
                    position = pos_el.get_text(strip=True) if pos_el else ''

                    grades.append(PFFPlayerGrade(
                        player_name=name,
                        team=team,
                        position=position,
                        pff_overall=overall,
                        season=2024,
                    ))
            except Exception:
                continue

        return grades

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_int(self, value) -> Optional[int]:
        """Safely convert to int."""
        if value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None

    def scrape_team_grades(self, team_name: str, team_url: str) -> List[PFFPlayerGrade]:
        """Scrape all player grades for a team."""
        print(f"  Scraping {team_name}...")

        data = self.fetch_page(team_url)
        if not data:
            print(f"    No data for {team_name}")
            return []

        grades = self.extract_player_grades(data, team_name)
        print(f"    Found {len(grades)} player grades")

        return grades

    def scrape_all_teams(self, teams: List[Dict[str, str]], delay_between: float = 4.0) -> pd.DataFrame:
        """
        Scrape grades for multiple teams.

        Args:
            teams: List of dicts with 'name' and 'url' keys
            delay_between: Seconds to wait between teams

        Returns:
            DataFrame with all player grades
        """
        all_grades = []

        print(f"\nScraping {len(teams)} teams...")
        print("="*50)

        for i, team in enumerate(teams):
            team_name = team.get('name', 'Unknown')
            team_url = team.get('url', '')

            if not team_url:
                continue

            print(f"\n[{i+1}/{len(teams)}] {team_name}")

            grades = self.scrape_team_grades(team_name, team_url)
            all_grades.extend(grades)

            # Delay between teams
            if i < len(teams) - 1:
                time.sleep(random.uniform(delay_between - 1, delay_between + 2))

        # Convert to DataFrame
        if all_grades:
            df = pd.DataFrame([asdict(g) for g in all_grades])
            print(f"\n{'='*50}")
            print(f"Total: {len(df)} player grades from {len(teams)} teams")
            return df

        return pd.DataFrame()

    def get_fbs_team_urls(self) -> List[Dict[str, str]]:
        """
        Generate PFF team URLs for all FBS teams.
        Uses CFBD to get team list, then constructs PFF URLs.
        """
        try:
            import cfbd
            from cfbd.rest import ApiException

            api_key = os.getenv("CFBD_API_KEY")
            if not api_key:
                print("CFBD_API_KEY not set, using fallback team list")
                return self._get_fallback_teams()

            config = cfbd.Configuration(access_token=api_key)
            api = cfbd.TeamsApi(cfbd.ApiClient(config))

            teams = api.get_fbs_teams(year=2024)
            team_urls = []

            for team in teams:
                # Convert team name to PFF URL slug
                # "Ohio State" -> "ohio-state", "Buckeyes" -> "buckeyes"
                school_slug = team.school.lower().replace(" ", "-").replace("&", "and")
                mascot_slug = (team.mascot or "").lower().replace(" ", "-")

                # Remove special characters
                import re
                school_slug = re.sub(r'[^a-z0-9-]', '', school_slug)
                mascot_slug = re.sub(r'[^a-z0-9-]', '', mascot_slug)

                # PFF URL format: /college/teams/{school}-{mascot}/grades
                url = f"https://www.pff.com/college/teams/{school_slug}-{mascot_slug}/grades"

                team_urls.append({
                    "name": team.school,
                    "url": url,
                    "conference": team.conference,
                })

            print(f"Generated URLs for {len(team_urls)} FBS teams")
            return team_urls

        except Exception as e:
            print(f"Error getting FBS teams: {e}")
            return self._get_fallback_teams()

    def _get_fallback_teams(self) -> List[Dict[str, str]]:
        """Fallback list of major teams if CFBD unavailable."""
        return [
            {"name": "Alabama", "url": "https://www.pff.com/college/teams/alabama-crimson-tide/grades"},
            {"name": "Ohio State", "url": "https://www.pff.com/college/teams/ohio-state-buckeyes/grades"},
            {"name": "Georgia", "url": "https://www.pff.com/college/teams/georgia-bulldogs/grades"},
            {"name": "Michigan", "url": "https://www.pff.com/college/teams/michigan-wolverines/grades"},
            {"name": "Texas", "url": "https://www.pff.com/college/teams/texas-longhorns/grades"},
            {"name": "USC", "url": "https://www.pff.com/college/teams/usc-trojans/grades"},
            {"name": "Oregon", "url": "https://www.pff.com/college/teams/oregon-ducks/grades"},
            {"name": "Penn State", "url": "https://www.pff.com/college/teams/penn-state-nittany-lions/grades"},
            {"name": "LSU", "url": "https://www.pff.com/college/teams/lsu-tigers/grades"},
            {"name": "Florida", "url": "https://www.pff.com/college/teams/florida-gators/grades"},
        ]


def main():
    """Test the PFF scraper."""
    scraper = PFFScraper()

    if not scraper.is_logged_in:
        print("Not logged in. Starting interactive login...")
        scraper.login_interactive()

    if scraper.is_logged_in:
        # Test with one team
        test_teams = scraper._get_fallback_teams()[:2]
        df = scraper.scrape_all_teams(test_teams)

        if not df.empty:
            output_path = scraper.data_dir / "processed" / "pff_player_grades_test.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")
            print(df.head(10))


if __name__ == "__main__":
    main()
