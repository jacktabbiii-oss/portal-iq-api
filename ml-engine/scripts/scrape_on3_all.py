"""
On3 Complete Data Scraper

Scrapes all NIL data from On3 including:
- College NIL rankings
- High School NIL rankings (all sports)
- Transfer Portal
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Rotate user agents to avoid detection (Updated Feb 2026)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# Modern browser headers to appear more legitimate
BROWSER_HEADERS = {
    "sec-ch-ua": '"Chromium";v="131", "Google Chrome";v="131", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
}

# All On3 URLs to scrape
ON3_PAGES = {
    # FOOTBALL FOCUSED
    # College Football NIL
    "college_nil_100": "https://www.on3.com/nil/rankings/player/nil-100/",
    "college_football": "https://www.on3.com/nil/rankings/player/college/football/",

    # Transfer Portal Football NIL
    "portal_football_nil": "https://www.on3.com/nil/rankings/player/college/football/?transfer-portal=true",

    # High School Football
    "hs_football": "https://www.on3.com/nil/rankings/player/high-school/football/",
}

# Transfer Portal pages - INFINITE PAGINATION (scrape all pages)
# Current year (2026)
PORTAL_PAGES = {
    "portal_wire_football_2026": "https://www.on3.com/transfer-portal/wire/football/",
    "portal_industry_football_2026": "https://www.on3.com/transfer-portal/industry/football/",
    "portal_team_rankings_2026": "https://www.on3.com/transfer-portal/team-rankings/football/2026/",
}

# Historical portal data - for ML training
PORTAL_PAGES_HISTORICAL = {
    # 2025 Industry Rankings (~4,040 entries)
    "portal_industry_football_2025": "https://www.on3.com/transfer-portal/industry/football/2025/",
    # 2024 Industry Rankings (~3,207 entries)
    "portal_industry_football_2024": "https://www.on3.com/transfer-portal/industry/football/2024/",

    # 2025 Team Rankings (which teams got the best portal classes)
    "portal_team_rankings_2025": "https://www.on3.com/transfer-portal/team-rankings/football/2025/",
    # 2024 Team Rankings
    "portal_team_rankings_2024": "https://www.on3.com/transfer-portal/team-rankings/football/2024/",

    # 2025 Portal Tracker (commitment tracking - 300 entries)
    "portal_tracker_2025": "https://www.on3.com/transfer-portal/tracker/football/2025/",
    # 2024 Portal Tracker (300 entries)
    "portal_tracker_2024": "https://www.on3.com/transfer-portal/tracker/football/2024/",
}

# OTHER SPORTS (commented out for now - uncomment to include)
OTHER_SPORTS = {
    # "college_basketball": "https://www.on3.com/nil/rankings/player/college/basketball/",
    # "college_womens_basketball": "https://www.on3.com/nil/rankings/player/college/womens-basketball/",
    # "college_volleyball": "https://www.on3.com/nil/rankings/player/college/volleyball/",
    # "college_gymnastics": "https://www.on3.com/nil/rankings/player/college/gymnastics/",
    # "portal_basketball": "https://www.on3.com/nil/rankings/player/college/basketball/?transfer-portal=true",
}


class On3Scraper:
    """Scrapes On3 using direct HTTP requests with saved cookies."""

    def __init__(self):
        self.cookies_path = project_root / "data" / "cache" / "on3_cookies.json"
        self.session = requests.Session()
        self._rotate_user_agent()
        self._load_cookies()
        self.request_count = 0

    def _rotate_user_agent(self):
        """Rotate user agent and headers to avoid detection."""
        ua = random.choice(USER_AGENTS)
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "max-age=0",
        }
        # Add modern browser security headers
        headers.update(BROWSER_HEADERS)
        self.session.headers.update(headers)

    def _load_cookies(self):
        """Load cookies from file."""
        if self.cookies_path.exists():
            with open(self.cookies_path, 'r') as f:
                cookies = json.load(f)
            for cookie in cookies:
                self.session.cookies.set(
                    cookie.get('name', ''),
                    cookie.get('value', ''),
                    domain=cookie.get('domain', '.on3.com'),
                )
            print(f"Loaded {len(cookies)} cookies")
        else:
            print(f"WARNING: No cookies found at {self.cookies_path}")
            print("Run: python scripts/on3_api_scraper.py --export-cookies")

    def fetch_page(self, url: str, max_retries: int = 3) -> dict:
        """Fetch a page and extract __NEXT_DATA__ JSON with retry logic."""
        self.request_count += 1

        # Rotate user agent periodically
        if self.request_count % 20 == 0:
            self._rotate_user_agent()

        # Add referer to look more natural
        self.session.headers["Referer"] = "https://www.on3.com/"

        for attempt in range(max_retries):
            try:
                # Random delay to appear more human-like
                if attempt > 0:
                    delay = random.uniform(2, 5)
                    time.sleep(delay)

                response = self.session.get(url, timeout=30)

                if response.status_code == 429:
                    # Rate limited - wait longer
                    print(f"  Rate limited, waiting...")
                    time.sleep(random.uniform(10, 20))
                    continue

                if response.status_code == 403:
                    # Forbidden - rotate UA and retry
                    print(f"  Forbidden, rotating user agent...")
                    self._rotate_user_agent()
                    time.sleep(random.uniform(3, 6))
                    continue

                if response.status_code == 404:
                    # Page not found - don't retry
                    return {}

                response.raise_for_status()

                # Extract __NEXT_DATA__ JSON
                match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', response.text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
                else:
                    print(f"  No __NEXT_DATA__ found in {len(response.text)} bytes")
                    return {}

            except requests.exceptions.Timeout:
                print(f"  Timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                print(f"  Request error on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"  Error fetching {url}: {e}")

        return {}

    def extract_nil_players(self, data: dict, category: str = "college") -> list:
        """Extract player data from __NEXT_DATA__ JSON."""
        players = []

        # Navigate to NIL rankings list (path may vary by page type)
        props = data.get('props', {}).get('pageProps', {})
        nil_list = (
            props.get('nilRankings', {}).get('list', []) or
            props.get('rankings', {}).get('list', []) or
            props.get('players', []) or
            []
        )

        for item in nil_list:
            try:
                person = item.get('person', {})
                valuation = item.get('valuation', {})
                commit_status = person.get('commitStatus', {}) or {}
                committed_org = commit_status.get('committedOrganization', {}) or {}
                rating = person.get('rating', {}) or {}

                # Get school name
                school = committed_org.get('name', '')
                if not school:
                    school = person.get('highSchoolName', '') or 'Unknown'

                players.append({
                    'name': person.get('name', 'Unknown'),
                    'school': school,
                    'position': person.get('positionAbbreviation', '') or rating.get('positionAbbr', 'Unknown'),
                    'nil_valuation': valuation.get('valuation', 0) or 0,
                    'nil_rank': valuation.get('rank'),
                    'class_year': commit_status.get('classRank') or person.get('classYear'),
                    'recruiting_stars': rating.get('stars'),
                    'recruiting_rating': rating.get('rating'),
                    'national_rank': rating.get('nationalRank'),
                    'state': rating.get('stateAbbr', ''),
                    'followers': valuation.get('followers', 0) or 0,
                    'headshot_url': person.get('defaultAssetUrl', ''),
                    'profile_url': f"https://www.on3.com/db/{person.get('slug', '')}/",
                    'category': category,
                })
            except Exception as e:
                continue

        return players

    def extract_portal_entries(self, data: dict, source: str) -> list:
        """Extract transfer portal entries from page data."""
        entries = []
        props = data.get('props', {}).get('pageProps', {})

        # Primary path for portal wire/industry pages: playerData.list
        player_data = props.get('playerData', {})
        portal_list = player_data.get('list', [])

        # Fallback paths for other portal page types
        if not portal_list:
            portal_lists = [
                props.get('transfers', {}).get('list', []),
                props.get('players', {}).get('list', []),
                props.get('wire', {}).get('list', []),
                props.get('rankings', {}).get('list', []),
                props.get('teamRankings', {}).get('list', []),
                props.get('list', []),
            ]
            for pl in portal_lists:
                if pl:
                    portal_list = pl
                    break

        for item in portal_list:
            try:
                # Get basic player info
                name = item.get('name', '')
                slug = item.get('slug', '')
                position = item.get('positionAbbreviation', '')
                class_year = item.get('classYear')
                headshot = item.get('defaultAssetUrl', '')

                # Get previous school (lastTeam)
                last_team = item.get('lastTeam', {}) or {}
                from_school = last_team.get('fullName', '') or last_team.get('name', '')

                # Get committed school (commitStatus.committedOrganization)
                commit_status = item.get('commitStatus', {}) or {}
                committed_org = commit_status.get('committedOrganization', {}) or {}
                to_school = committed_org.get('fullName', '') or committed_org.get('name', '')
                status = commit_status.get('type', 'In Portal')
                commit_date = commit_status.get('date', '')

                # Get NIL valuation
                valuation = item.get('valuation', {}) or {}
                nil_value = valuation.get('totalValue') or item.get('nilValue') or 0

                # Get transfer rating (more relevant than recruit rating)
                transfer_rating = item.get('transferRating', {}) or item.get('rosterRating', {}) or {}
                stars = transfer_rating.get('stars') or transfer_rating.get('consensusStars')
                rating = transfer_rating.get('rating') or transfer_rating.get('consensusRating')
                national_rank = transfer_rating.get('nationalRank') or transfer_rating.get('consensusNationalRank')
                position_rank = transfer_rating.get('positionRank') or transfer_rating.get('consensusPositionRank')

                entry = {
                    'name': name,
                    'position': position,
                    'from_school': from_school,
                    'to_school': to_school,
                    'status': status,
                    'nil_valuation': nil_value,
                    'stars': stars,
                    'rating': rating,
                    'national_rank': national_rank,
                    'position_rank': position_rank,
                    'class_year': class_year,
                    'commit_date': commit_date,
                    'headshot_url': headshot,
                    'profile_url': f"https://www.on3.com/db/{slug}/" if slug else '',
                    'source': source,
                }
                if entry['name']:
                    entries.append(entry)
            except Exception:
                continue

        return entries

    def extract_team_rankings(self, data: dict, source: str) -> list:
        """Extract team portal rankings from page data."""
        teams = []
        props = data.get('props', {}).get('pageProps', {})

        team_data = props.get('teamRankings', {})
        team_list = team_data.get('list', [])
        year = team_data.get('relatedModel', {}).get('year')

        for item in team_list:
            try:
                org = item.get('organization', {}) or {}

                team = {
                    'team': org.get('name', ''),
                    'team_full': org.get('fullName', ''),
                    'conference': org.get('conferenceAbbr', ''),
                    'year': year,
                    'overall_rank': item.get('overallRank'),
                    'overall_score': item.get('overallScore'),
                    'raw_score': item.get('rawScore'),
                    # Transfers in
                    'transfers_in': item.get('totalIn', 0),
                    'avg_rating_in': item.get('totalInAverageRating'),
                    'raw_score_in': item.get('rawScoreIn'),
                    # Transfers out
                    'transfers_out': item.get('totalOut', 0),
                    'avg_rating_out': item.get('totalOutAverageRating'),
                    'raw_score_out': item.get('rawScoreOut'),
                    # Star counts (net)
                    'five_stars_net': item.get('fiveStarsNet', 0),
                    'four_stars_net': item.get('fourStarsNet', 0),
                    'three_stars_net': item.get('threeStarsNet', 0),
                    # NIL impact
                    'original_nil_valuation': item.get('originalNilValuation'),
                    'adjusted_nil_valuation': item.get('adjustedNilValuation'),
                    'nil_valuation_change': item.get('nilValuationChange'),
                    'source': source,
                }
                if team['team']:
                    teams.append(team)
            except Exception:
                continue

        return teams

    def scrape_portal_infinite(self, name: str, base_url: str, max_pages: int = 120) -> list:
        """Scrape portal page with infinite pagination."""
        all_entries = []
        seen_names = set()
        total_pages = None

        print(f"\n[{name}] {base_url}")

        for page in range(1, max_pages + 1):
            # Build page URL
            if '?' in base_url:
                page_url = f"{base_url}&page={page}"
            else:
                page_url = f"{base_url}?page={page}"

            data = self.fetch_page(page_url)
            if not data:
                print(f"  Page {page}: Failed to fetch")
                break

            # Get pagination info on first page
            if page == 1:
                props = data.get('props', {}).get('pageProps', {})
                pagination = props.get('playerData', {}).get('pagination', {})
                total_count = pagination.get('count', 0)
                total_pages = pagination.get('pageCount', max_pages)
                print(f"  Found {total_count} total entries across {total_pages} pages")

            entries = self.extract_portal_entries(data, source=name)

            if not entries:
                print(f"  Page {page}: No more data")
                break

            # Check for duplicates
            new_entries = [e for e in entries if e['name'] not in seen_names]
            if not new_entries:
                print(f"  Page {page}: All duplicates, stopping")
                break

            for e in new_entries:
                seen_names.add(e['name'])

            all_entries.extend(new_entries)

            if page % 10 == 0:
                print(f"  Page {page}/{total_pages or '?'}: {len(all_entries)} entries collected")

            # Stop if we've reached the last page
            if total_pages and page >= total_pages:
                break

            # Random delay between pages (2-4 seconds for safer scraping)
            time.sleep(random.uniform(2.0, 4.0))

        print(f"  TOTAL: {len(all_entries)} entries scraped")
        return all_entries

    def scrape_all(self, include_historical: bool = False) -> tuple:
        """Scrape all On3 pages.

        Args:
            include_historical: If True, also scrape 2024-2025 portal data

        Returns:
            Tuple of (df_nil, df_portal, df_team_rankings)
        """
        all_players = []
        all_portal = []
        all_team_rankings = []

        print("=" * 70)
        print("ON3 COMPLETE DATA SCRAPER")
        print("=" * 70)

        # Scrape NIL rankings pages
        for name, url in ON3_PAGES.items():
            print(f"\n[{name}] {url}")

            data = self.fetch_page(url)
            if not data:
                print("  Failed to fetch page")
                continue

            players = self.extract_nil_players(data, category=name)
            print(f"  Found {len(players)} players")

            all_players.extend(players)
            time.sleep(random.uniform(3.0, 5.0))  # Random delay between pages

        # Scrape Transfer Portal pages with infinite pagination
        print("\n" + "=" * 70)
        print("TRANSFER PORTAL DATA (Infinite Pagination)")
        print("=" * 70)

        # Current year portal data
        for name, url in PORTAL_PAGES.items():
            if 'team_rankings' in name:
                # Team rankings use different extraction
                data = self.fetch_page(url)
                if data:
                    teams = self.extract_team_rankings(data, source=name)
                    print(f"\n[{name}] {url}")
                    print(f"  Found {len(teams)} teams")
                    all_team_rankings.extend(teams)
            else:
                entries = self.scrape_portal_infinite(name, url, max_pages=120)
                all_portal.extend(entries)
            time.sleep(random.uniform(4.0, 7.0))  # Random delay between sources

        # Historical portal data (if requested)
        if include_historical:
            print("\n" + "=" * 70)
            print("HISTORICAL TRANSFER PORTAL DATA (2024-2025)")
            print("=" * 70)
            for name, url in PORTAL_PAGES_HISTORICAL.items():
                if 'team_rankings' in name:
                    # Team rankings use different extraction
                    data = self.fetch_page(url)
                    if data:
                        teams = self.extract_team_rankings(data, source=name)
                        print(f"\n[{name}] {url}")
                        print(f"  Found {len(teams)} teams")
                        all_team_rankings.extend(teams)
                else:
                    entries = self.scrape_portal_infinite(name, url, max_pages=120)
                    all_portal.extend(entries)
                time.sleep(random.uniform(4.0, 7.0))  # Random delay between sources

        # Combine and deduplicate
        df_nil = pd.DataFrame(all_players)
        df_portal = pd.DataFrame(all_portal)
        df_team_rankings = pd.DataFrame(all_team_rankings)

        if not df_nil.empty:
            df_nil = df_nil.drop_duplicates(subset=['name', 'category'], keep='first')

        if not df_portal.empty:
            df_portal = df_portal.drop_duplicates(subset=['name', 'source'], keep='first')

        if not df_team_rankings.empty:
            df_team_rankings = df_team_rankings.drop_duplicates(subset=['team', 'year', 'source'], keep='first')

        return df_nil, df_portal, df_team_rankings


def main():
    import argparse
    parser = argparse.ArgumentParser(description="On3 Complete Data Scraper")
    parser.add_argument('--historical', action='store_true',
                        help='Include 2024-2025 historical portal data')
    args = parser.parse_args()

    scraper = On3Scraper()
    df_nil, df_portal, df_team_rankings = scraper.scrape_all(include_historical=args.historical)

    print(f"\n{'=' * 70}")
    print("SCRAPING COMPLETE")
    print("=" * 70)

    # NIL Rankings Summary
    if not df_nil.empty:
        print(f"\nNIL RANKINGS: {len(df_nil)} unique players")
        print("-" * 40)
        for cat in df_nil['category'].unique():
            count = len(df_nil[df_nil['category'] == cat])
            print(f"  {cat}: {count}")

        # Top 10 by NIL value
        print("\nTop 10 by NIL Value:")
        top = df_nil.nlargest(10, 'nil_valuation')
        for i, (_, row) in enumerate(top.iterrows(), 1):
            val = f"${row['nil_valuation']:,.0f}" if row['nil_valuation'] else "N/A"
            school = str(row['school'])[:20] if pd.notna(row['school']) else 'Unknown'
            print(f"{i:2}. {val:>12} | {row['name']:25} | {row['position']:4} | {school}")

        # Save NIL data
        output_path = project_root / "data" / "processed" / "on3_all_nil_rankings.csv"
        df_nil.to_csv(output_path, index=False)
        print(f"\nSaved NIL data to {output_path}")

        for cat in df_nil['category'].unique():
            cat_df = df_nil[df_nil['category'] == cat]
            cat_path = project_root / "data" / "processed" / f"on3_{cat}.csv"
            cat_df.to_csv(cat_path, index=False)

    # Transfer Portal Summary
    if not df_portal.empty:
        print(f"\nTRANSFER PORTAL: {len(df_portal)} total entries")
        print("-" * 40)
        for src in df_portal['source'].unique():
            count = len(df_portal[df_portal['source'] == src])
            print(f"  {src}: {count}")

        # Status breakdown
        if 'status' in df_portal.columns:
            print("\nBy status:")
            for status in df_portal['status'].unique():
                if status:
                    count = len(df_portal[df_portal['status'] == status])
                    print(f"  {status}: {count}")

        # Save portal data
        portal_path = project_root / "data" / "processed" / "on3_transfer_portal.csv"
        df_portal.to_csv(portal_path, index=False)
        print(f"\nSaved portal data to {portal_path}")

        for src in df_portal['source'].unique():
            src_df = df_portal[df_portal['source'] == src]
            src_path = project_root / "data" / "processed" / f"on3_{src}.csv"
            src_df.to_csv(src_path, index=False)

    # Team Rankings Summary
    if not df_team_rankings.empty:
        print(f"\nTEAM PORTAL RANKINGS: {len(df_team_rankings)} team-year entries")
        print("-" * 40)

        # Show by year
        if 'year' in df_team_rankings.columns:
            for year in sorted(df_team_rankings['year'].unique(), reverse=True):
                count = len(df_team_rankings[df_team_rankings['year'] == year])
                print(f"  {year}: {count} teams")

        # Top 10 portal classes (most recent year)
        if not df_team_rankings.empty:
            latest_year = df_team_rankings['year'].max()
            latest = df_team_rankings[df_team_rankings['year'] == latest_year]
            if 'overall_score' in latest.columns:
                top_teams = latest.nlargest(10, 'overall_score')
                print(f"\nTop 10 Portal Classes ({latest_year}):")
                for i, (_, row) in enumerate(top_teams.iterrows(), 1):
                    score = f"{row['overall_score']:.1f}" if pd.notna(row['overall_score']) else "N/A"
                    net = row.get('five_stars_net', 0) + row.get('four_stars_net', 0)
                    print(f"{i:2}. {row['team']:20} | Score: {score:>8} | Net 4-5*: {net:+d}")

        # Save team rankings
        team_path = project_root / "data" / "processed" / "on3_team_portal_rankings.csv"
        df_team_rankings.to_csv(team_path, index=False)
        print(f"\nSaved team rankings to {team_path}")


if __name__ == "__main__":
    main()
