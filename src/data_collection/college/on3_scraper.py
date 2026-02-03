"""
On3 NIL Data Scraper

Scrapes NIL valuations, recruiting rankings, and player data from On3.com
using Playwright for JavaScript-rendered content.

Features:
- NIL 100 rankings scraper with LOGIN SUPPORT
- Individual player profile scraper
- Recruiting rankings scraper
- Rate limiting and retry logic
- Data validation and cleaning
- Integration with performance stats from CFBD
- Session persistence (saves cookies for reuse)

Usage:
    from src.data_collection.college.on3_scraper import On3Scraper

    # With login for full NIL data
    async with On3Scraper() as scraper:
        await scraper.login("your_email@example.com", "your_password")
        nil_100 = await scraper.scrape_nil_100()

    # Or set credentials in .env:
    # ON3_EMAIL=your_email@example.com
    # ON3_PASSWORD=your_password
"""

import asyncio
import logging
import re
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import os
from dotenv import load_dotenv

# Playwright for JS rendering
try:
    from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class NILPlayer:
    """Data class for NIL player information."""
    name: str
    school: str
    position: str
    nil_valuation: float
    nil_rank: Optional[int] = None
    conference: Optional[str] = None
    class_year: Optional[str] = None
    hometown: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[int] = None
    instagram_followers: Optional[int] = None
    twitter_followers: Optional[int] = None
    tiktok_followers: Optional[int] = None
    recruiting_stars: Optional[int] = None
    recruiting_rank: Optional[int] = None
    composite_rating: Optional[float] = None
    profile_url: Optional[str] = None
    scraped_at: str = None

    def __post_init__(self):
        if self.scraped_at is None:
            self.scraped_at = datetime.now().isoformat()


@dataclass
class TransferPortalEntry:
    """Data class for transfer portal player information."""
    name: str
    position: str
    previous_school: str
    previous_conference: Optional[str] = None
    new_school: Optional[str] = None  # None if still in portal
    new_conference: Optional[str] = None
    transfer_status: str = "In Portal"  # "In Portal", "Committed", "Enrolled"
    stars: Optional[int] = None
    rating: Optional[float] = None
    class_year: Optional[str] = None
    entry_date: Optional[str] = None  # When they entered portal
    commitment_date: Optional[str] = None  # When they committed (if applicable)
    nil_valuation: Optional[float] = None
    profile_url: Optional[str] = None
    scraped_at: str = None

    def __post_init__(self):
        if self.scraped_at is None:
            self.scraped_at = datetime.now().isoformat()


class On3Scraper:
    """
    Scrapes NIL and recruiting data from On3.com using Playwright.

    On3 uses heavy JavaScript rendering, so we need a headless browser
    to properly extract the data.
    """

    BASE_URL = "https://www.on3.com"
    LOGIN_URL = f"{BASE_URL}/login/"
    NIL_100_URL = f"{BASE_URL}/nil/rankings/player/nil-100/"
    NIL_RANKINGS_URL = f"{BASE_URL}/nil/rankings/player/"
    RECRUITING_URL = f"{BASE_URL}/db/rankings/player/all/"
    PORTAL_URL = f"{BASE_URL}/db/transfer-portal/football/"  # Transfer portal page

    # Rate limiting
    MIN_DELAY = 2.0  # seconds between requests
    MAX_DELAY = 5.0

    # Authentication state
    is_logged_in: bool = False

    # Selectors (these may need updating if On3 changes their site)
    SELECTORS = {
        # NIL 100 page selectors
        "nil_player_row": "div[class*='RankingsItem'], tr[class*='ranking']",
        "nil_player_name": "a[class*='PlayerName'], span[class*='name']",
        "nil_player_school": "span[class*='school'], a[class*='team']",
        "nil_player_position": "span[class*='position']",
        "nil_valuation": "span[class*='valuation'], div[class*='nil-value']",
        "nil_rank": "span[class*='rank'], div[class*='ranking']",

        # Player profile selectors
        "profile_name": "h1[class*='name'], h1[class*='player']",
        "profile_school": "a[class*='team'], span[class*='school']",
        "profile_position": "span[class*='position']",
        "profile_nil_value": "div[class*='nil-value'], span[class*='valuation']",
        "profile_stats": "div[class*='stats'], div[class*='bio']",
        "social_instagram": "a[href*='instagram.com']",
        "social_twitter": "a[href*='twitter.com'], a[href*='x.com']",
        "social_tiktok": "a[href*='tiktok.com']",
        "recruiting_stars": "span[class*='stars'], div[class*='rating']",

        # Recruiting rankings selectors
        "recruit_row": "div[class*='PlayerRow'], tr[class*='recruit']",
        "recruit_name": "a[class*='name']",
        "recruit_rating": "span[class*='rating']",
        "recruit_stars": "span[class*='stars']",

        # Transfer portal selectors
        "portal_player_row": "div[class*='TransferPlayer'], div[class*='PlayerRow'], tr[class*='transfer']",
        "portal_player_name": "a[class*='name'], span[class*='player-name']",
        "portal_prev_school": "span[class*='from-team'], a[class*='previous'], span[class*='transfer-from']",
        "portal_new_school": "span[class*='to-team'], a[class*='destination'], span[class*='transfer-to']",
        "portal_position": "span[class*='position']",
        "portal_status": "span[class*='status'], div[class*='commitment']",
        "portal_stars": "span[class*='stars'], div[class*='rating']",
        "portal_date": "span[class*='date'], div[class*='entry-date']",
    }

    def __init__(self, data_dir: Optional[str] = None, headless: bool = True,
                 email: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the On3 scraper.

        Args:
            data_dir: Directory for storing scraped data
            headless: Run browser in headless mode (default True)
            email: On3 account email (or set ON3_EMAIL env var)
            password: On3 account password (or set ON3_PASSWORD env var)
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        # Get credentials from args or environment
        self.email = email or os.getenv("ON3_EMAIL")
        self.password = password or os.getenv("ON3_PASSWORD")

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
        self.raw_dir = self.data_dir / "raw" / "on3"
        self.cache_dir = self.data_dir / "cache"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.headless = headless
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._playwright = None
        self._context = None

        # Cookie storage for session persistence
        self.cookies_path = self.cache_dir / "on3_cookies.json"

        logger.info(f"On3Scraper initialized. Data dir: {self.data_dir}")
        if self.email:
            logger.info(f"Credentials provided for: {self.email}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._start_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_browser()

    async def _start_browser(self):
        """Start the Playwright browser."""
        self._playwright = await async_playwright().start()

        # Use persistent context for interactive login (saves Google login)
        if not self.headless:
            # Persistent browser profile for keeping Google login
            user_data_dir = self.cache_dir / "browser_profile"
            user_data_dir.mkdir(parents=True, exist_ok=True)

            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                headless=False,
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ]
            )
            self.browser = None  # Persistent context doesn't use separate browser
            self.page = self._context.pages[0] if self._context.pages else await self._context.new_page()
            logger.info("Browser started with persistent profile (Google login will be saved)")
        else:
            # Regular headless mode
            self.browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ]
            )

            self._context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                locale="en-US",
            )

            # Load saved cookies if available
            if self.cookies_path.exists():
                try:
                    with open(self.cookies_path, "r") as f:
                        cookies = json.load(f)
                    await self._context.add_cookies(cookies)
                    logger.info("Loaded saved session cookies")
                except Exception as e:
                    logger.debug(f"Could not load cookies: {e}")

            self.page = await self._context.new_page()

        # Only block ads/analytics, keep images for login to work
        await self.page.route("**/analytics**", lambda route: route.abort())
        await self.page.route("**/ads**", lambda route: route.abort())
        await self.page.route("**/*doubleclick*", lambda route: route.abort())
        await self.page.route("**/*google-analytics*", lambda route: route.abort())

        logger.info("Browser started")

    async def _close_browser(self):
        """Close the browser and save cookies."""
        # Save cookies before closing (for non-persistent contexts)
        if self._context and self.is_logged_in and self.browser:
            try:
                cookies = await self._context.cookies()
                with open(self.cookies_path, "w") as f:
                    json.dump(cookies, f)
                logger.info("Saved session cookies")
            except Exception as e:
                logger.debug(f"Could not save cookies: {e}")

        # Close browser or persistent context
        if self.browser:
            await self.browser.close()
        elif self._context:
            await self._context.close()

        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")

    async def login_interactive(self, timeout: int = 180) -> bool:
        """
        Interactive login - opens browser for manual Google/OAuth login.

        Use this if you signed up with Google. The browser will open,
        you complete the login manually, and cookies are saved for future use.

        Args:
            timeout: Seconds to wait for manual login (default 180)

        Returns:
            True if login successful
        """
        logger.info("Opening browser for manual login...")
        logger.info(f"You have {timeout} seconds to complete login.")

        # Set up popup handler for Google OAuth
        async def handle_popup(popup):
            logger.info(f"Popup opened: {popup.url}")
            await popup.wait_for_load_state()

        self._context.on("page", handle_popup)

        try:
            # Navigate to login page with longer timeout
            await self.page.goto(self.LOGIN_URL, wait_until="domcontentloaded", timeout=60000)

            print("\n" + "=" * 60)
            print("MANUAL LOGIN REQUIRED")
            print("=" * 60)
            print("A browser window should be open (or opening).")
            print("Please log in to On3 using Google or your credentials.")
            print(f"Waiting up to {timeout} seconds...")
            print("=" * 60 + "\n")

            # Wait for user to complete login
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                await asyncio.sleep(2)

                # Check if logged in
                if await self._check_logged_in():
                    self.is_logged_in = True
                    logger.info("Login detected!")

                    # Save cookies
                    cookies = await self._context.cookies()
                    with open(self.cookies_path, "w") as f:
                        json.dump(cookies, f)
                    logger.info("Session cookies saved for future use")

                    print("\nLogin successful! Cookies saved.")
                    print("Future runs will use saved session.\n")
                    return True

                # Also check URL - if they navigated away from login, might be logged in
                current_url = self.page.url
                if "/login" not in current_url and "on3.com" in current_url:
                    await asyncio.sleep(2)  # Give it a moment to settle
                    if await self._check_logged_in():
                        self.is_logged_in = True
                        cookies = await self._context.cookies()
                        with open(self.cookies_path, "w") as f:
                            json.dump(cookies, f)
                        print("\nLogin successful! Cookies saved.\n")
                        return True

            logger.error("Login timeout - did not detect successful login")
            return False

        except Exception as e:
            logger.error(f"Interactive login error: {e}")
            return False

    async def login(self, email: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Log in to On3 to access NIL valuations.

        Args:
            email: On3 account email (uses self.email if not provided)
            password: On3 account password (uses self.password if not provided)

        Returns:
            True if login successful, False otherwise
        """
        email = email or self.email
        password = password or self.password

        if not email or not password:
            logger.warning("No email/password credentials. Use login_interactive() for Google login.")
            return False

        logger.info(f"Logging in as: {email}")

        try:
            # Navigate to login page
            await self.page.goto(self.LOGIN_URL, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(2)

            # Look for login form - On3 uses various form structures
            # Try multiple selectors
            email_selectors = [
                'input[type="email"]',
                'input[name="email"]',
                'input[placeholder*="email" i]',
                'input[id*="email" i]',
            ]

            password_selectors = [
                'input[type="password"]',
                'input[name="password"]',
                'input[placeholder*="password" i]',
            ]

            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Log in")',
                'button:has-text("Sign in")',
                'button:has-text("Login")',
            ]

            # Find and fill email field
            email_field = None
            for selector in email_selectors:
                try:
                    email_field = await self.page.wait_for_selector(selector, timeout=5000)
                    if email_field:
                        break
                except:
                    continue

            if not email_field:
                logger.error("Could not find email input field")
                # Save screenshot for debugging
                await self.page.screenshot(path=str(self.raw_dir / "login_debug.png"))
                return False

            await email_field.fill(email)
            await asyncio.sleep(0.5)

            # Find and fill password field
            password_field = None
            for selector in password_selectors:
                try:
                    password_field = await self.page.query_selector(selector)
                    if password_field:
                        break
                except:
                    continue

            if not password_field:
                logger.error("Could not find password input field")
                return False

            await password_field.fill(password)
            await asyncio.sleep(0.5)

            # Find and click submit button
            submit_btn = None
            for selector in submit_selectors:
                try:
                    submit_btn = await self.page.query_selector(selector)
                    if submit_btn:
                        break
                except:
                    continue

            if not submit_btn:
                # Try pressing Enter instead
                await password_field.press("Enter")
            else:
                await submit_btn.click()

            # Wait for navigation/login to complete
            await asyncio.sleep(3)

            # Check if login was successful by looking for user indicators
            is_logged_in = await self._check_logged_in()

            if is_logged_in:
                self.is_logged_in = True
                logger.info("Login successful!")

                # Save cookies for future sessions
                cookies = await self._context.cookies()
                with open(self.cookies_path, "w") as f:
                    json.dump(cookies, f)
                logger.info("Session cookies saved")

                return True
            else:
                logger.error("Login failed - could not verify logged in state")
                await self.page.screenshot(path=str(self.raw_dir / "login_failed.png"))
                return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    async def _check_logged_in(self) -> bool:
        """Check if currently logged in to On3."""
        try:
            # Look for indicators of logged-in state
            logged_in_indicators = [
                'a[href*="/account"]',
                'a[href*="/profile"]',
                'button:has-text("Account")',
                'span:has-text("My Account")',
                '[class*="user-menu"]',
                '[class*="account"]',
            ]

            for selector in logged_in_indicators:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        return True
                except:
                    continue

            # Also check if we're NOT on the login page anymore
            current_url = self.page.url
            if "/login" not in current_url and "on3.com" in current_url:
                # Try accessing a protected page
                return True

            return False
        except:
            return False

    async def ensure_logged_in(self) -> bool:
        """Ensure we're logged in, attempting login if needed."""
        if self.is_logged_in:
            return True

        # Check if already logged in from saved cookies
        if await self._check_logged_in():
            self.is_logged_in = True
            return True

        # Try to login
        if self.email and self.password:
            return await self.login()

        return False

    async def _random_delay(self):
        """Add random delay between requests."""
        delay = random.uniform(self.MIN_DELAY, self.MAX_DELAY)
        await asyncio.sleep(delay)

    async def _navigate_with_retry(self, url: str, max_retries: int = 3) -> bool:
        """
        Navigate to URL with retry logic.

        Args:
            url: URL to navigate to
            max_retries: Maximum retry attempts

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                await self._random_delay()
                response = await self.page.goto(url, wait_until="networkidle", timeout=30000)

                if response and response.ok:
                    # Wait for dynamic content
                    await asyncio.sleep(2)
                    return True
                else:
                    logger.warning(f"Non-OK response from {url}: {response.status if response else 'None'}")

            except PlaywrightTimeout:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1} for {url}: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff

        return False

    def _parse_valuation(self, value_str: str) -> Optional[float]:
        """Parse NIL valuation string to float."""
        if not value_str:
            return None

        try:
            # Clean the string
            value_str = value_str.replace("$", "").replace(",", "").strip()

            # Handle M for millions, K for thousands
            if "M" in value_str.upper():
                return float(value_str.upper().replace("M", "")) * 1_000_000
            elif "K" in value_str.upper():
                return float(value_str.upper().replace("K", "")) * 1_000
            else:
                return float(value_str)
        except (ValueError, AttributeError):
            return None

    def _parse_followers(self, text: str) -> Optional[int]:
        """Parse follower count string to int."""
        if not text:
            return None

        try:
            text = text.lower().replace(",", "").strip()

            if "m" in text:
                return int(float(text.replace("m", "")) * 1_000_000)
            elif "k" in text:
                return int(float(text.replace("k", "")) * 1_000)
            else:
                return int(float(text))
        except (ValueError, AttributeError):
            return None

    async def scrape_nil_100(self, pages: int = 1) -> List[NILPlayer]:
        """
        Scrape the NIL 100 rankings.

        Args:
            pages: Number of pages to scrape (100 players per page)

        Returns:
            List of NILPlayer objects
        """
        logger.info(f"Scraping NIL 100 rankings ({pages} pages)...")
        players = []

        for page_num in range(1, pages + 1):
            url = self.NIL_100_URL if page_num == 1 else f"{self.NIL_100_URL}?page={page_num}"

            if not await self._navigate_with_retry(url):
                logger.error(f"Failed to load NIL 100 page {page_num}")
                continue

            # Try multiple selector strategies
            page_players = await self._extract_nil_rankings()

            if not page_players:
                # Fallback: try extracting from page source with regex
                page_players = await self._extract_nil_from_json()

            players.extend(page_players)
            logger.info(f"Page {page_num}: Found {len(page_players)} players")

        # Save raw data
        if players:
            self._save_players(players, "nil_100")

        return players

    async def _extract_nil_rankings(self) -> List[NILPlayer]:
        """Extract NIL rankings from page DOM."""
        players = []

        # Try to find player rows
        rows = await self.page.query_selector_all(self.SELECTORS["nil_player_row"])

        if not rows:
            # Try alternative: look for any ranking-like structure
            rows = await self.page.query_selector_all("[class*='rank'], [class*='player']")

        for row in rows:
            try:
                # Extract name
                name_el = await row.query_selector(self.SELECTORS["nil_player_name"])
                if not name_el:
                    name_el = await row.query_selector("a")
                name = await name_el.text_content() if name_el else None

                if not name or len(name) < 3:
                    continue

                # Extract school
                school_el = await row.query_selector(self.SELECTORS["nil_player_school"])
                school = await school_el.text_content() if school_el else None

                # Extract position
                pos_el = await row.query_selector(self.SELECTORS["nil_player_position"])
                position = await pos_el.text_content() if pos_el else None

                # Extract NIL valuation
                val_el = await row.query_selector(self.SELECTORS["nil_valuation"])
                val_text = await val_el.text_content() if val_el else None
                valuation = self._parse_valuation(val_text)

                # Extract rank
                rank_el = await row.query_selector(self.SELECTORS["nil_rank"])
                rank_text = await rank_el.text_content() if rank_el else None
                rank = int(re.sub(r'\D', '', rank_text)) if rank_text else None

                # Get profile URL
                link = await row.query_selector("a[href*='/db/']")
                profile_url = await link.get_attribute("href") if link else None
                if profile_url and not profile_url.startswith("http"):
                    profile_url = f"{self.BASE_URL}{profile_url}"

                if name and (valuation or school):
                    players.append(NILPlayer(
                        name=name.strip(),
                        school=school.strip() if school else "Unknown",
                        position=position.strip() if position else "Unknown",
                        nil_valuation=valuation or 0,
                        nil_rank=rank,
                        profile_url=profile_url,
                    ))

            except Exception as e:
                logger.debug(f"Error parsing player row: {e}")
                continue

        return players

    async def _extract_nil_from_json(self) -> List[NILPlayer]:
        """
        Try to extract NIL data from embedded JSON in page source.
        Many modern sites embed data in script tags for hydration.
        """
        players = []

        try:
            # Get page content
            content = await self.page.content()

            # Look for JSON data in script tags
            json_patterns = [
                r'__NEXT_DATA__.*?({.*?})</script>',
                r'window\.__PRELOADED_STATE__\s*=\s*({.*?});',
                r'"players"\s*:\s*(\[.*?\])',
                r'"rankings"\s*:\s*(\[.*?\])',
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        data = json.loads(match)
                        players.extend(self._parse_json_players(data))
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.debug(f"Error extracting JSON: {e}")

        return players

    def _parse_json_players(self, data: Any, depth: int = 0) -> List[NILPlayer]:
        """Recursively search JSON for player data."""
        players = []

        if depth > 10:  # Prevent infinite recursion
            return players

        if isinstance(data, dict):
            # Check if this looks like player data
            if "name" in data and any(k in data for k in ["nilValue", "valuation", "nil_valuation"]):
                try:
                    valuation = data.get("nilValue") or data.get("valuation") or data.get("nil_valuation")
                    if isinstance(valuation, str):
                        valuation = self._parse_valuation(valuation)

                    players.append(NILPlayer(
                        name=data.get("name", "Unknown"),
                        school=data.get("school") or data.get("team") or data.get("organization", "Unknown"),
                        position=data.get("position", "Unknown"),
                        nil_valuation=valuation or 0,
                        nil_rank=data.get("rank") or data.get("nil_rank"),
                        recruiting_stars=data.get("stars"),
                        composite_rating=data.get("rating") or data.get("composite"),
                    ))
                except Exception:
                    pass

            # Recurse into dict values
            for value in data.values():
                players.extend(self._parse_json_players(value, depth + 1))

        elif isinstance(data, list):
            for item in data:
                players.extend(self._parse_json_players(item, depth + 1))

        return players

    async def scrape_player_profile(self, player_slug: str) -> Optional[NILPlayer]:
        """
        Scrape detailed data for a specific player.

        Args:
            player_slug: Player URL slug (e.g., "travis-hunter-1")

        Returns:
            NILPlayer with detailed info, or None if not found
        """
        url = f"{self.BASE_URL}/db/{player_slug}/"

        if not await self._navigate_with_retry(url):
            logger.error(f"Failed to load player profile: {player_slug}")
            return None

        try:
            # Extract basic info
            name_el = await self.page.query_selector(self.SELECTORS["profile_name"])
            name = await name_el.text_content() if name_el else player_slug.replace("-", " ").title()

            school_el = await self.page.query_selector(self.SELECTORS["profile_school"])
            school = await school_el.text_content() if school_el else None

            pos_el = await self.page.query_selector(self.SELECTORS["profile_position"])
            position = await pos_el.text_content() if pos_el else None

            nil_el = await self.page.query_selector(self.SELECTORS["profile_nil_value"])
            nil_text = await nil_el.text_content() if nil_el else None
            nil_value = self._parse_valuation(nil_text)

            # Extract social media
            ig_el = await self.page.query_selector(self.SELECTORS["social_instagram"])
            ig_followers = None
            if ig_el:
                parent = await ig_el.query_selector("xpath=..")
                if parent:
                    text = await parent.text_content()
                    ig_followers = self._parse_followers(text)

            tw_el = await self.page.query_selector(self.SELECTORS["social_twitter"])
            tw_followers = None
            if tw_el:
                parent = await tw_el.query_selector("xpath=..")
                if parent:
                    text = await parent.text_content()
                    tw_followers = self._parse_followers(text)

            tt_el = await self.page.query_selector(self.SELECTORS["social_tiktok"])
            tt_followers = None
            if tt_el:
                parent = await tt_el.query_selector("xpath=..")
                if parent:
                    text = await parent.text_content()
                    tt_followers = self._parse_followers(text)

            # Extract recruiting info
            stars_el = await self.page.query_selector(self.SELECTORS["recruiting_stars"])
            stars = None
            if stars_el:
                stars_text = await stars_el.text_content()
                stars_match = re.search(r'(\d)', stars_text or "")
                if stars_match:
                    stars = int(stars_match.group(1))

            return NILPlayer(
                name=name.strip() if name else player_slug,
                school=school.strip() if school else "Unknown",
                position=position.strip() if position else "Unknown",
                nil_valuation=nil_value or 0,
                instagram_followers=ig_followers,
                twitter_followers=tw_followers,
                tiktok_followers=tt_followers,
                recruiting_stars=stars,
                profile_url=url,
            )

        except Exception as e:
            logger.error(f"Error parsing player profile {player_slug}: {e}")
            return None

    async def scrape_recruiting_rankings(
        self,
        year: int = 2025,
        sport: str = "football",
        pages: int = 5
    ) -> List[Dict]:
        """
        Scrape recruiting rankings.

        Args:
            year: Recruiting class year
            sport: Sport (football, basketball)
            pages: Number of pages to scrape

        Returns:
            List of recruit dictionaries
        """
        logger.info(f"Scraping {year} recruiting rankings ({pages} pages)...")
        recruits = []

        for page_num in range(1, pages + 1):
            url = f"{self.RECRUITING_URL}{year}/{sport}/?page={page_num}"

            if not await self._navigate_with_retry(url):
                logger.error(f"Failed to load recruiting page {page_num}")
                continue

            rows = await self.page.query_selector_all(self.SELECTORS["recruit_row"])

            for row in rows:
                try:
                    name_el = await row.query_selector(self.SELECTORS["recruit_name"])
                    name = await name_el.text_content() if name_el else None

                    rating_el = await row.query_selector(self.SELECTORS["recruit_rating"])
                    rating = await rating_el.text_content() if rating_el else None

                    stars_el = await row.query_selector(self.SELECTORS["recruit_stars"])
                    stars_text = await stars_el.text_content() if stars_el else None

                    if name:
                        recruits.append({
                            "name": name.strip(),
                            "rating": float(rating) if rating else None,
                            "stars": int(re.sub(r'\D', '', stars_text)) if stars_text else None,
                            "class_year": year,
                        })

                except Exception as e:
                    logger.debug(f"Error parsing recruit row: {e}")
                    continue

            logger.info(f"Page {page_num}: Found {len(rows)} recruits")

        return recruits

    async def scrape_transfer_portal(
        self,
        year: int = 2025,
        pages: int = 10,
        status_filter: Optional[str] = None
    ) -> List[TransferPortalEntry]:
        """
        Scrape the transfer portal.

        Args:
            year: Portal year (default current)
            pages: Number of pages to scrape (50 players per page typically)
            status_filter: Optional filter - "available", "committed", "all" (default all)

        Returns:
            List of TransferPortalEntry objects
        """
        logger.info(f"Scraping {year} transfer portal ({pages} pages)...")
        portal_entries = []

        for page_num in range(1, pages + 1):
            # Build URL with optional filters
            url = f"{self.PORTAL_URL}{year}/"
            if page_num > 1:
                url += f"?page={page_num}"

            if not await self._navigate_with_retry(url):
                logger.error(f"Failed to load portal page {page_num}")
                continue

            # Try DOM extraction first
            page_entries = await self._extract_portal_entries()

            if not page_entries:
                # Fallback: try extracting from embedded JSON
                page_entries = await self._extract_portal_from_json()

            # Apply status filter if specified
            if status_filter and status_filter != "all":
                if status_filter == "available":
                    page_entries = [e for e in page_entries if e.new_school is None]
                elif status_filter == "committed":
                    page_entries = [e for e in page_entries if e.new_school is not None]

            portal_entries.extend(page_entries)
            logger.info(f"Page {page_num}: Found {len(page_entries)} portal entries")

            # Stop if page was empty (reached end)
            if not page_entries and page_num > 1:
                logger.info(f"No more entries found, stopping at page {page_num}")
                break

        # Save raw data
        if portal_entries:
            self._save_portal_entries(portal_entries, f"transfer_portal_{year}")

        return portal_entries

    async def _extract_portal_entries(self) -> List[TransferPortalEntry]:
        """Extract transfer portal entries from page DOM."""
        entries = []

        # Try to find player rows
        rows = await self.page.query_selector_all(self.SELECTORS["portal_player_row"])

        if not rows:
            # Try alternative selectors
            rows = await self.page.query_selector_all("[class*='transfer'], [class*='portal-player']")

        for row in rows:
            try:
                # Extract name
                name_el = await row.query_selector(self.SELECTORS["portal_player_name"])
                if not name_el:
                    name_el = await row.query_selector("a")
                name = await name_el.text_content() if name_el else None

                if not name or len(name) < 3:
                    continue

                # Extract position
                pos_el = await row.query_selector(self.SELECTORS["portal_position"])
                position = await pos_el.text_content() if pos_el else "Unknown"

                # Extract previous school
                prev_school_el = await row.query_selector(self.SELECTORS["portal_prev_school"])
                prev_school = await prev_school_el.text_content() if prev_school_el else "Unknown"

                # Extract new school (if committed)
                new_school_el = await row.query_selector(self.SELECTORS["portal_new_school"])
                new_school = await new_school_el.text_content() if new_school_el else None
                if new_school:
                    new_school = new_school.strip()
                    # Check for placeholder text indicating no commitment yet
                    if new_school.lower() in ["--", "-", "n/a", "tbd", ""]:
                        new_school = None

                # Determine transfer status
                status_el = await row.query_selector(self.SELECTORS["portal_status"])
                status_text = await status_el.text_content() if status_el else None

                if status_text:
                    status_text = status_text.strip().lower()
                    if "commit" in status_text or "enrolled" in status_text:
                        transfer_status = "Committed"
                    elif "enrolled" in status_text:
                        transfer_status = "Enrolled"
                    else:
                        transfer_status = "In Portal"
                else:
                    transfer_status = "Committed" if new_school else "In Portal"

                # Extract stars/rating
                stars_el = await row.query_selector(self.SELECTORS["portal_stars"])
                stars = None
                if stars_el:
                    stars_text = await stars_el.text_content()
                    stars_match = re.search(r'(\d)', stars_text or "")
                    if stars_match:
                        stars = int(stars_match.group(1))

                # Extract entry date
                date_el = await row.query_selector(self.SELECTORS["portal_date"])
                entry_date = await date_el.text_content() if date_el else None

                # Get profile URL
                link = await row.query_selector("a[href*='/db/']")
                profile_url = await link.get_attribute("href") if link else None
                if profile_url and not profile_url.startswith("http"):
                    profile_url = f"{self.BASE_URL}{profile_url}"

                entries.append(TransferPortalEntry(
                    name=name.strip(),
                    position=position.strip() if position else "Unknown",
                    previous_school=prev_school.strip() if prev_school else "Unknown",
                    new_school=new_school,
                    transfer_status=transfer_status,
                    stars=stars,
                    entry_date=entry_date.strip() if entry_date else None,
                    profile_url=profile_url,
                ))

            except Exception as e:
                logger.debug(f"Error parsing portal row: {e}")
                continue

        return entries

    async def _extract_portal_from_json(self) -> List[TransferPortalEntry]:
        """
        Try to extract portal data from embedded JSON in page source.
        """
        entries = []

        try:
            content = await self.page.content()

            # Look for JSON data patterns
            json_patterns = [
                r'__NEXT_DATA__.*?({.*?})</script>',
                r'window\.__PRELOADED_STATE__\s*=\s*({.*?});',
                r'"transfers"\s*:\s*(\[.*?\])',
                r'"portalPlayers"\s*:\s*(\[.*?\])',
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        data = json.loads(match)
                        entries.extend(self._parse_json_portal_entries(data))
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.debug(f"Error extracting portal JSON: {e}")

        return entries

    def _parse_json_portal_entries(self, data: Any, depth: int = 0) -> List[TransferPortalEntry]:
        """Recursively search JSON for transfer portal data."""
        entries = []

        if depth > 10:
            return entries

        if isinstance(data, dict):
            # Check if this looks like portal entry data
            if "name" in data and any(k in data for k in ["previousSchool", "fromSchool", "previous_school"]):
                try:
                    entries.append(TransferPortalEntry(
                        name=data.get("name", "Unknown"),
                        position=data.get("position", "Unknown"),
                        previous_school=data.get("previousSchool") or data.get("fromSchool") or data.get("previous_school", "Unknown"),
                        new_school=data.get("newSchool") or data.get("toSchool") or data.get("destination"),
                        transfer_status="Committed" if data.get("newSchool") or data.get("toSchool") else "In Portal",
                        stars=data.get("stars"),
                        rating=data.get("rating"),
                        entry_date=data.get("entryDate") or data.get("entry_date"),
                    ))
                except Exception:
                    pass

            # Recurse into dict values
            for value in data.values():
                entries.extend(self._parse_json_portal_entries(value, depth + 1))

        elif isinstance(data, list):
            for item in data:
                entries.extend(self._parse_json_portal_entries(item, depth + 1))

        return entries

    def _save_portal_entries(self, entries: List[TransferPortalEntry], prefix: str):
        """Save portal entries to CSV."""
        if not entries:
            return

        df = pd.DataFrame([asdict(e) for e in entries])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save timestamped version
        path = self.raw_dir / f"{prefix}_{timestamp}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(entries)} portal entries to {path}")

        # Also save as latest
        latest_path = self.raw_dir / f"{prefix}_latest.csv"
        df.to_csv(latest_path, index=False)

        # Also save to main data directory for API access
        processed_path = self.data_dir / "processed" / "transfer_portal.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        logger.info(f"Updated {processed_path}")

    def load_latest_portal_data(self) -> pd.DataFrame:
        """Load the most recent portal scrape."""
        latest_path = self.raw_dir / "transfer_portal_2025_latest.csv"
        if latest_path.exists():
            return pd.read_csv(latest_path)
        # Fall back to processed directory
        processed_path = self.data_dir / "processed" / "transfer_portal.csv"
        if processed_path.exists():
            return pd.read_csv(processed_path)
        return pd.DataFrame()

    def _save_players(self, players: List[NILPlayer], prefix: str):
        """Save players to CSV."""
        if not players:
            return

        df = pd.DataFrame([asdict(p) for p in players])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save timestamped version
        path = self.raw_dir / f"{prefix}_{timestamp}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(players)} players to {path}")

        # Also save as latest
        latest_path = self.raw_dir / f"{prefix}_latest.csv"
        df.to_csv(latest_path, index=False)

    def load_latest_nil_data(self) -> pd.DataFrame:
        """Load the most recent NIL 100 scrape."""
        latest_path = self.raw_dir / "nil_100_latest.csv"
        if latest_path.exists():
            return pd.read_csv(latest_path)
        return pd.DataFrame()

    async def build_nil_performance_dataset(self) -> pd.DataFrame:
        """
        Build a merged dataset of NIL valuations + performance stats.

        This is the key integration point - combining scraped NIL data
        with performance data from CFBD for model training.

        Returns:
            Merged DataFrame with NIL + performance features
        """
        logger.info("Building NIL + performance dataset...")

        # Load NIL data
        nil_df = self.load_latest_nil_data()
        if nil_df.empty:
            logger.info("No NIL data found, scraping now...")
            players = await self.scrape_nil_100()
            nil_df = pd.DataFrame([asdict(p) for p in players])

        if nil_df.empty:
            logger.error("Could not get NIL data")
            return pd.DataFrame()

        # Load performance stats from cache
        stats_path = self.cache_dir / "cfb_player_stats_2020_2025_cache.csv"
        if stats_path.exists():
            stats_df = pd.read_csv(stats_path)
            logger.info(f"Loaded {len(stats_df)} performance records")
        else:
            logger.warning("No performance stats cache found - run cfb_stats.py first")
            stats_df = pd.DataFrame()

        # Load recruiting data
        recruiting_path = self.cache_dir / "cfb_recruiting_players_2018_2025_cache.csv"
        if recruiting_path.exists():
            recruiting_df = pd.read_csv(recruiting_path)
            logger.info(f"Loaded {len(recruiting_df)} recruiting records")
        else:
            recruiting_df = pd.DataFrame()

        # Merge datasets
        merged = self._merge_datasets(nil_df, stats_df, recruiting_df)

        # Save merged dataset
        if not merged.empty:
            output_path = self.data_dir / "processed" / "nil_performance_merged.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(output_path, index=False)
            logger.info(f"Saved merged dataset: {output_path} ({len(merged)} rows)")

        return merged

    def _merge_datasets(
        self,
        nil_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        recruiting_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge NIL, stats, and recruiting data using fuzzy name matching.
        """
        from rapidfuzz import fuzz, process

        merged = nil_df.copy()

        # Standardize name column
        merged["name_std"] = merged["name"].str.lower().str.strip()

        # Merge performance stats
        if not stats_df.empty:
            name_col = "player" if "player" in stats_df.columns else "player_name"
            if name_col in stats_df.columns:
                stats_df["name_std"] = stats_df[name_col].str.lower().str.strip()

                # Get latest season stats per player
                if "season" in stats_df.columns:
                    latest_stats = stats_df.sort_values("season", ascending=False).groupby("name_std").first().reset_index()
                else:
                    latest_stats = stats_df.groupby("name_std").first().reset_index()

                # Fuzzy match names
                matched_stats = []
                for _, row in merged.iterrows():
                    best_match = process.extractOne(
                        row["name_std"],
                        latest_stats["name_std"].tolist(),
                        scorer=fuzz.token_sort_ratio
                    )
                    if best_match and best_match[1] >= 85:  # 85% match threshold
                        match_row = latest_stats[latest_stats["name_std"] == best_match[0]].iloc[0]
                        matched_stats.append(match_row.to_dict())
                    else:
                        matched_stats.append({})

                stats_to_add = pd.DataFrame(matched_stats)
                # Add only non-overlapping columns
                new_cols = [c for c in stats_to_add.columns if c not in merged.columns and c != "name_std"]
                if new_cols:
                    merged = pd.concat([merged, stats_to_add[new_cols]], axis=1)
                    logger.info(f"Added {len(new_cols)} performance columns")

        # Merge recruiting data
        if not recruiting_df.empty:
            name_col = "name" if "name" in recruiting_df.columns else "player_name"
            if name_col in recruiting_df.columns:
                recruiting_df["name_std"] = recruiting_df[name_col].str.lower().str.strip()

                # Fuzzy match
                matched_recruiting = []
                for _, row in merged.iterrows():
                    best_match = process.extractOne(
                        row["name_std"],
                        recruiting_df["name_std"].tolist(),
                        scorer=fuzz.token_sort_ratio
                    )
                    if best_match and best_match[1] >= 85:
                        match_row = recruiting_df[recruiting_df["name_std"] == best_match[0]].iloc[0]
                        matched_recruiting.append(match_row.to_dict())
                    else:
                        matched_recruiting.append({})

                recruiting_to_add = pd.DataFrame(matched_recruiting)
                new_cols = [c for c in recruiting_to_add.columns if c not in merged.columns and c != "name_std"]
                if new_cols:
                    # Rename to avoid conflicts
                    renamed = {c: f"recruiting_{c}" if not c.startswith("recruiting") else c for c in new_cols}
                    recruiting_to_add = recruiting_to_add[new_cols].rename(columns=renamed)
                    merged = pd.concat([merged, recruiting_to_add], axis=1)
                    logger.info(f"Added {len(new_cols)} recruiting columns")

        # Clean up
        merged = merged.drop(columns=["name_std"], errors="ignore")

        return merged


async def main():
    """Example usage of the On3 scraper."""
    print("On3 NIL & Transfer Portal Scraper")
    print("=" * 50)

    async with On3Scraper(headless=True) as scraper:
        # Scrape NIL 100
        print("\n[1/4] Scraping NIL 100 rankings...")
        players = await scraper.scrape_nil_100(pages=1)
        print(f"Found {len(players)} players")

        if players:
            # Show top 10
            print("\nTop 10 NIL valuations:")
            sorted_players = sorted(players, key=lambda p: p.nil_valuation or 0, reverse=True)[:10]
            for i, p in enumerate(sorted_players, 1):
                print(f"  {i}. {p.name} ({p.school}) - ${p.nil_valuation:,.0f}")

        # Scrape transfer portal
        print("\n[2/4] Scraping transfer portal...")
        portal_entries = await scraper.scrape_transfer_portal(year=2025, pages=3)
        print(f"Found {len(portal_entries)} portal entries")

        if portal_entries:
            # Show summary
            in_portal = len([e for e in portal_entries if e.transfer_status == "In Portal"])
            committed = len([e for e in portal_entries if e.transfer_status != "In Portal"])
            print(f"\n  Still in portal: {in_portal}")
            print(f"  Committed: {committed}")

            # Show top available players
            available = [e for e in portal_entries if e.transfer_status == "In Portal"][:5]
            if available:
                print("\nTop 5 available players:")
                for i, e in enumerate(available, 1):
                    stars = f"{e.stars}★" if e.stars else "N/R"
                    print(f"  {i}. {e.name} ({e.position}) - from {e.previous_school} [{stars}]")

        # Build merged dataset
        print("\n[3/4] Building NIL + performance dataset...")
        merged = await scraper.build_nil_performance_dataset()
        print(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")

        print("\n[4/4] Done!")


if __name__ == "__main__":
    asyncio.run(main())
