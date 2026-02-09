"""
Custom NIL Valuation Model

A proprietary NIL valuation system based on measurable player attributes:
- On-field performance and production
- Value to team (win impact by position)
- Market factors (school brand, conference, media market)
- Social media reach and engagement
- Recruiting profile and potential

This model creates valuations WITHOUT relying on scraped deal data.
Instead, it builds a bottoms-up estimate of what a player SHOULD be worth.

The model can be calibrated against known deals when available.

Updated Feb 2026:
- Dynamic school tiers from CFBD data (wins, SP+, talent, recruiting)
- Manual overrides for championships and special cases
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Import dynamic school tier system
try:
    from .school_tiers import get_school_multiplier as dynamic_school_multiplier
    from .school_tiers import get_school_tier
    DYNAMIC_TIERS_AVAILABLE = True
except ImportError:
    DYNAMIC_TIERS_AVAILABLE = False
    logger.debug("Dynamic school tiers not available, using static multipliers")


@dataclass
class NILValuation:
    """Output of NIL valuation model."""
    player_name: str
    total_valuation: float
    valuation_tier: str  # mega, premium, solid, moderate, entry

    # Component breakdowns
    performance_value: float
    team_value: float
    market_value: float
    social_value: float
    potential_value: float

    # Confidence and explanation
    confidence: str  # high, medium, low
    factors: Dict[str, float]  # Individual factor contributions
    explanation: str


class CustomNILValuator:
    """
    Proprietary NIL valuation based on measurable factors.

    Formula:
    NIL Value = (Performance Score × Position Multiplier × School Multiplier)
                + Social Media Value
                + Potential Premium
                × Market Adjustment

    Each component is calculated from available data.
    """

    # ==========================================================================
    # POSITION VALUE WEIGHTS
    # Based on typical NIL market demand by position
    # ==========================================================================
    POSITION_BASE_VALUES = {
        "QB": 500_000,      # Highest demand, face of program
        "WR": 150_000,      # High visibility, skill position
        "RB": 120_000,      # Skill position, shorter career
        "TE": 100_000,      # Versatile, growing value
        "EDGE": 100_000,    # Premium defensive position
        "CB": 90_000,       # Coverage skills valued
        "S": 80_000,        # Secondary leader
        "LB": 75_000,       # Defensive anchor
        "DL": 70_000,       # Interior presence
        "OL": 60_000,       # Less visibility but critical
        "K": 30_000,        # Specialist
        "P": 25_000,        # Specialist
        "LS": 15_000,       # Specialist
        "ATH": 100_000,     # Versatile, depends on usage
    }

    # ==========================================================================
    # SCHOOL TIER MULTIPLIERS (Updated Feb 2026)
    # Based on brand value, NIL collective strength, media exposure, recent success
    # NOTE: These should be refreshed quarterly based on team performance
    # ==========================================================================
    SCHOOL_MULTIPLIERS = {
        # Tier 5: Blue Bloods + National Champions (massive NIL operations)
        "Ohio State": 3.0, "Texas": 3.0, "Alabama": 3.0, "Georgia": 2.8,
        "Indiana": 2.8,     # 2025 National Champions - undefeated season!
        "USC": 2.5, "Michigan": 2.5, "Notre Dame": 2.5, "Oklahoma": 2.3,

        # Tier 4: Elite programs (CFP contenders, top recruiting)
        "Oregon": 2.3, "Penn State": 2.2, "Clemson": 2.2, "LSU": 2.2,
        "Tennessee": 2.1, "Texas A&M": 2.0, "Florida": 2.0, "Miami": 2.0,
        "Colorado": 2.0,    # Prime effect + portal success

        # Tier 3: Strong Power 4 (consistent top 25, strong NIL)
        "Ole Miss": 1.9, "Auburn": 1.8, "Wisconsin": 1.8, "Iowa": 1.7,
        "UCLA": 1.7, "Florida State": 1.6, "Arkansas": 1.6, "Kentucky": 1.6,
        "South Carolina": 1.6, "Missouri": 1.5, "NC State": 1.5, "Utah": 1.5,
        "Kansas": 1.5,      # Recent program turnaround

        # Tier 2: Mid Power 4 (solid programs, growing NIL)
        "Michigan State": 1.4, "Virginia Tech": 1.4, "Louisville": 1.4,
        "Baylor": 1.4, "Kansas State": 1.4, "Pittsburgh": 1.3, "Syracuse": 1.3,
        "Duke": 1.3, "West Virginia": 1.3, "TCU": 1.3, "Iowa State": 1.3,
        "Arizona": 1.3, "Arizona State": 1.3, "BYU": 1.3, "Cincinnati": 1.3,
        "Washington": 1.3, "Washington State": 1.2, "Oregon State": 1.2,
        "Minnesota": 1.2, "Illinois": 1.2, "Purdue": 1.2, "Nebraska": 1.3,
        "Maryland": 1.2, "Rutgers": 1.1, "Northwestern": 1.2, "Vanderbilt": 1.1,

        # Tier 1: G5 Powers (strong programs, limited NIL but growing)
        "SMU": 1.2, "Boise State": 1.1, "Memphis": 1.0, "UCF": 1.1,
        "Tulane": 1.0, "Liberty": 0.9, "App State": 0.9, "James Madison": 0.9,
        "UNLV": 0.9, "San Jose State": 0.8, "Fresno State": 0.8,
        "Air Force": 0.8, "Navy": 0.8, "Army": 0.8,
        "Troy": 0.7, "Marshall": 0.7, "Southern Miss": 0.7,
    }
    DEFAULT_SCHOOL_MULTIPLIER = 0.8

    # ==========================================================================
    # CONFERENCE MULTIPLIERS
    # Applied when school not in list
    # ==========================================================================
    CONFERENCE_MULTIPLIERS = {
        "SEC": 1.8,
        "Big Ten": 1.7,
        "Big 12": 1.3,
        "ACC": 1.3,
        "Pac-12": 1.2,  # Legacy
        "American": 0.9,
        "Mountain West": 0.8,
        "Sun Belt": 0.7,
        "MAC": 0.6,
        "Conference USA": 0.6,
        "FCS": 0.4,
    }
    DEFAULT_CONFERENCE_MULTIPLIER = 0.8

    # ==========================================================================
    # SOCIAL MEDIA VALUATION
    # CPM-based calculation for sponsorship value
    # ==========================================================================
    SOCIAL_CPM_RATES = {
        "instagram": 12.0,   # $ per 1000 impressions
        "tiktok": 8.0,
        "twitter": 4.0,
        "youtube": 18.0,
    }
    ENGAGEMENT_RATES = {
        "instagram": 0.03,   # 3% of followers see content
        "tiktok": 0.05,      # Higher engagement
        "twitter": 0.01,
        "youtube": 0.08,
    }
    POSTS_PER_MONTH = {
        "instagram": 8,
        "tiktok": 12,
        "twitter": 15,
        "youtube": 2,
    }

    # ==========================================================================
    # RECRUITING PREMIUM
    # Higher-rated recruits command premium even without production yet
    # ==========================================================================
    STAR_PREMIUMS = {
        5: 200_000,   # 5-star gets $200K floor
        4: 50_000,    # 4-star gets $50K floor
        3: 10_000,    # 3-star gets $10K floor
        2: 5_000,
        1: 0,
        0: 0,
    }

    # ==========================================================================
    # TIER THRESHOLDS
    # ==========================================================================
    TIERS = {
        "mega": 2_000_000,
        "premium": 500_000,
        "solid": 100_000,
        "moderate": 25_000,
        "entry": 0,
    }

    def __init__(self, calibration_factor: float = 1.0, use_dynamic_tiers: bool = True):
        """
        Initialize the valuator.

        Args:
            calibration_factor: Multiply all valuations by this factor
                               to calibrate against known market data
            use_dynamic_tiers: Use CFBD-based dynamic school tiers (default True)
        """
        self.calibration_factor = calibration_factor
        self.use_dynamic_tiers = use_dynamic_tiers and DYNAMIC_TIERS_AVAILABLE

    def calculate_valuation(
        self,
        player_name: str,
        position: str,
        school: str,
        conference: Optional[str] = None,

        # Performance stats
        games_played: int = 0,
        games_started: int = 0,
        passing_yards: int = 0,
        passing_tds: int = 0,
        rushing_yards: int = 0,
        rushing_tds: int = 0,
        receiving_yards: int = 0,
        receiving_tds: int = 0,
        tackles: int = 0,
        sacks: float = 0,
        interceptions: int = 0,
        pff_grade: Optional[float] = None,

        # Social media
        instagram_followers: int = 0,
        twitter_followers: int = 0,
        tiktok_followers: int = 0,
        youtube_subscribers: int = 0,

        # Recruiting
        recruiting_stars: int = 0,
        national_rank: Optional[int] = None,

        # Other factors
        is_starter: bool = False,
        years_remaining: int = 1,
        awards: List[str] = None,
    ) -> NILValuation:
        """
        Calculate NIL valuation for a player.

        Returns detailed breakdown of valuation components.
        """
        factors = {}

        # ==========================================================================
        # 1. BASE POSITION VALUE
        # ==========================================================================
        position_group = self._normalize_position(position)
        base_value = self.POSITION_BASE_VALUES.get(position_group, 50_000)
        factors["position_base"] = base_value

        # ==========================================================================
        # 2. PERFORMANCE MULTIPLIER
        # ==========================================================================
        perf_multiplier = self._calculate_performance_multiplier(
            position_group, games_played, games_started,
            passing_yards, passing_tds, rushing_yards, rushing_tds,
            receiving_yards, receiving_tds, tackles, sacks, interceptions,
            pff_grade
        )
        performance_value = base_value * perf_multiplier
        factors["performance_multiplier"] = perf_multiplier
        factors["performance_value"] = performance_value

        # ==========================================================================
        # 3. SCHOOL/MARKET MULTIPLIER (Dynamic or Static)
        # ==========================================================================
        if self.use_dynamic_tiers:
            # Use CFBD-based dynamic tiers (wins, SP+, talent, recruiting)
            school_mult = dynamic_school_multiplier(school)
            tier_name, tier_info = get_school_tier(school)
            factors["school_tier"] = tier_name
            factors["school_score"] = tier_info.get("score", 0)
        else:
            # Fallback to static multipliers
            school_mult = self.SCHOOL_MULTIPLIERS.get(school, None)
            if school_mult is None:
                school_mult = self.CONFERENCE_MULTIPLIERS.get(
                    conference, self.DEFAULT_CONFERENCE_MULTIPLIER
                )
        market_value = performance_value * school_mult
        factors["school_multiplier"] = school_mult
        factors["market_value"] = market_value

        # ==========================================================================
        # 4. SOCIAL MEDIA VALUE
        # ==========================================================================
        social_value = self._calculate_social_value(
            instagram_followers, twitter_followers,
            tiktok_followers, youtube_subscribers
        )
        factors["social_value"] = social_value

        # ==========================================================================
        # 5. POTENTIAL/RECRUITING PREMIUM
        # ==========================================================================
        potential_value = self._calculate_potential_premium(
            recruiting_stars, national_rank, years_remaining
        )
        factors["potential_value"] = potential_value

        # ==========================================================================
        # 6. STARTER BONUS
        # ==========================================================================
        starter_bonus = 1.3 if is_starter else 1.0
        factors["starter_bonus"] = starter_bonus

        # ==========================================================================
        # 7. AWARDS BONUS
        # ==========================================================================
        awards_bonus = self._calculate_awards_bonus(awards or [])
        factors["awards_bonus"] = awards_bonus

        # ==========================================================================
        # FINAL CALCULATION
        # ==========================================================================
        total = (
            (market_value * starter_bonus)
            + social_value
            + potential_value
            + awards_bonus
        ) * self.calibration_factor

        # Ensure minimum based on stars
        star_floor = self.STAR_PREMIUMS.get(recruiting_stars, 0)
        total = max(total, star_floor)

        # Round to nearest $1000
        total = round(total / 1000) * 1000

        # Determine tier
        tier = self._get_tier(total)

        # Determine confidence
        confidence = self._assess_confidence(
            games_played, instagram_followers + twitter_followers + tiktok_followers,
            pff_grade is not None
        )

        # Generate explanation
        explanation = self._generate_explanation(
            player_name, position_group, school, total, factors
        )

        return NILValuation(
            player_name=player_name,
            total_valuation=total,
            valuation_tier=tier,
            performance_value=performance_value,
            team_value=market_value,
            market_value=market_value,
            social_value=social_value,
            potential_value=potential_value,
            confidence=confidence,
            factors=factors,
            explanation=explanation,
        )

    def _normalize_position(self, position: str) -> str:
        """Map position to standard group."""
        position = position.upper().strip()
        mappings = {
            "QUARTERBACK": "QB", "RUNNINGBACK": "RB", "HB": "RB", "FB": "RB",
            "RECEIVER": "WR", "WIDE RECEIVER": "WR", "SE": "WR", "FL": "WR",
            "TIGHT END": "TE",
            "OFFENSIVE LINE": "OL", "OT": "OL", "OG": "OL", "C": "OL", "T": "OL", "G": "OL",
            "DEFENSIVE LINE": "DL", "DT": "DL", "DE": "DL", "NT": "DL",
            "OUTSIDE LINEBACKER": "EDGE", "OLB": "EDGE", "RUSH": "EDGE",
            "LINEBACKER": "LB", "ILB": "LB", "MLB": "LB",
            "CORNERBACK": "CB", "DB": "CB",
            "SAFETY": "S", "FS": "S", "SS": "S",
            "KICKER": "K", "PK": "K",
            "PUNTER": "P",
            "LONG SNAPPER": "LS",
            "ATHLETE": "ATH",
        }
        return mappings.get(position, position if position in self.POSITION_BASE_VALUES else "ATH")

    def _calculate_performance_multiplier(
        self, position: str, games: int, starts: int,
        pass_yds: int, pass_tds: int, rush_yds: int, rush_tds: int,
        rec_yds: int, rec_tds: int, tackles: int, sacks: float, ints: int,
        pff_grade: Optional[float]
    ) -> float:
        """Calculate performance multiplier based on stats."""
        if games == 0:
            return 0.5  # Minimal multiplier for no games

        mult = 1.0

        # Position-specific performance scoring
        if position == "QB":
            # QBs: yards, TDs, efficiency
            if pass_yds > 3000: mult += 0.8
            elif pass_yds > 2000: mult += 0.5
            elif pass_yds > 1000: mult += 0.2
            if pass_tds > 25: mult += 0.5
            elif pass_tds > 15: mult += 0.3
            # Dual threat bonus
            if rush_yds > 500: mult += 0.3

        elif position in ["RB"]:
            if rush_yds > 1000: mult += 0.6
            elif rush_yds > 500: mult += 0.3
            if rush_tds > 10: mult += 0.4
            elif rush_tds > 5: mult += 0.2
            # Receiving bonus
            if rec_yds > 300: mult += 0.2

        elif position == "WR":
            if rec_yds > 1000: mult += 0.7
            elif rec_yds > 600: mult += 0.4
            elif rec_yds > 300: mult += 0.2
            if rec_tds > 10: mult += 0.4
            elif rec_tds > 5: mult += 0.2

        elif position == "TE":
            if rec_yds > 600: mult += 0.5
            elif rec_yds > 300: mult += 0.3
            if rec_tds > 5: mult += 0.3

        elif position in ["EDGE", "DL"]:
            if sacks > 10: mult += 0.7
            elif sacks > 5: mult += 0.4
            elif sacks > 2: mult += 0.2

        elif position == "LB":
            if tackles > 100: mult += 0.5
            elif tackles > 70: mult += 0.3
            if sacks > 3: mult += 0.2

        elif position in ["CB", "S"]:
            if ints > 5: mult += 0.6
            elif ints > 2: mult += 0.3
            if tackles > 50: mult += 0.2

        # PFF grade bonus (applies to all)
        if pff_grade:
            if pff_grade > 90: mult += 0.5
            elif pff_grade > 80: mult += 0.3
            elif pff_grade > 70: mult += 0.1

        # Starter bonus already applied elsewhere, but games started matters
        start_rate = starts / max(games, 1)
        if start_rate > 0.8: mult += 0.2

        return max(mult, 0.3)  # Minimum multiplier

    def _calculate_social_value(
        self, ig: int, tw: int, tt: int, yt: int
    ) -> float:
        """Calculate annual social media sponsorship value."""
        total = 0.0

        # Instagram value
        if ig > 0:
            impressions = ig * self.ENGAGEMENT_RATES["instagram"] * self.POSTS_PER_MONTH["instagram"] * 12
            total += impressions * self.SOCIAL_CPM_RATES["instagram"] / 1000

        # TikTok value
        if tt > 0:
            impressions = tt * self.ENGAGEMENT_RATES["tiktok"] * self.POSTS_PER_MONTH["tiktok"] * 12
            total += impressions * self.SOCIAL_CPM_RATES["tiktok"] / 1000

        # Twitter value
        if tw > 0:
            impressions = tw * self.ENGAGEMENT_RATES["twitter"] * self.POSTS_PER_MONTH["twitter"] * 12
            total += impressions * self.SOCIAL_CPM_RATES["twitter"] / 1000

        # YouTube value
        if yt > 0:
            impressions = yt * self.ENGAGEMENT_RATES["youtube"] * self.POSTS_PER_MONTH["youtube"] * 12
            total += impressions * self.SOCIAL_CPM_RATES["youtube"] / 1000

        return total

    def _calculate_potential_premium(
        self, stars: int, national_rank: Optional[int], years_remaining: int
    ) -> float:
        """Calculate premium for recruiting profile and remaining eligibility."""
        premium = self.STAR_PREMIUMS.get(stars, 0)

        # Top recruit bonus
        if national_rank and national_rank <= 10:
            premium += 100_000
        elif national_rank and national_rank <= 50:
            premium += 50_000
        elif national_rank and national_rank <= 100:
            premium += 25_000

        # Years remaining multiplier (more years = more total value)
        year_mult = 1.0 + (years_remaining - 1) * 0.1  # +10% per extra year
        premium *= year_mult

        return premium

    def _calculate_awards_bonus(self, awards: List[str]) -> float:
        """Calculate bonus for awards and accolades."""
        bonus = 0

        award_values = {
            "heisman": 500_000,
            "heisman finalist": 200_000,
            "all-american": 150_000,
            "first-team all-conference": 75_000,
            "conference player of year": 100_000,
            "freshman all-american": 50_000,
            "all-conference": 40_000,
        }

        for award in awards:
            award_lower = award.lower()
            for key, value in award_values.items():
                if key in award_lower:
                    bonus += value
                    break

        return bonus

    def _get_tier(self, value: float) -> str:
        """Get valuation tier."""
        for tier, threshold in self.TIERS.items():
            if value >= threshold:
                return tier
        return "entry"

    def _assess_confidence(
        self, games: int, total_followers: int, has_pff: bool
    ) -> str:
        """Assess confidence in valuation."""
        score = 0
        if games >= 10: score += 2
        elif games >= 5: score += 1
        if total_followers > 50000: score += 1
        if has_pff: score += 1

        if score >= 3:
            return "high"
        elif score >= 1:
            return "medium"
        return "low"

    def _generate_explanation(
        self, name: str, position: str, school: str,
        total: float, factors: Dict
    ) -> str:
        """Generate human-readable explanation."""
        parts = [
            f"{name} ({position}, {school})",
            f"Valuation: ${total:,.0f}",
            f"",
            f"Breakdown:",
            f"  Base position value: ${factors.get('position_base', 0):,.0f}",
            f"  Performance multiplier: {factors.get('performance_multiplier', 1):.2f}x",
            f"  School/market multiplier: {factors.get('school_multiplier', 1):.2f}x",
            f"  Social media value: ${factors.get('social_value', 0):,.0f}",
            f"  Potential premium: ${factors.get('potential_value', 0):,.0f}",
        ]
        return "\n".join(parts)

    def valuate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valuate all players in a DataFrame.

        Expected columns: player_name, position, school, and various stats.
        Returns DataFrame with valuation columns added.
        """
        valuations = []

        for _, row in df.iterrows():
            val = self.calculate_valuation(
                player_name=row.get("player_name", row.get("name", "Unknown")),
                position=row.get("position", "ATH"),
                school=row.get("school", "Unknown"),
                conference=row.get("conference"),
                games_played=int(row.get("games_played", 0) or 0),
                games_started=int(row.get("games_started", 0) or 0),
                passing_yards=int(row.get("passing_yards", 0) or 0),
                passing_tds=int(row.get("passing_tds", 0) or 0),
                rushing_yards=int(row.get("rushing_yards", 0) or 0),
                rushing_tds=int(row.get("rushing_tds", 0) or 0),
                receiving_yards=int(row.get("receiving_yards", 0) or 0),
                receiving_tds=int(row.get("receiving_tds", 0) or 0),
                tackles=int(row.get("tackles", 0) or 0),
                sacks=float(row.get("sacks", 0) or 0),
                interceptions=int(row.get("interceptions", 0) or 0),
                pff_grade=row.get("pff_grade"),
                instagram_followers=int(row.get("instagram_followers", 0) or 0),
                twitter_followers=int(row.get("twitter_followers", 0) or 0),
                tiktok_followers=int(row.get("tiktok_followers", 0) or 0),
                youtube_subscribers=int(row.get("youtube_subscribers", 0) or 0),
                recruiting_stars=int(row.get("recruiting_stars", 0) or 0),
                national_rank=row.get("national_rank") or row.get("recruiting_rank"),
                is_starter=row.get("is_starter", False),
                years_remaining=int(row.get("years_remaining", 1) or 1),
            )
            valuations.append({
                "custom_nil_value": val.total_valuation,
                "nil_tier": val.valuation_tier,
                "performance_value": val.performance_value,
                "market_value": val.market_value,
                "social_value": val.social_value,
                "potential_value": val.potential_value,
                "valuation_confidence": val.confidence,
            })

        val_df = pd.DataFrame(valuations)
        return pd.concat([df.reset_index(drop=True), val_df], axis=1)


# Example usage
if __name__ == "__main__":
    valuator = CustomNILValuator()

    # Example: Travis Hunter
    result = valuator.calculate_valuation(
        player_name="Travis Hunter",
        position="CB",
        school="Colorado",
        conference="Big 12",
        games_played=12,
        games_started=12,
        interceptions=4,
        tackles=55,
        receiving_yards=1150,
        receiving_tds=14,
        pff_grade=92.5,
        instagram_followers=2_500_000,
        twitter_followers=500_000,
        tiktok_followers=1_000_000,
        recruiting_stars=5,
        national_rank=1,
        is_starter=True,
        years_remaining=1,
        awards=["Heisman", "First-Team All-American"],
    )

    print(result.explanation)
    print(f"\nTotal: ${result.total_valuation:,.0f}")
    print(f"Tier: {result.valuation_tier}")
    print(f"Confidence: {result.confidence}")
