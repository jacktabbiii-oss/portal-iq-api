"""
Shared API Routes

Common endpoints used by both Portal IQ and Cap IQ.
Player lookup, search, matching, and cross-product features.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import logging

from ..auth import get_current_user
from .schemas import (
    PlayerSearchRequest,
    PlayerSearchResponse,
    PlayerLookupResponse,
    CollegeToNFLMappingResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# PLAYER SEARCH & LOOKUP
# =============================================================================

@router.post("/players/search", response_model=PlayerSearchResponse)
async def search_players(
    request: PlayerSearchRequest,
    user: dict = Depends(get_current_user),
):
    """
    Unified player search across college and NFL.

    Useful for finding players across products.
    """
    logger.info(f"Player search: '{request.query}' by {user['email']}")

    # TODO: Search across both databases
    return PlayerSearchResponse(
        results=[],
        total=0,
    )


@router.get("/players/{player_id}", response_model=PlayerLookupResponse)
async def lookup_player(
    player_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Look up a player by ID.

    Works for both college and NFL players.
    """
    # TODO: Look up player
    raise HTTPException(status_code=404, detail="Player not found")


@router.get("/players/match")
async def match_players(
    name: str = Query(..., description="Player name to match"),
    college: Optional[str] = Query(None, description="College for disambiguation"),
    position: Optional[str] = Query(None, description="Position for disambiguation"),
    user: dict = Depends(get_current_user),
):
    """
    Fuzzy match a player name to our database.

    Useful for matching external data sources.
    """
    # TODO: Use player matching logic
    return {
        "query": name,
        "matches": [],
        "best_match": None,
    }


# =============================================================================
# COLLEGE TO NFL MAPPING
# =============================================================================

@router.get("/players/college-to-nfl/{player_id}", response_model=CollegeToNFLMappingResponse)
async def get_college_to_nfl_mapping(
    player_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Map a college player to their NFL profile.

    Links Portal IQ and Cap IQ data for drafted players.
    """
    return CollegeToNFLMappingResponse(
        college_player_id=player_id,
        nfl_player_id=None,
        draft_year=None,
        draft_round=None,
        draft_pick=None,
        nfl_team=None,
    )


@router.get("/players/nfl-to-college/{nfl_player_id}")
async def get_nfl_to_college_mapping(
    nfl_player_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Map an NFL player to their college profile.
    """
    return {
        "nfl_player_id": nfl_player_id,
        "college_player_id": None,
        "college": None,
        "college_stats": {},
    }


# =============================================================================
# SCHOOL/TEAM ENDPOINTS
# =============================================================================

@router.get("/schools")
async def get_schools(
    conference: Optional[str] = Query(None),
    division: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    """
    Get list of college football schools.
    """
    return {"schools": [], "total": 0}


@router.get("/nfl-teams")
async def get_nfl_teams(
    conference: Optional[str] = Query(None),
    division: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    """
    Get list of NFL teams.
    """
    teams = [
        {"abbr": "ARI", "name": "Arizona Cardinals", "conference": "NFC", "division": "West"},
        {"abbr": "ATL", "name": "Atlanta Falcons", "conference": "NFC", "division": "South"},
        {"abbr": "BAL", "name": "Baltimore Ravens", "conference": "AFC", "division": "North"},
        {"abbr": "BUF", "name": "Buffalo Bills", "conference": "AFC", "division": "East"},
        # ... etc
    ]

    if conference:
        teams = [t for t in teams if t["conference"] == conference]
    if division:
        teams = [t for t in teams if t["division"] == division]

    return {"teams": teams, "total": len(teams)}


# =============================================================================
# POSITION MAPPINGS
# =============================================================================

@router.get("/positions")
async def get_positions(
    league: Optional[str] = Query(None, regex="^(college|nfl)$"),
    user: dict = Depends(get_current_user),
):
    """
    Get position definitions and mappings.
    """
    positions = {
        "offense": {
            "QB": "Quarterback",
            "RB": "Running Back",
            "FB": "Fullback",
            "WR": "Wide Receiver",
            "TE": "Tight End",
            "OT": "Offensive Tackle",
            "OG": "Offensive Guard",
            "C": "Center",
        },
        "defense": {
            "DE": "Defensive End",
            "DT": "Defensive Tackle",
            "NT": "Nose Tackle",
            "EDGE": "Edge Rusher",
            "LB": "Linebacker",
            "ILB": "Inside Linebacker",
            "OLB": "Outside Linebacker",
            "CB": "Cornerback",
            "S": "Safety",
            "FS": "Free Safety",
            "SS": "Strong Safety",
        },
        "special_teams": {
            "K": "Kicker",
            "P": "Punter",
            "LS": "Long Snapper",
        }
    }

    return positions


@router.get("/position-mappings")
async def get_position_mappings(
    user: dict = Depends(get_current_user),
):
    """
    Get mappings between college and NFL position nomenclature.
    """
    return {
        "college_to_nfl": {
            "EDGE": ["DE", "OLB"],
            "OL": ["OT", "OG", "C"],
            "DL": ["DE", "DT", "NT"],
            "DB": ["CB", "S", "FS", "SS"],
        },
        "nfl_to_college": {
            "DE": ["EDGE", "DL"],
            "OLB": ["EDGE", "LB"],
            # etc
        }
    }


# =============================================================================
# DATA REFRESH STATUS
# =============================================================================

@router.get("/data/status")
async def get_data_status(
    user: dict = Depends(get_current_user),
):
    """
    Get status of data sources and last refresh times.
    """
    return {
        "portal_iq": {
            "portal_data": {"last_refresh": None, "status": "ok"},
            "nil_data": {"last_refresh": None, "status": "ok"},
            "roster_data": {"last_refresh": None, "status": "ok"},
        },
        "cap_iq": {
            "contract_data": {"last_refresh": None, "status": "ok"},
            "cap_data": {"last_refresh": None, "status": "ok"},
            "injury_data": {"last_refresh": None, "status": "ok"},
        },
    }
