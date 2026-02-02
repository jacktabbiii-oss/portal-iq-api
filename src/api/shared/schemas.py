"""
Shared API Schemas

Common Pydantic models used across Portal IQ and Cap IQ.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class League(str, Enum):
    COLLEGE = "college"
    NFL = "nfl"


class PlayerType(str, Enum):
    COLLEGE = "college"
    NFL = "nfl"
    DRAFT_PROSPECT = "draft_prospect"


# =============================================================================
# PLAYER SEARCH SCHEMAS
# =============================================================================

class PlayerSearchRequest(BaseModel):
    """Request for player search."""
    query: str
    league: Optional[League] = None
    position: Optional[str] = None
    team: Optional[str] = None
    limit: int = 20

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Travis Hunter",
                "league": "college",
                "limit": 10,
            }
        }


class PlayerSearchResult(BaseModel):
    """Single player search result."""
    player_id: str
    player_name: str
    position: str
    team: str
    league: League
    match_score: float
    additional_info: Optional[Dict[str, Any]] = None


class PlayerSearchResponse(BaseModel):
    """Response for player search."""
    results: List[PlayerSearchResult]
    total: int


# =============================================================================
# PLAYER LOOKUP SCHEMAS
# =============================================================================

class CollegePlayerInfo(BaseModel):
    """College player information."""
    player_id: str
    player_name: str
    position: str
    school: str
    conference: str
    class_year: str
    height: Optional[str] = None
    weight: Optional[int] = None
    hometown: Optional[str] = None
    recruiting_stars: Optional[int] = None
    nil_valuation: Optional[float] = None


class NFLPlayerInfo(BaseModel):
    """NFL player information."""
    player_id: str
    player_name: str
    position: str
    team: str
    jersey_number: Optional[int] = None
    age: Optional[int] = None
    experience: Optional[int] = None
    college: Optional[str] = None
    draft_year: Optional[int] = None
    draft_round: Optional[int] = None
    draft_pick: Optional[int] = None
    contract_aav: Optional[float] = None
    war: Optional[float] = None


class PlayerLookupResponse(BaseModel):
    """Response for player lookup."""
    player_id: str
    player_type: PlayerType
    college_info: Optional[CollegePlayerInfo] = None
    nfl_info: Optional[NFLPlayerInfo] = None


# =============================================================================
# COLLEGE TO NFL MAPPING
# =============================================================================

class CollegeToNFLMappingResponse(BaseModel):
    """Response for college to NFL mapping."""
    college_player_id: str
    nfl_player_id: Optional[str] = None
    draft_year: Optional[int] = None
    draft_round: Optional[int] = None
    draft_pick: Optional[int] = None
    nfl_team: Optional[str] = None
    is_active: bool = False


# =============================================================================
# ERROR RESPONSES
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    status_code: int = 500


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = "Validation error"
    detail: List[Dict[str, Any]]
    status_code: int = 422


# =============================================================================
# PAGINATION
# =============================================================================

class PaginatedRequest(BaseModel):
    """Base model for paginated requests."""
    page: int = 1
    page_size: int = 50


class PaginatedResponse(BaseModel):
    """Base model for paginated responses."""
    total: int
    page: int
    page_size: int
    has_more: bool


# =============================================================================
# HEALTH & STATUS
# =============================================================================

class ServiceStatus(BaseModel):
    """Status of a service component."""
    status: str  # ok, degraded, error
    last_update: Optional[datetime] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    services: Dict[str, ServiceStatus]
