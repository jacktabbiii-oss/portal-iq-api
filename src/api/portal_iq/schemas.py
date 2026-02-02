"""
Portal IQ API Schemas

Pydantic models for request/response validation.
These schemas match the data structures expected by the PocketBase frontend.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class NILTier(str, Enum):
    MEGA = "mega"
    PREMIUM = "premium"
    SOLID = "solid"
    MODERATE = "moderate"
    ENTRY = "entry"


class PortalStatus(str, Enum):
    AVAILABLE = "available"
    COMMITTED = "committed"
    WITHDRAWN = "withdrawn"


class ComparisonType(str, Enum):
    NIL = "nil_comparison"
    PORTAL = "portal_comparison"
    DRAFT = "draft_comparison"


# =============================================================================
# NIL VALUATION SCHEMAS
# =============================================================================

class NILValuationRequest(BaseModel):
    """Request for NIL valuation."""
    player_id: Optional[str] = None
    player_name: str
    position: str
    school: str
    class_year: Optional[str] = None
    social_followers: Optional[int] = 0
    engagement_rate: Optional[float] = 0.0
    recruiting_stars: Optional[int] = None
    stats: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "John Smith",
                "position": "QB",
                "school": "Alabama",
                "class_year": "JR",
                "social_followers": 150000,
                "engagement_rate": 3.5,
                "recruiting_stars": 4,
            }
        }


class NILValuationResponse(BaseModel):
    """Response for NIL valuation."""
    player_name: str
    valuation: float
    valuation_low: float
    valuation_high: float
    nil_tier: str
    breakdown: Optional[Dict[str, float]] = None
    confidence: float = 0.7

    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "John Smith",
                "valuation": 750000,
                "valuation_low": 500000,
                "valuation_high": 1000000,
                "nil_tier": "premium",
                "confidence": 0.75,
            }
        }


class NILBulkValuationRequest(BaseModel):
    """Request for bulk NIL valuations."""
    players: List[NILValuationRequest]


class NILPlayerValuation(BaseModel):
    """Single player valuation in bulk response."""
    player_name: str
    position: str
    school: str
    valuation: float
    nil_tier: str


class NILBulkValuationResponse(BaseModel):
    """Response for bulk NIL valuations."""
    valuations: List[NILPlayerValuation]
    total_value: float
    count: int


class NILLeaderboardPlayer(BaseModel):
    """Player entry in NIL leaderboard."""
    rank: int
    player_id: str
    player_name: str
    position: str
    school: str
    valuation: float
    nil_tier: str
    social_followers: Optional[int] = None


class NILLeaderboardResponse(BaseModel):
    """NIL leaderboard response."""
    players: List[NILLeaderboardPlayer]
    total: int


# =============================================================================
# PORTAL SCHEMAS
# =============================================================================

class PortalPlayerResponse(BaseModel):
    """Portal player data."""
    player_id: str
    player_name: str
    position: str
    origin_school: str
    origin_conference: Optional[str] = None
    destination_school: Optional[str] = None
    stars: Optional[int] = None
    entry_date: Optional[datetime] = None
    status: PortalStatus
    nil_valuation: Optional[float] = None
    days_in_portal: Optional[int] = None


class PortalPredictionRequest(BaseModel):
    """Request for portal prediction."""
    player_id: Optional[str] = None
    player_name: str
    position: str
    current_school: str
    class_year: str
    is_starter: bool = False
    snap_trend: Optional[float] = None
    coaching_change: bool = False
    current_nil: Optional[float] = None


class DestinationPrediction(BaseModel):
    """Predicted destination."""
    school: str
    probability: float
    fit_score: float
    nil_potential: Optional[float] = None
    position_need: Optional[str] = None


class PortalPredictionResponse(BaseModel):
    """Response for portal prediction."""
    player_name: str
    entry_probability: float
    likely_destinations: List[DestinationPrediction]
    risk_factors: List[str]


class PortalSearchRequest(BaseModel):
    """Portal search parameters."""
    positions: Optional[List[str]] = None
    min_stars: Optional[int] = None
    max_nil: Optional[float] = None
    origin_conferences: Optional[List[str]] = None
    status: Optional[PortalStatus] = None


class AtRiskPlayer(BaseModel):
    """At-risk player data."""
    player_id: str
    player_name: str
    position: str
    class_year: str
    portal_probability: float
    risk_factors: List[str]
    depth_chart_position: Optional[int] = None
    snap_trend: Optional[float] = None


class AtRiskPlayersRequest(BaseModel):
    """Request for at-risk analysis."""
    school: str
    threshold: float = 0.5


class AtRiskPlayersResponse(BaseModel):
    """Response for at-risk analysis."""
    school: str
    at_risk_players: List[AtRiskPlayer]
    total_at_risk: int


# =============================================================================
# DRAFT PROJECTION SCHEMAS
# =============================================================================

class DraftProjectionRequest(BaseModel):
    """Request for draft projection."""
    player_id: Optional[str] = None
    player_name: str
    position: str
    school: str
    class_year: str
    height: Optional[int] = None  # inches
    weight: Optional[int] = None  # pounds
    career_stats: Optional[Dict[str, Any]] = None
    combine_data: Optional[Dict[str, float]] = None
    recruiting_stars: Optional[int] = None


class ComparablePlayer(BaseModel):
    """Historical comparable player."""
    name: str
    draft_year: int
    draft_pick: int
    nfl_team: str
    similarity_score: float
    career_summary: Optional[str] = None


class DraftProjectionResponse(BaseModel):
    """Response for draft projection."""
    player_name: str
    draft_probability: float
    projected_round: int
    projected_pick: int
    draft_grade: float
    comparable_players: List[ComparablePlayer]
    career_value_projection: Optional[Dict[str, float]] = None
    strengths: Optional[List[str]] = None
    weaknesses: Optional[List[str]] = None


class DraftBoardPlayer(BaseModel):
    """Player on draft board."""
    rank: int
    player_id: str
    player_name: str
    position: str
    school: str
    draft_grade: float
    projected_round: int
    projected_pick: int


class DraftBoardResponse(BaseModel):
    """Draft board response."""
    players: List[DraftBoardPlayer]
    last_updated: Optional[datetime] = None


# =============================================================================
# ROSTER OPTIMIZATION SCHEMAS
# =============================================================================

class PositionNeed(BaseModel):
    """Position need definition."""
    position: str
    priority: str  # critical, high, medium, low
    count_needed: int


class RosterOptimizationRequest(BaseModel):
    """Request for roster optimization."""
    school: str
    budget: float
    max_additions: int = 10
    position_needs: Optional[List[PositionNeed]] = None
    excluded_players: Optional[List[str]] = None
    must_have_players: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "school": "Texas",
                "budget": 5000000,
                "max_additions": 5,
                "position_needs": [
                    {"position": "CB", "priority": "critical", "count_needed": 2},
                    {"position": "WR", "priority": "high", "count_needed": 1},
                ],
            }
        }


class SelectedPlayer(BaseModel):
    """Player selected in optimization."""
    player_id: str
    player_name: str
    position: str
    origin_school: str
    war: float
    nil_cost: float
    fit_score: float


class RosterOptimizationResponse(BaseModel):
    """Response for roster optimization."""
    status: str
    selected_players: List[SelectedPlayer]
    total_war: float
    total_cost: float
    budget_remaining: float
    projected_wins_added: float
    position_needs_filled: Optional[Dict[str, int]] = None


class RosterMove(BaseModel):
    """Single roster move."""
    move_type: str  # add_portal, add_recruit, remove, transfer_out
    player_id: Optional[str] = None
    player_name: str
    position: str
    nil_cost: Optional[float] = None


class RosterScenarioRequest(BaseModel):
    """Request for roster scenario evaluation."""
    school: str
    name: str
    moves: List[RosterMove]


class RosterScenarioResponse(BaseModel):
    """Response for roster scenario."""
    school: str
    scenario_name: str
    current_projected_wins: float
    new_projected_wins: float
    win_delta: float
    roster_grade: str
    position_grades: Dict[str, str]
    move_impacts: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# COMPARISON SCHEMAS
# =============================================================================

class PlayerComparisonRequest(BaseModel):
    """Request for player comparison."""
    player_ids: List[str]
    comparison_type: ComparisonType


class ComparedPlayer(BaseModel):
    """Player in comparison."""
    player_id: str
    player_name: str
    position: str
    school: str
    metrics: Dict[str, Any]


class PlayerComparisonResponse(BaseModel):
    """Response for player comparison."""
    players: List[ComparedPlayer]
    comparison_type: ComparisonType
    metrics: Dict[str, Any]
    recommendation: Optional[str] = None
