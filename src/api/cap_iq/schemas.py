"""
Cap IQ API Schemas

Pydantic models for NFL salary cap analysis request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class ContractType(str, Enum):
    ROOKIE = "rookie"
    VETERAN = "veteran"
    FRANCHISE_TAG = "franchise_tag"
    TRANSITION_TAG = "transition_tag"


class FAType(str, Enum):
    UFA = "ufa"
    RFA = "rfa"
    ERFA = "erfa"


class MoveType(str, Enum):
    CUT = "cut"
    TRADE = "trade"
    RESTRUCTURE = "restructure"
    EXTENSION = "extension"
    SIGNING = "signing"


# =============================================================================
# CONTRACT SCHEMAS
# =============================================================================

class ContractPredictionRequest(BaseModel):
    """Request for contract prediction."""
    player_id: Optional[str] = None
    player_name: str
    position: str
    age: int
    experience: int
    stats: Optional[Dict[str, Any]] = None
    war: Optional[float] = None
    injuries: Optional[List[str]] = None
    market_year: int = 2025

    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "Ja'Marr Chase",
                "position": "WR",
                "age": 24,
                "experience": 3,
                "war": 3.5,
                "market_year": 2025,
            }
        }


class ComparableContract(BaseModel):
    """Historical comparable contract."""
    player_name: str
    position: str
    year_signed: int
    aav: float
    total_value: float
    years: int
    guaranteed: float
    similarity_score: float


class ContractPredictionResponse(BaseModel):
    """Response for contract prediction."""
    player_name: str
    position: str
    predicted_aav: float
    predicted_total: float
    predicted_years: int
    predicted_guaranteed: float
    confidence: float
    comparable_contracts: List[ComparableContract]
    range_low: Optional[float] = None
    range_high: Optional[float] = None


class ContractSearchRequest(BaseModel):
    """Contract search filters."""
    positions: Optional[List[str]] = None
    min_aav: Optional[float] = None
    max_aav: Optional[float] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    team: Optional[str] = None


class ContractEntry(BaseModel):
    """Contract entry."""
    player_id: str
    player_name: str
    position: str
    team: str
    aav: float
    total_value: float
    years: int
    guaranteed: float
    year_signed: int
    expiration_year: int


class ContractSearchResponse(BaseModel):
    """Contract search response."""
    contracts: List[ContractEntry]
    total: int


# =============================================================================
# SURPLUS VALUE SCHEMAS
# =============================================================================

class SurplusValueRequest(BaseModel):
    """Request for surplus value calculation."""
    player_id: Optional[str] = None
    player_name: str
    position: str
    cap_hit: float
    war: Optional[float] = None
    age: Optional[int] = None


class SurplusValueResponse(BaseModel):
    """Response for surplus value calculation."""
    player_name: str
    position: str
    expected_value: float
    cap_hit: float
    surplus_value: float
    surplus_rank: int
    percentile: float


# =============================================================================
# CAP ANALYSIS SCHEMAS
# =============================================================================

class PositionCapBreakdown(BaseModel):
    """Cap breakdown by position."""
    position: str
    total_cap: float
    player_count: int
    percentage: float


class TeamCapResponse(BaseModel):
    """Team cap situation response."""
    team: str
    year: int
    cap_limit: float
    cap_spent: float
    cap_space: float
    dead_money: float
    top_51: bool
    positions: Dict[str, PositionCapBreakdown]


class CapMove(BaseModel):
    """Hypothetical cap move."""
    move_type: MoveType
    player_id: Optional[str] = None
    player_name: str
    cap_impact: float
    dead_money: Optional[float] = None


class CapProjectionRequest(BaseModel):
    """Request for cap projection."""
    team: str
    years: int = 3
    hypothetical_moves: Optional[List[CapMove]] = None


class YearlyProjection(BaseModel):
    """Cap projection for a single year."""
    year: int
    projected_cap: float
    committed_cap: float
    projected_space: float
    notable_expirations: List[str]


class CapProjectionResponse(BaseModel):
    """Response for cap projection."""
    team: str
    projections: List[YearlyProjection]


# =============================================================================
# CAP OPTIMIZATION SCHEMAS
# =============================================================================

class CapOptimizationRequest(BaseModel):
    """Request for cap optimization."""
    team: str
    target_cap_space: float
    allow_cuts: bool = True
    allow_trades: bool = True
    allow_restructures: bool = True
    protected_players: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "team": "DAL",
                "target_cap_space": 20000000,
                "allow_cuts": True,
                "allow_trades": True,
                "allow_restructures": True,
            }
        }


class RecommendedMove(BaseModel):
    """Recommended roster move."""
    move_type: MoveType
    player_name: str
    position: str
    cap_savings: float
    dead_money: float
    war_lost: float
    trade_value: Optional[str] = None


class CapOptimizationResponse(BaseModel):
    """Response for cap optimization."""
    team: str
    status: str
    recommended_moves: List[RecommendedMove]
    cap_savings: float
    war_change: float
    new_cap_space: Optional[float] = None


# =============================================================================
# PLAYER VALUE SCHEMAS
# =============================================================================

class PlayerValueRequest(BaseModel):
    """Request for player value calculation."""
    player_id: Optional[str] = None
    player_name: str
    position: str
    age: int
    war: Optional[float] = None
    stats: Optional[Dict[str, Any]] = None


class PlayerValueResponse(BaseModel):
    """Response for player value calculation."""
    player_name: str
    position: str
    market_value: float
    war: float
    value_breakdown: Dict[str, float]
    percentile: Optional[float] = None


class PlayerAgingCurveRequest(BaseModel):
    """Request for aging curve."""
    player_id: Optional[str] = None
    player_name: str
    position: str
    current_age: int
    current_war: Optional[float] = None


class AgingProjection(BaseModel):
    """Single year aging projection."""
    age: int
    year: int
    projected_war: float
    projected_value: float
    confidence: float


class PlayerAgingCurveResponse(BaseModel):
    """Response for aging curve."""
    player_name: str
    position: str
    current_age: int
    peak_age: int
    projections: List[AgingProjection]


# =============================================================================
# COMPARISON SCHEMAS
# =============================================================================

class ContractComparisonRequest(BaseModel):
    """Request for contract comparison."""
    player_ids: List[str]


class ComparedContract(BaseModel):
    """Contract in comparison."""
    player_id: str
    player_name: str
    position: str
    team: str
    aav: float
    total_value: float
    years: int
    guaranteed: float
    surplus_value: float
    war: Optional[float] = None


class ContractComparisonResponse(BaseModel):
    """Response for contract comparison."""
    players: List[ComparedContract]
    metrics: Dict[str, Any]
    best_value: Optional[str] = None


# =============================================================================
# FREE AGENCY SCHEMAS
# =============================================================================

class FreeAgentEntry(BaseModel):
    """Free agent entry."""
    player_id: str
    player_name: str
    position: str
    current_team: str
    fa_type: FAType
    age: int
    projected_aav: float
    war: Optional[float] = None


class FreeAgentMarketResponse(BaseModel):
    """Free agent market response."""
    position: str
    year: int
    market_size: int
    projected_top_aav: float
    depth_rating: str  # shallow, average, deep
    top_free_agents: List[FreeAgentEntry]
