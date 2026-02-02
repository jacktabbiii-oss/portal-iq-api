"""Pydantic schemas for Portal IQ API request/response models."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# Base Response Models
# =============================================================================

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    status: str = Field(default="success", description="Response status: 'success' or 'error'")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response payload")
    message: Optional[str] = Field(default=None, description="Human-readable message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "data": {"predicted_value": 500000},
                "message": None,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = Field(default="error", description="Always 'error' for error responses")
    message: str = Field(..., description="Error description")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Invalid player data",
                "detail": "Missing required field: position",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# =============================================================================
# Shared Player Profile Models
# =============================================================================

class SocialMediaProfile(BaseModel):
    """Social media metrics for a player."""
    instagram_followers: Optional[int] = Field(default=0, ge=0, description="Instagram follower count")
    twitter_followers: Optional[int] = Field(default=0, ge=0, description="Twitter/X follower count")
    tiktok_followers: Optional[int] = Field(default=0, ge=0, description="TikTok follower count")
    engagement_rate: Optional[float] = Field(default=0.0, ge=0, le=1, description="Average engagement rate")

    class Config:
        json_schema_extra = {
            "example": {
                "instagram_followers": 150000,
                "twitter_followers": 50000,
                "tiktok_followers": 200000,
                "engagement_rate": 0.045
            }
        }


class RecruitingProfile(BaseModel):
    """Recruiting background information."""
    stars: Optional[int] = Field(default=3, ge=2, le=5, description="Recruiting star rating (2-5)")
    national_rank: Optional[int] = Field(default=None, ge=1, description="National recruiting rank")
    position_rank: Optional[int] = Field(default=None, ge=1, description="Position recruiting rank")
    state_rank: Optional[int] = Field(default=None, ge=1, description="State recruiting rank")

    class Config:
        json_schema_extra = {
            "example": {
                "stars": 4,
                "national_rank": 150,
                "position_rank": 12,
                "state_rank": 8
            }
        }


class PlayerStats(BaseModel):
    """Player performance statistics (position-flexible)."""
    games_played: Optional[int] = Field(default=0, ge=0, description="Games played")
    games_started: Optional[int] = Field(default=0, ge=0, description="Games started")

    # QB stats
    passing_yards: Optional[int] = Field(default=None, description="Passing yards (QB)")
    passing_tds: Optional[int] = Field(default=None, description="Passing touchdowns (QB)")
    interceptions: Optional[int] = Field(default=None, description="Interceptions thrown (QB)")
    completion_pct: Optional[float] = Field(default=None, description="Completion percentage (QB)")
    qbr: Optional[float] = Field(default=None, description="QBR rating (QB)")

    # Rushing stats (RB/QB)
    rushing_yards: Optional[int] = Field(default=None, description="Rushing yards")
    rushing_tds: Optional[int] = Field(default=None, description="Rushing touchdowns")
    yards_per_carry: Optional[float] = Field(default=None, description="Yards per carry")

    # Receiving stats (WR/TE/RB)
    receptions: Optional[int] = Field(default=None, description="Receptions")
    receiving_yards: Optional[int] = Field(default=None, description="Receiving yards")
    receiving_tds: Optional[int] = Field(default=None, description="Receiving touchdowns")
    yards_per_reception: Optional[float] = Field(default=None, description="Yards per reception")

    # Defensive stats
    tackles: Optional[int] = Field(default=None, description="Total tackles (DEF)")
    sacks: Optional[float] = Field(default=None, description="Sacks (DEF)")
    interceptions_def: Optional[int] = Field(default=None, description="Interceptions (DEF)")
    passes_defended: Optional[int] = Field(default=None, description="Passes defended (DEF)")
    forced_fumbles: Optional[int] = Field(default=None, description="Forced fumbles (DEF)")

    # Advanced metrics
    pff_grade: Optional[float] = Field(default=None, ge=0, le=100, description="PFF grade (0-100)")

    class Config:
        json_schema_extra = {
            "example": {
                "games_played": 12,
                "games_started": 12,
                "passing_yards": 3500,
                "passing_tds": 32,
                "interceptions": 5,
                "completion_pct": 68.5,
                "qbr": 85.2
            }
        }


class CombineMeasurables(BaseModel):
    """Combine/pro day measurables."""
    height: Optional[float] = Field(default=None, description="Height in inches")
    weight: Optional[float] = Field(default=None, description="Weight in pounds")
    forty_yard: Optional[float] = Field(default=None, description="40-yard dash time")
    vertical: Optional[float] = Field(default=None, description="Vertical jump in inches")
    broad_jump: Optional[float] = Field(default=None, description="Broad jump in inches")
    bench_press: Optional[int] = Field(default=None, description="Bench press reps")
    three_cone: Optional[float] = Field(default=None, description="3-cone drill time")
    shuttle: Optional[float] = Field(default=None, description="20-yard shuttle time")

    class Config:
        json_schema_extra = {
            "example": {
                "height": 74,
                "weight": 215,
                "forty_yard": 4.42,
                "vertical": 38.5,
                "broad_jump": 128
            }
        }


class PlayerProfile(BaseModel):
    """Complete player profile for predictions."""
    name: str = Field(..., min_length=1, description="Player full name")
    school: str = Field(..., min_length=1, description="Current school name")
    position: str = Field(..., min_length=1, description="Position (QB, RB, WR, etc.)")

    # Basic info
    class_year: Optional[str] = Field(default="Junior", description="Class year (Freshman, Sophomore, Junior, Senior)")
    eligibility_remaining: Optional[int] = Field(default=2, ge=0, le=4, description="Years of eligibility remaining")
    hometown: Optional[str] = Field(default=None, description="Hometown")
    state: Optional[str] = Field(default=None, description="Home state")

    # Nested profiles
    stats: Optional[PlayerStats] = Field(default=None, description="Performance statistics")
    social_media: Optional[SocialMediaProfile] = Field(default=None, description="Social media metrics")
    recruiting: Optional[RecruitingProfile] = Field(default=None, description="Recruiting background")
    measurables: Optional[CombineMeasurables] = Field(default=None, description="Physical measurements")

    # Additional context
    overall_rating: Optional[float] = Field(default=None, ge=0, le=1, description="Overall player rating (0-1)")
    is_starter: Optional[bool] = Field(default=True, description="Whether player is a starter")
    awards: Optional[List[str]] = Field(default=None, description="List of awards/honors")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Arch Manning",
                "school": "Texas",
                "position": "QB",
                "class_year": "Sophomore",
                "eligibility_remaining": 3,
                "stats": {
                    "games_played": 12,
                    "passing_yards": 3200,
                    "passing_tds": 28,
                    "interceptions": 6
                },
                "social_media": {
                    "instagram_followers": 500000,
                    "twitter_followers": 200000
                },
                "recruiting": {
                    "stars": 5,
                    "national_rank": 1
                },
                "overall_rating": 0.95
            }
        }


# =============================================================================
# NIL Endpoint Schemas
# =============================================================================

class NILPredictRequest(BaseModel):
    """Request body for NIL prediction."""
    player: PlayerProfile = Field(..., description="Complete player profile")

    class Config:
        json_schema_extra = {
            "example": {
                "player": {
                    "name": "Travis Hunter",
                    "school": "Colorado",
                    "position": "CB",
                    "class_year": "Junior",
                    "stats": {"games_played": 12, "interceptions_def": 4},
                    "social_media": {"instagram_followers": 1200000},
                    "recruiting": {"stars": 5, "national_rank": 2}
                }
            }
        }


class NILValueBreakdown(BaseModel):
    """Breakdown of NIL value components."""
    base_value: float = Field(..., description="Base value from performance")
    social_media_premium: float = Field(..., description="Social media contribution")
    school_brand_factor: float = Field(..., description="School brand contribution")
    position_market_factor: float = Field(..., description="Position market adjustment")
    draft_potential_premium: float = Field(default=0, description="Draft prospect premium")

    class Config:
        json_schema_extra = {
            "example": {
                "base_value": 400000,
                "social_media_premium": 350000,
                "school_brand_factor": 150000,
                "position_market_factor": 100000,
                "draft_potential_premium": 200000
            }
        }


class NILPredictResponse(BaseModel):
    """Response for NIL prediction."""
    player_name: str = Field(..., description="Player name")
    school: str = Field(..., description="Current school")
    position: str = Field(..., description="Position")
    predicted_value: float = Field(..., description="Predicted NIL value in dollars")
    value_tier: str = Field(..., description="Value tier (mega, premium, solid, moderate, entry)")
    tier_probabilities: Dict[str, float] = Field(..., description="Probability for each tier")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    value_breakdown: NILValueBreakdown = Field(..., description="Value component breakdown")
    comparable_players: List[Dict[str, Any]] = Field(default=[], description="Similar players for reference")
    percentile: Optional[float] = Field(default=None, description="Percentile among all players")

    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "Travis Hunter",
                "school": "Colorado",
                "position": "CB",
                "predicted_value": 1200000,
                "value_tier": "mega",
                "tier_probabilities": {"mega": 0.75, "premium": 0.20, "solid": 0.05},
                "confidence": 0.85,
                "value_breakdown": {
                    "base_value": 400000,
                    "social_media_premium": 350000,
                    "school_brand_factor": 150000,
                    "position_market_factor": 100000,
                    "draft_potential_premium": 200000
                },
                "comparable_players": [
                    {"name": "Derek Stingley Jr", "nil_value": 1000000}
                ],
                "percentile": 99.5
            }
        }


class TransferImpactRequest(BaseModel):
    """Request for transfer impact analysis."""
    player: PlayerProfile = Field(..., description="Player profile")
    target_school: str = Field(..., min_length=1, description="Potential transfer destination")

    class Config:
        json_schema_extra = {
            "example": {
                "player": {
                    "name": "Jalen Milroe",
                    "school": "Alabama",
                    "position": "QB",
                    "stats": {"passing_yards": 2800, "passing_tds": 23}
                },
                "target_school": "USC"
            }
        }


class TransferImpactResponse(BaseModel):
    """Response for transfer impact analysis."""
    player_name: str
    current_school: str
    target_school: str
    current_value: float = Field(..., description="Current NIL value")
    projected_value: float = Field(..., description="Projected value at target school")
    value_change: float = Field(..., description="Change in value (can be negative)")
    value_change_pct: float = Field(..., description="Percentage change")
    factors: Dict[str, str] = Field(..., description="Factors affecting the change")
    recommendation: str = Field(..., description="Transfer recommendation")

    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "Jalen Milroe",
                "current_school": "Alabama",
                "target_school": "USC",
                "current_value": 800000,
                "projected_value": 950000,
                "value_change": 150000,
                "value_change_pct": 18.75,
                "factors": {
                    "market_size": "+$100K LA market premium",
                    "qb_room": "+$50K less competition"
                },
                "recommendation": "Transfer would increase NIL value"
            }
        }


class MarketReportRequest(BaseModel):
    """Request for NIL market report."""
    position: Optional[str] = Field(default=None, description="Filter by position group")
    conference: Optional[str] = Field(default=None, description="Filter by conference")
    school_tier: Optional[str] = Field(default=None, description="Filter by school tier")

    class Config:
        json_schema_extra = {
            "example": {
                "position": "QB",
                "conference": "SEC"
            }
        }


class MarketReportResponse(BaseModel):
    """Response for NIL market report."""
    filters_applied: Dict[str, str]
    total_players: int
    average_value: float
    median_value: float
    total_market_value: float
    value_by_tier: Dict[str, Dict[str, float]]
    top_players: List[Dict[str, Any]]
    position_breakdown: Optional[Dict[str, Dict[str, float]]] = None
    market_trends: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "filters_applied": {"position": "QB", "conference": "SEC"},
                "total_players": 42,
                "average_value": 425000,
                "median_value": 250000,
                "total_market_value": 17850000,
                "value_by_tier": {
                    "mega": {"count": 3, "avg_value": 1500000},
                    "premium": {"count": 8, "avg_value": 650000}
                },
                "top_players": [
                    {"name": "Player A", "school": "Georgia", "value": 2100000}
                ],
                "market_trends": ["QB values up 15% YoY"]
            }
        }


# =============================================================================
# Portal Endpoint Schemas
# =============================================================================

class FlightRiskRequest(BaseModel):
    """Request for flight risk prediction."""
    player: PlayerProfile = Field(..., description="Player profile with team context")
    team_context: Optional[Dict[str, Any]] = Field(default=None, description="Additional team context")

    class Config:
        json_schema_extra = {
            "example": {
                "player": {
                    "name": "Sample Player",
                    "school": "Florida State",
                    "position": "WR",
                    "class_year": "Junior",
                    "overall_rating": 0.82
                },
                "team_context": {
                    "recent_coaching_change": True,
                    "team_record": "6-6"
                }
            }
        }


class FlightRiskResponse(BaseModel):
    """Response for flight risk prediction."""
    player_name: str
    school: str
    flight_risk_probability: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., description="critical, high, moderate, low")
    risk_factors: List[Dict[str, Any]] = Field(..., description="Contributing factors")
    retention_recommendations: List[str]
    estimated_replacement_cost: float
    comparable_transfers: List[Dict[str, Any]] = Field(default=[])

    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "Sample Player",
                "school": "Florida State",
                "flight_risk_probability": 0.72,
                "risk_level": "high",
                "risk_factors": [
                    {"factor": "coaching_change", "impact": 0.25},
                    {"factor": "below_market_nil", "impact": 0.20}
                ],
                "retention_recommendations": [
                    "Increase NIL to market rate",
                    "Ensure playing time commitment"
                ],
                "estimated_replacement_cost": 350000,
                "comparable_transfers": []
            }
        }


class TeamReportRequest(BaseModel):
    """Request for team flight risk report."""
    school: str = Field(..., min_length=1, description="School name")
    include_walkons: bool = Field(default=False, description="Include walk-ons in analysis")

    class Config:
        json_schema_extra = {
            "example": {
                "school": "Michigan",
                "include_walkons": False
            }
        }


class TeamReportResponse(BaseModel):
    """Response for team flight risk report."""
    school: str
    analysis_date: datetime
    total_roster_size: int
    total_at_risk: int
    critical_risk_players: List[Dict[str, Any]]
    high_risk_players: List[Dict[str, Any]]
    estimated_wins_at_risk: float
    total_retention_budget_needed: float
    position_vulnerability: Dict[str, Dict[str, Any]]
    recommendations: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "school": "Michigan",
                "analysis_date": "2024-01-15T10:30:00Z",
                "total_roster_size": 85,
                "total_at_risk": 12,
                "critical_risk_players": [
                    {"name": "Player A", "position": "WR", "risk": 0.85}
                ],
                "high_risk_players": [],
                "estimated_wins_at_risk": 2.5,
                "total_retention_budget_needed": 1500000,
                "position_vulnerability": {
                    "WR": {"count": 3, "avg_risk": 0.65}
                },
                "recommendations": ["Prioritize WR retention"]
            }
        }


class PortalFitRequest(BaseModel):
    """Request for portal fit scoring."""
    player: PlayerProfile = Field(..., description="Portal player profile")
    target_school: str = Field(..., min_length=1, description="Target school")

    class Config:
        json_schema_extra = {
            "example": {
                "player": {
                    "name": "Transfer Player",
                    "school": "UCLA",
                    "position": "RB",
                    "stats": {"rushing_yards": 800}
                },
                "target_school": "Oregon"
            }
        }


class PortalFitResponse(BaseModel):
    """Response for portal fit scoring."""
    player_name: str
    origin_school: str
    target_school: str
    fit_score: float = Field(..., ge=0, le=1)
    fit_grade: str = Field(..., description="A/B/C/D/F grade")
    fit_breakdown: Dict[str, float]
    projected_nil_at_target: float
    projected_playing_time: str
    concerns: List[str]
    strengths: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "Transfer Player",
                "origin_school": "UCLA",
                "target_school": "Oregon",
                "fit_score": 0.82,
                "fit_grade": "B+",
                "fit_breakdown": {
                    "scheme_fit": 0.85,
                    "competition_level": 0.75,
                    "geographic_fit": 0.90
                },
                "projected_nil_at_target": 180000,
                "projected_playing_time": "Starter",
                "concerns": ["Crowded RB room"],
                "strengths": ["Zone scheme experience"]
            }
        }


class PortalRecommendationsRequest(BaseModel):
    """Request for portal recommendations."""
    school: str = Field(..., min_length=1, description="School seeking players")
    budget: float = Field(..., gt=0, description="Available NIL budget")
    positions_of_need: Optional[List[str]] = Field(default=None, description="Priority positions")
    max_targets: int = Field(default=20, ge=1, le=50, description="Maximum targets to return")

    class Config:
        json_schema_extra = {
            "example": {
                "school": "Tennessee",
                "budget": 2000000,
                "positions_of_need": ["QB", "EDGE", "CB"],
                "max_targets": 15
            }
        }


class PortalRecommendationsResponse(BaseModel):
    """Response for portal recommendations."""
    school: str
    budget: float
    targets: List[Dict[str, Any]]
    positions_prioritized: List[str]
    budget_allocation_suggestion: Dict[str, float]
    projected_roster_improvement: float
    acquisition_strategy: str

    class Config:
        json_schema_extra = {
            "example": {
                "school": "Tennessee",
                "budget": 2000000,
                "targets": [
                    {
                        "name": "Portal QB",
                        "position": "QB",
                        "origin_school": "Oregon",
                        "projected_nil": 500000,
                        "fit_score": 0.88,
                        "win_impact": 1.2
                    }
                ],
                "positions_prioritized": ["QB", "EDGE", "CB"],
                "budget_allocation_suggestion": {"QB": 600000, "EDGE": 400000},
                "projected_roster_improvement": 2.5,
                "acquisition_strategy": "Focus on elite QB, then depth"
            }
        }


# =============================================================================
# Draft Endpoint Schemas
# =============================================================================

class DraftProjectRequest(BaseModel):
    """Request for draft projection."""
    player: PlayerProfile = Field(..., description="Player profile with measurables")

    class Config:
        json_schema_extra = {
            "example": {
                "player": {
                    "name": "Caleb Williams",
                    "school": "USC",
                    "position": "QB",
                    "class_year": "Junior",
                    "stats": {"passing_yards": 4500, "passing_tds": 42},
                    "measurables": {"height": 73, "weight": 215}
                }
            }
        }


class DraftProjectResponse(BaseModel):
    """Response for draft projection."""
    player_name: str
    position: str
    draft_eligible: bool
    projected_round: Optional[int] = Field(default=None, ge=1, le=7)
    projected_pick_range: Optional[str] = Field(default=None, description="e.g., '1-5'")
    draft_probability: float = Field(..., ge=0, le=1)
    draft_grade: str
    expected_draft_value: float = Field(..., description="Jimmy Johnson chart value")
    rookie_contract_estimate: float
    career_earnings_estimate: float
    strengths: List[str]
    weaknesses: List[str]
    comparable_prospects: List[Dict[str, Any]]
    stock_trend: str = Field(..., description="rising, stable, falling")

    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "Caleb Williams",
                "position": "QB",
                "draft_eligible": True,
                "projected_round": 1,
                "projected_pick_range": "1-3",
                "draft_probability": 0.99,
                "draft_grade": "A+",
                "expected_draft_value": 3000,
                "rookie_contract_estimate": 38000000,
                "career_earnings_estimate": 250000000,
                "strengths": ["Elite arm talent", "Pocket mobility"],
                "weaknesses": ["Scheme diversity"],
                "comparable_prospects": [
                    {"name": "Patrick Mahomes", "draft_position": 10}
                ],
                "stock_trend": "stable"
            }
        }


class MockDraftRequest(BaseModel):
    """Request for mock draft generation."""
    season_year: int = Field(..., ge=2020, le=2030, description="Draft year")
    num_rounds: int = Field(default=3, ge=1, le=7, description="Number of rounds")
    include_trades: bool = Field(default=False, description="Include trade predictions")

    class Config:
        json_schema_extra = {
            "example": {
                "season_year": 2025,
                "num_rounds": 3,
                "include_trades": False
            }
        }


class MockDraftResponse(BaseModel):
    """Response for mock draft."""
    season_year: int
    num_rounds: int
    total_picks: int
    draft_board: List[Dict[str, Any]]
    position_distribution: Dict[str, int]
    top_prospects_by_position: Dict[str, List[Dict[str, Any]]]
    generated_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "season_year": 2025,
                "num_rounds": 3,
                "total_picks": 96,
                "draft_board": [
                    {"pick": 1, "player": "Player A", "position": "QB", "school": "Texas"}
                ],
                "position_distribution": {"QB": 5, "WR": 12, "CB": 8},
                "top_prospects_by_position": {},
                "generated_at": "2024-01-15T10:30:00Z"
            }
        }


# =============================================================================
# Roster Endpoint Schemas
# =============================================================================

class RosterOptimizeRequest(BaseModel):
    """Request for roster/NIL optimization."""
    school: str = Field(..., min_length=1, description="School name")
    total_budget: float = Field(..., gt=0, description="Total NIL budget")
    win_target: Optional[float] = Field(default=None, ge=0, le=15, description="Target wins")
    prioritize_retention: bool = Field(default=True, description="Prioritize keeping current players")

    class Config:
        json_schema_extra = {
            "example": {
                "school": "Georgia",
                "total_budget": 12000000,
                "win_target": 11,
                "prioritize_retention": True
            }
        }


class RosterOptimizeResponse(BaseModel):
    """Response for roster optimization."""
    school: str
    total_budget: float
    total_allocated: float
    budget_remaining: float
    expected_wins: float
    optimization_status: str
    allocations: List[Dict[str, Any]]
    position_breakdown: Dict[str, float]
    retention_priorities: List[str]
    efficiency_score: float

    class Config:
        json_schema_extra = {
            "example": {
                "school": "Georgia",
                "total_budget": 12000000,
                "total_allocated": 11500000,
                "budget_remaining": 500000,
                "expected_wins": 11.2,
                "optimization_status": "optimal",
                "allocations": [
                    {"player": "QB1", "recommended_nil": 2000000}
                ],
                "position_breakdown": {"QB": 2500000, "WR": 1800000},
                "retention_priorities": ["QB1", "EDGE1"],
                "efficiency_score": 0.92
            }
        }


class ScenarioPlayer(BaseModel):
    """Player add/remove for scenario analysis."""
    name: str = Field(..., description="Player name")
    position: str = Field(..., description="Position")
    action: str = Field(..., description="'add' or 'remove'")
    overall_rating: Optional[float] = Field(default=0.75, ge=0, le=1)
    nil_cost: Optional[float] = Field(default=0, ge=0, description="NIL cost if adding")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Portal QB",
                "position": "QB",
                "action": "add",
                "overall_rating": 0.88,
                "nil_cost": 500000
            }
        }


class ScenarioRequest(BaseModel):
    """Request for scenario analysis."""
    school: str = Field(..., min_length=1)
    changes: List[ScenarioPlayer] = Field(..., min_length=1, description="List of player changes")

    class Config:
        json_schema_extra = {
            "example": {
                "school": "Oklahoma",
                "changes": [
                    {"name": "New QB", "position": "QB", "action": "add", "overall_rating": 0.90},
                    {"name": "Old LB", "position": "LB", "action": "remove"}
                ]
            }
        }


class ScenarioResponse(BaseModel):
    """Response for scenario analysis."""
    school: str
    changes_analyzed: int
    current_projected_wins: float
    new_projected_wins: float
    win_delta: float
    total_nil_cost: float
    cost_per_win: Optional[float]
    position_impacts: Dict[str, float]
    recommendation: str
    risks: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "school": "Oklahoma",
                "changes_analyzed": 2,
                "current_projected_wins": 8.5,
                "new_projected_wins": 10.2,
                "win_delta": 1.7,
                "total_nil_cost": 500000,
                "cost_per_win": 294118,
                "position_impacts": {"QB": 2.0, "LB": -0.3},
                "recommendation": "Proceed - strong value",
                "risks": ["QB injury risk with no proven backup"]
            }
        }


class RosterReportResponse(BaseModel):
    """Response for comprehensive roster report."""
    school: str
    school_tier: str
    generated_at: datetime
    executive_summary: List[str]
    roster_summary: Dict[str, Any]
    nil_optimization: Dict[str, Any]
    portal_shopping: Dict[str, Any]
    flight_risk: Dict[str, Any]
    win_projection: Dict[str, Any]
    gap_analysis: Dict[str, Any]
    output_files: Dict[str, str]

    class Config:
        json_schema_extra = {
            "example": {
                "school": "Texas",
                "school_tier": "blue_blood",
                "generated_at": "2024-01-15T10:30:00Z",
                "executive_summary": [
                    "Projected 10.5 wins with current roster",
                    "3 critical retention targets"
                ],
                "roster_summary": {"total_players": 85},
                "nil_optimization": {},
                "portal_shopping": {},
                "flight_risk": {},
                "win_projection": {},
                "gap_analysis": {},
                "output_files": {
                    "json": "outputs/reports/texas_full_roster_report.json"
                }
            }
        }
