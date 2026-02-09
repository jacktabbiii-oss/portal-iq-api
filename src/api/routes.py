"""API routes for Portal IQ.

All endpoints for NIL valuation, portal intelligence, draft projections,
and roster optimization.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

# Import data loader for real R2 data (no local fallback)
from ..utils.data_loader import (
    get_nil_players,
    get_portal_players,
    get_database_stats,
    get_pff_player_stats,
    calculate_portal_iq_value,
    _get_nil_tier,
)
from ..utils.s3_storage import R2NotConfiguredError, R2DataLoadError

from .schemas import (
    # Base
    APIResponse,
    # NIL
    NILPredictRequest,
    NILPredictResponse,
    NILValueBreakdown,
    TransferImpactRequest,
    TransferImpactResponse,
    MarketReportRequest,
    MarketReportResponse,
    # Portal
    FlightRiskRequest,
    FlightRiskResponse,
    TeamReportRequest,
    TeamReportResponse,
    PortalFitRequest,
    PortalFitResponse,
    PortalRecommendationsRequest,
    PortalRecommendationsResponse,
    # Draft
    DraftProjectRequest,
    DraftProjectResponse,
    MockDraftRequest,
    MockDraftResponse,
    # Roster
    RosterOptimizeRequest,
    RosterOptimizeResponse,
    ScenarioRequest,
    ScenarioResponse,
    RosterReportResponse,
)

logger = logging.getLogger("portal_iq_api")

router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================

def get_models(request: Request) -> Dict[str, Any]:
    """Get loaded models from app state."""
    return request.app.state.get_models()


async def require_api_key(request: Request) -> str:
    """Require API key authentication."""
    api_key = request.headers.get("X-API-Key")
    from fastapi import HTTPException

    valid_keys = {"dev-key-123", "64c60545aa2809c7fc69d4bb6cc9743a8690df00e19c28ad32b022befa9c2ec1"}

    # Also try to get from app state
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        keys_env = os.getenv("PORTAL_IQ_API_KEYS", "dev-key-123")
        valid_keys.update(k.strip() for k in keys_env.split(",") if k.strip())
    except:
        pass

    if not api_key:
        raise HTTPException(status_code=401, detail={"error": "Missing API key", "message": "Include X-API-Key header"})
    if api_key not in valid_keys:
        raise HTTPException(status_code=401, detail={"error": "Invalid API key", "message": "The provided API key is not valid"})
    return api_key


def player_to_dataframe(player_data: Dict[str, Any]) -> pd.DataFrame:
    """Convert player profile dict to DataFrame for model input."""
    # Flatten nested structures
    flat_data = {
        "name": player_data.get("name"),
        "player_name": player_data.get("name"),
        "school": player_data.get("school"),
        "position": player_data.get("position"),
        "class_year": player_data.get("class_year", "Junior"),
        "eligibility_remaining": player_data.get("eligibility_remaining", 2),
        "overall_rating": player_data.get("overall_rating", 0.75),
        "is_starter": player_data.get("is_starter", True),
    }

    # Flatten stats
    if player_data.get("stats"):
        for key, value in player_data["stats"].items():
            flat_data[key] = value

    # Flatten social media
    if player_data.get("social_media"):
        for key, value in player_data["social_media"].items():
            flat_data[key] = value

    # Flatten recruiting
    if player_data.get("recruiting"):
        for key, value in player_data["recruiting"].items():
            flat_data[key] = value

    # Flatten measurables
    if player_data.get("measurables"):
        for key, value in player_data["measurables"].items():
            flat_data[key] = value

    return pd.DataFrame([flat_data])


# =============================================================================
# Valuation Helper Functions
# =============================================================================

def _get_position_base_value(position: str) -> int:
    """Get base NIL value by position."""
    position_values = {
        "QB": 250000,
        "WR": 120000,
        "RB": 100000,
        "TE": 80000,
        "OT": 70000,
        "OG": 60000,
        "C": 60000,
        "OL": 65000,
        "EDGE": 90000,
        "DT": 75000,
        "DE": 85000,
        "DL": 80000,
        "LB": 75000,
        "CB": 95000,
        "S": 70000,
        "K": 30000,
        "P": 25000,
        "LS": 20000,
    }
    return position_values.get(position.upper(), 50000)


def _get_school_multiplier(school: str) -> float:
    """Get school brand multiplier based on tier."""
    tier1 = ["Ohio State", "Alabama", "Georgia", "Michigan", "Texas", "USC", "LSU", "Oregon", "Clemson", "Notre Dame"]
    tier2 = ["Oklahoma", "Penn State", "Florida", "Tennessee", "Miami", "Auburn", "Texas A&M", "Ole Miss", "Wisconsin", "Utah"]
    tier3 = ["Colorado", "Florida State", "Washington", "UCLA", "Arizona", "Michigan State", "Iowa", "Nebraska", "Arkansas", "NC State"]

    school_lower = school.lower()
    for s in tier1:
        if s.lower() in school_lower:
            return 2.0
    for s in tier2:
        if s.lower() in school_lower:
            return 1.5
    for s in tier3:
        if s.lower() in school_lower:
            return 1.25
    return 1.0


def _generate_valuation_reasoning(
    position: str,
    school: str,
    pff_overall: float,
    stars: int,
    nil_value: float,
) -> List[str]:
    """Generate human-readable reasoning for NIL valuation."""
    reasons = []

    # Position reasoning
    high_value_positions = ["QB", "WR", "CB", "EDGE"]
    if position.upper() in high_value_positions:
        reasons.append(f"{position} is a premium NIL position with high market demand")
    else:
        reasons.append(f"{position} has moderate NIL market value")

    # School reasoning
    school_mult = _get_school_multiplier(school)
    if school_mult >= 2.0:
        reasons.append(f"{school} is a Tier 1 NIL market with elite brand value")
    elif school_mult >= 1.5:
        reasons.append(f"{school} is a strong NIL market with good brand visibility")
    elif school_mult >= 1.25:
        reasons.append(f"{school} is an emerging NIL market")
    else:
        reasons.append(f"{school} has developing NIL opportunities")

    # Performance reasoning
    if pff_overall >= 85:
        reasons.append(f"Elite PFF grade ({pff_overall:.1f}) significantly boosts valuation")
    elif pff_overall >= 75:
        reasons.append(f"Strong PFF grade ({pff_overall:.1f}) supports higher valuation")
    elif pff_overall >= 65:
        reasons.append(f"Solid PFF grade ({pff_overall:.1f}) at position average")
    elif pff_overall > 0:
        reasons.append(f"PFF grade ({pff_overall:.1f}) indicates room for development")

    # Recruiting reasoning
    if stars >= 5:
        reasons.append("5-star recruit with maximum recruiting premium")
    elif stars >= 4:
        reasons.append("4-star recruit with significant brand recognition")
    elif stars >= 3:
        reasons.append("3-star recruit with solid foundation")

    return reasons


# =============================================================================
# NIL Endpoints
# =============================================================================

@router.post(
    "/nil/predict",
    response_model=APIResponse,
    tags=["NIL"],
    summary="Predict NIL value",
    description="Get NIL valuation prediction for a player with detailed breakdown.",
)
async def predict_nil(
    request: Request,
    body: NILPredictRequest,
    api_key: str = Depends(require_api_key),
):
    """Predict NIL value for a player.

    This is the primary endpoint PlaymakerVC calls to show NIL data on client profiles.
    """
    models = get_models(request)
    nil_valuator = models.get("nil_valuator")

    player_dict = body.player.model_dump()

    if nil_valuator is not None:
        try:
            player_df = player_to_dataframe(player_dict)
            predictions = nil_valuator.predict(player_df)

            if predictions and len(predictions) > 0:
                pred = predictions[0]

                response_data = NILPredictResponse(
                    player_name=body.player.name,
                    school=body.player.school,
                    position=body.player.position,
                    predicted_value=pred.get("predicted_value", 0),
                    value_tier=pred.get("tier", "moderate"),
                    tier_probabilities=pred.get("tier_probabilities", {}),
                    confidence=pred.get("confidence", 0.7),
                    value_breakdown=NILValueBreakdown(
                        base_value=pred.get("value_breakdown", {}).get("base_value", 0),
                        social_media_premium=pred.get("value_breakdown", {}).get("social_premium", 0),
                        school_brand_factor=pred.get("value_breakdown", {}).get("school_factor", 0),
                        position_market_factor=pred.get("value_breakdown", {}).get("position_factor", 0),
                        draft_potential_premium=pred.get("value_breakdown", {}).get("draft_premium", 0),
                    ),
                    comparable_players=pred.get("comparable_players", []),
                    percentile=pred.get("percentile"),
                )

                return APIResponse(status="success", data=response_data.model_dump())

        except Exception as e:
            logger.warning(f"ML NIL prediction failed: {e}, falling back to CustomNILValuator")
            # Fall through to CustomNILValuator below

    # Use CustomNILValuator for formula-based valuation (primary method or fallback)
    try:
        from ..models.custom_nil_valuator import CustomNILValuator

        custom_valuator = CustomNILValuator()

        # Extract stats - handle None values
        stats = player_dict.get("stats") or {}
        social = player_dict.get("social_media") or {}
        recruiting = player_dict.get("recruiting") or {}

        val = custom_valuator.calculate_valuation(
            player_name=player_dict.get("name", "Unknown"),
            position=player_dict.get("position", "ATH"),
            school=player_dict.get("school", "Unknown"),
            conference=player_dict.get("conference"),
            games_played=stats.get("games_played", 0),
            games_started=stats.get("games_started", 0),
            passing_yards=stats.get("passing_yards", 0),
            passing_tds=stats.get("passing_tds", 0),
            rushing_yards=stats.get("rushing_yards", 0),
            rushing_tds=stats.get("rushing_tds", 0),
            receiving_yards=stats.get("receiving_yards", 0),
            receiving_tds=stats.get("receiving_tds", 0),
            tackles=stats.get("tackles", 0),
            sacks=stats.get("sacks", 0),
            interceptions=stats.get("interceptions", 0),
            instagram_followers=social.get("instagram_followers", 0),
            twitter_followers=social.get("twitter_followers", 0),
            tiktok_followers=social.get("tiktok_followers", 0),
            recruiting_stars=recruiting.get("stars", 0),
            national_rank=recruiting.get("national_rank"),
            is_starter=player_dict.get("is_starter", True),
        )

        response_data = NILPredictResponse(
            player_name=body.player.name,
            school=body.player.school,
            position=body.player.position,
            predicted_value=val.total_valuation,
            value_tier=val.valuation_tier,
            tier_probabilities={val.valuation_tier: 0.8},
            confidence=0.75 if val.confidence == "high" else 0.6 if val.confidence == "medium" else 0.4,
            value_breakdown=NILValueBreakdown(
                base_value=val.factors.get("position_base", 0),
                social_media_premium=val.social_value,
                school_brand_factor=val.market_value - val.performance_value,
                position_market_factor=val.performance_value,
                draft_potential_premium=val.potential_value,
            ),
            comparable_players=[],
            percentile=50.0,
        )

        return APIResponse(
            status="success",
            data=response_data.model_dump(),
            message="Valuation from CustomNILValuator (performance-based)",
        )

    except Exception as e:
        import traceback
        logger.error(f"CustomNILValuator error: {e}\n{traceback.format_exc()}")

        # Final fallback to simple demo calculation
        demo_value = _calculate_demo_nil_value(player_dict)

        response_data = NILPredictResponse(
            player_name=body.player.name,
            school=body.player.school,
            position=body.player.position,
            predicted_value=demo_value,
            value_tier=_get_nil_tier(demo_value),
            tier_probabilities={"moderate": 0.6, "solid": 0.3, "entry": 0.1},
            confidence=0.65,
            value_breakdown=NILValueBreakdown(
                base_value=demo_value * 0.4,
                social_media_premium=demo_value * 0.2,
                school_brand_factor=demo_value * 0.2,
                position_market_factor=demo_value * 0.15,
                draft_potential_premium=demo_value * 0.05,
            ),
            comparable_players=[],
            percentile=50.0,
        )

        return APIResponse(
            status="success",
            data=response_data.model_dump(),
            message="Demo mode - fallback calculation",
        )


@router.post(
    "/nil/transfer-impact",
    response_model=APIResponse,
    tags=["NIL"],
    summary="Analyze transfer impact on NIL",
    description="Compare current NIL value vs projected value at a target school.",
)
async def transfer_impact(
    request: Request,
    body: TransferImpactRequest,
    api_key: str = Depends(require_api_key),
):
    """Analyze how transferring would impact a player's NIL value."""
    models = get_models(request)
    nil_valuator = models.get("nil_valuator")

    player_dict = body.player.model_dump()

    if nil_valuator is not None:
        try:
            result = nil_valuator.transfer_impact(
                player_to_dataframe(player_dict),
                body.target_school,
            )

            if result:
                response_data = TransferImpactResponse(
                    player_name=body.player.name,
                    current_school=body.player.school,
                    target_school=body.target_school,
                    current_value=result.get("current_value", 0),
                    projected_value=result.get("projected_value", 0),
                    value_change=result.get("value_change", 0),
                    value_change_pct=result.get("value_change_pct", 0),
                    factors=result.get("factors", {}),
                    recommendation=result.get("recommendation", ""),
                )
                return APIResponse(status="success", data=response_data.model_dump())

        except Exception as e:
            logger.error(f"Transfer impact error: {e}")

    # Demo fallback
    current = _calculate_demo_nil_value(player_dict)
    target_mult = _get_school_multiplier(body.target_school)
    current_mult = _get_school_multiplier(body.player.school)
    projected = current * (target_mult / current_mult) if current_mult else current

    response_data = TransferImpactResponse(
        player_name=body.player.name,
        current_school=body.player.school,
        target_school=body.target_school,
        current_value=current,
        projected_value=projected,
        value_change=projected - current,
        value_change_pct=((projected - current) / current * 100) if current else 0,
        factors={
            "market_size": "Market size adjustment",
            "program_brand": "Program brand factor",
        },
        recommendation="Transfer analysis based on market factors",
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode",
    )


@router.get(
    "/nil/leaderboard",
    response_model=APIResponse,
    tags=["NIL"],
    summary="NIL Leaderboard",
    description="Get top players ranked by NIL valuation from real data.",
)
async def nil_leaderboard(
    request: Request,
    limit: int = Query(100, ge=1, le=25000, description="Max results to return (up to 25000)"),
    offset: int = Query(0, ge=0, description="Skip first N results for pagination"),
    position: Optional[str] = Query(None, description="Filter by position (QB, WR, etc.)"),
    school: Optional[str] = Query(None, description="Filter by school name"),
    conference: Optional[str] = Query(None, description="Filter by conference"),
    search: Optional[str] = Query(None, description="Search player names"),
    api_key: str = Depends(require_api_key),
):
    """Get NIL leaderboard with real data from Cloudflare R2.

    Supports pagination via offset/limit and search across all players.
    """
    try:
        # Get ALL matching players first (no limit) to get total count
        df = get_nil_players(
            position=position,
            school=school,
            conference=conference,
            limit=50000,  # Get all data
        )

        # Apply search filter if provided
        if search and not df.empty and "name" in df.columns:
            df = df[df["name"].str.contains(search, case=False, na=False)]

        total_count = len(df)

        # Apply pagination
        df = df.iloc[offset:offset + limit]

        if df.empty:
            return APIResponse(
                status="success",
                data={"players": [], "total": 0},
                message="No players found matching criteria"
            )

        # Build response list with Portal IQ calculated valuations
        players = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            on3_value = float(row.get("nil_value", 0)) if pd.notna(row.get("nil_value")) else 0
            player_name = str(row.get("name", "Unknown"))
            pos = str(row.get("position", "")).upper()
            school_name = str(row.get("school", ""))
            pff_overall = float(row.get("pff_overall", 0)) if pd.notna(row.get("pff_overall")) else 0
            stars = int(row.get("stars", 0)) if pd.notna(row.get("stars")) else 0

            # Use the proper valuation model
            val = calculate_portal_iq_value(pos, school_name, pff_overall, stars)

            player_data = {
                "rank": rank,
                "player_id": str(row.get("player_id", player_name.replace(" ", "_").lower())),
                "player_name": player_name,
                "position": pos,
                "school": school_name,
                "conference": str(row.get("conference", "")) if pd.notna(row.get("conference")) else None,
                "valuation": val["value"],
                "on3_value": on3_value if on3_value > 0 else None,
                "nil_tier": val["tier"],
                "headshot_url": str(row.get("headshot_url")) if pd.notna(row.get("headshot_url")) else None,
                "valuation_source": "Portal IQ",
                "stars": stars if stars > 0 else None,
                "height": float(row.get("height", 0)) if pd.notna(row.get("height")) else None,
                "weight": float(row.get("weight", 0)) if pd.notna(row.get("weight")) else None,
                "pff_overall": pff_overall if pff_overall > 0 else None,
                "pff_offense": float(row.get("pff_offense", 0)) if pd.notna(row.get("pff_offense")) else None,
                "pff_defense": float(row.get("pff_defense", 0)) if pd.notna(row.get("pff_defense")) else None,
            }
            players.append(player_data)

        return APIResponse(
            status="success",
            data={
                "players": players,
                "total": len(players),
                "total_count": total_count,  # Total matching players (for pagination)
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(players) < total_count,
                "filters_applied": {
                    "position": position,
                    "school": school,
                    "conference": conference,
                    "search": search,
                }
            }
        )

    except R2NotConfiguredError as e:
        logger.error(f"R2 not configured: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Storage not configured",
                "message": "R2 storage is required but not configured. Contact support.",
            }
        )
    except R2DataLoadError as e:
        logger.error(f"R2 data load failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Data unavailable",
                "message": "Could not load NIL data from storage. Try again later.",
            }
        )
    except Exception as e:
        logger.error(f"NIL leaderboard error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to load NIL data: {str(e)}")


@router.get(
    "/players/search",
    response_model=APIResponse,
    tags=["Players"],
    summary="Search all players",
    description="Search across NIL and portal data for players by name, school, or position.",
)
async def search_players(
    request: Request,
    query: str = Query(..., min_length=2, description="Search query (name, school, or position)"),
    data_type: str = Query("all", pattern="^(nil|portal|all)$", description="Search in nil, portal, or all"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
    api_key: str = Depends(require_api_key),
):
    """Search players across all available data.

    This endpoint searches the full dataset (21,000+ players) and returns
    matching results quickly for autocomplete and search functionality.
    """
    try:
        results = []
        query_lower = query.lower()

        # Search NIL data
        if data_type in ("nil", "all"):
            nil_df = get_nil_players(limit=50000)
            if not nil_df.empty:
                # Search name, school, position
                mask = (
                    nil_df["name"].str.lower().str.contains(query_lower, na=False) |
                    nil_df.get("school", pd.Series()).str.lower().str.contains(query_lower, na=False) |
                    nil_df.get("position", pd.Series()).str.lower().str.contains(query_lower, na=False)
                )
                matches = nil_df[mask].head(limit if data_type == "nil" else limit // 2)
                for _, row in matches.iterrows():
                    pos = str(row.get("position", "")).upper()
                    school_name = str(row.get("school", ""))
                    pff_overall = float(row.get("pff_overall", 0)) if pd.notna(row.get("pff_overall")) else 0
                    stars = int(row.get("stars", 0)) if pd.notna(row.get("stars")) else 0
                    on3_value = float(row.get("nil_value", 0)) if pd.notna(row.get("nil_value")) else 0

                    val = calculate_portal_iq_value(pos, school_name, pff_overall, stars)

                    results.append({
                        "player_id": str(row.get("player_id", row.get("name", "").replace(" ", "_").lower())),
                        "player_name": str(row.get("name", "")),
                        "position": pos,
                        "school": school_name,
                        "valuation": val["value"],
                        "on3_value": on3_value if on3_value > 0 else None,
                        "nil_tier": val["tier"],
                        "headshot_url": str(row.get("headshot_url")) if pd.notna(row.get("headshot_url")) else None,
                        "source": "nil",
                    })

        # Search Portal data
        if data_type in ("portal", "all"):
            portal_df = get_portal_players(limit=50000)
            if not portal_df.empty:
                mask = (
                    portal_df["name"].str.lower().str.contains(query_lower, na=False) |
                    portal_df.get("origin_school", pd.Series()).str.lower().str.contains(query_lower, na=False) |
                    portal_df.get("position", pd.Series()).str.lower().str.contains(query_lower, na=False)
                )
                matches = portal_df[mask].head(limit if data_type == "portal" else limit // 2)
                for _, row in matches.iterrows():
                    player_name = str(row.get("name", ""))
                    # Avoid duplicates from NIL search
                    if not any(r["player_name"] == player_name for r in results):
                        pos = str(row.get("position", "")).upper()
                        school_name = str(row.get("origin_school", ""))
                        stars = int(row.get("stars", 0)) if pd.notna(row.get("stars")) else 0

                        val = calculate_portal_iq_value(pos, school_name, 0, stars)

                        results.append({
                            "player_id": str(row.get("player_id", player_name.replace(" ", "_").lower())),
                            "player_name": player_name,
                            "position": pos,
                            "school": school_name,
                            "valuation": val["value"],
                            "nil_tier": val["tier"],
                            "destination_school": str(row.get("destination_school")) if pd.notna(row.get("destination_school")) else None,
                            "status": str(row.get("status", "available")),
                            "stars": stars if stars > 0 else None,
                            "headshot_url": str(row.get("headshot_url")) if pd.notna(row.get("headshot_url")) else None,
                            "source": "portal",
                        })

        return APIResponse(
            status="success",
            data={
                "players": results[:limit],
                "total": len(results[:limit]),
                "query": query,
                "data_type": data_type,
            }
        )

    except R2NotConfiguredError as e:
        logger.error(f"R2 not configured: {e}")
        raise HTTPException(status_code=503, detail={"error": "Storage not configured"})
    except R2DataLoadError as e:
        logger.error(f"R2 data load failed: {e}")
        raise HTTPException(status_code=503, detail={"error": "Data unavailable"})
    except Exception as e:
        logger.error(f"Player search error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# NOTE: The comprehensive /players/{player_name}/stats endpoint is defined below
# (around line 1897) to avoid duplicate route definitions. That version returns
# the full PlayerStats format expected by the frontend with PFF grades,
# position-specific stats, and valuation data.


@router.post(
    "/nil/market-report",
    response_model=APIResponse,
    tags=["NIL"],
    summary="NIL market report",
    description="Get market overview with average values, top players, and trends.",
)
async def market_report(
    request: Request,
    body: MarketReportRequest,
    api_key: str = Depends(require_api_key),
):
    """Generate NIL market report with optional position/conference filters."""
    from pathlib import Path

    # Try to load real data from CSV first
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    valuations_file = data_dir / "nil_valuations_2025.csv"
    if not valuations_file.exists():
        valuations_file = data_dir / "nil_valuations_2024.csv"

    if valuations_file.exists():
        try:
            df = pd.read_csv(valuations_file)

            # Apply filters
            filters = {}
            if body.position:
                df = df[df['position'].str.upper() == body.position.upper()]
                filters["position"] = body.position
            if body.conference:
                df = df[df['conference'].str.lower() == body.conference.lower()]
                filters["conference"] = body.conference

            # Calculate stats
            total_players = len(df)
            average_value = df['custom_nil_value'].mean() if total_players > 0 else 0
            median_value = df['custom_nil_value'].median() if total_players > 0 else 0
            total_market_value = df['custom_nil_value'].sum()

            # Value by tier
            value_by_tier = {}
            if 'nil_tier' in df.columns:
                for tier in df['nil_tier'].unique():
                    tier_df = df[df['nil_tier'] == tier]
                    value_by_tier[tier] = {
                        "count": len(tier_df),
                        "avg_value": tier_df['custom_nil_value'].mean()
                    }

            # Top players (map to frontend expected fields)
            top_df = df.nlargest(25, 'custom_nil_value')
            top_players = []
            for _, row in top_df.iterrows():
                player = {
                    "name": row.get('player_name', 'Unknown'),
                    "school": row.get('school', 'Unknown'),
                    "position": row.get('position', 'Unknown'),
                    "value": row.get('custom_nil_value', 0),
                }
                # Add optional fields if they exist
                if 'espn_headshot_url' in row and pd.notna(row['espn_headshot_url']):
                    player["headshot_url"] = row['espn_headshot_url']
                if 'nil_tier' in row:
                    player["tier"] = row['nil_tier']
                top_players.append(player)

            response_data = MarketReportResponse(
                filters_applied=filters,
                total_players=total_players,
                average_value=average_value,
                median_value=median_value,
                total_market_value=total_market_value,
                value_by_tier=value_by_tier,
                top_players=top_players,
                market_trends=[
                    f"Total market value: ${total_market_value:,.0f}",
                    f"Average player value: ${average_value:,.0f}",
                    f"Top valued positions: QB, WR, EDGE",
                ],
            )

            return APIResponse(
                status="success",
                data=response_data.model_dump(),
                message="Real data from valuations",
            )

        except Exception as e:
            logger.error(f"Market report CSV error: {e}")

    # Demo fallback if no CSV available
    models = get_models(request)
    nil_valuator = models.get("nil_valuator")

    if nil_valuator is not None:
        try:
            report = nil_valuator.generate_position_market_report(
                position=body.position,
                conference=body.conference,
            )
            if report:
                return APIResponse(status="success", data=report)
        except Exception as e:
            logger.error(f"Market report error: {e}")

    # Final demo fallback
    filters = {}
    if body.position:
        filters["position"] = body.position
    if body.conference:
        filters["conference"] = body.conference

    response_data = MarketReportResponse(
        filters_applied=filters,
        total_players=500,
        average_value=185000,
        median_value=75000,
        total_market_value=92500000,
        value_by_tier={
            "mega": {"count": 15, "avg_value": 1500000},
            "premium": {"count": 50, "avg_value": 600000},
            "solid": {"count": 100, "avg_value": 175000},
            "moderate": {"count": 150, "avg_value": 50000},
            "entry": {"count": 185, "avg_value": 15000},
        },
        top_players=[
            {"name": "Top QB", "school": "Georgia", "position": "QB", "value": 2500000},
            {"name": "Star WR", "school": "Ohio State", "position": "WR", "value": 1800000},
        ],
        market_trends=[
            "QB values increased 18% year-over-year",
            "Social media following driving premium valuations",
            "Blue blood schools command 2-3x market premium",
        ],
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode - run generate_nil_valuations.py for real data",
    )


# =============================================================================
# Portal Endpoints
# =============================================================================

@router.get(
    "/portal/active",
    response_model=APIResponse,
    tags=["Portal"],
    summary="Active portal players",
    description="Get current transfer portal players with real data from On3.",
)
async def portal_active(
    request: Request,
    position: Optional[str] = Query(None, description="Filter by position"),
    origin_school: Optional[str] = Query(None, description="Filter by origin school"),
    origin_conference: Optional[str] = Query(None, description="Filter by origin conference"),
    min_stars: Optional[int] = Query(None, ge=1, le=5, description="Minimum star rating"),
    status: Optional[str] = Query("all", pattern="^(available|committed|all)$"),
    limit: int = Query(200, ge=1, le=25000, description="Max results (up to 25000)"),
    offset: int = Query(0, ge=0, description="Skip first N results for pagination"),
    search: Optional[str] = Query(None, description="Search player names"),
    api_key: str = Depends(require_api_key),
):
    """Get active transfer portal players with real data from Cloudflare R2.

    Supports pagination via offset/limit and search across all portal players.
    """
    try:
        # Map status to filter
        status_filter = None
        if status == "available":
            status_filter = "Entered"
        elif status == "committed":
            status_filter = "Committed"

        # Use data_loader which pulls from S3/R2 - get ALL matching data
        df = get_portal_players(
            status=status_filter,
            position=position,
            origin_school=origin_school,
            origin_conference=origin_conference,
            min_stars=min_stars,
            limit=50000,  # Get all data
        )

        # Apply search filter if provided
        if search and not df.empty and "name" in df.columns:
            df = df[df["name"].str.contains(search, case=False, na=False)]

        total_count = len(df)

        # Compute aggregate stats BEFORE pagination (across all matching data)
        active_count = 0
        committed_count = 0
        schools_set = set()

        if not df.empty and "status" in df.columns:
            status_lower = df["status"].str.lower()
            active_count = int((~status_lower.isin(["committed", "withdrawn"])).sum())
            committed_count = int((status_lower == "committed").sum())

        if not df.empty and "origin_school" in df.columns:
            schools_set = set(df["origin_school"].dropna().unique())
        if not df.empty and "destination_school" in df.columns:
            dest_schools = set(df["destination_school"].dropna().unique())
            schools_set = schools_set | dest_schools

        # Apply pagination
        df = df.iloc[offset:offset + limit]

        if df.empty:
            return APIResponse(
                status="success",
                data={
                    "players": [],
                    "total": 0,
                    "total_count": total_count,
                    "active_in_portal": active_count,
                    "committed": committed_count,
                    "schools_active": len(schools_set),
                },
                message="No portal players found matching criteria"
            )

        # Build response list with Portal IQ valuations
        players = []
        for _, row in df.iterrows():
            player_name = str(row.get("name", "Unknown"))
            raw_status = str(row.get("status", "Entered")).lower()

            # Map status string
            if raw_status == "committed":
                player_status = "committed"
            elif raw_status == "withdrawn":
                player_status = "withdrawn"
            else:
                player_status = "available"

            # Calculate Portal IQ valuation for this portal player
            pos = str(row.get("position", "")).upper()
            school_name = str(row.get("origin_school", ""))
            stars = int(row.get("stars", 0)) if pd.notna(row.get("stars")) else 0
            pff_overall = float(row.get("pff_overall", 0)) if pd.notna(row.get("pff_overall")) else 0
            on3_value = float(row.get("nil_value", 0)) if pd.notna(row.get("nil_value")) else 0

            val = calculate_portal_iq_value(pos, school_name, pff_overall, stars)

            player_data = {
                "player_id": str(row.get("player_id", player_name.replace(" ", "_").lower())),
                "player_name": player_name,
                "position": pos,
                "origin_school": school_name,
                "origin_conference": str(row.get("conference")) if pd.notna(row.get("conference")) else None,
                "destination_school": str(row.get("destination_school")) if pd.notna(row.get("destination_school")) else None,
                "stars": stars if stars > 0 else None,
                "status": player_status,
                "nil_valuation": val["value"],  # Portal IQ calculated value
                "on3_value": on3_value if on3_value > 0 else None,
                "nil_tier": val["tier"],
                "headshot_url": str(row.get("headshot_url")) if pd.notna(row.get("headshot_url")) else None,
                "height": float(row.get("height", 0)) if pd.notna(row.get("height")) else None,
                "weight": float(row.get("weight", 0)) if pd.notna(row.get("weight")) else None,
                "pff_overall": pff_overall if pff_overall > 0 else None,
            }
            players.append(player_data)

        return APIResponse(
            status="success",
            data={
                "players": players,
                "total": len(players),
                "total_count": total_count,
                "active_in_portal": active_count,
                "committed": committed_count,
                "schools_active": len(schools_set),
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(players) < total_count,
                "filters_applied": {
                    "position": position,
                    "origin_school": origin_school,
                    "status": status,
                    "search": search,
                }
            }
        )

    except R2NotConfiguredError as e:
        logger.error(f"R2 not configured: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Storage not configured",
                "message": "R2 storage is required but not configured. Contact support.",
            }
        )
    except R2DataLoadError as e:
        logger.error(f"R2 data load failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Data unavailable",
                "message": "Could not load portal data from storage. Try again later.",
            }
        )
    except Exception as e:
        logger.error(f"Portal active error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to load portal data: {str(e)}")


@router.post(
    "/portal/flight-risk",
    response_model=APIResponse,
    tags=["Portal"],
    summary="Predict flight risk",
    description="Predict probability a player enters the transfer portal.",
)
async def flight_risk(
    request: Request,
    body: FlightRiskRequest,
    api_key: str = Depends(require_api_key),
):
    """Predict flight risk for a player."""
    models = get_models(request)
    portal_predictor = models.get("portal_predictor")

    player_dict = body.player.model_dump()
    if body.team_context:
        player_dict.update(body.team_context)

    if portal_predictor is not None:
        try:
            predictions = portal_predictor.predict_flight_risk(
                player_to_dataframe(player_dict)
            )

            if predictions and len(predictions) > 0:
                pred = predictions[0]

                response_data = FlightRiskResponse(
                    player_name=body.player.name,
                    school=body.player.school,
                    flight_risk_probability=pred.get("flight_risk_probability", 0.3),
                    risk_level=pred.get("risk_level", "moderate"),
                    risk_factors=pred.get("risk_factors", []),
                    retention_recommendations=pred.get("retention_recommendations", []),
                    estimated_replacement_cost=pred.get("replacement_cost", 100000),
                    comparable_transfers=pred.get("comparable_transfers", []),
                )
                return APIResponse(status="success", data=response_data.model_dump())

        except Exception as e:
            logger.error(f"Flight risk error: {e}")

    # Demo fallback
    risk_prob = 0.35
    if body.team_context and body.team_context.get("recent_coaching_change"):
        risk_prob += 0.2

    response_data = FlightRiskResponse(
        player_name=body.player.name,
        school=body.player.school,
        flight_risk_probability=risk_prob,
        risk_level="moderate" if risk_prob < 0.5 else "high",
        risk_factors=[
            {"factor": "playing_time", "impact": 0.15},
            {"factor": "nil_market", "impact": 0.10},
        ],
        retention_recommendations=[
            "Ensure competitive NIL compensation",
            "Discuss role in upcoming season",
        ],
        estimated_replacement_cost=150000,
        comparable_transfers=[],
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode",
    )


@router.post(
    "/portal/team-report",
    response_model=APIResponse,
    tags=["Portal"],
    summary="Team flight risk report",
    description="Get comprehensive flight risk analysis for entire roster.",
)
async def team_report(
    request: Request,
    body: TeamReportRequest,
    api_key: str = Depends(require_api_key),
):
    """Generate team-wide flight risk report."""
    models = get_models(request)
    portal_predictor = models.get("portal_predictor")

    if portal_predictor is not None:
        try:
            # Would need roster data - using placeholder
            report = portal_predictor.team_flight_risk_report(
                pd.DataFrame(),  # Would load from database
                body.school,
            )
            if report:
                return APIResponse(status="success", data=report)
        except Exception as e:
            logger.error(f"Team report error: {e}")

    # Demo fallback
    response_data = TeamReportResponse(
        school=body.school,
        analysis_date=datetime.utcnow(),
        total_roster_size=85,
        total_at_risk=8,
        critical_risk_players=[
            {"name": "WR1", "position": "WR", "risk": 0.78, "nil_value": 200000},
        ],
        high_risk_players=[
            {"name": "CB2", "position": "CB", "risk": 0.62, "nil_value": 150000},
            {"name": "RB1", "position": "RB", "risk": 0.58, "nil_value": 120000},
        ],
        estimated_wins_at_risk=1.8,
        total_retention_budget_needed=850000,
        position_vulnerability={
            "WR": {"count": 2, "avg_risk": 0.65},
            "CB": {"count": 2, "avg_risk": 0.55},
        },
        recommendations=[
            "Prioritize WR retention - highest flight risk position group",
            "Address CB depth concerns before portal window",
        ],
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode - provide roster data for full analysis",
    )


@router.post(
    "/portal/fit-score",
    response_model=APIResponse,
    tags=["Portal"],
    summary="Calculate portal fit",
    description="Score how well a portal player fits a target school.",
)
async def portal_fit(
    request: Request,
    body: PortalFitRequest,
    api_key: str = Depends(require_api_key),
):
    """Calculate fit score for portal player at target school."""
    models = get_models(request)
    portal_predictor = models.get("portal_predictor")

    player_dict = body.player.model_dump()

    if portal_predictor is not None:
        try:
            predictions = portal_predictor.predict_portal_fit(
                player_to_dataframe(player_dict),
                body.target_school,
            )

            if predictions and len(predictions) > 0:
                pred = predictions[0]

                response_data = PortalFitResponse(
                    player_name=body.player.name,
                    origin_school=body.player.school,
                    target_school=body.target_school,
                    fit_score=pred.get("fit_score", 0.7),
                    fit_grade=pred.get("fit_grade", "B"),
                    fit_breakdown=pred.get("fit_breakdown", {}),
                    projected_nil_at_target=pred.get("projected_nil", 100000),
                    projected_playing_time=pred.get("playing_time", "Starter"),
                    concerns=pred.get("concerns", []),
                    strengths=pred.get("strengths", []),
                )
                return APIResponse(status="success", data=response_data.model_dump())

        except Exception as e:
            logger.error(f"Portal fit error: {e}")

    # Demo fallback
    fit_score = 0.75
    response_data = PortalFitResponse(
        player_name=body.player.name,
        origin_school=body.player.school,
        target_school=body.target_school,
        fit_score=fit_score,
        fit_grade="B+" if fit_score >= 0.75 else "B",
        fit_breakdown={
            "scheme_fit": 0.80,
            "competition_level": 0.70,
            "geographic_fit": 0.75,
        },
        projected_nil_at_target=_calculate_demo_nil_value(player_dict) * _get_school_multiplier(body.target_school),
        projected_playing_time="Starter",
        concerns=["Adjustment to new system"],
        strengths=["Experience level", "Production history"],
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode",
    )


@router.post(
    "/portal/recommendations",
    response_model=APIResponse,
    tags=["Portal"],
    summary="Portal recommendations",
    description="Get ranked portal targets for a school based on needs and budget.",
)
async def portal_recommendations(
    request: Request,
    body: PortalRecommendationsRequest,
    api_key: str = Depends(require_api_key),
):
    """Get ranked portal transfer recommendations."""
    models = get_models(request)
    roster_optimizer = models.get("roster_optimizer")

    if roster_optimizer is not None:
        try:
            result = roster_optimizer.portal_shopping_list(
                school=body.school,
                roster_df=pd.DataFrame(),  # Would load from database
                budget_remaining=body.budget,
                positions_of_need=body.positions_of_need,
            )

            if result:
                response_data = PortalRecommendationsResponse(
                    school=body.school,
                    budget=body.budget,
                    targets=result.get("shopping_list", [])[:body.max_targets],
                    positions_prioritized=result.get("positions_prioritized", []),
                    budget_allocation_suggestion=result.get("position_needs", {}),
                    projected_roster_improvement=result.get("projected_improvement", 0),
                    acquisition_strategy=result.get("budget_strategy", ""),
                )
                return APIResponse(status="success", data=response_data.model_dump())

        except Exception as e:
            logger.error(f"Portal recommendations error: {e}")

    # Demo fallback
    positions = body.positions_of_need or ["QB", "EDGE", "CB"]

    demo_targets = []
    for i, pos in enumerate(positions[:5]):
        demo_targets.append({
            "name": f"Portal Target {i+1}",
            "position": pos,
            "origin_school": "Previous School",
            "projected_nil": body.budget * (0.3 - i * 0.05),
            "fit_score": 0.85 - i * 0.05,
            "win_impact": 0.8 - i * 0.1,
            "value_rating": 0.9 - i * 0.1,
        })

    response_data = PortalRecommendationsResponse(
        school=body.school,
        budget=body.budget,
        targets=demo_targets,
        positions_prioritized=positions,
        budget_allocation_suggestion={pos: body.budget / len(positions) for pos in positions},
        projected_roster_improvement=1.5,
        acquisition_strategy="Focus on elite talent at positions of need",
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode",
    )


# =============================================================================
# Draft Endpoints
# =============================================================================

@router.post(
    "/draft/project",
    response_model=APIResponse,
    tags=["Draft"],
    summary="Project draft position",
    description="Get NFL draft projection with contract and earnings estimates.",
)
async def draft_project(
    request: Request,
    body: DraftProjectRequest,
    api_key: str = Depends(require_api_key),
):
    """Project draft position and NFL value for a player."""
    models = get_models(request)
    draft_projector = models.get("draft_projector")

    player_dict = body.player.model_dump()

    if draft_projector is not None:
        try:
            predictions = draft_projector.predict(player_to_dataframe(player_dict))

            if predictions and len(predictions) > 0:
                pred = predictions[0]

                response_data = DraftProjectResponse(
                    player_name=body.player.name,
                    position=body.player.position,
                    draft_eligible=pred.get("draft_eligible", True),
                    projected_round=pred.get("projected_round"),
                    projected_pick_range=pred.get("projected_pick_range"),
                    draft_probability=pred.get("draft_probability", 0.5),
                    draft_grade=pred.get("draft_grade", "B"),
                    expected_draft_value=pred.get("expected_draft_value", 500),
                    rookie_contract_estimate=pred.get("rookie_contract", 5000000),
                    career_earnings_estimate=pred.get("career_earnings", 25000000),
                    strengths=pred.get("strengths", []),
                    weaknesses=pred.get("weaknesses", []),
                    comparable_prospects=pred.get("comparables", []),
                    stock_trend=pred.get("stock_trend", "stable"),
                )
                return APIResponse(status="success", data=response_data.model_dump())

        except Exception as e:
            logger.error(f"Draft projection error: {e}")

    # Demo fallback
    draft_prob = 0.4
    if player_dict.get("overall_rating", 0.75) >= 0.90:
        draft_prob = 0.85
    elif player_dict.get("overall_rating", 0.75) >= 0.85:
        draft_prob = 0.6

    response_data = DraftProjectResponse(
        player_name=body.player.name,
        position=body.player.position,
        draft_eligible=body.player.class_year in ["Junior", "Senior"],
        projected_round=3 if draft_prob > 0.5 else None,
        projected_pick_range="65-100" if draft_prob > 0.5 else None,
        draft_probability=draft_prob,
        draft_grade="B" if draft_prob > 0.5 else "C",
        expected_draft_value=500 if draft_prob > 0.5 else 100,
        rookie_contract_estimate=5000000 if draft_prob > 0.5 else 0,
        career_earnings_estimate=25000000 if draft_prob > 0.5 else 0,
        strengths=["Production", "Experience"],
        weaknesses=["Athleticism testing needed"],
        comparable_prospects=[],
        stock_trend="stable",
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode",
    )


@router.post(
    "/draft/mock",
    response_model=APIResponse,
    tags=["Draft"],
    summary="Generate mock draft",
    description="Generate full mock draft board for specified year and rounds.",
)
async def mock_draft(
    request: Request,
    body: MockDraftRequest,
    api_key: str = Depends(require_api_key),
):
    """Generate mock draft board."""
    models = get_models(request)
    draft_projector = models.get("draft_projector")

    if draft_projector is not None:
        try:
            result = draft_projector.generate_mock_draft(
                season_year=body.season_year,
                num_rounds=body.num_rounds,
            )

            if result:
                response_data = MockDraftResponse(
                    season_year=body.season_year,
                    num_rounds=body.num_rounds,
                    total_picks=result.get("total_picks", 0),
                    draft_board=result.get("draft_board", []),
                    position_distribution=result.get("position_distribution", {}),
                    top_prospects_by_position=result.get("top_by_position", {}),
                    generated_at=datetime.utcnow(),
                )
                return APIResponse(status="success", data=response_data.model_dump())

        except Exception as e:
            logger.error(f"Mock draft error: {e}")

    # Demo fallback
    picks_per_round = 32
    total_picks = body.num_rounds * picks_per_round

    demo_board = []
    positions = ["QB", "WR", "CB", "EDGE", "OT", "DT", "LB", "S", "RB", "TE"]
    for i in range(min(total_picks, 50)):  # Limit demo to 50 picks
        demo_board.append({
            "pick": i + 1,
            "round": (i // picks_per_round) + 1,
            "player": f"Prospect {i + 1}",
            "position": positions[i % len(positions)],
            "school": "University",
            "grade": "A" if i < 10 else "B" if i < 32 else "C",
        })

    response_data = MockDraftResponse(
        season_year=body.season_year,
        num_rounds=body.num_rounds,
        total_picks=total_picks,
        draft_board=demo_board,
        position_distribution={"QB": 5, "WR": 8, "CB": 6, "EDGE": 5},
        top_prospects_by_position={},
        generated_at=datetime.utcnow(),
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode",
    )


# =============================================================================
# Roster Endpoints
# =============================================================================

@router.post(
    "/roster/optimize",
    response_model=APIResponse,
    tags=["Roster"],
    summary="Optimize NIL budget",
    description="Get optimal NIL budget allocation across roster.",
)
async def roster_optimize(
    request: Request,
    body: RosterOptimizeRequest,
    api_key: str = Depends(require_api_key),
):
    """Optimize NIL budget allocation."""
    models = get_models(request)
    roster_optimizer = models.get("roster_optimizer")

    if roster_optimizer is not None:
        try:
            result = roster_optimizer.optimize_nil_budget(
                school=body.school,
                total_budget=body.total_budget,
                roster_df=pd.DataFrame(),  # Would load from database
                win_target=body.win_target,
            )

            if result:
                response_data = RosterOptimizeResponse(
                    school=body.school,
                    total_budget=body.total_budget,
                    total_allocated=result.get("total_allocated", 0),
                    budget_remaining=result.get("budget_remaining", 0),
                    expected_wins=result.get("expected_wins", 0),
                    optimization_status=result.get("optimization_status", "unknown"),
                    allocations=result.get("allocations", []),
                    position_breakdown=result.get("position_breakdown", {}),
                    retention_priorities=result.get("retention_priority", []),
                    efficiency_score=result.get("budget_efficiency", 0),
                )
                return APIResponse(status="success", data=response_data.model_dump())

        except Exception as e:
            logger.error(f"Roster optimization error: {e}")

    # Demo fallback
    allocated = body.total_budget * 0.92

    demo_allocations = [
        {"player": "QB1", "position": "QB", "recommended_nil": body.total_budget * 0.18},
        {"player": "WR1", "position": "WR", "recommended_nil": body.total_budget * 0.10},
        {"player": "EDGE1", "position": "EDGE", "recommended_nil": body.total_budget * 0.08},
    ]

    response_data = RosterOptimizeResponse(
        school=body.school,
        total_budget=body.total_budget,
        total_allocated=allocated,
        budget_remaining=body.total_budget - allocated,
        expected_wins=body.win_target or 9.0,
        optimization_status="demo",
        allocations=demo_allocations,
        position_breakdown={
            "QB": body.total_budget * 0.20,
            "WR": body.total_budget * 0.15,
            "EDGE": body.total_budget * 0.12,
        },
        retention_priorities=["QB1", "WR1"],
        efficiency_score=0.85,
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode - provide roster data for full optimization",
    )


@router.post(
    "/roster/scenario",
    response_model=APIResponse,
    tags=["Roster"],
    summary="Scenario analysis",
    description="Analyze win impact of roster changes (adds/removes).",
)
async def roster_scenario(
    request: Request,
    body: ScenarioRequest,
    api_key: str = Depends(require_api_key),
):
    """Analyze win impact of roster changes."""
    models = get_models(request)
    win_model = models.get("win_model")

    if win_model is not None:
        try:
            # Convert changes to format expected by model
            additions = []
            removals = []

            for change in body.changes:
                player_data = {
                    "name": change.name,
                    "position": change.position,
                    "overall_rating": change.overall_rating,
                    "nil_cost": change.nil_cost,
                }
                if change.action == "add":
                    additions.append(player_data)
                else:
                    removals.append(player_data)

            result = win_model.scenario_analysis(
                pd.DataFrame(),  # Current roster
                body.school,
                additions=additions,
                removals=removals,
            )

            if result:
                response_data = ScenarioResponse(
                    school=body.school,
                    changes_analyzed=len(body.changes),
                    current_projected_wins=result.get("current_wins", 8),
                    new_projected_wins=result.get("new_wins", 8),
                    win_delta=result.get("win_delta", 0),
                    total_nil_cost=result.get("total_cost", 0),
                    cost_per_win=result.get("cost_per_win"),
                    position_impacts=result.get("position_impacts", {}),
                    recommendation=result.get("recommendation", ""),
                    risks=result.get("risks", []),
                )
                return APIResponse(status="success", data=response_data.model_dump())

        except Exception as e:
            logger.error(f"Scenario analysis error: {e}")

    # Demo fallback
    win_delta = 0
    total_cost = 0
    position_impacts = {}

    for change in body.changes:
        impact = (change.overall_rating - 0.75) * 2
        if change.action == "remove":
            impact = -impact
        win_delta += impact
        total_cost += change.nil_cost or 0
        position_impacts[change.position] = impact

    response_data = ScenarioResponse(
        school=body.school,
        changes_analyzed=len(body.changes),
        current_projected_wins=8.5,
        new_projected_wins=8.5 + win_delta,
        win_delta=win_delta,
        total_nil_cost=total_cost,
        cost_per_win=total_cost / win_delta if win_delta > 0 else None,
        position_impacts=position_impacts,
        recommendation="Proceed" if win_delta > 0 else "Reconsider",
        risks=["Depth concerns"] if any(c.action == "remove" for c in body.changes) else [],
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode",
    )


@router.get(
    "/roster/{school}/report",
    response_model=APIResponse,
    tags=["Roster"],
    summary="Full roster report",
    description="Get comprehensive roster analysis report for a school.",
)
async def roster_report(
    request: Request,
    school: str,
    api_key: str = Depends(require_api_key),
):
    """Generate comprehensive roster report."""
    models = get_models(request)
    roster_optimizer = models.get("roster_optimizer")

    if roster_optimizer is not None:
        try:
            result = roster_optimizer.full_roster_report(school=school)

            if result:
                response_data = RosterReportResponse(
                    school=result.get("school", school),
                    school_tier=result.get("school_tier", "p4_mid"),
                    generated_at=datetime.utcnow(),
                    executive_summary=result.get("executive_summary", []),
                    roster_summary=result.get("sections", {}).get("roster_summary", {}),
                    nil_optimization=result.get("sections", {}).get("nil_optimization", {}),
                    portal_shopping=result.get("sections", {}).get("portal_shopping", {}),
                    flight_risk=result.get("sections", {}).get("flight_risk", {}),
                    win_projection=result.get("sections", {}).get("win_projection", {}),
                    gap_analysis=result.get("sections", {}).get("gap_analysis", {}),
                    output_files=result.get("output_files", {}),
                )
                return APIResponse(status="success", data=response_data.model_dump())

        except Exception as e:
            logger.error(f"Roster report error: {e}")

    # Demo fallback
    school_tier = _get_school_tier(school)

    response_data = RosterReportResponse(
        school=school,
        school_tier=school_tier,
        generated_at=datetime.utcnow(),
        executive_summary=[
            f"Analysis for {school} ({school_tier} tier)",
            "Provide roster data for complete analysis",
        ],
        roster_summary={"message": "Requires roster data"},
        nil_optimization={"message": "Requires roster data"},
        portal_shopping={"message": "Requires roster data"},
        flight_risk={"message": "Requires roster data"},
        win_projection={"message": "Requires roster data"},
        gap_analysis={"message": "Requires roster data"},
        output_files={},
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
        message="Demo mode - provide roster data for full report",
    )


# =============================================================================
# Demo Helper Functions
# =============================================================================

def _calculate_demo_nil_value(player_dict: Dict[str, Any]) -> float:
    """Calculate demo NIL value based on player attributes."""
    base_value = 50000

    # Position multiplier
    position_mult = {
        "QB": 3.0, "WR": 1.5, "RB": 1.2, "TE": 1.0,
        "OT": 0.9, "OG": 0.7, "C": 0.7,
        "EDGE": 1.3, "DT": 0.9, "LB": 0.8,
        "CB": 1.2, "S": 0.9,
    }.get(player_dict.get("position", "").upper(), 1.0)

    # Rating multiplier
    rating = player_dict.get("overall_rating")
    if rating is None:
        rating = 0.75
    rating_mult = 1.0 + (rating - 0.75) * 5

    # Social media bonus
    social = player_dict.get("social_media") or {}
    followers = (
        (social.get("instagram_followers") or 0) +
        (social.get("twitter_followers") or 0) +
        (social.get("tiktok_followers") or 0)
    )
    social_bonus = min(followers / 10, 500000)  # Cap at 500k bonus

    # School multiplier
    school = player_dict.get("school", "")
    school_mult = _get_school_multiplier(school)

    return (base_value * position_mult * rating_mult * school_mult) + social_bonus


def _get_school_multiplier(school: str) -> float:
    """Get NIL multiplier based on school brand."""
    blue_bloods = ["Alabama", "Ohio State", "Georgia", "Texas", "USC", "Michigan", "Notre Dame", "Oklahoma"]
    elite = ["LSU", "Florida", "Penn State", "Oregon", "Clemson", "Tennessee", "Texas A&M"]
    power_brand = ["Miami", "Florida State", "Auburn", "Wisconsin", "Iowa", "UCLA"]

    school_clean = school.strip()
    if school_clean in blue_bloods:
        return 2.5
    elif school_clean in elite:
        return 1.8
    elif school_clean in power_brand:
        return 1.4
    else:
        return 1.0


def _get_school_tier(school: str) -> str:
    """Get school tier classification."""
    blue_bloods = ["Alabama", "Ohio State", "Georgia", "Texas", "USC", "Michigan", "Notre Dame", "Oklahoma"]
    elite = ["LSU", "Florida", "Penn State", "Oregon", "Clemson", "Tennessee", "Texas A&M"]
    power_brand = ["Miami", "Florida State", "Auburn", "Wisconsin", "Iowa", "UCLA"]

    school_clean = school.strip()
    if school_clean in blue_bloods:
        return "blue_blood"
    elif school_clean in elite:
        return "elite"
    elif school_clean in power_brand:
        return "power_brand"
    else:
        return "p4_mid"


def _get_nil_tier(value: float) -> str:
    """Get NIL tier from value."""
    if value >= 1000000:
        return "mega"
    elif value >= 500000:
        return "premium"
    elif value >= 100000:
        return "solid"
    elif value >= 25000:
        return "moderate"
    else:
        return "entry"


# =============================================================================
# AI Search Endpoints
# =============================================================================

@router.post(
    "/search",
    response_model=APIResponse,
    tags=["AI Search"],
    summary="AI-powered natural language search",
    description="Search the player database using natural language queries powered by Claude.",
)
async def ai_search(
    request: Request,
    body: dict,
    api_key: str = Depends(require_api_key),
):
    """
    Search the player database using natural language.

    Example queries:
    - "Show me 4-star QBs in the portal"
    - "Top NIL prospects from SEC schools"
    - "Undervalued players outperforming their recruiting ranking"
    - "Portal WRs with 1000+ receiving yards"
    """
    from .ai_search import get_ai_search

    query = body.get("query", "")
    max_results = body.get("max_results", 25)

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    ai = get_ai_search()

    if not ai.is_available():
        return APIResponse(
            status="error",
            message="AI search not available - ANTHROPIC_API_KEY not configured",
            data={"results": []}
        )

    result = await ai.search(query, max_results)

    if "error" in result:
        return APIResponse(
            status="error",
            message=result["error"],
            data=result
        )

    return APIResponse(
        status="success",
        data=result,
        message=f"Found {result.get('count', 0)} results"
    )


@router.get(
    "/search/status",
    response_model=APIResponse,
    tags=["AI Search"],
    summary="Check AI search availability",
)
async def ai_search_status(
    request: Request,
    api_key: str = Depends(require_api_key),
):
    """Check if AI search is available and what data is loaded."""
    from .ai_search import get_ai_search

    ai = get_ai_search()

    datasets = {}
    for name, df in ai.data.items():
        datasets[name] = {
            "records": len(df),
            "columns": list(df.columns)[:10]
        }

    return APIResponse(
        status="success",
        data={
            "available": ai.is_available(),
            "anthropic_configured": ai.client is not None,
            "datasets_loaded": len(ai.data),
            "datasets": datasets
        }
    )


# =============================================================================
# Player Search Endpoints
# =============================================================================

@router.get(
    "/players/search",
    response_model=APIResponse,
    tags=["Players"],
    summary="Search players",
    description="Search for players by name across NIL and portal data.",
)
async def search_players(
    request: Request,
    query: str = Query(..., min_length=2, description="Search query"),
    data_type: str = Query("all", regex="^(nil|portal|all)$", description="Data source to search"),
    limit: int = Query(25, ge=1, le=100),
    api_key: str = Depends(require_api_key),
):
    """Search players by name across NIL and portal data."""
    try:
        results = []
        query_lower = query.lower()

        # Search NIL data
        if data_type in ("nil", "all"):
            nil_df = get_nil_players(limit=500)
            if not nil_df.empty:
                # Filter by name match
                mask = nil_df["name"].str.lower().str.contains(query_lower, na=False)
                matching = nil_df[mask].head(limit if data_type == "nil" else limit // 2)

                for _, row in matching.iterrows():
                    results.append({
                        "name": str(row.get("name", "Unknown")),
                        "position": str(row.get("position", "")),
                        "school": str(row.get("school", "")),
                        "nil_value": float(row.get("nil_value", 0)) if pd.notna(row.get("nil_value")) else None,
                        "stars": int(row.get("stars", 0)) if pd.notna(row.get("stars")) else None,
                        "headshot_url": str(row.get("headshot_url")) if pd.notna(row.get("headshot_url")) else None,
                        "pff_overall": float(row.get("pff_overall", 0)) if pd.notna(row.get("pff_overall")) else None,
                        "status": None,
                        "destination_school": None,
                        "data_source": "nil",
                    })

        # Search Portal data
        if data_type in ("portal", "all"):
            portal_df = get_portal_players(limit=500)
            if not portal_df.empty:
                mask = portal_df["name"].str.lower().str.contains(query_lower, na=False)
                matching = portal_df[mask].head(limit if data_type == "portal" else limit // 2)

                for _, row in matching.iterrows():
                    # Check if already in results (from NIL)
                    player_name = str(row.get("name", "Unknown"))
                    if any(r["name"] == player_name for r in results):
                        continue

                    raw_status = str(row.get("status", "Entered")).lower()
                    if raw_status == "committed":
                        player_status = "committed"
                    elif raw_status == "withdrawn":
                        player_status = "withdrawn"
                    else:
                        player_status = "available"

                    results.append({
                        "name": player_name,
                        "position": str(row.get("position", "")),
                        "school": str(row.get("origin_school", "")),
                        "nil_value": float(row.get("nil_value", 0)) if pd.notna(row.get("nil_value")) else None,
                        "stars": int(row.get("stars", 0)) if pd.notna(row.get("stars")) else None,
                        "headshot_url": str(row.get("headshot_url")) if pd.notna(row.get("headshot_url")) else None,
                        "pff_overall": float(row.get("pff_overall", 0)) if pd.notna(row.get("pff_overall")) else None,
                        "status": player_status,
                        "destination_school": str(row.get("destination_school")) if pd.notna(row.get("destination_school")) else None,
                        "data_source": "portal",
                    })

        # Sort by NIL value descending
        results.sort(key=lambda x: x.get("nil_value") or 0, reverse=True)

        return APIResponse(
            status="success",
            data={
                "players": results[:limit],
                "total": len(results),
                "query": query,
            }
        )

    except R2NotConfiguredError as e:
        logger.error(f"R2 not configured: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Storage not configured",
                "message": "R2 storage is required but not configured. Contact support.",
            }
        )
    except R2DataLoadError as e:
        logger.error(f"R2 data load failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Data unavailable",
                "message": "Could not load player data from storage. Try again later.",
            }
        )
    except Exception as e:
        logger.error(f"Player search error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/players/{player_name}/stats",
    response_model=APIResponse,
    tags=["Players"],
    summary="Get player stats",
    description="Get comprehensive stats for a specific player.",
)
async def get_player_stats(
    request: Request,
    player_name: str,
    season: int = Query(2025, description="Season year"),
    api_key: str = Depends(require_api_key),
):
    """Get detailed stats for a player by name."""
    try:
        from urllib.parse import unquote
        player_name = unquote(player_name)
        player_name_lower = player_name.lower()

        # Search in NIL data first - use full dataset to find any player
        nil_df = get_nil_players(limit=50000)
        player_row = None

        if not nil_df.empty:
            mask = nil_df["name"].str.lower() == player_name_lower
            matches = nil_df[mask]
            if not matches.empty:
                player_row = matches.iloc[0]

        # If not found in NIL, check portal
        if player_row is None:
            portal_df = get_portal_players(limit=50000)
            if not portal_df.empty:
                mask = portal_df["name"].str.lower() == player_name_lower
                matches = portal_df[mask]
                if not matches.empty:
                    player_row = matches.iloc[0]

        if player_row is None:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

        # Extract base data from player row
        position = str(player_row.get("position", "")).upper()
        school = str(player_row.get("school", player_row.get("origin_school", "")))
        pff_overall = float(player_row.get("pff_overall", 0)) if pd.notna(player_row.get("pff_overall")) else 0
        stars = int(player_row.get("stars", 0)) if pd.notna(player_row.get("stars")) else 0

        # Get On3's value as market reference
        on3_raw_value = float(player_row.get("nil_value", 0)) if pd.notna(player_row.get("nil_value")) else 0
        on3_explicit = float(player_row.get("on3_nil_value", 0)) if pd.notna(player_row.get("on3_nil_value")) else None
        on3_market_value = on3_explicit if on3_explicit else on3_raw_value

        # Load detailed PFF stats (box score data: yards, TDs, receptions, etc.)
        pff_detailed = get_pff_player_stats(player_name, season)

        # Use PFF detailed data to enrich grades if available
        if pff_detailed:
            pff_overall = pff_detailed.get("pff_overall") or pff_overall

        # Calculate Portal IQ's valuation using the proper model
        valuation_result = calculate_portal_iq_value(
            position=position,
            school=school,
            pff_overall=pff_overall,
            stars=stars,
        )
        calculated_nil_value = valuation_result["value"]

        # Helper to safely get float from either pff_detailed or player_row
        def _get_stat(key, fallback_key=None):
            """Get stat from PFF detailed data first, then player_row."""
            if pff_detailed:
                val = pff_detailed.get(key)
                if val is not None:
                    return float(val)
            # Fallback to player_row
            fk = fallback_key or key
            val = player_row.get(fk)
            if pd.notna(val) if hasattr(val, '__class__') else val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
            return None

        # Build comprehensive stats response
        stats = {
            "name": str(player_row.get("name", player_name)),
            "position": position,
            "school": school,
            "headshot_url": str(player_row.get("headshot_url")) if pd.notna(player_row.get("headshot_url")) else None,
            "season": season,
            "nil_value": calculated_nil_value,
            "nil_tier": valuation_result["tier"],
            "stars": stars if stars > 0 else None,
            "height": float(player_row.get("height", 0)) if pd.notna(player_row.get("height")) else None,
            "weight": float(player_row.get("weight", 0)) if pd.notna(player_row.get("weight")) else None,
            "pff": {
                "overall": _get_stat("pff_overall"),
                "offense": _get_stat("pff_offense"),
                "defense": _get_stat("pff_defense"),
                "passing": _get_stat("pff_passing"),
                "rushing": _get_stat("pff_rushing"),
                "receiving": _get_stat("pff_receiving"),
                "pass_block": _get_stat("pff_pass_block"),
                "run_block": _get_stat("pff_run_block"),
                "pass_rush": _get_stat("pff_pass_rush"),
                "run_defense": _get_stat("pff_run_defense"),
                "tackling": _get_stat("pff_tackling"),
                "coverage": _get_stat("pff_coverage"),
            },
        }

        # Add position-specific box score stats from PFF detailed data
        if position == "QB":
            stats["passing"] = {
                "passer_rating": _get_stat("passer_rating"),
                "completion_pct": _get_stat("completion_pct") or _get_stat("adjusted_completion_pct"),
                "big_time_throws": _get_stat("big_time_throws"),
                "big_time_throw_pct": _get_stat("big_time_throw_pct"),
                "turnover_worthy_plays": _get_stat("turnover_worthy_plays"),
                "yards": _get_stat("yards", "passing_yards"),
                "touchdowns": _get_stat("touchdowns", "passing_tds"),
                "avg_depth_of_target": _get_stat("avg_depth_of_target"),
                "time_to_throw": _get_stat("time_to_throw") or _get_stat("avg_time_to_throw"),
            }

        if position in ("RB", "QB"):
            stats["rushing"] = {
                "elusive_rating": _get_stat("elusive_rating"),
                "yards_after_contact": _get_stat("yards_after_contact"),
                "breakaway_pct": _get_stat("breakaway_pct"),
                "yards": _get_stat("yards", "rushing_yards") if position == "RB" else _get_stat("rushing_yards"),
                "touchdowns": _get_stat("touchdowns", "rushing_tds") if position == "RB" else _get_stat("rushing_tds"),
                "yards_per_carry": _get_stat("yards_per_carry") or _get_stat("ypa"),
                "attempts": _get_stat("attempts"),
                "missed_tackles_forced": _get_stat("missed_tackles_forced"),
                "breakaway_yards": _get_stat("breakaway_yards"),
            }

        if position in ("WR", "TE", "RB"):
            stats["receiving"] = {
                "receptions": _get_stat("receptions"),
                "targets": _get_stat("targets"),
                "yards": _get_stat("rec_yards") or _get_stat("yards", "receiving_yards"),
                "touchdowns": _get_stat("touchdowns", "receiving_tds") if position != "RB" else _get_stat("receiving_tds"),
                "yards_per_route_run": _get_stat("yards_per_route_run"),
                "catch_rate": _get_stat("catch_rate"),
                "drop_rate": _get_stat("drop_rate"),
                "drops": _get_stat("drops"),
                "contested_catch_rate": _get_stat("contested_catch_rate"),
                "yards_after_catch": _get_stat("yards_after_catch"),
                "routes_run": _get_stat("routes_run"),
                "longest": _get_stat("longest"),
            }

        if position in ("EDGE", "DT", "DL", "DE", "LB"):
            stats["pass_rush"] = {
                "pass_rushing_productivity": _get_stat("pass_rushing_productivity"),
                "pass_rush_win_rate": _get_stat("pass_rush_win_rate"),
                "pressures": _get_stat("pressures"),
                "hurries": _get_stat("hurries"),
                "hits": _get_stat("hits"),
                "sacks": _get_stat("sacks"),
                "batted_passes": _get_stat("batted_passes"),
            }

        if position in ("CB", "S", "LB"):
            stats["coverage"] = {
                "passer_rating_allowed": _get_stat("passer_rating_allowed"),
                "yards_per_coverage_snap": _get_stat("yards_per_coverage_snap"),
                "interceptions": _get_stat("interceptions") or _get_stat("ints"),
                "pass_breakups": _get_stat("pass_breakups") or _get_stat("pbus"),
                "targets_allowed": _get_stat("targets_allowed"),
                "completions_allowed": _get_stat("completions_allowed"),
                "yards_allowed": _get_stat("yards_allowed"),
                "forced_incompletes": _get_stat("forced_incompletes"),
            }

        if position in ("OT", "OG", "C", "OL"):
            stats["blocking"] = {
                "pass_blocking_efficiency": _get_stat("pass_blocking_efficiency"),
                "pressures_allowed": _get_stat("pressures_allowed"),
                "sacks_allowed": _get_stat("sacks_allowed"),
                "hits_allowed": _get_stat("hits_allowed"),
                "hurries_allowed": _get_stat("hurries_allowed"),
                "run_block_percent": _get_stat("run_block_percent"),
            }

        # Add tackling stats for all defensive players
        if position in ("CB", "S", "LB", "EDGE", "DT", "DL", "DE"):
            stats["tackling"] = {
                "tackles": _get_stat("tackles"),
                "assists": _get_stat("assists") or _get_stat("tackle_assists"),
                "tackles_for_loss": _get_stat("tackles_for_loss"),
                "missed_tackles": _get_stat("missed_tackles"),
                "missed_tackle_pct": _get_stat("missed_tackle_pct"),
                "forced_fumbles": _get_stat("forced_fumbles"),
            }

        # Add snap counts if available
        off_snaps = _get_stat("offensive_snaps")
        def_snaps = _get_stat("defensive_snaps")
        if off_snaps or def_snaps:
            stats["snaps"] = {
                "offensive": off_snaps,
                "defensive": def_snaps,
                "pass_rush": _get_stat("pass_rush_snaps"),
                "coverage": _get_stat("coverage_snaps"),
            }

        # Add games played if available
        games = _get_stat("games_played")
        if games:
            stats["games_played"] = int(games)

        # Build valuation with the new calculator
        has_on3_data = on3_market_value > 0
        stats["valuation"] = {
            "on3_value": on3_market_value if has_on3_data else None,
            "portal_iq_value": calculated_nil_value,
            "portal_iq_tier": valuation_result["tier"],
            "confidence": "high" if has_on3_data else ("medium" if pff_overall > 0 else "low"),
            "has_on3_data": has_on3_data,
            "breakdown": valuation_result["breakdown"],
            "reasoning": valuation_result["reasoning"],
        }

        # Add market comparison if On3 data available
        if has_on3_data:
            diff = calculated_nil_value - on3_market_value
            diff_pct = (diff / on3_market_value * 100) if on3_market_value > 0 else 0
            stats["valuation"]["market_comparison"] = {
                "on3_reference": on3_market_value,
                "portal_iq_value": calculated_nil_value,
                "difference": diff,
                "difference_pct": round(diff_pct, 1),
                "assessment": "undervalued" if diff > 0 else ("overvalued" if diff < 0 else "fair value"),
            }

        return APIResponse(status="success", data=stats)

    except HTTPException:
        raise
    except R2NotConfiguredError as e:
        logger.error(f"R2 not configured: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Storage not configured",
                "message": "R2 storage is required but not configured. Contact support.",
            }
        )
    except R2DataLoadError as e:
        logger.error(f"R2 data load failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Data unavailable",
                "message": "Could not load player stats from storage. Try again later.",
            }
        )
    except Exception as e:
        logger.error(f"Player stats error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get player stats: {str(e)}")


# =============================================================================
# Player Comparison Endpoints
# =============================================================================

@router.get(
    "/players/{player_name}/comparisons",
    response_model=APIResponse,
    tags=["Players"],
    summary="Get player comparisons",
    description="Find similar NFL and college players based on stats and measurables.",
)
async def get_player_comparisons_endpoint(
    request: Request,
    player_name: str,
    include_nfl: bool = Query(True, description="Include NFL player comparisons"),
    include_college: bool = Query(True, description="Include college player comparisons"),
    limit: int = Query(5, ge=1, le=20, description="Max comparisons per category"),
    api_key: str = Depends(require_api_key),
):
    """Get comparable players for draft/portal analysis."""
    try:
        from urllib.parse import unquote
        player_name = unquote(player_name)

        # Get player position for position-based comparisons
        nil_df = get_nil_players(limit=50000)
        position = "Unknown"
        if not nil_df.empty:
            match = nil_df[nil_df["name"].str.lower() == player_name.lower()]
            if not match.empty:
                position = str(match.iloc[0].get("position", "Unknown"))

        # Return placeholder comparisons (full implementation requires similarity engine)
        return APIResponse(
            status="success",
            data={
                "player": player_name,
                "position": position,
                "nfl_comparisons": [],
                "college_comparisons": [],
                "message": "Player comparison engine in development",
            }
        )

    except Exception as e:
        logger.error(f"Player comparison error: {e}")
        return APIResponse(
            status="success",
            data={
                "player": player_name,
                "position": "Unknown",
                "nfl_comparisons": [],
                "college_comparisons": [],
                "message": "Comparison data not available",
            }
        )


@router.get(
    "/players/{player_name}/career",
    response_model=APIResponse,
    tags=["Players"],
    summary="Get player career stats",
    description="Get multi-season career trajectory for a player.",
)
async def get_player_career_endpoint(
    request: Request,
    player_name: str,
    api_key: str = Depends(require_api_key),
):
    """Get career stats across multiple seasons."""
    try:
        from urllib.parse import unquote
        player_name = unquote(player_name)

        # Return placeholder career data
        return APIResponse(
            status="success",
            data={
                "player_name": player_name,
                "college_seasons": [],
                "nfl_seasons": [],
                "combine_data": None,
                "total_seasons": 0,
                "message": "Career tracking in development",
            }
        )

    except Exception as e:
        logger.error(f"Player career error: {e}")
        return APIResponse(
            status="success",
            data={
                "player_name": player_name,
                "college_seasons": [],
                "nfl_seasons": [],
                "combine_data": None,
                "total_seasons": 0,
            }
        )


@router.get(
    "/players/{player_name}/elite-profile",
    response_model=APIResponse,
    tags=["Players"],
    summary="Get elite athlete profile",
    description="Get elite measurables and trait analysis for a player.",
)
async def get_player_elite_profile(
    request: Request,
    player_name: str,
    api_key: str = Depends(require_api_key),
):
    """Get elite athlete profile with measurables."""
    try:
        from urllib.parse import unquote
        player_name = unquote(player_name)

        # Get player data
        nil_df = get_nil_players(limit=50000)
        position = "Unknown"
        height = None
        weight = None

        if not nil_df.empty:
            match = nil_df[nil_df["name"].str.lower() == player_name.lower()]
            if not match.empty:
                row = match.iloc[0]
                position = str(row.get("position", "Unknown"))
                height = float(row.get("height", 0)) if pd.notna(row.get("height")) else None
                weight = float(row.get("weight", 0)) if pd.notna(row.get("weight")) else None

        return APIResponse(
            status="success",
            data={
                "player": player_name,
                "position": position,
                "elite_traits": [],
                "elite_trait_count": 0,
                "elite_bonus": 1.0,
                "draft_adjustment": 0,
                "measurables": {
                    "height": height,
                    "weight": weight,
                    "forty": None,
                    "vertical": None,
                    "broad_jump": None,
                    "bench": None,
                    "three_cone": None,
                    "shuttle": None,
                },
                "thresholds": {},
                "message": "Elite trait analysis in development",
            }
        )

    except Exception as e:
        logger.error(f"Elite profile error: {e}")
        return APIResponse(
            status="success",
            data={
                "player": player_name,
                "position": "Unknown",
                "elite_traits": [],
                "elite_trait_count": 0,
                "elite_bonus": 1.0,
                "draft_adjustment": 0,
                "measurables": {},
                "thresholds": {},
            }
        )


# =============================================================================
# Draft Projection Endpoint
# =============================================================================

@router.get(
    "/draft/project/{player_name}",
    response_model=APIResponse,
    tags=["Draft"],
    summary="Get draft projection for player",
    description="Get NFL draft projection with round, pick range, and contract estimates.",
)
async def get_draft_projection_endpoint(
    request: Request,
    player_name: str,
    api_key: str = Depends(require_api_key),
):
    """Get draft projection for a specific player."""
    try:
        from urllib.parse import unquote
        player_name = unquote(player_name)

        # Get player data
        nil_df = get_nil_players(limit=50000)
        position = "Unknown"
        pff_overall = 0
        stars = 0

        if not nil_df.empty:
            match = nil_df[nil_df["name"].str.lower() == player_name.lower()]
            if not match.empty:
                row = match.iloc[0]
                position = str(row.get("position", "Unknown"))
                pff_overall = float(row.get("pff_overall", 0)) if pd.notna(row.get("pff_overall")) else 0
                stars = int(row.get("stars", 0)) if pd.notna(row.get("stars")) else 0

        # Calculate draft grade based on PFF and recruiting
        draft_grade = min(100, (pff_overall * 0.7) + (stars * 8))

        # Determine round projection
        if draft_grade >= 85:
            projected_round = 1
            pick_range = "1-15"
            draft_probability = 0.95
        elif draft_grade >= 75:
            projected_round = 2
            pick_range = "32-64"
            draft_probability = 0.85
        elif draft_grade >= 65:
            projected_round = 3
            pick_range = "65-100"
            draft_probability = 0.70
        elif draft_grade >= 55:
            projected_round = 5
            pick_range = "140-180"
            draft_probability = 0.45
        else:
            projected_round = None
            pick_range = "UDFA"
            draft_probability = 0.20

        # Calculate letter grade
        if draft_grade >= 90:
            letter_grade = "A+"
        elif draft_grade >= 85:
            letter_grade = "A"
        elif draft_grade >= 80:
            letter_grade = "A-"
        elif draft_grade >= 75:
            letter_grade = "B+"
        elif draft_grade >= 70:
            letter_grade = "B"
        elif draft_grade >= 65:
            letter_grade = "B-"
        elif draft_grade >= 60:
            letter_grade = "C+"
        elif draft_grade >= 55:
            letter_grade = "C"
        else:
            letter_grade = "C-"

        # Estimate contract values
        if projected_round == 1:
            rookie_contract = 15000000 + (15 - int(pick_range.split("-")[0])) * 1000000
            career_estimate = 100000000
        elif projected_round == 2:
            rookie_contract = 5000000
            career_estimate = 40000000
        elif projected_round:
            rookie_contract = 1000000
            career_estimate = 15000000
        else:
            rookie_contract = 750000
            career_estimate = 5000000

        return APIResponse(
            status="success",
            data={
                "player": player_name,
                "position": position,
                "draft_grade": round(draft_grade, 1),
                "draft_letter_grade": letter_grade,
                "projected_round": projected_round,
                "projected_pick": int(pick_range.split("-")[0]) if projected_round else 250,
                "pick_range": pick_range,
                "draft_probability": draft_probability,
                "elite_bonus": 1.0,
                "elite_traits": [],
                "elite_adjustment": 0,
                "rookie_contract_estimate": rookie_contract,
                "career_earnings_estimate": career_estimate,
                "expected_draft_value": rookie_contract * draft_probability,
            }
        )

    except Exception as e:
        logger.error(f"Draft projection error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return APIResponse(
            status="success",
            data={
                "player": player_name,
                "position": "Unknown",
                "draft_grade": 0,
                "draft_letter_grade": "N/A",
                "projected_round": None,
                "projected_pick": 250,
                "pick_range": "UDFA",
                "draft_probability": 0.1,
                "elite_bonus": 1.0,
                "elite_traits": [],
                "elite_adjustment": 0,
                "rookie_contract_estimate": 750000,
                "career_earnings_estimate": 3000000,
                "expected_draft_value": 75000,
            }
        )
