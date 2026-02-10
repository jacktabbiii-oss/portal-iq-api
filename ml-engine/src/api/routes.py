"""API routes for Portal IQ.

All endpoints for NIL valuation, portal intelligence, draft projections,
and roster optimization.
"""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

# Import data loader for real R2 data (no local fallback)
from ..utils.data_loader import (
    get_nil_players,
    get_portal_players,
    get_database_stats,
    get_player_pff_stats,
    get_player_dual_valuation,
    _get_nil_tier,
    normalize_school_name,
    calculate_player_war,
    get_team_roster_composition,
    get_team_pff_summary,
    get_team_cfbd_profile,
    get_on3_team_portal_rankings,
    get_roster_needs,
    get_team_logo_url,
    IDEAL_ROSTER,
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
    """Require API key authentication.

    API keys MUST be set via PORTAL_IQ_API_KEYS environment variable.
    No hardcoded keys — fails closed if env var is missing.
    """
    api_key = request.headers.get("X-API-Key")
    from fastapi import HTTPException

    import os
    keys_env = os.getenv("PORTAL_IQ_API_KEYS", "")
    if not keys_env.strip():
        # In development, allow ENABLE_AUTH=false to skip key check entirely
        if os.getenv("ENABLE_AUTH", "true").lower() == "false":
            return api_key or "dev-no-auth"
        raise HTTPException(
            status_code=503,
            detail={"error": "API keys not configured", "message": "Set PORTAL_IQ_API_KEYS environment variable"},
        )

    valid_keys = {k.strip() for k in keys_env.split(",") if k.strip()}

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

        # Fallback: use CalibratedNILValuator (rank-based, always available if data loaded)
        try:
            from ..models.calibrated_valuator import CalibratedNILValuator
            cal_valuator = CalibratedNILValuator()
            cal_result = cal_valuator.predict(
                name=body.player.name,
                school=body.player.school,
                position=body.player.position,
                stars=player_dict.get("stars"),
            )
            fallback_value = cal_result.nil_value

            response_data = NILPredictResponse(
                player_name=body.player.name,
                school=body.player.school,
                position=body.player.position,
                predicted_value=fallback_value,
                value_tier=_get_nil_tier(fallback_value),
                tier_probabilities={cal_result.nil_tier: 0.7, "moderate": 0.2, "entry": 0.1},
                confidence=0.5,
                value_breakdown=NILValueBreakdown(
                    base_value=fallback_value * 0.5,
                    social_media_premium=0,
                    school_brand_factor=fallback_value * 0.25,
                    position_market_factor=fallback_value * 0.2,
                    draft_potential_premium=fallback_value * 0.05,
                ),
                comparable_players=[],
                percentile=cal_result.percentile if hasattr(cal_result, "percentile") else 50.0,
            )

            return APIResponse(
                status="success",
                data=response_data.model_dump(),
                message="Fallback to calibrated rank-based valuation",
            )
        except Exception as cal_err:
            logger.error(f"CalibratedNILValuator fallback also failed: {cal_err}")
            raise HTTPException(
                status_code=503,
                detail={"error": "Valuation unavailable", "message": "All valuation models failed. Check R2 data availability."},
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

    # Data-driven transfer impact using real NIL data and school tiers
    try:
        # Look up player's real NIL value
        nil_df = get_nil_players(limit=50000)
        name_col = "name" if "name" in nil_df.columns else "player_name"
        val_col = "nil_value" if "nil_value" in nil_df.columns else "predicted_nil"
        player_name_lower = body.player.name.lower()

        current_value = 0
        if not nil_df.empty:
            match = nil_df[nil_df[name_col].str.lower() == player_name_lower]
            if match.empty:
                match = nil_df[nil_df[name_col].str.contains(body.player.name, case=False, na=False)]
            if not match.empty and val_col in match.columns:
                current_value = float(match[val_col].iloc[0])

        if current_value == 0:
            # Use CalibratedNILValuator instead of demo calculation
            try:
                from ..models.calibrated_valuator import CalibratedNILValuator
                cal = CalibratedNILValuator()
                cal_result = cal.predict(
                    name=body.player.name,
                    school=body.player.school,
                    position=body.player.position,
                    stars=player_dict.get("stars"),
                )
                current_value = cal_result.nil_value
            except Exception:
                # Last resort: position-based median from real data
                pos = body.player.position.upper()
                pos_medians = {"QB": 25000, "WR": 8000, "RB": 6000, "EDGE": 10000, "CB": 7000, "OT": 5000, "LB": 5000, "S": 5000, "TE": 5000, "DT": 5000}
                current_value = pos_medians.get(pos, 5000)

        # Get school tier multipliers
        try:
            from ..models.school_tiers import get_school_multiplier
            target_mult = get_school_multiplier(body.target_school)
            current_mult = get_school_multiplier(body.player.school)
        except Exception:
            target_mult = _get_school_multiplier(body.target_school)
            current_mult = _get_school_multiplier(body.player.school)

        projected_value = current_value * (target_mult / max(current_mult, 0.5))

        # Build detailed factors
        factors = {}
        mult_ratio = target_mult / max(current_mult, 0.5)
        if mult_ratio > 1.2:
            factors["program_upgrade"] = f"Moving from {current_mult:.1f}x to {target_mult:.1f}x school multiplier (+{(mult_ratio - 1) * 100:.0f}%)"
        elif mult_ratio < 0.8:
            factors["program_downgrade"] = f"Moving from {current_mult:.1f}x to {target_mult:.1f}x school multiplier ({(mult_ratio - 1) * 100:.0f}%)"
        else:
            factors["lateral_move"] = "Similar program tier — minimal NIL impact from school change"

        # Media market factor
        if target_mult >= 2.3:
            factors["media_market"] = "Elite program provides national visibility — premium NIL opportunities"
        elif target_mult >= 1.4:
            factors["media_market"] = "Strong P4 program — solid regional and national exposure"

        value_change = projected_value - current_value
        value_change_pct = (value_change / current_value * 100) if current_value > 0 else 0

        # Recommendation
        if value_change_pct > 50:
            recommendation = f"Strong NIL upgrade: projected +${value_change:,.0f} ({value_change_pct:+.0f}%). Transfer would significantly increase earning potential."
        elif value_change_pct > 10:
            recommendation = f"Moderate NIL increase: projected +${value_change:,.0f} ({value_change_pct:+.0f}%). Transfer offers improved market positioning."
        elif value_change_pct > -10:
            recommendation = f"NIL-neutral move (${value_change:+,.0f}). Decision should be based on playing time, development, and fit."
        else:
            recommendation = f"NIL decrease: projected ${value_change:,.0f} ({value_change_pct:+.0f}%). Consider whether other factors outweigh financial impact."

        response_data = TransferImpactResponse(
            player_name=body.player.name,
            current_school=body.player.school,
            target_school=body.target_school,
            current_value=round(current_value),
            projected_value=round(projected_value),
            value_change=round(value_change),
            value_change_pct=round(value_change_pct, 1),
            factors=factors,
            recommendation=recommendation,
        )

        return APIResponse(status="success", data=response_data.model_dump())

    except Exception as e:
        logger.error(f"Transfer impact fallback error: {e}")
        raise HTTPException(status_code=500, detail=f"Transfer impact analysis failed: {str(e)}")


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

        # Calculate market stats BEFORE pagination (on full filtered dataset)
        if "nil_value" in df.columns and not df.empty:
            market_cap = float(df["nil_value"].sum())
            avg_value = float(df["nil_value"].mean())
        else:
            market_cap = 0.0
            avg_value = 0.0

        # Apply pagination
        df = df.iloc[offset:offset + limit]

        if df.empty:
            return APIResponse(
                status="success",
                data={"players": [], "total": 0, "total_count": total_count, "avg_value": avg_value, "market_cap": market_cap},
                message="No players found matching criteria"
            )

        # Build response list
        players = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            nil_value = float(row.get("nil_value", 0)) if pd.notna(row.get("nil_value")) else 0
            player_name = str(row.get("name", "Unknown"))

            # Use field names that match frontend expectations
            player_data = {
                "rank": rank,
                "player_id": str(row.get("player_id", player_name.replace(" ", "_").lower())),
                "player_name": player_name,  # Frontend expects player_name
                "position": str(row.get("position", "")),
                "school": str(row.get("school", "")),
                "conference": str(row.get("conference", "")) if pd.notna(row.get("conference")) else None,
                "valuation": nil_value,  # Frontend expects valuation
                "nil_tier": str(row.get("tier", _get_nil_tier(nil_value))),  # Frontend expects nil_tier
                "headshot_url": str(row.get("headshot_url")) if pd.notna(row.get("headshot_url")) else None,
                "valuation_source": str(row.get("valuation_source", "On3")) if pd.notna(row.get("valuation_source")) else "On3",
                # Additional fields for detail view
                "stars": int(row.get("stars", 0)) if pd.notna(row.get("stars")) else None,
                "height": float(row.get("height", 0)) if pd.notna(row.get("height")) else None,
                "weight": float(row.get("weight", 0)) if pd.notna(row.get("weight")) else None,
                "pff_overall": float(row.get("pff_overall", 0)) if pd.notna(row.get("pff_overall")) else None,
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
                "avg_value": avg_value,      # Average NIL value across all filtered players
                "market_cap": market_cap,    # Sum of all NIL values across all filtered players
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

    Response fields match the frontend PlayerSearchResult interface:
    name, position, school, nil_value, stars, headshot_url, pff_overall,
    status, destination_school, data_source
    """
    try:
        results = []
        seen_names = set()
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
                    name = str(row.get("name", ""))
                    seen_names.add(name.lower())
                    results.append({
                        "name": name,
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
            portal_df = get_portal_players(limit=50000)
            if not portal_df.empty:
                mask = (
                    portal_df["name"].str.lower().str.contains(query_lower, na=False) |
                    portal_df.get("origin_school", pd.Series()).str.lower().str.contains(query_lower, na=False) |
                    portal_df.get("position", pd.Series()).str.lower().str.contains(query_lower, na=False)
                )
                matches = portal_df[mask].head(limit if data_type == "portal" else limit // 2)
                for _, row in matches.iterrows():
                    name = str(row.get("name", ""))
                    # Avoid duplicates - but merge portal status into existing NIL result
                    if name.lower() in seen_names:
                        for r in results:
                            if r["name"].lower() == name.lower():
                                r["status"] = str(row.get("status", "available"))
                                r["destination_school"] = str(row.get("destination_school")) if pd.notna(row.get("destination_school")) else None
                                if not r.get("stars"):
                                    r["stars"] = int(row.get("stars", 0)) if pd.notna(row.get("stars")) else None
                                break
                        continue
                    seen_names.add(name.lower())
                    results.append({
                        "name": name,
                        "position": str(row.get("position", "")),
                        "school": str(row.get("origin_school", "")),
                        "nil_value": float(row.get("nil_value", 0)) if pd.notna(row.get("nil_value")) else None,
                        "stars": int(row.get("stars", 0)) if pd.notna(row.get("stars")) else None,
                        "headshot_url": str(row.get("headshot_url")) if pd.notna(row.get("headshot_url")) else None,
                        "pff_overall": None,
                        "status": str(row.get("status", "available")),
                        "destination_school": str(row.get("destination_school")) if pd.notna(row.get("destination_school")) else None,
                        "data_source": "portal",
                    })

        # Sort: exact name matches first, then by NIL value descending
        def sort_key(r):
            name_lower = r["name"].lower()
            exact = 0 if name_lower == query_lower else (1 if name_lower.startswith(query_lower) else 2)
            val = -(r.get("nil_value") or 0)
            return (exact, val)

        results.sort(key=sort_key)

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
    try:
        # Load real data from R2
        df = get_nil_players(limit=50000)
        if df.empty:
            raise HTTPException(status_code=503, detail="NIL data unavailable")

        # Determine value column
        val_col = "nil_value"
        for candidate in ["nil_value", "custom_nil_value", "predicted_nil"]:
            if candidate in df.columns:
                val_col = candidate
                break

        # Apply filters
        filters = {}
        if body.position:
            if "position" in df.columns:
                df = df[df["position"].str.upper() == body.position.upper()]
            filters["position"] = body.position
        if body.conference:
            if "conference" in df.columns:
                df = df[df["conference"].str.lower() == body.conference.lower()]
            filters["conference"] = body.conference

        # Calculate stats
        total_players = len(df)
        values = pd.to_numeric(df[val_col], errors="coerce").fillna(0)
        average_value = float(values.mean()) if total_players > 0 else 0
        median_value = float(values.median()) if total_players > 0 else 0
        total_market_value = float(values.sum())

        # Value by tier
        value_by_tier = {}
        tier_col = "nil_tier" if "nil_tier" in df.columns else None
        if tier_col:
            for tier in df[tier_col].dropna().unique():
                tier_mask = df[tier_col] == tier
                tier_vals = values[tier_mask]
                value_by_tier[str(tier)] = {
                    "count": int(tier_mask.sum()),
                    "avg_value": float(tier_vals.mean()) if len(tier_vals) > 0 else 0,
                }
        else:
            # Calculate tiers from values
            for tier_name, tier_min in [("mega", 2000000), ("premium", 500000),
                                         ("solid", 100000), ("moderate", 25000), ("entry", 0)]:
                if tier_name == "entry":
                    tier_mask = values < 25000
                elif tier_name == "moderate":
                    tier_mask = (values >= 25000) & (values < 100000)
                elif tier_name == "solid":
                    tier_mask = (values >= 100000) & (values < 500000)
                elif tier_name == "premium":
                    tier_mask = (values >= 500000) & (values < 2000000)
                else:
                    tier_mask = values >= 2000000
                tier_vals = values[tier_mask]
                if len(tier_vals) > 0:
                    value_by_tier[tier_name] = {
                        "count": int(tier_mask.sum()),
                        "avg_value": float(tier_vals.mean()),
                    }

        # Top players
        df["_sort_val"] = values
        top_df = df.nlargest(25, "_sort_val")
        top_players = []
        name_col = "name" if "name" in df.columns else "player_name"
        school_col = "school" if "school" in df.columns else "team"
        for _, row in top_df.iterrows():
            player = {
                "name": str(row.get(name_col, "Unknown")),
                "school": str(row.get(school_col, "Unknown")),
                "position": str(row.get("position", "Unknown")),
                "value": float(row.get("_sort_val", 0)),
            }
            for img_col in ["espn_headshot_url", "headshot_url", "image_url"]:
                if img_col in row and pd.notna(row[img_col]):
                    player["headshot_url"] = str(row[img_col])
                    break
            if tier_col and pd.notna(row.get(tier_col)):
                player["tier"] = str(row[tier_col])
            top_players.append(player)

        # Position breakdown for trends
        pos_avgs = {}
        if "position" in df.columns:
            for pos in ["QB", "WR", "RB", "EDGE", "CB", "OT", "TE", "LB", "S", "DT"]:
                pos_mask = df["position"].str.upper() == pos
                pos_vals = values[pos_mask]
                if len(pos_vals) > 0:
                    pos_avgs[pos] = float(pos_vals.mean())
            top_positions = sorted(pos_avgs.items(), key=lambda x: x[1], reverse=True)[:3]
            top_pos_str = ", ".join(f"{p}" for p, _ in top_positions)
        else:
            top_pos_str = "QB, WR, EDGE"

        response_data = MarketReportResponse(
            filters_applied=filters,
            total_players=total_players,
            average_value=average_value,
            median_value=median_value,
            total_market_value=total_market_value,
            value_by_tier=value_by_tier,
            top_players=top_players,
            market_trends=[
                f"Total FBS market value: ${total_market_value:,.0f} across {total_players:,} players",
                f"Average: ${average_value:,.0f} | Median: ${median_value:,.0f}",
                f"Highest valued positions: {top_pos_str}",
            ],
        )

        return APIResponse(
            status="success",
            data=response_data.model_dump(),
        )

    except HTTPException:
        raise
    except (R2NotConfiguredError, R2DataLoadError) as e:
        logger.error(f"R2 error in market report: {e}")
        raise HTTPException(status_code=503, detail="Data storage unavailable")
    except Exception as e:
        logger.error(f"Market report error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Market report failed: {str(e)}")


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

        # Calculate portal stats BEFORE pagination (on full filtered dataset)
        active_in_portal = 0
        committed_count = 0
        schools_active = 0

        if not df.empty and "status" in df.columns:
            status_lower = df["status"].str.lower()
            active_in_portal = int((status_lower == "entered").sum() + (status_lower == "available").sum())
            committed_count = int((status_lower == "committed").sum())

            # Count unique origin schools
            if "origin_school" in df.columns:
                schools_active = int(df["origin_school"].nunique())

        # Apply pagination
        df = df.iloc[offset:offset + limit]

        if df.empty:
            return APIResponse(
                status="success",
                data={
                    "players": [],
                    "total": 0,
                    "total_count": total_count,
                    "active_in_portal": active_in_portal,
                    "committed": committed_count,
                    "schools_active": schools_active,
                },
                message="No portal players found matching criteria"
            )

        # Build response list
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

            origin_school = str(row.get("origin_school", ""))
            dest_school = str(row.get("destination_school", "")) if pd.notna(row.get("destination_school")) else None
            nil_val = float(row.get("nil_value", 0)) if pd.notna(row.get("nil_value")) else None

            player_data = {
                "player_id": str(row.get("player_id", player_name.replace(" ", "_").lower())),
                "player_name": player_name,
                "position": str(row.get("position", "")),
                "origin_school": origin_school,
                "origin_conference": str(row.get("conference")) if pd.notna(row.get("conference")) else None,
                "destination_school": dest_school,
                "stars": int(row.get("stars", 0)) if pd.notna(row.get("stars")) else None,
                "status": player_status,
                "nil_valuation": nil_val,
                "nil_tier": str(row.get("nil_tier", "")) if pd.notna(row.get("nil_tier")) else (_get_nil_tier(nil_val) if nil_val else "entry"),
                "headshot_url": str(row.get("headshot_url")) if pd.notna(row.get("headshot_url")) else None,
                # School logos
                "origin_logo": get_team_logo_url(origin_school),
                "destination_logo": get_team_logo_url(dest_school) if dest_school else None,
                # Additional fields for detail view
                "height": float(row.get("height", 0)) if pd.notna(row.get("height")) else None,
                "weight": float(row.get("weight", 0)) if pd.notna(row.get("weight")) else None,
                "pff_overall": float(row.get("pff_overall", 0)) if pd.notna(row.get("pff_overall")) else None,
            }
            players.append(player_data)

        return APIResponse(
            status="success",
            data={
                "players": players,
                "total": len(players),
                "total_count": total_count,  # Total matching players (for pagination)
                "active_in_portal": active_in_portal,  # Players still in portal
                "committed": committed_count,          # Players who committed
                "schools_active": schools_active,      # Unique origin schools
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

    # Data-driven flight risk calculation
    risk_prob = 0.20  # Base risk: ~20% of players enter portal each cycle
    risk_factors = []
    recommendations = []

    player_name = body.player.name
    school = body.player.school
    position = (body.player.position or "").upper()
    stars = getattr(body.player, "stars", None) or 3

    # Factor 1: School tier (lower tier → higher risk of losing to bigger programs)
    try:
        from ..models.school_tiers import get_school_tier
        tier_name, tier_info = get_school_tier(school)
        tier_mult = tier_info.get("multiplier", 1.0)
        # Low-tier schools lose more players
        if tier_mult <= 0.8:
            risk_prob += 0.15
            risk_factors.append({"factor": "school_tier", "impact": 0.15, "detail": f"{school} is {tier_info.get('label', 'lower tier')} — players often seek P4 opportunities"})
        elif tier_mult <= 1.1:
            risk_prob += 0.05
            risk_factors.append({"factor": "school_tier", "impact": 0.05, "detail": f"{school} is mid-tier — some upward portal movement expected"})
        else:
            risk_prob -= 0.05  # Blue bloods retain better
    except Exception:
        pass

    # Factor 2: Position scarcity — high-demand positions get more offers
    high_demand = {"QB": 0.10, "EDGE": 0.08, "WR": 0.06, "CB": 0.06, "OT": 0.05}
    pos_impact = high_demand.get(position, 0.0)
    if pos_impact > 0:
        risk_prob += pos_impact
        risk_factors.append({"factor": "position_demand", "impact": pos_impact, "detail": f"{position} is a high-demand portal position"})

    # Factor 3: Star rating — high-star players get recruited away more
    if stars >= 4:
        star_impact = 0.10
        risk_prob += star_impact
        risk_factors.append({"factor": "talent_level", "impact": star_impact, "detail": f"{stars}-star talent attracts portal interest from top programs"})
    elif stars <= 2:
        risk_prob -= 0.05  # Lower stars, fewer offers

    # Factor 4: NIL value gap — if player's value is below school average, they may seek better NIL
    try:
        nil_df = get_nil_players(limit=50000)
        val_col = "nil_value" if "nil_value" in nil_df.columns else "predicted_nil"
        if not nil_df.empty and val_col in nil_df.columns:
            school_norm = normalize_school_name(school).lower()
            name_col = "name" if "name" in nil_df.columns else "player_name"
            school_col = "school" if "school" in nil_df.columns else "team"
            # Find player's NIL value
            player_match = nil_df[nil_df[name_col].str.lower() == player_name.lower()]
            player_nil = float(player_match[val_col].iloc[0]) if not player_match.empty else 0
            # School average
            school_players = nil_df[nil_df[school_col].apply(lambda x: normalize_school_name(str(x)).lower()) == school_norm]
            school_avg = float(school_players[val_col].mean()) if not school_players.empty else 50000
            if player_nil > 0 and player_nil < school_avg * 0.5:
                nil_impact = 0.10
                risk_prob += nil_impact
                risk_factors.append({"factor": "nil_undervalued", "impact": nil_impact, "detail": f"Player NIL (${player_nil:,.0f}) is below school average (${school_avg:,.0f})"})
                recommendations.append(f"Increase NIL to at least ${school_avg * 0.7:,.0f} to reduce flight risk")
            replacement_cost = max(player_nil * 1.2, school_avg)
        else:
            replacement_cost = 50000
    except Exception:
        replacement_cost = 50000

    # Factor 5: Team context (coaching changes, losing season, etc.)
    if body.team_context:
        if body.team_context.get("recent_coaching_change"):
            risk_prob += 0.20
            risk_factors.append({"factor": "coaching_change", "impact": 0.20, "detail": "Recent coaching change significantly increases portal risk"})
            recommendations.append("New coaching staff should meet individually with key players")
        if body.team_context.get("losing_season"):
            risk_prob += 0.10
            risk_factors.append({"factor": "team_performance", "impact": 0.10, "detail": "Losing season increases player dissatisfaction"})

    # Factor 6: Check if similar players at this school have transferred
    try:
        portal_df = get_portal_players()
        if not portal_df.empty:
            origin_col = "origin_school" if "origin_school" in portal_df.columns else "from_school"
            if origin_col in portal_df.columns:
                school_departures = portal_df[portal_df[origin_col].apply(
                    lambda x: normalize_school_name(str(x)).lower()) == normalize_school_name(school).lower()]
                if "position" in portal_df.columns:
                    pos_departures = school_departures[school_departures["position"].str.upper() == position]
                    if len(pos_departures) >= 2:
                        risk_prob += 0.08
                        risk_factors.append({"factor": "position_exodus", "impact": 0.08, "detail": f"{len(pos_departures)} {position}s have already left {school} this cycle"})

                # Find comparable transfers for context
                comparable_transfers = []
                pos_transfers = portal_df[portal_df["position"].str.upper() == position] if "position" in portal_df.columns else pd.DataFrame()
                if not pos_transfers.empty and "stars" in pos_transfers.columns:
                    similar = pos_transfers[(pos_transfers["stars"] >= stars - 1) & (pos_transfers["stars"] <= stars + 1)]
                    for _, t in similar.head(3).iterrows():
                        comparable_transfers.append({
                            "name": str(t.get("name", "")),
                            "position": str(t.get("position", "")),
                            "from_school": str(t.get(origin_col, "")),
                            "stars": int(t.get("stars", 0)),
                        })
    except Exception:
        comparable_transfers = []

    # Clamp probability
    risk_prob = max(0.05, min(0.95, risk_prob))

    # Determine risk level
    if risk_prob >= 0.70:
        risk_level = "critical"
    elif risk_prob >= 0.50:
        risk_level = "high"
    elif risk_prob >= 0.30:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # Default recommendations
    if not recommendations:
        if risk_level in ("critical", "high"):
            recommendations = [
                "Schedule immediate meeting to discuss player's role and development plan",
                f"Review NIL compensation — replacement cost estimated at ${replacement_cost:,.0f}",
                "Ensure position coach relationship is strong",
            ]
        elif risk_level == "moderate":
            recommendations = [
                "Maintain regular check-ins on player satisfaction",
                "Ensure competitive NIL compensation within position group",
            ]
        else:
            recommendations = [
                "Continue current development plan",
                "Player appears settled — standard retention practices sufficient",
            ]

    response_data = FlightRiskResponse(
        player_name=body.player.name,
        school=body.player.school,
        flight_risk_probability=round(risk_prob, 3),
        risk_level=risk_level,
        risk_factors=risk_factors,
        retention_recommendations=recommendations,
        estimated_replacement_cost=round(replacement_cost),
        comparable_transfers=comparable_transfers if "comparable_transfers" in dir() else [],
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
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
    """Generate team-wide flight risk report using real roster and portal data."""
    school = body.school
    school_norm = normalize_school_name(school).lower()

    try:
        # Load real data
        nil_df = get_nil_players(limit=50000)
        portal_df = get_portal_players()

        # Find school roster players from NIL data
        name_col = "name" if "name" in nil_df.columns else "player_name"
        school_col = "school" if "school" in nil_df.columns else "team"
        val_col = "nil_value" if "nil_value" in nil_df.columns else "predicted_nil"

        roster_players = nil_df[nil_df[school_col].apply(
            lambda x: normalize_school_name(str(x)).lower()) == school_norm].copy()
        total_roster_size = len(roster_players)

        if total_roster_size == 0:
            raise HTTPException(status_code=404, detail=f"No roster data found for {school}")

        # Get school tier info
        try:
            from ..models.school_tiers import get_school_tier
            tier_name, tier_info = get_school_tier(school)
            tier_mult = tier_info.get("multiplier", 1.0)
        except Exception:
            tier_mult = 1.0

        # Get players who already left via portal
        origin_col = "origin_school" if "origin_school" in portal_df.columns else "from_school"
        departed = portal_df[portal_df[origin_col].apply(
            lambda x: normalize_school_name(str(x)).lower()) == school_norm] if not portal_df.empty else pd.DataFrame()
        departed_positions = departed["position"].str.upper().tolist() if "position" in departed.columns else []

        # Calculate risk for each roster player
        high_demand_positions = {"QB": 0.10, "EDGE": 0.08, "WR": 0.06, "CB": 0.06, "OT": 0.05}
        school_avg_nil = float(roster_players[val_col].mean()) if val_col in roster_players.columns else 50000

        critical_risk = []
        high_risk = []
        position_vulnerability = {}

        for _, player in roster_players.iterrows():
            p_name = str(player.get(name_col, "Unknown"))
            p_pos = str(player.get("position", "")).upper()
            p_nil = float(player.get(val_col, 0)) if val_col in player.index else 0
            p_stars = int(player.get("stars", 3)) if "stars" in player.index else 3

            # Calculate individual risk
            risk = 0.15  # Base
            if tier_mult <= 0.8:
                risk += 0.12
            elif tier_mult <= 1.1:
                risk += 0.04

            risk += high_demand_positions.get(p_pos, 0.0)

            if p_stars >= 4:
                risk += 0.08

            # NIL undervaluation
            if p_nil > 0 and p_nil < school_avg_nil * 0.4:
                risk += 0.12

            # Position already hemorrhaging players
            pos_departed = departed_positions.count(p_pos)
            if pos_departed >= 2:
                risk += 0.05  # Position group is unstable

            risk = max(0.05, min(0.95, risk))

            player_entry = {
                "name": p_name,
                "position": p_pos,
                "risk": round(risk, 3),
                "nil_value": round(p_nil),
            }

            if risk >= 0.55:
                critical_risk.append(player_entry)
            elif risk >= 0.35:
                high_risk.append(player_entry)

            # Track position vulnerability
            if p_pos not in position_vulnerability:
                position_vulnerability[p_pos] = {"count": 0, "total_risk": 0}
            if risk >= 0.30:
                position_vulnerability[p_pos]["count"] += 1
                position_vulnerability[p_pos]["total_risk"] += risk

        # Sort by risk
        critical_risk.sort(key=lambda x: x["risk"], reverse=True)
        high_risk.sort(key=lambda x: x["risk"], reverse=True)

        # Finalize position vulnerability
        pos_vuln_final = {}
        for pos, data in position_vulnerability.items():
            if data["count"] > 0:
                pos_vuln_final[pos] = {
                    "count": data["count"],
                    "avg_risk": round(data["total_risk"] / data["count"], 3),
                }

        total_at_risk = len(critical_risk) + len(high_risk)

        # Estimate WAR at risk
        war_at_risk = sum(
            calculate_player_war(p["position"], p.get("stars", 3) if "stars" in p else 3, p["nil_value"], school)
            for p in critical_risk
        )

        # Retention budget = sum of NIL values for at-risk players * 1.3 (raise needed)
        retention_budget = sum(p["nil_value"] for p in critical_risk + high_risk) * 1.3

        # Build recommendations
        recommendations = []
        if critical_risk:
            top_pos = critical_risk[0]["position"]
            recommendations.append(f"Immediate retention priority: {critical_risk[0]['name']} ({top_pos}) — highest flight risk")
        vulnerable_positions = sorted(pos_vuln_final.items(), key=lambda x: x[1]["avg_risk"], reverse=True)
        for pos, data in vulnerable_positions[:2]:
            recommendations.append(f"Address {pos} depth — {data['count']} players at elevated risk (avg {data['avg_risk']:.0%})")
        if departed_positions:
            recommendations.append(f"Already lost {len(departed_positions)} players to portal this cycle — monitor remaining roster closely")

        response_data = TeamReportResponse(
            school=school,
            analysis_date=datetime.utcnow(),
            total_roster_size=total_roster_size,
            total_at_risk=total_at_risk,
            critical_risk_players=critical_risk[:10],
            high_risk_players=high_risk[:10],
            estimated_wins_at_risk=round(war_at_risk, 1),
            total_retention_budget_needed=round(retention_budget),
            position_vulnerability=pos_vuln_final,
            recommendations=recommendations,
        )

        return APIResponse(status="success", data=response_data.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Team report error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Team report failed: {str(e)}")


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

    # Data-driven fit calculation using real roster, PFF, and NIL data
    position = player_dict.get("position", "").upper()
    stars = player_dict.get("stars") or player_dict.get("overall_rating", 0.75) * 5
    if isinstance(stars, float) and stars <= 1.0:
        stars = round(stars * 5)
    player_nil = player_dict.get("nil_value", 0)
    if not player_nil:
        # Use CalibratedNILValuator for missing NIL values
        try:
            from ..models.calibrated_valuator import CalibratedNILValuator
            cal = CalibratedNILValuator()
            cal_result = cal.predict(
                name=player_dict.get("name", ""),
                school=player_dict.get("school", ""),
                position=position,
                stars=int(stars) if stars else None,
            )
            player_nil = cal_result.nil_value
        except Exception:
            player_nil = 5000  # Conservative floor

    # 1. Position need at target school (35% weight)
    try:
        roster = get_team_roster_composition(body.target_school)
        pos_count = roster.get(position, 0)
        ideal = IDEAL_ROSTER.get(position, 3)
        # Account for outgoing players at that position
        portal_df = get_portal_players()
        outgoing_at_pos = 0
        if not portal_df.empty:
            origin_col = "origin_school" if "origin_school" in portal_df.columns else None
            if origin_col:
                portal_df["_origin_norm"] = portal_df[origin_col].apply(normalize_school_name).str.lower()
                target_norm = normalize_school_name(body.target_school).lower()
                team_outgoing = portal_df[portal_df["_origin_norm"] == target_norm]
                if "position" in team_outgoing.columns:
                    outgoing_at_pos = len(team_outgoing[team_outgoing["position"].str.upper() == position])
        adjusted = pos_count - outgoing_at_pos
        need_ratio = max(0.0, min(1.0, (ideal - adjusted) / max(ideal, 1)))
    except Exception:
        need_ratio = 0.5

    # 2. School tier compatibility (25% weight)
    target_mult = _get_school_multiplier(body.target_school)
    origin_mult = _get_school_multiplier(body.player.school)
    # Higher fit when player's tier matches target's tier
    tier_gap = abs(target_mult - origin_mult) / max(target_mult, origin_mult, 1)
    tier_fit = max(0.3, 1.0 - tier_gap * 0.5)

    # 3. Performance fit via PFF (20% weight)
    try:
        team_pff = get_team_pff_summary(body.target_school)
        player_pff_data = get_player_pff_stats(body.player.name)
        if player_pff_data and team_pff.get("avg_overall", 0) > 0:
            player_grade = player_pff_data.get("pff_overall", 65)
            pff_fit = min(1.0, player_grade / max(team_pff["avg_overall"], 50))
        else:
            # Estimate from stars
            estimated_grade = 50 + (float(stars) * 6)
            team_avg = team_pff.get("avg_overall", 65) if team_pff else 65
            pff_fit = min(1.0, estimated_grade / max(team_avg, 50))
    except Exception:
        pff_fit = 0.65

    # 4. NIL capacity fit (20% weight)
    try:
        nil_players = get_nil_players(school=body.target_school, limit=100)
        if not nil_players.empty and "nil_value" in nil_players.columns:
            team_avg_nil = nil_players["nil_value"].mean()
            nil_ratio = player_nil / max(team_avg_nil, 5000)
            nil_fit = 1.0 if nil_ratio <= 1.5 else max(0.4, 1.0 - (nil_ratio - 1.5) * 0.2)
        else:
            nil_fit = 0.7
    except Exception:
        nil_fit = 0.7

    # Weighted composite fit score
    fit_score = (need_ratio * 0.35 + tier_fit * 0.25 + pff_fit * 0.20 + nil_fit * 0.20)
    fit_score = max(0.1, min(1.0, fit_score))

    # Grade from fit score
    if fit_score >= 0.85:
        fit_grade = "A+"
    elif fit_score >= 0.75:
        fit_grade = "A"
    elif fit_score >= 0.65:
        fit_grade = "B+"
    elif fit_score >= 0.55:
        fit_grade = "B"
    elif fit_score >= 0.45:
        fit_grade = "C+"
    else:
        fit_grade = "C"

    # Projected NIL at target (scale by school multiplier ratio)
    projected_nil = player_nil * (target_mult / max(origin_mult, 0.5))

    # Projected playing time based on fit
    if fit_score >= 0.7 and need_ratio >= 0.5:
        playing_time = "Day 1 Starter"
    elif fit_score >= 0.5:
        playing_time = "Starter"
    elif fit_score >= 0.35:
        playing_time = "Rotational"
    else:
        playing_time = "Depth"

    # Build concerns and strengths
    concerns = []
    strengths = []
    if tier_gap > 0.4:
        concerns.append("Significant program tier gap")
    if need_ratio < 0.2:
        concerns.append(f"Limited snaps available at {position}")
    if nil_fit < 0.5:
        concerns.append("NIL expectations may exceed team budget")
    if need_ratio >= 0.5:
        strengths.append(f"Fills critical {position} need")
    if tier_fit >= 0.8:
        strengths.append("Strong program tier match")
    if pff_fit >= 0.8:
        strengths.append("Above-average production for this roster")
    if not strengths:
        strengths.append("Solid portal addition")

    response_data = PortalFitResponse(
        player_name=body.player.name,
        origin_school=body.player.school,
        target_school=body.target_school,
        fit_score=round(fit_score, 3),
        fit_grade=fit_grade,
        fit_breakdown={
            "position_need": round(need_ratio, 3),
            "tier_fit": round(tier_fit, 3),
            "performance_fit": round(pff_fit, 3),
            "nil_fit": round(nil_fit, 3),
        },
        projected_nil_at_target=round(projected_nil, 0),
        projected_playing_time=playing_time,
        concerns=concerns,
        strengths=strengths,
    )

    return APIResponse(
        status="success",
        data=response_data.model_dump(),
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

    # Data-driven recommendations from real portal data
    try:
        portal_df = get_portal_players()
        if portal_df.empty:
            raise HTTPException(status_code=503, detail="Portal data unavailable")

        school_norm = normalize_school_name(body.school).lower()

        # Get roster needs to determine priority positions
        needs = get_roster_needs(body.school, portal_df)
        priority_positions = needs.get("priority_positions", [])

        # Use provided positions or fall back to calculated needs
        target_positions = body.positions_of_need or priority_positions or ["QB", "WR", "EDGE", "CB"]

        # Filter available portal players
        status_col = "status" if "status" in portal_df.columns else None
        if status_col:
            available = portal_df[portal_df[status_col].fillna("").str.lower() == "available"].copy()
        else:
            available = portal_df.copy()

        # Filter by target positions
        if "position" in available.columns and target_positions:
            pos_upper = [p.upper() for p in target_positions]
            available = available[available["position"].str.upper().isin(pos_upper)]

        if available.empty:
            # Broaden search if no matches
            available = portal_df.copy()
            if status_col:
                available = available[available[status_col].fillna("").str.lower() == "available"]

        # Get NIL values and calculate WAR for each
        val_col = "nil_value" if "nil_value" in available.columns else "nil_valuation"
        targets = []
        for _, player in available.iterrows():
            p_name = str(player.get("name", "Unknown"))
            p_pos = str(player.get("position", "")).upper()
            p_nil = float(player.get(val_col, 0)) if val_col in player.index and pd.notna(player.get(val_col)) else 0
            p_stars = int(player.get("stars", 3)) if "stars" in player.index and pd.notna(player.get("stars")) else 3
            origin_col = "origin_school" if "origin_school" in player.index else "from_school"
            p_origin = str(player.get(origin_col, ""))

            # Skip if over budget
            if p_nil > body.budget * 0.5:
                continue

            war = calculate_player_war(p_pos, p_stars, p_nil, body.school)

            # Fit score: position need + talent level
            pos_need_boost = 1.2 if p_pos in target_positions else 0.8
            fit = min(1.0, (war / 3.0) * pos_need_boost)
            value_rating = min(1.0, war / max(p_nil / 100000, 0.5))  # WAR per $100K

            targets.append({
                "name": p_name,
                "position": p_pos,
                "origin_school": normalize_school_name(p_origin),
                "stars": p_stars,
                "projected_nil": round(p_nil),
                "fit_score": round(fit, 3),
                "win_impact": round(war, 2),
                "value_rating": round(min(1.0, value_rating), 3),
            })

        # Sort by fit_score * win_impact for best overall targets
        targets.sort(key=lambda x: x["fit_score"] * x["win_impact"], reverse=True)
        targets = targets[:body.max_targets]

        # Budget allocation by position
        budget_allocation = {}
        total_projected = sum(t["projected_nil"] for t in targets)
        for pos in target_positions:
            pos_targets = [t for t in targets if t["position"] == pos]
            pos_nil = sum(t["projected_nil"] for t in pos_targets)
            budget_allocation[pos] = round(pos_nil)

        # Projected WAR improvement
        total_war = sum(t["win_impact"] for t in targets)

        # Strategy text
        if total_war >= 5:
            strategy = f"Aggressive portal strategy: {len(targets)} targets adding {total_war:.1f} projected WAR"
        elif total_war >= 2:
            strategy = f"Targeted portal additions: {len(targets)} players filling key gaps, {total_war:.1f} WAR"
        else:
            strategy = f"Selective portal approach: focus on highest-impact {target_positions[0]} targets"

        response_data = PortalRecommendationsResponse(
            school=body.school,
            budget=body.budget,
            targets=targets,
            positions_prioritized=target_positions,
            budget_allocation_suggestion=budget_allocation,
            projected_roster_improvement=round(total_war, 1),
            acquisition_strategy=strategy,
        )

        return APIResponse(status="success", data=response_data.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portal recommendations error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


@router.get(
    "/portal/rankings",
    response_model=APIResponse,
    tags=["Portal"],
    summary="Team portal rankings",
    description="Get teams ranked by portal success based on committed transfers.",
)
async def portal_rankings(
    request: Request,
    limit: int = 25,
    api_key: str = Depends(require_api_key),
):
    """Get team rankings based on transfer portal success."""
    try:
        # Load portal data
        df = get_portal_players()
        if df.empty:
            return APIResponse(
                status="success",
                data={"rankings": [], "total_teams": 0}
            )

        # Only count committed players for rankings
        status_col = "status" if "status" in df.columns else None
        if status_col:
            status_lower = df[status_col].fillna("").str.lower()
            committed_df = df[status_lower == "committed"].copy()
        else:
            committed_df = df.copy()

        # Find destination column
        dest_col = None
        for col in ["destination_school", "destination", "new_school", "committed_school"]:
            if col in committed_df.columns:
                dest_col = col
                break

        if not dest_col or committed_df.empty:
            return APIResponse(
                status="success",
                data={"rankings": [], "total_teams": 0}
            )

        # Normalize school names for proper matching
        committed_df["_dest_normalized"] = committed_df[dest_col].apply(normalize_school_name)

        # Load On3 team portal rankings for comparison
        try:
            on3_rankings_df = get_on3_team_portal_rankings()
            on3_lookup = {}
            if not on3_rankings_df.empty:
                for _, row in on3_rankings_df.iterrows():
                    team_name = normalize_school_name(str(row.get("team", row.get("school", ""))))
                    on3_lookup[team_name.lower()] = {
                        "rank": int(row.get("overall_rank", 0)) if pd.notna(row.get("overall_rank")) else None,
                        "score": float(row.get("overall_score", 0)) if pd.notna(row.get("overall_score")) else None,
                    }
        except Exception:
            on3_lookup = {}

        # Group by normalized destination school
        rankings = []
        grouped = committed_df.groupby("_dest_normalized")

        for team, group in grouped:
            if pd.isna(team) or str(team).strip() == "":
                continue

            team_str = str(team)
            transfers_in = len(group)
            total_stars = group["stars"].sum() if "stars" in group.columns else 0
            avg_stars = group["stars"].mean() if "stars" in group.columns and transfers_in > 0 else 0
            total_nil = group["nil_value"].sum() if "nil_value" in group.columns else 0

            # Calculate real WAR for each player
            war_added = 0.0
            for _, player in group.iterrows():
                pos = str(player.get("position", "")) or ""
                stars = player.get("stars", 0)
                nil_val = player.get("nil_value", 0)
                war_added += calculate_player_war(pos, stars, nil_val if pd.notna(nil_val) else 0, team_str)

            # Multi-factor portal score
            # WAR score (40% weight) - normalize to 0-100 scale
            war_score = min(war_added / 15.0, 1.0) * 100

            # NIL score (15% weight) - log scale normalized
            nil_total = float(total_nil) if pd.notna(total_nil) else 0
            nil_score = math.log1p(nil_total) / math.log1p(5_000_000) * 100

            # Talent score (20% weight) - avg stars normalized
            avg_stars_val = float(avg_stars) if pd.notna(avg_stars) else 0
            talent_score = (avg_stars_val / 5.0) * 100

            # Volume score (10% weight) - more transfers = better, cap at 20
            volume_score = min(transfers_in / 20.0, 1.0) * 100

            # Position diversity score (15% weight)
            if "position" in group.columns:
                unique_positions = group["position"].dropna().nunique()
                balance_score = min(unique_positions / min(transfers_in, 8), 1.0) * 100
            else:
                balance_score = 50.0

            portal_score = (
                war_score * 0.40 +
                nil_score * 0.15 +
                talent_score * 0.20 +
                volume_score * 0.10 +
                balance_score * 0.15
            )

            # Grade thresholds calibrated for realistic distribution
            if portal_score >= 85:
                grade = "A+"
            elif portal_score >= 70:
                grade = "A"
            elif portal_score >= 55:
                grade = "B+"
            elif portal_score >= 40:
                grade = "B"
            elif portal_score >= 25:
                grade = "C+"
            elif portal_score >= 15:
                grade = "C"
            else:
                grade = "D"

            # Top acquisitions (sorted by WAR contribution)
            top_acquisitions = []
            acq_list = []
            for _, player in group.iterrows():
                pos = str(player.get("position", "")) or ""
                stars_val = player.get("stars", 0)
                nil_val = player.get("nil_value", 0)
                p_war = calculate_player_war(pos, stars_val, nil_val if pd.notna(nil_val) else 0, team_str)
                acq_list.append({
                    "name": str(player.get("name", "Unknown")),
                    "position": pos,
                    "stars": int(stars_val) if pd.notna(stars_val) else 0,
                    "nil_value": float(nil_val) if pd.notna(nil_val) else 0,
                    "war": round(p_war, 2),
                })
            acq_list.sort(key=lambda x: x["war"], reverse=True)
            top_acquisitions = acq_list[:5]

            # On3 comparison
            on3_info = on3_lookup.get(team_str.lower(), {})

            rankings.append({
                "team": team_str,
                "team_logo": get_team_logo_url(team_str),
                "grade": grade,
                "portal_score": round(portal_score, 1),
                "war_added": round(war_added, 2),
                "total_nil_invested": round(nil_total, 0),
                "breakdown": {
                    "transfers_in": transfers_in,
                    "total_stars": int(total_stars) if pd.notna(total_stars) else 0,
                    "avg_stars": round(avg_stars_val, 1),
                    "war_score": round(war_score, 1),
                    "nil_score": round(nil_score, 1),
                    "talent_score": round(talent_score, 1),
                    "balance_score": round(balance_score, 1),
                },
                "on3_rank": on3_info.get("rank"),
                "on3_score": on3_info.get("score"),
                "top_acquisitions": top_acquisitions,
            })

        # Sort by portal score
        rankings.sort(key=lambda x: x["portal_score"], reverse=True)

        return APIResponse(
            status="success",
            data={
                "rankings": rankings[:limit],
                "total_teams": len(rankings),
            }
        )

    except Exception as e:
        logger.error(f"Portal rankings error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to calculate rankings: {str(e)}")


@router.get(
    "/portal/team/{team}",
    response_model=APIResponse,
    tags=["Portal"],
    summary="Team portal activity",
    description="Get incoming and outgoing transfers for a specific team.",
)
async def portal_team_activity(
    request: Request,
    team: str,
    season: int = 2026,
    api_key: str = Depends(require_api_key),
):
    """Get incoming and outgoing transfers for a team."""
    try:
        df = get_portal_players()
        if df.empty:
            return APIResponse(
                status="success",
                data={
                    "team": team,
                    "season": season,
                    "incoming": [],
                    "outgoing": [],
                    "net_talent_change": 0,
                    "summary": {"incoming_count": 0, "outgoing_count": 0, "net_count": 0,
                                "incoming_war": 0, "outgoing_war": 0, "net_war": 0},
                }
            )

        team_normalized = normalize_school_name(team).lower()

        # Find column names
        origin_col = None
        dest_col = None
        for col in ["origin_school", "origin", "from_school", "school"]:
            if col in df.columns:
                origin_col = col
                break
        for col in ["destination_school", "destination", "new_school", "committed_school"]:
            if col in df.columns:
                dest_col = col
                break

        # Normalize school names in the data for matching
        if origin_col:
            df["_origin_norm"] = df[origin_col].apply(normalize_school_name).str.lower()
        if dest_col:
            df["_dest_norm"] = df[dest_col].apply(normalize_school_name).str.lower()

        # Outgoing - players who left this team
        outgoing = []
        outgoing_war_total = 0.0
        if origin_col:
            outgoing_df = df[df["_origin_norm"] == team_normalized]
            for _, row in outgoing_df.iterrows():
                pos = str(row.get("position", "")) or ""
                stars = row.get("stars", 0)
                nil_val = row.get("nil_value", 0)
                p_war = calculate_player_war(pos, stars, nil_val if pd.notna(nil_val) else 0, team)
                outgoing_war_total += p_war
                outgoing.append({
                    "player_id": str(row.get("player_id", "")),
                    "player_name": str(row.get("name", "Unknown")),
                    "position": pos,
                    "origin_school": normalize_school_name(str(row.get(origin_col, ""))),
                    "destination_school": normalize_school_name(str(row.get(dest_col, ""))) if dest_col and pd.notna(row.get(dest_col)) else None,
                    "stars": int(stars) if pd.notna(stars) else None,
                    "status": str(row.get("status", "available")).lower(),
                    "nil_valuation": float(nil_val) if pd.notna(nil_val) else None,
                    "war": round(p_war, 2),
                })

        # Incoming - players who committed to this team
        incoming = []
        incoming_war_total = 0.0
        incoming_nil_total = 0.0
        if dest_col:
            incoming_df = df[df["_dest_norm"] == team_normalized]
            for _, row in incoming_df.iterrows():
                pos = str(row.get("position", "")) or ""
                stars = row.get("stars", 0)
                nil_val = row.get("nil_value", 0)
                nil_float = float(nil_val) if pd.notna(nil_val) else 0
                p_war = calculate_player_war(pos, stars, nil_float, team)
                incoming_war_total += p_war
                incoming_nil_total += nil_float
                incoming.append({
                    "player_id": str(row.get("player_id", "")),
                    "player_name": str(row.get("name", "Unknown")),
                    "position": pos,
                    "origin_school": normalize_school_name(str(row.get(origin_col, ""))) if origin_col else "",
                    "destination_school": normalize_school_name(str(row.get(dest_col, ""))),
                    "stars": int(stars) if pd.notna(stars) else None,
                    "status": "committed",
                    "nil_valuation": nil_float,
                    "war": round(p_war, 2),
                })

        # Sort by WAR
        incoming.sort(key=lambda x: x.get("war", 0), reverse=True)
        outgoing.sort(key=lambda x: x.get("war", 0), reverse=True)

        # Calculate roster needs from portal activity
        try:
            roster_needs = get_roster_needs(team, incoming, outgoing)
        except Exception:
            roster_needs = None

        net_war = incoming_war_total - outgoing_war_total

        return APIResponse(
            status="success",
            data={
                "team": team,
                "season": season,
                "incoming": incoming,
                "outgoing": outgoing,
                "net_talent_change": sum(p.get("stars", 0) or 0 for p in incoming) - sum(p.get("stars", 0) or 0 for p in outgoing),
                "net_war": round(net_war, 2),
                "total_nil_invested": round(incoming_nil_total, 0),
                "summary": {
                    "incoming_count": len(incoming),
                    "outgoing_count": len(outgoing),
                    "net_count": len(incoming) - len(outgoing),
                    "incoming_war": round(incoming_war_total, 2),
                    "outgoing_war": round(outgoing_war_total, 2),
                    "net_war": round(net_war, 2),
                },
                "roster_needs": roster_needs,
            }
        )

    except Exception as e:
        logger.error(f"Team portal activity error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get team activity: {str(e)}")


@router.get(
    "/roster/{team}/needs",
    response_model=APIResponse,
    tags=["Portal"],
    summary="Team roster needs",
    description="Get position-by-position roster needs analysis for a team.",
)
async def team_roster_needs(
    request: Request,
    team: str,
    api_key: str = Depends(require_api_key),
):
    """Get roster needs analysis for a team based on current roster and portal activity."""
    try:
        # Get portal data for incoming/outgoing
        df = get_portal_players()
        incoming_players = []
        outgoing_players = []

        if not df.empty:
            origin_col = None
            dest_col = None
            for col in ["origin_school", "origin", "from_school", "school"]:
                if col in df.columns:
                    origin_col = col
                    break
            for col in ["destination_school", "destination", "new_school", "committed_school"]:
                if col in df.columns:
                    dest_col = col
                    break

            team_normalized = normalize_school_name(team).lower()

            if origin_col:
                df["_origin_norm"] = df[origin_col].apply(normalize_school_name).str.lower()
                outgoing_df = df[df["_origin_norm"] == team_normalized]
                for _, row in outgoing_df.iterrows():
                    outgoing_players.append({
                        "position": str(row.get("position", "")),
                        "stars": int(row.get("stars", 0)) if pd.notna(row.get("stars")) else 0,
                    })

            if dest_col:
                df["_dest_norm"] = df[dest_col].apply(normalize_school_name).str.lower()
                committed = df[df["_dest_norm"] == team_normalized]
                if "status" in df.columns:
                    committed = committed[committed["status"].fillna("").str.lower() == "committed"]
                for _, row in committed.iterrows():
                    incoming_players.append({
                        "position": str(row.get("position", "")),
                        "stars": int(row.get("stars", 0)) if pd.notna(row.get("stars")) else 0,
                    })

        # Calculate roster needs
        needs_data = get_roster_needs(team, incoming_players, outgoing_players)

        # Build analysis text
        priority = needs_data.get("priority_positions", [])
        if priority:
            analysis = f"Priority needs at {team}: {', '.join(priority)}. "
            for pos in priority[:3]:
                pos_need = needs_data["needs"].get(pos, {})
                net = pos_need.get("net", 0)
                if net < 0:
                    analysis += f"Lost {abs(net)} {pos}{'s' if abs(net) > 1 else ''} to portal. "
                elif pos_need.get("need_level") == "critical":
                    analysis += f"{pos} depth is below ideal roster size. "
        else:
            analysis = f"{team} has no critical roster needs currently."

        return APIResponse(
            status="success",
            data={
                "team": team,
                "needs": needs_data.get("needs", {}),
                "priority_positions": priority,
                "analysis": analysis,
            }
        )

    except Exception as e:
        logger.error(f"Roster needs error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to calculate roster needs: {str(e)}")


@router.get(
    "/teams/logo/{team}",
    response_model=APIResponse,
    tags=["Teams"],
    summary="Get team logo URL",
    description="Get ESPN team logo URL for a specific school.",
)
async def team_logo(
    team: str,
    api_key: str = Depends(require_api_key),
):
    """Get team logo URL from ESPN CDN."""
    logo_url = get_team_logo_url(team)
    return APIResponse(
        status="success",
        data={
            "team": team,
            "logo_url": logo_url,
        }
    )


@router.get(
    "/teams/logos",
    response_model=APIResponse,
    tags=["Teams"],
    summary="Get all team logos",
    description="Get ESPN team logo URLs for all FBS schools.",
)
async def all_team_logos(
    api_key: str = Depends(require_api_key),
):
    """Get all available team logo URLs."""
    from ..utils.data_loader import get_all_team_logos
    logos = get_all_team_logos()
    return APIResponse(
        status="success",
        data={
            "logos": logos,
            "count": len(logos),
        }
    )


@router.get(
    "/schools/tiers",
    response_model=APIResponse,
    tags=["Teams"],
    summary="Get all school tiers",
    description="Get data-driven school tier classifications from CFBD data (wins, SP+, talent).",
)
async def school_tiers_endpoint(
    api_key: str = Depends(require_api_key),
):
    """Get all school tiers calculated from real CFBD performance data."""
    try:
        from ..models.school_tiers import get_all_school_tiers_for_api, get_school_tiers

        tier_data = get_all_school_tiers_for_api()

        # Also add a flat list sorted by score for easy consumption
        all_tiers = get_school_tiers()
        flat_list = sorted(
            [{"school": school, **info} for school, info in all_tiers.items()],
            key=lambda x: x.get("score", 0),
            reverse=True,
        )

        # Add team logo URLs
        for entry in flat_list:
            logo = get_team_logo_url(entry["school"])
            if logo:
                entry["logo_url"] = logo

        tier_data["all_schools"] = flat_list

        return APIResponse(status="success", data=tier_data)

    except Exception as e:
        logger.error(f"School tiers error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"School tiers failed: {str(e)}")


@router.get(
    "/schools/{school}/tier",
    response_model=APIResponse,
    tags=["Teams"],
    summary="Get school tier",
    description="Get tier classification for a specific school.",
)
async def school_tier_detail(
    school: str,
    api_key: str = Depends(require_api_key),
):
    """Get tier info for a single school from CFBD data."""
    try:
        from urllib.parse import unquote
        school = unquote(school)

        from ..models.school_tiers import get_school_tier

        tier_name, tier_info = get_school_tier(school)

        # Add logo
        logo = get_team_logo_url(school)

        return APIResponse(
            status="success",
            data={
                "school": school,
                "logo_url": logo,
                **tier_info,
            }
        )

    except Exception as e:
        logger.error(f"School tier detail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

    # Data-driven draft projection from real player data
    try:
        nil_df = get_nil_players(limit=50000)
        name_col = "name" if "name" in nil_df.columns else "player_name"
        val_col = "nil_value" if "nil_value" in nil_df.columns else "predicted_nil"
        player_name_lower = body.player.name.lower()
        position = (body.player.position or "").upper()

        # Find player in data
        player_match = nil_df[nil_df[name_col].str.lower() == player_name_lower] if not nil_df.empty else pd.DataFrame()
        if player_match.empty and not nil_df.empty:
            player_match = nil_df[nil_df[name_col].str.contains(body.player.name, case=False, na=False)]

        stars = int(player_dict.get("stars", 3))
        nil_value = 0
        pff_grade = 0
        school = body.player.school or ""

        if not player_match.empty:
            row = player_match.iloc[0]
            stars = int(row.get("stars", stars)) if pd.notna(row.get("stars")) else stars
            nil_value = float(row.get(val_col, 0)) if pd.notna(row.get(val_col)) else 0
            pff_grade = float(row.get("pff_overall", 0)) if pd.notna(row.get("pff_overall")) else 0
            school = str(row.get("school", school)) if pd.notna(row.get("school")) else school

        # Draft probability model using stars + PFF + NIL + position
        position_draft_rates = {
            "QB": 0.12, "WR": 0.08, "RB": 0.06, "TE": 0.05, "OT": 0.07,
            "OG": 0.04, "C": 0.03, "EDGE": 0.09, "DT": 0.05, "LB": 0.06,
            "CB": 0.08, "S": 0.05, "K": 0.01, "P": 0.01,
        }
        base_prob = position_draft_rates.get(position, 0.04)

        # Stars multiplier
        star_mult = {5: 6.0, 4: 3.5, 3: 1.5, 2: 0.5, 1: 0.2}
        draft_prob = base_prob * star_mult.get(stars, 1.0)

        # PFF boost
        if pff_grade >= 85:
            draft_prob *= 2.0
        elif pff_grade >= 75:
            draft_prob *= 1.5
        elif pff_grade >= 65:
            draft_prob *= 1.2

        # NIL indicates market perception
        if nil_value >= 500000:
            draft_prob *= 1.5
        elif nil_value >= 100000:
            draft_prob *= 1.2

        draft_prob = min(0.95, draft_prob)

        # Project round based on probability
        if draft_prob >= 0.80:
            projected_round = 1
            pick_range = "1-32"
            grade = "A+"
            contract_est = 20000000
            career_est = 100000000
        elif draft_prob >= 0.60:
            projected_round = 2
            pick_range = "33-64"
            grade = "A"
            contract_est = 10000000
            career_est = 50000000
        elif draft_prob >= 0.40:
            projected_round = 3
            pick_range = "65-100"
            grade = "B+"
            contract_est = 5000000
            career_est = 25000000
        elif draft_prob >= 0.25:
            projected_round = 4
            pick_range = "101-140"
            grade = "B"
            contract_est = 4000000
            career_est = 15000000
        elif draft_prob >= 0.15:
            projected_round = 5
            pick_range = "141-180"
            grade = "C+"
            contract_est = 3500000
            career_est = 8000000
        else:
            projected_round = None
            pick_range = None
            grade = "C"
            contract_est = 2800000  # UDFA
            career_est = 4000000

        # Strengths/weaknesses from data
        strengths = []
        weaknesses = []
        if pff_grade >= 75:
            strengths.append(f"Strong PFF grade ({pff_grade:.1f})")
        if stars >= 4:
            strengths.append(f"{stars}-star recruit — elite pedigree")
        if nil_value >= 200000:
            strengths.append("High market value indicates strong production/brand")
        if not strengths:
            strengths.append("Solid production at position")

        if pff_grade > 0 and pff_grade < 65:
            weaknesses.append(f"Below-average PFF grade ({pff_grade:.1f})")
        if stars <= 2:
            weaknesses.append("Low recruiting ranking — must prove at combine")
        if not weaknesses:
            weaknesses.append("Needs strong pro day/combine numbers")

        # Stock trend
        stock_trend = "rising" if draft_prob >= 0.5 and pff_grade >= 70 else "stable" if draft_prob >= 0.25 else "falling"

        response_data = DraftProjectResponse(
            player_name=body.player.name,
            position=body.player.position,
            draft_eligible=body.player.class_year in ["Junior", "Senior", "Redshirt Junior", "Redshirt Senior"],
            projected_round=projected_round,
            projected_pick_range=pick_range,
            draft_probability=round(draft_prob, 3),
            draft_grade=grade,
            expected_draft_value=projected_round * 100 if projected_round else 50,
            rookie_contract_estimate=contract_est,
            career_earnings_estimate=career_est,
            strengths=strengths,
            weaknesses=weaknesses,
            comparable_prospects=[],
            stock_trend=stock_trend,
        )

        return APIResponse(status="success", data=response_data.model_dump())

    except Exception as e:
        logger.error(f"Draft projection fallback error: {e}")
        raise HTTPException(status_code=500, detail=f"Draft projection failed: {str(e)}")


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

    # Data-driven mock draft from real player data
    try:
        nil_df = get_nil_players(limit=50000)
        if nil_df.empty:
            raise HTTPException(status_code=503, detail="Player data unavailable")

        name_col = "name" if "name" in nil_df.columns else "player_name"
        val_col = "nil_value" if "nil_value" in nil_df.columns else "predicted_nil"
        school_col = "school" if "school" in nil_df.columns else "team"
        picks_per_round = 32
        total_picks = body.num_rounds * picks_per_round

        # Calculate draft score for each player
        position_draft_rates = {
            "QB": 0.12, "WR": 0.08, "RB": 0.06, "TE": 0.05, "OT": 0.07,
            "OG": 0.04, "C": 0.03, "EDGE": 0.09, "DT": 0.05, "LB": 0.06,
            "CB": 0.08, "S": 0.05, "K": 0.01, "P": 0.01,
        }
        star_mult = {5: 6.0, 4: 3.5, 3: 1.5, 2: 0.5, 1: 0.2}

        prospects = []
        for _, row in nil_df.iterrows():
            pos = str(row.get("position", "")).upper()
            if pos in ("K", "P"):
                continue  # Skip specialists for draft
            stars = int(row.get("stars", 3)) if pd.notna(row.get("stars")) else 3
            nil_val = float(row.get(val_col, 0)) if pd.notna(row.get(val_col)) else 0
            pff = float(row.get("pff_overall", 0)) if pd.notna(row.get("pff_overall")) else 0

            base = position_draft_rates.get(pos, 0.04)
            score = base * star_mult.get(stars, 1.0)
            if pff >= 80:
                score *= 2.0
            elif pff >= 70:
                score *= 1.5
            if nil_val >= 500000:
                score *= 1.5
            elif nil_val >= 100000:
                score *= 1.2

            prospects.append({
                "name": str(row.get(name_col, "")),
                "position": pos,
                "school": str(row.get(school_col, "")),
                "stars": stars,
                "nil_value": nil_val,
                "pff_grade": pff,
                "draft_score": score,
            })

        # Sort by draft score
        prospects.sort(key=lambda x: x["draft_score"], reverse=True)

        # Build draft board
        draft_board = []
        position_dist = {}
        top_by_position = {}

        for i, p in enumerate(prospects[:total_picks]):
            rnd = (i // picks_per_round) + 1
            if rnd == 1:
                grade = "A+" if i < 5 else "A"
            elif rnd == 2:
                grade = "B+"
            elif rnd == 3:
                grade = "B"
            else:
                grade = "C+" if rnd <= 5 else "C"

            draft_board.append({
                "pick": i + 1,
                "round": rnd,
                "player": p["name"],
                "position": p["position"],
                "school": normalize_school_name(p["school"]),
                "grade": grade,
                "stars": p["stars"],
                "nil_value": round(p["nil_value"]),
            })

            pos = p["position"]
            position_dist[pos] = position_dist.get(pos, 0) + 1
            if pos not in top_by_position:
                top_by_position[pos] = []
            if len(top_by_position[pos]) < 3:
                top_by_position[pos].append({
                    "name": p["name"],
                    "school": normalize_school_name(p["school"]),
                    "pick": i + 1,
                })

        response_data = MockDraftResponse(
            season_year=body.season_year,
            num_rounds=body.num_rounds,
            total_picks=len(draft_board),
            draft_board=draft_board,
            position_distribution=position_dist,
            top_prospects_by_position=top_by_position,
            generated_at=datetime.utcnow(),
        )

        return APIResponse(status="success", data=response_data.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mock draft fallback error: {e}")
        raise HTTPException(status_code=500, detail=f"Mock draft failed: {str(e)}")


@router.get(
    "/draft/project/{player_name}",
    response_model=APIResponse,
    tags=["Draft"],
    summary="Project draft position for player",
    description="Get NFL draft projection for a specific player by name.",
)
async def draft_project_by_name(
    request: Request,
    player_name: str,
    api_key: str = Depends(require_api_key),
):
    """Project draft position for a player by name."""
    try:
        from urllib.parse import unquote
        player_name = unquote(player_name)

        from ..models.draft_projector import project_draft_position
        from ..utils.data_loader import get_nil_players

        # Find player in NIL data
        nil_df = get_nil_players(limit=50000)
        player_data = {}

        if not nil_df.empty and "name" in nil_df.columns:
            match = nil_df[nil_df["name"].str.contains(player_name, case=False, na=False)]
            if not match.empty:
                player_data = match.iloc[0].to_dict()

        if not player_data:
            raise HTTPException(status_code=404, detail=f"Player not found: {player_name}")

        projection = project_draft_position(player_data)

        return APIResponse(
            status="success",
            data=projection,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Draft projection error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/draft/comparables/{player_name}",
    response_model=APIResponse,
    tags=["Draft"],
    summary="Get draft comparables",
    description="Get historical draft comparables for a player.",
)
async def draft_comparables(
    request: Request,
    player_name: str,
    limit: int = Query(5, ge=1, le=20, description="Max comparables"),
    api_key: str = Depends(require_api_key),
):
    """Get historical draft comparables for a player."""
    try:
        from urllib.parse import unquote
        player_name = unquote(player_name)

        from ..models.draft_projector import get_historical_comparables
        from ..utils.data_loader import get_nil_players

        # Find player in NIL data
        nil_df = get_nil_players(limit=50000)
        player_data = {}

        if not nil_df.empty and "name" in nil_df.columns:
            match = nil_df[nil_df["name"].str.contains(player_name, case=False, na=False)]
            if not match.empty:
                player_data = match.iloc[0].to_dict()

        if not player_data:
            return APIResponse(
                status="success",
                data={"player": player_name, "comparables": []},
            )

        comparables = get_historical_comparables(player_data, limit=limit)

        return APIResponse(
            status="success",
            data={
                "player": player_name,
                "position": player_data.get("position", "Unknown"),
                "comparables": comparables,
            },
        )

    except Exception as e:
        logger.error(f"Draft comparables error: {e}")
        return APIResponse(
            status="success",
            data={"player": player_name, "comparables": []},
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

    # Data-driven roster optimization from real player data
    try:
        nil_df = get_nil_players(limit=50000)
        school_norm = normalize_school_name(body.school).lower()
        name_col = "name" if "name" in nil_df.columns else "player_name"
        school_col = "school" if "school" in nil_df.columns else "team"
        val_col = "nil_value" if "nil_value" in nil_df.columns else "predicted_nil"

        # Get school's players
        roster = nil_df[nil_df[school_col].apply(
            lambda x: normalize_school_name(str(x)).lower()) == school_norm].copy()

        if roster.empty:
            raise HTTPException(status_code=404, detail=f"No roster data found for {body.school}")

        # Sort by WAR (calculated from position, stars, NIL)
        allocations = []
        position_breakdown = {}
        total_allocated = 0

        for _, row in roster.iterrows():
            p_name = str(row.get(name_col, ""))
            p_pos = str(row.get("position", "")).upper()
            p_nil = float(row.get(val_col, 0)) if pd.notna(row.get(val_col)) else 0
            p_stars = int(row.get("stars", 3)) if pd.notna(row.get("stars")) else 3
            war = calculate_player_war(p_pos, p_stars, p_nil, body.school)

            allocations.append({
                "player": p_name,
                "position": p_pos,
                "current_nil": round(p_nil),
                "recommended_nil": round(p_nil),  # Will be adjusted below
                "war": round(war, 2),
                "stars": p_stars,
            })

        # Sort by WAR to prioritize top players
        allocations.sort(key=lambda x: x["war"], reverse=True)

        # Budget-constrained allocation: distribute budget proportional to WAR
        total_war = sum(a["war"] for a in allocations) or 1
        budget = body.total_budget

        for alloc in allocations:
            war_share = alloc["war"] / total_war
            recommended = budget * war_share
            # Cap individual allocation at 25% of total budget
            recommended = min(recommended, budget * 0.25)
            # Don't recommend less than current NIL (avoid pay cuts)
            recommended = max(recommended, alloc["current_nil"] * 0.8)
            alloc["recommended_nil"] = round(recommended)
            total_allocated += recommended

            pos = alloc["position"]
            position_breakdown[pos] = position_breakdown.get(pos, 0) + recommended

        # Round position breakdown
        position_breakdown = {k: round(v) for k, v in position_breakdown.items()}

        # Retention priorities: top WAR players whose current NIL is below recommended
        retention_priorities = [
            a["player"] for a in allocations[:10]
            if a["recommended_nil"] > a["current_nil"] * 1.1
        ][:5]

        # Efficiency score: how well current spend aligns with WAR
        current_total = sum(a["current_nil"] for a in allocations)
        if current_total > 0 and budget > 0:
            # Score based on how close current spend matches optimal
            efficiency = 1.0 - abs(total_allocated - current_total) / max(budget, current_total)
            efficiency = max(0.3, min(1.0, efficiency))
        else:
            efficiency = 0.5

        # Expected wins from CFBD data if available
        try:
            cfbd = get_team_cfbd_profile(body.school)
            current_wins = cfbd.get("wins", 7) if cfbd else 7
        except Exception:
            current_wins = 7
        expected_wins = current_wins + (total_war / len(allocations) * 0.5) if allocations else current_wins

        response_data = RosterOptimizeResponse(
            school=body.school,
            total_budget=body.total_budget,
            total_allocated=round(total_allocated),
            budget_remaining=round(budget - total_allocated),
            expected_wins=round(min(expected_wins, body.win_target or 15), 1),
            optimization_status="optimized",
            allocations=allocations[:20],  # Top 20 allocations
            position_breakdown=position_breakdown,
            retention_priorities=retention_priorities,
            efficiency_score=round(efficiency, 3),
        )

        return APIResponse(status="success", data=response_data.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Roster optimization fallback error: {e}")
        raise HTTPException(status_code=500, detail=f"Roster optimization failed: {str(e)}")


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

    # Data-driven scenario analysis using WAR and CFBD data
    try:
        # Get current wins from CFBD
        try:
            cfbd = get_team_cfbd_profile(body.school)
            current_wins = cfbd.get("wins", 7) if cfbd else 7
        except Exception:
            current_wins = 7

        win_delta = 0
        total_cost = 0
        position_impacts = {}
        risks = []

        for change in body.changes:
            pos = change.position.upper()
            stars_est = max(1, min(5, int(change.overall_rating * 5)))
            nil_cost = change.nil_cost or 0
            war = calculate_player_war(pos, stars_est, nil_cost, body.school)

            if change.action == "remove":
                impact = -war
                risks.append(f"Losing {change.name} ({pos}) removes {war:.1f} WAR")
            else:
                impact = war

            win_delta += impact
            total_cost += nil_cost
            position_impacts[pos] = round(position_impacts.get(pos, 0) + impact, 2)

        new_wins = current_wins + win_delta

        # Build recommendation
        if win_delta >= 2:
            recommendation = f"Strong roster improvement: +{win_delta:.1f} wins projected. Proceed with acquisitions."
        elif win_delta >= 0.5:
            recommendation = f"Moderate improvement: +{win_delta:.1f} wins. Good value if within budget."
        elif win_delta >= 0:
            recommendation = "Marginal improvement. Consider whether cost justifies impact."
        elif win_delta >= -1:
            recommendation = "Slight negative impact. Ensure departures are offset by incoming talent."
        else:
            recommendation = f"Significant roster downgrade: {win_delta:.1f} wins. Prioritize replacements."

        if not risks:
            risks = ["No significant depth concerns identified"]

        response_data = ScenarioResponse(
            school=body.school,
            changes_analyzed=len(body.changes),
            current_projected_wins=round(current_wins, 1),
            new_projected_wins=round(max(0, new_wins), 1),
            win_delta=round(win_delta, 2),
            total_nil_cost=total_cost,
            cost_per_win=round(total_cost / win_delta) if win_delta > 0 else None,
            position_impacts=position_impacts,
            recommendation=recommendation,
            risks=risks,
        )

        return APIResponse(status="success", data=response_data.model_dump())

    except Exception as e:
        logger.error(f"Scenario analysis fallback error: {e}")
        raise HTTPException(status_code=500, detail=f"Scenario analysis failed: {str(e)}")


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

    # Data-driven roster report from real data
    try:
        school_norm = normalize_school_name(school).lower()

        # Get school tier
        try:
            from ..models.school_tiers import get_school_tier
            tier_name, tier_info = get_school_tier(school)
            school_tier = tier_name
        except Exception:
            school_tier = _get_school_tier(school)

        # Get roster composition
        roster_comp = get_team_roster_composition(school)

        # Get NIL data for this school
        nil_df = get_nil_players(limit=50000)
        name_col = "name" if "name" in nil_df.columns else "player_name"
        school_col_name = "school" if "school" in nil_df.columns else "team"
        val_col = "nil_value" if "nil_value" in nil_df.columns else "predicted_nil"

        school_players = nil_df[nil_df[school_col_name].apply(
            lambda x: normalize_school_name(str(x)).lower()) == school_norm]

        roster_size = len(school_players) if not school_players.empty else sum(roster_comp.values()) if roster_comp else 0
        total_nil = float(school_players[val_col].sum()) if not school_players.empty and val_col in school_players.columns else 0
        avg_nil = float(school_players[val_col].mean()) if not school_players.empty and val_col in school_players.columns else 0

        # Top players by NIL
        top_players = []
        if not school_players.empty:
            top_df = school_players.nlargest(5, val_col) if val_col in school_players.columns else school_players.head(5)
            for _, row in top_df.iterrows():
                top_players.append({
                    "name": str(row.get(name_col, "")),
                    "position": str(row.get("position", "")),
                    "nil_value": float(row.get(val_col, 0)) if pd.notna(row.get(val_col)) else 0,
                })

        # PFF summary
        pff_summary = get_team_pff_summary(school)

        # CFBD profile
        try:
            cfbd = get_team_cfbd_profile(school)
        except Exception:
            cfbd = {}

        # Portal activity
        portal_df = get_portal_players()
        origin_col = "origin_school" if "origin_school" in portal_df.columns else "from_school"
        dest_col = "destination_school" if "destination_school" in portal_df.columns else "to_school"
        outgoing = portal_df[portal_df[origin_col].apply(
            lambda x: normalize_school_name(str(x)).lower()) == school_norm] if not portal_df.empty else pd.DataFrame()
        incoming = portal_df[
            (portal_df[dest_col].apply(lambda x: normalize_school_name(str(x)).lower()) == school_norm) &
            (portal_df["status"].fillna("").str.lower() == "committed")
        ] if not portal_df.empty and "status" in portal_df.columns else pd.DataFrame()

        # Roster needs
        needs = get_roster_needs(school, portal_df)

        # Build executive summary
        exec_summary = [
            f"{school} ({tier_info.get('label', school_tier)}) — {roster_size} roster players",
            f"Total NIL investment: ${total_nil:,.0f} (avg ${avg_nil:,.0f}/player)",
        ]
        if cfbd:
            exec_summary.append(f"Record: {cfbd.get('wins', '?')}-{cfbd.get('losses', '?')} | SP+: {cfbd.get('sp_overall', 'N/A')}")
        if pff_summary:
            exec_summary.append(f"Team PFF avg: {pff_summary.get('avg_overall', 'N/A'):.1f}")

        # Roster summary section
        roster_summary = {
            "total_players": roster_size,
            "position_counts": roster_comp or {},
            "top_players": top_players,
        }

        # NIL optimization section
        nil_optimization = {
            "total_nil_spend": round(total_nil),
            "average_nil": round(avg_nil),
            "top_5_spend": sum(p["nil_value"] for p in top_players),
            "concentration": round(sum(p["nil_value"] for p in top_players[:3]) / max(total_nil, 1) * 100, 1) if top_players else 0,
        }

        # Portal shopping section
        priority_positions = needs.get("priority_positions", [])
        portal_shopping = {
            "positions_of_need": priority_positions,
            "incoming_count": len(incoming),
            "outgoing_count": len(outgoing),
            "net_transfers": len(incoming) - len(outgoing),
        }

        # Flight risk section (simplified)
        flight_risk_section = {
            "estimated_at_risk": len(outgoing),
            "position_vulnerability": {pos: departed_positions.count(pos) for pos in set(departed_positions)} if "departed_positions" in dir() else {},
        }

        # Win projection
        wins = cfbd.get("wins", 7) if cfbd else 7
        win_projection = {
            "current_wins": wins,
            "sp_plus_rating": cfbd.get("sp_overall") if cfbd else None,
            "talent_composite": cfbd.get("talent") if cfbd else None,
        }

        # Gap analysis
        gap_analysis = {
            "needs": needs.get("needs", {}),
            "priority_positions": priority_positions,
            "ideal_vs_actual": {
                pos: {"ideal": IDEAL_ROSTER.get(pos, 0), "actual": roster_comp.get(pos, 0)}
                for pos in IDEAL_ROSTER
            } if roster_comp else {},
        }

        response_data = RosterReportResponse(
            school=school,
            school_tier=school_tier,
            generated_at=datetime.utcnow(),
            executive_summary=exec_summary,
            roster_summary=roster_summary,
            nil_optimization=nil_optimization,
            portal_shopping=portal_shopping,
            flight_risk=flight_risk_section,
            win_projection=win_projection,
            gap_analysis=gap_analysis,
            output_files={},
        )

        return APIResponse(status="success", data=response_data.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Roster report fallback error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Roster report failed: {str(e)}")


# =============================================================================
# Helper Functions
# =============================================================================


def _get_school_multiplier(school: str) -> float:
    """Get NIL multiplier based on school brand from real CFBD data."""
    from ..models.school_tiers import get_school_multiplier as dynamic_multiplier
    return dynamic_multiplier(school.strip())


def _get_school_tier(school: str) -> str:
    """Get school tier classification from real CFBD data."""
    from ..models.school_tiers import get_school_tier as dynamic_tier
    tier_name, _ = dynamic_tier(school.strip())
    return tier_name


def _get_nil_tier(value: float) -> str:
    """Get NIL tier from value. Must match calibrated_valuator.py thresholds."""
    if value >= 2000000:
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

## Duplicate search endpoint removed - consolidated into the primary /players/search above


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

        # Get detailed PFF stats from the PFF grades file
        pff_stats = get_player_pff_stats(player_name, season)

        # Build comprehensive stats response
        stats = {
            "name": str(player_row.get("name", player_name)),
            "position": str(player_row.get("position", "")),
            "school": str(player_row.get("school", player_row.get("origin_school", ""))),
            "headshot_url": str(player_row.get("headshot_url")) if pd.notna(player_row.get("headshot_url")) else None,
            "season": season,
            "nil_value": float(player_row.get("nil_value", 0)) if pd.notna(player_row.get("nil_value")) else None,
            "nil_tier": str(player_row.get("tier", _get_nil_tier(float(player_row.get("nil_value", 0)) if pd.notna(player_row.get("nil_value")) else 0))),
            "stars": int(player_row.get("stars", 0)) if pd.notna(player_row.get("stars")) else None,
            "height": float(player_row.get("height", 0)) if pd.notna(player_row.get("height")) else None,
            "weight": float(player_row.get("weight", 0)) if pd.notna(player_row.get("weight")) else None,
        }

        # Use PFF stats if available, otherwise try from player_row
        if pff_stats:
            stats["pff"] = {
                "overall": pff_stats.get("pff_overall"),
                "offense": pff_stats.get("pff_offense"),
                "defense": pff_stats.get("pff_defense"),
                "passing": pff_stats.get("pff_passing"),
                "rushing": pff_stats.get("pff_rushing"),
                "receiving": pff_stats.get("pff_receiving"),
                "pass_block": pff_stats.get("pff_pass_block"),
                "run_block": pff_stats.get("pff_run_block"),
                "pass_rush": pff_stats.get("pff_pass_rush"),
                "run_defense": pff_stats.get("pff_run_defense"),
                "tackling": pff_stats.get("pff_tackling"),
                "coverage": pff_stats.get("pff_coverage"),
            }
        else:
            stats["pff"] = {
                "overall": float(player_row.get("pff_overall", 0)) if pd.notna(player_row.get("pff_overall")) else None,
                "offense": float(player_row.get("pff_offense", 0)) if pd.notna(player_row.get("pff_offense")) else None,
                "defense": float(player_row.get("pff_defense", 0)) if pd.notna(player_row.get("pff_defense")) else None,
                "passing": None,
                "rushing": None,
                "receiving": None,
                "pass_block": None,
                "run_block": None,
                "pass_rush": None,
                "run_defense": None,
                "tackling": None,
                "coverage": None,
            }

        # Add position-specific stats from PFF data
        position = str(player_row.get("position", "")).upper()

        if position == "QB" and pff_stats:
            stats["passing"] = {
                "passer_rating": pff_stats.get("passer_rating"),
                "completion_pct": pff_stats.get("completion_pct"),
                "big_time_throws": pff_stats.get("big_time_throws"),
                "big_time_throw_pct": pff_stats.get("big_time_throw_pct"),
                "turnover_worthy_plays": pff_stats.get("turnover_worthy_plays"),
                "pressure_completion_pct": pff_stats.get("pressure_completion_pct"),
                "pressure_qb_rating": pff_stats.get("pressure_qb_rating"),
                "yards": pff_stats.get("yards"),
                "touchdowns": pff_stats.get("touchdowns"),
            }

        if position in ("RB", "QB") and pff_stats:
            stats["rushing"] = {
                "elusive_rating": pff_stats.get("elusive_rating"),
                "yards_after_contact": pff_stats.get("yards_after_contact"),
                "yaco_per_attempt": pff_stats.get("yaco_per_attempt"),
                "breakaway_pct": pff_stats.get("breakaway_pct"),
                "missed_tackles_forced": pff_stats.get("missed_tackles_forced"),
                "yards": pff_stats.get("yards"),
                "touchdowns": pff_stats.get("touchdowns"),
                "yards_per_carry": pff_stats.get("yards_per_carry"),
            }

        if position in ("WR", "TE", "RB") and pff_stats:
            stats["receiving"] = {
                "yards_per_route_run": pff_stats.get("yards_per_route_run"),
                "drop_rate": pff_stats.get("drop_rate"),
                "contested_catch_rate": pff_stats.get("contested_catch_rate"),
                "yards_after_catch": pff_stats.get("yards_after_catch"),
                "targets": pff_stats.get("targets"),
                "receptions": pff_stats.get("receptions"),
                "yards": pff_stats.get("rec_yards"),
                "touchdowns": pff_stats.get("touchdowns"),
            }

        if position in ("EDGE", "DT", "DL", "DE") and pff_stats:
            stats["pass_rush"] = {
                "pass_rushing_productivity": pff_stats.get("pass_rushing_productivity"),
                "pass_rush_win_rate": pff_stats.get("pass_rush_win_rate"),
                "pressures": pff_stats.get("pressures"),
                "sacks": pff_stats.get("sacks"),
                "hurries": pff_stats.get("hurries"),
                "hits": pff_stats.get("hits"),
            }

        if position in ("CB", "S", "LB") and pff_stats:
            stats["coverage"] = {
                "passer_rating_allowed": pff_stats.get("passer_rating_allowed"),
                "yards_per_coverage_snap": pff_stats.get("yards_per_coverage_snap"),
                "forced_incompletes": pff_stats.get("forced_incompletes"),
                "interceptions": pff_stats.get("interceptions"),
                "pass_breakups": pff_stats.get("pass_breakups"),
                "missed_tackle_rate": pff_stats.get("missed_tackle_rate"),
            }

        if position in ("OT", "OG", "C", "OL", "IOL") and pff_stats:
            stats["blocking"] = {
                "pass_blocking_efficiency": pff_stats.get("pass_blocking_efficiency"),
                "pressures_allowed": pff_stats.get("pressures_allowed"),
                "sacks_allowed": pff_stats.get("sacks_allowed"),
                "run_block_percent": pff_stats.get("run_block_percent"),
            }

        # Add dual valuation (On3 + Portal IQ)
        dual_val = get_player_dual_valuation(player_name)
        if dual_val:
            stats["valuation"] = {
                "on3_value": dual_val["on3_value"],
                "portal_iq_value": dual_val["portal_iq_value"],
                "portal_iq_tier": dual_val["portal_iq_tier"],
                "confidence": dual_val["confidence"],
                "has_on3_data": dual_val["has_on3_data"],
                "breakdown": dual_val["value_breakdown"],
                "reasoning": dual_val["reasoning"],
            }
        else:
            # Fallback to existing NIL value if dual valuation not available
            stats["valuation"] = {
                "on3_value": None,
                "portal_iq_value": stats.get("nil_value"),
                "portal_iq_tier": stats.get("nil_tier"),
                "confidence": "low",
                "has_on3_data": False,
                "breakdown": None,
                "reasoning": ["Limited data available for valuation"],
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

        from ..models.player_similarity import get_player_comparisons as get_comps

        result = get_comps(
            player_name=player_name,
            include_nfl=include_nfl,
            include_college=include_college,
            limit=limit,
        )

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return APIResponse(
            status="success",
            data=result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Player comparison error: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Return empty comparisons instead of error
        return APIResponse(
            status="success",
            data={
                "player": player_name,
                "position": "Unknown",
                "nfl_comparisons": [],
                "college_comparisons": [],
                "message": "Comparison data not yet available for this player",
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

        from ..utils.historical_loader import get_player_career_stats

        career = get_player_career_stats(player_name)

        return APIResponse(
            status="success",
            data=career,
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
                "message": "Career data not yet available for this player",
            }
        )


@router.get(
    "/players/{player_name}/elite-profile",
    response_model=APIResponse,
    tags=["Players"],
    summary="Get elite athlete profile",
    description="Get elite trait analysis based on combine measurables.",
)
async def get_player_elite_profile_endpoint(
    request: Request,
    player_name: str,
    api_key: str = Depends(require_api_key),
):
    """Get elite athlete profile with bonus calculations."""
    try:
        from urllib.parse import unquote
        player_name = unquote(player_name)

        from ..models.elite_traits import get_player_elite_profile as get_elite

        profile = get_elite(player_name)

        return APIResponse(
            status="success",
            data=profile,
        )

    except Exception as e:
        logger.error(f"Elite profile error: {e}")
        return APIResponse(
            status="success",
            data={
                "player": player_name,
                "elite_traits": [],
                "elite_bonus": 1.0,
                "measurables": {},
                "message": "Elite profile data not available for this player",
            }
        )


@router.get(
    "/reference/historical-data",
    response_model=APIResponse,
    tags=["Reference"],
    summary="List available historical data",
    description="Check what NFL, college, and combine data is available in storage.",
)
async def list_historical_data_endpoint(
    request: Request,
    api_key: str = Depends(require_api_key),
):
    """List available historical data for comparisons."""
    try:
        from ..utils.historical_loader import list_available_historical_data

        available = list_available_historical_data()

        return APIResponse(
            status="success",
            data=available,
        )

    except Exception as e:
        logger.error(f"Historical data list error: {e}")
        return APIResponse(
            status="success",
            data={
                "nfl_seasons": [],
                "fbs_seasons": [],
                "fcs_seasons": [],
                "combine_years": [],
                "message": "Could not enumerate historical data",
            }
        )
