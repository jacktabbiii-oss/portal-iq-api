"""API routes for Portal IQ.

All endpoints for NIL valuation, portal intelligence, draft projections,
and roster optimization.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

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
    return await request.app.state.verify_api_key()


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
            logger.error(f"NIL prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Demo mode fallback
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
        message="Demo mode - models not loaded",
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

    # Demo fallback
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
        message="Demo mode",
    )


# =============================================================================
# Portal Endpoints
# =============================================================================

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
    rating = player_dict.get("overall_rating", 0.75)
    rating_mult = 1.0 + (rating - 0.75) * 5

    # Social media bonus
    social = player_dict.get("social_media", {})
    followers = (
        social.get("instagram_followers", 0) +
        social.get("twitter_followers", 0) +
        social.get("tiktok_followers", 0)
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
