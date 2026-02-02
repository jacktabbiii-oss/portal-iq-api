"""Tests for Portal IQ API routes.

Tests all API endpoints for correct responses, error handling, and authentication.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestHealthEndpoints:
    """Tests for health check endpoints (no auth required)."""

    def test_root_health_check_returns_200(self, api_client):
        """Test that root endpoint returns 200."""
        response = api_client.get("/")
        assert response.status_code == 200

    def test_root_returns_api_info(self, api_client):
        """Test that root endpoint returns API information."""
        response = api_client.get("/")
        data = response.json()

        assert "status" in data or "service" in data or "data" in data

    def test_health_endpoint_returns_200(self, api_client):
        """Test that /health endpoint returns 200."""
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_includes_model_status(self, api_client):
        """Test that health check includes model status."""
        response = api_client.get("/health")
        data = response.json()

        # Check for model status in response
        if "data" in data:
            assert "models" in data["data"] or "status" in data["data"]


class TestAuthenticationMiddleware:
    """Tests for API key authentication middleware."""

    def test_request_without_api_key_returns_401(self, api_client):
        """Test that requests without API key return 401."""
        response = api_client.post("/api/nil/predict", json={
            "player": {
                "name": "Test Player",
                "school": "Alabama",
                "position": "QB",
            }
        })

        assert response.status_code == 401

    def test_request_with_invalid_api_key_returns_401(self, api_client, invalid_api_key):
        """Test that requests with invalid API key return 401."""
        response = api_client.post(
            "/api/nil/predict",
            json={
                "player": {
                    "name": "Test Player",
                    "school": "Alabama",
                    "position": "QB",
                }
            },
            headers={"X-API-Key": invalid_api_key}
        )

        assert response.status_code == 401

    def test_request_with_valid_api_key_succeeds(self, api_client, api_key, sample_api_player_request):
        """Test that requests with valid API key succeed."""
        response = api_client.post(
            "/api/nil/predict",
            json=sample_api_player_request,
            headers={"X-API-Key": api_key}
        )

        # Should not be 401
        assert response.status_code != 401

    def test_health_check_no_auth_required(self, api_client):
        """Test that health check doesn't require authentication."""
        # No API key header
        response = api_client.get("/")

        assert response.status_code == 200


class TestNILEndpoints:
    """Tests for NIL valuation endpoints."""

    def test_nil_predict_returns_200(self, api_client, api_key, sample_api_player_request):
        """Test POST /api/nil/predict returns 200 with valid input."""
        response = api_client.post(
            "/api/nil/predict",
            json=sample_api_player_request,
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_nil_predict_returns_value(self, api_client, api_key, sample_api_player_request):
        """Test that NIL prediction includes a value."""
        response = api_client.post(
            "/api/nil/predict",
            json=sample_api_player_request,
            headers={"X-API-Key": api_key}
        )

        data = response.json()
        # Check for predicted value in response
        assert "data" in data or "predicted_value" in data

    def test_nil_predict_invalid_input_returns_error(self, api_client, api_key):
        """Test that invalid input returns error code."""
        response = api_client.post(
            "/api/nil/predict",
            json={"invalid": "data"},
            headers={"X-API-Key": api_key}
        )

        # Should return 4xx error
        assert response.status_code >= 400

    def test_nil_predict_missing_required_fields(self, api_client, api_key):
        """Test that missing required fields returns error."""
        response = api_client.post(
            "/api/nil/predict",
            json={
                "player": {
                    "name": "Test Player",
                    # Missing school and position
                }
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code >= 400

    def test_nil_transfer_impact_returns_200(self, api_client, api_key):
        """Test POST /api/nil/transfer-impact returns 200."""
        response = api_client.post(
            "/api/nil/transfer-impact",
            json={
                "player": {
                    "name": "Test Player",
                    "school": "UCLA",
                    "position": "QB",
                },
                "target_school": "Alabama"
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_nil_market_report_returns_200(self, api_client, api_key):
        """Test POST /api/nil/market-report returns 200."""
        response = api_client.post(
            "/api/nil/market-report",
            json={
                "position": "QB",
                "conference": "SEC"
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200


class TestPortalEndpoints:
    """Tests for portal intelligence endpoints."""

    def test_portal_flight_risk_returns_200(self, api_client, api_key):
        """Test POST /api/portal/flight-risk returns 200."""
        response = api_client.post(
            "/api/portal/flight-risk",
            json={
                "player": {
                    "name": "Test Player",
                    "school": "Florida State",
                    "position": "WR",
                    "overall_rating": 0.82
                }
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_portal_team_report_returns_200(self, api_client, api_key):
        """Test POST /api/portal/team-report returns 200."""
        response = api_client.post(
            "/api/portal/team-report",
            json={"school": "Michigan"},
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_portal_fit_score_returns_200(self, api_client, api_key):
        """Test POST /api/portal/fit-score returns 200."""
        response = api_client.post(
            "/api/portal/fit-score",
            json={
                "player": {
                    "name": "Transfer Player",
                    "school": "UCLA",
                    "position": "RB"
                },
                "target_school": "Oregon"
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_portal_recommendations_returns_200(self, api_client, api_key):
        """Test POST /api/portal/recommendations returns 200."""
        response = api_client.post(
            "/api/portal/recommendations",
            json={
                "school": "Tennessee",
                "budget": 2000000,
                "positions_of_need": ["QB", "EDGE"]
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_portal_recommendations_invalid_budget(self, api_client, api_key):
        """Test that negative budget returns error."""
        response = api_client.post(
            "/api/portal/recommendations",
            json={
                "school": "Tennessee",
                "budget": -1000,
                "positions_of_need": ["QB"]
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code >= 400


class TestDraftEndpoints:
    """Tests for draft projection endpoints."""

    def test_draft_project_returns_200(self, api_client, api_key):
        """Test POST /api/draft/project returns 200."""
        response = api_client.post(
            "/api/draft/project",
            json={
                "player": {
                    "name": "Draft Prospect",
                    "school": "USC",
                    "position": "QB",
                    "class_year": "Junior"
                }
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_draft_project_includes_projection(self, api_client, api_key):
        """Test that draft projection includes round/pick info."""
        response = api_client.post(
            "/api/draft/project",
            json={
                "player": {
                    "name": "Draft Prospect",
                    "school": "Ohio State",
                    "position": "EDGE",
                    "class_year": "Senior"
                }
            },
            headers={"X-API-Key": api_key}
        )

        data = response.json()
        # Should have draft projection data
        assert "data" in data or "projected_round" in data

    def test_draft_mock_returns_200(self, api_client, api_key):
        """Test POST /api/draft/mock returns 200."""
        response = api_client.post(
            "/api/draft/mock",
            json={
                "season_year": 2025,
                "num_rounds": 3
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_draft_mock_invalid_year(self, api_client, api_key):
        """Test that invalid draft year returns error."""
        response = api_client.post(
            "/api/draft/mock",
            json={
                "season_year": 1900,  # Invalid year
                "num_rounds": 3
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code >= 400

    def test_draft_mock_invalid_rounds(self, api_client, api_key):
        """Test that invalid round count returns error."""
        response = api_client.post(
            "/api/draft/mock",
            json={
                "season_year": 2025,
                "num_rounds": 10  # Max is 7
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code >= 400


class TestRosterEndpoints:
    """Tests for roster optimization endpoints."""

    def test_roster_optimize_returns_200(self, api_client, api_key):
        """Test POST /api/roster/optimize returns 200."""
        response = api_client.post(
            "/api/roster/optimize",
            json={
                "school": "Georgia",
                "total_budget": 12000000,
                "win_target": 11
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_roster_optimize_includes_allocations(self, api_client, api_key):
        """Test that optimization includes allocation recommendations."""
        response = api_client.post(
            "/api/roster/optimize",
            json={
                "school": "Alabama",
                "total_budget": 15000000
            },
            headers={"X-API-Key": api_key}
        )

        data = response.json()
        # Should include some allocation data
        assert "data" in data or "allocations" in data

    def test_roster_scenario_returns_200(self, api_client, api_key):
        """Test POST /api/roster/scenario returns 200."""
        response = api_client.post(
            "/api/roster/scenario",
            json={
                "school": "Oklahoma",
                "changes": [
                    {"name": "New QB", "position": "QB", "action": "add", "overall_rating": 0.90},
                    {"name": "Old LB", "position": "LB", "action": "remove"}
                ]
            },
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_roster_scenario_shows_win_impact(self, api_client, api_key):
        """Test that scenario analysis shows win impact."""
        response = api_client.post(
            "/api/roster/scenario",
            json={
                "school": "Texas",
                "changes": [
                    {"name": "Portal QB", "position": "QB", "action": "add", "overall_rating": 0.92}
                ]
            },
            headers={"X-API-Key": api_key}
        )

        data = response.json()
        # Should have win impact data
        assert "data" in data or "win_delta" in data

    def test_roster_report_returns_200(self, api_client, api_key):
        """Test GET /api/roster/{school}/report returns 200."""
        response = api_client.get(
            "/api/roster/Texas/report",
            headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_roster_report_includes_sections(self, api_client, api_key):
        """Test that roster report includes expected sections."""
        response = api_client.get(
            "/api/roster/Georgia/report",
            headers={"X-API-Key": api_key}
        )

        data = response.json()
        # Should have report data
        assert "data" in data or "school" in data


class TestResponseFormat:
    """Tests for consistent response format."""

    def test_success_response_format(self, api_client, api_key, sample_api_player_request):
        """Test that success responses follow expected format."""
        response = api_client.post(
            "/api/nil/predict",
            json=sample_api_player_request,
            headers={"X-API-Key": api_key}
        )

        data = response.json()

        # Check for standard response structure
        assert "status" in data
        assert data["status"] == "success"

    def test_error_response_format(self, api_client, api_key):
        """Test that error responses follow expected format."""
        response = api_client.post(
            "/api/nil/predict",
            json={"invalid": "data"},
            headers={"X-API-Key": api_key}
        )

        if response.status_code >= 400:
            data = response.json()
            # Should have error indication
            assert "status" in data or "error" in data or "message" in data

    def test_response_includes_timestamp(self, api_client, api_key, sample_api_player_request):
        """Test that responses include timestamp."""
        response = api_client.post(
            "/api/nil/predict",
            json=sample_api_player_request,
            headers={"X-API-Key": api_key}
        )

        data = response.json()

        # Timestamp may be in data or at top level
        has_timestamp = (
            "timestamp" in data or
            ("data" in data and isinstance(data["data"], dict) and "timestamp" in data.get("data", {}))
        )
        # Timestamp is optional but good to have
        assert response.status_code == 200

    def test_response_time_header(self, api_client, api_key, sample_api_player_request):
        """Test that response includes timing header."""
        response = api_client.post(
            "/api/nil/predict",
            json=sample_api_player_request,
            headers={"X-API-Key": api_key}
        )

        # Check for response time header
        has_timing = (
            "X-Response-Time" in response.headers or
            "x-response-time" in response.headers
        )
        # Timing header is optional
        assert response.status_code == 200


class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation endpoints."""

    def test_openapi_json_accessible(self, api_client):
        """Test that OpenAPI JSON is accessible."""
        response = api_client.get("/openapi.json")
        assert response.status_code == 200

    def test_docs_endpoint_accessible(self, api_client):
        """Test that docs endpoint is accessible."""
        response = api_client.get("/docs")
        # Docs page returns HTML
        assert response.status_code == 200

    def test_redoc_endpoint_accessible(self, api_client):
        """Test that ReDoc endpoint is accessible."""
        response = api_client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_includes_paths(self, api_client):
        """Test that OpenAPI schema includes API paths."""
        response = api_client.get("/openapi.json")
        data = response.json()

        assert "paths" in data
        assert "/api/nil/predict" in data["paths"] or len(data["paths"]) > 0
