"""
Tests for FastAPI Application

Tests main app endpoints and configuration.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_includes_timestamp(self, client):
        """Health endpoint includes timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_welcome(self, client):
        """Root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "ML Engine" in data["message"]


@pytest.fixture
def client():
    """Create test client."""
    try:
        from src.api.app import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI app not available")
