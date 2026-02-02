"""
PocketBase Authentication Middleware

Validates JWT tokens from PocketBase for authenticated requests.
Allows public endpoints while protecting authenticated ones.
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

import jwt
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}

# Endpoints that only require valid token (any tier)
AUTHENTICATED_ENDPOINTS = {
    "/portal-iq/",
    "/cap-iq/",
    "/shared/",
}

# Endpoints requiring Pro tier or higher
PRO_ENDPOINTS = {
    "/portal-iq/roster/optimize",
    "/portal-iq/nil/bulk-valuate",
    "/cap-iq/contract/predict",
    "/cap-iq/cap/optimize",
}

# Endpoints requiring Enterprise tier
ENTERPRISE_ENDPOINTS = {
    "/portal-iq/api/export",
    "/cap-iq/api/export",
}


class PocketBaseAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate PocketBase JWT tokens.

    Extracts user info from token and attaches to request state.
    """

    def __init__(self, app):
        super().__init__(app)
        self.pb_jwt_secret = os.getenv("POCKETBASE_JWT_SECRET", "")
        self.pb_url = os.getenv("POCKETBASE_URL", "http://localhost:8090")

        if not self.pb_jwt_secret:
            logger.warning(
                "POCKETBASE_JWT_SECRET not set. "
                "Authentication will be disabled in development mode."
            )

    async def dispatch(self, request: Request, call_next):
        """Process request and validate authentication."""

        path = request.url.path

        # Allow public endpoints
        if self._is_public_endpoint(path):
            return await call_next(request)

        # Extract token from header
        auth_header = request.headers.get("Authorization", "")
        token = self._extract_token(auth_header)

        # Development mode: allow requests without token if no secret configured
        if not self.pb_jwt_secret:
            request.state.user = self._get_dev_user()
            return await call_next(request)

        # Validate token
        if not token:
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication required", "detail": "Missing authorization token"}
            )

        try:
            user = self._validate_token(token)
            request.state.user = user

            # Check tier requirements
            if not self._check_tier_access(path, user):
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Insufficient permissions",
                        "detail": f"This endpoint requires a higher subscription tier",
                        "required_tier": self._get_required_tier(path),
                        "current_tier": user.get("subscription_tier", "free"),
                    }
                )

        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"error": "Token expired", "detail": "Please log in again"}
            )
        except jwt.InvalidTokenError as e:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid token", "detail": str(e)}
            )
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication failed", "detail": str(e)}
            )

        return await call_next(request)

    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public."""
        for public_path in PUBLIC_ENDPOINTS:
            if path == public_path or path.startswith(public_path.rstrip("/") + "/"):
                return True
        return False

    def _extract_token(self, auth_header: str) -> Optional[str]:
        """Extract JWT token from Authorization header."""
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]

        return None

    def _validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate PocketBase JWT token.

        Returns:
            User data from token
        """
        # PocketBase uses HS256 by default
        payload = jwt.decode(
            token,
            self.pb_jwt_secret,
            algorithms=["HS256"],
            options={"verify_exp": True}
        )

        # PocketBase token structure
        return {
            "id": payload.get("id"),
            "email": payload.get("email"),
            "name": payload.get("name", ""),
            "organization": payload.get("organization", ""),
            "organization_type": payload.get("organization_type", "independent"),
            "subscription_tier": payload.get("subscription_tier", "free"),
            "exp": payload.get("exp"),
        }

    def _get_dev_user(self) -> Dict[str, Any]:
        """Return a development user for testing without auth."""
        return {
            "id": "dev_user",
            "email": "dev@localhost",
            "name": "Development User",
            "organization": "Development",
            "organization_type": "independent",
            "subscription_tier": "enterprise",  # Full access in dev
            "exp": None,
        }

    def _check_tier_access(self, path: str, user: Dict[str, Any]) -> bool:
        """Check if user's tier allows access to endpoint."""
        tier = user.get("subscription_tier", "free")
        tier_levels = {"free": 0, "pro": 1, "enterprise": 2}
        user_level = tier_levels.get(tier, 0)

        # Check enterprise endpoints
        for endpoint in ENTERPRISE_ENDPOINTS:
            if path.startswith(endpoint):
                return user_level >= 2

        # Check pro endpoints
        for endpoint in PRO_ENDPOINTS:
            if path.startswith(endpoint):
                return user_level >= 1

        # All other authenticated endpoints are free tier
        return True

    def _get_required_tier(self, path: str) -> str:
        """Get required tier for an endpoint."""
        for endpoint in ENTERPRISE_ENDPOINTS:
            if path.startswith(endpoint):
                return "enterprise"

        for endpoint in PRO_ENDPOINTS:
            if path.startswith(endpoint):
                return "pro"

        return "free"


def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Dependency to get current user from request state.

    Usage in routes:
        @router.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user": user}
    """
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.state.user


def require_tier(required_tier: str):
    """
    Dependency factory to require a specific tier.

    Usage:
        @router.get("/pro-feature")
        async def pro_feature(user: dict = Depends(require_tier("pro"))):
            return {"data": "pro content"}
    """
    tier_levels = {"free": 0, "pro": 1, "enterprise": 2}
    required_level = tier_levels.get(required_tier, 0)

    def check_tier(request: Request) -> Dict[str, Any]:
        user = get_current_user(request)
        user_level = tier_levels.get(user.get("subscription_tier", "free"), 0)

        if user_level < required_level:
            raise HTTPException(
                status_code=403,
                detail=f"This feature requires {required_tier} tier or higher"
            )

        return user

    return check_tier
