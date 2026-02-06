"""FastAPI application for Portal IQ API.

Provides ML-powered college football analytics including:
- NIL valuation predictions
- Transfer portal intelligence
- Draft projections
- Roster optimization

Designed for easy integration with PlaymakerVC and other clients.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse

from .routes import router
from .schemas import APIResponse, ErrorResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("portal_iq_api")

# =============================================================================
# Configuration
# =============================================================================

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# API Key configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_api_keys_env = os.getenv("PORTAL_IQ_API_KEYS", "dev-key-123,playmaker-api-key")
logger.info(f"Loading API keys from env (length={len(_api_keys_env)}): {_api_keys_env[:30]}...")
VALID_API_KEYS = set(
    key.strip()
    for key in _api_keys_env.split(",")
    if key.strip()
)
logger.info(f"Loaded {len(VALID_API_KEYS)} valid API keys: {[k[:8] + '...' for k in VALID_API_KEYS]}")

# =============================================================================
# Model Storage
# =============================================================================

# Models loaded on startup
models = {
    "nil_valuator": None,
    "portal_predictor": None,
    "draft_projector": None,
    "win_model": None,
    "roster_optimizer": None,
}


def load_models():
    """Load all ML models on startup."""
    logger.info("=" * 60)
    logger.info("Loading Portal IQ ML Models...")
    logger.info("=" * 60)

    try:
        from ..models import (
            NILValuator,
            PortalPredictor,
            DraftProjector,
            WinImpactModel,
            RosterOptimizer,
        )

        # Initialize models
        models["nil_valuator"] = NILValuator()
        logger.info("  [OK] NILValuator loaded")

        models["portal_predictor"] = PortalPredictor()
        logger.info("  [OK] PortalPredictor loaded")

        models["draft_projector"] = DraftProjector()
        logger.info("  [OK] DraftProjector loaded")

        models["win_model"] = WinImpactModel()
        logger.info("  [OK] WinImpactModel loaded")

        # RosterOptimizer integrates other models
        models["roster_optimizer"] = RosterOptimizer(
            nil_valuator=models["nil_valuator"],
            portal_predictor=models["portal_predictor"],
            draft_projector=models["draft_projector"],
            win_model=models["win_model"],
        )
        logger.info("  [OK] RosterOptimizer loaded")

        logger.info("=" * 60)
        logger.info("All models loaded successfully!")
        logger.info("=" * 60)

    except ImportError as e:
        logger.warning(f"Could not import models: {e}")
        logger.info("Running in DEMO MODE without ML models")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("Running in DEMO MODE without ML models")


def get_models():
    """Get loaded models dict for use in routes."""
    return models


# =============================================================================
# API Key Authentication
# =============================================================================

async def verify_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """Verify API key from X-API-Key header.

    Raises:
        HTTPException: 401 if key missing or invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Missing API key",
                "message": "Include X-API-Key header with your request",
            },
        )
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid API key",
                "message": "The provided API key is not valid",
            },
        )
    return api_key


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifespan - load models on startup."""
    # Startup
    logger.info("Starting Portal IQ API...")
    load_models()

    # Store helpers in app state for routes
    app.state.get_models = get_models
    app.state.verify_api_key = verify_api_key

    logger.info("Portal IQ API ready to serve requests")
    yield

    # Shutdown
    logger.info("Shutting down Portal IQ API...")


# =============================================================================
# Create FastAPI Application
# =============================================================================

app = FastAPI(
    title="Portal IQ API",
    version="1.0.0",
    description="""
## Portal IQ - College Football Intelligence Platform

AI-powered analytics API for NIL valuation, transfer portal intelligence,
draft projections, and roster optimization.

### Products
- **NIL Valuation**: Predict player NIL values with detailed breakdowns
- **Portal Intelligence**: Flight risk analysis and transfer fit scoring
- **Draft Projection**: NFL draft projections with earnings estimates
- **Roster Optimization**: Budget allocation and scenario analysis

### Authentication
All endpoints (except health check) require an API key via the `X-API-Key` header.

```
X-API-Key: your-api-key-here
```

### Response Format
All responses follow a consistent JSON format:
```json
{
    "status": "success",
    "data": { ... },
    "message": null,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Handling
Errors return appropriate HTTP status codes with details:
```json
{
    "status": "error",
    "message": "Error description",
    "detail": "Additional context"
}
```

### Integration
Designed for easy integration with PlaymakerVC and other sports analytics platforms.
Simply make HTTP calls with JSON payloads.

---
*Powered by Elite Sports Solutions*
    """,
    contact={
        "name": "Elite Sports Solutions",
        "email": "api@elitesportssolutions.com",
        "url": "https://elitesportssolutions.com",
    },
    license_info={
        "name": "Proprietary",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# =============================================================================
# CORS Middleware
# =============================================================================

if IS_PRODUCTION:
    # Restrict origins in production
    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "https://playmakervc.com,https://app.playmakervc.com,https://portaliq.app"
        ).split(",")
    ]
    logger.info(f"CORS: Production mode - allowed origins: {allowed_origins}")
else:
    # Allow all origins in development
    allowed_origins = ["*"]
    logger.info("CORS: Development mode - allowing all origins")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Response-Time", "X-Request-ID"],
)


# =============================================================================
# Request Logging & Response Time Middleware
# =============================================================================

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add response time header and log all requests."""
    import uuid

    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Log incoming request
    logger.info(
        f"[{request_id}] --> {request.method} {request.url.path} "
        f"(client: {request.client.host if request.client else 'unknown'})"
    )

    # Process request
    response = await call_next(request)

    # Calculate response time
    process_time = time.time() - start_time
    process_time_ms = round(process_time * 1000, 2)

    # Add headers
    response.headers["X-Response-Time"] = f"{process_time_ms}ms"
    response.headers["X-Request-ID"] = request_id

    # Log response
    logger.info(
        f"[{request_id}] <-- {response.status_code} "
        f"({process_time_ms}ms)"
    )

    return response


# =============================================================================
# Health Check Endpoints (No Auth Required)
# =============================================================================

@app.get("/debug/keys", tags=["Debug"])
async def debug_keys():
    """Debug endpoint to show loaded keys (DEV ONLY)."""
    return {
        "keys_count": len(VALID_API_KEYS),
        "keys_preview": [k[:10] + "..." for k in VALID_API_KEYS],
    }


@app.get(
    "/",
    response_model=APIResponse,
    tags=["Health"],
    summary="Health check",
    description="Returns API status. No authentication required.",
)
async def health_check():
    """Health check endpoint - no authentication required."""
    models_status = {
        name: "loaded" if model is not None else "not_loaded"
        for name, model in models.items()
    }

    all_loaded = all(m is not None for m in models.values())

    return APIResponse(
        status="success",
        data={
            "service": "Portal IQ API",
            "version": "1.0.0",
            "environment": ENVIRONMENT,
            "status": "operational" if all_loaded else "degraded",
            "models": models_status,
        },
        message="API is healthy" if all_loaded else "API running in demo mode",
    )


@app.get(
    "/health",
    response_model=APIResponse,
    tags=["Health"],
    summary="Detailed health check",
    description="Returns detailed API and model status.",
)
async def detailed_health():
    """Detailed health check with model information."""
    models_loaded = sum(1 for m in models.values() if m is not None)
    total_models = len(models)

    model_details = {}
    for name, model in models.items():
        if model is not None:
            model_details[name] = {
                "loaded": True,
                "type": type(model).__name__,
                "ready": True,
            }
        else:
            model_details[name] = {
                "loaded": False,
                "type": None,
                "ready": False,
            }

    return APIResponse(
        status="success",
        data={
            "service": "Portal IQ API",
            "version": "1.0.0",
            "environment": ENVIRONMENT,
            "models_loaded": f"{models_loaded}/{total_models}",
            "models": model_details,
            "endpoints": {
                "nil": "/api/nil/*",
                "portal": "/api/portal/*",
                "draft": "/api/draft/*",
                "roster": "/api/roster/*",
            },
        },
    )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="error",
            message=str(exc.detail) if isinstance(exc.detail, str) else exc.detail.get("message", "Error"),
            detail=exc.detail.get("error") if isinstance(exc.detail, dict) else None,
        ).model_dump(mode="json"),
        headers={"X-Error": "true"},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            status="error",
            message="Invalid request data",
            detail=str(exc),
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error on {request.url.path}: {exc}", exc_info=True)

    # Don't expose internal errors in production
    detail = str(exc) if not IS_PRODUCTION else None

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            status="error",
            message="Internal server error",
            detail=detail,
        ).model_dump(mode="json"),
        headers={"X-Error": "true"},
    )


# =============================================================================
# Include API Routes
# =============================================================================

app.include_router(router, prefix="/api")


# =============================================================================
# WebSocket Routes
# =============================================================================

from fastapi import WebSocket, WebSocketDisconnect
from .websocket import manager, Channels

@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    """
    WebSocket endpoint for real-time updates.

    Channels:
    - nil: NIL valuation updates
    - portal: Transfer portal updates
    - draft: Draft projection updates
    - roster: Roster changes
    - all: All updates
    """
    # Validate channel
    valid_channels = [Channels.NIL, Channels.PORTAL, Channels.DRAFT, Channels.ROSTER, Channels.ALL]
    if channel not in valid_channels:
        await websocket.close(code=4000)
        return

    await manager.connect(websocket, channel)
    try:
        # Send welcome message
        await manager.send_to_client(websocket, {
            "type": "connected",
            "channel": channel,
            "message": f"Connected to {channel} channel"
        })

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_json()
            # Handle ping/pong for keepalive
            if data.get("type") == "ping":
                await manager.send_to_client(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, channel)


@app.get("/ws/status", tags=["WebSocket"])
async def websocket_status():
    """Get WebSocket connection statistics."""
    return {
        "total_connections": manager.get_total_connections(),
        "channels": {
            channel: manager.get_channel_count(channel)
            for channel in [Channels.NIL, Channels.PORTAL, Channels.DRAFT, Channels.ROSTER, Channels.ALL]
        }
    }


# =============================================================================
# Custom OpenAPI Schema
# =============================================================================

def custom_openapi():
    """Customize OpenAPI schema with security definitions."""
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        contact=app.contact,
        license_info=app.license_info,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication. Contact Elite Sports Solutions for access.",
        }
    }

    # Apply security to all /api/* paths
    for path, methods in openapi_schema["paths"].items():
        if path.startswith("/api"):
            for method in methods.values():
                if isinstance(method, dict) and "security" not in method:
                    method["security"] = [{"ApiKeyAuth": []}]

    # Add servers
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development"},
        {"url": "https://api.portaliq.app", "description": "Production"},
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# =============================================================================
# Run with uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Railway sets PORT env var - use it if available
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=not IS_PRODUCTION,
        log_level="info",
    )
