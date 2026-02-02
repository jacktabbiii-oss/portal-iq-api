"""
ML Engine API Module

FastAPI application serving Portal IQ endpoints for:
- NIL valuation predictions
- Transfer portal intelligence
- Draft projections
- Roster optimization
"""

from .app import app, get_models, verify_api_key
from .routes import router

__all__ = [
    "app",
    "router",
    "get_models",
    "verify_api_key",
]
