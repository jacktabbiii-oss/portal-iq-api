"""
Portal IQ API - Main Application
FastAPI application with comprehensive college football analytics endpoints
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

from .routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Portal IQ API",
    description="Comprehensive college football analytics API with NIL valuations, transfer portal intelligence, and team rankings",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
CORS_ALLOWED_ORIGINS = [
    "https://portal-iq-web.vercel.app",
    "https://www.portaliq.io",
    "http://localhost:3000",
    "https://*.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return JSONResponse({
        "status": "success",
        "data": {
            "service": "Portal IQ API",
            "version": "2.0.0",
            "environment": "production",
            "status": "operational",
            "features": [
                "NIL Valuations (21K+ players)",
                "Transfer Portal Intelligence (11K+ entries)",
                "Team Rankings (136 FBS schools)",
                "Win Impact Analysis",
                "Draft Projections",
                "Player Comparisons"
            ]
        },
        "message": "API is healthy",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
