"""FastAPI application entry point for Telescope Backend."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import models, graph, weights, activations


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown events."""
    # Startup: initialize model manager, etc.
    print("🔭 Telescope Backend starting up...")
    yield
    # Shutdown: cleanup resources
    print("🔭 Telescope Backend shutting down...")


app = FastAPI(
    title="Telescope Backend",
    description="Deep learning model weight/activation inspector for Mamba-transformer hybrid models",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(graph.router, prefix="/graph", tags=["graph"])
app.include_router(weights.router, prefix="/weights", tags=["weights"])
app.include_router(activations.router, prefix="/activations", tags=["activations"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "telescope-backend"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "Telescope Backend",
        "version": "0.1.0",
        "docs": "/docs",
    }
