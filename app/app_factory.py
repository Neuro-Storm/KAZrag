"""Module for creating and configuring the FastAPI application."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# Import routes registration
from app.routes import register_routes

# Import exception handlers
from core.utils.exception_handlers import add_exception_handlers

# Import resource path resolver
from config.resource_path import resource_path

logger = logging.getLogger(__name__)


def _parse_allowed_origins(allowed_origins_env: str) -> list:
    """Parse ALLOWED_ORIGINS environment variable into a list of origins.
    
    Args:
        allowed_origins_env: Environment variable value
        
    Returns:
        List of origins for CORS middleware
    """
    # Handle special case of "*" (allow all origins)
    if allowed_origins_env.strip() == "*":
        return ["*"]
    
    # Split by comma and strip whitespace
    origins = [origin.strip() for origin in allowed_origins_env.split(',') if origin.strip()]
    
    # Validate and normalize origins
    normalized_origins = []
    for origin in origins:
        # Skip empty origins
        if not origin:
            continue
            
        # Add scheme if missing (assume https for security)
        if not origin.startswith(('http://', 'https://')):
            origin = f"https://{origin}"
            
        normalized_origins.append(origin)
    
    return normalized_origins


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Create the main FastAPI application
    app = FastAPI(
        title="KAZrag",
        description="Поисковая система на основе векторного поиска"
    )
    
    # Use asgi-request-id middleware for generating request IDs
    from asgi_request_id import RequestIDMiddleware
    app.add_middleware(RequestIDMiddleware)
    
    # Add centralized exception handlers
    add_exception_handlers(app)
    
    # Add CORS middleware
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*") or "*"
    allowed_origins = _parse_allowed_origins(allowed_origins_env)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    static_dir = resource_path("web/static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:
        logger.warning(f"Static directory {static_dir} does not exist")
    
    # Register routes
    register_routes(app)
    
    # Add root route for redirecting to search page
    @app.get("/", response_class=RedirectResponse)
    async def root():
        """Root route - redirects to search page"""
        return RedirectResponse(url="/api/search/")
    
    # Add settings route that redirects to admin
    @app.get("/settings", response_class=RedirectResponse)
    async def settings_redirect():
        """Settings route - redirects to admin panel"""
        return RedirectResponse(url="/api/admin/settings/")
    
    return app