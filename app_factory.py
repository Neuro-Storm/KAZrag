"""Module for creating and configuring the FastAPI application."""

import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Import routes registration
from routes import register_routes
# Import exception handlers
from core.exception_handlers import add_exception_handlers

logger = logging.getLogger(__name__)


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
    
    # Add centralized exception handlers
    add_exception_handlers(app)
    
    # Add CORS middleware
    # Normalize ALLOWED_ORIGINS: '*' -> ['*'], otherwise comma-separated list of origins
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*") or "*"
    if allowed_origins_env.strip() == "*":
        allowed_origins = ["*"]
    else:
        allowed_origins = [o.strip() for o in allowed_origins_env.split(',') if o.strip()]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    static_dir = Path("web/static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:
        logger.warning(f"Static directory {static_dir} does not exist")
    
    # Add favicon.ico route to prevent 404 errors
    @app.get("/favicon.ico")
    async def favicon():
        """Favicon route - returns 204 No Content to prevent 404 errors"""
        from fastapi.responses import Response
        return Response(status_code=204)
    
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