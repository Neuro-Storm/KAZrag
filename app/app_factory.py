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
    
    # Use asgi-request-id middleware for generating request IDs
    from asgi_request_id import RequestIDMiddleware
    app.add_middleware(RequestIDMiddleware)
    
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
    
    # Add favicon route that serves the static favicon
    from fastapi.responses import FileResponse
    @app.get("/favicon.ico")
    async def favicon():
        """Favicon route - serves the static favicon"""
        favicon_path = Path("web/static/favicon.ico")
        if favicon_path.exists():
            return FileResponse(str(favicon_path))
        else:
            # Return empty response if favicon not found
            from fastapi.responses import Response
            return Response(status_code=204)
    
    return app