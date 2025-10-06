"""Module for registering application routes."""

from fastapi import FastAPI

from web.admin_app import app as admin_router
from web.search_app import app as search_router
from web.indexing_app import app as indexing_router


def register_routes(app: FastAPI) -> None:
    """Register all application routes.
    
    Args:
        app (FastAPI): The FastAPI application instance
    """
    # Register the search router with prefix
    app.include_router(search_router, prefix="/api/search")
    
    # Register the admin router with prefix
    app.include_router(admin_router, prefix="/api/admin")
    
    # Register the indexing router with prefix
    app.include_router(indexing_router, prefix="/api/indexing")