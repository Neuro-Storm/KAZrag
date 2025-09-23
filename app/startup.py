"""Module for application startup and shutdown event handlers."""

import logging

from config.config_manager import ConfigManager
from core.search.search_strategy import SearchStrategy
from core.qdrant.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)


def startup_event_handler() -> None:
    """Logic to execute when the application starts.
    
    This includes loading configuration and checking Qdrant availability.
    """
    logger.info("Application startup event triggered")
    
    # Load configuration at startup (without Qdrant check to avoid blocking startup)
    try:
        logger.info("Loading configuration")
        config_manager = ConfigManager.get_instance()
        config = config_manager.get()
        logger.info("Configuration successfully loaded")
        
        # Initialize BM25 collection if enabled
        if config.use_bm25:
            logger.info("Initializing BM25 collection configuration")
            try:
                client = get_qdrant_client(config)
                strategy = SearchStrategy(client, config.collection_name, None)
                strategy.create_or_update_collection_for_bm25()
                logger.info("BM25 collection configuration initialized successfully")
            except Exception as e:
                logger.exception(f"Error initializing BM25 collection configuration: {e}")
    except Exception as e:
        logger.exception(f"Error loading configuration: {e}")
        # Don't exit here, as we want the app to start even if config has issues
        # The app will handle config errors when needed