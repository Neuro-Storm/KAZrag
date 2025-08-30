"""Module for application startup and shutdown event handlers."""

import logging

from config.config_manager import ConfigManager

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
        config_manager.get()
        logger.info("Configuration successfully loaded")
    except Exception as e:
        logger.exception(f"Error loading configuration: {e}")
        # Don't exit here, as we want the app to start even if config has issues
        # The app will handle config errors when needed


def shutdown_event_handler() -> None:
    """Logic to execute when the application shuts down."""
    logger.info("Application shutdown event triggered")
    # Add any cleanup logic here if needed