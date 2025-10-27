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
                from core.embedding.sparse_embedding_adapter import SparseEmbeddingAdapter
                sparse_emb = SparseEmbeddingAdapter(config)
                strategy = SearchStrategy(client, config.collection_name, None, sparse_emb)
                strategy.create_or_update_collection_for_bm25()
                logger.info("BM25 collection configuration initialized successfully")
            except Exception as e:
                logger.exception(f"Error initializing BM25 collection configuration: {e}")
        
        # Initialize Docling converter - only verify it's available
        logger.info("Checking Docling converter availability")
        try:
            from core.converting.docling_converter import DoclingConverter
            # Don't create instance here to avoid loading models at startup
            logger.info("Docling converter available")
        except Exception as e:
            logger.exception(f"Error checking Docling converter: {e}")
            
        # Check if granite backend is configured
        try:
            config = config_manager.get()
            if config.docling_backend == "granite":
                from pathlib import Path
                granite_dir = config.granite_models_dir
                granite_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Granite backend selected, model will be downloaded on-demand")
        except Exception as e:
            logger.warning(f"Error checking Granite backend: {e}")
            
    except Exception as e:
        logger.exception(f"Error loading configuration: {e}")
        # Don't exit here, as we want the app to start even if config has issues
        # The app will handle config errors when needed