"""Module for centralized Qdrant collection management."""

import logging
import time
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import CollectionInfo

from config.config_manager import ConfigManager
from core.qdrant.qdrant_client import get_qdrant_client
from config.settings import Config

logger = logging.getLogger(__name__)


class CollectionError(Exception):
    """Custom exception for collection management errors."""
    pass


class CollectionManager:
    """Centralized manager for Qdrant collections with caching."""
    
    _instance: Optional['CollectionManager'] = None
    
    def __init__(self, cache_ttl: int = 60):
        """Initialize CollectionManager.
        
        Args:
            cache_ttl: Time to live for cached collections in seconds
        """
        self.cache_ttl = cache_ttl
        self._cached_collections: Optional[Dict[str, CollectionInfo]] = None
        self._cached_collections_time: float = 0
        self.config_manager = ConfigManager.get_instance()
        
    @classmethod
    def get_instance(cls, cache_ttl: int = 60) -> 'CollectionManager':
        """Get singleton instance of CollectionManager.
        
        Args:
            cache_ttl: Time to live for cached collections in seconds
            
        Returns:
            CollectionManager: Singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(cache_ttl)
        return cls._instance
    
    def get_collections(self, client: Optional[QdrantClient] = None) -> Dict[str, CollectionInfo]:
        """Get collections from Qdrant with caching.
        
        Args:
            client: Qdrant client (optional, will create new if not provided)
            
        Returns:
            Dict[str, CollectionInfo]: Dictionary of collection names and their info
            
        Raises:
            CollectionError: If there's an error retrieving collections
        """
        current_time = time.time()
        
        # Check if we have valid cached collections
        if (self._cached_collections is not None and 
            current_time - self._cached_collections_time <= self.cache_ttl):
            logger.debug("Returning cached collections")
            return self._cached_collections
            
        # Get client if not provided
        if client is None:
            config = self.config_manager.get()
            client = get_qdrant_client(config)
            
        try:
            # Get collections from Qdrant
            collections_response = client.get_collections()
            collections_dict = {coll.name: coll for coll in collections_response.collections}
            
            # Update cache
            self._cached_collections = collections_dict
            self._cached_collections_time = current_time
            
            logger.debug(f"Loaded {len(collections_dict)} collections from Qdrant")
            return collections_dict
            
        except Exception as e:
            logger.exception(f"Error retrieving collections from Qdrant: {e}")
            raise CollectionError(f"Failed to retrieve collections: {e}")
    
    def refresh_collections(self, client: Optional[QdrantClient] = None) -> Dict[str, CollectionInfo]:
        """Force refresh collections from Qdrant, bypassing cache.
        
        Args:
            client: Qdrant client (optional, will create new if not provided)
            
        Returns:
            Dict[str, CollectionInfo]: Dictionary of collection names and their info
            
        Raises:
            CollectionError: If there's an error retrieving collections
        """
        # Clear cache
        self._cached_collections = None
        self._cached_collections_time = 0
        
        # Get fresh collections
        return self.get_collections(client)
    
    def recreate_collection(
        self, 
        collection_name: str, 
        vectors_config: Any,
        client: Optional[QdrantClient] = None,
        recreate: bool = True
    ) -> bool:
        """Create or recreate a collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vectors_config: Vector configuration for the collection
            client: Qdrant client (optional, will create new if not provided)
            recreate: Whether to delete existing collection before creating
            
        Returns:
            bool: True if collection was created/recreated successfully
            
        Raises:
            CollectionError: If there's an error creating the collection
        """
        # Get client if not provided
        if client is None:
            config = self.config_manager.get()
            client = get_qdrant_client(config)
            
        try:
            # Check if collection exists
            collections = self.get_collections(client)
            collection_exists = collection_name in collections
            
            # Delete existing collection if requested
            if collection_exists and recreate:
                logger.info(f"Deleting existing collection '{collection_name}'")
                client.delete_collection(collection_name)
                # Clear cache after deletion
                self._cached_collections = None
                self._cached_collections_time = 0
            
            # Create new collection
            logger.info(f"Creating collection '{collection_name}'")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config
            )
            
            # Clear cache after creation
            self._cached_collections = None
            self._cached_collections_time = 0
            
            logger.info(f"Collection '{collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error creating collection '{collection_name}': {e}")
            raise CollectionError(f"Failed to create collection '{collection_name}': {e}")
    
    def collection_exists(self, collection_name: str, client: Optional[QdrantClient] = None) -> bool:
        """Check if a collection exists in Qdrant.
        
        Args:
            collection_name: Name of the collection to check
            client: Qdrant client (optional, will create new if not provided)
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            collections = self.get_collections(client)
            return collection_name in collections
        except CollectionError:
            return False
    
    def get_collection_info(
        self, 
        collection_name: str, 
        client: Optional[QdrantClient] = None
    ) -> Optional[CollectionInfo]:
        """Get information about a specific collection.
        
        Args:
            collection_name: Name of the collection
            client: Qdrant client (optional, will create new if not provided)
            
        Returns:
            Optional[CollectionInfo]: Collection information or None if not found
        """
        try:
            collections = self.get_collections(client)
            return collections.get(collection_name)
        except CollectionError:
            return None