"""Unit tests for the CollectionManager class."""

from unittest.mock import Mock, patch

import pytest
from qdrant_client.models import CollectionInfo

from core.collection_manager import CollectionError, CollectionManager


class TestCollectionManager:
    """Test suite for CollectionManager class."""

    def test_singleton_instance(self):
        """Test that CollectionManager is a singleton."""
        instance1 = CollectionManager.get_instance()
        instance2 = CollectionManager.get_instance()
        assert instance1 is instance2

    def test_get_collections_from_cache(self):
        """Test getting collections from cache."""
        manager = CollectionManager()
        
        # Set up mock cache
        mock_collections = {
            "test-collection": Mock(spec=CollectionInfo)
        }
        manager._cached_collections = mock_collections
        manager._cached_collections_time = 9999999999  # Future time to avoid expiration
        
        with patch('time.time', return_value=10000000000):
            collections = manager.get_collections()
            assert collections == mock_collections

    def test_get_collections_from_qdrant(self):
        """Test getting collections from Qdrant."""
        manager = CollectionManager()
        
        # Clear cache
        manager._cached_collections = None
        manager._cached_collections_time = 0
        
        # Create mock collections response
        mock_collections_response = Mock()
        mock_collection_obj = Mock()
        mock_collection_obj.name = "test-collection"
        mock_collections_response.collections = [mock_collection_obj]
        
        mock_client = Mock()
        mock_client.get_collections.return_value = mock_collections_response
        
        with patch('core.collection_manager.get_qdrant_client', return_value=mock_client):
            with patch('core.collection_manager.time.time', return_value=10000000000):
                collections = manager.get_collections(mock_client)
                assert "test-collection" in collections

    def test_refresh_collections(self):
        """Test refreshing collections cache."""
        manager = CollectionManager()
        
        # Set up existing cache
        manager._cached_collections = {"old-collection": Mock()}
        manager._cached_collections_time = 9999999999
        
        # Create mock collections response
        mock_collections_response = Mock()
        mock_collection_obj = Mock()
        mock_collection_obj.name = "new-collection"
        mock_collections_response.collections = [mock_collection_obj]
        
        mock_client = Mock()
        mock_client.get_collections.return_value = mock_collections_response
        
        with patch('core.collection_manager.get_qdrant_client', return_value=mock_client):
            collections = manager.refresh_collections(mock_client)
            assert "new-collection" in collections
            assert "old-collection" not in collections

    def test_collection_exists(self):
        """Test checking if collection exists."""
        manager = CollectionManager()
        
        # Set up mock cache
        mock_collections = {"test-collection": Mock()}
        manager._cached_collections = mock_collections
        manager._cached_collections_time = 9999999999
        
        with patch('time.time', return_value=10000000000):
            # Test existing collection
            exists = manager.collection_exists("test-collection")
            assert exists is True
            
            # Test non-existing collection
            exists = manager.collection_exists("nonexistent-collection")
            assert exists is False

    def test_get_collection_info(self):
        """Test getting specific collection info."""
        manager = CollectionManager()
        
        # Set up mock cache
        mock_collection_info = Mock(spec=CollectionInfo)
        mock_collections = {"test-collection": mock_collection_info}
        manager._cached_collections = mock_collections
        manager._cached_collections_time = 9999999999
        
        with patch('time.time', return_value=10000000000):
            # Test existing collection
            info = manager.get_collection_info("test-collection")
            assert info is mock_collection_info
            
            # Test non-existing collection
            info = manager.get_collection_info("nonexistent-collection")
            assert info is None

    def test_recreate_collection_success(self):
        """Test successfully recreating a collection."""
        manager = CollectionManager()
        
        # Set up mock cache
        mock_collections = {"test-collection": Mock()}
        manager._cached_collections = mock_collections
        manager._cached_collections_time = 9999999999
        
        mock_client = Mock()
        mock_collections_response = Mock()
        mock_collection_obj = Mock()
        mock_collection_obj.name = "test-collection"
        mock_collections_response.collections = [mock_collection_obj]
        mock_client.get_collections.return_value = mock_collections_response
        
        with patch('core.collection_manager.get_qdrant_client', return_value=mock_client):
            with patch.object(manager, 'refresh_collections'):
                result = manager.recreate_collection(
                    collection_name="test-collection",
                    vectors_config=Mock(),
                    client=mock_client,
                    recreate=True
                )
                assert result is True
                mock_client.delete_collection.assert_called_once_with("test-collection")
                mock_client.create_collection.assert_called_once()

    def test_collection_error_handling(self):
        """Test handling of CollectionError exceptions."""
        manager = CollectionManager()
        
        mock_client = Mock()
        mock_client.get_collections.side_effect = Exception("Connection failed")
        
        with pytest.raises(CollectionError):
            manager.get_collections(mock_client)