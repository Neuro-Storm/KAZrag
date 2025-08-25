"""Unit tests for the search components."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.collection_analyzer import CollectionAnalyzer, SearchError
from core.search_strategy import SearchStrategy
from core.search_executor import SearchExecutor
from langchain_qdrant import QdrantVectorStore


class TestCollectionAnalyzer:
    """Test suite for CollectionAnalyzer class."""

    def test_analyze_collection_dense_only(self):
        """Test analyzing collection with only dense vectors."""
        analyzer = CollectionAnalyzer()
        
        # Create mock collection info with dense vectors only
        mock_collection_info = Mock()
        mock_config = Mock()
        mock_params = Mock()
        mock_params.vectors = {"size": 768}
        mock_params.sparse_vectors = {}
        mock_config.params = mock_params
        mock_collection_info.config = mock_config
        
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection_info
        
        has_dense, has_sparse, sparse_vector_name = analyzer.analyze_collection(mock_client, "test-collection")
        
        assert has_dense is True
        assert has_sparse is False
        assert sparse_vector_name == "sparse_vector"

    def test_analyze_collection_sparse_only(self):
        """Test analyzing collection with only sparse vectors."""
        analyzer = CollectionAnalyzer()
        
        # Create mock collection info with sparse vectors only
        mock_collection_info = Mock()
        mock_config = Mock()
        mock_params = Mock()
        mock_params.vectors = {}
        mock_params.sparse_vectors = {"sparse_vector": {}}
        mock_config.params = mock_params
        mock_collection_info.config = mock_config
        
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection_info
        
        has_dense, has_sparse, sparse_vector_name = analyzer.analyze_collection(mock_client, "test-collection")
        
        assert has_dense is False
        assert has_sparse is True
        assert sparse_vector_name == "sparse_vector"

    def test_analyze_collection_hybrid(self):
        """Test analyzing collection with both dense and sparse vectors."""
        analyzer = CollectionAnalyzer()
        
        # Create mock collection info with both vector types
        mock_collection_info = Mock()
        mock_config = Mock()
        mock_params = Mock()
        mock_params.vectors = {"size": 768}
        mock_params.sparse_vectors = {"sparse_vector": {}}
        mock_config.params = mock_params
        mock_collection_info.config = mock_config
        
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection_info
        
        has_dense, has_sparse, sparse_vector_name = analyzer.analyze_collection(mock_client, "test-collection")
        
        assert has_dense is True
        assert has_sparse is True
        assert sparse_vector_name == "sparse_vector"

    def test_analyze_collection_error(self):
        """Test analyzing collection when Qdrant client fails."""
        analyzer = CollectionAnalyzer()
        
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Connection failed")
        
        has_dense, has_sparse, sparse_vector_name = analyzer.analyze_collection(mock_client, "test-collection")
        
        # Should fall back to default values
        assert has_dense is True  # Default assumption
        assert has_sparse is False
        assert sparse_vector_name == "sparse_vector"


class TestSearchStrategy:
    """Test suite for SearchStrategy class."""

    def test_determine_search_mode_hybrid(self):
        """Test determining hybrid search mode."""
        mock_client = Mock()
        mock_embedder = Mock()
        mock_sparse_emb = Mock()
        
        strategy = SearchStrategy(mock_client, "test-collection", mock_embedder, mock_sparse_emb)
        strategy.has_dense = True
        strategy.has_sparse = True
        
        mode = strategy.determine_search_mode(hybrid=True)
        assert mode == "hybrid"

    def test_determine_search_mode_hybrid_no_sparse(self):
        """Test determining hybrid search mode when sparse vectors are not available."""
        mock_client = Mock()
        mock_embedder = Mock()
        mock_sparse_emb = Mock()
        
        strategy = SearchStrategy(mock_client, "test-collection", mock_embedder, mock_sparse_emb)
        strategy.has_dense = True
        strategy.has_sparse = False
        
        with patch('core.search_strategy.logger.warning') as mock_warning:
            mode = strategy.determine_search_mode(hybrid=True)
            assert mode == "dense"
            mock_warning.assert_called_once()

    def test_determine_search_mode_sparse_only(self):
        """Test determining search mode for sparse-only collection."""
        mock_client = Mock()
        mock_embedder = Mock()
        mock_sparse_emb = Mock()
        
        strategy = SearchStrategy(mock_client, "test-collection", mock_embedder, mock_sparse_emb)
        strategy.has_dense = False
        strategy.has_sparse = True
        
        mode = strategy.determine_search_mode(hybrid=False)
        assert mode == "sparse"

    def test_determine_search_mode_dense_default(self):
        """Test determining search mode defaults to dense."""
        mock_client = Mock()
        mock_embedder = Mock()
        mock_sparse_emb = Mock()
        
        strategy = SearchStrategy(mock_client, "test-collection", mock_embedder, mock_sparse_emb)
        strategy.has_dense = True
        strategy.has_sparse = False
        
        mode = strategy.determine_search_mode(hybrid=False)
        assert mode == "dense"

    def test_create_qdrant_store_hybrid(self):
        """Test creating QdrantVectorStore for hybrid search."""
        mock_client = Mock()
        mock_embedder = Mock()
        mock_sparse_emb = Mock()
        
        strategy = SearchStrategy(mock_client, "test-collection", mock_embedder, mock_sparse_emb)
        strategy.has_dense = True
        strategy.has_sparse = True
        strategy.sparse_vector_name = "sparse_vector"
        
        with patch('core.search_strategy.QdrantVectorStore') as mock_qdrant_store:
            mock_instance = Mock()
            mock_qdrant_store.return_value = mock_instance
            
            qdrant_store = strategy.create_qdrant_store("hybrid")
            assert qdrant_store is mock_instance
            mock_qdrant_store.assert_called_once()

    def test_create_qdrant_store_sparse(self):
        """Test creating QdrantVectorStore for sparse-only search."""
        mock_client = Mock()
        mock_embedder = Mock()
        mock_sparse_emb = Mock()
        
        strategy = SearchStrategy(mock_client, "test-collection", mock_embedder, mock_sparse_emb)
        strategy.has_dense = False
        strategy.has_sparse = True
        strategy.sparse_vector_name = "sparse_vector"
        
        with patch('core.search_strategy.QdrantVectorStore') as mock_qdrant_store:
            mock_instance = Mock()
            mock_qdrant_store.return_value = mock_instance
            
            qdrant_store = strategy.create_qdrant_store("sparse")
            assert qdrant_store is mock_instance

    def test_create_qdrant_store_dense(self):
        """Test creating QdrantVectorStore for dense-only search."""
        mock_client = Mock()
        mock_embedder = Mock()
        mock_sparse_emb = Mock()
        
        strategy = SearchStrategy(mock_client, "test-collection", mock_embedder, mock_sparse_emb)
        strategy.has_dense = True
        strategy.has_sparse = False
        
        with patch('core.search_strategy.QdrantVectorStore') as mock_qdrant_store:
            mock_instance = Mock()
            mock_qdrant_store.return_value = mock_instance
            
            qdrant_store = strategy.create_qdrant_store("dense")
            assert qdrant_store is mock_instance


class TestSearchExecutor:
    """Test suite for SearchExecutor class."""

    @pytest.mark.asyncio
    async def test_execute_search_without_filter(self):
        """Test executing search without metadata filter."""
        executor = SearchExecutor()
        
        mock_qdrant = Mock()
        mock_results = [("doc1", 0.9), ("doc2", 0.8)]
        # Create a proper async mock
        async def mock_search(*args, **kwargs):
            return mock_results
        mock_qdrant.asimilarity_search_with_score = mock_search
        
        results, error = await executor.execute_search(
            qdrant=mock_qdrant,
            query="test query",
            k=5
        )
        
        assert results == mock_results
        assert error is None

    @pytest.mark.asyncio
    async def test_execute_search_with_filter(self):
        """Test executing search with metadata filter."""
        executor = SearchExecutor()
        
        mock_qdrant = Mock()
        mock_results = [("doc1", 0.9)]
        # Create a proper async mock
        async def mock_search(*args, **kwargs):
            return mock_results
        mock_qdrant.asimilarity_search_with_score = mock_search
        
        metadata_filter = {
            "author": "Test Author",
            "year": {"$gt": 2020}
        }
        
        results, error = await executor.execute_search(
            qdrant=mock_qdrant,
            query="test query",
            k=5,
            metadata_filter=metadata_filter
        )
        
        assert results == mock_results
        assert error is None

    @pytest.mark.asyncio
    async def test_execute_search_error(self):
        """Test executing search when an error occurs."""
        executor = SearchExecutor()
        
        mock_qdrant = Mock()
        # Create a proper async mock that raises an exception
        async def mock_search(*args, **kwargs):
            raise Exception("Search failed")
        mock_qdrant.asimilarity_search_with_score = mock_search
        
        results, error = await executor.execute_search(
            qdrant=mock_qdrant,
            query="test query",
            k=5
        )
        
        assert results == []
        assert error is not None
        assert "Search failed" in error