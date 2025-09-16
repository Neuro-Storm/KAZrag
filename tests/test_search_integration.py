"""Integration tests for search functionality."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from config.settings import Config
from core.search.searcher import search_in_collection


class TestSearchIntegration:
    """Integration tests for search functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.query = "test query"
        self.collection_name = "test-collection"
        self.device = "cpu"
        self.k = 5

    @pytest.mark.asyncio
    async def test_search_in_collection_dense_success(self):
        """Test successful search in a collection with dense vectors only."""
        mock_results = [
            (Document(page_content='Test result 1', metadata={"source": "doc1.txt"}), 0.9),
            (Document(page_content='Test result 2', metadata={"source": "doc2.txt"}), 0.8)
        ]
        
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.config_manager') as mock_config_manager:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Setup mocks
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5
                                    mock_config_manager.get_instance().get.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Mock collection analysis
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Mock search strategy
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Mock search executor
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Execute search
                                    results, error = await search_in_collection(
                                        self.query, self.collection_name, self.device, self.k
                                    )
                                    
                                    # Verify results
                                    assert results == mock_results
                                    assert error is None

    @pytest.mark.asyncio
    async def test_search_in_collection_hybrid_success(self):
        """Test successful search in a collection with hybrid vectors."""
        mock_results = [
            (Document(page_content="Hybrid result 1", metadata={"source": "doc1.txt"}), 0.95),
            (Document(page_content="Hybrid result 2", metadata={"source": "doc2.txt"}), 0.85)
        ]
        
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    with patch('importlib.import_module') as mock_import_module:
                                        # Setup mocks
                                        mock_client = Mock()
                                        mock_get_client.return_value = mock_client
                                        
                                        mock_config = Mock()
                                        mock_config.search_default_k = 5
                                        mock_config.sparse_embedding = "Qdrant/bm25"
                                        mock_load_config.return_value = mock_config
                                        
                                        mock_get_device.return_value = "cpu"
                                        mock_get_embedder.return_value = Mock()
                                        
                                        # Mock SparseEmbeddingAdapter import
                                        mock_sparse_module = Mock()
                                        mock_sparse_adapter_class = Mock()
                                        mock_sparse_adapter_instance = Mock()
                                        mock_sparse_adapter_class.return_value = mock_sparse_adapter_instance
                                        mock_sparse_module.SparseEmbeddingAdapter = mock_sparse_adapter_class
                                        mock_import_module.return_value = mock_sparse_module
                                        
                                        # Mock collection analysis
                                        mock_analyzer_instance = Mock()
                                        mock_analyzer.return_value = mock_analyzer_instance
                                        mock_analyzer_instance.analyze_collection.return_value = (True, True, "sparse_vector")
                                        
                                        # Mock search strategy
                                        mock_strategy_instance = Mock()
                                        mock_strategy.return_value = mock_strategy_instance
                                        mock_strategy_instance.determine_search_mode.return_value = "hybrid"
                                        
                                        mock_qdrant_store = Mock()
                                        mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                        
                                        # Mock search executor
                                        mock_executor_instance = Mock()
                                        mock_executor.return_value = mock_executor_instance
                                        async def mock_execute_search(*args, **kwargs):
                                            return (mock_results, None)
                                        mock_executor_instance.execute_search = mock_execute_search
                                        
                                        # Execute search with hybrid=True
                                        results, error = await search_in_collection(
                                            self.query, self.collection_name, self.device, self.k, hybrid=True
                                        )
                                        
                                        # Verify results
                                        assert results == mock_results
                                        assert error is None

    @pytest.mark.asyncio
    async def test_search_in_collection_sparse_success(self):
        """Test successful search in a collection with sparse vectors only."""
        mock_results = [
            (Document(page_content='Sparse result 1', metadata={"source": "doc1.txt"}), 0.9),
            (Document(page_content='Sparse result 2', metadata={"source": "doc2.txt"}), 0.8)
        ]
        
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    with patch('importlib.import_module') as mock_import_module:
                                        # Setup mocks
                                        mock_client = Mock()
                                        mock_get_client.return_value = mock_client
                                        
                                        mock_config = Mock()
                                        mock_config.search_default_k = 5
                                        mock_config.sparse_embedding = "Qdrant/bm25"
                                        mock_load_config.return_value = mock_config
                                        
                                        mock_get_device.return_value = "cpu"
                                        mock_get_embedder.return_value = Mock()
                                        
                                        # Mock SparseEmbeddingAdapter import
                                        mock_sparse_module = Mock()
                                        mock_sparse_adapter_class = Mock()
                                        mock_sparse_adapter_instance = Mock()
                                        mock_sparse_adapter_class.return_value = mock_sparse_adapter_instance
                                        mock_sparse_module.SparseEmbeddingAdapter = mock_sparse_adapter_class
                                        mock_import_module.return_value = mock_sparse_module
                                        
                                        # Mock collection analysis
                                        mock_analyzer_instance = Mock()
                                        mock_analyzer.return_value = mock_analyzer_instance
                                        mock_analyzer_instance.analyze_collection.return_value = (False, True, "sparse_vector")
                                        
                                        # Mock search strategy
                                        mock_strategy_instance = Mock()
                                        mock_strategy.return_value = mock_strategy_instance
                                        mock_strategy_instance.determine_search_mode.return_value = "sparse"
                                        
                                        mock_qdrant_store = Mock()
                                        mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                        
                                        # Mock search executor
                                        mock_executor_instance = Mock()
                                        mock_executor.return_value = mock_executor_instance
                                        async def mock_execute_search(*args, **kwargs):
                                            return (mock_results, None)
                                        mock_executor_instance.execute_search = mock_execute_search
                                        
                                        # Execute search
                                        results, error = await search_in_collection(
                                            self.query, self.collection_name, self.device, self.k
                                        )
                                        
                                        # Verify results
                                        assert results == mock_results
                                        assert error is None

    @pytest.mark.asyncio
    async def test_search_in_collection_with_metadata_filter(self):
        """Test search with metadata filter applied."""
        metadata_filter = {
            "author": "Test Author",
            "year": {"$gt": 2020}
        }
        
        mock_results = [
            (Document(page_content='Filtered result', metadata={
                "source": "doc1.txt", 
                "author": "Test Author"
            }), 0.9)
        ]
        
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Setup mocks
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Mock collection analysis
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Mock search strategy
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Mock search executor
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Execute search with metadata filter
                                    results, error = await search_in_collection(
                                        self.query, self.collection_name, self.device, self.k, metadata_filter=metadata_filter
                                    )
                                    
                                    # Verify results
                                    assert results == mock_results
                                    assert error is None

    @pytest.mark.asyncio
    async def test_search_in_collection_with_k_override(self):
        """Test search with custom k value."""
        custom_k = 10  # Override default k value
        
        mock_results = [
            (Document(page_content="Result 1", metadata={"source": "doc1.txt"}), 0.9)
        ]
        
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Setup mocks
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5  # Default from config
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Mock collection analysis
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Mock search strategy
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Mock search executor
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Execute search with custom k
                                    results, error = await search_in_collection(
                                        self.query, self.collection_name, self.device, custom_k
                                    )
                                    
                                    # Verify results
                                    assert results == mock_results
                                    assert error is None

    @pytest.mark.asyncio
    async def test_search_in_collection_default_k(self):
        """Test search using default k value from config."""
        mock_results = [
            (Document(page_content="Result 1", metadata={"source": "doc1.txt"}), 0.9)
        ]
        
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Setup mocks
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5  # Default value
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Mock collection analysis
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Mock search strategy
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Mock search executor
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Execute search with None k (use default)
                                    results, error = await search_in_collection(
                                        self.query, self.collection_name, self.device, None
                                    )
                                    
                                    # Verify results
                                    assert results == mock_results
                                    assert error is None


class TestSearchErrorHandling:
    """Tests for error handling in search process."""

    def setup_method(self):
        """Set up test fixtures."""
        self.query = "test query"
        self.collection_name = "test-collection"
        self.device = "cpu"
        self.k = 5

    @pytest.mark.asyncio
    async def test_search_in_collection_client_error(self):
        """Test search when Qdrant client fails to initialize."""
        # Mock client initialization error
        with patch('core.search.searcher.aget_qdrant_client', side_effect=Exception("Client initialization failed")):
            with patch('core.search.searcher.load_config') as mock_load_config:
                mock_config = Mock()
                mock_config.search_default_k = 5
                mock_load_config.return_value = mock_config
                
                results, error = await search_in_collection(
                    self.query, self.collection_name, self.device
                )
                
                assert results == []
                assert error is not None
                assert "Client initialization failed" in error

    @pytest.mark.asyncio
    async def test_search_in_collection_no_vectors_available(self):
        """Test search when no vectors are available for search."""
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            # Setup mocks
                            mock_client = Mock()
                            mock_get_client.return_value = mock_client
                            
                            mock_config = Mock()
                            mock_config.search_default_k = 5
                            mock_load_config.return_value = mock_config
                            
                            mock_get_device.return_value = "cpu"
                            mock_get_embedder.return_value = Mock()
                            
                            # Mock collection analysis - no dense or sparse vectors
                            mock_analyzer_instance = Mock()
                            mock_analyzer.return_value = mock_analyzer_instance
                            mock_analyzer_instance.analyze_collection.return_value = (False, False, "sparse_vector")
                            
                            results, error = await search_in_collection(
                                self.query, self.collection_name, self.device
                            )
                            
                            assert results == []
                            assert error is not None
                            assert "Нет доступных векторов для поиска" in error

    @pytest.mark.asyncio
    async def test_search_in_collection_sparse_embedding_error(self):
        """Test search when sparse embedding initialization fails."""
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('importlib.import_module', side_effect=Exception("Sparse embedding failed")):
                                # Setup mocks
                                mock_client = Mock()
                                mock_get_client.return_value = mock_client
                                
                                mock_config = Mock()
                                mock_config.search_default_k = 5
                                mock_config.sparse_embedding = "Qdrant/bm25"
                                mock_load_config.return_value = mock_config
                                
                                mock_get_device.return_value = "cpu"
                                mock_get_embedder.return_value = Mock()
                                
                                # Mock collection analysis - has sparse vectors but no dense
                                mock_analyzer_instance = Mock()
                                mock_analyzer.return_value = mock_analyzer_instance
                                mock_analyzer_instance.analyze_collection.return_value = (False, True, "sparse_vector")
                                
                                results, error = await search_in_collection(
                                    self.query, self.collection_name, self.device, self.k
                                )
                                
                                assert results == []
                                assert error is not None
                                assert "Sparse embedding failed" in error

    @pytest.mark.asyncio
    async def test_search_in_collection_execution_error(self):
        """Test search when execution fails."""
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Setup mocks
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Mock collection analysis
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Mock search strategy
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Mock search execution error
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    async def mock_execute_search(*args, **kwargs):
                                        return ([], "Search execution failed")
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Execute search
                                    results, error = await search_in_collection(
                                        self.query, self.collection_name, self.device, self.k
                                    )
                                    
                                    # Verify results
                                    assert results == []
                                    assert error is not None
                                    assert "Search execution failed" in error

    @pytest.mark.asyncio
    async def test_search_in_collection_config_error(self):
        """Test search when config loading fails."""
        # Mock config loading error
        with patch('core.search.searcher.load_config', side_effect=Exception("Config loading failed")):
            results, error = await search_in_collection(
                self.query, self.collection_name, self.device
            )
            
            assert results == []
            assert error is not None
            assert "Config loading failed" in error

    @pytest.mark.asyncio
    async def test_search_in_collection_empty_query(self):
        """Test search with empty query."""
        empty_query = ""
        mock_results = [
            (Document(page_content="Result 1", metadata={"source": "doc1.txt"}), 0.9)
        ]
        
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Setup mocks
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Mock collection analysis
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Mock search strategy
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Mock search executor
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Execute search with empty query
                                    results, error = await search_in_collection(
                                        empty_query, self.collection_name, self.device, self.k
                                    )
                                    
                                    # Should still work with empty query
                                    assert results == mock_results
                                    assert error is None


class TestSearchParametrized:
    """Parametrized tests for search functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.query = "test query"
        self.collection_name = "test-collection"
        self.device = "cpu"
        self.k = 5

    @pytest.mark.parametrize("search_mode", ["dense", "sparse", "hybrid"])
    @pytest.mark.asyncio
    async def test_search_in_collection_parametrized_search_modes(self, search_mode):
        """Parametrized test for search in collection with different search modes."""
        hybrid = (search_mode == "hybrid")
        
        mock_results = [
            (Document(page_content=f"{search_mode} result", metadata={"source": f"{search_mode}.txt"}), 0.9)
        ]
        
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    with patch('importlib.import_module'):
                                        # Setup mocks
                                        mock_client = Mock()
                                        mock_get_client.return_value = mock_client
                                        
                                        mock_config = Mock()
                                        mock_config.search_default_k = 5
                                        mock_config.sparse_embedding = "Qdrant/bm25"
                                        mock_load_config.return_value = mock_config
                                        
                                        mock_get_device.return_value = "cpu"
                                        mock_get_embedder.return_value = Mock()
                                        
                                        # Mock collection analysis
                                        mock_analyzer_instance = Mock()
                                        mock_analyzer.return_value = mock_analyzer_instance
                                        has_dense = search_mode in ["dense", "hybrid"]
                                        has_sparse = search_mode in ["sparse", "hybrid"]
                                        mock_analyzer_instance.analyze_collection.return_value = (has_dense, has_sparse, "sparse_vector")
                                        
                                        # Mock search strategy
                                        mock_strategy_instance = Mock()
                                        mock_strategy.return_value = mock_strategy_instance
                                        mock_strategy_instance.determine_search_mode.return_value = search_mode
                                        
                                        mock_qdrant_store = Mock()
                                        mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                        
                                        # Mock search executor
                                        mock_executor_instance = Mock()
                                        mock_executor.return_value = mock_executor_instance
                                        async def mock_execute_search(*args, **kwargs):
                                            return (mock_results, None)
                                        mock_executor_instance.execute_search = mock_execute_search
                                        
                                        # Execute search
                                        results, error = await search_in_collection(
                                            self.query, self.collection_name, self.device, self.k, hybrid=hybrid
                                        )
                                        
                                        # Verify results
                                        assert results == mock_results
                                        assert error is None

    @pytest.mark.parametrize("k_value", [1, 5, 10, 20])
    @pytest.mark.asyncio
    async def test_search_in_collection_parametrized_k_values(self, k_value):
        """Parametrized test for search with different k values."""
        mock_results = [
            (Document(page_content="Test result", metadata={"source": "doc.txt"}), 0.9)
        ]
        
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Setup mocks
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Mock collection analysis
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Mock search strategy
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Mock search executor
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Execute search
                                    results, error = await search_in_collection(
                                        self.query, self.collection_name, self.device, k_value
                                    )
                                    
                                    # Verify results
                                    assert results == mock_results
                                    assert error is None

    @pytest.mark.parametrize("device_type", ["cpu", "cuda", "auto"])
    @pytest.mark.asyncio
    async def test_search_in_collection_parametrized_devices(self, device_type):
        """Parametrized test for search with different devices."""
        mock_results = [
            (Document(page_content="Test result", metadata={"source": "doc.txt"}), 0.9)
        ]
        
        # Mock all dependencies
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Setup mocks
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = device_type
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Mock collection analysis
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Mock search strategy
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Mock search executor
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Execute search
                                    results, error = await search_in_collection(
                                        self.query, self.collection_name, device_type, self.k
                                    )
                                    
                                    # Verify results
                                    assert results == mock_results
                                    assert error is None