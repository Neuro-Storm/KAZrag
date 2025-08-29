"""Integration tests for the search process."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import importlib
from core.search.searcher import search_in_collection
from langchain_core.documents import Document


class TestSearchProcess:
    """Integration tests for the search process."""

    @pytest.mark.asyncio
    async def test_search_in_collection_dense_success(self):
        """Test successful search in a collection with dense vectors only."""
        query = "test query"
        collection_name = "test-collection"
        device = "cpu"
        k = 5
        
        mock_results = [
            ({'content': 'Test result 1', 'metadata': {"source": "doc1.txt"}, 'score': 0.9}, 0.9),
            ({'content': 'Test result 2', 'metadata': {"source": "doc2.txt"}, 'score': 0.8}, 0.8)
        ]
        
        # Мокаем все зависимости
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Настраиваем моки
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Мокаем анализ коллекции
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Мокаем стратегию поиска
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Мокаем исполнитель поиска - используем AsyncMock для async метода
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    # Создаем AsyncMock для асинхронного метода
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Выполняем поиск
                                    results, error = await search_in_collection(query, collection_name, device, k)
                                    
                                    # Проверяем результаты
                                    assert results == mock_results
                                    assert error is None

    @pytest.mark.asyncio
    async def test_search_in_collection_hybrid_success(self):
        """Test successful search in a collection with hybrid vectors."""
        query = "test query"
        collection_name = "test-collection"
        device = "cpu"
        k = 5
        hybrid = True
        
        mock_results = [
            (Document(page_content="Hybrid result 1", metadata={"source": "doc1.txt"}), 0.95),
            (Document(page_content="Hybrid result 2", metadata={"source": "doc2.txt"}), 0.85)
        ]
        
        # Мокаем все зависимости
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    with patch('importlib.import_module') as mock_import_module:
                                        # Настраиваем моки
                                        mock_client = Mock()
                                        mock_get_client.return_value = mock_client
                                        
                                        mock_config = Mock()
                                        mock_config.search_default_k = 5
                                        mock_config.sparse_embedding = "Qdrant/bm25"
                                        mock_load_config.return_value = mock_config
                                        
                                        mock_get_device.return_value = "cpu"
                                        mock_get_embedder.return_value = Mock()
                                        
                                        # Мокаем импорт SparseEmbeddingAdapter
                                        mock_sparse_module = Mock()
                                        mock_sparse_adapter_class = Mock()
                                        mock_sparse_adapter_instance = Mock()
                                        mock_sparse_adapter_class.return_value = mock_sparse_adapter_instance
                                        mock_sparse_module.SparseEmbeddingAdapter = mock_sparse_adapter_class
                                        mock_import_module.return_value = mock_sparse_module
                                        
                                        # Мокаем анализ коллекции
                                        mock_analyzer_instance = Mock()
                                        mock_analyzer.return_value = mock_analyzer_instance
                                        mock_analyzer_instance.analyze_collection.return_value = (True, True, "sparse_vector")
                                        
                                        # Мокаем стратегию поиска
                                        mock_strategy_instance = Mock()
                                        mock_strategy.return_value = mock_strategy_instance
                                        mock_strategy_instance.determine_search_mode.return_value = "hybrid"
                                        
                                        mock_qdrant_store = Mock()
                                        mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                        
                                        # Мокаем исполнитель поиска - используем AsyncMock для async метода
                                        mock_executor_instance = Mock()
                                        mock_executor.return_value = mock_executor_instance
                                        # Создаем AsyncMock для асинхронного метода
                                        async def mock_execute_search(*args, **kwargs):
                                            return (mock_results, None)
                                        mock_executor_instance.execute_search = mock_execute_search
                                        
                                        # Выполняем поиск
                                        results, error = await search_in_collection(query, collection_name, device, k, hybrid)
                                        
                                        # Проверяем результаты
                                        assert results == mock_results
                                        assert error is None

    @pytest.mark.asyncio
    async def test_search_in_collection_sparse_success(self):
        """Test successful search in a collection with sparse vectors only."""
        query = "test query"
        collection_name = "test-collection"
        device = "cpu"
        k = 5
        
        mock_results = [
            ({'content': 'Sparse result 1', 'metadata': {"source": "doc1.txt"}, 'score': 0.9}, 0.9),
            ({'content': 'Sparse result 2', 'metadata': {"source": "doc2.txt"}, 'score': 0.8}, 0.8)
        ]
        
        # Мокаем все зависимости
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    with patch('importlib.import_module') as mock_import_module:
                                        # Настраиваем моки
                                        mock_client = Mock()
                                        mock_get_client.return_value = mock_client
                                        
                                        mock_config = Mock()
                                        mock_config.search_default_k = 5
                                        mock_config.sparse_embedding = "Qdrant/bm25"
                                        mock_load_config.return_value = mock_config
                                        
                                        mock_get_device.return_value = "cpu"
                                        mock_get_embedder.return_value = Mock()
                                        
                                        # Мокаем импорт SparseEmbeddingAdapter
                                        mock_sparse_module = Mock()
                                        mock_sparse_adapter_class = Mock()
                                        mock_sparse_adapter_instance = Mock()
                                        mock_sparse_adapter_class.return_value = mock_sparse_adapter_instance
                                        mock_sparse_module.SparseEmbeddingAdapter = mock_sparse_adapter_class
                                        mock_import_module.return_value = mock_sparse_module
                                        
                                        # Мокаем анализ коллекции
                                        mock_analyzer_instance = Mock()
                                        mock_analyzer.return_value = mock_analyzer_instance
                                        mock_analyzer_instance.analyze_collection.return_value = (False, True, "sparse_vector")
                                        
                                        # Мокаем стратегию поиска
                                        mock_strategy_instance = Mock()
                                        mock_strategy.return_value = mock_strategy_instance
                                        mock_strategy_instance.determine_search_mode.return_value = "sparse"
                                        
                                        mock_qdrant_store = Mock()
                                        mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                        
                                        # Мокаем исполнитель поиска - используем AsyncMock для async метода
                                        mock_executor_instance = Mock()
                                        mock_executor.return_value = mock_executor_instance
                                        # Создаем AsyncMock для асинхронного метода
                                        async def mock_execute_search(*args, **kwargs):
                                            return (mock_results, None)
                                        mock_executor_instance.execute_search = mock_execute_search
                                        
                                        # Выполняем поиск
                                        results, error = await search_in_collection(query, collection_name, device, k)
                                        
                                        # Проверяем результаты
                                        assert results == mock_results
                                        assert error is None

    @pytest.mark.asyncio
    async def test_search_in_collection_with_metadata_filter(self):
        """Test search with metadata filter applied."""
        query = "test query"
        collection_name = "test-collection"
        device = "cpu"
        k = 5
        
        metadata_filter = {
            "author": "Test Author",
            "year": {"$gt": 2020}
        }
        
        mock_results = [
            ({'content': 'Filtered result', 'metadata': {
                "source": "doc1.txt", 
                "author": "Test Author"
            }, 'score': 0.9}, 0.9)
        ]
        
        # Мокаем все зависимости
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Настраиваем моки
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Мокаем анализ коллекции
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Мокаем стратегию поиска
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Мокаем исполнитель поиска - используем AsyncMock для async метода
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    # Создаем AsyncMock для асинхронного метода
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Выполняем поиск
                                    results, error = await search_in_collection(
                                        query, collection_name, device, k, metadata_filter=metadata_filter
                                    )
                                    
                                    # Проверяем результаты
                                    assert results == mock_results
                                    assert error is None

    @pytest.mark.asyncio
    async def test_search_in_collection_with_k_override(self):
        """Test search with custom k value."""
        query = "test query"
        collection_name = "test-collection"
        device = "cpu"
        k = 10  # Переопределяем значение из конфига
        
        mock_results = [
            ({"content": "Result 1", "metadata": {"source": "doc1.txt"}, "score": 0.9}, 0.9)
        ]
        
        # Мокаем все зависимости
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Настраиваем моки
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5  # Значение из конфига
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Мокаем анализ коллекции
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Мокаем стратегию поиска
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Мокаем исполнитель поиска - используем AsyncMock для async метода
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    # Создаем AsyncMock для асинхронного метода
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Выполняем поиск
                                    results, error = await search_in_collection(query, collection_name, device, k)
                                    
                                    # Проверяем результаты
                                    assert results == mock_results
                                    assert error is None

    @pytest.mark.asyncio
    async def test_search_in_collection_default_k(self):
        """Test search using default k value from config."""
        query = "test query"
        collection_name = "test-collection"
        device = "cpu"
        k = None  # Используем значение по умолчанию из конфига
        
        mock_results = [
            ({"content": "Result 1", "metadata": {"source": "doc1.txt"}, "score": 0.9}, 0.9)
        ]
        
        # Мокаем все зависимости
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Настраиваем моки
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5  # Значение по умолчанию
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Мокаем анализ коллекции
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Мокаем стратегию поиска
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Мокаем исполнитель поиска - используем AsyncMock для async метода
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    # Создаем AsyncMock для асинхронного метода
                                    async def mock_execute_search(*args, **kwargs):
                                        return (mock_results, None)
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Выполняем поиск
                                    results, error = await search_in_collection(query, collection_name, device, k)
                                    
                                    # Проверяем результаты
                                    assert results == mock_results
                                    assert error is None


class TestSearchErrorHandling:
    """Tests for error handling in search process."""

    @pytest.mark.asyncio
    async def test_search_in_collection_client_error(self):
        """Test search when Qdrant client fails to initialize."""
        query = "test query"
        collection_name = "test-collection"
        device = "cpu"
        
        # Мокаем ошибку при создании клиента
        with patch('core.search.searcher.aget_qdrant_client', side_effect=Exception("Client initialization failed")):
            with patch('core.search.searcher.load_config') as mock_load_config:
                mock_config = Mock()
                mock_config.search_default_k = 5
                mock_load_config.return_value = mock_config
                
                results, error = await search_in_collection(query, collection_name, device)
                
                assert results == []
                assert error is not None
                assert "Client initialization failed" in error

    @pytest.mark.asyncio
    async def test_search_in_collection_no_vectors_available(self):
        """Test search when no vectors are available for search."""
        query = "test query"
        collection_name = "test-collection"
        device = "cpu"
        
        # Мокаем все зависимости
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            # Настраиваем моки
                            mock_client = Mock()
                            mock_get_client.return_value = mock_client
                            
                            mock_config = Mock()
                            mock_config.search_default_k = 5
                            mock_load_config.return_value = mock_config
                            
                            mock_get_device.return_value = "cpu"
                            mock_get_embedder.return_value = Mock()
                            
                            # Мокаем анализ коллекции - нет ни dense, ни sparse векторов
                            mock_analyzer_instance = Mock()
                            mock_analyzer.return_value = mock_analyzer_instance
                            mock_analyzer_instance.analyze_collection.return_value = (False, False, "sparse_vector")
                            
                            results, error = await search_in_collection(query, collection_name, device)
                            
                            assert results == []
                            assert error is not None
                            assert "Нет доступных векторов для поиска" in error

    @pytest.mark.asyncio
    async def test_search_in_collection_sparse_embedding_error(self):
        """Test search when sparse embedding initialization fails."""
        query = "test query"
        collection_name = "test-collection"
        device = "cpu"
        k = 5
        
        # Мокаем все зависимости
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('importlib.import_module', side_effect=Exception("Sparse embedding failed")):
                                # Настраиваем моки
                                mock_client = Mock()
                                mock_get_client.return_value = mock_client
                                
                                mock_config = Mock()
                                mock_config.search_default_k = 5
                                mock_config.sparse_embedding = "Qdrant/bm25"
                                mock_load_config.return_value = mock_config
                                
                                mock_get_device.return_value = "cpu"
                                mock_get_embedder.return_value = Mock()
                                
                                # Мокаем анализ коллекции - есть sparse векторы, но dense нет
                                mock_analyzer_instance = Mock()
                                mock_analyzer.return_value = mock_analyzer_instance
                                mock_analyzer_instance.analyze_collection.return_value = (False, True, "sparse_vector")
                                
                                results, error = await search_in_collection(query, collection_name, device, k)
                                
                                assert results == []
                                assert error is not None
                                assert "Sparse embedding failed" in error

    @pytest.mark.asyncio
    async def test_search_in_collection_execution_error(self):
        """Test search when execution fails."""
        query = "test query"
        collection_name = "test-collection"
        device = "cpu"
        k = 5
        
        # Мокаем все зависимости
        with patch('core.search.searcher.aget_qdrant_client') as mock_get_client:
            with patch('core.search.searcher.load_config') as mock_load_config:
                with patch('core.search.searcher.get_dense_embedder') as mock_get_embedder:
                    with patch('core.search.searcher.get_search_device') as mock_get_device:
                        with patch('core.search.searcher.CollectionAnalyzer') as mock_analyzer:
                            with patch('core.search.searcher.SearchStrategy') as mock_strategy:
                                with patch('core.search.searcher.SearchExecutor') as mock_executor:
                                    # Настраиваем моки
                                    mock_client = Mock()
                                    mock_get_client.return_value = mock_client
                                    
                                    mock_config = Mock()
                                    mock_config.search_default_k = 5
                                    mock_load_config.return_value = mock_config
                                    
                                    mock_get_device.return_value = "cpu"
                                    mock_get_embedder.return_value = Mock()
                                    
                                    # Мокаем анализ коллекции
                                    mock_analyzer_instance = Mock()
                                    mock_analyzer.return_value = mock_analyzer_instance
                                    mock_analyzer_instance.analyze_collection.return_value = (True, False, "sparse_vector")
                                    
                                    # Мокаем стратегию поиска
                                    mock_strategy_instance = Mock()
                                    mock_strategy.return_value = mock_strategy_instance
                                    mock_strategy_instance.determine_search_mode.return_value = "dense"
                                    
                                    mock_qdrant_store = Mock()
                                    mock_strategy_instance.create_qdrant_store.return_value = mock_qdrant_store
                                    
                                    # Мокаем ошибку при выполнении поиска
                                    mock_executor_instance = Mock()
                                    mock_executor.return_value = mock_executor_instance
                                    # Создаем AsyncMock для асинхронного метода, который возвращает ошибку
                                    async def mock_execute_search(*args, **kwargs):
                                        return ([], "Search execution failed")
                                    mock_executor_instance.execute_search = mock_execute_search
                                    
                                    # Выполняем поиск
                                    results, error = await search_in_collection(query, collection_name, device, k)
                                    
                                    # Проверяем результаты
                                    assert results == []
                                    assert error is not None
                                    assert "Search execution failed" in error
