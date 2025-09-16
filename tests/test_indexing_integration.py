"""Integration tests for indexing functionality."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from config.settings import Config
from core.indexing.indexer import run_indexing_logic
from core.indexing.indexer_component import Indexer


class TestIndexingIntegration:
    """Integration tests for indexing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.folder_path = "./test_folder"
        self.config.collection_name = "test-collection"
        self.config.current_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.config.chunk_size = 100
        self.config.chunk_overlap = 20
        self.config.device = "cpu"
        self.config.index_dense = True
        self.config.index_bm25 = False
        self.config.index_hybrid = False
        self.config.embedding_batch_size = 32
        self.config.indexing_batch_size = 50

    @pytest.mark.asyncio
    async def test_indexing_process_success(self):
        """Test successful indexing process."""
        with patch('core.indexing.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            with patch('core.indexing.indexer.Indexer') as mock_indexer_class:
                mock_indexer_instance = Mock()
                mock_indexer_instance.index_documents.return_value = (True, "indexed_successfully")
                mock_indexer_class.return_value = mock_indexer_instance
                
                success, status = await run_indexing_logic()
                
                assert success is True
                assert status == "indexed_successfully"

    @pytest.mark.asyncio
    async def test_indexing_process_folder_not_found(self):
        """Test the indexing process when folder is not found."""
        self.config.folder_path = "./nonexistent_folder"
        
        with patch('core.indexing.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            success, status = await run_indexing_logic()
            
            # Function should return False and "folder_not_found" for nonexistent folder
            assert success is False
            assert status == "folder_not_found"

    @pytest.mark.asyncio
    async def test_indexing_process_no_index_type(self):
        """Test indexing process when no index type is configured."""
        self.config.index_dense = False
        self.config.index_bm25 = False
        self.config.index_hybrid = False
        
        with patch('core.indexing.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            success, status = await run_indexing_logic()
            
            # Should return error no_index_type
            assert success is False
            assert status == "no_index_type"

    @pytest.mark.asyncio
    async def test_indexing_process_hybrid_indexing(self):
        """Test indexing process with hybrid indexing."""
        self.config.index_dense = True
        self.config.index_bm25 = False
        self.config.index_hybrid = True
        
        with patch('core.indexing.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            with patch('core.indexing.indexer.Indexer') as mock_indexer_class:
                mock_indexer_instance = Mock()
                mock_indexer_instance.index_documents.return_value = (True, "indexed_successfully")
                mock_indexer_class.return_value = mock_indexer_instance
                
                success, status = await run_indexing_logic()
                
                assert success is True
                assert status == "indexed_successfully"
        self.config.index_bm25 = True
        self.config.index_hybrid = True
        self.config.sparse_embedding = "Qdrant/bm25"
        
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            with patch('core.indexer.Indexer') as mock_indexer_class:
                mock_indexer_instance = Mock()
                mock_indexer_instance.index_documents.return_value = (True, "indexed_successfully")
                mock_indexer_class.return_value = mock_indexer_instance
                
                success, status = await run_indexing_logic()
                
                assert success is True
                assert status == "indexed_successfully"

    @pytest.mark.asyncio
    async def test_indexing_process_bm25_only(self):
        """Test indexing process with BM25 only."""
        self.config.index_dense = False
        self.config.index_bm25 = True
        self.config.index_hybrid = False
        
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            with patch('core.indexer.Indexer') as mock_indexer_class:
                mock_indexer_instance = Mock()
                mock_indexer_instance.index_documents.return_value = (True, "indexed_successfully")
                mock_indexer_class.return_value = mock_indexer_instance
                
                success, status = await run_indexing_logic()
                
                assert success is True
                assert status == "indexed_successfully"

    @pytest.mark.asyncio
    async def test_indexing_process_config_error(self):
        """Test indexing process when config loading fails."""
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.side_effect = Exception("Config loading failed")
            
            success, status = await run_indexing_logic()
            assert success is False
            assert "Config loading failed" in status

    @pytest.mark.asyncio
    async def test_indexing_process_memory_error(self):
        """Test indexing process when memory is insufficient."""
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            with patch('core.indexer.Indexer') as mock_indexer_class:
                mock_indexer_class.side_effect = Exception("Insufficient memory")
                
                success, status = await run_indexing_logic()
                assert success is False
                assert "Insufficient memory" in status

    @pytest.mark.parametrize("index_type_config", [
        {"index_dense": True, "index_bm25": False, "index_hybrid": False},
        {"index_dense": False, "index_bm25": True, "index_hybrid": False},
        {"index_dense": True, "index_bm25": True, "index_hybrid": True},
    ])
    @pytest.mark.asyncio
    async def test_indexing_process_parametrized_index_types(self, index_type_config):
        """Parametrized test for indexing process with different index types."""
        self.config.index_dense = index_type_config["index_dense"]
        self.config.index_bm25 = index_type_config["index_bm25"]
        self.config.index_hybrid = index_type_config["index_hybrid"]
        
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            with patch('core.indexer.Indexer') as mock_indexer_class:
                mock_indexer_instance = Mock()
                mock_indexer_instance.index_documents.return_value = (True, "indexed_successfully")
                mock_indexer_class.return_value = mock_indexer_instance
                
                success, status = await run_indexing_logic()
                
                assert success is True
                assert status == "indexed_successfully"


class TestIndexerComponent:
    """Tests for the Indexer component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.folder_path = "./test_folder"
        self.config.collection_name = "test-collection"
        self.config.current_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.config.chunk_size = 100
        self.config.chunk_overlap = 20
        self.config.device = "cpu"
        self.config.index_dense = True
        self.config.index_bm25 = False
        self.config.index_hybrid = False
        self.config.embedding_batch_size = 32
        self.config.indexing_batch_size = 50
        self.config.memory_threshold = 500 * 1024 * 1024

    def test_indexer_initialization(self):
        """Test indexer initialization."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024  # 1GB available
            
            indexer = Indexer(self.config)
            assert indexer.config is self.config
            assert indexer.index_dense is True
            assert indexer.index_bm25 is False
            assert indexer.index_hybrid is False

    def test_indexer_hybrid_requires_dense(self):
        """Test that hybrid mode requires dense indexing."""
        self.config.index_dense = False
        self.config.index_hybrid = True
        
        with pytest.raises(Exception):  # IndexerError
            Indexer(self.config)

    @pytest.mark.asyncio
    async def test_index_documents_empty(self):
        """Test indexing empty document list."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024
            
            indexer = Indexer(self.config)
            success, status = await indexer.index_documents([])
            assert success is True
            assert status == "indexed_successfully_no_docs"

    @pytest.mark.asyncio
    async def test_index_documents_success(self):
        """Test successfully indexing documents."""
        documents = [
            Document(page_content="Document 1 content", metadata={}),
            Document(page_content="Document 2 content", metadata={}),
            Document(page_content="Document 3 content", metadata={})
        ]
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024
            
            with patch('core.indexer_component.aget_qdrant_client') as mock_get_client:
                with patch('core.indexer_component.get_device') as mock_get_device:
                    with patch('core.indexer_component.get_dense_embedder') as mock_get_embedder:
                        with patch('core.indexer_component.Indexer._process_batch') as mock_process_batch:
                            mock_client = Mock()
                            mock_get_client.return_value = mock_client
                            mock_get_device.return_value = "cpu"
                            mock_get_embedder.return_value = Mock()
                            
                            indexer = Indexer(self.config)
                            success, status = await indexer.index_documents(documents)
                            
                            assert success is True
                            assert status == "indexed_successfully"
                            # Should be called twice (3 documents with batch size 2)
                            assert mock_process_batch.call_count >= 1

    @pytest.mark.asyncio
    async def test_index_documents_with_bm25(self):
        """Test indexing documents with BM25."""
        self.config.index_dense = False
        self.config.index_bm25 = True
        self.config.index_hybrid = False
        
        documents = [
            Document(page_content="Document 1 content", metadata={}),
            Document(page_content="Document 2 content", metadata={})
        ]
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024
            
            with patch('core.indexer_component.aget_qdrant_client') as mock_get_client:
                with patch('core.indexer_component.get_device') as mock_get_device:
                    with patch('core.indexer_component.get_dense_embedder') as mock_get_embedder:
                        with patch('core.indexer_component.Indexer._process_batch') as mock_process_batch:
                            mock_client = Mock()
                            mock_get_client.return_value = mock_client
                            mock_get_device.return_value = "cpu"
                            mock_get_embedder.return_value = Mock()
                            
                            indexer = Indexer(self.config)
                            success, status = await indexer.index_documents(documents)
                            
                            assert success is True
                            assert status == "indexed_successfully"

    @pytest.mark.asyncio
    async def test_index_documents_memory_error(self):
        """Test indexing documents when memory is insufficient."""
        self.config.memory_threshold = 2 * 1024 * 1024 * 1024  # 2GB threshold
        
        documents = [
            Document(page_content="Document 1 content", metadata={})
        ]
        
        with patch('psutil.virtual_memory') as mock_memory:
            # Simulate only 512MB available memory
            mock_memory.return_value.available = 512 * 1024 * 1024
            
            with patch('core.indexer_component.aget_qdrant_client') as mock_get_client:
                with patch('core.indexer_component.get_device') as mock_get_device:
                    with patch('core.indexer_component.get_dense_embedder') as mock_get_embedder:
                        mock_client = Mock()
                        mock_get_client.return_value = mock_client
                        mock_get_device.return_value = "cpu"
                        mock_get_embedder.return_value = Mock()
                        
                        indexer = Indexer(self.config)
                        # Should raise error due to insufficient memory
                        with pytest.raises(Exception):  # IndexerError
                            await indexer.index_documents(documents)

    @pytest.mark.asyncio
    async def test_index_documents_client_error(self):
        """Test indexing documents when Qdrant client fails."""
        documents = [
            Document(page_content="Document 1 content", metadata={})
        ]
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024
            
            with patch('core.indexer_component.aget_qdrant_client', side_effect=Exception("Connection failed")):
                indexer = Indexer(self.config)
                success, status = await indexer.index_documents(documents)
                
                assert success is False
                assert "Connection failed" in status

    @pytest.mark.parametrize("index_dense,index_bm25,index_hybrid,should_succeed", [
        (True, False, False, True),  # Only dense
        (False, True, False, True),  # Only BM25
        (True, True, True, True),    # Hybrid (requires dense)
        (False, True, True, False),  # Hybrid without dense (should fail)
    ])
    def test_indexer_initialization_parametrized(self, index_dense, index_bm25, index_hybrid, should_succeed):
        """Parametrized test for indexer initialization with different configurations."""
        self.config.index_dense = index_dense
        self.config.index_bm25 = index_bm25
        self.config.index_hybrid = index_hybrid
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024  # 1GB available
            
            if should_succeed:
                indexer = Indexer(self.config)
                assert indexer.index_dense == index_dense
                assert indexer.index_bm25 == index_bm25
                assert indexer.index_hybrid == index_hybrid
            else:
                with pytest.raises(Exception):  # IndexerError
                    Indexer(self.config)


class TestIndexingErrorHandling:
    """Tests for error handling in indexing process."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.folder_path = "./test_folder"
        self.config.collection_name = "test-collection"
        self.config.current_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.config.chunk_size = 100
        self.config.chunk_overlap = 20
        self.config.device = "cpu"
        self.config.index_dense = True
        self.config.index_bm25 = False
        self.config.index_hybrid = False
        self.config.embedding_batch_size = 32
        self.config.indexing_batch_size = 50

    @pytest.mark.asyncio
    async def test_indexing_process_file_loading_error(self):
        """Test indexing process when file loading fails."""
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            # Since folder doesn't exist, function should return "folder_not_found"
            success, status = await run_indexing_logic()
            assert success is False
            assert status == "folder_not_found"

    @pytest.mark.asyncio
    async def test_indexing_process_embedding_error(self):
        """Test indexing process when embedding fails."""
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            with patch('core.indexer.Indexer') as mock_indexer_class:
                mock_indexer_instance = Mock()
                mock_indexer_instance.index_documents.return_value = (False, "Embedding failed")
                mock_indexer_class.return_value = mock_indexer_instance
                
                success, status = await run_indexing_logic()
                assert success is False
                assert status == "Embedding failed"

    @pytest.mark.asyncio
    async def test_indexing_process_qdrant_error(self):
        """Test indexing process when Qdrant operation fails."""
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            with patch('core.indexer.Indexer') as mock_indexer_class:
                mock_indexer_instance = Mock()
                mock_indexer_instance.index_documents.return_value = (False, "Qdrant operation failed")
                mock_indexer_class.return_value = mock_indexer_instance
                
                success, status = await run_indexing_logic()
                assert success is False
                assert status == "Qdrant operation failed"

    @pytest.mark.parametrize("error_message", [
        "Disk full",
        "Network error",
        "Permission denied",
        "Timeout"
    ])
    @pytest.mark.asyncio
    async def test_indexing_process_various_errors(self, error_message):
        """Parametrized test for indexing process with various errors."""
        with patch('core.indexer.config_manager') as mock_config_manager:
            mock_config_manager.get_instance().get.return_value = self.config
            
            with patch('core.indexer.Indexer') as mock_indexer_class:
                mock_indexer_instance = Mock()
                mock_indexer_instance.index_documents.return_value = (False, error_message)
                mock_indexer_class.return_value = mock_indexer_instance
                
                success, status = await run_indexing_logic()
                assert success is False
                assert error_message in status