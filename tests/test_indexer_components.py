"""Unit tests for the indexer components."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from core.document_loader import DocumentLoader, IndexingError
from core.text_splitter import TextSplitter
from core.indexer_component import Indexer, IndexingError as IndexerError
from langchain_core.documents import Document
from config.settings import Config


class TestDocumentLoader:
    """Test suite for DocumentLoader class."""

    def test_load_text_file_success(self):
        """Test successfully loading a text file."""
        loader = DocumentLoader()
        
        # Create a mock TextLoader
        mock_text_loader = Mock()
        mock_documents = [Document(page_content="Test content", metadata={})]
        mock_text_loader.load.return_value = mock_documents
        
        with patch('core.document_loader.TextLoader', return_value=mock_text_loader):
            documents = loader.load_text_file(Path("test.txt"))
            assert len(documents) == 1
            assert documents[0].page_content == "Test content"

    def test_load_text_file_error(self):
        """Test handling errors when loading a text file."""
        loader = DocumentLoader()
        
        with patch('core.document_loader.TextLoader') as mock_text_loader_class:
            mock_text_loader_class.side_effect = Exception("File not found")
            
            with pytest.raises(Exception):
                loader.load_text_file(Path("nonexistent.txt"))


class TestTextSplitter:
    """Test suite for TextSplitter class."""

    def test_split_documents_success(self):
        """Test successfully splitting documents."""
        # Create mock config
        config = Config()
        config.chunk_size = 500
        config.chunk_overlap = 100
        
        # Create mock text splitter
        mock_splitter = Mock()
        mock_split_documents = [Document(page_content="Chunk 1"), Document(page_content="Chunk 2")]
        mock_splitter.split_documents.return_value = mock_split_documents
        
        with patch('core.text_splitter.get_text_splitter', return_value=mock_splitter):
            splitter = TextSplitter(config)
            documents = [Document(page_content="Long document content that will be split")]
            chunks = splitter.split_documents(documents)
            
            assert len(chunks) == 2
            assert chunks[0].page_content == "Chunk 1"
            assert chunks[1].page_content == "Chunk 2"

    def test_split_documents_error(self):
        """Test handling errors when splitting documents."""
        # Create mock config
        config = Config()
        config.chunk_size = 500
        config.chunk_overlap = 100
        
        # Create mock text splitter that raises an exception
        mock_splitter = Mock()
        mock_splitter.split_documents.side_effect = Exception("Splitting failed")
        
        with patch('core.text_splitter.get_text_splitter', return_value=mock_splitter):
            splitter = TextSplitter(config)
            documents = [Document(page_content="Test content")]
            
            with pytest.raises(Exception):
                splitter.split_documents(documents)


class TestIndexer:
    """Test suite for Indexer class."""

    def test_indexer_initialization(self):
        """Test indexer initialization."""
        config = Config()
        config.index_dense = True
        config.index_bm25 = False
        config.index_hybrid = False
        config.device = "cpu"
        config.indexing_batch_size = 32
        config.memory_threshold = 500 * 1024 * 1024
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024  # 1GB available
            
            indexer = Indexer(config)
            assert indexer.config is config
            assert indexer.index_dense is True
            assert indexer.index_bm25 is False
            assert indexer.index_hybrid is False

    def test_indexer_hybrid_requires_dense(self):
        """Test that hybrid mode requires dense indexing."""
        config = Config()
        config.index_dense = False
        config.index_bm25 = False
        config.index_hybrid = True
        
        with pytest.raises(IndexerError):
            Indexer(config)

    @pytest.mark.asyncio
    async def test_index_documents_empty(self):
        """Test indexing empty document list."""
        config = Config()
        config.index_dense = True
        config.index_bm25 = False
        config.index_hybrid = False
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024
            
            indexer = Indexer(config)
            success, status = await indexer.index_documents([])
            assert success is True
            assert status == "indexed_successfully_no_docs"

    @pytest.mark.asyncio
    async def test_index_documents_success(self):
        """Test successfully indexing documents."""
        config = Config()
        config.index_dense = True
        config.index_bm25 = False
        config.index_hybrid = False
        config.collection_name = "test-collection"
        config.device = "cpu"
        config.indexing_batch_size = 2  # Small batch size for testing
        config.memory_threshold = 500 * 1024 * 1024
        
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
                            
                            indexer = Indexer(config)
                            success, status = await indexer.index_documents(documents)
                            
                            assert success is True
                            assert status == "indexed_successfully"
                            # Should be called twice (3 documents with batch size 2)
                            assert mock_process_batch.call_count == 2

    @pytest.mark.asyncio
    async def test_process_batch_create_collection(self):
        """Test processing batch and creating collection."""
        config = Config()
        config.index_dense = True
        config.index_bm25 = False
        config.index_hybrid = False
        config.collection_name = "test-collection"
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024 * 1024 * 1024
            
            indexer = Indexer(config)
            documents = [Document(page_content="Test content", metadata={})]
            mock_embedder = Mock()
            mock_client = Mock()
            
            with patch('core.indexer_component.QdrantVectorStore.from_documents') as mock_from_documents:
                await indexer._create_collection(documents, mock_embedder, mock_client)
                mock_from_documents.assert_called_once()