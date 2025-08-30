"""Configuration file for pytest tests."""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import tempfile
from unittest.mock import Mock

import pytest


# Fixture for temporary directory
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

# Fixture for mock config
@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.folder_path = "./data_to_index"
    config.collection_name = "test-collection"
    config.current_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.chunk_size = 500
    config.chunk_overlap = 100
    config.device = "cpu"
    config.index_dense = True
    config.index_bm25 = False
    config.index_hybrid = False
    config.embedding_batch_size = 32
    config.indexing_batch_size = 50
    config.qdrant_url = "http://localhost:6333"
    config.sparse_embedding = "Qdrant/bm25"
    config.search_default_k = 5
    return config

# Fixture for sample documents
@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="This is a test document about artificial intelligence and machine learning.",
            metadata={"source": "test1.txt", "author": "Test Author"}
        ),
        Document(
            page_content="Another test document focusing on natural language processing and transformers.",
            metadata={"source": "test2.txt", "author": "Another Author"}
        )
    ]