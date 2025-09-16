"""Common fixtures for all tests."""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_project_structure():
    """Mock the project structure to avoid import errors."""
    # Mock core modules that might not be available during testing
    with patch('core.document_loader', create=True), \
         patch('core.text_splitter', create=True), \
         patch('core.embedding_manager', create=True), \
         patch('core.indexer', create=True), \
         patch('core.search.searcher', create=True):
        yield