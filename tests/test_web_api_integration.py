"""Integration tests for the web API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
from app.app_factory import create_app
from config.settings import Config
from core.searcher import search_in_collection
from langchain_core.documents import Document


class TestWebAPIIntegration:
    """Integration tests for the web API."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI app."""
        return create_app()

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return TestClient(app)

    def test_search_endpoint_success(self, client):
        """Test successful search endpoint call."""
        # Prepare search data
        search_data = {
            "query": "artificial intelligence",
            "collection": "test-collection",
            "search_device": "cpu",
            "k": "5",
            "search_type": "dense"
        }
        
        # Mock search results
        mock_results = [
            (Document(page_content="AI is transforming industries", metadata={"source": "doc1.txt"}), 0.95),
            (Document(page_content="Machine learning applications", metadata={"source": "doc2.txt"}), 0.85)
        ]
        
        with patch('web.search_app.search_in_collection') as mock_search:
            with patch('web.search_app.get_cached_collections', return_value=['test-collection']):
                mock_search.return_value = (mock_results, None)
                
                # Make the request
                response = client.post("/api/search/", data=search_data)
                
                # Check response
                assert response.status_code == 200
                assert "AI is transforming industries" in response.text
                assert "Machine learning applications" in response.text

    def test_search_endpoint_with_metadata_filter(self, client):
        """Test search endpoint with metadata filter."""
        # Prepare search data with metadata filter
        search_data = {
            "query": "data science",
            "collection": "test-collection",
            "search_device": "cpu",
            "k": "3",
            "search_type": "dense",
            "filter_author": "John Doe",
            "filter_source": "research.pdf"
        }
        
        # Mock search results
        mock_results = [
            (Document(page_content="Data science research paper", metadata={
                "source": "research.pdf", 
                "author": "John Doe"
            }), 0.92)
        ]
        
        with patch('web.search_app.search_in_collection') as mock_search:
            with patch('web.search_app.get_cached_collections', return_value=['test-collection']):
                mock_search.return_value = (mock_results, None)
                
                # Make the request
                response = client.post("/api/search/", data=search_data)
                
                # Check response
                assert response.status_code == 200
                assert "Data science research paper" in response.text

    def test_search_endpoint_hybrid_search(self, client):
        """Test search endpoint with hybrid search."""
        # Prepare search data for hybrid search
        search_data = {
            "query": "neural networks",
            "collection": "test-collection",
            "search_device": "cpu",
            "k": "5",
            "search_type": "hybrid"
        }
        
        # Mock search results
        mock_results = [
            (Document(page_content="Deep learning with neural networks", metadata={"source": "dl.pdf"}), 0.98)
        ]
        
        with patch('web.search_app.search_in_collection') as mock_search:
            with patch('web.search_app.get_cached_collections', return_value=['test-collection']):
                mock_search.return_value = (mock_results, None)
                
                # Make the request
                response = client.post("/api/search/", data=search_data)
                
                # Check response
                assert response.status_code == 200
                assert "Deep learning with neural networks" in response.text

    def test_search_endpoint_ajax_request(self, client):
        """Test search endpoint with AJAX request."""
        # Prepare search data
        search_data = {
            "query": "machine learning",
            "collection": "test-collection",
            "search_device": "cpu",
            "k": "5",
            "search_type": "dense"
        }
        
        # Mock search results
        mock_results = [
            (Document(page_content="ML algorithms overview", metadata={"source": "ml.txt"}), 0.9)
        ]
        
        with patch('web.search_app.search_in_collection') as mock_search:
            with patch('web.search_app.get_cached_collections', return_value=['test-collection']):
                mock_search.return_value = (mock_results, None)
                
                # Make the AJAX request
                response = client.post(
                    "/api/search/", 
                    data=search_data,
                    headers={"X-Requested-With": "XMLHttpRequest"}
                )
                
                # Check response
                assert response.status_code == 200
                # For AJAX requests, we expect HTML content with results
                assert "ML algorithms overview" in response.text

    def test_search_endpoint_error_handling(self, client):
        """Test search endpoint error handling."""
        # Prepare search data
        search_data = {
            "query": "test query",
            "collection": "test-collection",
            "search_device": "cpu",
            "k": "5",
            "search_type": "dense"
        }
        
        with patch('web.search_app.search_in_collection') as mock_search:
            with patch('web.search_app.get_cached_collections', return_value=['test-collection']):
                # Mock search error
                mock_search.return_value = ([], "Search service unavailable")
                
                # Make the request
                response = client.post("/api/search/", data=search_data)
                
                # Check response
                assert response.status_code == 200
                assert "Ошибка" in response.text

    def test_settings_endpoint_success(self, client):
        """Test settings endpoint with successful authentication."""
        with patch('web.admin_app.verify_admin_access_from_form', return_value="admin"):
            with patch('web.admin_app.get_cached_collections', return_value=['test-collection']):
                with patch('os.environ.get', return_value="./models/fastembed_cache"):
                    # Make the request
                    response = client.get("/api/admin/settings/")
                    
                    # Check response
                    assert response.status_code == 200
                    assert "Настройки" in response.text

    def test_settings_endpoint_authentication_error(self, client):
        """Test settings endpoint with authentication error."""
        with patch('web.admin_app.verify_admin_access_from_form') as mock_auth:
            mock_auth.side_effect = Exception("Authentication failed")
            
            # Make the request
            response = client.get("/api/admin/settings/")
            
            # Check response - should redirect or return error
            assert response.status_code in [401, 403, 303, 200]

    def test_update_settings_endpoint(self, client):
        """Test update settings endpoint."""
        # Prepare settings data
        settings_data = {
            "action": "save_index_settings",
            "folder_path": "./test_data",
            "collection_name": "test-collection",
            "hf_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": "500",
            "chunk_overlap": "100",
            "device": "cpu",
            "indexing_type": "dense"
        }
        
        with patch('web.admin_app.verify_admin_access_from_form', return_value="admin"):
            with patch('web.admin_app.config_manager') as mock_config_manager:
                mock_config = Config()
                mock_config_manager.get_instance().get.return_value = mock_config
                
                with patch('web.admin_app.get_cached_collections', return_value=['test-collection']):
                    # Make the request
                    response = client.post(
                        "/api/admin/update-settings",
                        data=settings_data,
                        headers={"X-Requested-With": "XMLHttpRequest"}
                    )
                    
                    # Check response
                    assert response.status_code == 200
                    response_data = response.json()
                    assert response_data["status"] == "saved"

    def test_delete_collection_endpoint(self, client):
        """Test delete collection endpoint."""
        # Prepare delete data
        delete_data = {
            "collection_name": "test-collection"
        }
        
        with patch('web.admin_app.verify_admin_access_from_form', return_value="admin"):
            with patch('web.admin_app.get_cached_collections', return_value=['test-collection']):
                with patch('web.admin_app.refresh_collections_cache') as mock_refresh:
                    # Make the request
                    response = client.post(
                        "/api/admin/settings/delete-collection",
                        data=delete_data,
                        headers={"X-Requested-With": "XMLHttpRequest"}
                    )
                    
                    # Check response
                    assert response.status_code == 200
                    response_data = response.json()
                    assert "deleted_" in response_data["status"]

    def test_run_indexing_endpoint(self, client):
        """Test run indexing endpoint."""
        with patch('web.admin_app.verify_admin_access_from_form', return_value="admin"):
            with patch('web.admin_app.BackgroundTasks') as mock_background_tasks:
                mock_task_instance = Mock()
                mock_background_tasks.return_value = mock_task_instance
                
                with patch('web.admin_app.run_indexing_logic') as mock_run_indexing:
                    # Make the request
                    response = client.post(
                        "/api/admin/run-indexing",
                        headers={"X-Requested-With": "XMLHttpRequest"}
                    )
                    
                    # Check response
                    assert response.status_code == 200
                    response_data = response.json()
                    assert response_data["status"] == "indexing_started"

    def test_process_pdfs_endpoint(self, client):
        """Test process PDFs endpoint."""
        with patch('web.admin_app.verify_admin_access_from_form', return_value="admin"):
            with patch('web.admin_app.BackgroundTasks') as mock_background_tasks:
                mock_task_instance = Mock()
                mock_background_tasks.return_value = mock_task_instance
                
                with patch('web.admin_app.run_pdf_processing_from_config') as mock_process_pdfs:
                    # Make the request
                    response = client.post(
                        "/api/admin/process-pdfs",
                        headers={"X-Requested-With": "XMLHttpRequest"}
                    )
                    
                    # Check response
                    assert response.status_code == 200
                    response_data = response.json()
                    assert response_data["status"] == "processing_started"

    def test_process_files_endpoint(self, client):
        """Test process files endpoint."""
        with patch('web.admin_app.verify_admin_access_from_form', return_value="admin"):
            with patch('web.admin_app.BackgroundTasks') as mock_background_tasks:
                mock_task_instance = Mock()
                mock_background_tasks.return_value = mock_task_instance
                
                with patch('web.admin_app.run_multi_format_processing_from_config') as mock_process_files:
                    # Make the request
                    response = client.post(
                        "/api/admin/process-files",
                        headers={"X-Requested-With": "XMLHttpRequest"}
                    )
                    
                    # Check response
                    assert response.status_code == 200
                    response_data = response.json()
                    assert response_data["status"] == "processing_started"

    def test_root_redirect(self, client):
        """Test root endpoint redirects to search."""
        response = client.get("/")
        assert response.status_code == 200  # Follows redirect
        # Should contain search page content
        assert "Поиск" in response.text or "Search" in response.text

    def test_settings_redirect(self, client):
        """Test settings redirect endpoint."""
        response = client.get("/settings")
        assert response.status_code == 200  # Follows redirect
        # Should contain settings page content
        assert "Настройки" in response.text or "Settings" in response.text


class TestWebAPIErrorHandling:
    """Tests for error handling in web API."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI app."""
        return create_app()

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return TestClient(app)

    def test_search_endpoint_internal_error(self, client):
        """Test search endpoint handling of internal errors."""
        # Prepare search data
        search_data = {
            "query": "test query",
            "collection": "test-collection",
            "search_device": "cpu",
            "k": "5",
            "search_type": "dense"
        }
        
        with patch('web.search_app.search_in_collection') as mock_search:
            with patch('web.search_app.get_cached_collections', return_value=['test-collection']):
                # Mock internal server error
                mock_search.side_effect = Exception("Internal server error")
                
                # Make the request
                response = client.post("/api/search/", data=search_data)
                
                # Check response
                assert response.status_code == 200  # Still returns 200 but with error message
                assert "Ошибка" in response.text

    def test_settings_endpoint_config_error(self, client):
        """Test settings endpoint handling of configuration errors."""
        with patch('web.admin_app.verify_admin_access_from_form', return_value="admin"):
            with patch('web.admin_app.get_cached_collections', side_effect=Exception("Config error")):
                # Make the request
                response = client.get("/api/admin/settings/")
                
                # Check response
                assert response.status_code == 200  # Still returns 200 but with error message
                assert "Ошибка" in response.text

    def test_update_settings_endpoint_validation_error(self, client):
        """Test update settings endpoint handling of validation errors."""
        # Prepare invalid settings data
        settings_data = {
            "action": "save_index_settings",
            "folder_path": "",  # Empty folder path should cause validation error
            "collection_name": "test-collection"
        }
        
        with patch('web.admin_app.verify_admin_access_from_form', return_value="admin"):
            with patch('web.admin_app.config_manager') as mock_config_manager:
                mock_config = Config()
                mock_config_manager.get_instance().get.return_value = mock_config
                
                # Make the request
                response = client.post(
                    "/api/admin/update-settings",
                    data=settings_data,
                    headers={"X-Requested-With": "XMLHttpRequest"}
                )
                
                # Check response
                assert response.status_code == 200
                response_data = response.json()
                assert response_data["status"] == "error"