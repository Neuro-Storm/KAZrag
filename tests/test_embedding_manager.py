"""Unit tests for the EmbeddingManager class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.embedding_manager import EmbeddingManager, EmbeddingError
from config.settings import Config


class TestEmbeddingManager:
    """Test suite for EmbeddingManager class."""

    def test_singleton_instance(self):
        """Test that EmbeddingManager is a singleton."""
        instance1 = EmbeddingManager.get_instance()
        instance2 = EmbeddingManager.get_instance()
        assert instance1 is instance2

    def test_get_device_cuda_available(self):
        """Test getting device when CUDA is available."""
        manager = EmbeddingManager()
        
        with patch('torch.cuda.is_available', return_value=True):
            device = manager.get_device("cuda")
            assert device == "cuda"
            
            device = manager.get_device("auto")
            assert device == "cuda"

    def test_get_device_cpu_only(self):
        """Test getting device when only CPU is available."""
        manager = EmbeddingManager()
        
        with patch('torch.cuda.is_available', return_value=False):
            device = manager.get_device("cuda")
            assert device == "cpu"
            
            device = manager.get_device("cpu")
            assert device == "cpu"
            
            device = manager.get_device("auto")
            assert device == "cpu"

    def test_get_search_device(self):
        """Test getting search device."""
        manager = EmbeddingManager()
        
        with patch('torch.cuda.is_available', return_value=True):
            device = manager.get_search_device("cuda")
            assert device == "cuda"
            
        device = manager.get_search_device("cpu")
        assert device == "cpu"

    def test_get_embedder_hf_model(self):
        """Test getting HuggingFace embedder."""
        manager = EmbeddingManager()
        
        config = Config()
        config.current_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding_batch_size = 32
        config.device = "cpu"
        
        with patch('core.embedding_manager.HuggingFaceEmbeddings') as mock_hf_embeddings:
            mock_embedder = Mock()
            mock_hf_embeddings.return_value = mock_embedder
            
            embedder = manager.get_embedder(config, "cpu")
            assert embedder is mock_embedder
            mock_hf_embeddings.assert_called_once()

    def test_get_embedder_gguf_model(self):
        """Test getting GGUF embedder."""
        manager = EmbeddingManager()
        
        config = Config()
        config.current_hf_model = "model.gguf"
        config.embedding_batch_size = 32
        config.device = "cpu"
        
        with patch('core.embedding_manager.GGUF_AVAILABLE', True):
            with patch('core.embedding_manager.GGUFEmbeddings') as mock_gguf_embeddings:
                mock_embedder = Mock()
                mock_gguf_embeddings.return_value = mock_embedder
                
                embedder = manager.get_embedder(config, "cpu")
                assert embedder is mock_embedder
                mock_gguf_embeddings.assert_called_once()

    def test_get_embedder_gguf_not_available(self):
        """Test getting GGUF embedder when not available."""
        manager = EmbeddingManager()
        
        config = Config()
        config.current_hf_model = "model.gguf"
        config.embedding_batch_size = 32
        config.device = "cpu"
        
        with patch('core.embedding_manager.GGUF_AVAILABLE', False):
            with pytest.raises(EmbeddingError):
                manager.get_embedder(config, "cpu")

    def test_embedder_caching(self):
        """Test embedder caching functionality."""
        manager = EmbeddingManager()
        
        config = Config()
        config.current_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding_batch_size = 32
        config.device = "cpu"
        
        with patch('core.embedding_manager.HuggingFaceEmbeddings') as mock_hf_embeddings:
            mock_embedder1 = Mock()
            mock_embedder2 = Mock()
            mock_hf_embeddings.side_effect = [mock_embedder1, mock_embedder2]
            
            # Get embedder first time
            embedder1 = manager.get_embedder(config, "cpu")
            assert embedder1 is mock_embedder1
            
            # Get embedder second time with same parameters - should use cache
            embedder2 = manager.get_embedder(config, "cpu")
            assert embedder2 is mock_embedder1  # Same instance from cache
            assert mock_hf_embeddings.call_count == 1  # Should not be called again
            
            # Get embedder with different device - should create new instance
            embedder3 = manager.get_embedder(config, "cuda")
            assert embedder3 is mock_embedder2  # New instance
            assert mock_hf_embeddings.call_count == 2  # Should be called again

    def test_clear_cache(self):
        """Test clearing embedder cache."""
        manager = EmbeddingManager()
        
        # Add something to cache
        manager._embedder_cache[("model", "cpu")] = Mock()
        
        # Clear cache
        manager.clear_cache()
        
        # Cache should be empty
        assert len(manager._embedder_cache) == 0

    def test_get_cache_info(self):
        """Test getting cache information."""
        manager = EmbeddingManager()
        
        # Add something to cache
        manager._embedder_cache[("model1", "cpu")] = Mock()
        manager._embedder_cache[("model2", "cuda")] = Mock()
        
        cache_info = manager.get_cache_info()
        assert cache_info["cache_size"] == 2
        assert cache_info["max_cache_size"] == 3
        assert len(cache_info["cached_models"]) == 2

    def test_embedder_error_handling(self):
        """Test handling of EmbeddingError exceptions."""
        manager = EmbeddingManager()
        
        config = Config()
        config.current_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding_batch_size = 32
        config.device = "cpu"
        
        with patch('core.embedding_manager.HuggingFaceEmbeddings', side_effect=Exception("Model load failed")):
            with pytest.raises(EmbeddingError):
                manager.get_embedder(config, "cpu")