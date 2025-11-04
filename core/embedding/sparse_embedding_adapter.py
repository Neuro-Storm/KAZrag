"""Адаптер для разреженных эмбеддингов с native BM25."""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector
from config.config_manager import ConfigManager

# Попробуем импортировать Embeddings из langchain для совместимости
try:
    from langchain_core.embeddings import Embeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    class Embeddings:
        """Заглушка для Embeddings если langchain недоступен"""
        pass

logger = logging.getLogger(__name__)

def get_sparse_model(model_name: str = None):
    """Получение sparse модели с автоматическим скачиванием в папку models."""
    config_manager = ConfigManager.get_instance()
    config = config_manager.get()
    
    if model_name is None:
        model_name = config.sparse_embedding
    
    local_path = Path(config.local_models_path / "fastembed" / model_name)
    
    if local_path.exists():
        logger.info(f"Используется локальная sparse модель: {local_path}")
        try:
            return SparseTextEmbedding(
                model_name=model_name,
                cache_dir=str(local_path),
                local_files_only=True
            )
        except Exception as e:
            logger.warning(f"Ошибка использования локальной sparse модели: {e}. Fallback на стандартный кэш.")
            return SparseTextEmbedding(
                model_name=model_name,
                cache_dir=str(config.fastembed_cache_path)
            )
    else:
        logger.info(f"Sparse модель {model_name} не найдена локально. Скачивание...")
        
        try:
            # Создаем директорию, если она не существует
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Скачиваем модель
            model = SparseTextEmbedding(
                model_name=model_name,
                cache_dir=str(local_path)
            )
            
            # Инициализация: encode один текст для preload
            try:
                _ = list(model.encode(["test text"]))[0]  # SparseEmbedding object
                logger.info(f"Sparse модель успешно инициализирована в: {local_path}")
            except Exception as init_e:
                logger.warning(f"Инициализация sparse модели failed: {init_e}")
            
            return model
        except Exception as e:
            logger.error(f"Ошибка при скачивании sparse модели {model_name}: {e}")
            # Возвращаем модель с стандартным кэшем
            return SparseTextEmbedding(
                model_name=model_name,
                cache_dir=str(config.fastembed_cache_path)
            )


class SparseEmbeddingAdapter(Embeddings):
    """Адаптер для sparse эмбеддингов, совместимый с langchain Embeddings интерфейсом."""
    
    def __init__(self, config=None):
        """Инициализирует адаптер."""
        self.config = config or ConfigManager.get_instance().get()
        if not self.config.use_bm25:
            raise ValueError("BM25 not enabled in config")

    def encode(self, texts: List[str], return_sparse: bool = True) -> List[Dict[str, Any]]:
        """
        Генерирует sparse vectors для текстов (native BM25-like: токены + TF weights).
        Returns raw dict for Qdrant JSON: {"indices": [int], "values": [float]}.
        
        Args:
            texts: List of texts.
            return_sparse: Always True for sparse.
            
        Returns:
            List of dict (for SparseVector).
        """
        sparse_vectors = []
        for text in texts:
            # Advanced tokenization to better match Qdrant native BM25
            # Use more sophisticated tokenization similar to what Qdrant uses
            tokens = self._tokenize_text(text)
            tokens = [t for t in tokens if len(t) >= self.config.bm25_min_token_len]
            
            # TF weights (frequency)
            token_counts = Counter(tokens)
            total_terms = len(tokens) or 1  # Avoid div/0
            indices = [abs(hash(token)) % (2**31 - 1) for token in token_counts.keys()]  # Positive unique-ish ints
            values = [count / total_terms for count in token_counts.values()]
            
            # Ensure unique/sorted (Qdrant req)
            unique_pairs = sorted(set(zip(indices, values)), key=lambda x: x[0])
            indices = [idx for idx, _ in unique_pairs]
            values = [val for _, val in unique_pairs]
            
            # Создаем объект SparseVector как ожидает Qdrant/langchain
            from qdrant_client.models import SparseVector
            sparse_vector = SparseVector(indices=indices, values=values)
            sparse_vectors.append(sparse_vector)
        
        logger.debug(f"Generated {len(sparse_vectors)} sparse vectors for BM25")
        return sparse_vectors

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Advanced tokenization that better matches Qdrant native BM25.
        Qdrant typically uses word-level tokenization with some normalization.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Use regex to find word-like tokens (letters, numbers, and hyphens within words)
        # This should better match the Qdrant native tokenization
        tokens = re.findall(r'\b\w+(?:-\w+)*\b', text)
        
        return tokens

    def embed_query(self, query: str) -> Any:  # Returns SparseVector
        """Single query embedding."""
        from qdrant_client.models import SparseVector
        # Encode returns a list, so get the first element
        encoded_result = self.encode([query], return_sparse=True)[0]
        # If it's already a SparseVector, return it, otherwise create one from dict
        if hasattr(encoded_result, 'indices') and hasattr(encoded_result, 'values'):
            return encoded_result
        else:  # It's a dict, so create SparseVector from it
            return SparseVector(indices=encoded_result["indices"], values=encoded_result["values"])
    
    def embed_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate sparse vectors for multiple documents."""
        return self.encode(texts, return_sparse=True)