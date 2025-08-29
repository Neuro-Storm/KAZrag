"""Модуль для управления векторизаторами (эмбеддерами)."""

import torch
import logging
from config.settings import Config
from core.embedding.embedding_manager import EmbeddingManager

# Импортируем GGUF эмбеддер
try:
    from core.embedding.gguf_embeddings import GGUFEmbeddings
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    # raise ImportError("GGUF эмбеддер не доступен. Установите ctransformers для работы с GGUF моделями.")
    # Временно оставляем print для совместимости, но в будущем лучше использовать raise

logger = logging.getLogger(__name__)

# Get singleton instance of EmbeddingManager
embedding_manager = EmbeddingManager.get_instance()


def get_device(config_device: str) -> str:
    """Определяет устройство для индексации из конфига."""
    return embedding_manager.get_device(config_device)


def get_search_device(search_device_param: str) -> str:
    """Определяет устройство для поиска из параметра формы."""
    return embedding_manager.get_search_device(search_device_param)


# --- Кэширование HuggingFaceEmbeddings ---
def get_dense_embedder(config: Config, device=None):
    """Получает или создает кэшированный экземпляр эмбеддера."""
    return embedding_manager.get_embedder(config, device)


def clear_embedder_cache():
    """Очищает кэш эмбеддеров."""
    embedding_manager.clear_cache()