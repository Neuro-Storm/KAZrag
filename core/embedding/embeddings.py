"""Модуль для управления векторизаторами (эмбеддерами)."""

import logging
from typing import List, Optional

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
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # Если уже есть запущенный loop, создаем задачу
        import concurrent.futures
        import threading
        
        def run_get_embedder():
            return asyncio.run(embedding_manager.get_embedder(config, device))
        
        # Выполняем в отдельном потоке с новым event loop
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_get_embedder)
            return future.result()
    except RuntimeError:
        # Нет запущенного loop, можем использовать asyncio.run
        return asyncio.run(embedding_manager.get_embedder(config, device))


async def aget_dense_embedder(config: Config, device=None):
    """Асинхронная версия получения эмбеддера."""
    return await embedding_manager.get_embedder(config, device)


def embed_query_sync(text: str, config: Config = None, device = None):
    """Синхронная версия embed_query для использования в синхронных контекстах."""
    return embedding_manager.embed_query(text, config, device)


def embed_texts_sync(texts: List[str], config: Config = None, device = None):
    """Синхронная версия embed_texts для использования в синхронных контекстах."""
    return embedding_manager.embed_texts(texts, config, device)


def clear_embedder_cache():
    """Очищает кэш эмбеддеров."""
    embedding_manager.clear_cache()