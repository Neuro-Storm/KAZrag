"""Module for centralized embedding management."""

import logging
import torch
from collections import OrderedDict
from typing import Optional, Dict, Tuple, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from config.config_manager import ConfigManager
from config.settings import Config

# Импортируем GGUF эмбеддер
try:
    from core.gguf_embeddings import GGUFEmbeddings
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("GGUF эмбеддер не доступен. Установите llama-cpp-python для работы с GGUF моделями.")

logger = logging.getLogger(__name__)

# Get singleton instance of ConfigManager
config_manager = ConfigManager.get_instance()


class EmbeddingError(Exception):
    """Custom exception for embedding management errors."""
    pass


class EmbeddingManager:
    """Centralized manager for embeddings with caching."""
    
    _instance: Optional['EmbeddingManager'] = None
    MAX_CACHE_SIZE = 3  # Максимум 3 модели в кэше одновременно
    
    def __init__(self):
        """Initialize EmbeddingManager."""
        self._embedder_cache: OrderedDict = OrderedDict()
        
    @classmethod
    def get_instance(cls) -> 'EmbeddingManager':
        """Get singleton instance of EmbeddingManager.
        
        Returns:
            EmbeddingManager: Singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_device(self, config_device: str) -> str:
        """Определяет устройство для индексации из конфига."""
        if config_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if config_device == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_search_device(self, search_device_param: str) -> str:
        """Определяет устройство для поиска из параметра формы."""
        if search_device_param == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def get_embedder(self, config: Config, device: Optional[str] = None) -> Embeddings:
        """Получает или создает кэшированный экземпляр эмбеддера.
        
        Args:
            config: Configuration object
            device: Device to use for embeddings (optional)
            
        Returns:
            Embeddings: Embedder instance
            
        Raises:
            EmbeddingError: If there's an error creating the embedder
        """
        try:
            model_name = config.current_hf_model
            batch_size = config.embedding_batch_size
            if device is None:
                device = self.get_device(config.device)
            
            # Проверяем, является ли модель GGUF
            if model_name.lower().endswith('.gguf'):
                if GGUF_AVAILABLE:
                    return self._get_gguf_embedder(config, device)
                else:
                    raise EmbeddingError("GGUF эмбеддер не доступен. Установите llama-cpp-python для работы с GGUF моделями.")
            
            # Ключ кэша - кортеж (модель, устройство)
            cache_key = (model_name, device)
            
            # Проверяем, есть ли в кэше модель с такими параметрами
            if cache_key in self._embedder_cache:
                # Проверяем, соответствует ли batch_size
                cached_embedder = self._embedder_cache[cache_key]
                if getattr(cached_embedder, "_batch_size", None) == batch_size:
                    # Перемещаем модель в начало OrderedDict (обновляем время последнего использования)
                    self._embedder_cache.move_to_end(cache_key, last=True)
                    return cached_embedder
            
            # Если кэш переполнен, удаляем самую старую запись
            if len(self._embedder_cache) >= self.MAX_CACHE_SIZE:
                # Удаляем самую старую запись (первую в OrderedDict)
                oldest_key, _ = self._embedder_cache.popitem(last=False)
                logger.info(f"Удалена старая модель из кэша: {oldest_key}")
            
            # Подготовка параметров для загрузки модели
            model_kwargs = {"device": device}
            # Если используем CUDA, указываем dtype через model_kwargs вместо обращения к приватным полям
            if device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16

            # Если модель не найдена в кеше или batch_size изменился, создаем новую
            embedder = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={"batch_size": batch_size}
            )

            # Устанавливаем batch_size для последующего использования
            embedder._batch_size = batch_size  # Для отслеживания
            
            # Сохраняем модель в кэш
            self._embedder_cache[cache_key] = embedder
            
            return embedder
            
        except Exception as e:
            logger.exception(f"Error creating embedder: {e}")
            raise EmbeddingError(f"Failed to create embedder: {e}")
    
    def _get_gguf_embedder(self, config: Config, device: str) -> Embeddings:
        """Получает GGUF эмбеддер.
        
        Args:
            config: Configuration object
            device: Device to use for embeddings
            
        Returns:
            Embeddings: GGUF embedder instance
        """
        if not GGUF_AVAILABLE:
            raise EmbeddingError("GGUF эмбеддер не доступен")
            
        model_name = config.current_hf_model
        
        # Ключ кэша - кортеж (модель, устройство)
        cache_key = (model_name, device)
        
        # Проверяем, есть ли в кэше модель с такими параметрами
        if cache_key in self._embedder_cache:
            # Перемещаем модель в начало OrderedDict (обновляем время последнего использования)
            self._embedder_cache.move_to_end(cache_key, last=True)
            return self._embedder_cache[cache_key]
        
        # Если кэш переполнен, удаляем самую старую запись
        if len(self._embedder_cache) >= self.MAX_CACHE_SIZE:
            # Удаляем самую старую запись (первую в OrderedDict)
            oldest_key, _ = self._embedder_cache.popitem(last=False)
            logger.info(f"Удалена старая модель из кэша: {oldest_key}")
        
        # Создаем новый GGUF эмбеддер
        # Определяем путь к модели (предполагаем, что он находится в каталоге моделей)
        model_path = model_name  # Предполагаем, что model_name уже содержит полный путь
        
        embedder = GGUFEmbeddings(
            model_path=model_path,
            device=device
        )
        
        # Сохраняем модель в кэш
        self._embedder_cache[cache_key] = embedder
        
        return embedder
    
    def clear_cache(self) -> None:
        """Очищает кэш эмбеддеров."""
        self._embedder_cache.clear()
        logger.info("Кэш эмбеддеров очищен")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Получает информацию о кэше эмбеддеров.
        
        Returns:
            Dict[str, Any]: Cache information
        """
        return {
            "cache_size": len(self._embedder_cache),
            "max_cache_size": self.MAX_CACHE_SIZE,
            "cached_models": list(self._embedder_cache.keys())
        }