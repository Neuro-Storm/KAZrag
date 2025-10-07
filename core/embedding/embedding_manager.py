"""Module for centralized embedding management."""

import asyncio
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

from config.config_manager import ConfigManager
from config.settings import Config

# Импортируем GGUF эмбеддер
try:
    from core.embedding.gguf_embeddings import GGUFEmbeddings
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("GGUF эмбеддер не доступен. Установите llama-cpp-python для работы с GGUF моделями.")

logger = logging.getLogger(__name__)

def get_embedding_model(model_name: str = None):
    """Получение модели эмбеддингов с автоматическим скачиванием в папку models."""
    config_manager = ConfigManager.get_instance()
    config = config_manager.get()
    
    if model_name is None:
        model_name = config.current_hf_model
    
    local_path = Path(config.local_models_path / "embeddings" / model_name)
    
    if local_path.exists():
        logger.info(f"Используется локальная модель эмбеддингов: {local_path}")
        return SentenceTransformer(
            str(local_path), 
            local_files_only=True,
            trust_remote_code=False
        )
    else:
        if config.auto_download_models:
            logger.info(f"Модель {model_name} не найдена локально. Скачивание в папку models...")
            
            try:
                # Создаем директорию, если она не существует
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Скачиваем модель напрямую в нашу папку
                downloaded_path = snapshot_download(
                    repo_id=model_name,
                    cache_dir=str(config.huggingface_cache_path),  # Временный кэш
                    local_dir=str(local_path),  # Куда сохранять модель
                    local_dir_use_symlinks=False,  # Не использовать символические ссылки
                    token=config.huggingface_token  # Для приватных моделей
                )
                
                logger.info(f"Модель успешно скачана в: {downloaded_path}")
                
                # Возвращаем модель из нашей папки
                return SentenceTransformer(
                    str(local_path), 
                    local_files_only=True,
                    trust_remote_code=False
                )
            except Exception as e:
                logger.error(f"Ошибка при скачивании модели {model_name}: {e}")
        
        # Fallback на кэш HuggingFace, если скачивание не удалось или отключено
        logger.info(f"Использование кэша HuggingFace для модели {model_name}")
        return SentenceTransformer(
            model_name, 
            cache_folder=str(config.huggingface_cache_path),
            local_files_only=config.use_local_only,
            trust_remote_code=False
        )

# Get singleton instance of ConfigManager
config_manager = ConfigManager.get_instance()


class EmbeddingError(Exception):
    """Custom exception for embedding management errors."""
    pass


class EmbeddingManager:
    """Centralized manager for embeddings with caching."""
    
    _instance: Optional['EmbeddingManager'] = None
    MAX_CACHE_SIZE = 1  # Максимум 1 модель в кэше одновременно
    
    def __init__(self):
        """Initialize EmbeddingManager."""
        self._embedder_cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()  # Синхронный лок для синхронизации доступа к кэшу
        
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
            encode_kwargs = {"batch_size": batch_size}
            
            # Если указан токен HuggingFace, добавляем его в model_kwargs
            if config.huggingface_token:
                model_kwargs["token"] = config.huggingface_token
            
            # Проверяем, содержит ли модель имя, которое требует trust_remote_code
            models_requiring_trust = [
                "ai-sage/giga-embeddings-instruct",
                "ai-sage/giga-embeddings",
                "giga-embeddings",
                "giga-embeddings-instruct",
                "jina-ai/jina-embeddings-v3",
                "mixedbread-ai/mxbai-embed-large-v1",
                "nomic-ai/nomic-embed-text-v1",
                "nomic-ai/nomic-embed-text-v1.5",
                "nomic-ai/nomic-embed-text-v1.5-f16"
            ]
            model_lower = model_name.lower()
            
            if any(model_to_trust in model_lower for model_to_trust in models_requiring_trust):
                model_kwargs["trust_remote_code"] = True
                logger.info(f"Включено trust_remote_code для модели {model_name}")

            # Если модель не найдена в кеше или batch_size измнился, создаем новую
            try:
                embedder = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )

                # Устанавливаем batch_size для последующего использования
                embedder._batch_size = batch_size  # Для отслеживания
                
                # Сохраняем модель в кэш
                self._embedder_cache[cache_key] = embedder
                
                return embedder
            except Exception as e:
                logger.error(f"Failed to create HuggingFaceEmbeddings for model {model_name}: {e}")
                # Повторный вызов ошибки с более подробным описанием
                raise
            
        except Exception as e:
            # Проверяем, является ли это ошибкой совместимости PyTorch/Torchvision
            error_str = str(e)
            if "torchvision::nms does not exist" in error_str or "operator torchvision" in error_str:
                logger.error("PyTorch/TorchVision compatibility issue detected.")
                logger.error("SOLUTION: Update PyTorch and TorchVision to compatible versions:")
                logger.error("For CPU: pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu")
                logger.error("For CUDA 11.8: pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118")
                logger.error("For CUDA 12.1: pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121")
                raise EmbeddingError(f"PyTorch/TorchVision compatibility error. Please update PyTorch and TorchVision to compatible versions: {e}")
            else:
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
        
        # Принудительно очищаем память PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
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
    
    def embed_query(self, text: str, config: Optional[Config] = None, device: Optional[str] = None) -> List[float]:
        """Генерация эмбеддинга для одного запроса.
        
        Args:
            text: Текст для кодирования
            config: Configuration object (optional, если не указан, используется конфиг по умолчанию)
            device: Device to use for embeddings (optional)
            
        Returns:
            List[float]: Эмбеддинг текста
            
        Raises:
            EmbeddingError: Если возникает ошибка при создании эмбеддинга
        """
        try:
            # Если конфиг не передан, получаем его из config_manager
            if config is None:
                config = config_manager.get()
            
            # Получаем эмбеддер
            embedder = self.get_embedder(config, device)
            
            # Проверяем, есть ли у эмбеддера метод embed_query
            if hasattr(embedder, 'embed_query'):
                return embedder.embed_query(text)
            else:
                # Если метода embed_query нет, используем embed_documents с одним текстом
                embeddings = embedder.embed_documents([text])
                return embeddings[0] if embeddings else []
                
        except Exception as e:
            logger.exception(f"Error generating embedding for query: {e}")
            raise EmbeddingError(f"Failed to generate embedding for query: {e}")
    
    def embed_texts(self, texts: List[str], config: Optional[Config] = None, device: Optional[str] = None) -> List[List[float]]:
        """Генерация эмбеддингов для списка текстов.
        
        Args:
            texts: Список текстов для кодирования
            config: Configuration object (optional, если не указан, используется конфиг по умолчанию)
            device: Device to use for embeddings (optional)
            
        Returns:
            List[List[float]]: Список эмбеддингов текстов
            
        Raises:
            EmbeddingError: Если возникает ошибка при создании эмбеддингов
        """
        try:
            # Если конфиг не передан, получаем его из config_manager
            if config is None:
                config = config_manager.get()
            
            # Получаем эмбеддер
            embedder = self.get_embedder(config, device)
            
            # Используем метод embed_documents для генерации эмбеддингов
            return embedder.embed_documents(texts)
                
        except Exception as e:
            logger.exception(f"Error generating embeddings for texts: {e}")
            raise EmbeddingError(f"Failed to generate embeddings for texts: {e}")

    async def aget_embedder(self, config: Config, device: Optional[str] = None) -> Embeddings:
        """Получает или создает кэшированный экземпляр эмбеддера (асинхронная версия).
        
        Args:
            config: Configuration object
            device: Device to use for embeddings (optional)
            
        Returns:
            Embeddings: Embedder instance
            
        Raises:
            EmbeddingError: If there's an error creating the embedder
        """
        # Обертка для вызова синхронного метода в отдельном потоке
        import asyncio
        import concurrent.futures
        
        def run_get_embedder():
            return self.get_embedder(config, device)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_get_embedder)
            return await asyncio.get_event_loop().run_in_executor(None, future.result)

    async def aembed_query(self, text: str, config: Config = None, device: Optional[str] = None) -> List[float]:
        """Асинхронная версия embed_query.
        
        Args:
            text: Текст для эмбеддинга
            config: Конфигурация (опционально, по умолчанию используется текущая)
            device: Устройство для эмбеддинга (опционально)
            
        Returns:
            List[float]: Вектор эмбеддинга
            
        Raises:
            EmbeddingError: If there's an error during embedding
        """
        import asyncio
        import concurrent.futures
        
        def run_embed_query():
            return self.embed_query(text, config, device)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_embed_query)
            return await asyncio.get_event_loop().run_in_executor(None, future.result)

    async def aembed_texts(self, texts: List[str], config: Config = None, device: Optional[str] = None) -> List[List[float]]:
        """Асинхронная версия embed_texts.
        
        Args:
            texts: Список текстов для эмбеддинга
            config: Конфигурация (опционально, по умолчанию используется текущая)
            device: Устройство для эмбеддинга (опционально)
            
        Returns:
            List[List[float]]: Список векторов эмбеддингов
            
        Raises:
            EmbeddingError: If there's an error during embedding
        """
        import asyncio
        import concurrent.futures
        
        def run_embed_texts():
            return self.embed_texts(texts, config, device)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_embed_texts)
            return await asyncio.get_event_loop().run_in_executor(None, future.result)