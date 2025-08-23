"""Модуль для управления векторизаторами (эмбеддерами)."""

import torch
import logging
from collections import OrderedDict
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import Config

# Импортируем GGUF эмбеддер
try:
    from core.gguf_embeddings import get_gguf_embedder
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    # raise ImportError("GGUF эмбеддер не доступен. Установите ctransformers для работы с GGUF моделями.")
    # Временно оставляем print для совместимости, но в будущем лучше использовать raise

logger = logging.getLogger(__name__)

# --- Кэш для embedder'а ---
# Используем словарь для хранения моделей с ключом (модель, устройство)
# Добавляем ограничение по размеру кэша для предотвращения утечек памяти
MAX_CACHE_SIZE = 3  # Максимум 3 модели в кэше одновременно

_dense_embedder_cache = OrderedDict()


@lru_cache
def get_device(config_device: str) -> str:
    """Определяет устройство для индексации из конфига."""
    if config_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if config_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_search_device(search_device_param: str) -> str:
    """Определяет устройство для поиска из параметра формы."""
    if search_device_param == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --- Кэширование HuggingFaceEmbeddings ---
def get_dense_embedder(config: Config, device=None):
    """Получает или создает кэшированный экземпляр эмбеддера."""
    global _dense_embedder_cache
    model_name = config.current_hf_model
    batch_size = config.embedding_batch_size
    if device is None:
        device = get_device(config.device)
    
    # Проверяем, является ли модель GGUF
    if model_name.lower().endswith('.gguf'):
        if GGUF_AVAILABLE:
            return get_gguf_embedder(config, device)
        else:
            raise ImportError("GGUF эмбеддер не доступен. Установите llama-cpp-python для работы с GGUF моделями.")
    
    # Ключ кэша - кортеж (модель, устройство)
    cache_key = (model_name, device)
    
    # Проверяем, есть ли в кэше модель с такими параметрами
    if cache_key in _dense_embedder_cache:
        # Проверяем, соответствует ли batch_size
        cached_embedder = _dense_embedder_cache[cache_key]
        if getattr(cached_embedder, "_batch_size", None) == batch_size:
            # Перемещаем модель в начало OrderedDict (обновляем время последнего использования)
            _dense_embedder_cache.move_to_end(cache_key, last=True)
            return cached_embedder
    
    # Если кэш переполнен, удаляем самую старую запись
    if len(_dense_embedder_cache) >= MAX_CACHE_SIZE:
        # Удаляем самую старую запись (первую в OrderedDict)
        oldest_key, _ = _dense_embedder_cache.popitem(last=False)
        logger.info(f"Удалена старая модель из кэша: {oldest_key}")
    
    # Подготовка параметров для загрузки модели
    model_kwargs = {"device": device}
    # Если используем CUDA, указываем dtype через model_kwargs вместо обращения к приватным полям
    if device == "cuda":
        import torch as _torch
        model_kwargs["torch_dtype"] = _torch.float16

    # Если модель не найдена в кеше или batch_size изменился, создаем новую
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"batch_size": batch_size}
    )

    # Устанавливаем batch_size для последующего использования
    embedder._batch_size = batch_size  # Для отслеживания
    
    # Сохраняем модель в кэш
    _dense_embedder_cache[cache_key] = embedder
    
    return embedder


def clear_embedder_cache():
    """Очищает кэш эмбеддеров."""
    global _dense_embedder_cache
    _dense_embedder_cache.clear()
    logger.info("Кэш эмбеддеров очищен")