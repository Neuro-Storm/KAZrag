"""Модуль для управления векторизаторами (эмбеддерами)."""

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import load_config

# Импортируем GGUF эмбеддер
try:
    from core.gguf_embeddings import get_gguf_embedder, GGUFEmbeddings
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    print("GGUF эмбеддер не доступен. Установите ctransformers для работы с GGUF моделями.")

# --- Кэш для embedder'а ---
# Используем словарь для хранения моделей с ключом (модель, устройство)
_dense_embedder_cache = {}


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
def get_dense_embedder(config, device=None):
    """Получает или создает кэшированный экземпляр эмбеддера."""
    global _dense_embedder_cache
    model_name = config["current_hf_model"]
    batch_size = config.get("embedding_batch_size", 32)
    if device is None:
        device = get_device(config["device"])
    
    # Проверяем, является ли модель GGUF
    if ("-gguf" in model_name.lower() or ".gguf" in model_name.lower()) and GGUF_AVAILABLE:
        return get_gguf_embedder(config, device)
    
    # Ключ кэша - кортеж (модель, устройство)
    cache_key = (model_name, device)
    
    # Проверяем, есть ли в кэше модель с такими параметрами
    if cache_key in _dense_embedder_cache:
        # Проверяем, соответствует ли batch_size
        cached_embedder = _dense_embedder_cache[cache_key]
        if getattr(cached_embedder, "_batch_size", None) == batch_size:
            return cached_embedder
    
    # Подготовка параметров для загрузки модели
    model_kwargs = {"device": device}
    
    # Если модель не найдена в кэше или batch_size изменился, создаем новую
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"batch_size": batch_size}
    )
    
    # Установка torch_dtype после создания модели, если необходимо
    if device == "cuda":
        embedder._client[0].half()  # Преобразование весов модели в float16
    embedder._batch_size = batch_size  # Для отслеживания
    
    # Сохраняем модель в кэш
    _dense_embedder_cache[cache_key] = embedder
    
    return embedder