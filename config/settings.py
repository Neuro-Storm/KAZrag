"""Модуль для работы с конфигурацией приложения."""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONFIG_FILE = Path("config.json")

# --- Кэш для конфигурации ---
_cached_config: Optional['Config'] = None
_cached_config_time = 0

# Значения по умолчанию для констант
DEFAULT_MEMORY_THRESHOLD = 500 * 1024 * 1024  # 500MB
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_WAIT_TIME = 2
DEFAULT_SEARCH_DEFAULT_K = 5
DEFAULT_INDEXING_DEFAULT_BATCH_SIZE = 32
DEFAULT_MINERU_SUBPROCESS_TIMEOUT = 600
DEFAULT_GGUF_MODEL_N_CTX = 4096
DEFAULT_CACHE_TTL = 60


class Config(BaseModel):
    """Модель конфигурации приложения."""
    
    # Настройки индексации
    folder_path: str = "./data_to_index"
    collection_name: str = "final-dense-collection"
    current_hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_model_history: List[str] = Field(default_factory=lambda: ["sentence-transformers/all-MiniLM-L6-v2"])
    chunk_size: int = 500
    chunk_overlap: int = 100
    device: str = "auto"
    use_dense_vectors: bool = True
    # Новые отдельные режимы индексации
    index_dense: bool = True
    index_bm25: bool = False
    index_hybrid: bool = False
    is_indexed: bool = False
    embedding_batch_size: int = Field(default=32, ge=1)  # Минимум 1
    indexing_batch_size: int = Field(default=50, ge=1)   # Минимум 1
    force_recreate: bool = True
    memory_threshold: int = DEFAULT_MEMORY_THRESHOLD  # 500MB по умолчанию
    sparse_embedding: Optional[str] = "Qdrant/bm25"
    use_hybrid: bool = False
    
    # Настройки Qdrant
    qdrant_url: str = DEFAULT_QDRANT_URL
    
    # Настройки MinerU
    mineru_input_pdf_dir: str = "./pdfs_to_process"
    mineru_output_md_dir: str = "./data_to_index"
    mineru_enable_formula_parsing: bool = False
    mineru_enable_table_parsing: bool = False
    mineru_model_source: str = "huggingface"
    mineru_models_dir: str = ""
    mineru_backend: str = "pipeline"
    mineru_method: str = "auto"
    mineru_lang: str = "east_slavic"
    mineru_sglang_url: str = ""
    mineru_subprocess_timeout: int = DEFAULT_MINERU_SUBPROCESS_TIMEOUT  # Таймаут для subprocess вызова mineru в секундах
    
    # Настройки кэширования
    config_cache_ttl: int = Field(default=DEFAULT_CACHE_TTL, ge=1)  # Минимум 1 секунда
    qdrant_client_cache_ttl: int = Field(default=DEFAULT_CACHE_TTL, ge=1)  # Минимум 1 секунда
    collections_cache_ttl: int = Field(default=DEFAULT_CACHE_TTL, ge=1)  # Минимум 1 секунда
    
    # Настройки GGUF моделей
    gguf_model_n_ctx: int = Field(default=DEFAULT_GGUF_MODEL_N_CTX, ge=1)  # Минимум 1
    
    # Настройки поиска
    search_default_k: int = Field(default=DEFAULT_SEARCH_DEFAULT_K, ge=1)  # Минимум 1 результат
    
    # Настройки подключения к Qdrant
    qdrant_retry_attempts: int = Field(default=DEFAULT_RETRY_ATTEMPTS, ge=1)  # Минимум 1 попытка
    qdrant_retry_wait_time: int = Field(default=DEFAULT_RETRY_WAIT_TIME, ge=1)  # Минимум 1 секунда
    
    # Настройки индексации документов
    indexing_default_batch_size: int = Field(default=DEFAULT_INDEXING_DEFAULT_BATCH_SIZE, ge=1)  # Минимум 1


def load_config() -> Config:
    """Загружает конфигурацию из файла config.json.
    
    Если файл не существует, создает его с настройками по умолчанию.
    """
    global _cached_config, _cached_config_time
    
    from pydantic import ValidationError
    
    # Простая проверка времени без загрузки всего конфига
    current_time = time.time()
    if _cached_config is not None:
        if current_time - _cached_config_time <= _cached_config.config_cache_ttl:
            return _cached_config
    
    if not CONFIG_FILE.exists():
        default_cfg = Config()
        save_config(default_cfg)
        _cached_config = default_cfg
        _cached_config_time = current_time
        return _cached_config
        
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
        
    # Создаем объект Config из словаря
    try:
        config = Config(**config_dict)
    except ValidationError as e:
        logger.exception(f"Validation errors: {e.errors()}")
        # Создаем конфигурацию по умолчанию и сохраняем её
        config = Config()
        save_config(config)
        _cached_config = config
        _cached_config_time = current_time
        return _cached_config
    except Exception as e:
        logger.exception(f"Ошибка валидации конфигурации: {e}")
        # Создаем конфигурацию по умолчанию и сохраняем её
        config = Config()
        save_config(config)
        _cached_config = config
        _cached_config_time = current_time
        return _cached_config
        
    # Backwards compatibility: if new flags not provided, derive them from old ones
    # (some older configs may only have use_dense_vectors/use_hybrid)
    if 'index_dense' not in config_dict:
        config.index_dense = config_dict.get('use_dense_vectors', True)
    if 'index_bm25' not in config_dict:
        # default False
        config.index_bm25 = config_dict.get('index_bm25', False)
    if 'index_hybrid' not in config_dict:
        config.index_hybrid = config_dict.get('use_hybrid', False)
    # Backwards compatibility for cache settings
    if 'config_cache_ttl' not in config_dict:
        config.config_cache_ttl = DEFAULT_CACHE_TTL
    if 'qdrant_client_cache_ttl' not in config_dict:
        config.qdrant_client_cache_ttl = DEFAULT_CACHE_TTL
    if 'collections_cache_ttl' not in config_dict:
        config.collections_cache_ttl = DEFAULT_CACHE_TTL
    # Backwards compatibility for new settings
    if 'mineru_subprocess_timeout' not in config_dict:
        config.mineru_subprocess_timeout = DEFAULT_MINERU_SUBPROCESS_TIMEOUT
    if 'gguf_model_n_ctx' not in config_dict:
        config.gguf_model_n_ctx = DEFAULT_GGUF_MODEL_N_CTX
    if 'search_default_k' not in config_dict:
        config.search_default_k = DEFAULT_SEARCH_DEFAULT_K
    if 'qdrant_retry_attempts' not in config_dict:
        config.qdrant_retry_attempts = DEFAULT_RETRY_ATTEMPTS
    if 'qdrant_retry_wait_time' not in config_dict:
        config.qdrant_retry_wait_time = DEFAULT_RETRY_WAIT_TIME
    if 'indexing_default_batch_size' not in config_dict:
        config.indexing_default_batch_size = DEFAULT_INDEXING_DEFAULT_BATCH_SIZE
    if 'memory_threshold' not in config_dict:
        config.memory_threshold = DEFAULT_MEMORY_THRESHOLD
    if 'qdrant_url' not in config_dict:
        config.qdrant_url = DEFAULT_QDRANT_URL

    # Проверяем доступность Qdrant через централизованный фабричный модуль
    try:
        from core.qdrant_client import get_qdrant_client
        client = get_qdrant_client(config)
        # Попытка подключения к Qdrant
        client.get_collections()
    except Exception as e:
        raise RuntimeError(f"Qdrant не доступен по адресу {config.qdrant_url}. Проверьте, запущен ли сервис Qdrant. Ошибка: {e}")
        
    _cached_config = config
    _cached_config_time = current_time
    return _cached_config


def save_config(config: Config):
    """Сохраняет конфигурацию в файл config.json."""
    global _cached_config, _cached_config_time
    
    # Преобразуем Config в словарь
    config_dict = config.model_dump()
    
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
    # Сбрасываем кэш, чтобы при следующем запросе конфигурация была перезагружена
    _cached_config = None
    _cached_config_time = 0