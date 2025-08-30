"""Модуль для работы с конфигурацией приложения."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONFIG_FILE = Path("config/config.json")



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
    # Новые параметры для чанкинга
    chunking_strategy: str = "character"  # "character", "paragraph" или "sentence"
    # Параметры для чанкинга по абзацам
    paragraphs_per_chunk: int = 3
    paragraph_overlap: int = 1
    # Параметры для чанкинга по предложениям
    sentences_per_chunk: int = 5
    sentence_overlap: int = 1
    # Параметры для многоуровневого чанкинга
    use_multilevel_chunking: bool = False
    # Параметры макро-чанкинга
    multilevel_macro_strategy: str = "character"  # "character", "paragraph" или "sentence"
    multilevel_macro_chunk_size: int = 10000
    multilevel_macro_chunk_overlap: int = 1000
    multilevel_macro_paragraphs_per_chunk: int = 5
    multilevel_macro_paragraph_overlap: int = 1
    multilevel_macro_sentences_per_chunk: int = 10
    multilevel_macro_sentence_overlap: int = 1
    # Параметры микро-чанкинга
    multilevel_micro_strategy: str = "character"  # "character", "paragraph" или "sentence"
    multilevel_micro_chunk_size: int = 1000
    multilevel_micro_chunk_overlap: int = 100
    multilevel_micro_paragraphs_per_chunk: int = 3
    multilevel_micro_paragraph_overlap: int = 1
    multilevel_micro_sentences_per_chunk: int = 5
    multilevel_micro_sentence_overlap: int = 1
    device: str = "auto"
    # Новые отдельные режимы индексации
    index_dense: bool = True
    index_bm25: bool = False
    index_hybrid: bool = False
    embedding_batch_size: int = Field(default=32, ge=1)  # Минимум 1
    indexing_batch_size: int = Field(default=50, ge=1)   # Минимум 1
    force_recreate: bool = True
    memory_threshold: int = DEFAULT_MEMORY_THRESHOLD  # 500MB по умолчанию
    sparse_embedding: Optional[str] = "Qdrant/bm25"
    is_indexed: bool = False  # Флаг, указывающий, была ли выполнена индексация
    
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
    
    # Настройки метаданных
    enable_metadata_extraction: bool = True
    metadata_custom_fields: Dict[str, Any] = Field(default_factory=dict)
    metadata_extract_pdf: bool = True
    metadata_extract_image: bool = True
    metadata_extract_docx: bool = True


def load_config() -> Config:
    """Загружает конфигурацию из файла config.json.
    
    Если файл не существует, создает его с настройками по умолчанию.
    """
    from config.config_manager import ConfigManager
    
    config_manager = ConfigManager.get_instance()
    return config_manager.get()


def save_config(config: Config):
    """Сохраняет конфигурацию в файл config.json."""
    from config.config_manager import ConfigManager
    
    config_manager = ConfigManager.get_instance()
    config_manager.save(config)