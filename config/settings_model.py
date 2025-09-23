"""Модуль для работы с конфигурацией приложения с использованием pydantic-settings."""

from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import resource path resolver
from config.resource_path import resource_path


class Config(BaseSettings):
    """Модель конфигурации приложения с автоматической загрузкой из различных источников."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Настройки индексации
    folder_path: str = "./data_to_index"
    collection_name: str = "final-dense-collection"
    current_hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_model_history: List[str] = Field(default=["sentence-transformers/all-MiniLM-L6-v2"])
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
    
    # Оптимизация для оценки моделей
    enable_pre_chunking_optimization: bool = True  # Включить предварительный chunking при оценке множества моделей
    
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
    memory_threshold: int = 500 * 1024 * 1024  # 500MB по умолчанию
    sparse_embedding: Optional[str] = "Qdrant/bm25"
    is_indexed: bool = False  # Флаг, указывающий, была ли выполнена индексация
    
    # Настройки Qdrant
    qdrant_url: str = "http://localhost:6333"
    
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
    mineru_subprocess_timeout: int = 600  # Таймаут для subprocess вызова mineru в секундах
    
    # Настройки кэширования
    config_cache_ttl: int = Field(default=60, ge=1)  # Минимум 1 секунда
    qdrant_client_cache_ttl: int = Field(default=60, ge=1)  # Минимум 1 секунда
    collections_cache_ttl: int = Field(default=60, ge=1)  # Минимум 1 секунда
    
    # Настройки GGUF моделей
    gguf_model_n_ctx: int = Field(default=4096, ge=1)  # Минимум 1
    
    # Настройки поиска
    search_default_k: int = Field(default=5, ge=1)  # Минимум 1 результат
    use_hybrid: bool = False  # Использовать гибридный поиск (dense + sparse)
    
    # BM25 Native Sparse Configuration
    use_bm25: bool = True  # Enable native BM25 via sparse vectors with IDF
    sparse_vector_name: str = "bm25_text"  # Name of the sparse vector field
    bm25_tokenizer: str = "word"  # Tokenizer type: word, whitespace, prefix
    bm25_min_token_len: int = 2
    
    # Настройки reranker
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = Field(default=5, ge=1)  # Минимум 1 результат
    
    # Настройки подключения к Qdrant
    qdrant_retry_attempts: int = Field(default=3, ge=1)  # Минимум 1 попытка
    qdrant_retry_wait_time: int = Field(default=2, ge=1)  # Минимум 1 секунда
    
    # Настройки индексации документов
    indexing_default_batch_size: int = Field(default=32, ge=1)  # Минимум 1
    
    # Настройки метаданных
    enable_metadata_extraction: bool = True
    metadata_custom_fields: Dict[str, Any] = Field(default={})
    metadata_extract_pdf: bool = True
    metadata_extract_image: bool = True
    metadata_extract_docx: bool = True
    
    # Путь к файлу конфигурации (для совместимости)
    config_file_path: str = str(resource_path("config/config.json"))
    
    # Токен HuggingFace для приватных моделей
    huggingface_token: Optional[str] = Field(default=None, exclude=True)

    def save_to_file(self, file_path: Optional[str] = None) -> None:
        """Сохранить конфигурацию в JSON файл.
        
        Args:
            file_path: Путь к файлу для сохранения (по умолчанию используется config_file_path)
        """
        import json
        if file_path is None:
            file_path = self.config_file_path
            
        config_dict = self.model_dump()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)