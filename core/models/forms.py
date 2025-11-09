"""
Модели Pydantic для форм FastAPI в проекте KAZrag.
"""
from typing import Optional, List, Dict, Any
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from fastapi import Form
from typing_extensions import Annotated


class DoclingForm(BaseModel):
    """Форма для настроек Docling."""
    use_ocr: bool = Form(default=True)
    use_tables: bool = Form(default=True)
    use_formulas: bool = Form(default=True)
    model_backend: str = Form(default="huggingface")
    ocr_model: str = Form(default="easyocr")
    ocr_lang: str = Form(default="ru")
    images_dir: str = Form(default="images")
    table_mode: str = Form(default="accurate")
    enable_page_images: bool = Form(default=True)
    table_detection_advanced: bool = Form(default=True)
    formula_detection_advanced: bool = Form(default=True)
    backend: str = Form(default="classic")
    device: str = Form(default="auto")
    granite_models_dir: str = Form(default="./models/granite")

    @field_validator('model_backend')
    def validate_model_backend(cls, v):
        if v not in ["huggingface", "local", "api"]:
            raise ValueError("model_backend должен быть одним из: huggingface, local, api")
        return v

    @field_validator('ocr_lang')
    def validate_ocr_lang(cls, v):
        if v not in ["ru", "en", "kk", "east_slavic"]:
            raise ValueError("ocr_lang должен быть одним из: ru, en, kk, east_slavic")
        return v

    @field_validator('table_mode')
    def validate_table_mode(cls, v):
        if v not in ["fast", "accurate"]:
            raise ValueError("table_mode должен быть одним из: fast, accurate")
        return v

    @field_validator('backend')
    def validate_backend(cls, v):
        if v not in ["classic", "granite"]:
            raise ValueError("backend должен быть одним из: classic, granite")
        return v

    @field_validator('device')
    def validate_device(cls, v):
        if v not in ["cpu", "gpu", "auto"]:
            raise ValueError("device должен быть одним из: cpu, gpu, auto")
        return v


class SearchForm(BaseModel):
    """Форма для настроек поиска."""
    default_k: int = Form(default=5, ge=1)
    default_collection: str = Form(default="final-dense-collection")
    default_device: str = Form(default="cpu")
    default_type: str = Form(default="dense")
    default_use_reranker: bool = Form(default=True)
    use_hybrid: bool = Form(default=False)
    hybrid_alpha: float = Form(default=0.7, ge=0.0, le=1.0)

    @field_validator('default_type')
    def validate_default_type(cls, v):
        if v not in ["dense", "sparse", "hybrid"]:
            raise ValueError("default_type должен быть одним из: dense, sparse, hybrid")
        return v


class RerankerForm(BaseModel):
    """Форма для настроек reranker."""
    enabled: bool = Form(default=False)
    model: str = Form(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k: int = Form(default=5, ge=1)


class EmbeddingForm(BaseModel):
    """Форма для настроек эмбеддингов."""
    current_hf_model: str = Form(default="sentence-transformers/all-MiniLM-L6-v2")
    device: str = Form(default="auto")
    batch_size: int = Form(default=32, ge=1)
    gguf_model_n_ctx: int = Form(default=4096, ge=1)


class QdrantForm(BaseModel):
    """Форма для настроек Qdrant."""
    url: str = Form(default="http://localhost:6333")
    retry_attempts: int = Form(default=3, ge=1)
    retry_wait_time: int = Form(default=2, ge=1)


class BM25Form(BaseModel):
    """Форма для настроек BM25."""
    enabled: bool = Form(default=True)
    sparse_vector_name: str = Form(default="bm25_text")
    tokenizer: str = Form(default="word")
    min_token_len: int = Form(default=2, ge=1)

    @field_validator('tokenizer')
    def validate_tokenizer(cls, v):
        if v not in ["word", "whitespace", "prefix"]:
            raise ValueError("tokenizer должен быть одним из: word, whitespace, prefix")
        return v


class RAGForm(BaseModel):
    """Форма для настроек RAG."""
    enabled: bool = Form(default=False)
    model_path: str = Form(default="models/Qwen3-4B-Instruct-2507-Q8_0.gguf")
    system_prompt: str = Form(default="You are a helpful assistant. Use the following context to answer the user's question accurately.")
    top_k: int = Form(default=3, ge=1, le=10)
    max_tokens: int = Form(default=512, ge=1)
    temperature: float = Form(default=0.7, ge=0.0, le=1.0)
    context_size: int = Form(default=4096, ge=512)
    gpu_layers: int = Form(default=-1, ge=-1)
    threads: int = Form(default=4, ge=1)
    batch_size: int = Form(default=512, ge=1)
    beam_size: int = Form(default=1, ge=1)

    @model_validator(mode='after')
    def validate_rag_constraints(self):
        """Проверяем, что top_k <= max_tokens."""
        if self.top_k > self.max_tokens:
            raise ValueError("top_k must be less than or equal to max_tokens")
        return self


class IndexingForm(BaseModel):
    """Форма для настроек индексации."""
    folder_path: str = Form(default="./data_to_index")
    collection_name: str = Form(default="final-dense-collection")
    index_dense: bool = Form(default=True)
    index_bm25: bool = Form(default=False)
    index_hybrid: bool = Form(default=False)
    chunk_size: int = Form(default=500, ge=1)
    chunk_overlap: int = Form(default=100, ge=0)
    chunking_strategy: str = Form(default="character")
    paragraphs_per_chunk: int = Form(default=3, ge=1)
    paragraph_overlap: int = Form(default=1, ge=0)
    sentences_per_chunk: int = Form(default=5, ge=1)
    sentence_overlap: int = Form(default=1, ge=0)
    use_multilevel_chunking: bool = Form(default=False)
    batch_size: int = Form(default=50, ge=1)
    force_recreate: bool = Form(default=False)
    memory_threshold: int = Form(default=500 * 1024 * 1024, ge=1)
    sparse_embedding: Optional[str] = Form(default="Qdrant/bm25")

    @field_validator('chunking_strategy')
    def validate_chunking_strategy(cls, v):
        if v not in ["character", "paragraph", "sentence"]:
            raise ValueError("chunking_strategy должен быть одним из: character, paragraph, sentence")
        return v


class MultilevelChunkerForm(BaseModel):
    """Форма для настроек многоуровневого чанкинга."""
    # Макро-чанкинг
    macro_strategy: str = Form(default="character")
    macro_chunk_size: int = Form(default=10000, ge=100)
    macro_chunk_overlap: int = Form(default=1000, ge=0)
    macro_paragraphs_per_chunk: int = Form(default=5, ge=1)
    macro_paragraph_overlap: int = Form(default=1, ge=0)
    macro_sentences_per_chunk: int = Form(default=10, ge=1)
    macro_sentence_overlap: int = Form(default=1, ge=0)
    
    # Микро-чанкинг
    micro_strategy: str = Form(default="character")
    micro_chunk_size: int = Form(default=1000, ge=100)
    micro_chunk_overlap: int = Form(default=100, ge=0)
    micro_paragraphs_per_chunk: int = Form(default=3, ge=1)
    micro_paragraph_overlap: int = Form(default=1, ge=0)
    micro_sentences_per_chunk: int = Form(default=5, ge=1)
    micro_sentence_overlap: int = Form(default=1, ge=0)

    @field_validator('macro_strategy', 'micro_strategy')
    def validate_strategy(cls, v):
        if v not in ["character", "paragraph", "sentence"]:
            raise ValueError("strategy должен быть одним из: character, paragraph, sentence")
        return v


class MetadataForm(BaseModel):
    """Форма для настроек метаданных."""
    enable_extraction: bool = Form(default=True)
    extract_pdf: bool = Form(default=True)
    extract_image: bool = Form(default=True)
    extract_docx: bool = Form(default=True)


class CacheForm(BaseModel):
    """Форма для настроек кэширования."""
    config_cache_ttl: int = Form(default=60, ge=1)
    qdrant_client_cache_ttl: int = Form(default=60, ge=1)
    collections_cache_ttl: int = Form(default=60, ge=1)


class ModelPathsForm(BaseModel):
    """Форма для настроек путей к моделям."""
    local_models_path: str = Form(default="./models")
    huggingface_cache_path: str = Form(default="./models/huggingface_cache")
    easyocr_models_path: str = Form(default="./models/easyocr")
    fastembed_cache_path: str = Form(default="./models/fastembed")
    use_local_only: bool = Form(default=True)
    auto_download_models: bool = Form(default=True)


class MinerUForm(BaseModel):
    """Форма для настроек MinerU (совместимость)."""
    input_pdf_dir: str = Form(default="./pdfs_to_process")
    output_md_dir: str = Form(default="./data_to_index")
    enable_formula_parsing: bool = Form(default=False)
    enable_table_parsing: bool = Form(default=False)
    model_source: str = Form(default="huggingface")
    models_dir: str = Form(default="")
    backend: str = Form(default="pipeline")
    method: str = Form(default="auto")
    lang: str = Form(default="east_slavic")
    sglang_url: str = Form(default="")
    subprocess_timeout: int = Form(default=600, ge=1)


# Класс для получения форм из FastAPI
class FormParser:
    """Класс-помощник для парсинга форм из FastAPI запросов."""
    
    @classmethod
    def as_form(cls):
        """
        Декоратор для использования Pydantic моделей как форм в FastAPI.
        """
        def field2formparam(field) -> Form:
            # Проверяем, есть ли у поля значения по умолчанию
            if field.is_required():
                default = ...
            else:
                default = field.get_default()
            
            # Возвращаем Form параметр с настройками поля
            return Form(
                default=default,
                description=field.description,
                title=field.title,
                gt=field.gt,
                ge=field.ge,
                lt=field.lt,
                le=field.le,
                multiple_of=field.multiple_of,
                max_length=field.max_length,
                min_length=field.min_length,
            )
        
        # Создаем параметры для всех полей модели
        params = {
            name: (field.annotation, field2formparam(field))
            for name, field in cls.model_fields.items()
        }
        
        # Создаем новый класс с Form параметрами
        form_cls = type(cls.__name__, (BaseModel,), {
            '__annotations__': {k: v[0] for k, v in params.items()},
            **{k: v[1] for k, v in params.items()}
        })
        
        return form_cls