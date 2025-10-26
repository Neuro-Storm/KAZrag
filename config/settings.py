"""Модуль для работы с конфигурацией приложения с использованием pydantic-settings."""

from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import resource path resolver
from config.resource_path import resource_path
from core.models.config import MainConfig


class Config(BaseSettings):
    """Модель конфигурации приложения с автоматической загрузкой из различных источников."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Вложенные конфигурации
    main: MainConfig = Field(default_factory=MainConfig, description="Основная конфигурация")
    
    # Старые поля для обратной совместимости - свойства, которые ссылаются на вложенные настройки
    @property
    def folder_path(self) -> str:
        return self.main.indexing.folder_path
    
    @folder_path.setter
    def folder_path(self, value: str) -> None:
        self.main.indexing.folder_path = value
    
    @property
    def collection_name(self) -> str:
        return self.main.indexing.collection_name
    
    @collection_name.setter
    def collection_name(self, value: str) -> None:
        self.main.indexing.collection_name = value
    
    @property
    def current_hf_model(self) -> str:
        return self.main.embedding.current_hf_model
    
    @current_hf_model.setter
    def current_hf_model(self, value: str) -> None:
        self.main.embedding.current_hf_model = value
    
    @property
    def hf_model_history(self) -> List[str]:
        return self.main.embedding.hf_model_history
    
    @hf_model_history.setter
    def hf_model_history(self, value: List[str]) -> None:
        self.main.embedding.hf_model_history = value
    
    @property
    def chunk_size(self) -> int:
        return self.main.indexing.chunk_size
    
    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        self.main.indexing.chunk_size = value
    
    @property
    def chunk_overlap(self) -> int:
        return self.main.indexing.chunk_overlap
    
    @chunk_overlap.setter
    def chunk_overlap(self, value: int) -> None:
        self.main.indexing.chunk_overlap = value
    
    @property
    def chunking_strategy(self) -> str:
        return self.main.indexing.chunking_strategy
    
    @chunking_strategy.setter
    def chunking_strategy(self, value: str) -> None:
        self.main.indexing.chunking_strategy = value
    
    @property
    def paragraphs_per_chunk(self) -> int:
        return self.main.indexing.paragraphs_per_chunk
    
    @paragraphs_per_chunk.setter
    def paragraphs_per_chunk(self, value: int) -> None:
        self.main.indexing.paragraphs_per_chunk = value
    
    @property
    def paragraph_overlap(self) -> int:
        return self.main.indexing.paragraph_overlap
    
    @paragraph_overlap.setter
    def paragraph_overlap(self, value: int) -> None:
        self.main.indexing.paragraph_overlap = value
    
    @property
    def sentences_per_chunk(self) -> int:
        return self.main.indexing.sentences_per_chunk
    
    @sentences_per_chunk.setter
    def sentences_per_chunk(self, value: int) -> None:
        self.main.indexing.sentences_per_chunk = value
    
    @property
    def sentence_overlap(self) -> int:
        return self.main.indexing.sentence_overlap
    
    @sentence_overlap.setter
    def sentence_overlap(self, value: int) -> None:
        self.main.indexing.sentence_overlap = value
    
    @property
    def use_multilevel_chunking(self) -> bool:
        return self.main.indexing.use_multilevel_chunking
    
    @use_multilevel_chunking.setter
    def use_multilevel_chunking(self, value: bool) -> None:
        self.main.indexing.use_multilevel_chunking = value
    
    @property
    def enable_pre_chunking_optimization(self) -> bool:
        return True  # Временная заглушка, так как это не используется в новой структуре
    
    @enable_pre_chunking_optimization.setter
    def enable_pre_chunking_optimization(self, value: bool) -> None:
        # Временная заглушка, так как это не используется в новой структуре
        pass
    
    @property
    def multilevel_macro_strategy(self) -> str:
        return self.main.multilevel_chunker.macro_strategy
    
    @multilevel_macro_strategy.setter
    def multilevel_macro_strategy(self, value: str) -> None:
        self.main.multilevel_chunker.macro_strategy = value
    
    @property
    def multilevel_macro_chunk_size(self) -> int:
        return self.main.multilevel_chunker.macro_chunk_size
    
    @multilevel_macro_chunk_size.setter
    def multilevel_macro_chunk_size(self, value: int) -> None:
        self.main.multilevel_chunker.macro_chunk_size = value
    
    @property
    def multilevel_macro_chunk_overlap(self) -> int:
        return self.main.multilevel_chunker.macro_chunk_overlap
    
    @multilevel_macro_chunk_overlap.setter
    def multilevel_macro_chunk_overlap(self, value: int) -> None:
        self.main.multilevel_chunker.macro_chunk_overlap = value
    
    @property
    def multilevel_macro_paragraphs_per_chunk(self) -> int:
        return self.main.multilevel_chunker.macro_paragraphs_per_chunk
    
    @multilevel_macro_paragraphs_per_chunk.setter
    def multilevel_macro_paragraphs_per_chunk(self, value: int) -> None:
        self.main.multilevel_chunker.macro_paragraphs_per_chunk = value
    
    @property
    def multilevel_macro_paragraph_overlap(self) -> int:
        return self.main.multilevel_chunker.macro_paragraph_overlap
    
    @multilevel_macro_paragraph_overlap.setter
    def multilevel_macro_paragraph_overlap(self, value: int) -> None:
        self.main.multilevel_chunker.macro_paragraph_overlap = value
    
    @property
    def multilevel_macro_sentences_per_chunk(self) -> int:
        return self.main.multilevel_chunker.macro_sentences_per_chunk
    
    @multilevel_macro_sentences_per_chunk.setter
    def multilevel_macro_sentences_per_chunk(self, value: int) -> None:
        self.main.multilevel_chunker.macro_sentences_per_chunk = value
    
    @property
    def multilevel_macro_sentence_overlap(self) -> int:
        return self.main.multilevel_chunker.macro_sentence_overlap
    
    @multilevel_macro_sentence_overlap.setter
    def multilevel_macro_sentence_overlap(self, value: int) -> None:
        self.main.multilevel_chunker.macro_sentence_overlap = value
    
    @property
    def multilevel_micro_strategy(self) -> str:
        return self.main.multilevel_chunker.micro_strategy
    
    @multilevel_micro_strategy.setter
    def multilevel_micro_strategy(self, value: str) -> None:
        self.main.multilevel_chunker.micro_strategy = value
    
    @property
    def multilevel_micro_chunk_size(self) -> int:
        return self.main.multilevel_chunker.micro_chunk_size
    
    @multilevel_micro_chunk_size.setter
    def multilevel_micro_chunk_size(self, value: int) -> None:
        self.main.multilevel_chunker.micro_chunk_size = value
    
    @property
    def multilevel_micro_chunk_overlap(self) -> int:
        return self.main.multilevel_chunker.micro_chunk_overlap
    
    @multilevel_micro_chunk_overlap.setter
    def multilevel_micro_chunk_overlap(self, value: int) -> None:
        self.main.multilevel_chunker.micro_chunk_overlap = value
    
    @property
    def multilevel_micro_paragraphs_per_chunk(self) -> int:
        return self.main.multilevel_chunker.micro_paragraphs_per_chunk
    
    @multilevel_micro_paragraphs_per_chunk.setter
    def multilevel_micro_paragraphs_per_chunk(self, value: int) -> None:
        self.main.multilevel_chunker.micro_paragraphs_per_chunk = value
    
    @property
    def multilevel_micro_paragraph_overlap(self) -> int:
        return self.main.multilevel_chunker.micro_paragraph_overlap
    
    @multilevel_micro_paragraph_overlap.setter
    def multilevel_micro_paragraph_overlap(self, value: int) -> None:
        self.main.multilevel_chunker.micro_paragraph_overlap = value
    
    @property
    def multilevel_micro_sentences_per_chunk(self) -> int:
        return self.main.multilevel_chunker.micro_sentences_per_chunk
    
    @multilevel_micro_sentences_per_chunk.setter
    def multilevel_micro_sentences_per_chunk(self, value: int) -> None:
        self.main.multilevel_chunker.micro_sentences_per_chunk = value
    
    @property
    def multilevel_micro_sentence_overlap(self) -> int:
        return self.main.multilevel_chunker.micro_sentence_overlap
    
    @multilevel_micro_sentence_overlap.setter
    def multilevel_micro_sentence_overlap(self, value: int) -> None:
        self.main.multilevel_chunker.micro_sentence_overlap = value
    
    @property
    def device(self) -> str:
        return self.main.embedding.device
    
    @device.setter
    def device(self, value: str) -> None:
        self.main.embedding.device = value
    
    @property
    def index_dense(self) -> bool:
        return self.main.indexing.index_dense
    
    @index_dense.setter
    def index_dense(self, value: bool) -> None:
        self.main.indexing.index_dense = value
    
    @property
    def index_bm25(self) -> bool:
        return self.main.indexing.index_bm25
    
    @index_bm25.setter
    def index_bm25(self, value: bool) -> None:
        self.main.indexing.index_bm25 = value
    
    @property
    def index_hybrid(self) -> bool:
        return self.main.indexing.index_hybrid
    
    @index_hybrid.setter
    def index_hybrid(self, value: bool) -> None:
        self.main.indexing.index_hybrid = value
    
    @property
    def embedding_batch_size(self) -> int:
        return self.main.embedding.batch_size
    
    @embedding_batch_size.setter
    def embedding_batch_size(self, value: int) -> None:
        self.main.embedding.batch_size = value
    
    @property
    def indexing_batch_size(self) -> int:
        return self.main.indexing.batch_size
    
    @indexing_batch_size.setter
    def indexing_batch_size(self, value: int) -> None:
        self.main.indexing.batch_size = value
    
    @property
    def force_recreate(self) -> bool:
        return self.main.indexing.force_recreate
    
    @force_recreate.setter
    def force_recreate(self, value: bool) -> None:
        self.main.indexing.force_recreate = value
    
    @property
    def memory_threshold(self) -> int:
        return self.main.indexing.memory_threshold
    
    @memory_threshold.setter
    def memory_threshold(self, value: int) -> None:
        self.main.indexing.memory_threshold = value
    
    @property
    def sparse_embedding(self) -> Optional[str]:
        return self.main.indexing.sparse_embedding
    
    @sparse_embedding.setter
    def sparse_embedding(self, value: Optional[str]) -> None:
        self.main.indexing.sparse_embedding = value
    
    @property
    def is_indexed(self) -> bool:
        return self.main.indexing.is_indexed
    
    @is_indexed.setter
    def is_indexed(self, value: bool) -> None:
        self.main.indexing.is_indexed = value
    
    @property
    def qdrant_url(self) -> str:
        return self.main.qdrant.url
    
    @qdrant_url.setter
    def qdrant_url(self, value: str) -> None:
        self.main.qdrant.url = value
    
    @property
    def mineru_input_pdf_dir(self) -> str:
        return self.main.mineru.input_pdf_dir
    
    @mineru_input_pdf_dir.setter
    def mineru_input_pdf_dir(self, value: str) -> None:
        self.main.mineru.input_pdf_dir = value
    
    @property
    def mineru_output_md_dir(self) -> str:
        return self.main.mineru.output_md_dir
    
    @mineru_output_md_dir.setter
    def mineru_output_md_dir(self, value: str) -> None:
        self.main.mineru.output_md_dir = value
    
    @property
    def mineru_enable_formula_parsing(self) -> bool:
        return self.main.mineru.enable_formula_parsing
    
    @mineru_enable_formula_parsing.setter
    def mineru_enable_formula_parsing(self, value: bool) -> None:
        self.main.mineru.enable_formula_parsing = value
    
    @property
    def mineru_enable_table_parsing(self) -> bool:
        return self.main.mineru.enable_table_parsing
    
    @mineru_enable_table_parsing.setter
    def mineru_enable_table_parsing(self, value: bool) -> None:
        self.main.mineru.enable_table_parsing = value
    
    @property
    def mineru_model_source(self) -> str:
        return self.main.mineru.model_source
    
    @mineru_model_source.setter
    def mineru_model_source(self, value: str) -> None:
        self.main.mineru.model_source = value
    
    @property
    def mineru_models_dir(self) -> str:
        return self.main.mineru.models_dir
    
    @mineru_models_dir.setter
    def mineru_models_dir(self, value: str) -> None:
        self.main.mineru.models_dir = value
    
    @property
    def mineru_backend(self) -> str:
        return self.main.mineru.backend
    
    @mineru_backend.setter
    def mineru_backend(self, value: str) -> None:
        self.main.mineru.backend = value
    
    @property
    def mineru_method(self) -> str:
        return self.main.mineru.method
    
    @mineru_method.setter
    def mineru_method(self, value: str) -> None:
        self.main.mineru.method = value
    
    @property
    def mineru_lang(self) -> str:
        return self.main.mineru.lang
    
    @mineru_lang.setter
    def mineru_lang(self, value: str) -> None:
        self.main.mineru.lang = value
    
    @property
    def mineru_sglang_url(self) -> str:
        return self.main.mineru.sglang_url
    
    @mineru_sglang_url.setter
    def mineru_sglang_url(self, value: str) -> None:
        self.main.mineru.sglang_url = value
    
    @property
    def mineru_subprocess_timeout(self) -> int:
        return self.main.mineru.subprocess_timeout
    
    @mineru_subprocess_timeout.setter
    def mineru_subprocess_timeout(self, value: int) -> None:
        self.main.mineru.subprocess_timeout = value
    
    @property
    def docling_use_ocr(self) -> bool:
        return self.main.docling.use_ocr
    
    @docling_use_ocr.setter
    def docling_use_ocr(self, value: bool) -> None:
        self.main.docling.use_ocr = value
    
    @property
    def docling_use_tables(self) -> bool:
        return self.main.docling.use_tables
    
    @docling_use_tables.setter
    def docling_use_tables(self, value: bool) -> None:
        self.main.docling.use_tables = value
    
    @property
    def docling_use_formulas(self) -> bool:
        return self.main.docling.use_formulas
    
    @docling_use_formulas.setter
    def docling_use_formulas(self, value: bool) -> None:
        self.main.docling.use_formulas = value
    
    @property
    def docling_model_backend(self) -> str:
        return self.main.docling.model_backend
    
    @docling_model_backend.setter
    def docling_model_backend(self, value: str) -> None:
        self.main.docling.model_backend = value
    
    @property
    def docling_ocr_model(self) -> str:
        return self.main.docling.ocr_model
    
    @docling_ocr_model.setter
    def docling_ocr_model(self, value: str) -> None:
        self.main.docling.ocr_model = value
    
    @property
    def docling_ocr_lang(self) -> str:
        return self.main.docling.ocr_lang
    
    @docling_ocr_lang.setter
    def docling_ocr_lang(self, value: str) -> None:
        self.main.docling.ocr_lang = value
    
    @property
    def docling_images_dir(self) -> str:
        return self.main.docling.images_dir
    
    @docling_images_dir.setter
    def docling_images_dir(self, value: str) -> None:
        self.main.docling.images_dir = value
    
    @property
    def docling_table_mode(self) -> str:
        return self.main.docling.table_mode
    
    @docling_table_mode.setter
    def docling_table_mode(self, value: str) -> None:
        self.main.docling.table_mode = value
    
    @property
    def docling_enable_page_images(self) -> bool:
        return self.main.docling.enable_page_images
    
    @docling_enable_page_images.setter
    def docling_enable_page_images(self, value: bool) -> None:
        self.main.docling.enable_page_images = value
    
    @property
    def docling_table_detection_advanced(self) -> bool:
        return self.main.docling.table_detection_advanced
    
    @docling_table_detection_advanced.setter
    def docling_table_detection_advanced(self, value: bool) -> None:
        self.main.docling.table_detection_advanced = value
    
    @property
    def docling_formula_detection_advanced(self) -> bool:
        return self.main.docling.formula_detection_advanced
    
    @docling_formula_detection_advanced.setter
    def docling_formula_detection_advanced(self, value: bool) -> None:
        self.main.docling.formula_detection_advanced = value
    
    @property
    def docling_backend(self) -> str:
        return self.main.docling.backend
    
    @docling_backend.setter
    def docling_backend(self, value: str) -> None:
        self.main.docling.backend = value
    
    @property
    def docling_device(self) -> str:
        return self.main.docling.device
    
    @docling_device.setter
    def docling_device(self, value: str) -> None:
        self.main.docling.device = value
    
    @property
    def docling_granite_models_dir(self) -> Path:
        return self.main.docling.granite_models_dir
    
    @docling_granite_models_dir.setter
    def docling_granite_models_dir(self, value: Path) -> None:
        self.main.docling.granite_models_dir = value
    
    @property
    def granite_models_dir(self) -> Path:
        return self.main.docling.granite_models_dir
    
    @granite_models_dir.setter
    def granite_models_dir(self, value: Path) -> None:
        self.main.docling.granite_models_dir = value
    
    @property
    def config_cache_ttl(self) -> int:
        return self.main.cache.config_cache_ttl
    
    @config_cache_ttl.setter
    def config_cache_ttl(self, value: int) -> None:
        self.main.cache.config_cache_ttl = value
    
    @property
    def qdrant_client_cache_ttl(self) -> int:
        return self.main.cache.qdrant_client_cache_ttl
    
    @qdrant_client_cache_ttl.setter
    def qdrant_client_cache_ttl(self, value: int) -> None:
        self.main.cache.qdrant_client_cache_ttl = value
    
    @property
    def collections_cache_ttl(self) -> int:
        return self.main.cache.collections_cache_ttl
    
    @collections_cache_ttl.setter
    def collections_cache_ttl(self, value: int) -> None:
        self.main.cache.collections_cache_ttl = value
    
    @property
    def gguf_model_n_ctx(self) -> int:
        return self.main.embedding.gguf.model_n_ctx
    
    @gguf_model_n_ctx.setter
    def gguf_model_n_ctx(self, value: int) -> None:
        self.main.embedding.gguf.model_n_ctx = value
    
    @property
    def search_default_k(self) -> int:
        return self.main.search.default_k
    
    @search_default_k.setter
    def search_default_k(self, value: int) -> None:
        self.main.search.default_k = value
    
    @property
    def use_hybrid(self) -> bool:
        return self.main.search.use_hybrid
    
    @use_hybrid.setter
    def use_hybrid(self, value: bool) -> None:
        self.main.search.use_hybrid = value
    
    @property
    def hybrid_alpha(self) -> float:
        return self.main.search.hybrid_alpha
    
    @hybrid_alpha.setter
    def hybrid_alpha(self, value: float) -> None:
        self.main.search.hybrid_alpha = value
    
    @property
    def search_default_collection(self) -> str:
        return self.main.search.default_collection
    
    @search_default_collection.setter
    def search_default_collection(self, value: str) -> None:
        self.main.search.default_collection = value
    
    @property
    def search_default_device(self) -> str:
        return self.main.search.default_device
    
    @search_default_device.setter
    def search_default_device(self, value: str) -> None:
        self.main.search.default_device = value
    
    @property
    def search_default_type(self) -> str:
        return self.main.search.default_type
    
    @search_default_type.setter
    def search_default_type(self, value: str) -> None:
        self.main.search.default_type = value
    
    @property
    def search_default_use_reranker(self) -> bool:
        return self.main.search.default_use_reranker
    
    @search_default_use_reranker.setter
    def search_default_use_reranker(self, value: bool) -> None:
        self.main.search.default_use_reranker = value
    
    @property
    def use_bm25(self) -> bool:
        return self.main.bm25.enabled
    
    @use_bm25.setter
    def use_bm25(self, value: bool) -> None:
        self.main.bm25.enabled = value
    
    @property
    def sparse_vector_name(self) -> str:
        return self.main.bm25.sparse_vector_name
    
    @sparse_vector_name.setter
    def sparse_vector_name(self, value: str) -> None:
        self.main.bm25.sparse_vector_name = value
    
    @property
    def bm25_tokenizer(self) -> str:
        return self.main.bm25.tokenizer
    
    @bm25_tokenizer.setter
    def bm25_tokenizer(self, value: str) -> None:
        self.main.bm25.tokenizer = value
    
    @property
    def bm25_min_token_len(self) -> int:
        return self.main.bm25.min_token_len
    
    @bm25_min_token_len.setter
    def bm25_min_token_len(self, value: int) -> None:
        self.main.bm25.min_token_len = value
    
    @property
    def reranker_enabled(self) -> bool:
        return self.main.reranker.enabled
    
    @reranker_enabled.setter
    def reranker_enabled(self, value: bool) -> None:
        self.main.reranker.enabled = value
    
    @property
    def reranker_model(self) -> str:
        return self.main.reranker.model
    
    @reranker_model.setter
    def reranker_model(self, value: str) -> None:
        self.main.reranker.model = value
    
    @property
    def reranker_top_k(self) -> int:
        return self.main.reranker.top_k
    
    @reranker_top_k.setter
    def reranker_top_k(self, value: int) -> None:
        self.main.reranker.top_k = value
    
    @property
    def qdrant_retry_attempts(self) -> int:
        return self.main.qdrant.retry_attempts
    
    @qdrant_retry_attempts.setter
    def qdrant_retry_attempts(self, value: int) -> None:
        self.main.qdrant.retry_attempts = value
    
    @property
    def qdrant_retry_wait_time(self) -> int:
        return self.main.qdrant.retry_wait_time
    
    @qdrant_retry_wait_time.setter
    def qdrant_retry_wait_time(self, value: int) -> None:
        self.main.qdrant.retry_wait_time = value
    
    @property
    def indexing_default_batch_size(self) -> int:
        return self.main.indexing.batch_size  # Используем то же поле
    
    @indexing_default_batch_size.setter
    def indexing_default_batch_size(self, value: int) -> None:
        self.main.indexing.batch_size = value
    
    @property
    def enable_metadata_extraction(self) -> bool:
        return self.main.metadata.enable_extraction
    
    @enable_metadata_extraction.setter
    def enable_metadata_extraction(self, value: bool) -> None:
        self.main.metadata.enable_extraction = value
    
    @property
    def metadata_custom_fields(self) -> Dict[str, Any]:
        return self.main.metadata.custom_fields
    
    @metadata_custom_fields.setter
    def metadata_custom_fields(self, value: Dict[str, Any]) -> None:
        self.main.metadata.custom_fields = value
    
    @property
    def metadata_extract_pdf(self) -> bool:
        return self.main.metadata.extract_pdf
    
    @metadata_extract_pdf.setter
    def metadata_extract_pdf(self, value: bool) -> None:
        self.main.metadata.extract_pdf = value
    
    @property
    def metadata_extract_image(self) -> bool:
        return self.main.metadata.extract_image
    
    @metadata_extract_image.setter
    def metadata_extract_image(self, value: bool) -> None:
        self.main.metadata.extract_image = value
    
    @property
    def metadata_extract_docx(self) -> bool:
        return self.main.metadata.extract_docx
    
    @metadata_extract_docx.setter
    def metadata_extract_docx(self, value: bool) -> None:
        self.main.metadata.extract_docx = value
    
    @property
    def config_file_path(self) -> str:
        return str(resource_path("config/config.json"))
    
    @property
    def huggingface_token(self) -> Optional[str]:
        return getattr(self, '_huggingface_token', None)  # Временная заглушка
    
    @huggingface_token.setter
    def huggingface_token(self, value: Optional[str]) -> None:
        self._huggingface_token = value
    
    @property
    def rag_enabled(self) -> bool:
        return self.main.rag.enabled
    
    @rag_enabled.setter
    def rag_enabled(self, value: bool) -> None:
        self.main.rag.enabled = value
    
    @property
    def rag_model_path(self) -> str:
        return self.main.rag.model_path
    
    @rag_model_path.setter
    def rag_model_path(self, value: str) -> None:
        self.main.rag.model_path = value
    
    @property
    def rag_system_prompt(self) -> str:
        return self.main.rag.system_prompt
    
    @rag_system_prompt.setter
    def rag_system_prompt(self, value: str) -> None:
        self.main.rag.system_prompt = value
    
    @property
    def rag_top_k(self) -> int:
        return self.main.rag.top_k
    
    @rag_top_k.setter
    def rag_top_k(self, value: int) -> None:
        self.main.rag.top_k = value
    
    @property
    def rag_max_tokens(self) -> int:
        return self.main.rag.max_tokens
    
    @rag_max_tokens.setter
    def rag_max_tokens(self, value: int) -> None:
        self.main.rag.max_tokens = value
    
    @property
    def rag_temperature(self) -> float:
        return self.main.rag.temperature
    
    @rag_temperature.setter
    def rag_temperature(self, value: float) -> None:
        self.main.rag.temperature = value
    
    @property
    def rag_context_size(self) -> int:
        return self.main.rag.context_size
    
    @rag_context_size.setter
    def rag_context_size(self, value: int) -> None:
        self.main.rag.context_size = value
    
    @property
    def rag_gpu_layers(self) -> int:
        return self.main.rag.gpu_layers
    
    @rag_gpu_layers.setter
    def rag_gpu_layers(self, value: int) -> None:
        self.main.rag.gpu_layers = value
    
    @property
    def rag_threads(self) -> int:
        return self.main.rag.threads
    
    @rag_threads.setter
    def rag_threads(self, value: int) -> None:
        self.main.rag.threads = value
    
    @property
    def rag_batch_size(self) -> int:
        return self.main.rag.batch_size
    
    @rag_batch_size.setter
    def rag_batch_size(self, value: int) -> None:
        self.main.rag.batch_size = value
    
    @property
    def rag_beam_size(self) -> int:
        return self.main.rag.beam_size
    
    @rag_beam_size.setter
    def rag_beam_size(self, value: int) -> None:
        self.main.rag.beam_size = value
    
    @property
    def local_models_path(self) -> Path:
        return self.main.model_paths.local_models_path
    
    @local_models_path.setter
    def local_models_path(self, value: Path) -> None:
        self.main.model_paths.local_models_path = value
    
    @property
    def huggingface_cache_path(self) -> Path:
        return self.main.model_paths.huggingface_cache_path
    
    @huggingface_cache_path.setter
    def huggingface_cache_path(self, value: Path) -> None:
        self.main.model_paths.huggingface_cache_path = value
    
    @property
    def easyocr_models_path(self) -> Path:
        return self.main.model_paths.easyocr_models_path
    
    @easyocr_models_path.setter
    def easyocr_models_path(self, value: Path) -> None:
        self.main.model_paths.easyocr_models_path = value
    
    @property
    def fastembed_cache_path(self) -> Path:
        return self.main.model_paths.fastembed_cache_path
    
    @fastembed_cache_path.setter
    def fastembed_cache_path(self, value: Path) -> None:
        self.main.model_paths.fastembed_cache_path = value
    
    @property
    def use_local_only(self) -> bool:
        return self.main.model_paths.use_local_only
    
    @use_local_only.setter
    def use_local_only(self, value: bool) -> None:
        self.main.model_paths.use_local_only = value
    
    @property
    def auto_download_models(self) -> bool:
        return self.main.model_paths.auto_download_models
    
    @auto_download_models.setter
    def auto_download_models(self, value: bool) -> None:
        self.main.model_paths.auto_download_models = value

    def save_to_file(self, file_path: Optional[str] = None) -> None:
        """Сохранить конфигурацию в JSON файл.
        
        Args:
            file_path: Путь к файлу для сохранения (по умолчанию используется config_file_path)
        """
        import json
        from pathlib import Path
        if file_path is None:
            file_path = self.config_file_path
            
        # Используем вложенные данные из main для сериализации
        config_dict = self.main.model_dump()
        
        # Преобразуем объекты Path в строки для сериализации в JSON
        def convert_path_to_str(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: convert_path_to_str(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_path_to_str(item) for item in obj]
            else:
                return obj
        
        config_dict = convert_path_to_str(config_dict)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)


# Для обратной совместимости
CONFIG_FILE = "config/config.json"

# Функция для обратной совместимости
def load_config(reload: bool = False) -> Config:
    """Загрузить конфигурацию для обратной совместимости.
    
    Args:
        reload: Если True, принудительно перезагрузить конфигурацию
        
    Returns:
        Config: Объект конфигурации
    """
    from config.config_manager import ConfigManager
    manager = ConfigManager.get_instance()
    if reload:
        return manager.reload()
    return manager.get()

__all__ = ["Config", "CONFIG_FILE", "load_config"]