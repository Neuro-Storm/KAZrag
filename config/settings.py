"""Модуль для работы с конфигурацией приложения с использованием pydantic-settings."""

from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import resource path resolver
from config.resource_path import resource_path
from core.models.config import MainConfig


# Определение алиасов вне класса, чтобы избежать конфликта с Pydantic
# Все алиасы, требуемые приложением, включая стандартные имена для обратной совместимости
_ALIAS_MAP = {
    # indexing aliases - both simple and prefixed versions for compatibility
    'folder_path': 'indexing.folder_path',
    'collection_name': 'indexing.collection_name',
    'chunk_size': 'indexing.chunk_size',
    'indexing_chunk_size': 'indexing.chunk_size',  # Original prefixed version
    'chunk_overlap': 'indexing.chunk_overlap',
    'indexing_chunk_overlap': 'indexing.chunk_overlap',  # Original prefixed version
    'chunking_strategy': 'indexing.chunking_strategy',
    'indexing_chunking_strategy': 'indexing.chunking_strategy',  # Original prefixed version
    'paragraphs_per_chunk': 'indexing.paragraphs_per_chunk',
    'indexing_paragraphs_per_chunk': 'indexing.paragraphs_per_chunk',  # Original prefixed version
    'paragraph_overlap': 'indexing.paragraph_overlap',
    'indexing_paragraph_overlap': 'indexing.paragraph_overlap',  # Original prefixed version
    'sentences_per_chunk': 'indexing.sentences_per_chunk',
    'indexing_sentences_per_chunk': 'indexing.sentences_per_chunk',  # Original prefixed version
    'sentence_overlap': 'indexing.sentence_overlap',
    'indexing_sentence_overlap': 'indexing.sentence_overlap',  # Original prefixed version
    'use_multilevel_chunking': 'indexing.use_multilevel_chunking',
    'index_dense': 'indexing.index_dense',
    'indexing_index_dense': 'indexing.index_dense',  # Original prefixed version
    'index_bm25': 'indexing.index_bm25',
    'indexing_index_bm25': 'indexing.index_bm25',  # Original prefixed version
    'index_hybrid': 'indexing.index_hybrid',
    'indexing_index_hybrid': 'indexing.index_hybrid',  # Original prefixed version
    'embedding_batch_size': 'embedding.batch_size',
    'indexing_batch_size': 'indexing.batch_size',
    'force_recreate': 'indexing.force_recreate',
    'indexing_force_recreate': 'indexing.force_recreate',  # Original prefixed version
    'memory_threshold': 'indexing.memory_threshold',
    'indexing_memory_threshold': 'indexing.memory_threshold',  # Original prefixed version
    'sparse_embedding': 'indexing.sparse_embedding',
    'indexing_sparse_embedding': 'indexing.sparse_embedding',  # Original prefixed version
    'is_indexed': 'indexing.is_indexed',
    'indexing_is_indexed': 'indexing.is_indexed',  # Original prefixed version
    
    # embedding aliases
    'current_hf_model': 'embedding.current_hf_model',
    'hf_model_history': 'embedding.hf_model_history',
    'device': 'embedding.device',
    
    # qdrant aliases
    'qdrant_url': 'qdrant.url',
    'qdrant_retry_attempts': 'qdrant.retry_attempts',
    'qdrant_retry_wait_time': 'qdrant.retry_wait_time',
    
    # mineru aliases
    'mineru_input_pdf_dir': 'mineru.input_pdf_dir',
    'mineru_output_md_dir': 'mineru.output_md_dir',
    'mineru_enable_formula_parsing': 'mineru.enable_formula_parsing',
    'mineru_enable_table_parsing': 'mineru.enable_table_parsing',
    'mineru_model_source': 'mineru.model_source',
    'mineru_models_dir': 'mineru.models_dir',
    'mineru_backend': 'mineru.backend',
    'mineru_method': 'mineru.method',
    'mineru_lang': 'mineru.lang',
    'mineru_sglang_url': 'mineru.sglang_url',
    'mineru_subprocess_timeout': 'mineru.subprocess_timeout',
    
    # docling aliases - These were removed from MainConfig but kept for backward compatibility
    'docling_use_ocr': 'docling.use_ocr',
    'docling_use_tables': 'docling.use_tables',
    'docling_use_formulas': 'docling.use_formulas',
    'docling_model_backend': 'docling.model_backend',
    'docling_ocr_model': 'docling.ocr_model',
    'docling_ocr_lang': 'docling.ocr_lang',
    'docling_images_dir': 'docling.images_dir',
    'docling_table_mode': 'docling.table_mode',
    'docling_enable_page_images': 'docling.enable_page_images',
    'docling_table_detection_advanced': 'docling.table_detection_advanced',
    'docling_formula_detection_advanced': 'docling.formula_detection_advanced',
    'docling_backend': 'docling.backend',
    'docling_device': 'docling.device',
    'docling_granite_models_dir': 'docling.granite_models_dir',
    'max_file_size_mb': 'docling.max_file_size_mb',  # Maximum file size in MB for processing
    
    # search aliases
    'search_default_k': 'search.default_k',
    'use_hybrid': 'search.use_hybrid',
    'search_use_hybrid': 'search.use_hybrid',  # Original prefixed version
    'hybrid_alpha': 'search.hybrid_alpha',
    'search_hybrid_alpha': 'search.hybrid_alpha',  # Original prefixed version
    'search_default_collection': 'search.default_collection',
    'search_default_device': 'search.default_device',
    'search_default_type': 'search.default_type',
    'search_default_use_reranker': 'search.default_use_reranker',
    
    # reranker aliases
    'reranker_enabled': 'reranker.enabled',
    'reranker_model': 'reranker.model',
    'reranker_top_k': 'reranker.top_k',
    
    # bm25 aliases
    'use_bm25': 'bm25.enabled',
    'sparse_vector_name': 'bm25.sparse_vector_name',
    'bm25_sparse_vector_name': 'bm25.sparse_vector_name',  # Original prefixed version
    'bm25_tokenizer': 'bm25.tokenizer',
    'bm25_min_token_len': 'bm25.min_token_len',
    
    # cache aliases
    'config_cache_ttl': 'cache.config_cache_ttl',
    'qdrant_client_cache_ttl': 'cache.qdrant_client_cache_ttl',
    'collections_cache_ttl': 'cache.collections_cache_ttl',
    
    # embedding specific aliases
    'gguf_model_n_ctx': 'embedding.gguf.model_n_ctx',
    
    # metadata aliases
    'enable_metadata_extraction': 'metadata.enable_extraction',
    'custom_fields': 'metadata.custom_fields',  # Also support non-prefixed version
    'metadata_custom_fields': 'metadata.custom_fields',
    'metadata_extract_pdf': 'metadata.extract_pdf',
    'metadata_extract_image': 'metadata.extract_image',
    'metadata_extract_docx': 'metadata.extract_docx',
    
    # rag aliases
    'rag_enabled': 'rag.enabled',
    'rag_model_path': 'rag.model_path',
    'rag_system_prompt': 'rag.system_prompt',
    'rag_top_k': 'rag.top_k',
    'rag_max_tokens': 'rag.max_tokens',
    'rag_temperature': 'rag.temperature',
    'rag_context_size': 'rag.context_size',
    'rag_gpu_layers': 'rag.gpu_layers',
    'rag_threads': 'rag.threads',
    'rag_batch_size': 'rag.batch_size',
    'rag_beam_size': 'rag.beam_size',
    
    # model path aliases
    'local_models_path': 'model_paths.local_models_path',
    'huggingface_cache_path': 'model_paths.huggingface_cache_path',
    'easyocr_models_path': 'model_paths.easyocr_models_path',
    'fastembed_cache_path': 'model_paths.fastembed_cache_path',
    'use_local_only': 'model_paths.use_local_only',
    'auto_download_models': 'model_paths.auto_download_models',
    
    # Multilevel chunker aliases - These were removed from MainConfig but kept for backward compatibility
    'multilevel_macro_strategy': 'multilevel_chunker.macro_strategy',
    'multilevel_macro_chunk_size': 'multilevel_chunker.macro_chunk_size',
    'multilevel_macro_chunk_overlap': 'multilevel_chunker.macro_chunk_overlap',
    'multilevel_macro_paragraphs_per_chunk': 'multilevel_chunker.macro_paragraphs_per_chunk',
    'multilevel_macro_paragraph_overlap': 'multilevel_chunker.macro_paragraph_overlap',
    'multilevel_macro_sentences_per_chunk': 'multilevel_chunker.macro_sentences_per_chunk',
    'multilevel_macro_sentence_overlap': 'multilevel_chunker.macro_sentence_overlap',
    'multilevel_micro_strategy': 'multilevel_chunker.micro_strategy',
    'multilevel_micro_chunk_size': 'multilevel_chunker.micro_chunk_size',
    'multilevel_micro_chunk_overlap': 'multilevel_chunker.micro_chunk_overlap',
    'multilevel_micro_paragraphs_per_chunk': 'multilevel_chunker.micro_paragraphs_per_chunk',
    'multilevel_micro_paragraph_overlap': 'multilevel_chunker.micro_paragraph_overlap',
    'multilevel_micro_sentences_per_chunk': 'multilevel_chunker.micro_sentences_per_chunk',
    'multilevel_micro_sentence_overlap': 'multilevel_chunker.micro_sentence_overlap',
    
    # Additional consistency aliases
    'granite_models_dir': 'docling.granite_models_dir',  # Duplicate alias
    'indexing_default_batch_size': 'indexing.batch_size',  # Same as indexing.batch_size
}


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

    def __getattribute__(self, name: str):
        """Delegate attribute access to main config using dot-path notation."""
        # Use object.__getattribute__ to avoid recursion and bypass Pydantic's attribute access
        # First, try to get the attribute normally
        try:
            # Try to get it using object's __getattribute__ to bypass Pydantic's logic
            result = object.__getattribute__(self, name)
            return result
        except AttributeError:
            # If it's not a normal attribute, try delegation to main config
            try:
                main = object.__getattribute__(self, "main")
            except AttributeError:
                raise AttributeError(f"{self.__class__.__name__} has no attribute {name!r} (main not set)")
            
            # Handle dot-notation paths
            if "." in name:
                if hasattr(main, "get_nested"):
                    return main.get_nested(name)
                # fallback simple traversal
                obj = main
                for part in name.split('.'):
                    obj = getattr(obj, part)
                return obj
            # Try the alias mapping first (for prefixed/legacy names)
            elif name in _ALIAS_MAP:
                actual_path = _ALIAS_MAP[name]
                if hasattr(main, "get_nested"):
                    return main.get_nested(actual_path)
                else:
                    # fallback: split path and traverse
                    obj = main
                    for part in actual_path.split('.'):
                        obj = getattr(obj, part)
                    return obj
            # Handle special cases like huggingface_token that was stored as _huggingface_token
            elif name == 'huggingface_token':
                return getattr(self, '_huggingface_token', None)
            # For non-alias direct attribute access, try to find it in main or its nested structures
            else:
                # If not found in alias map, try to find it in nested structures
                # For backward compatibility, non-prefixed names might map to nested paths
                # We'll try to find attributes that don't have prefixes
                # This is handled by checking if the attribute exists in any of the nested configs
                try:
                    # Try common paths where non-prefixed attributes might be
                    nested_names = [
                        f"indexing.{name}",
                        f"embedding.{name}",
                        f"qdrant.{name}",
                        f"search.{name}",
                        f"docling.{name}",
                        f"bm25.{name}",
                        f"cache.{name}",
                        f"metadata.{name}",
                        f"rag.{name}",
                        f"model_paths.{name}",
                        f"mineru.{name}",
                        f"multilevel_chunker.{name}",
                    ]
                    
                    for nested_name in nested_names:
                        try:
                            if hasattr(main, "get_nested"):
                                return main.get_nested(nested_name)
                            else:
                                # fallback: split path and traverse
                                obj = main
                                for part in nested_name.split('.'):
                                    obj = getattr(obj, part)
                                return obj
                        except (AttributeError, KeyError):
                            continue  # Try the next path
                except Exception:
                    pass  # Fall through to final error
                
                # If no alias matches and no nested attribute exists, raise AttributeError
                raise AttributeError(f"{self.__class__.__name__} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to main config using dot-path notation."""
        # Handle dot-notation access by delegating to main config
        if "." in name:
            try:
                main = super().__getattribute__("main")
            except Exception:
                return super().__setattr__(name, value)
            
            if hasattr(main, "set_nested"):
                main.set_nested(name, value)
                return
            # fallback
            parts = name.split('.')
            obj = main
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            return
        
        # For 'main' attribute and class attributes, set directly
        if name == "main" or name in self.__class__.__dict__:
            super().__setattr__(name, value)
            return
        
        # Handle alias mappings for backward compatibility first
        # Map common aliases to their actual paths
        # Note: These are the aliases for legacy flattened access patterns
        if name in _ALIAS_MAP:
            actual_path = _ALIAS_MAP[name]
            try:
                main = super().__getattribute__("main")
            except Exception:
                return super().__setattr__(name, value)
            
            if hasattr(main, "set_nested"):
                main.set_nested(actual_path, value)
                return
            else:
                # fallback: split path and set
                parts = actual_path.split('.')
                obj = main
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
                return
        
        # For other attributes, check if they exist in main config
        try:
            main = super().__getattribute__("main")
        except Exception:
            super().__setattr__(name, value)
            return
        
        # Only try to set direct attributes if they're not in alias mapping
        # This avoids the issue with properties that only have getters
        if hasattr(main, name) and name not in _ALIAS_MAP:
            setattr(main, name, value)
            return
        
        # Handle special cases like huggingface_token
        if name == 'huggingface_token':
            setattr(self, '_huggingface_token', value)
            return
        
        # If the attribute doesn't exist anywhere, let the default behavior handle it
        # which might raise an error for Pydantic models - this is expected behavior
        super().__setattr__(name, value)

    @property
    def config_file_path(self) -> str:
        """Путь к файлу конфигурации."""
        from config.resource_path import resource_path
        return str(resource_path("config/config.json"))

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