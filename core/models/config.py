"""
Вложенные модели Pydantic для конфигурации KAZrag.
"""
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class DoclingConfig(BaseModel):
    """Конфигурация для Docling - обработки различных форматов документов."""
    
    # Основные настройки обработки
    use_ocr: bool = Field(default=True, description="Использовать OCR для изображений")
    force_full_ocr: bool = Field(default=False, description="Принудительно использовать OCR для всего PDF документа, включая внедренный текст")
    use_tables: bool = Field(default=True, description="Обрабатывать таблицы")
    use_formulas: bool = Field(default=True, description="Обрабатывать формулы")
    model_backend: str = Field(default="huggingface", pattern=r"^(huggingface|local|api)$", description="Бэкенд для моделей")
    ocr_model: str = Field(default="easyocr", description="Модель OCR")
    ocr_lang: Union[str, List[str]] = Field(default=["ru"], description="Язык(и) OCR")
    
    # Ограничения на размер файлов
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Максимальный размер файла в МБ для обработки")
    
    @field_validator('ocr_lang', mode='before')
    @classmethod
    def convert_lang_to_list(cls, v):
        """Преобразует строку языка в список, если необходимо и проверяет формат языков."""
        # Если строка - конвертируем в список с одним элементом
        if isinstance(v, str):
            langs = [v]
        elif isinstance(v, list):
            langs = v
        else:
            raise ValueError("ocr_lang must be a string or list of strings")
        
        # Валидация каждого языка в списке
        valid_langs = {"ru", "en", "kk", "east_slavic"}
        for lang in langs:
            if lang not in valid_langs:
                raise ValueError(f"Invalid language: {lang}. Must be one of {valid_langs}")
        
        return langs
    
    # Настройки обработки
    images_dir: str = Field(default="images", description="Поддиректория для изображений")
    table_mode: str = Field(default="accurate", pattern=r"^(fast|accurate)$", description="Режим обработки таблиц")
    enable_page_images: bool = Field(default=True, description="Генерировать ли изображения страниц")
    table_detection_advanced: bool = Field(default=True, ge=0, le=1, description="Расширенное обнаружение таблиц")
    formula_detection_advanced: bool = Field(default=True, ge=0, le=1, description="Расширенное обнаружение формул")
    
    # Режим бэкенда и устройство
    backend: str = Field(default="classic", pattern=r"^(classic|granite)$", description="Режим Docling backend")
    device: str = Field(default="auto", pattern=r"^(cpu|gpu|auto)$", description="Устройство для обработки")
    
    # Настройки для Granite
    granite_models_dir: Path = Field(default=Path("./models/granite"), description="Папка для Granite-моделей")


class SearchConfig(BaseModel):
    """Конфигурация для поиска."""
    
    # Настройки поиска
    default_k: int = Field(default=5, ge=1, description="Количество результатов по умолчанию")
    default_collection: str = Field(default="final-dense-collection", description="Коллекция по умолчанию")
    default_device: str = Field(default="cpu", description="Устройство по умолчанию")
    default_type: str = Field(default="dense", pattern=r"^(dense|sparse|hybrid)$", description="Тип поиска по умолчанию")
    default_use_reranker: bool = Field(default=True, description="Использовать reranker по умолчанию")
    
    # Гибридный поиск
    use_hybrid: bool = Field(default=False, description="Использовать гибридный поиск (dense + sparse)")
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Вес для dense в гибридном поиске")


class RerankerConfig(BaseModel):
    """Конфигурация для reranker."""
    
    enabled: bool = Field(default=False, description="Включить reranker")
    model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Модель reranker")
    top_k: int = Field(default=5, ge=1, description="Количество топ-результатов для rerank")


class GGUFConfig(BaseModel):
    """Конфигурация для GGUF моделей."""
    
    model_n_ctx: int = Field(default=512, ge=1, description="Контекст GGUF модели")


class EmbeddingConfig(BaseModel):
    """Конфигурация для эмбеддингов."""
    
    # Основные настройки
    current_hf_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Текущая модель HF")
    hf_model_history: List[str] = Field(default=["sentence-transformers/all-MiniLM-L6-v2"], description="История моделей")
    device: str = Field(default="auto", description="Устройство для эмбеддингов")
    
    # Настройки производительности
    batch_size: int = Field(default=32, ge=1, description="Размер батча для эмбеддингов")
    gguf: GGUFConfig = Field(default_factory=GGUFConfig, description="Конфигурация GGUF")


class QdrantConfig(BaseModel):
    """Конфигурация для Qdrant."""
    
    # Подключение
    url: str = Field(default="http://localhost:6333", description="URL Qdrant")
    
    # Настройки подключения
    retry_attempts: int = Field(default=3, ge=1, description="Количество попыток подключения")
    retry_wait_time: int = Field(default=2, ge=1, description="Время ожидания между попытками")


class BM25Config(BaseModel):
    """Конфигурация для BM25."""
    
    enabled: bool = Field(default=True, description="Включить BM25")
    sparse_vector_name: str = Field(default="bm25_text", description="Имя sparse векторного поля")
    tokenizer: str = Field(default="word", pattern=r"^(word|whitespace|prefix)$", description="Тип токенизатора")
    min_token_len: int = Field(default=2, ge=1, description="Минимальная длина токена")


class RAGConfig(BaseModel):
    """Конфигурация для RAG (Retrieval-Augmented Generation)."""
    
    enabled: bool = Field(default=False, description="Включить RAG")
    model_path: str = Field(default="models/Qwen3-4B-Instruct-2507-Q8_0.gguf", description="Путь к GGUF-модели LLM")
    system_prompt: str = Field(
        default="You are a helpful assistant. Use the following context to answer the user's question accurately.", 
        description="Системный промпт"
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Количество топ-результатов для контекста")
    max_tokens: int = Field(default=512, ge=1, description="Максимум токенов в генерации")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Температура генерации")
    context_size: int = Field(default=4096, ge=512, description="Размер контекста для модели")
    gpu_layers: int = Field(default=-1, ge=-1, description="Количество слоев для GPU (-1 = все слои)")
    threads: int = Field(default=4, ge=1, description="Количество потоков для генерации")
    batch_size: int = Field(default=512, ge=1, description="Размер батча для обработки")
    beam_size: int = Field(default=1, ge=1, description="Размер beam для генерации")
    
    @field_validator('top_k')
    def validate_top_k_le_max_tokens(cls, v, values):
        """Проверяем, что top_k <= max_tokens."""
        # Обратите внимание, что в field_validator у нас нет доступа к другим полям через values
        # Вместо этого будем использовать отдельный validator для модели
        return v


class IndexingConfig(BaseModel):
    """Конфигурация для индексации."""
    
    # Пути и общие настройки
    folder_path: str = Field(default="./data_to_index", description="Путь к папке с документами для индексации")
    collection_name: str = Field(default="final-dense-collection", description="Имя коллекции")
    
    # Настройки режимов индексации
    index_dense: bool = Field(default=True, description="Индексировать dense векторы")
    index_bm25: bool = Field(default=False, description="Индексировать sparse векторы (BM25)")
    index_hybrid: bool = Field(default=False, description="Индексировать гибридные векторы")
    
    # Настройки чанкинга
    chunk_size: int = Field(default=500, ge=1, description="Размер чанка")
    chunk_overlap: int = Field(default=100, ge=0, description="Перекрытие чанков")
    chunking_strategy: str = Field(default="character", pattern=r"^(character|paragraph|sentence)$", description="Стратегия чанкинга")
    
    # Настройки чанкинга по абзацам
    paragraphs_per_chunk: int = Field(default=3, ge=1, description="Абзацев в чанке")
    paragraph_overlap: int = Field(default=1, ge=0, description="Перекрытие абзацев")
    
    # Настройки чанкинга по предложениям
    sentences_per_chunk: int = Field(default=5, ge=1, description="Предложений в чанке")
    sentence_overlap: int = Field(default=1, ge=0, description="Перекрытие предложений")
    
    # Настройки многоуровневого чанкинга
    use_multilevel_chunking: bool = Field(default=False, description="Использовать многоуровневый чанкинг")
    
    # Настройки производительности
    batch_size: int = Field(default=50, ge=1, description="Размер батча для индексации")
    force_recreate: bool = Field(default=True, description="Пересоздавать коллекцию")
    memory_threshold: int = Field(default=500 * 1024 * 1024, ge=1, description="Порог памяти в байтах")
    
    # Дополнительные настройки
    sparse_embedding: Optional[str] = Field(default="Qdrant/bm25", description="Sparse эмбеддинг")
    is_indexed: bool = Field(default=False, description="Флаг, указывающий, была ли выполнена индексация")


class MultilevelChunkerConfig(BaseModel):
    """Конфигурация для многоуровневого чанкинга."""
    
    # Настройки макро-чанкинга
    macro_strategy: str = Field(default="character", pattern=r"^(character|paragraph|sentence)$", description="Стратегия макро-чанкинга")
    macro_chunk_size: int = Field(default=10000, ge=100, description="Размер макро-чанка")
    macro_chunk_overlap: int = Field(default=1000, ge=0, description="Перекрытие макро-чанков")
    macro_paragraphs_per_chunk: int = Field(default=5, ge=1, description="Абзацев в макро-чанке")
    macro_paragraph_overlap: int = Field(default=1, ge=0, description="Перекрытие абзацев в макро-чанке")
    macro_sentences_per_chunk: int = Field(default=10, ge=1, description="Предложений в макро-чанке")
    macro_sentence_overlap: int = Field(default=1, ge=0, description="Перекрытие предложений в макро-чанке")
    
    # Настройки микро-чанкинга
    micro_strategy: str = Field(default="character", pattern=r"^(character|paragraph|sentence)$", description="Стратегия микро-чанкинга")
    micro_chunk_size: int = Field(default=1000, ge=100, description="Размер микро-чанка")
    micro_chunk_overlap: int = Field(default=100, ge=0, description="Перекрытие микро-чанков")
    micro_paragraphs_per_chunk: int = Field(default=3, ge=1, description="Абзацев в микро-чанке")
    micro_paragraph_overlap: int = Field(default=1, ge=0, description="Перекрытие абзацев в микро-чанке")
    micro_sentences_per_chunk: int = Field(default=5, ge=1, description="Предложений в микро-чанке")
    micro_sentence_overlap: int = Field(default=1, ge=0, description="Перекрытие предложений в микро-чанке")


class MetadataConfig(BaseModel):
    """Конфигурация для метаданных."""
    
    enable_extraction: bool = Field(default=True, description="Включить извлечение метаданных")
    custom_fields: Dict[str, Any] = Field(default={}, description="Пользовательские поля метаданных")
    extract_pdf: bool = Field(default=True, description="Извлекать метаданные из PDF")
    extract_image: bool = Field(default=True, description="Извлекать метаданные из изображений")
    extract_docx: bool = Field(default=True, description="Извлекать метаданные из DOCX")


class CacheConfig(BaseModel):
    """Конфигурация для кэширования."""
    
    config_cache_ttl: int = Field(default=60, ge=1, description="TTL кэша конфигурации")
    qdrant_client_cache_ttl: int = Field(default=60, ge=1, description="TTL кэша клиента Qdrant")
    collections_cache_ttl: int = Field(default=60, ge=1, description="TTL кэша коллекций")


class ModelPathsConfig(BaseModel):
    """Конфигурация для путей к моделям."""
    
    local_models_path: Path = Field(default=Path("./models"), description="Путь к локальным моделям")
    huggingface_cache_path: Path = Field(default=Path("./models/huggingface_cache"), description="Путь к кэшу HuggingFace")
    easyocr_models_path: Path = Field(default=Path("./models/easyocr"), description="Путь к моделям EasyOCR")
    fastembed_cache_path: Path = Field(default=Path("./models/fastembed"), description="Путь к кэшу FastEmbed")
    use_local_only: bool = Field(default=True, description="Флаг для строгого оффлайна")
    auto_download_models: bool = Field(default=True, description="Флаг автоматического скачивания моделей")


class MinerUConfig(BaseModel):
    """Конфигурация для MinerU (для совместимости)."""
    
    # Настройки для совместимости
    input_pdf_dir: str = Field(default="./pdfs_to_process", description="Входная директория PDF")
    output_md_dir: str = Field(default="./data_to_index", description="Выходная директория MD")
    enable_formula_parsing: bool = Field(default=False, description="Включить парсинг формул")
    enable_table_parsing: bool = Field(default=False, description="Включить парсинг таблиц")
    model_source: str = Field(default="huggingface", description="Источник моделей")
    models_dir: str = Field(default="", description="Директория моделей")
    backend: str = Field(default="pipeline", description="Бэкенд")
    method: str = Field(default="auto", description="Метод")
    lang: str = Field(default="east_slavic", description="Язык")
    sglang_url: str = Field(default="", description="URL SGLang")
    subprocess_timeout: int = Field(default=600, ge=1, description="Таймаут подпроцесса")


class MainConfig(BaseModel):
    """Главная конфигурация приложения с вложенными объектами."""
    
    # Вложенные конфигурации
    docling: DoclingConfig = Field(default_factory=DoclingConfig, description="Конфигурация Docling")
    search: SearchConfig = Field(default_factory=SearchConfig, description="Конфигурация поиска")
    reranker: RerankerConfig = Field(default_factory=RerankerConfig, description="Конфигурация reranker")
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig, description="Конфигурация эмбеддингов")
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig, description="Конфигурация Qdrant")
    bm25: BM25Config = Field(default_factory=BM25Config, description="Конфигурация BM25")
    rag: RAGConfig = Field(default_factory=RAGConfig, description="Конфигурация RAG")
    indexing: IndexingConfig = Field(default_factory=IndexingConfig, description="Конфигурация индексации")
    multilevel_chunker: MultilevelChunkerConfig = Field(default_factory=MultilevelChunkerConfig, description="Конфигурация многоуровневого чанкинга")
    metadata: MetadataConfig = Field(default_factory=MetadataConfig, description="Конфигурация метаданных")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Конфигурация кэширования")
    model_paths: ModelPathsConfig = Field(default_factory=ModelPathsConfig, description="Конфигурация путей к моделям")
    mineru: MinerUConfig = Field(default_factory=MinerUConfig, description="Конфигурация MinerU (совместимость)")
    
    def get_nested(self, path: str) -> Any:
        """Get nested attribute using dot notation path."""
        obj: Any = self
        for part in path.split('.'):
            if obj is None:
                raise AttributeError(f"Cannot traverse '{part}' because parent is None (path='{path}')")
            # dict-like support
            if isinstance(obj, dict):
                try:
                    obj = obj[part]
                except KeyError:
                    raise KeyError(f"Key '{part}' not found while resolving '{path}'")
            else:
                if not hasattr(obj, part):
                    raise AttributeError(f"Attribute '{part}' not found while resolving '{path}'")
                obj = getattr(obj, part)
        return obj

    def set_nested(self, path: str, value: Any) -> None:
        """Set nested attribute using dot notation path."""
        parts = path.split('.')
        if not parts:
            raise ValueError("Empty path")
        obj: Any = self
        for part in parts[:-1]:
            if obj is None:
                raise AttributeError(f"Cannot traverse '{part}' because parent is None (path='{path}')")
            if isinstance(obj, dict):
                if part not in obj:
                    raise KeyError(f"Key '{part}' not found while resolving '{path}'")
                obj = obj[part]
            else:
                if not hasattr(obj, part):
                    raise AttributeError(f"Attribute '{part}' not found while resolving '{path}'")
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)
    
    # Валидатор для проверки ограничений RAG
    @model_validator(mode='after')
    def validate_rag_constraints(self):
        """Проверяем, что top_k <= max_tokens для RAG."""
        if self.rag.top_k > self.rag.max_tokens:
            raise ValueError("rag_top_k must be less than or equal to rag_max_tokens")
        return self