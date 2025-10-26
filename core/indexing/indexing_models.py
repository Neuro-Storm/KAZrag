"""
Модели Pydantic для индексации в проекте KAZrag.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path


class IndexingParams(BaseModel):
    """Модель для параметров индексации."""
    folder_path: str = Field(default="./data_to_index", min_length=1, description="Путь к папке с документами")
    collection_name: str = Field(default="final-dense-collection", min_length=1, max_length=100, description="Имя коллекции")
    index_dense: bool = Field(default=True, description="Индексировать dense векторы")
    index_bm25: bool = Field(default=False, description="Индексировать sparse векторы (BM25)")
    index_hybrid: bool = Field(default=False, description="Индексировать гибридные векторы")
    chunk_size: int = Field(default=500, ge=1, le=10000, description="Размер чанка")
    chunk_overlap: int = Field(default=100, ge=0, le=1000, description="Перекрытие чанков")
    chunking_strategy: str = Field(default="character", pattern=r"^(character|paragraph|sentence)$", description="Стратегия чанкинга")
    
    # Параметры для чанкинга по абзацам
    paragraphs_per_chunk: int = Field(default=3, ge=1, le=50, description="Абзацев в чанке")
    paragraph_overlap: int = Field(default=1, ge=0, le=10, description="Перекрытие абзацев")
    
    # Параметры для чанкинга по предложениям
    sentences_per_chunk: int = Field(default=5, ge=1, le=100, description="Предложений в чанке")
    sentence_overlap: int = Field(default=1, ge=0, le=20, description="Перекрытие предложений")
    
    # Параметры для многоуровневого чанкинга
    use_multilevel_chunking: bool = Field(default=False, description="Использовать многоуровневый чанкинг")
    
    # Параметры макро-чанкинга
    multilevel_macro_strategy: str = Field(default="character", pattern=r"^(character|paragraph|sentence)$", description="Стратегия макро-чанкинга")
    multilevel_macro_chunk_size: int = Field(default=10000, ge=100, le=100000, description="Размер макро-чанка")
    multilevel_macro_chunk_overlap: int = Field(default=1000, ge=0, le=10000, description="Перекрытие макро-чанков")
    multilevel_macro_paragraphs_per_chunk: int = Field(default=5, ge=1, le=50, description="Абзацев в макро-чанке")
    multilevel_macro_paragraph_overlap: int = Field(default=1, ge=0, le=10, description="Перекрытие абзацев в макро-чанке")
    multilevel_macro_sentences_per_chunk: int = Field(default=10, ge=1, le=100, description="Предложений в макро-чанке")
    multilevel_macro_sentence_overlap: int = Field(default=1, ge=0, le=20, description="Перекрытие предложений в макро-чанке")
    
    # Параметры микро-чанкинга
    multilevel_micro_strategy: str = Field(default="character", pattern=r"^(character|paragraph|sentence)$", description="Стратегия микро-чанкинга")
    multilevel_micro_chunk_size: int = Field(default=1000, ge=100, le=10000, description="Размер микро-чанка")
    multilevel_micro_chunk_overlap: int = Field(default=100, ge=0, le=1000, description="Перекрытие микро-чанков")
    multilevel_micro_paragraphs_per_chunk: int = Field(default=3, ge=1, le=50, description="Абзацев в микро-чанке")
    multilevel_micro_paragraph_overlap: int = Field(default=1, ge=0, le=10, description="Перекрытие абзацев в микро-чанке")
    multilevel_micro_sentences_per_chunk: int = Field(default=5, ge=1, le=100, description="Предложений в микро-чанке")
    multilevel_micro_sentence_overlap: int = Field(default=1, ge=0, le=20, description="Перекрытие предложений в микро-чанке")
    
    # Настройки производительности
    embedding_batch_size: int = Field(default=32, ge=1, le=128, description="Размер батча для эмбеддингов")
    indexing_batch_size: int = Field(default=50, ge=1, le=500, description="Размер батча для индексации")
    force_recreate: bool = Field(default=True, description="Пересоздавать коллекцию")
    memory_threshold: int = Field(default=500 * 1024 * 1024, ge=1024 * 1024, description="Порог памяти в байтах")
    
    # Дополнительные настройки
    sparse_embedding: Optional[str] = Field(default="Qdrant/bm25", description="Sparse эмбеддинг")
    is_indexed: bool = Field(default=False, description="Флаг, указывающий, была ли выполнена индексация")
    
    @field_validator('folder_path')
    def validate_folder_path(cls, v):
        if not v or not v.strip():
            raise ValueError('Путь к папке не может быть пустым')
        return v.strip()
    
    @field_validator('collection_name')
    def validate_collection_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Имя коллекции не может быть пустым')
        if len(v) > 100:
            raise ValueError('Имя коллекции не может быть длиннее 100 символов')
        return v.strip()
    
    @model_validator(mode='after')
    def validate_multilevel_params_consistency(self):
        """Проверяем логическую согласованность параметров многоуровневого чанкинга."""
        if self.use_multilevel_chunking:
            # Проверяем, что размеры микро-чанков меньше размеров макро-чанков
            if self.multilevel_macro_strategy == 'character' and self.multilevel_micro_strategy == 'character':
                if self.multilevel_micro_chunk_size >= self.multilevel_macro_chunk_size:
                    raise ValueError('Размер микро-чанка должен быть меньше размера макро-чанка при чанкинге по символам')
            
            # Проверяем, что перекрытия корректны
            if self.multilevel_macro_chunk_overlap >= self.multilevel_macro_chunk_size:
                raise ValueError('Перекрытие макро-чанков должно быть меньше размера макро-чанка')
            
            if self.multilevel_micro_chunk_overlap >= self.multilevel_micro_chunk_size:
                raise ValueError('Перекрытие микро-чанков должно быть меньше размера микро-чанка')
        
        # Проверяем, что перекрытие чанков меньше размера чанка
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('Перекрытие чанков должно быть меньше размера чанка')
        
        return self


class DocumentIndexingResult(BaseModel):
    """Модель для результата индексации документа."""
    success: bool = Field(description="Успешно ли проиндексирован документ")
    document_id: str = Field(description="ID документа")
    chunks_count: int = Field(description="Количество созданных чанков")
    error: Optional[str] = Field(default=None, description="Сообщение об ошибке, если была")
    
    @field_validator('chunks_count')
    def validate_chunks_count(cls, v):
        if v < 0:
            raise ValueError('Количество чанков не может быть отрицательным')
        return v


class MultiLevelChunkData(BaseModel):
    """Модель для данных многоуровневого чанка."""
    macro_chunk: Dict[str, Any] = Field(description="Макро-чанк")
    micro_chunks: List[str] = Field(description="Микро-чанки")
    chunk_index: int = Field(ge=0, description="Индекс чанка")
    
    @field_validator('chunk_index')
    def validate_chunk_index(cls, v):
        if v < 0:
            raise ValueError('Индекс чанка не может быть отрицательным')
        return v
    
    @field_validator('micro_chunks')
    def validate_micro_chunks(cls, v):
        if not v:
            raise ValueError('Должен быть хотя бы один микро-чанк')
        return v