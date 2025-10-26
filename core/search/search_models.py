"""
Модели Pydantic для поиска в проекте KAZrag.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator


class SearchQuery(BaseModel):
    """Модель для параметров поискового запроса."""
    query: str = Field(..., min_length=1, max_length=1000, description="Поисковый запрос")
    k: int = Field(default=5, ge=1, le=100, description="Количество результатов")
    search_mode: str = Field(default="dense", pattern=r"^(dense|sparse|hybrid)$", description="Режим поиска")
    collection_name: Optional[str] = Field(default=None, min_length=1, max_length=100, description="Название коллекции")
    use_hybrid: bool = Field(default=False, description="Использовать гибридный поиск")
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Вес для dense в гибридном поиске")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Фильтр по метаданным")

    @field_validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Запрос не может быть пустым')
        return v.strip()

    @field_validator('collection_name')
    def validate_collection_name(cls, v):
        if v is not None:
            if not v or not v.strip():
                raise ValueError('Название коллекции не может быть пустым')
        return v


class SearchResult(BaseModel):
    """Модель для результата поиска."""
    content: str = Field(default="", description="Содержимое результата")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Метаданные результата")
    original_score: float = Field(default=0.0, description="Оригинальная оценка результата")
    source: Optional[str] = Field(default=None, description="Источник результата")
    micro_contents: Optional[List[str]] = Field(default=None, description="Содержимое микро-чанков")
    
    @field_validator('content')
    def validate_content(cls, v):
        return v or ""


class SearchFilter(BaseModel):
    """Модель для фильтра по метаданным."""
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Фильтр по метаданным")
    
    @field_validator('metadata_filter')
    def validate_metadata_filter(cls, v):
        if v is None:
            return v
        
        # Проверяем, что фильтр не пустой
        if not isinstance(v, dict):
            raise ValueError('metadata_filter должен быть словарем')
        
        # Проверяем корректность структуры фильтра
        for key, value in v.items():
            if isinstance(value, dict):
                # Проверяем сложные условия (например, {"$gt": 2020})
                for op, op_value in value.items():
                    if op not in ["$gt", "$gte", "$lt", "$lte", "$contains"]:
                        raise ValueError(f'Неподдерживаемая операция фильтрации: {op}')
            # Простые условия (равенство) проходят без дополнительной проверки
        
        return v