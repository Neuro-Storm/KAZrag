"""Модуль для выполнения поиска."""

import logging
from typing import List, Tuple, Any, Optional, Dict
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
from langchain_qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


class SearchExecutor:
    """Класс для выполнения поисковых запросов."""
    
    @staticmethod
    async def execute_search(
        qdrant: QdrantVectorStore, 
        query: str, 
        k: int, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Tuple[Any, float]], Optional[str]]:
        """
        Выполняет поиск с опциональной фильтрацией по метаданным.
        
        Args:
            qdrant (QdrantVectorStore): Настроенный экземпляр QdrantVectorStore.
            query (str): Поисковый запрос.
            k (int): Количество результатов.
            metadata_filter (Optional[Dict[str, Any]]): Фильтр по метаданным.
            
        Returns:
            Tuple[List[Tuple[Any, float]], Optional[str]]: (результаты поиска, ошибка)
        """
        try:
            # Выполняем поиск с опциональной фильтрацией по метаданным
            if metadata_filter:
                # Создаем фильтр для Qdrant
                must_conditions = []
                
                for key, value in metadata_filter.items():
                    # Обрабатываем различные типы условий
                    if isinstance(value, dict):
                        # Сложные условия (например, {"$gt": 2020})
                        for op, op_value in value.items():
                            if op == "$gt":
                                must_conditions.append(FieldCondition(
                                    key=f"metadata.{key}",
                                    range=Range(gt=op_value)
                                ))
                            elif op == "$gte":
                                must_conditions.append(FieldCondition(
                                    key=f"metadata.{key}",
                                    range=Range(gte=op_value)
                                ))
                            elif op == "$lt":
                                must_conditions.append(FieldCondition(
                                    key=f"metadata.{key}",
                                    range=Range(lt=op_value)
                                ))
                            elif op == "$lte":
                                must_conditions.append(FieldCondition(
                                    key=f"metadata.{key}",
                                    range=Range(lte=op_value)
                                ))
                            elif op == "$contains":
                                # Для массивов или строк
                                must_conditions.append(FieldCondition(
                                    key=f"metadata.{key}",
                                    match=MatchValue(value=op_value)
                                ))
                    else:
                        # Простое равенство
                        must_conditions.append(FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        ))
                
                search_filter = Filter(must=must_conditions)
                results = await qdrant.asimilarity_search_with_score(query, k=k, filter=search_filter)
            else:
                results = await qdrant.asimilarity_search_with_score(query, k=k)
                
            return results, None
            
        except Exception as e:
            logger.exception(f"Ошибка при поиске: {e}")
            return [], str(e)