"""Модуль для выполнения поиска."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range

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
                
            # Обрабатываем результаты для извлечения содержимого чанков
            processed_results = []
            for doc, score in results:
                # Извлекаем дополнительную информацию из metadata
                content = getattr(doc, 'page_content', '')
                metadata = getattr(doc, 'metadata', {})
                
                # Для многоуровневых чанков содержимое может быть в payload
                if not content and hasattr(doc, 'payload'):
                    payload = doc.payload
                    content = payload.get('content', '')
                    # Если content все еще пустой, попробуем другие поля
                    if not content:
                        content = payload.get('page_content', '')
                
                # Если content все еще пустой, попробуем получить из __dict__
                if not content and hasattr(doc, '__dict__'):
                    doc_dict = doc.__dict__
                    content = doc_dict.get('page_content', '')
                    if not content:
                        content = doc_dict.get('content', '')
                
                # Создаем расширенный объект результата
                extended_result = {
                    'content': content if content is not None else '',
                    'metadata': metadata,
                    'score': score
                }
                
                # Если это многоуровневый чанк, добавляем информацию о микро-чанках
                if 'micro_contents' in metadata:
                    extended_result['micro_contents'] = metadata['micro_contents']
                elif hasattr(doc, 'payload') and 'micro_contents' in doc.payload:
                    extended_result['micro_contents'] = doc.payload['micro_contents']
                    
                # Добавляем source если есть
                if 'source' in metadata:
                    extended_result['source'] = metadata['source']
                elif hasattr(doc, 'payload') and 'source' in doc.payload:
                    extended_result['source'] = doc.payload['source']
                elif 'source' not in extended_result and 'source' in metadata:
                    extended_result['source'] = metadata.get('source', '')
                    
                processed_results.append((extended_result, score))
                
            logger.debug(f"Search returned {len(processed_results)} results")
            return processed_results, None
            
        except Exception as e:
            logger.exception(f"Ошибка при поиске: {e}")
            return [], str(e)