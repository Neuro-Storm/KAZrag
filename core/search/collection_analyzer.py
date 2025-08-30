"""Модуль для анализа коллекций Qdrant."""

import logging
from typing import Tuple

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class CollectionAnalyzer:
    """Класс для анализа коллекций Qdrant."""
    
    @staticmethod
    def analyze_collection(client: QdrantClient, collection_name: str) -> Tuple[bool, bool, str]:
        """
        Анализирует коллекцию Qdrant для определения типов векторов.
        
        Args:
            client (QdrantClient): Клиент Qdrant.
            collection_name (str): Название коллекции.
            
        Returns:
            Tuple[bool, bool, str]: (has_dense, has_sparse, sparse_vector_name)
        """
        has_dense = False
        has_sparse = False
        sparse_vector_name = "sparse_vector"
        
        try:
            coll_info = client.get_collection(collection_name)
            logger.debug(f"Collection info for '{collection_name}': {coll_info}")
            
            # Получаем параметры коллекции
            collection_config = getattr(coll_info, 'config', None)
            if collection_config:
                params = getattr(collection_config, 'params', None)
                if params:
                    # Проверяем наличие dense векторов
                    vectors = getattr(params, 'vectors', {})
                    # Векторы могут быть представлены по-разному
                    if isinstance(vectors, dict):
                        has_dense = len(vectors) > 0
                        logger.debug(f"Dense vectors config (dict): {vectors}")
                    elif hasattr(vectors, 'size'):  # Это VectorParams
                        has_dense = True
                        logger.debug(f"Dense vectors config (VectorParams): {vectors}")
                    else:
                        has_dense = bool(vectors)
                        logger.debug(f"Dense vectors config (other): {vectors}")
                    
                    # Проверяем наличие sparse векторов в разных возможных местах
                    sparse_vectors = getattr(params, 'sparse_vectors', {})
                    if not sparse_vectors:
                        sparse_vectors = getattr(params, 'sparse_vectors_config', {})
                    
                    has_sparse = bool(sparse_vectors) if isinstance(sparse_vectors, dict) else False
                    logger.debug(f"Sparse vectors config: {sparse_vectors}")
                    
                    # Сохраняем имя sparse вектора для использования при поиске
                    if has_sparse and isinstance(sparse_vectors, dict):
                        sparse_vector_names = list(sparse_vectors.keys())
                        if sparse_vector_names:
                            sparse_vector_name = sparse_vector_names[0]  # Берем первое имя
                        else:
                            sparse_vector_name = "sparse_vector"
                    else:
                        sparse_vector_name = "sparse_vector"
                        
                    logger.debug(f"Analysis result: has_dense={has_dense}, has_sparse={has_sparse}, sparse_vector_name={sparse_vector_name}")
                else:
                    # По умолчанию предполагаем, что есть dense векторы
                    has_dense = True
                    sparse_vector_name = "sparse_vector"
                    logger.debug("Using default values: has_dense=True, has_sparse=False")
            else:
                # По умолчанию предполагаем, что есть dense векторы
                has_dense = True
                sparse_vector_name = "sparse_vector"
                logger.debug("Using default values: has_dense=True, has_sparse=False")
        except Exception as e:
            logger.debug(f"Не удалось получить информацию о коллекции '{collection_name}': {e}")
            # По умолчанию предполагаем, что есть dense векторы (старое поведение)
            has_dense = True
            sparse_vector_name = "sparse_vector"
            logger.debug("Using default values due to error: has_dense=True, has_sparse=False")
            
        return has_dense, has_sparse, sparse_vector_name


class SearchError(Exception):
    """Исключение для ошибок поиска."""
    pass