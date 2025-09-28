"""Модуль для анализа коллекций Qdrant."""

import logging
from typing import Tuple

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class CollectionAnalyzer:
    """Класс для анализа коллекций Qdrant."""
    
    @staticmethod
    def analyze_collection(client: QdrantClient, collection_name: str, sparse_name: str = None) -> Tuple[bool, bool, str]:
        """
        Анализирует коллекцию Qdrant для определения типов векторов.
        
        Args:
            client (QdrantClient): Клиент Qdrant.
            collection_name (str): Название коллекции.
            sparse_name (str, optional): Имя sparse вектора. По умолчанию None.
            
        Returns:
            Tuple[bool, bool, str]: (has_dense, has_sparse, sparse_vector_name)
        """
        has_dense = False
        has_sparse = False
        sparse_vector_name = sparse_name or "sparse_vector"
        
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
                            logger.info(f"Detected sparse vector name: {sparse_vector_name}")
                        else:
                            sparse_vector_name = sparse_name or "sparse_vector"
                            logger.info(f"Using provided sparse vector name: {sparse_vector_name}")
                    else:
                        # Try to detect the sparse vector name from the collection info more thoroughly
                        try:
                            # Check if there are any sparse vectors in the collection's points structure
                            points_sample = client.scroll(
                                collection_name=collection_name,
                                limit=1,
                                with_payload=False,
                                with_vectors=True
                            )
                            if points_sample[0] if isinstance(points_sample, list) and len(points_sample) > 0 else False:
                                point = points_sample[0]
                                if hasattr(point, 'vector') and isinstance(point.vector, dict):
                                    vector_names = list(point.vector.keys())
                                    sparse_candidates = [name for name in vector_names if 'sparse' in name.lower() or 'bm25' in name.lower()]
                                    if sparse_candidates:
                                        sparse_vector_name = sparse_candidates[0]
                                        has_sparse = True
                                        logger.info(f"Detected sparse vector name from point vectors: {sparse_vector_name}")
                                    else:
                                        # Fallback to checking if 'bm25_text' exists
                                        if 'bm25_text' in vector_names:
                                            sparse_vector_name = 'bm25_text'
                                            has_sparse = True
                                            logger.info(f"Found 'bm25_text' sparse vector: {sparse_vector_name}")
                                        else:
                                            sparse_vector_name = sparse_name or "sparse_vector"
                                            logger.info(f"Using provided/fallback sparse vector name: {sparse_vector_name}")
                                else:
                                    sparse_vector_name = sparse_name or "sparse_vector"
                                    logger.info(f"Using provided/fallback sparse vector name: {sparse_vector_name}")
                        except Exception as e:
                            logger.debug(f"Could not detect sparse vector names from points: {e}")
                            sparse_vector_name = sparse_name or "sparse_vector"
                        
                    logger.debug(f"Analysis result: has_dense={has_dense}, has_sparse={has_sparse}, sparse_vector_name={sparse_vector_name}")
                else:
                    # По умолчанию предполагаем, что есть dense векторы
                    has_dense = True
                    sparse_vector_name = sparse_name or "sparse_vector"
                    logger.debug("Using default values: has_dense=True, has_sparse=False")
            else:
                # По умолчанию предполагаем, что есть dense векторы
                has_dense = True
                sparse_vector_name = sparse_name or "sparse_vector"
                logger.debug("Using default values: has_dense=True, has_sparse=False")
        except Exception as e:
            logger.debug(f"Не удалось получить информацию о коллекции '{collection_name}': {e}")
            # По умолчанию предполагаем, что есть dense векторы (старое поведение)
            has_dense = True
            sparse_vector_name = sparse_name or "sparse_vector"
            logger.debug("Using default values due to error: has_dense=True, has_sparse=False")
            
        return has_dense, has_sparse, sparse_vector_name


class SearchError(Exception):
    """Исключение для ошибок поиска."""
    pass