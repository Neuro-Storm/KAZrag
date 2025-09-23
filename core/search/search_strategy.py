"""Модуль для определения стратегии поиска."""

import logging
from typing import Tuple, List, Optional, Dict, Any

from qdrant_client.http.models import RecommendRequest, SearchRequest, ScoredPoint

from core.search.collection_analyzer import CollectionAnalyzer

logger = logging.getLogger(__name__)


class SearchStrategy:
    """Класс для определения стратегии поиска."""
    
    def __init__(self, client, collection_name: str, embedder, sparse_emb=None):
        """
        Инициализирует стратегию поиска.
        
        Args:
            client: Клиент Qdrant.
            collection_name (str): Название коллекции.
            embedder: Эмбеддер для dense векторов.
            sparse_emb: Эмбеддер для sparse векторов (опционально).
        """
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder
        self.sparse_emb = sparse_emb
        
        # Анализируем коллекцию
        analyzer = CollectionAnalyzer()
        self.has_dense, self.has_sparse, self.sparse_vector_name = analyzer.analyze_collection(
            client, collection_name
        )
    
    def determine_search_mode(self, hybrid: bool) -> str:
        """
        Определяет режим поиска на основе параметров.
        
        Args:
            hybrid (bool): Требуется ли гибридный поиск.
            
        Returns:
            str: Режим поиска ("hybrid", "dense", "sparse").
        """
        logger.info(f"Determining search mode: hybrid={hybrid}, has_dense={self.has_dense}, has_sparse={self.has_sparse}")
        
        # Если запрошен гибридный поиск, проверяем возможность его выполнения
        if hybrid:
            if not self.has_sparse:
                logger.warning(f"Коллекция '{self.collection_name}' не содержит sparse-векторов. Выполняется только dense search.")
                return "dense"
            elif not self.has_dense:
                logger.warning(f"Коллекция '{self.collection_name}' не содержит dense-векторов. Выполняется только sparse search.")
                return "sparse"
            else:
                logger.info("Hybrid search mode selected")
                return "hybrid"
        elif self.has_sparse and not self.has_dense:
            # Только sparse векторы
            logger.info("Sparse-only search mode selected")
            return "sparse"
        else:
            # По умолчанию dense (или гибрид невозможен)
            logger.info("Dense search mode selected (default)")
            return "dense"
    
    def create_qdrant_searcher(self, search_mode: str):
        """
        Возвращает настроенный клиент и параметры для поиска.
        
        Args:
            search_mode (str): Режим поиска.
            
        Returns:
            Tuple[QdrantClient, str, Optional[str], Optional[Any]]: (client, mode, vector_name, sparse_params)
        """
        try:
            if search_mode == "hybrid":
                logger.info(f"Configuring hybrid search for collection '{self.collection_name}'")
                sparse_params = {
                    "sparse_vector_name": self.sparse_vector_name,
                    "sparse_embedding": self.sparse_emb
                }
                return self.client, "hybrid", "dense_vector", sparse_params
            elif search_mode == "sparse":
                logger.info(f"Configuring sparse-only search for collection '{self.collection_name}' with sparse_vector_name='{self.sparse_vector_name}'")
                return self.client, "sparse", None, {
                    "sparse_vector_name": self.sparse_vector_name,
                    "sparse_embedding": self.sparse_emb
                }
            else:
                # Dense-only поиск (по умолчанию)
                logger.info(f"Configuring dense-only search for collection '{self.collection_name}'")
                return self.client, "dense", "dense_vector", None
        except Exception as e:
            # Если возникла ошибка, проверим, не связана ли она с конфликтом размерностей
            err_msg = str(e)
            if "dimensions" in err_msg and "force_recreate" in err_msg:
                logger.warning(f"Конфликт размерностей векторов в коллекции '{self.collection_name}'. Попытка пересоздать коллекцию.")
                # В новой версии нативного клиента этот блок остается для совместимости, но теперь не нужен для нативного
                try:  # Этот блок остается для совместимости, но теперь не нужен для нативного
                    # Удаляем существующую коллекцию
                    self.client.delete_collection(self.collection_name)
                    logger.info(f"Коллекция '{self.collection_name}' удалена. Повторная попытка создания QdrantVectorStore.")
                    
                    # Повторная попытка
                    if search_mode == "hybrid":
                        sparse_params = {
                            "sparse_vector_name": self.sparse_vector_name,
                            "sparse_embedding": self.sparse_emb
                        }
                        return self.client, "hybrid", "dense_vector", sparse_params
                    elif search_mode == "sparse":
                        return self.client, "sparse", None, {
                            "sparse_vector_name": self.sparse_vector_name,
                            "sparse_embedding": self.sparse_emb
                        }
                    else:
                        return self.client, "dense", "dense_vector", None
                except Exception as recreate_error:
                    logger.exception(f"Не удалось пересоздать коллекцию '{self.collection_name}': {recreate_error}")
                    raise recreate_error
            else:
                # Если ошибка не связана с конфликтом размерностей, пробрасываем её дальше
                logger.exception(f"Ошибка при создании QdrantVectorStore: {e}")
                raise e