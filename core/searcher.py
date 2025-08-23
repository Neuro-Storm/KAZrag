"""Модуль для выполнения поиска в Qdrant."""

import logging
from typing import List, Tuple, Any, Optional
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from core.qdrant_client import aget_qdrant_client
from config.settings import load_config, Config
from core.embeddings import get_dense_embedder, get_search_device

logger = logging.getLogger(__name__)


# Используем централизованные клиенты из core.qdrant_client


async def search_in_collection(query: str, collection_name: str, device: str, k: int = None, hybrid: bool = False, client = None) -> Tuple[List[Tuple[Any, float]], Optional[str]]:
    """
    Выполняет поиск в указанной коллекции Qdrant.
    
    Args:
        query (str): Поисковый запрос.
        collection_name (str): Название коллекции.
        device (str): Устройство для поиска ("cpu" или "cuda").
        k (int): Количество результатов. Если None, используется значение из конфигурации.
        hybrid (bool): Использовать hybrid search (dense + sparse).
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, создается новый.
        
    Returns:
        Tuple[List[Tuple[Any, float]], Optional[str]]: (результаты поиска, ошибка)
    """
    try:
        config: Config = load_config()
        
        # Если k не указано, используем значение из конфигурации
        if k is None:
            k = config.search_default_k
            
    # Если клиент не передан, создаем новый (асинхронно чтобы не блокировать loop)
        if client is None:
            client = await aget_qdrant_client(config)
            
        # Получаем эмбеддер для поиска
        search_device = get_search_device(device)
        embedder = get_dense_embedder(config, search_device)
        
        # Создаем QdrantVectorStore для поиска
        sparse_emb = None
        # Определяем тип коллекции и выбираем правильный режим поиска
        has_dense = False
        has_sparse = False
        
        # Получаем информацию о коллекции для определения типа векторов
        try:
            coll_info = client.get_collection(collection_name)
            # Получаем параметры коллекции
            collection_config = getattr(coll_info, 'config', None)
            if collection_config:
                params = getattr(collection_config, 'params', None)
                if params:
                    # Проверяем наличие dense векторов
                    vectors = getattr(params, 'vectors', {})
                    has_dense = bool(vectors) if isinstance(vectors, dict) else False
                    
                    # Проверяем наличие sparse векторов
                    sparse_vectors = getattr(params, 'sparse_vectors', {})
                    has_sparse = bool(sparse_vectors) if isinstance(sparse_vectors, dict) else False
                    
                    # Сохраняем имя sparse вектора для использования при поиске
                    if has_sparse and isinstance(sparse_vectors, dict):
                        sparse_vector_names = list(sparse_vectors.keys())
                        if sparse_vector_names:
                            sparse_vector_name = sparse_vector_names[0]  # Берем первое имя
                        else:
                            sparse_vector_name = "sparse_vector"
                    else:
                        sparse_vector_name = "sparse_vector"
                else:
                    # По умолчанию предполагаем, что есть dense векторы
                    has_dense = True
                    sparse_vector_name = "sparse_vector"
            else:
                # По умолчанию предполагаем, что есть dense векторы
                has_dense = True
                sparse_vector_name = "sparse_vector"
        except Exception as e:
            logger.debug(f"Не удалось получить информацию о коллекции '{collection_name}': {e}")
            # По умолчанию предполагаем, что есть dense векторы (старое поведение)
            has_dense = True
            sparse_vector_name = "sparse_vector"
        
        # Если запрошен гибридный поиск, проверяем возможность его выполнения
        if hybrid:
            if not has_sparse:
                logger.warning(f"Коллекция '{collection_name}' не содержит sparse-векторов. Выполняется только dense search.")
                hybrid = False
            elif not has_dense:
                logger.warning(f"Коллекция '{collection_name}' не содержит dense-векторов. Выполняется только sparse search.")
                hybrid = False
        
        # Инициализируем sparse embedding если он нужен
        if has_sparse and (hybrid or (not hybrid and not has_dense)):
            try:
                from core.sparse_embedding_adapter import SparseEmbeddingAdapter
                sparse_emb = SparseEmbeddingAdapter(config.sparse_embedding)  # Используем config
                logger.info(f"Sparse embedding adapter initialized: {config.sparse_embedding}")
            except ImportError:
                logger.warning("fastembed не доступен — отключение sparse search.")
                has_sparse = False
                if not has_dense:
                    return [], "Нет доступных векторов для поиска (dense и sparse недоступны)"
            except Exception as e:
                logger.exception(f"Ошибка при инициализации sparse embedding: {e}")
                has_sparse = False
                if not has_dense:
                    return [], f"Ошибка при инициализации sparse embedding: {str(e)}"
        
        # Выбираем правильный режим поиска в зависимости от типа коллекции и запроса
        logger.info(f"Search mode selection: hybrid={hybrid}, has_dense={has_dense}, has_sparse={has_sparse}")
        if hybrid and has_dense and has_sparse:
            # Гибридный поиск
            logger.info(f"Initializing hybrid QdrantVectorStore for collection '{collection_name}'")
            qdrant = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedder,
                vector_name="dense_vector",
                sparse_embedding=sparse_emb,
                sparse_vector_name=sparse_vector_name,
                retrieval_mode=RetrievalMode.HYBRID
            )
        elif not hybrid and has_sparse and not has_dense:
            # Sparse-only поиск
            logger.info(f"Initializing sparse-only QdrantVectorStore for collection '{collection_name}' with sparse_vector_name='{sparse_vector_name}'")
            qdrant = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=None,
                vector_name=None,
                sparse_embedding=sparse_emb,
                sparse_vector_name=sparse_vector_name,
                retrieval_mode=RetrievalMode.SPARSE
            )
        else:
            # Dense-only поиск (по умолчанию)
            logger.info(f"Initializing dense-only QdrantVectorStore for collection '{collection_name}' (fallback)")
            logger.info(f"  hybrid={hybrid}, has_dense={has_dense}, has_sparse={has_sparse}")
            qdrant = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedder,
                vector_name="dense_vector"
            )
        
        # Выполняем поиск
        results = await qdrant.asimilarity_search_with_score(query, k=k)
            
        return results, None
        
    except Exception as e:
        logger.exception(f"Ошибка при поиске: {e}")
        return [], str(e)


def search_collections(query: str, k: int = None) -> List[Tuple[Any, float]]:
    """
    Выполняет поиск по коллекциям, используя настройки из config.json.
    
    Args:
        query (str): Поисковый запрос.
        k (int): Количество результатов. Если None, используется значение из конфигурации.
        
    Returns:
        List[Tuple[Any, float]]: Результаты поиска.
    """
    import asyncio
    try:
        config: Config = load_config()
        if k is None:
            k = config.search_default_k
            
        # Получаем устройство для поиска
        device = get_search_device(config.device)
        
        # Выполняем асинхронный поиск
        results, error = asyncio.run(search_in_collection(
            query=query,
            collection_name=config.collection_name,
            device=device,
            k=k,
            hybrid=config.use_hybrid
        ))
        
        if error:
            logger.error(f"Ошибка при поиске: {error}")
            return []
            
        return results
    except Exception as e:
        logger.exception(f"Ошибка при поиске: {e}")
        return []