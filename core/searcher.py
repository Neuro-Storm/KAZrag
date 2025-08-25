"""Модуль для выполнения поиска в Qdrant."""

import logging
from typing import List, Tuple, Any, Optional, Dict
from core.qdrant_client import aget_qdrant_client
from config.settings import load_config, Config
from core.embeddings import get_dense_embedder, get_search_device
from core.collection_analyzer import CollectionAnalyzer, SearchError
from core.search_strategy import SearchStrategy
from core.search_executor import SearchExecutor

logger = logging.getLogger(__name__)


async def search_in_collection(query: str, collection_name: str, device: str, k: int = None, hybrid: bool = False, 
                              metadata_filter: Optional[Dict[str, Any]] = None, client = None) -> Tuple[List[Tuple[Any, float]], Optional[str]]:
    """
    Выполняет поиск в указанной коллекции Qdrant.
    
    Args:
        query (str): Поисковый запрос.
        collection_name (str): Название коллекции.
        device (str): Устройство для поиска ("cpu" или "cuda").
        k (int): Количество результатов. Если None, используется значение из конфигурации.
        hybrid (bool): Использовать hybrid search (dense + sparse).
        metadata_filter (Optional[Dict[str, Any]]): Фильтр по метаданным.
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
        
        # Инициализируем sparse embedding если он нужен
        sparse_emb = None
        analyzer = CollectionAnalyzer()
        has_dense, has_sparse, sparse_vector_name = analyzer.analyze_collection(client, collection_name)
        
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
        
        # Определяем стратегию поиска
        strategy = SearchStrategy(client, collection_name, embedder, sparse_emb)
        search_mode = strategy.determine_search_mode(hybrid)
        
        # Создаем QdrantVectorStore с соответствующими параметрами
        qdrant = strategy.create_qdrant_store(search_mode)
        
        # Выполняем поиск
        executor = SearchExecutor()
        results, error = await executor.execute_search(qdrant, query, k, metadata_filter)
        
        return results, error
        
    except Exception as e:
        logger.exception(f"Ошибка при поиске: {e}")
        return [], str(e)


def search_collections(query: str, k: int = None, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Any, float]]:
    """
    Выполняет поиск по коллекциям, используя настройки из config.json.
    
    Args:
        query (str): Поисковый запрос.
        k (int): Количество результатов. Если None, используется значение из конфигурации.
        metadata_filter (Optional[Dict[str, Any]]): Фильтр по метаданным.
        
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
            hybrid=config.use_hybrid,
            metadata_filter=metadata_filter
        ))
        
        if error:
            logger.error(f"Ошибка при поиске: {error}")
            return []
            
        return results
    except Exception as e:
        logger.exception(f"Ошибка при поиске: {e}")
        return []