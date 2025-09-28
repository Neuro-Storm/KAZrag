"""Модуль для выполнения поиска в Qdrant."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client.http.models import CountResult

from config.config_manager import ConfigManager
from config.settings import Config
from core.embedding.embeddings import get_dense_embedder, get_search_device
from core.qdrant.qdrant_client import aget_qdrant_client
from core.search.collection_analyzer import CollectionAnalyzer
from core.search.search_executor import SearchExecutor
from core.search.search_strategy import SearchStrategy
from core.search.query_cache import QueryCache

logger = logging.getLogger(__name__)


async def search_in_collection(query: str, collection_name: str, device: str, k: int = None, hybrid: bool = False, 
                              search_mode_override: str = None, metadata_filter: Optional[Dict[str, Any]] = None, client = None) -> Tuple[List[Tuple[Any, float]], Optional[str]]:
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
        config_manager = ConfigManager.get_instance()
        config: Config = config_manager.get()
        
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
        
        logger.info(f"Collection analysis: has_dense={has_dense}, has_sparse={has_sparse}, sparse_vector_name={sparse_vector_name}")
        logger.info(f"Search parameters: hybrid={hybrid}, device={device}")
        
        # Инициализируем sparse embedding если он нужен и включен в конфигурации
        sparse_emb = None
        if has_sparse and config.use_bm25:
            try:
                from core.embedding.sparse_embedding_adapter import (
                    SparseEmbeddingAdapter,
                )
                sparse_emb = SparseEmbeddingAdapter(config)  # Используем полный config объект
                logger.info(f"Sparse embedding adapter initialized: {config.sparse_embedding}")
                
                # Debug sample doc sparse
                try:
                    sample_points = client.scroll(
                        collection_name=collection_name,
                        limit=1,
                        with_vectors=True
                    )
                    if sample_points and len(sample_points[0]) > 0:
                        sample_point = sample_points[0][0]  # First point
                        if hasattr(sample_point, 'vectors') and sample_point.vectors:
                            sparse_vec_sample = sample_point.vectors.get(config.sparse_vector_name)
                            if sparse_vec_sample:
                                logger.info(f"Sample doc sparse: indices={sparse_vec_sample.indices[:5]}..., non-zero={sum(1 for v in sparse_vec_sample.values if v > 0)}")
                            else:
                                logger.warning("No sparse vector in sample point")
                        else:
                            logger.warning("No vectors in sample point")
                except Exception as scroll_error:
                    logger.error(f"Debug scroll error: {scroll_error}")
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
        elif has_sparse and not config.use_bm25:
            # Если коллекция поддерживает sparse, но BM25 отключен в конфиге, 
            # не инициализируем sparse embedding, но помечаем, что sparse векторы есть
            logger.info("Sparse vectors exist in collection but BM25 is disabled in config - sparse search will be skipped")
            has_sparse = False  # Принудительно отключаем sparse если он отключен в конфиге
        
        # Определяем стратегию поиска
        strategy = SearchStrategy(client, collection_name, embedder, sparse_emb)
        # Если задан режим поиска явно, используем его, иначе определяем автоматически
        if search_mode_override and search_mode_override in ["dense", "sparse", "hybrid"]:
            search_mode = search_mode_override
            logger.info(f"Using search mode override: {search_mode_override}")
        else:
            # Определяем тип поиска на основе гибридного параметра и доступных векторов
            search_type = "hybrid" if hybrid else "auto"  # "auto" для старого поведения, если не hybrid
            search_mode = strategy.determine_search_mode(hybrid, search_type)
        
        logger.info(f"Search mode determined: {search_mode}")
        
        # Теперь возвращаем client, search_mode, etc.
        client, search_mode, vector_name, sparse_params = strategy.create_qdrant_searcher(search_mode)
        
        # Инициализируем кэш запросов
        query_cache = QueryCache(client, config)
        
        # Получаем размерность вектора для текущей модели
        # Для получения размерности создаем тестовый эмбеддинг
        test_embedding = embedder.embed_query("test")
        vector_size = len(test_embedding)
        
        # Получаем или создаем эмбеддинг запроса с использованием кэша
        query_vector = query_cache.get_or_create_query_embedding(query, collection_name, vector_size)
        
        # Проверяем количество точек в коллекции
        count = client.count(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' has {count.count} points")
        
        # Для других режимов: использовать SearchExecutor.execute_search(client, search_mode, vector_name, sparse_params, query, k, metadata_filter)
        logger.info(f"Calling SearchExecutor.execute_search with mode '{search_mode}', vector_name '{vector_name}', sparse_params keys: {list(sparse_params.keys()) if sparse_params else 'None'}")
        if search_mode != "dense":
            results, error = await SearchExecutor.execute_search(client, search_mode, vector_name, sparse_params, query, k, metadata_filter)
        else:
            # Выполняем поиск с использованием кэшированного вектора
            # Используем SearchExecutor для плотного поиска тоже
            results, error = await SearchExecutor.execute_search(client, search_mode, vector_name, sparse_params, query, k, metadata_filter)
        
        logger.info(f"SearchExecutor returned: {len(results) if results else 0} results, error: {error}")
        
        # Debug logging to check results from SearchExecutor
        if results:
            logger.info(f"Results from SearchExecutor: {len(results)} items")
            for i, (result, score) in enumerate(results[:3]):  # Log first 3 results
                logger.info(f"  Result {i}: score={score}, type={type(result)}, content_len={len(result.get('content', '')) if isinstance(result, dict) else 'N/A'}, source={result.get('source', 'N/A') if isinstance(result, dict) else 'N/A'}")
                if isinstance(result, dict):
                    logger.info(f"    Keys: {list(result.keys())}")
                    logger.info(f"    Content preview: {str(result.get('content', ''))[:100] if result.get('content') else 'NO CONTENT'}")
                    logger.info(f"    Original score: {result.get('original_score', 'NO ORIGINAL SCORE')}")
        else:
            logger.debug("No results returned from SearchExecutor")
        
        # Log the error returned from SearchExecutor
        if error:
            logger.error(f"Error from SearchExecutor: {error}")
        
        # Store original results count for comparison after further processing
        original_results_count = len(results) if results else 0
        logger.debug(f"Before additional processing: {original_results_count} results")
        
        logger.debug(f"Starting additional processing for {len(results)} results")
        # Дополнительно обрабатываем результаты для веб-интерфейса
        logger.debug(f"Starting additional processing for {len(results)} results")
        processed_results = []
        for i, (result, score) in enumerate(results):
            logger.debug(f"Processing result {i}: type={type(result)}, score={score}")
            if isinstance(result, dict):
                # Это уже расширенный результат из SearchExecutor
                logger.debug(f"  Dict result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                logger.debug(f"  Dict content: '{result.get('content', 'NO CONTENT')[:50]}...' if result.get('content') else 'NO CONTENT'")
                logger.debug(f"  Dict source: {result.get('source', 'NO SOURCE')}")
                logger.debug(f"  Dict original_score: {result.get('original_score', 'NO ORIGINAL SCORE')}")
                processed_results.append((result, score))
            else:
                # Стандартный формат - пытаемся извлечь содержимое
                logger.debug(f"  Processing non-dict result: {type(result)}")
                content = getattr(result, 'page_content', '')
                metadata = getattr(result, 'metadata', {})
                
                # Проверяем, есть ли payload с содержимым
                if not content and hasattr(result, 'payload'):
                    payload = result.payload
                    content = payload.get('content', '')
                    if not content:
                        content = payload.get('page_content', '')
                    # Копируем метаданные из payload если они есть
                    if 'metadata' in payload:
                        metadata = payload['metadata']
                
                # Если content все еще пустый, проверяем __dict__
                if not content and hasattr(result, '__dict__'):
                    result_dict = result.__dict__
                    content = result_dict.get('content', '')
                    if not content:
                        content = result_dict.get('page_content', '')
                    if not metadata and 'metadata' in result_dict:
                        metadata = result_dict['metadata']
                
                processed_results.append(
                    (
                        {
                            'content': content if content is not None else '',
                            'metadata': metadata,
                            'original_score': score,  # Сохраняем оригинальную оценку
                            'score': score
                        }, 
                        score
                    )
                )
        
        logger.debug(f"Completed additional processing: {len(processed_results)} results")
        for i, (result, score) in enumerate(processed_results[:3]):
            logger.debug(f"  Processed result {i}: score={score}, content_len={len(result.get('content', ''))}, source={result.get('source', 'N/A')}, original_score={result.get('original_score', 'N/A')}")
            logger.debug(f"    Full result keys: {list(result.keys())}")
            logger.debug(f"    Content preview: {str(result.get('content', ''))[:100] if result.get('content') else 'NO CONTENT'}")
        
        # Log results before potential content fetching from Qdrant
        logger.info(f"Before Qdrant content fetch: {len(processed_results)} results")
        for i, (result, score) in enumerate(processed_results[:2]):
            logger.info(f"  Before Qdrant fetch {i}: score={score}, orig_score={result.get('original_score', 'NO ORIG')}, content_len={len(result.get('content', ''))}")
        
        # Если контент пустой, пытаемся получить его напрямую из Qdrant
        if all(not result.get('content', '') for result, _ in processed_results) and client:
            logger.debug("Content is empty in all results, attempting to fetch directly from Qdrant")
            try:
                # Получаем точки напрямую из Qdrant по ID результатов
                # Note: This only works if we have the original point IDs available
                # Since RRF query processes the points, we need to make sure they were stored properly
                # This is a fallback mechanism for when content extraction from RRF payload fails
                point_ids = []
                # We need to extract IDs differently - since we're using RRF, the original point IDs might not be available
                # The RRF query_points returns ScoredPoint objects with IDs, but we need to maintain that info
                
                # This is tricky because after RRF processing, we lose the original point ID references
                # That's why it's important to have proper content extraction in SearchExecutor
                logger.debug("Skipping direct Qdrant retrieval as it requires original point IDs")
            except Exception as retrieve_error:
                logger.warning(f"Failed to retrieve content directly from Qdrant: {retrieve_error}")
        
        # Применяем reranker если включён (после обновления контента)
        config_manager = ConfigManager.get_instance()
        config = config_manager.get()
        if config.reranker_enabled and processed_results:
            from core.search.reranker_manager import RerankerManager
            reranker_manager = RerankerManager.get_instance()
            # Логируем информацию перед reranking для отладки
            logger.debug(f"Before reranking: {len(processed_results)} results")
            for i, (result, score) in enumerate(processed_results[:3]):  # Логируем первые 3
                logger.debug(f"  Result {i}: score={score}, content_len={len(result.get('content', ''))}, source={result.get('source', 'N/A')}")
            
            processed_results = reranker_manager.rerank_documents(query, processed_results, config)
            
            logger.debug(f"After reranking: {len(processed_results)} results")
            for i, (result, score) in enumerate(processed_results[:3]):  # Логируем первые 3
                logger.debug(f"  Result {i}: score={score}, content_len={len(result.get('content', ''))}, source={result.get('source', 'N/A')}")
            logger.debug(f"Reranked to {len(processed_results)} results")
        else:
            logger.debug(f"Reranker not enabled or no results: reranker_enabled={config.reranker_enabled}, results_count={len(processed_results)}")
        
        logger.info(f"Search completed successfully with {len(processed_results)} results")
        return processed_results, error
        
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
        config_manager = ConfigManager.get_instance()
        config: Config = config_manager.get()
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