"""Модуль для выполнения поиска в Qdrant."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from config.config_manager import ConfigManager
from config.settings import Config
from core.embedding.embeddings import get_dense_embedder, get_search_device
from core.qdrant.qdrant_client import aget_qdrant_client
from core.search.collection_analyzer import CollectionAnalyzer
from core.search.search_executor import SearchExecutor
from core.search.search_strategy import SearchStrategy

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
        
        # Инициализируем sparse embedding для всех режимов поиска, где он может быть нужен
        if has_sparse:
            try:
                from core.embedding.sparse_embedding_adapter import (
                    SparseEmbeddingAdapter,
                )
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
        
        logger.info(f"Search mode determined: {search_mode}")
        
        # Создаем QdrantVectorStore с соответствующими параметрами
        qdrant = strategy.create_qdrant_store(search_mode)
        
        # Выполняем поиск
        executor = SearchExecutor()
        results, error = await executor.execute_search(qdrant, query, k, metadata_filter)
        
        # Дополнительно обрабатываем результаты для веб-интерфейса
        processed_results = []
        for result, score in results:
            if isinstance(result, dict):
                # Это уже расширенный результат из SearchExecutor
                processed_results.append((result, score))
            else:
                # Стандартный формат - пытаемся извлечь содержимое
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
                
                # Если content все еще пустой, проверяем __dict__
                if not content and hasattr(result, '__dict__'):
                    result_dict = result.__dict__
                    content = result_dict.get('content', '')
                    if not content:
                        content = result_dict.get('page_content', '')
                    if not metadata and 'metadata' in result_dict:
                        metadata = result_dict['metadata']
                
                processed_results.append(({
                    'content': content if content is not None else '',
                    'metadata': metadata,
                    'original_score': score,  # Сохраняем оригинальную оценку
                    'score': score
                }, score))
        
        # Если контент пустой, пытаемся получить его напрямую из Qdrant
        if all(not result.get('content', '') for result, _ in processed_results) and client:
            logger.debug("Content is empty in all results, attempting to fetch directly from Qdrant")
            try:
                # Получаем точки напрямую из Qdrant по ID результатов
                point_ids = []
                for result, _ in processed_results:
                    if 'metadata' in result and '_id' in result['metadata']:
                        point_ids.append(result['metadata']['_id'])
                
                if point_ids:
                    # Получаем точки напрямую
                    points_response = client.retrieve(
                        collection_name=collection_name,
                        ids=point_ids,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    # Создаем словарь ID -> payload для быстрого поиска
                    payload_dict = {}
                    for point in points_response:
                        if hasattr(point, 'id') and hasattr(point, 'payload'):
                            payload_dict[str(point.id)] = point.payload
                    
                    # Обновляем содержимое в результатах
                    for i, (result, score) in enumerate(processed_results):
                        point_id = result['metadata'].get('_id')
                        if point_id and str(point_id) in payload_dict:
                            payload = payload_dict[str(point_id)]
                            content = payload.get('content', '')
                            if not content:
                                content = payload.get('page_content', '')
                            
                            # Обновляем содержимое в результате, сохраняя все существующие поля
                            updated_result = result.copy()  # Копируем все существующие поля
                            
                            # Обновляем контент только если он не пустой
                            if content and content.strip():
                                updated_result['content'] = content
                            # Если контент пустой, оставляем существующий контент
                            
                            processed_results[i] = (updated_result, score)
                    
                    logger.debug(f"Updated {len(processed_results)} results with content from direct Qdrant query")
            except Exception as retrieve_error:
                logger.warning(f"Failed to retrieve content directly from Qdrant: {retrieve_error}")
        
        # Применяем reranker если включён (после обновления контента)
        config_manager = ConfigManager.get_instance()
        config = config_manager.get()
        if config.reranker_enabled and processed_results:
            from core.search.reranker_manager import RerankerManager
            reranker_manager = RerankerManager.get_instance()
            processed_results = reranker_manager.rerank_documents(query, processed_results, config)
            logger.debug(f"Reranked to {len(processed_results)} results")
        
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