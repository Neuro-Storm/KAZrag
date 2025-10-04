"""Модуль для выполнения поиска."""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import models
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range
from qdrant_client.models import SparseVector, Fusion, FusionQuery, Prefetch

from config.config_manager import ConfigManager
from core.embedding.embeddings import get_dense_embedder, get_search_device

logger = logging.getLogger(__name__)


class SearchExecutor:
    """Класс для выполнения поисковых запросов."""
    
    @staticmethod
    async def execute_search_with_vector(
        client, 
        query: str,
        query_vector: List[float],
        k: int, 
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: str = None
    ) -> Tuple[List[Tuple[Any, float]], Optional[str]]:
        """
        Выполняет поиск с использованием предварительно вычисленного вектора запроса.
        
        Args:
            client: QdrantClient.
            query (str): Оригинальный текстовый запрос (для логирования и reranking).
            query_vector (List[float]): Предварительно вычисленный вектор запроса.
            k (int): Количество результатов.
            metadata_filter (Optional[Dict[str, Any]]): Фильтр по метаданным.
            collection_name (str): Название коллекции для поиска.
            
        Returns:
            Tuple[List[Tuple[Any, float]], Optional[str]]: (результаты поиска, ошибка)
        """
        try:
            if collection_name is None:
                collection_name = ConfigManager.get_instance().get().collection_name  # Получаем из конфига
            # Выполняем поиск с опциональной фильтрацией по метаданным, используя готовый вектор
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
                # Используем нативный поиск Qdrant с именованным вектором
                results = client.search(
                    collection_name=collection_name,
                    query_vector=("dense_vector", query_vector),
                    limit=k,
                    query_filter=search_filter,
                    with_payload=True,
                    with_vectors=False
                )
            else:
                # Используем нативный поиск Qdrant без фильтров с именованным вектором
                results = client.search(
                    collection_name=collection_name,
                    query_vector=("dense_vector", query_vector),
                    limit=k,
                    with_payload=True,
                    with_vectors=False
                )

            # Обрабатываем результаты для извлечения содержимого чанков
            processed_results = []
            for point in results:  # Теперь results - это список PointStruct
                # Извлекаем дополнительную информацию из payload
                payload = point.payload if hasattr(point, 'payload') else {}
                content = payload.get('content', '') or payload.get('page_content', '')
                metadata = payload.get('metadata', {})
                
                # Создаем расширенный объект результата
                extended_result = {
                    'content': content if content is not None else '',
                    'metadata': metadata,
                    'original_score': point.score if hasattr(point, 'score') else 0  # Сохраняем оригинальную оценку
                }
                
                # Если это многоуровневый чанк, добавляем информацию о микро-чанках
                if 'micro_contents' in metadata:
                    extended_result['micro_contents'] = metadata['micro_contents']
                elif 'micro_contents' in payload:
                    extended_result['micro_contents'] = payload['micro_contents']
                    
                # Добавляем source если есть
                if 'source' in metadata:
                    extended_result['source'] = metadata['source']
                elif 'source' in payload:
                    extended_result['source'] = payload['source']
                elif 'source' not in extended_result and 'source' in metadata:
                    extended_result['source'] = metadata.get('source', '')
                    
                processed_results.append((extended_result, point.score if hasattr(point, 'score') else 0))
                
            logger.debug(f"Search with vector returned {len(processed_results)} results")
            
            # Log first result before returning to see what we have
            if processed_results:
                first_result, first_score = processed_results[0]
                logger.debug(f"Before returning - First result score: {first_score}, keys: {list(first_result.keys()) if isinstance(first_result, dict) else 'not dict'}")
                if isinstance(first_result, dict):
                    logger.debug(f"Before returning - original_score: {first_result.get('original_score')}")
            
            return processed_results, None
            
        except Exception as e:
            logger.exception(f"Ошибка при поиске с вектором: {e}")
            return [], str(e)

    @staticmethod
    async def execute_hybrid_search(
        client, query: str, embedder, sparse_params: Dict, k: int, metadata_filter: Optional[Dict[str, Any]] = None, collection_name: str = None
    ) -> Tuple[List[Any], Optional[str]]:  # Changed to return raw ScoredPoint objects
        """
        Выполняет hybrid поиск с dense + sparse через Query API.
        """
        try:
            config = ConfigManager.get_instance().get()
            if collection_name is None:
                collection_name = config.collection_name
            
            # Dense vector
            dense_vector = embedder.embed_query(query)
            
            # Sparse vector (с валидацией)
            sparse_embedding = sparse_params.get("sparse_embedding") if sparse_params else None
            sparse_dict = sparse_embedding.embed_query(query) if sparse_embedding else None
            sparse_name = sparse_params.get("sparse_vector_name", "bm25_text") if sparse_params else "bm25_text"
            sparse_vec = None
            if sparse_dict:
                indices = sorted(set(sparse_dict["indices"]))
                values = [sparse_dict["values"][i] for i in range(len(sparse_dict["indices"])) if sparse_dict["indices"][i] in indices]
                non_zero = sum(1 for v in values if v > 0)
                logger.info(f"Sparse query: {len(indices)} indices, {non_zero} non-zero values")
                if non_zero > 0:
                    sparse_vec = models.SparseVector(indices=indices, values=values)
            
            # Filter на top-level
            search_filter = SearchExecutor._create_filter(metadata_filter) if metadata_filter else None
            
            # Если sparse вектора нет, используем только dense поиск
            if sparse_vec:
                logger.info(f"Sending hybrid query with dense vector (len={len(dense_vector)}) and sparse vector (len={len(sparse_vec.indices)}) to {sparse_name}")
                
                # Prefetch для RRF (только если есть оба вектора)
                prefetch = []
                
                # Dense prefetch
                prefetch.append(
                    models.Prefetch(
                        query=dense_vector,
                        using="dense_vector",
                        limit=k * 2
                    )
                )
                
                # Sparse prefetch (только если есть sparse вектор)
                prefetch.append(
                    models.Prefetch(
                        query=sparse_vec,
                        using=sparse_name,
                        limit=k * 2
                    )
                )
                
                # Fusion RRF без rrf_k (default k=60)
                fusion = models.FusionQuery(fusion=models.Fusion.RRF)
                
                # Query points
                logger.info(f"Executing query_points with prefetch and RRF fusion for collection {collection_name}")
                response = client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch,
                    query=fusion,
                    limit=k,
                    query_filter=search_filter,  # Top-level!
                    with_payload=True,
                    with_vectors=False
                )
                
                results = response.points  # List[ScoredPoint]
                
                logger.info(f"RRF query returned {len(results)} results")
                
                # Log details about the scores to debug the issue
                if results:
                    logger.info(f"RRF scores - first few: {[f'{point.score:.6f}' for point in results[:5]]}")
                    logger.info(f"RRF scores - all: {[f'{point.score:.6f}' for point in results]}")
                
                # Проверим качество результатов RRF - если оценки обнулены, используем dense fallback
                if results:
                    # Проверим, есть ли значимые оценки (не все близки к 0)
                    significant_scores = [point for point in results if point.score > 0.0001]
                    if not significant_scores:
                        logger.warning(f"RRF returned low-quality scores ({len(significant_scores)} significant out of {len(results)}) — fallback to dense-only")
                        results = client.search(
                            collection_name=collection_name,
                            query_vector=("dense_vector", dense_vector),
                            limit=k,
                            query_filter=search_filter,
                            with_payload=True,
                            with_vectors=False
                        )
                    else:
                        logger.info(f"RRF returned {len(significant_scores)} significant scores out of {len(results)} total")
            else:
                # Если sparse вектора нет, используем только dense поиск
                logger.info("No sparse vector available, using dense-only search")
                results = client.search(
                    collection_name=collection_name,
                    query_vector=("dense_vector", dense_vector),
                    limit=k,
                    query_filter=search_filter,
                    with_payload=True,
                    with_vectors=False
                )
            
            # Check if results exist and log scores appropriately
            if not results:
                logger.warning("Hybrid returned empty — fallback to dense-only")
                results = client.search(
                    collection_name=collection_name,
                    query_vector=("dense_vector", dense_vector),
                    limit=k,
                    query_filter=search_filter,
                    with_payload=True,
                    with_vectors=False
                )
            
            logger.info(f"Hybrid results: {len(results)}, first score: {results[0].score if results else 'empty'}")
            
            # CRITICAL FIX: The fallback check was incorrectly triggering even with good RRF scores
            # Only perform dense fallback if RRF results are truly poor (all scores extremely low)
            # This was the source of the issue where good RRF scores were being replaced with dense fallback
            all_scores_are_poor = False
            if results:
                scores_list = [point.score for point in results]
                logger.info(f"RRF scores for fallback check: {scores_list}")
                
                # Check if ALL scores are extremely low
                all_scores_are_poor = all(score < 0.001 for score in scores_list)  # Increased threshold to be more conservative
                logger.info(f"All scores are poor: {all_scores_are_poor}, score details: {scores_list}")
            
            # Only do fallback if ALL scores are genuinely poor (very low)
            if results and all_scores_are_poor:
                logger.warning("All RRF scores are extremely low - fallback to dense-only")
                results = client.search(
                    collection_name=collection_name,
                    query_vector=("dense_vector", dense_vector),
                    limit=k,
                    query_filter=search_filter,
                    with_payload=True,
                    with_vectors=False
                )
            else:
                logger.info("RRF scores are acceptable - no fallback needed")
            
            return results, None  # Return raw ScoredPoint objects for processing in execute_search
            
        except Exception as e:
            logger.exception(f"Ошибка в hybrid search: {e}")
            # Explicit fallback в except
            try:
                logger.info("Fallback to dense in except block")
                config = ConfigManager.get_instance().get()
                from core.embedding.embeddings import get_dense_embedder, get_search_device
                search_device = get_search_device(None)  # используем дефолтное устройство для поиска
                embedder = get_dense_embedder(config, search_device)  # Re-init if needed
                dense_vector = embedder.embed_query(query)
                search_filter = SearchExecutor._create_filter(metadata_filter) if metadata_filter else None
                fallback_results = client.search(
                    collection_name=collection_name,
                    query_vector=("dense_vector", dense_vector),
                    limit=k,
                    query_filter=search_filter,
                    with_payload=True,
                    with_vectors=False
                )
                return fallback_results, None  # Return raw ScoredPoint objects for processing in execute_search
            except Exception as fallback_e:
                logger.exception(f"Fallback failed: {fallback_e}")
                return [], str(e)

    @staticmethod
    async def execute_search(
        client, 
        search_mode: str,
        vector_name: Optional[str],
        sparse_params: Optional[Dict],
        query: str, 
        k: int, 
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: str = None
    ) -> Tuple[List[Tuple[Any, float]], Optional[str]]:
        """
        Выполняет поиск с опциональной фильтрацией по метаданным.
        
        Args:
            client: QdrantClient.
            search_mode (str): Режим поиска ("dense", "sparse", "hybrid").
            vector_name (str): Имя dense vector.
            sparse_params (Dict): Параметры sparse.
            query (str): Поисковый запрос.
            k (int): Количество результатов.
            metadata_filter (Optional[Dict[str, Any]]): Фильтр по метаданным.
            collection_name (str): Название коллекции для поиска.
            
        Returns:
            Tuple[List[Tuple[Any, float]], Optional[str]]: (результаты поиска, ошибка)
        """
        try:
            config = ConfigManager.get_instance().get()
            if collection_name is None:
                collection_name = config.collection_name
            
            # Создаем фильтр, если нужно
            search_filter = SearchExecutor._create_filter(metadata_filter) if metadata_filter else None
            
            if search_mode == "hybrid":
                logger.info(f"Starting hybrid search mode for collection {collection_name}")
                # Для hybrid нужен dense embedder
                search_device = get_search_device(None)  # используем дефолтное устройство для поиска
                embedder = get_dense_embedder(config, search_device)
                query_vector = embedder.embed_query(query)
                
                # Проверяем, есть ли валидный sparse embedding в sparse_params
                sparse_embedding = sparse_params.get("sparse_embedding") if sparse_params else None
                logger.info(f"Sparse embedding in params: {sparse_embedding is not None}, use_bm25 config: {config.use_bm25}")
                
                # Если sparse embedding отсутствует или BM25 отключен в конфиге, используем dense-only
                if not config.use_bm25 or sparse_embedding is None:
                    logger.info("Sparse embedding not available, using dense-only search for hybrid mode")
                    results = client.search(
                        collection_name=collection_name,
                        query_vector=("dense_vector", query_vector),
                        limit=k,
                        query_filter=search_filter,
                        with_payload=True,
                        with_vectors=False
                    )
                    error = None
                else:
                    logger.info(f"Executing hybrid search with sparse vector name: {sparse_params.get('sparse_vector_name', 'NOT SPECIFIED')}")
                    # Обычный гибридный поиск
                    results, error = await SearchExecutor.execute_hybrid_search(client, query, embedder, sparse_params, k, metadata_filter, collection_name)
                    if error:
                        logger.warning(f"Hybrid search failed: {error}, falling back to dense search")
                        # Fallback to dense search
                        results = client.search(
                            collection_name=collection_name,
                            query_vector=("dense_vector", query_vector),
                            limit=k,
                            query_filter=search_filter,
                            with_payload=True,
                            with_vectors=False
                        )
                        error = None
            elif search_mode == "sparse":
                # Sparse поиск — НЕ инициализируем dense!
                sparse_vector_name = sparse_params["sparse_vector_name"] if sparse_params else config.sparse_vector_name
                # Получаем sparse вектор через sparse embedding
                sparse_embedding = sparse_params.get("sparse_embedding")
                sparse_vector = None
                if sparse_embedding:
                    # sparse_result уже в формате {"indices": [...], "values": [...]} от адаптера
                    sparse_result = sparse_embedding.embed_query(query)
                    sparse_vector = sparse_result  # уже в правильном формате
                
                if sparse_vector:
                    # Валидация и нормализация sparse vector
                    indices = sorted(set(sparse_vector["indices"]))  # Unique + sorted (Qdrant requires)
                    values = [sparse_vector["values"][i] for i in range(len(sparse_vector["indices"])) if sparse_vector["indices"][i] in indices]  # Match to unique indices
                    
                    # Log для debug
                    non_zero = sum(1 for v in values if abs(v) > 1e-10)  # Check for non-zero values (with tolerance)
                    logger.info(f"Sparse query: {len(indices)} indices, {non_zero} non-zero values")
                    
                    if non_zero == 0:
                        logger.warning("Sparse query has 0 non-zero values — skipping sparse search")
                        results = []
                    else:
                        sparse_vec = models.SparseVector(indices=indices, values=values)
                        named_sparse = models.NamedSparseVector(name=sparse_vector_name, vector=sparse_vec)
                        
                        results = client.search(
                            collection_name=collection_name,
                            query_vector=named_sparse,  # Передаем sparse как query_vector
                            limit=k,
                            query_filter=search_filter,
                            with_payload=True,
                            with_vectors=False
                        )
                else:
                    # Если sparse вектор не создан, возвращаем пустой результат
                    results = []
            else:
                # Dense — инициализируем dense embedder
                search_device = get_search_device(None)  # используем дефолтное устройство для поиска
                embedder = get_dense_embedder(config, search_device)
                query_vector = embedder.embed_query(query)
                # Для dense векторов используем именованный вектор
                if vector_name:
                    results = client.search(
                        collection_name=collection_name,
                        query_vector=(vector_name, query_vector),
                        limit=k,
                        query_filter=search_filter,
                        with_payload=True,
                        with_vectors=False
                    )
                else:
                    # Если имя вектора не указано, используем имя по умолчанию "dense_vector"
                    results = client.search(
                        collection_name=collection_name,
                        query_vector=("dense_vector", query_vector),
                        limit=k,
                        query_filter=search_filter,
                        with_payload=True,
                        with_vectors=False
                    )
        
            # Обрабатываем результаты для извлечения содержимого чанков
            processed_results = []
            for point in results:  # Теперь results - это список PointStruct
                # Извлекаем дополнительную информацию из payload
                payload = point.payload if hasattr(point, 'payload') else {}
                content = payload.get('content', '') or payload.get('page_content', '')
                metadata = payload.get('metadata', {})
                
                # Создаем расширенный объект результата
                extended_result = {
                    'content': content if content is not None else '',
                    'metadata': metadata,
                    'original_score': point.score if hasattr(point, 'score') else 0  # Сохраняем оригинальную оценку
                }
                
                # Если это многоуровневый чанк, добавляем информацию о микро-чанках
                if 'micro_contents' in metadata:
                    extended_result['micro_contents'] = metadata['micro_contents']
                elif 'micro_contents' in payload:
                    extended_result['micro_contents'] = payload['micro_contents']
                    
                # Добавляем source если есть
                if 'source' in metadata:
                    extended_result['source'] = metadata['source']
                elif 'source' in payload:
                    extended_result['source'] = payload['source']
                elif 'source' not in extended_result and 'source' in metadata:
                    extended_result['source'] = metadata.get('source', '')
                    
                processed_results.append((extended_result, point.score if hasattr(point, 'score') else 0))
                
            logger.debug(f"Search returned {len(processed_results)} results")
            
            # Log first result before returning to see what we have
            if processed_results:
                first_result, first_score = processed_results[0]
                logger.debug(f"Before returning - First result score: {first_score}, keys: {list(first_result.keys()) if isinstance(first_result, dict) else 'not dict'}")
                if isinstance(first_result, dict):
                    logger.debug(f"Before returning - original_score: {first_result.get('original_score')}")
                    
            # Explicit logging to show which vector names are used in the search
            logger.info(f"Search execution completed with mode '{search_mode}', dense vector: 'dense_vector', sparse vector: {sparse_params.get('sparse_vector_name', 'None') if sparse_params else 'None'}")
        
            return processed_results, None
            
        except Exception as e:
            logger.exception(f"Ошибка при поиске: {e}")
            return [], str(e)

    @staticmethod
    def _create_filter(metadata_filter: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Вспомогательный метод для создания фильтра."""
        if not metadata_filter:
            return None
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
        return Filter(must=must_conditions)