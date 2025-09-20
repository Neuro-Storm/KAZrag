"""Модуль для кэширования поисковых запросов и их эмбеддингов в Qdrant."""

import hashlib
import logging
from typing import List, Optional, Tuple

from config.config_manager import ConfigManager
from config.settings import Config
from core.embedding.embedding_manager import EmbeddingManager
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

# Префикс для коллекций кэша
CACHE_COLLECTION_PREFIX = "query_cache_"


class QueryCache:
    """Управление кэшированием поисковых запросов и их эмбеддингов в Qdrant."""
    
    def __init__(self, client: QdrantClient, config: Config):
        """Инициализация QueryCache с клиентом Qdrant и конфигурацией.
        
        Args:
            client: Экземпляр клиента Qdrant
            config: Конфигурация приложения
        """
        self.client = client
        self.config = config
        self.embedding_manager = EmbeddingManager.get_instance()
    
    def _get_cache_collection_name(self, vector_size: int) -> str:
        """Получить имя коллекции кэша для заданной размерности вектора.
        
        Args:
            vector_size: Размерность вектора
            
        Returns:
            str: Имя коллекции кэша
        """
        return f"{CACHE_COLLECTION_PREFIX}{vector_size}"
    
    def _ensure_cache_collection(self, vector_size: int) -> None:
        """Убедиться, что коллекция кэша существует в Qdrant для заданной размерности.
        
        Args:
            vector_size: Размерность вектора
        """
        try:
            collection_name = self._get_cache_collection_name(vector_size)
            
            # Проверить, существует ли коллекция
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if collection_name not in collection_names:
                # Создать коллекцию кэша
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Создана коллекция кэша '{collection_name}' с размерностью {vector_size}")
            else:
                logger.debug(f"Коллекция кэша '{collection_name}' уже существует")
        except Exception as e:
            logger.exception(f"Ошибка при проверке существования коллекции кэша: {e}")
            raise
    
    def _generate_query_id(self, query_text: str, collection_name: str) -> str:
        """Сгенерировать уникальный ID для запроса на основе его текста и целевой коллекции.
        
        Args:
            query_text: Текст поискового запроса
            collection_name: Имя коллекции, в которой выполняется поиск
            
        Returns:
            str: Уникальный хэш-идентификатор для запроса
        """
        # Создать уникальный идентификатор на основе текста запроса и имени коллекции
        unique_string = f"{query_text}:{collection_name}"
        # Используем только первые 32 символа хэша для совместимости с UUID
        return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()[:32]
    
    def get_cached_query(self, query_text: str, collection_name: str, vector_size: int) -> Optional[Tuple[List[float], str]]:
        """Получить кэшированный эмбеддинг запроса, если он существует.
        
        Args:
            query_text: Текст поискового запроса
            collection_name: Имя коллекции, в которой выполняется поиск
            vector_size: Размерность вектора
            
        Returns:
            Optional[Tuple[List[float], str]]: (вектор эмбеддинга, текст запроса) если найден, иначе None
        """
        try:
            cache_collection_name = self._get_cache_collection_name(vector_size)
            query_id = self._generate_query_id(query_text, collection_name)
            
            # Проверить, существует ли коллекция кэша
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if cache_collection_name not in collection_names:
                # Коллекция не существует, возвращаем None
                logger.debug(f"Коллекция кэша '{cache_collection_name}' не существует")
                return None
            
            # Поиск кэшированного запроса по ID
            response = self.client.retrieve(
                collection_name=cache_collection_name,
                ids=[query_id],
                with_vectors=True,
                with_payload=True
            )
            
            if response:
                point = response[0]
                vector = point.vector if hasattr(point, 'vector') else point.vectors
                payload = point.payload if hasattr(point, 'payload') else {}
                
                # Проверить, что текст запроса совпадает (двойная проверка)
                if payload.get('query_text') == query_text:
                    logger.debug(f"Найден кэшированный запрос для '{query_text}' в коллекции '{collection_name}'")
                    return (vector, query_text)
            
            logger.debug(f"Кэшированный запрос не найден для '{query_text}' в коллекции '{collection_name}'")
            return None
            
        except Exception as e:
            logger.exception(f"Ошибка при получении кэшированного запроса: {e}")
            return None
    
    def cache_query(self, query_text: str, collection_name: str, vector: List[float]) -> bool:
        """Кэшировать эмбеддинг запроса в Qdrant.
        
        Args:
            query_text: Текст поискового запроса
            collection_name: Имя коллекции, в которой выполняется поиск
            vector: Вектор эмбеддинга для запроса
            
        Returns:
            bool: True если кэширование прошло успешно, False в противном случае
        """
        try:
            vector_size = len(vector)
            cache_collection_name = self._get_cache_collection_name(vector_size)
            
            # Убедиться, что коллекция кэша существует
            self._ensure_cache_collection(vector_size)
            
            query_id = self._generate_query_id(query_text, collection_name)
            
            # Подготовить payload с метаданными
            payload = {
                "query_text": query_text,
                "collection_name": collection_name,
                "query_hash": query_id
            }
            
            # Вставить или обновить запрос в коллекции кэша
            self.client.upsert(
                collection_name=cache_collection_name,
                points=[
                    models.PointStruct(
                        id=query_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            
            logger.debug(f"Закэширован запрос '{query_text}' для коллекции '{collection_name}' в коллекции '{cache_collection_name}'")
            return True
            
        except Exception as e:
            logger.exception(f"Ошибка при кэшировании запроса: {e}")
            return False
    
    def get_or_create_query_embedding(self, query_text: str, collection_name: str, vector_size: int) -> List[float]:
        """Получить кэшированный эмбеддинг запроса или создать новый.
        
        Args:
            query_text: Текст поискового запроса
            collection_name: Имя коллекции, в которой выполняется поиск
            vector_size: Размерность вектора
            
        Returns:
            List[float]: Вектор эмбеддинга для запроса
        """
        # Сначала попробовать получить кэшированный эмбеддинг
        cached_result = self.get_cached_query(query_text, collection_name, vector_size)
        if cached_result:
            return cached_result[0]  # Вернуть вектор
        
        # Если не закэширован, создать новый эмбеддинг
        logger.debug(f"Запрос не найден в кэше, создание нового эмбеддинга для '{query_text}'")
        embedding = self.embedding_manager.embed_query(query_text, self.config)
        
        # Кэшировать новый эмбеддинг
        self.cache_query(query_text, collection_name, embedding)
        
        return embedding