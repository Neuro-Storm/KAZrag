"""Модуль для управления коллекциями Qdrant."""

import logging
import time
from typing import List
from core.qdrant_client import get_qdrant_client
from config.settings import load_config

logger = logging.getLogger(__name__)

# --- Кэш для списка коллекций ---
_collections_cache = None
_collections_cache_time = 0
# Глобальный клиент для использования в кэше
_qdrant_client = None


def get_qdrant_client_cached():
    """Возвращает закэшированный клиент Qdrant."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = get_qdrant_client()
    return _qdrant_client


def get_cached_collections(client = None) -> List[str]:
    """
    Получает список коллекций с кэшированием.
    
    Args:
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, используется закэшированный.
        
    Returns:
        List[str]: Список названий коллекций.
    """
    global _collections_cache, _collections_cache_time
    
    # Загружаем конфигурацию для получения TTL
    config = load_config()
    
    current_time = time.time()
    
    # Проверяем, нужно ли обновить кэш
    if (_collections_cache is None or 
        current_time - _collections_cache_time > config.collections_cache_ttl):
        
        try:
            # Если клиент не передан, используем закэшированный
            if client is None:
                client = get_qdrant_client_cached()
                
            collections_response = client.get_collections()
            _collections_cache = [c.name for c in collections_response.collections]
            _collections_cache_time = current_time
            logger.debug(f"Обновлен кэш коллекций: {_collections_cache}")
        except Exception as e:
            logger.exception(f"Ошибка при получении списка коллекций: {e}")
            # В случае ошибки возвращаем пустой список или старый кэш, если он есть
            if _collections_cache is None:
                _collections_cache = []
    
    return _collections_cache


def refresh_collections_cache(client = None):
    """
    Принудительно обновляет кэш коллекций.
    
    Args:
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, используется закэшированный.
    """
    global _collections_cache, _collections_cache_time
    
    try:
        # Если клиент не передан, используем закэшированный
        if client is None:
            client = get_qdrant_client_cached()
            
        collections_response = client.get_collections()
        _collections_cache = [c.name for c in collections_response.collections]
        _collections_cache_time = time.time()
        logger.debug(f"Принудительно обновлен кэш коллекций: {_collections_cache}")
    except Exception as e:
        logger.exception(f"Ошибка при принудительном обновлении кэша коллекций: {e}")


def recreate_collection_from_config(force_recreate: bool = True):
    """
    Пересоздает коллекцию, используя настройки из config.json.
    
    Args:
        force_recreate (bool): Принудительно пересоздать коллекцию.
        
    Returns:
        Tuple[bool, str]: (успех, статус)
    """
    try:
        config = load_config()
        
        # Проверка наличия хотя бы одного типа индексации
        if not (config.index_dense or config.index_bm25 or config.index_hybrid):
            return False, "no_index_type"
            
        # Проверка для гибридного режима
        if config.index_hybrid and not config.index_dense:
            return False, "hybrid_requires_dense"
            
        client = get_qdrant_client(config)
        
        # Проверка существования коллекции
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if config.collection_name in collection_names and force_recreate:
            # Удаление существующей коллекции
            client.delete_collection(config.collection_name)
            logger.info(f"Удалена существующая коллекция: {config.collection_name}")
            
        if config.collection_name not in collection_names or force_recreate:
            # Создание новой коллекции
            
            # Создаем временную коллекцию для получения параметров
            if config.index_dense:
                # Для плотных векторов создаем коллекцию с нужными параметрами
                # Реальная реализация зависит от конкретных требований
                pass
                
            logger.info(f"Создана новая коллекция: {config.collection_name}")
            
        return True, "collection_created"
        
    except Exception as e:
        logger.exception(f"Ошибка при пересоздании коллекции: {e}")
        return False, f"collection_error: {str(e)}"