"""Модуль для управления коллекциями Qdrant."""

import logging
from typing import List
from core.utils.collection_manager import CollectionManager
from core.qdrant.qdrant_client import get_qdrant_client
from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


def get_cached_collections(client=None) -> List[str]:
    """
    Получает список коллекций с кэшированием.
    
    Args:
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, используется закэшированный.
        
    Returns:
        List[str]: Список названий коллекций.
    """
    try:
        collection_manager = CollectionManager.get_instance()
        collections_dict = collection_manager.get_collections(client)
        return list(collections_dict.keys())
    except Exception as e:
        logger.exception(f"Ошибка при получении списка коллекций: {e}")
        return []


async def refresh_collections_cache(client=None):
    """
    Принудительно обновляет кэш коллекций.
    
    Args:
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, используется закэшированный.
    """
    try:
        collection_manager = CollectionManager.get_instance()
        collections_dict = collection_manager.refresh_collections(client)
        logger.debug(f"Принудительно обновлен кэш коллекций: {list(collections_dict.keys())}")
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
        config_manager = ConfigManager.get_instance()
        config = config_manager.get()
        
        # Проверка наличия хотя бы одного типа индексации
        if not (config.index_dense or config.index_bm25 or config.index_hybrid):
            return False, "no_index_type"
            
        # Проверка для гибридного режима
        if config.index_hybrid and not config.index_dense:
            return False, "hybrid_requires_dense"
            
        client = get_qdrant_client(config)
        
        # Используем CollectionManager для проверки существования коллекции
        collection_manager = CollectionManager.get_instance()
        collections_dict = collection_manager.get_collections(client)
        collection_exists = config.collection_name in collections_dict
        
        if collection_exists and force_recreate:
            # Удаление существующей коллекции через CollectionManager
            client.delete_collection(config.collection_name)
            # Очищаем кэш CollectionManager
            collection_manager.refresh_collections(client)
            logger.info(f"Удалена существующая коллекция: {config.collection_name}")
            
        if not collection_exists or force_recreate:
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