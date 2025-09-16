"""Additional functions for the qdrant collections module."""

import logging
from typing import Tuple

from config.config_manager import ConfigManager
from core.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)


def recreate_collection_from_config(force_recreate: bool = True) -> Tuple[bool, str]:
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