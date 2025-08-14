"""Модуль для управления кэшем коллекций Qdrant."""

import time
import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import CollectionDescription
from config.settings import load_config, Config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Кэш для списка коллекций ---
_cached_collections: Optional[List[str]] = None
_cached_collections_time = 0
COLLECTIONS_CACHE_TTL = 300  # 5 минут


def get_cached_collections(client: QdrantClient = None) -> List[str]:
    """
    Получает список коллекций с кэшированием.
    
    Args:
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, создается новый.
        
    Returns:
        List[str]: Список названий коллекций.
    """
    global _cached_collections, _cached_collections_time
    
    current_time = time.time()
    
    # Проверяем, нужно ли обновить кэш
    if (_cached_collections is None or 
        current_time - _cached_collections_time > COLLECTIONS_CACHE_TTL):
        
        try:
            # Если клиент не передан, создаем новый
            if client is None:
                config: Config = load_config()
                client = QdrantClient(url=config.qdrant_url)
                
            collections_response = client.get_collections()
            _cached_collections = [c.name for c in collections_response.collections]
            _cached_collections_time = current_time
            logger.debug(f"Обновлен кэш коллекций: {_cached_collections}")
        except Exception as e:
            logger.error(f"Ошибка при получении списка коллекций: {e}")
            # В случае ошибки возвращаем пустой список или старый кэш, если он есть
            if _cached_collections is None:
                _cached_collections = []
    
    return _cached_collections


def refresh_collections_cache(client: QdrantClient = None):
    """
    Принудительно обновляет кэш коллекций.
    
    Args:
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, создается новый.
    """
    global _cached_collections, _cached_collections_time
    
    try:
        # Если клиент не передан, создаем новый
        if client is None:
            config: Config = load_config()
            client = QdrantClient(url=config.qdrant_url)
            
        collections_response = client.get_collections()
        _cached_collections = [c.name for c in collections_response.collections]
        _cached_collections_time = time.time()
        logger.debug(f"Принудительно обновлен кэш коллекций: {_cached_collections}")
    except Exception as e:
        logger.error(f"Ошибка при принудительном обновлении кэша коллекций: {e}")