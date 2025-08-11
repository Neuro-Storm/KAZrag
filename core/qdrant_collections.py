"""Модуль для управления кэшем коллекций Qdrant."""

import time
import logging
from qdrant_client import QdrantClient
from config.settings import load_config, Config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Кэш для списка коллекций ---
_cached_collections = None
_cached_collections_time = 0
COLLECTIONS_CACHE_TTL = 300  # 5 минут


def refresh_collections_cache():
    """Обновляет кэш списка коллекций"""
    global _cached_collections, _cached_collections_time
    config: Config = load_config()
    client = QdrantClient(url=config.qdrant_url)
    try:
        _cached_collections = [c.name for c in client.get_collections().collections]
        _cached_collections_time = time.time()
    except Exception as e:
        logger.error(f"Ошибка при получении списка коллекций: {e}")
        _cached_collections = []
    return _cached_collections


def get_cached_collections():
    """Возвращает кэшированный список коллекций"""
    global _cached_collections, _cached_collections_time
    if (_cached_collections is None or
        (time.time() - _cached_collections_time) > COLLECTIONS_CACHE_TTL):
        return refresh_collections_cache()
    return _cached_collections