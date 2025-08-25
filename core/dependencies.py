"""Модуль для зависимостей FastAPI."""

import logging
import time
from fastapi import Depends
from config.config_manager import ConfigManager
from config.settings import Config
from tenacity import retry, stop_after_attempt, wait_fixed
from core.constants import RETRY_ATTEMPTS, RETRY_WAIT_TIME
from core.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)

# --- Кэш для проверки соединения ---
_last_connection_check = 0

# Get singleton instance of ConfigManager
config_manager = ConfigManager.get_instance()


def get_config() -> Config:
    """Зависимость для получения конфигурации."""
    return config_manager.get()


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_fixed(RETRY_WAIT_TIME))
def get_client(config: Config = Depends(get_config)):
    """Зависимость для получения клиента Qdrant."""
    global _last_connection_check
    
    client = get_qdrant_client(config)
    
    # Проверяем соединение только если прошло больше CONNECTION_CHECK_TTL секунд
    current_time = time.time()
    if current_time - _last_connection_check > config.qdrant_client_cache_ttl:
        try:
            client.get_collections()  # Проверка соединения
            _last_connection_check = current_time
            logger.debug("Проверка соединения с Qdrant выполнена")
        except Exception as e:
            logger.exception(f"Ошибка при проверке соединения с Qdrant: {e}")
            raise
    else:
        logger.debug("Проверка соединения с Qdrant пропущена (кэш)")
    
    return client


def verify_admin_access():
    """Зависимость для проверки прав администратора.
    
    В текущей реализации это заглушка, которая всегда возвращает 'admin'.
    В будущем можно реализовать полноценную аутентификацию.
    """
    # В текущей реализации у нас нет полноценной аутентификации,
    # поэтому просто возвращаем 'admin' как идентификатор пользователя
    return "admin"