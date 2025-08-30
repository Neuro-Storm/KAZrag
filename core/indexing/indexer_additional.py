"""Дополнительные функции для индексации."""

import logging
from typing import Tuple

from core.indexer import run_indexing_logic

logger = logging.getLogger(__name__)


def run_indexing_from_config() -> Tuple[bool, str]:
    """
    Запускает индексацию, используя настройки из config.json.
    
    Returns:
        Tuple[bool, str]: (успех, статус)
    """
    try:
        # Вызов функции индексации с настройками из конфига
        success, status = run_indexing_logic()
        return success, status
    except Exception as e:
        logger.exception(f"Ошибка при индексации: {e}")
        return False, f"indexing_error: {str(e)}"