"""Модуль для преобразования PDF в Markdown с помощью утилиты MinerU."""

import logging
from typing import Tuple

from config.config_manager import ConfigManager

# Импорт функции get_device из embeddings, чтобы не дублировать код
# Импорт нового мультиформатного конвертера
from .multi_format_converter import convert_files_to_md

logger = logging.getLogger(__name__)

# Get singleton instance of ConfigManager
config_manager = ConfigManager.get_instance()


def run_multi_format_processing_from_config() -> Tuple[bool, str]:
    """
    Запускает обработку файлов различных форматов, используя настройки из config.json.
    
    Returns:
        Tuple[bool, str]: (успех, статус)
    """
    config = config_manager.get()
    try:
        # Вызов функции обработки для всех форматов
        success, status = convert_files_to_md(
            input_dir=config.mineru_input_pdf_dir,  # Используем ту же директорию для всех форматов
            output_dir=config.mineru_output_md_dir
        )
        if success:
            return True, f"files_processed_successfully_{status}"
        else:
            return False, status
    except Exception as e:
        logger.exception(f"Ошибка при обработке файлов: {e}")
        return False, "file_processing_error"