"""Модуль для преобразования PDF в Markdown с помощью утилиты MinerU."""

import logging
from typing import Tuple
from config.config_manager import ConfigManager
# Импорт основной функции обработки из pdf_to_md_chunker (для обратной совместимости)
from .pdf_to_md_chunker import process_pdfs_and_chunk
# Импорт функции get_device из embeddings, чтобы не дублировать код
from core.embeddings import get_device
# Импорт нового мультиформатного конвертера
from .multi_format_converter import convert_files_to_md

logger = logging.getLogger(__name__)

# Get singleton instance of ConfigManager
config_manager = ConfigManager.get_instance()


def run_pdf_processing_from_config() -> Tuple[bool, str]:
    """
    Запускает обработку PDF, используя настройки из config.json.
    
    Returns:
        Tuple[bool, str]: (успех, статус)
    """
    config = config_manager.get()
    try:
        # Вызов функции обработки
        process_pdfs_and_chunk(
            input_pdf_dir=config.mineru_input_pdf_dir,
            output_md_dir=config.mineru_output_md_dir,
            enable_formula_parsing=config.mineru_enable_formula_parsing,
            enable_table_parsing=config.mineru_enable_table_parsing,
            model_source=config.mineru_model_source,
            models_dir=config.mineru_models_dir if config.mineru_models_dir else None,
            backend=config.mineru_backend,
            method=config.mineru_method,
            lang=config.mineru_lang,
            sglang_url=config.mineru_sglang_url if config.mineru_sglang_url else None,
            device=get_device(config.device) # Передаем устройство из настроек индексации
        )
        return True, "pdfs_processed_successfully"
    except Exception as e:
        logger.exception(f"Ошибка при обработке PDF: {e}")
        return False, "pdf_processing_error"


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