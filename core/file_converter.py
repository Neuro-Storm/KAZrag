"""Модуль для преобразования PDF в Markdown с помощью утилиты MinerU."""

import logging
from typing import Tuple
from config.settings import load_config, Config
# Импорт основной функции обработки из pdf_to_md_chunker
from .pdf_to_md_chunker import process_pdfs_and_chunk
# Импорт функции get_device из embeddings, чтобы не дублировать код
from core.embeddings import get_device

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


def run_pdf_processing_from_config() -> Tuple[bool, str]:
    """
    Запускает обработку PDF, используя настройки из config.json.
    
    Returns:
        Tuple[bool, str]: (успех, статус)
    """
    config: Config = load_config()
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
        logger.error(f"Ошибка при обработке PDF: {e}")
        import traceback
        traceback.print_exc()
        return False, f"pdf_processing_error_{str(e).replace(' ', '_')}"