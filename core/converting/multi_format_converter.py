"""Упрощенный конвертер форматов с использованием Docling."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable

from core.converting.docling_converter import DoclingConverter
from core.converting.file_tracker import FileTracker

logger = logging.getLogger(__name__)


class MultiFormatConverter:
    """Упрощенный конвертер форматов с использованием Docling."""
    
    def __init__(self):
        """Инициализация конвертера."""
        self.docling_converter = DoclingConverter()
        self.tracker = FileTracker()
        logger.info("MultiFormatConverter инициализирован с Docling и FileTracker")
    
    def convert_files(
        self, 
        input_dir: str, 
        output_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, List[Path]]:
        """Конвертировать файлы из входной директории в выходную.
        
        Args:
            input_dir: Путь к входной директории
            output_dir: Путь к выходной директории
            progress_callback: Опциональная функция обратного вызова для прогресса
            
        Returns:
            Dict[str, List[Path]]: Результаты конвертации по файлам
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Проверка существования входной директории
        if not input_path.exists():
            raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")
        
        # Создание выходной директории
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сбросить флаг остановки перед началом
        self.tracker.reset_stop_flag()
        
        # Получаем список необработанных файлов
        unprocessed_files = self.tracker.get_unprocessed_files(input_path)
        total_files = len(unprocessed_files)
        
        results = {}
        processed_count = 0
        
        for file_path in unprocessed_files:
            # Проверить, не остановлена ли конвертация
            if self.tracker.is_conversion_stopped():
                logger.info("Конвертация остановлена пользователем")
                break
                
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, total_files)
                
            # Обновляем статус файла на "в процессе"
            self.tracker.update_file(file_path, "in_progress")
            
            try:
                file_results = self.docling_converter.convert_file(file_path, output_path)
                if file_results:
                    self.tracker.update_file(file_path, "converted")
                    results[str(file_path)] = file_results
                else:
                    self.tracker.update_file(file_path, "error")
                    results[str(file_path)] = []
            except Exception as e:
                logger.error(f"Ошибка при конвертации {file_path}: {e}")
                self.tracker.update_file(file_path, "error")
                results[str(file_path)] = []
        
        return results
    
    def is_supported(self, file_path: str) -> bool:
        """Проверить, поддерживается ли формат файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True, если формат поддерживается
        """
        return self.docling_converter.is_supported(Path(file_path))
    
    def get_supported_formats(self) -> List[str]:
        """Получить список поддерживаемых форматов.
        
        Returns:
            List[str]: Список поддерживаемых форматов
        """
        return self.docling_converter.get_supported_formats()
    
    def reload_config(self):
        """Перезагрузить конфигурацию конвертера."""
        self.docling_converter.reload_config()
        logger.info("Конфигурация MultiFormatConverter перезагружена")
    
    def stop_conversion(self):
        """Остановить процесс конвертации."""
        self.tracker.stop_conversion()


def convert_files_to_md(input_dir: str, output_dir: str) -> tuple[bool, str]:
    """
    Convert files of various formats to Markdown using Docling.
    
    Args:
        input_dir (str): Directory containing files to convert
        output_dir (str): Directory to save converted Markdown files
        
    Returns:
        Tuple[bool, str]: (success, status message)
    """
    try:
        converter = MultiFormatConverter()
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists() or not input_path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert files
        results = converter.convert_files(input_dir, output_dir)
        
        # Count total files processed
        successful_conversions = 0
        for file_results in results.values():
            if file_results:  # If there are results for this file
                successful_conversions += 1
        
        if successful_conversions == 0:
            logger.warning("No files were successfully converted")
        else:
            logger.info(f"Successfully converted {successful_conversions} files to Markdown")
        
        return True, f"converted_{successful_conversions}_files"
        
    except Exception as e:
        logger.exception(f"Error converting files: {e}")
        return False, f"conversion_error: {str(e)}"


def get_supported_formats() -> List[str]:
    """
    Get list of supported file formats.
    
    Returns:
        List[str]: List of supported formats
    """
    converter = MultiFormatConverter()
    return converter.get_supported_formats()