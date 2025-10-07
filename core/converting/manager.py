"""Упрощенный менеджер конвертеров с использованием только Docling."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

from core.converting.docling_converter import DoclingConverter

logger = logging.getLogger(__name__)


class ConverterManager:
    """Упрощенный менеджер конвертеров с использованием только Docling."""
    
    def __init__(self):
        """Инициализация менеджера конвертеров."""
        self.docling_converter = DoclingConverter()
        self.docling_converter.reload_config()  # Перезагрузка после инициализации
        logger.info("ConverterManager инициализирован с Docling конвертером")
    
    def convert_file(self, file_path: Path, output_dir: Path) -> List[Path]:
        """Конвертировать файл с использованием Docling.
        
        Args:
            file_path: Путь к исходному файлу
            output_dir: Директория для сохранения результатов
            
        Returns:
            List[Path]: Список путей к созданным файлам
        """
        return self.docling_converter.convert_file(file_path, output_dir)
    
    def convert_directory(
        self, 
        input_dir: Path, 
        output_dir: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, List[Path]]:
        """Конвертировать все файлы в директории.
        
        Args:
            input_dir: Директория с исходными файлами
            output_dir: Директория для сохранения результатов
            progress_callback: Опциональная функция обратного вызова для прогресса
            
        Returns:
            Dict[str, List[Path]]: Словарь с результатами конвертации по файлам
        """
        return self.docling_converter.convert_directory(input_dir, output_dir, progress_callback)
    
    def is_supported(self, file_path: Path) -> bool:
        """Проверить, поддерживается ли формат файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True, если формат поддерживается
        """
        return self.docling_converter.is_supported(file_path)
    
    def get_supported_formats(self) -> List[str]:
        """Получить список поддерживаемых форматов файлов.
        
        Returns:
            List[str]: Список поддерживаемых форматов
        """
        return self.docling_converter.get_supported_formats()
    
    def reload_config(self):
        """Перезагрузить конфигурацию конвертеров."""
        self.docling_converter.reload_config()
        logger.info("Конфигурация ConverterManager перезагружена")