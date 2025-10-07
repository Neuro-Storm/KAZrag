"""Упрощенный обработчик файлов с использованием Docling."""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable

from core.converting.docling_converter import DoclingConverter

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Перечисление поддерживаемых типов файлов."""
    PDF = "pdf"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"
    HTML = "html"
    UNKNOWN = "unknown"


@dataclass
class FileProcessingResult:
    """Класс для результатов обработки файлов."""
    file_path: Path
    output_path: Optional[Path]
    success: bool
    error_message: Optional[str]
    processing_time: float = 0.0
    metadata: Dict = None


class FileProcessor:
    """Упрощенный обработчик файлов с использованием Docling."""
    
    def __init__(self, progress_callback: Optional[Callable[[int, int], None]] = None):
        """Инициализация обработчика файлов.
        
        Args:
            progress_callback: Опциональная функция обратного вызова для отчета о прогрессе
        """
        self.progress_callback = progress_callback
        self.docling_converter = DoclingConverter()
        
    def scan_directory(
        self, 
        directory: Path, 
        recursive: bool = True
    ) -> Dict[FileType, List[Path]]:
        """Сканировать директорию и найти поддерживаемые файлы.
        
        Args:
            directory: Директория для сканирования
            recursive: Сканировать поддиректории рекурсивно
            
        Returns:
            Dict[FileType, List[Path]]: Словарь файлов, сгруппированных по типам
        """
        if not directory.exists():
            raise FileNotFoundError(f"Директория не найдена: {directory}")
            
        if not directory.is_dir():
            raise ValueError(f"Путь не является директорией: {directory}")
            
        try:
            # Использование pathlib для сканирования файлов
            if recursive:
                all_files = list(directory.rglob("*"))
            else:
                all_files = list(directory.iterdir())
                
            # Фильтрация только файлов (не директорий)
            files = [f for f in all_files if f.is_file()]
            
            # Инициализация словаря для всех типов файлов
            files_by_type = {file_type: [] for file_type in FileType}
            
            # Классификация файлов по типам
            for file_path in files:
                file_type = self._classify_file(file_path)
                files_by_type[file_type].append(file_path)
                
            # Логирование результатов
            total_files = sum(len(files) for files in files_by_type.values())
            logger.info(f"Найдено {total_files} файлов в {directory}")
            
            return files_by_type
            
        except PermissionError:
            raise PermissionError(f"Нет доступа к директории: {directory}")
        except Exception as e:
            logger.exception(f"Ошибка при сканировании директории {directory}: {e}")
            raise
    
    def _classify_file(self, file_path: Path) -> FileType:
        """Классифицировать файл по типу.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            FileType: Тип файла
        """
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return FileType.PDF
        elif extension in {'.docx', '.doc', '.txt', '.md', '.rtf'}:
            return FileType.DOCUMENT
        elif extension in {'.pptx', '.ppt'}:
            return FileType.PRESENTATION
        elif extension in {'.xlsx', '.xls', '.csv'}:
            return FileType.SPREADSHEET
        elif extension in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}:
            return FileType.IMAGE
        elif extension in {'.html', '.htm'}:
            return FileType.HTML
        else:
            return FileType.UNKNOWN
    
    def process_files(
        self, 
        files_by_type: Dict[FileType, List[Path]], 
        output_dir: Path
    ) -> List[FileProcessingResult]:
        """Обработать файлы с использованием Docling.
        
        Args:
            files_by_type: Файлы, сгруппированные по типам
            output_dir: Директория для сохранения результатов
            
        Returns:
            List[FileProcessingResult]: Результаты обработки
        """
        results = []
        
        # Создание выходной директории, если она не существует
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Подсчет общего количества файлов для прогресса
        total_files = sum(len(files) for files in files_by_type.values())
        processed_count = 0
        
        # Обработка каждого типа файла
        for file_type, files in files_by_type.items():
            if not files:
                continue
                
            logger.info(f"Обработка {len(files)} файлов типа {file_type.value}")
            
            for file_path in files:
                try:
                    # Отчет о прогрессе
                    processed_count += 1
                    if self.progress_callback:
                        self.progress_callback(processed_count, total_files)
                    
                    # Обработка файла
                    file_results = self.docling_converter.convert_file(file_path, output_dir)
                    
                    # Проверка успешности обработки
                    if file_results:
                        results.append(FileProcessingResult(
                            file_path=file_path,
                            output_path=file_results[0] if file_results else None,
                            success=True,
                            error_message=None,
                            metadata={"file_type": file_type.value, "all_files": file_results}
                        ))
                    else:
                        results.append(FileProcessingResult(
                            file_path=file_path,
                            output_path=None,
                            success=False,
                            error_message="Обработка файла не удалась (нет выходных файлов)",
                            metadata={"file_type": file_type.value}
                        ))
                    
                except Exception as e:
                    logger.error(f"Ошибка при обработке {file_path}: {e}")
                    results.append(FileProcessingResult(
                        file_path=file_path,
                        output_path=None,
                        success=False,
                        error_message=str(e),
                        metadata={"file_type": file_type.value}
                    ))
        
        return results