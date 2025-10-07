"""Унифицированный конвертер документов на основе Docling с EasyOCR."""

import logging
import shutil
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TableStructureOptions
)

from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class DoclingConverter:
    """Унифицированный конвертер документов на основе Docling с EasyOCR."""
    
    def __init__(self):
        """Инициализация конвертера."""
        self.config_manager = ConfigManager.get_instance()
        self._initialize_converter()
    
    def _initialize_converter(self):
        """Инициализация конвертера Docling с настройками из конфигурации."""
        config = self.config_manager.get()
        
        try:
            # Создаем опции для OCR с указанием на локальные модели
            ocr_options = EasyOcrOptions(
                lang=[config.docling_ocr_lang],
                model_storage_directory=str(config.easyocr_models_path),
                download_enabled=config.auto_download_models
            )
            
            # Создаем PdfPipelineOptions с учетом совместимости версий
            try:
                if config.docling_use_tables:
                    pipeline_options = PdfPipelineOptions(
                        do_ocr=config.docling_use_ocr,
                        do_table_structure=config.docling_use_tables,
                        ocr_options=ocr_options,
                        table_structure_options=TableStructureOptions(
                            do_table_structure=config.docling_use_tables
                        )
                    )
                else:
                    pipeline_options = PdfPipelineOptions(
                        do_ocr=config.docling_use_ocr,
                        do_table_structure=config.docling_use_tables,
                        ocr_options=ocr_options
                    )
            except TypeError:
                # Fallback для старых версий Docling
                pipeline_options = PdfPipelineOptions(
                    do_ocr=config.docling_use_ocr,
                    ocr_options=ocr_options
                )
            
            # Инициализация конвертера с настроенными опциями
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pipeline_options,
                },
            )
            
            # Preload-тест для оффлайна
            try:
                # Тестовый вызов без текста, чтобы проверить модели
                from easyocr import Reader
                test_reader = Reader([config.docling_ocr_lang], model_storage_directory=str(config.easyocr_models_path), download_enabled=False)
                logger.info("EasyOCR preload успешен: модели найдены локально")
            except Exception as preload_e:
                logger.warning(f"EasyOCR preload failed (но fallback на Docling): {preload_e}")
            
            logger.info("Docling конвертер инициализирован с локальными моделями EasyOCR")
        except Exception as e:
            logger.warning(f"Ошибка с локальными опциями: {e}. Fallback на дефолт.")
            self.converter = DocumentConverter()
    
    def convert_file(self, file_path: Path, output_dir: Path) -> List[Path]:
        """Конвертировать файл в Markdown.
        
        Args:
            file_path: Путь к исходному файлу
            output_dir: Директория для сохранения результатов
            
        Returns:
            List[Path]: Список путей к созданным файлам
        """
        try:
            config = self.config_manager.get()
            
            # В новой версии Docling структура PdfPipelineOptions может быть несовместима
            # Попробуем создать конвертер с корректными опциями
            try:
                # Попробуем сначала создать опции с минимальным набором параметров
                # В зависимости от версии Docling, структура может отличаться
                
                # Сначала создаем ocr_options
                ocr_options = EasyOcrOptions(
                    lang=[config.docling_ocr_lang]
                )
                
                # Создаем PdfPipelineOptions с разным набором параметров в зависимости от поддержки
                try:
                    # Пробуем создать с полным набором параметров
                    if config.docling_use_tables:
                        pipeline_options = PdfPipelineOptions(
                            do_ocr=config.docling_use_ocr,
                            do_table_structure=config.docling_use_tables,
                            ocr_options=ocr_options,
                            table_structure_options=TableStructureOptions(
                                do_table_structure=config.docling_use_tables
                            )
                        )
                    else:
                        pipeline_options = PdfPipelineOptions(
                            do_ocr=config.docling_use_ocr,
                            do_table_structure=config.docling_use_tables,
                            ocr_options=ocr_options
                        )
                except TypeError:
                    # Если это не сработало, пробуем с минимальным набором параметров
                    pipeline_options = PdfPipelineOptions(
                        do_ocr=config.docling_use_ocr,
                        ocr_options=ocr_options
                    )
                
                # Добавление опций для распознавания формул, если включено
                # Note: Formula recognition options may not be available in all versions of Docling
                if config.docling_use_formulas:
                    logger.warning("Распознавание формул запрошено, но настройки формулы могут отличаться в текущей версии Docling")
                
                # Создание нового конвертера с определенными опциями
                temp_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: pipeline_options,
                    },
                )
                
                # Конвертация документа
                result = temp_converter.convert(file_path)
            except Exception as e:
                logger.warning(f"Ошибка при использовании настроенных опций: {e}. Пробуем использовать конвертер по умолчанию.")
                
                # Если с настройками не получается, используем базовый конвертер
                temp_converter = DocumentConverter()
                result = temp_converter.convert(file_path)
            
            # Формирование пути для выходного файла
            output_path = output_dir / f"{file_path.stem}.md"
            
            # Сохранение результата в Markdown
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.document.export_to_markdown())
            
            # Обработка изображений, если они есть
            image_paths = []
            try:
                # Экспорт в словарь для извлечения изображений
                doc_dict = result.document.export_to_dict()
                
                if "images" in doc_dict and doc_dict["images"]:
                    images_dir = output_dir / f"{file_path.stem}_images"
                    images_dir.mkdir(exist_ok=True)
                    
                    for img_id, img_data in doc_dict["images"].items():
                        if isinstance(img_data, dict) and "data" in img_data:
                            # Декодирование base64 данных изображения
                            img_bytes = base64.b64decode(img_data["data"])
                            
                            # Определение расширения из MIME типа или использование ID
                            mime_type = img_data.get("mime_type", "")
                            if "jpeg" in mime_type or "jpg" in mime_type:
                                ext = ".jpg"
                            elif "png" in mime_type:
                                ext = ".png"
                            else:
                                ext = ".png"  # По умолчанию
                            
                            img_path = images_dir / f"{img_id}{ext}"
                            with open(img_path, 'wb') as img_file:
                                img_file.write(img_bytes)
                            image_paths.append(img_path)
            except Exception as e:
                logger.warning(f"Не удалось извлечь изображения из {file_path}: {e}")
            
            logger.info(f"Документ {file_path} успешно конвертирован в {output_path}")
            return [output_path] + image_paths
            
        except Exception as e:
            logger.error(f"Ошибка при конвертации {file_path}: {e}")
            return []
    
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
        results = {}
        
        # Создание выходной директории, если она не существует
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Получение списка всех файлов в директории
        all_files = list(input_dir.rglob("*")) if input_dir.is_dir() else []
        files = [f for f in all_files if f.is_file() and self.is_supported(f)]
        
        total_files = len(files)
        processed_count = 0
        
        # Обработка каждого файла в директории
        for file_path in files:
            # Отчет о прогрессе
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, total_files)
            
            file_results = self.convert_file(file_path, output_dir)
            results[str(file_path)] = file_results
        
        return results
    
    def is_supported(self, file_path: Path) -> bool:
        """Проверить, поддерживается ли формат файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True, если формат поддерживается
        """
        try:
            # Проверка расширения файла
            extension = file_path.suffix.lower()
            supported_extensions = {
                '.pdf', '.docx', '.pptx', '.xlsx', 
                '.png', '.jpg', '.jpeg', '.tiff', '.bmp',
                '.html', '.htm', '.md'
            }
            return extension in supported_extensions
        except Exception:
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Получить список поддерживаемых форматов файлов.
        
        Returns:
            List[str]: Список поддерживаемых форматов
        """
        return [
            'PDF', 'DOCX', 'PPTX', 'XLSX',
            'PNG', 'JPG', 'JPEG', 'TIFF', 'BMP',
            'HTML', 'HTM', 'MD'
        ]
    
    def reload_config(self):
        """Перезагрузить конфигурацию и переинициализировать конвертер."""
        self._initialize_converter()
        logger.info("Конфигурация Docling конвертера перезагружена")