"""Унифицированный конвертер документов на основе Docling с EasyOCR."""

import logging
import base64
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    EasyOcrOptions
)
from docling.document_converter import PdfFormatOption
# Импорт для TableFormerMode (если используется; fallback на default)
try:
    from docling.datamodel.table_structure import TableFormerMode
except ImportError:
    TableFormerMode = None  # Default будет ACCURATE

from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class DoclingConverter:
    """Унифицированный конвертер документов на основе Docling с EasyOCR."""

    def __init__(self):
        """Инициализация конвертера."""
        self.config_manager = ConfigManager.get_instance()
        self.config = self.config_manager.get()
        self._setup_model_paths()
        self._initialize_converter()

    def _setup_model_paths(self):
        """Настройка режима оффлайн/онлайн для Docling."""
        # Устанавливаем только переменную оффлайн режима, не создаем локальные директории
        config = self.config_manager.get()
        auto_download = getattr(config, "auto_download_models", True)
        use_local_only = getattr(config, "use_local_only", True)

        # ВАЖНО: Docling 2.55+ проверяет только этот флаг для скачивания моделей
        if use_local_only and not auto_download:
            os.environ["DOCLING_OFFLINE_MODE"] = "1"
            logger.info("DOCLING_OFFLINE_MODE=1 (режим оффлайн)")
        else:
            # Разрешаем скачивание - удаляем переменную окружения полностью
            os.environ.pop("DOCLING_OFFLINE_MODE", None)
            logger.info("DOCLING_OFFLINE_MODE снят (режим онлайн/скачивание разрешено)")

        logger.info(f"DOCLING_OFFLINE_MODE={os.environ.get('DOCLING_OFFLINE_MODE', 'NOT SET')}")



    def _initialize_converter(self):
        """Инициализация конвертера Docling с настройками из конфигурации."""
        # Убедимся, что оффлайн-режим снят при необходимости
        auto_download = getattr(self.config, "auto_download_models", True)
        orig_offline = os.environ.get("DOCLING_OFFLINE_MODE")
        
        if auto_download and orig_offline is not None:
            os.environ.pop("DOCLING_OFFLINE_MODE", None)
            logger.info(f"DOCLING_OFFLINE_MODE снят перед инициализацией (был {orig_offline})")
        elif auto_download:
            logger.info("DOCLING_OFFLINE_MODE уже не установлен")
        else:
            logger.info(f"DOCLING_OFFLINE_MODE оставлен как есть: {orig_offline}")
        
        try:
            # Создаем основной конвертер с учетом текущих настроек окружения
            self.converter = DocumentConverter()
            logger.info("Главный Docling конвертер успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка при инициализации конвертера: {e}")
            raise
        
        logger.info("Docling конвертер инициализирован (использует стандартные пути)")
    
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
            
            # Используем стандартные пути для моделей Docling
            artifacts_path = None  # Не указываем специфический путь, пусть Docling использует стандартные
            
            # Опции для таблиц (если включено)
            table_options = None
            if hasattr(config, 'docling_use_tables') and config.docling_use_tables:
                # Устанавливаем режим таблицы, используя правильное значение
                table_mode = 'accurate'  # Значение по умолчанию
                if hasattr(config, 'docling_table_mode'):
                    if config.docling_table_mode.lower() in ['fast', 'accurate']:
                        table_mode = config.docling_table_mode.lower()
                
                table_options = TableStructureOptions(
                    do_cell_matching=True,  # Default: True (улучшает mapping к PDF cells)
                    mode=table_mode  # Правильное строковое значение
                )

            # Опции для OCR с языком из конфигурации
            ocr_options = None
            if config.docling_use_ocr:
                # Используем язык из конфигурации
                ocr_lang = getattr(config, 'docling_ocr_lang', 'en')  # по умолчанию английский
                # Для EasyOCR язык указывается как список
                if isinstance(ocr_lang, str):
                    ocr_lang_list = [ocr_lang]
                else:
                    ocr_lang_list = ocr_lang
                
                ocr_options = EasyOcrOptions(
                    lang=ocr_lang_list,
                    download_enabled=True  # Явно разрешить скачивание в EasyOCR
                )

            # Определяем, включать ли расширенное обнаружение таблиц
            enable_table_detection = (
                getattr(config, 'docling_use_tables', False) and 
                getattr(config, 'docling_table_detection_advanced', True)
            )
            
            # PDF опции (OCR по умолчанию, если EasyOCR доступен)
            if artifacts_path:
                pdf_pipeline = PdfPipelineOptions(
                    artifacts_path=artifacts_path,
                    enable_remote_services=config.auto_download_models,  # Разрешить скачивание, если auto_download=True
                    do_table_structure=enable_table_detection,  # Управление расширенным обнаружением таблиц
                    table_structure_options=table_options,
                    do_ocr=config.docling_use_ocr,  # Включаем OCR если нужно
                    ocr_options=ocr_options,  # Используем наши настройки OCR
                    generate_page_images=getattr(config, 'docling_enable_page_images', True)  # Управление генерацией изображений страниц
                )
            else:
                pdf_pipeline = PdfPipelineOptions(
                    enable_remote_services=config.auto_download_models,  # Разрешить скачивание, если auto_download=True
                    do_table_structure=enable_table_detection,  # Управление расширенным обнаружением таблиц
                    table_structure_options=table_options,
                    do_ocr=config.docling_use_ocr,  # Включаем OCR если нужно
                    ocr_options=ocr_options,  # Используем наши настройки OCR
                    generate_page_images=getattr(config, 'docling_enable_page_images', True)  # Управление генерацией изображений страниц
                )

            # format_options для PDF (остальные — default)
            format_options = {
                InputFormat.PDF: PdfFormatOption(  # Используем PdfFormatOption вместо прямого PdfPipelineOptions
                    pipeline_options=pdf_pipeline
                ),
            }

            # Проверяем и устанавливаем режим оффлайн при необходимости
            auto_download = getattr(config, "auto_download_models", True)
            if auto_download:
                # Убедимся, что оффлайн режим не установлен
                os.environ.pop("DOCLING_OFFLINE_MODE", None)
                logger.info("DOCLING_OFFLINE_MODE снят для конвертации")
            else:
                # Установим оффлайн режим, если автозагрузка отключена
                os.environ["DOCLING_OFFLINE_MODE"] = "1"
                logger.info("DOCLING_OFFLINE_MODE=1 (режим оффлайн)")
            
            try:
                # Создание нового конвертера с опциями
                temp_converter = DocumentConverter(
                    format_options=format_options,
                )
                
                # Конвертация документа (с лимитами для памяти)
                result = temp_converter.convert(file_path)
            finally:
                # Никогда не восстанавливаем DOCLING_OFFLINE_MODE в рамках этой операции, 
                # т.к. он управляется на уровне _setup_model_paths
                # Восстановление происходит при изменении конфигурации
                pass
            
            # Формирование пути для выходного файла
            output_path = output_dir / f"{file_path.stem}.md"
            
            # Сохранение результата в Markdown
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.document.export_to_markdown())
            except MemoryError as e:
                logger.error(f"Ошибка памяти при записи результата для {file_path}: {e}")
                return []
            
            # Обработка изображений (только для PDF, где Docling извлекает встраиваемые figs)
            image_paths = []
            try:
                doc_dict = result.document.export_to_dict()
                if "images" in doc_dict and doc_dict["images"]:
                    images_dir = output_dir / f"{file_path.stem}_images"
                    images_dir.mkdir(exist_ok=True)
                    
                    for img_id, img_data in doc_dict["images"].items():
                        if isinstance(img_data, dict) and "data" in img_data:
                            img_bytes = base64.b64decode(img_data["data"])
                            img_path = images_dir / f"{img_id}.png"
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_bytes)
                            image_paths.append(img_path)
                            
                            # Замена base64 в MD на относительный путь
                            with open(output_path, 'r', encoding='utf-8') as f:
                                md_content = f.read()
                            md_content = md_content.replace(
                                f"data:image/png;base64,{img_data['data']}",
                                f"./{file_path.stem}_images/{img_id}.png"
                            )
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(md_content)
            except Exception as e:
                logger.warning(f"Ошибка при обработке изображений для {file_path}: {e}")
            
            return [output_path] + image_paths
            
        except MemoryError as e:
            logger.error(f"Ошибка памяти при конвертации {file_path}: {e}")
            logger.info(f"Попробуйте закрыть другие приложения.")
            return []
        except Exception as e:
            logger.exception(f"Ошибка при конвертации файла {file_path}: {e}")
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
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_files = list(input_dir.rglob("*")) if input_dir.is_dir() else []
        files = [f for f in all_files if f.is_file() and self.is_supported(f)]
        
        total_files = len(files)
        processed_count = 0
        
        for file_path in files:
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, total_files)
            
            file_results = self.convert_file(file_path, output_dir)
            results[str(file_path)] = file_results
        
        return results
    
    def is_supported(self, file_path: Path) -> bool:
        """Проверить, поддерживается ли формат файла."""
        try:
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
        """Получить список поддерживаемых форматов файлов."""
        return [
            'PDF', 'DOCX', 'PPTX', 'XLSX',
            'PNG', 'JPG', 'JPEG', 'TIFF', 'BMP',
            'HTML', 'HTM', 'MD'
        ]
    
    def reload_config(self):
        """Перезагрузить конфигурацию и переинициализировать конвертер."""
        self.config = self.config_manager.get()
        self._setup_model_paths()
        self._initialize_converter()
        logger.info("Конфигурация Docling конвертера перезагружена")