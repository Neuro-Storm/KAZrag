"""Унифицированный конвертер документов на основе Docling с EasyOCR."""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
    PdfPipelineOptions,
    TableStructureOptions,
    EasyOcrOptions
)
from docling.datamodel import vlm_model_specs  # Правильный импорт для Granite specs
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

try:
    from docling.datamodel.table_structure import TableFormerMode
    TABLE_FORMER_MODE = TableFormerMode.ACCURATE
except ImportError:
    TABLE_FORMER_MODE = "accurate"  # Fallback

import base64  # Moved down to avoid import issues if needed

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
        config = self.config_manager.get()
        backend = config.docling_backend
        auto_download = getattr(config, "auto_download_models", True)
        use_local_only = getattr(config, "use_local_only", True)

        # Для Granite: env для HF-моделей (TRANSFORMERS)
        if backend == "granite":
            if use_local_only:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                logger.info("TRANSFORMERS_OFFLINE=1 для Granite (локальный режим)")
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            # Для classic: общий Docling оффлайн
            if use_local_only and not auto_download:
                os.environ["DOCLING_OFFLINE_MODE"] = "1"
                logger.info("DOCLING_OFFLINE_MODE=1 (режим оффлайн)")
            else:
                os.environ.pop("DOCLING_OFFLINE_MODE", None)
                logger.info("DOCLING_OFFLINE_MODE снят (режим онлайн/скачивание разрешено)")

        logger.info(f"DOCLING_OFFLINE_MODE={os.environ.get('DOCLING_OFFLINE_MODE', 'NOT SET')}")


    def _get_accelerator(self, device_str: str) -> AcceleratorOptions:
        """Get accelerator options based on device string."""
        device_map = {
            "cpu": AcceleratorDevice.CPU,
            "gpu": AcceleratorDevice.CUDA,
            "auto": AcceleratorDevice.AUTO,
        }
        device = device_map.get(device_str.lower(), AcceleratorDevice.AUTO)
        return AcceleratorOptions(device=device)



    def _ensure_ocr_lang_list(self, lang):
        """Ensure lang is a list of strings for EasyOcrOptions."""
        if isinstance(lang, str):
            return [lang]
        elif isinstance(lang, list):
            return lang
        else:
            # Default to Russian if unexpected type
            return ["ru"]

    def _initialize_converter(self):
        """Инициализация конвертера Docling с настройками из конфигурации."""
        config = self.config_manager.get()
        backend = config.docling_backend
        device_str = config.docling_device.lower()
        accelerator = self._get_accelerator(device_str)

        # Установка HF_HOME для кэша (избегаем deprecated TRANSFORMERS_CACHE)
        cache_dir = config.huggingface_cache_path
        os.environ["HF_HOME"] = str(cache_dir)

        # Для Granite: путь к локальным моделям
        if backend == "granite":
            granite_dir = config.granite_models_dir
            logger.info(f"Granite setup: device={device_str}, cache={cache_dir}, models={granite_dir}")
        else:
            logger.info(f"Classic Docling setup: device={device_str}")

        if backend == "granite":
            # Granite uses VlmPipeline with TRANSFORMERS (for CPU/GPU)
            # Ensure lang is always a list for EasyOcrOptions
            ocr_lang_list = self._ensure_ocr_lang_list(config.docling_ocr_lang)
            
            vlm_options = VlmPipelineOptions(
                vlm_model_specs=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,  # Правильная константа для CPU/GPU
                accelerator_options=accelerator,
                table_structure_options=TableStructureOptions(
                    table_former_mode=TABLE_FORMER_MODE,
                    ocr_options=EasyOcrOptions(lang=ocr_lang_list),
                ) if config.docling_use_ocr else None,
                local_files_only=(os.environ.get("TRANSFORMERS_OFFLINE") == "1"),
                enable_ocr=config.docling_use_ocr,
                enable_tables=config.docling_use_tables,
                enable_formulas=config.docling_use_formulas,
            )
            format_opts = {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=vlm_options,
                )
            }
            self.converter = DocumentConverter(format_options=format_opts)
        else:
            # Classic mode with OCR and other options
            # Apply OCR and table options to classic pipeline
            kwargs = {
                "accelerator_options": accelerator,
            }
            
            if config.docling_use_ocr:
                # Ensure lang is always a list for EasyOcrOptions
                ocr_lang_list = self._ensure_ocr_lang_list(config.docling_ocr_lang)
                kwargs["ocr_options"] = EasyOcrOptions(lang=ocr_lang_list)
            
            if config.docling_use_tables:
                kwargs["table_structure_options"] = TableStructureOptions(
                    table_former_mode=TABLE_FORMER_MODE,
                )
            
            pdf_opts = PdfPipelineOptions(**kwargs)
            format_opts = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
            self.converter = DocumentConverter(format_options=format_opts)
    
    def convert_file(self, file_path: Path, output_dir: Path) -> List[Path]:
        """Конвертировать файл в Markdown.
        
        Args:
            file_path: Путь к исходному файлу
            output_dir: Директория для сохранения результатов
            
            Returns: List[Path]
        """
        try:
            config = self.config_manager.get()
            
            # Проверяем и устанавливаем режим оффлайн при необходимости
            auto_download = getattr(config, "auto_download_models", True)
            # Offline mode handled in _setup_model_paths; refresh if config changed
            self._setup_model_paths()
            
            try:
                # Используем основной конвертер, инициализированный в _initialize_converter
                result = self.converter.convert(file_path)
            finally:
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
        self.config = self.config_manager.reload()
        self._setup_model_paths()
        self._initialize_converter()
        logger.info("Конфигурация Docling конвертера перезагружена")