"""Модуль для управления метаданными для чанков."""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from docx import Document
    HAS_PYTHON_DOCX = True
except ImportError:
    HAS_PYTHON_DOCX = False

logger = logging.getLogger(__name__)

class MetadataManager:
    """Класс для управления метаданными для чанков."""
    
    def __init__(self):
        """Инициализация MetadataManager."""
        self.custom_schemas = {}
        
    def add_metadata_to_chunk(self, chunk: Any, file_path: Path, custom_fields: Optional[Dict[str, Any]] = None) -> Any:
        """
        Добавляет метаданные к чанку.
        
        Args:
            chunk (Any): Чанк документа.
            file_path (Path): Путь к исходному файлу.
            custom_fields (Optional[Dict[str, Any]]): Пользовательские поля метаданных.
            
        Returns:
            Any: Чанк с добавленными метаданными.
        """
        # Инициализируем метаданные, если их нет
        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
            chunk.metadata = {}
            
        # Добавляем базовые метаданные
        chunk.metadata["source"] = str(file_path)
        chunk.metadata["file_name"] = file_path.name
        chunk.metadata["file_extension"] = file_path.suffix.lower()
        chunk.metadata["file_size"] = file_path.stat().st_size if file_path.exists() else 0
        chunk.metadata["file_modified"] = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
        
        # Добавляем хэш файла для идентификации
        chunk.metadata["file_hash"] = self._get_file_hash(file_path)
        
        # Извлекаем метаданные из файла
        file_metadata = self.extract_metadata_from_file(file_path)
        chunk.metadata.update(file_metadata)
        
        # Добавляем пользовательские поля
        if custom_fields:
            chunk.metadata.update(custom_fields)
            
        return chunk
    
    def extract_metadata_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Извлекает метаданные из исходного документа.
        
        Args:
            file_path (Path): Путь к файлу.
            
        Returns:
            Dict[str, Any]: Словарь с метаданными.
        """
        metadata = {}
        
        try:
            if file_path.suffix.lower() == '.pdf' and HAS_PYPDF2:
                metadata.update(self._extract_pdf_metadata(file_path))
            elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif'] and HAS_PIL:
                metadata.update(self._extract_image_metadata(file_path))
            elif file_path.suffix.lower() == '.docx' and HAS_PYTHON_DOCX:
                metadata.update(self._extract_docx_metadata(file_path))
            else:
                logger.debug(f"Не поддерживаемый тип файла для извлечения метаданных: {file_path}")
        except Exception as e:
            logger.warning(f"Ошибка при извлечении метаданных из файла {file_path}: {e}")
            
        return metadata
    
    def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Извлекает метаданные из PDF файла.
        
        Args:
            file_path (Path): Путь к PDF файлу.
            
        Returns:
            Dict[str, Any]: Словарь с метаданными PDF.
        """
        metadata = {}
        try:
            with open(file_path, 'rb') as f:
                pdf = PdfReader(f)
                if pdf.metadata:
                    # Извлекаем стандартные метаданные
                    pdf_metadata = {
                        "title": pdf.metadata.get("/Title"),
                        "author": pdf.metadata.get("/Author"),
                        "subject": pdf.metadata.get("/Subject"),
                        "creator": pdf.metadata.get("/Creator"),
                        "producer": pdf.metadata.get("/Producer"),
                        "creation_date": pdf.metadata.get("/CreationDate"),
                        "modification_date": pdf.metadata.get("/ModDate"),
                    }
                    # Фильтруем None значения
                    metadata.update({k: v for k, v in pdf_metadata.items() if v is not None})
                    
                    # Добавляем количество страниц
                    metadata["page_count"] = len(pdf.pages)
        except Exception as e:
            logger.warning(f"Ошибка при извлечении метаданных из PDF {file_path}: {e}")
            
        return metadata
    
    def _extract_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Извлекает метаданные из изображения.
        
        Args:
            file_path (Path): Путь к изображению.
            
        Returns:
            Dict[str, Any]: Словарь с метаданными изображения.
        """
        metadata = {}
        try:
            image = Image.open(file_path)
            metadata["image_format"] = image.format
            metadata["image_mode"] = image.mode
            metadata["image_size"] = image.size  # (width, height)
            
            # Извлекаем EXIF данные
            if hasattr(image, '_getexif') and image._getexif():
                exif_data = image._getexif()
                if exif_data:
                    exif = {}
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif[tag] = value
                    metadata["exif"] = exif
        except Exception as e:
            logger.warning(f"Ошибка при извлечении метаданных из изображения {file_path}: {e}")
            
        return metadata
    
    def _extract_docx_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Извлекает метаданные из DOCX файла.
        
        Args:
            file_path (Path): Путь к DOCX файлу.
            
        Returns:
            Dict[str, Any]: Словарь с метаданными DOCX.
        """
        metadata = {}
        try:
            doc = Document(file_path)
            # Извлекаем стандартные метаданные
            core_props = doc.core_properties
            doc_metadata = {
                "title": core_props.title,
                "author": core_props.author,
                "subject": core_props.subject,
                "creator": core_props.creator,
                "keywords": core_props.keywords,
                "description": core_props.description,
                "category": core_props.category,
                "created": core_props.created.isoformat() if core_props.created else None,
                "modified": core_props.modified.isoformat() if core_props.modified else None,
            }
            # Фильтруем None значения
            metadata.update({k: v for k, v in doc_metadata.items() if v is not None})
            
            # Добавляем количество страниц (приблизительно)
            paragraph_count = len(doc.paragraphs)
            metadata["approximate_paragraph_count"] = paragraph_count
        except Exception as e:
            logger.warning(f"Ошибка при извлечении метаданных из DOCX {file_path}: {e}")
            
        return metadata
    
    def _get_file_hash(self, file_path: Path) -> str:
        """
        Вычисляет хэш файла.
        
        Args:
            file_path (Path): Путь к файлу.
            
        Returns:
            str: Хэш файла.
        """
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Ошибка при вычислении хэша файла {file_path}: {e}")
            return ""
    
    def validate_and_normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Валидирует и нормализует метаданные.
        
        Args:
            metadata (Dict[str, Any]): Словарь с метаданными.
            
        Returns:
            Dict[str, Any]: Нормализованный словарь с метаданными.
        """
        normalized = {}
        
        for key, value in metadata.items():
            # Приводим ключи к нижнему регистру и заменяем пробелы на подчеркивания
            normalized_key = key.lower().replace(" ", "_")
            
            # Обрабатываем значения в зависимости от типа
            if isinstance(value, datetime):
                normalized[normalized_key] = value.isoformat()
            elif isinstance(value, (str, int, float, bool)) or value is None:
                normalized[normalized_key] = value
            else:
                # Для сложных объектов преобразуем в строку
                normalized[normalized_key] = str(value)
                
        return normalized
    
    def register_custom_schema(self, schema_name: str, schema: Dict[str, Any]):
        """
        Регистрирует пользовательскую схему метаданных.
        
        Args:
            schema_name (str): Имя схемы.
            schema (Dict[str, Any]): Схема метаданных.
        """
        self.custom_schemas[schema_name] = schema
        logger.info(f"Зарегистрирована пользовательская схема метаданных: {schema_name}")
    
    def get_custom_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """
        Получает пользовательскую схему метаданных.
        
        Args:
            schema_name (str): Имя схемы.
            
        Returns:
            Optional[Dict[str, Any]]: Схема метаданных или None.
        """
        return self.custom_schemas.get(schema_name)

# Создаем глобальный экземпляр MetadataManager
metadata_manager = MetadataManager()