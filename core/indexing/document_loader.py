"""Модуль для загрузки документов."""

import logging
from pathlib import Path
from typing import List, Tuple
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Класс для загрузки документов из файлов."""

    def __init__(self):
        """Инициализация загрузчика документов."""
        pass

    def load_text_file(self, filepath: Path) -> List[Document]:
        """
        Загрузка текстового файла (.txt или .md).
        
        Args:
            filepath (Path): Путь к файлу.
            
        Returns:
            List[Document]: Список загруженных документов.
        """
        try:
            loader = TextLoader(str(filepath), encoding="utf-8")
            return loader.load()
        except Exception as e:
            logger.exception(f"Ошибка при загрузке файла {filepath}: {e}")
            raise


class IndexingError(Exception):
    """Пользовательское исключение для ошибок индексации."""
    pass