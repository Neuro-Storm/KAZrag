"""Модуль для разделения текста на чанки."""

import logging
from typing import List
from langchain_core.documents import Document
from core.chunker import get_text_splitter
from config.settings import Config

logger = logging.getLogger(__name__)


class TextSplitter:
    """Класс для разделения текста на чанки."""

    def __init__(self, config: Config):
        """
        Инициализация разделителя текста.
        
        Args:
            config (Config): Конфигурация приложения.
        """
        self.text_splitter = get_text_splitter(config.chunk_size, config.chunk_overlap)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Разделение документов на чанки.
        
        Args:
            documents (List[Document]): Список документов для разделения.
            
        Returns:
            List[Document]: Список чанков.
        """
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.exception(f"Ошибка при разделении документов на чанки: {e}")
            raise


class IndexingError(Exception):
    """Пользовательское исключение для ошибок индексации."""
    pass