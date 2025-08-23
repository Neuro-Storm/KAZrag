"""Модуль для нарезки текста на чанки."""

import logging
from functools import lru_cache
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import load_config, Config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)  # Ограничиваем размер кэша до 32 элементов
def get_text_splitter(chunk_size: int = None, chunk_overlap: int = None):
    """Создает и возвращает экземпляр RecursiveCharacterTextSplitter на основе конфигурации."""
    config: Config = load_config()
    
    # Используем переданные параметры, если они есть, иначе параметры из конфигурации
    size = chunk_size if chunk_size is not None else config.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
    
    logger.debug(f"Создание text splitter с chunk_size={size}, chunk_overlap={overlap}")
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)