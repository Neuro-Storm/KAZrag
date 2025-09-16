"""Модуль для нарезки текста на чанки."""

import logging
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

from config.config_manager import ConfigManager
from config.settings import Config
from core.indexing.paragraph_chunker import ParagraphTextSplitter
from core.indexing.sentence_chunker import SentenceTextSplitter

logger = logging.getLogger(__name__)


def create_text_splitter(config: Config) -> TextSplitter:
    """
    Создает и возвращает экземпляр TextSplitter на основе конфигурации.
    
    Args:
        config (Config): Конфигурация приложения
        
    Returns:
        TextSplitter: Экземпляр подходящего разделителя текста
    """
    if config.chunking_strategy == "paragraph":
        logger.debug(
            f"Создание paragraph text splitter с paragraphs_per_chunk={config.paragraphs_per_chunk}, "
            f"paragraph_overlap={config.paragraph_overlap}"
        )
        return ParagraphTextSplitter(
            paragraphs_per_chunk=config.paragraphs_per_chunk,
            paragraph_overlap=config.paragraph_overlap
        )
    elif config.chunking_strategy == "sentence":
        logger.debug(
            f"Создание sentence text splitter с sentences_per_chunk={config.sentences_per_chunk}, "
            f"sentence_overlap={config.sentence_overlap}"
        )
        return SentenceTextSplitter(
            sentences_per_chunk=config.sentences_per_chunk,
            sentence_overlap=config.sentence_overlap
        )
    else:  # По умолчанию используем стратегию по символам
        logger.debug(
            f"Создание character text splitter с chunk_size={config.chunk_size}, "
            f"chunk_overlap={config.chunk_overlap}"
        )
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )


@lru_cache(maxsize=32)  # Ограничиваем размер кэша до 32 элементов
def get_text_splitter(
    chunk_size: int = None, 
    chunk_overlap: int = None,
    chunking_strategy: str = None,
    paragraphs_per_chunk: int = None,
    paragraph_overlap: int = None,
    sentences_per_chunk: int = None,
    sentence_overlap: int = None
) -> TextSplitter:
    """
    Создает и возвращает экземпляр TextSplitter на основе конфигурации или переданных параметров.
    
    Args:
        chunk_size (int, optional): Размер чанка в символах
        chunk_overlap (int, optional): Перекрытие чанков в символах
        chunking_strategy (str, optional): Стратегия чанкинга ("character", "paragraph" или "sentence")
        paragraphs_per_chunk (int, optional): Количество абзацев в одном чанке
        paragraph_overlap (int, optional): Количество абзацев перекрытия между чанками
        sentences_per_chunk (int, optional): Количество предложений в одном чанке
        sentence_overlap (int, optional): Количество предложений перекрытия между чанками
        
    Returns:
        TextSplitter: Экземпляр подходящего разделителя текста
    """
    config_manager = ConfigManager.get_instance()
    config: Config = config_manager.get()
    
    # Используем переданные параметры, если они есть, иначе параметры из конфигурации
    strategy = chunking_strategy if chunking_strategy is not None else config.chunking_strategy
    size = chunk_size if chunk_size is not None else config.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
    para_per_chunk = paragraphs_per_chunk if paragraphs_per_chunk is not None else config.paragraphs_per_chunk
    para_overlap = paragraph_overlap if paragraph_overlap is not None else config.paragraph_overlap
    sent_per_chunk = sentences_per_chunk if sentences_per_chunk is not None else config.sentences_per_chunk
    sent_overlap = sentence_overlap if sentence_overlap is not None else config.sentence_overlap
    
    if strategy == "paragraph":
        logger.debug(
            f"Создание paragraph text splitter с paragraphs_per_chunk={para_per_chunk}, "
            f"paragraph_overlap={para_overlap}"
        )
        return ParagraphTextSplitter(
            paragraphs_per_chunk=para_per_chunk,
            paragraph_overlap=para_overlap
        )
    elif strategy == "sentence":
        logger.debug(
            f"Создание sentence text splitter с sentences_per_chunk={sent_per_chunk}, "
            f"sentence_overlap={sent_overlap}"
        )
        return SentenceTextSplitter(
            sentences_per_chunk=sent_per_chunk,
            sentence_overlap=sent_overlap
        )
    else:  # По умолчанию используем стратегию по символам
        logger.debug(
            f"Создание character text splitter с chunk_size={size}, "
            f"chunk_overlap={overlap}"
        )
        return RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap
        )