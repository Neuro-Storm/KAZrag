"""Модуль для нарезки текста на чанки."""

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import load_config, Config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_text_splitter():
    """Создает и возвращает экземпляр RecursiveCharacterTextSplitter на основе конфигурации."""
    config: Config = load_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, 
        chunk_overlap=config.chunk_overlap
    )