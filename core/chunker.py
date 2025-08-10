"""Модуль для нарезки текста на чанки."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import load_config


def get_text_splitter():
    """Создает и возвращает экземпляр RecursiveCharacterTextSplitter на основе конфигурации."""
    config = load_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"], 
        chunk_overlap=config["chunk_overlap"]
    )