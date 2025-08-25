"""Модуль для управления индексацией документов в Qdrant."""

import logging
from pathlib import Path
from typing import Tuple, List
import asyncio

from langchain_core.documents import Document

from config.config_manager import ConfigManager
from config.settings import Config
from core.metadata_manager import metadata_manager
from core.document_loader import DocumentLoader, IndexingError
from core.text_splitter import TextSplitter
from core.indexer_component import Indexer

logger = logging.getLogger(__name__)

# Get singleton instance of ConfigManager
config_manager = ConfigManager.get_instance()


async def run_indexing_logic(client=None) -> Tuple[bool, str]:
    """
    Основная логика индексации документов.
    
    Args:
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, создается новый.
        
    Returns:
        Tuple[bool, str]: (успех, статус)
    """
    try:
        config: Config = config_manager.get()
        
        # Проверка режимов индексации
        index_dense = getattr(config, 'index_dense', True)
        index_bm25 = getattr(config, 'index_bm25', False)
        index_hybrid = getattr(config, 'index_hybrid', False)

        if not (index_dense or index_bm25 or index_hybrid):
            return False, "no_index_type"
        
        folder_path = Path(config.folder_path)
        folder_path_resolved = folder_path.resolve()  # Для получения относительных путей
        if not folder_path.is_dir():
            return False, "folder_not_found"
        
        # Создаем компоненты
        document_loader = DocumentLoader()
        text_splitter = TextSplitter(config)
        indexer = Indexer(config)
        
        # Собираем все документы
        all_documents = []
        
        # Обрабатываем .txt файлы
        for filepath in folder_path.rglob("*.txt"):
            try:
                loaded_docs = document_loader.load_text_file(filepath)
                chunks = text_splitter.split_documents(loaded_docs)
                processed_chunks = _process_chunks(chunks, filepath, folder_path_resolved, config)
                all_documents.extend(processed_chunks)
            except Exception as e:
                logger.exception(f"Ошибка при обработке файла {filepath}: {e}")
                raise IndexingError(f"Ошибка при обработке файла {filepath}: {e}")
        
        # Обрабатываем .md файлы
        for filepath in folder_path.rglob("*.md"):
            try:
                loaded_docs = document_loader.load_text_file(filepath)
                chunks = text_splitter.split_documents(loaded_docs)
                processed_chunks = _process_chunks(chunks, filepath, folder_path_resolved, config)
                all_documents.extend(processed_chunks)
            except Exception as e:
                logger.exception(f"Ошибка при обработке файла {filepath}: {e}")
                raise IndexingError(f"Ошибка при обработке файла {filepath}: {e}")
        
        # Индексируем все документы
        success, status = await indexer.index_documents(all_documents, client, folder_path_resolved)
        
        # Обновляем конфигурацию
        if success:
            config.is_indexed = True
            config_manager.save(config)
        
        return success, status
        
    except IndexingError as e:
        logger.exception(f"Ошибка индексации: {e}")
        return False, f"indexing_error: {str(e)}"
    except Exception as e:
        logger.exception(f"Неожиданная ошибка при индексации: {e}")
        return False, f"indexing_error: {str(e)}"


def _process_chunks(chunks: List[Document], filepath: Path, folder_path_resolved: Path, config: Config) -> List[Document]:
    """
    Обработка чанков: добавление метаданных и вычисление относительных путей.
    
    Args:
        chunks (List[Document]): Список чанков.
        filepath (Path): Путь к файлу.
        folder_path_resolved (Path): Путь к корневой папке для вычисления относительных путей.
        config (Config): Конфигурация приложения.
        
    Returns:
        List[Document]: Список обработанных чанков.
    """
    try:
        # Получение относительного пути файла от корневой папки
        abs_filepath = filepath.resolve()
        relative_source_path = abs_filepath.relative_to(folder_path_resolved)
    except ValueError:
        # Если файл не в корневой папке, используем только имя файла
        logger.warning(f"Файл {filepath} не находится внутри {folder_path_resolved}. Используется только имя файла.")
        relative_source_path = abs_filepath.name
    
    # Добавляем метаданные к чанкам с помощью MetadataManager
    # Учитываем настройки из конфигурации
    if config.enable_metadata_extraction:
        for chunk in chunks:
            # Добавляем пользовательские поля из конфигурации
            chunk = metadata_manager.add_metadata_to_chunk(chunk, filepath, config.metadata_custom_fields)
            # Обновляем source в метаданных для совместимости
            chunk.metadata["source"] = str(relative_source_path)
    else:
        # Если извлечение метаданных отключено, добавляем только базовые метаданные
        for chunk in chunks:
            chunk.metadata["source"] = str(relative_source_path)
    
    return chunks


def run_indexing_from_config():
    """
    Запускает индексацию, используя настройки из config.json.
    
    Returns:
        Tuple[bool, str]: (успех, статус)
    """
    try:
        # Вызов асинхронной функции индексации
        result = asyncio.run(run_indexing_logic())
        return result
    except Exception as e:
        logger.exception(f"Ошибка при индексации: {e}")
        return False, f"indexing_error: {str(e)}"