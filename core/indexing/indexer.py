"""Модуль для управления индексацией документов в Qdrant."""

import asyncio
import logging
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

from config.config_manager import ConfigManager
from config.settings import Config
from core.indexing.document_loader import DocumentLoader, IndexingError
from core.indexing.indexer_component import Indexer
from core.indexing.metadata_manager import metadata_manager
from core.indexing.multilevel_indexer import MultiLevelIndexer
from core.indexing.text_splitter import TextSplitter

logger = logging.getLogger(__name__)

# Get singleton instance of ConfigManager
config_manager = ConfigManager.get_instance()


async def run_indexing_logic(client=None, pre_chunked_documents=None) -> Tuple[bool, str]:
    """
    Основная логика индексации документов.
    
    Args:
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, создается новый.
        pre_chunked_documents (List[Document], optional): Предварительно нарезанные документы. 
            Если указаны, используются вместо загрузки и нарезки файлов.
        
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
        
        # Если предоставлены предварительно нарезанные документы, используем их
        if pre_chunked_documents is not None:
            logger.info(f"Используются предварительно нарезанные документы: {len(pre_chunked_documents)} чанков")
            all_documents = pre_chunked_documents
            folder_path_resolved = Path(config.folder_path).resolve()
            
            # Создаем индексер и индексируем документы
            indexer = Indexer(config)
            success, status = await indexer.index_documents(all_documents, client, folder_path_resolved)
        else:
            folder_path = Path(config.folder_path)
            folder_path_resolved = folder_path.resolve()  # Для получения относительных путей
            if not folder_path.is_dir():
                return False, "folder_not_found"
            
            # Проверяем, нужно ли использовать многоуровневый чанкинг
            use_multilevel_chunking = getattr(config, 'use_multilevel_chunking', False)
            
            if use_multilevel_chunking:
                # Используем многоуровневый индексатор
                logger.info("Используется многоуровневый индексатор")
                multilevel_indexer = MultiLevelIndexer(config)
                
                # Собираем все документы
                all_documents = []
                document_loader = DocumentLoader()
                
                # Обрабатываем .txt файлы
                for filepath in folder_path.rglob("*.txt"):
                    try:
                        loaded_docs = document_loader.load_text_file(filepath)
                        # Для многоуровневого чанкинга не нужно предварительно разбивать на чанки
                        processed_docs = _process_documents_without_chunking(loaded_docs, filepath, folder_path_resolved, config)
                        all_documents.extend(processed_docs)
                    except Exception as e:
                        logger.exception(f"Ошибка при обработке файла {filepath}: {e}")
                        raise IndexingError(f"Ошибка при обработке файла {filepath}: {e}")
                
                # Обрабатываем .md файлы
                for filepath in folder_path.rglob("*.md"):
                    try:
                        logger.info(f"Загрузка .md файла: {filepath}")
                        loaded_docs = document_loader.load_text_file(filepath)
                        logger.info(f"Загружено {len(loaded_docs)} документов из файла: {filepath}")
                        # Для многоуровневого чанкинга не нужно предварительно разбивать на чанки
                        processed_docs = _process_documents_without_chunking(loaded_docs, filepath, folder_path_resolved, config)
                        all_documents.extend(processed_docs)
                        logger.info(f"После обработки файла {filepath} всего документов: {len(all_documents)}")
                    except Exception as e:
                        logger.exception(f"Ошибка при обработке файла {filepath}: {e}")
                        raise IndexingError(f"Ошибка при обработке файла {filepath}: {e}")
                
                logger.info(f"Всего собрано документов: {len(all_documents)}")
                
                # Индексируем все документы с использованием многоуровневого индексатора
                # Временно используем синхронный вызов, так как multilevel_indexer.index_documents_multilevel не async
                stats = multilevel_indexer.index_documents_multilevel(all_documents)
                logger.info(f"Индексация завершена: {stats}")
                
                success = True
                status = "indexed_successfully"
            else:
                # Используем обычный индексатор
                logger.info("Используется обычный индексатор")
                
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
        
        # Обновляем конфигурацию (только если не используются предварительно нарезанные документы)
        if success and pre_chunked_documents is None:
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


def _process_documents_without_chunking(documents: List[Document], filepath: Path, folder_path_resolved: Path, config: Config) -> List[Document]:
    """
    Обработка документов без предварительного разбиения на чанки (для многоуровневого чанкинга).
    
    Args:
        documents (List[Document]): Список документов.
        filepath (Path): Путь к файлу.
        folder_path_resolved (Path): Путь к корневой папке для вычисления относительных путей.
        config (Config): Конфигурация приложения.
        
    Returns:
        List[Document]: Список обработанных документов.
    """
    try:
        # Получение относительного пути файла от корневой папки
        abs_filepath = filepath.resolve()
        relative_source_path = abs_filepath.relative_to(folder_path_resolved)
    except ValueError:
        # Если файл не в корневой папке, используем только имя файла
        logger.warning(f"Файл {filepath} не находится внутри {folder_path_resolved}. Используется только имя файла.")
        relative_source_path = abs_filepath.name
    
    # Добавляем метаданные к документам с помощью MetadataManager
    # Учитываем настройки из конфигурации
    if config.enable_metadata_extraction:
        for doc in documents:
            # Добавляем пользовательские поля из конфигурации
            doc = metadata_manager.add_metadata_to_chunk(doc, filepath, config.metadata_custom_fields)
            # Обновляем source в метаданных для совместимости
            doc.metadata["source"] = str(relative_source_path)
    else:
        # Если извлечение метаданных отключено, добавляем только базовые метаданные
        for doc in documents:
            doc.metadata["source"] = str(relative_source_path)
    
    return documents


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