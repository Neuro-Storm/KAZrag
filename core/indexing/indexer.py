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
from core.indexing.indexing_tracker import IndexingTracker
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
    tracker = IndexingTracker()

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

            # Начинаем новую сессию отслеживания
            tracker.start_new_session(config.collection_name)
            tracker.reset_stop_flag()

            # Получаем все файлы
            all_files = []
            for ext in ['.txt', '.md']:
                all_files.extend(folder_path.rglob(f"*{ext}"))
                all_files.extend(folder_path.rglob(f"*{ext.upper()}"))

            # Добавляем все файлы в текущую сессию
            files_to_process = []
            already_indexed_files = []

            for file_path in all_files:
                if file_path.is_file():
                    # Проверяем, был ли файл уже проиндексирован в предыдущих сессиях С УЧЕТОМ КОЛЛЕКЦИИ
                    status = tracker.get_file_status_from_any_session(file_path, config.collection_name)
                    if status == "indexed":
                        # Файл уже был проиндексирован для этой коллекции, добавляем как indexed в отдельный список
                        already_indexed_files.append(file_path)
                        tracker.update_file(file_path, "indexed")
                    else:
                        # Файл не был проиндексирован для этой коллекции, добавляем как pending и в список для обработки
                        tracker.update_file(file_path, "pending")
                        files_to_process.append(file_path)

            # Устанавливаем total только для файлов, которые нужно обработать
            tracker.set_total_files(len(files_to_process))

            # Добавляем уже проиндексированные файлы в статистику (но не в total)
            for file_path in already_indexed_files:
                tracker.update_file(file_path, "indexed", skip_counters=True)

            if len(files_to_process) == 0:
                tracker.end_session("completed_no_files")
                return True, "indexed_successfully_no_docs"

            # Проверяем, нужно ли использовать многоуровневый чанкинг
            use_multilevel_chunking = getattr(config, 'use_multilevel_chunking', False)

            if use_multilevel_chunking:
                # Используем многоуровневый индексатор
                logger.info("Используется многоуровневый индексатор")
                multilevel_indexer = MultiLevelIndexer(config)

                # Собираем все документы с отслеживанием
                all_documents = []
                document_loader = DocumentLoader()

                for filepath in files_to_process:
                    # Проверяем, не остановлена ли индексация
                    if tracker.is_indexing_stopped():
                        logger.info("Индексация остановлена пользователем")
                        tracker.end_session("stopped")
                        return False, "indexing_stopped"

                    try:
                        # Этап загрузки файла (многоуровневый)
                        logger.info(f"Начало обработки файла (многоуровневый): {filepath} - этап загрузки")
                        tracker.update_file(filepath, "loading")

                        # Дополнительная проверка остановки перед загрузкой
                        if tracker.is_indexing_stopped():
                            logger.info("Индексация остановлена перед загрузкой файла")
                            tracker.end_session("stopped")
                            return False, "indexing_stopped"

                        loaded_docs = document_loader.load_text_file(filepath)
                        logger.info(f"Файл {filepath} успешно загружен, документов: {len(loaded_docs)}")

                        # Небольшая задержка для наглядности прогресса
                        import time
                        time.sleep(0.3)

                        # Проверка остановки после загрузки
                        if tracker.is_indexing_stopped():
                            logger.info("Индексация остановлена после загрузки файла")
                            tracker.end_session("stopped")
                            return False, "indexing_stopped"

                        # Этап обработки документов (многоуровневый)
                        logger.info(f"Файл {filepath} - этап обработки для многоуровневой индексации")
                        tracker.update_file(filepath, "indexing")

                        # Для многоуровневого чанкинга не нужно предварительно разбивать на чанки
                        processed_docs = _process_documents_without_chunking(loaded_docs, filepath, folder_path_resolved, config)
                        all_documents.extend(processed_docs)
                        tracker.update_file(filepath, "indexed")
                        logger.info(f"Файл {filepath} успешно обработан для многоуровневой индексации")
                    except Exception as e:
                        logger.exception(f"Ошибка при обработке файла {filepath}: {e}")
                        tracker.update_file(filepath, "error", str(e))
                        # Не прерываем индексацию из-за одной ошибки файла
                        continue

                logger.info(f"Всего собрано документов: {len(all_documents)}")

                if len(all_documents) == 0:
                    tracker.end_session("completed_no_docs")
                    return True, "indexed_successfully_no_docs"

                # Финальная проверка остановки перед многоуровневой индексацией
                if tracker.is_indexing_stopped():
                    logger.info("Многоуровневая индексация остановлена перед индексацией")
                    tracker.end_session("stopped")
                    return False, "indexing_stopped"

                # Индексируем все документы с использованием многоуровневого индексатора
                try:
                    # Обновляем статусы файлов на chunked перед многоуровневой индексацией
                    for filepath in files_to_process:
                        tracker.update_file(filepath, "chunked")

                    # Показываем, что началась многоуровневая индексация
                    for filepath in files_to_process:
                        tracker.update_file(filepath, "multilevel_indexing")

                    stats = multilevel_indexer.index_documents_multilevel(all_documents)
                    logger.info(f"Многоуровневая индексация завершена: {stats}")

                    # Обновляем статусы файлов на indexed после завершения
                    for filepath in files_to_process:
                        tracker.update_file(filepath, "indexed")

                    success = True
                    status = "indexed_successfully"
                    tracker.end_session("completed")
                except Exception as e:
                    if "indexing_stopped" in str(e):
                        logger.info("Многоуровневая индексация остановлена")
                        tracker.end_session("stopped")
                        return False, "indexing_stopped"
                    else:
                        logger.exception(f"Ошибка при многоуровневой индексации: {e}")
                        tracker.end_session("error")
                        return False, f"indexing_error: {str(e)}"
            else:
                # Используем обычный индексатор
                logger.info("Используется обычный индексатор")

                # Создаем компоненты
                document_loader = DocumentLoader()
                text_splitter = TextSplitter(config)
                indexer = Indexer(config)

                # Собираем все документы с отслеживанием
                all_documents = []

                for filepath in files_to_process:
                    # Проверяем, не остановлена ли индексация
                    if tracker.is_indexing_stopped():
                        logger.info("Индексация остановлена пользователем")
                        tracker.end_session("stopped")
                        return False, "indexing_stopped"

                    try:
                        # Этап загрузки файла
                        logger.info(f"Начало обработки файла: {filepath} - этап загрузки")
                        tracker.update_file(filepath, "loading")

                        # Дополнительная проверка остановки перед загрузкой
                        if tracker.is_indexing_stopped():
                            logger.info("Индексация остановлена перед загрузкой файла")
                            tracker.end_session("stopped")
                            return False, "indexing_stopped"

                        loaded_docs = document_loader.load_text_file(filepath)
                        logger.info(f"Файл {filepath} успешно загружен, документов: {len(loaded_docs)}")

                        # Небольшая задержка для наглядности прогресса
                        import time
                        time.sleep(0.3)

                        # Проверка остановки после загрузки
                        if tracker.is_indexing_stopped():
                            logger.info("Индексация остановлена после загрузки файла")
                            tracker.end_session("stopped")
                            return False, "indexing_stopped"

                        # Этап разделения на чанки
                        logger.info(f"Файл {filepath} - этап разделения на чанки")
                        tracker.update_file(filepath, "chunking")

                        chunks = text_splitter.split_documents(loaded_docs)
                        logger.info(f"Файл {filepath} разделен на {len(chunks)} чанков")

                        # Небольшая задержка для наглядности прогресса
                        time.sleep(0.3)

                        # Проверка остановки после разделения на чанки
                        if tracker.is_indexing_stopped():
                            logger.info("Индексация остановлена после разделения на чанки")
                            tracker.end_session("stopped")
                            return False, "indexing_stopped"

                        # Этап обработки чанков (добавление метаданных)
                        logger.info(f"Файл {filepath} - этап обработки чанков")
                        tracker.update_file(filepath, "indexing")

                        # Небольшая задержка для наглядности прогресса
                        import time
                        time.sleep(0.5)

                        processed_chunks = _process_chunks(chunks, filepath, folder_path_resolved, config)
                        all_documents.extend(processed_chunks)
                        tracker.update_file(filepath, "chunked")
                        logger.info(f"Файл {filepath} разбит на чанки и готов к индексации")
                    except Exception as e:
                        logger.exception(f"Ошибка при обработке файла {filepath}: {e}")
                        tracker.update_file(filepath, "error", str(e))
                        # Не прерываем индексацию из-за одной ошибки файла
                        continue

                # Финальная проверка остановки перед индексацией в Qdrant
                if tracker.is_indexing_stopped():
                    logger.info("Индексация остановлена перед добавлением в Qdrant")
                    tracker.end_session("stopped")
                    return False, "indexing_stopped"

                # Индексируем все документы
                try:
                    # Передаем информацию о файлах для отслеживания
                    file_paths = [str(f) for f in files_to_process]
                    success, status = await indexer.index_documents(all_documents, client, folder_path_resolved, file_paths)

                    if success:
                        tracker.end_session("completed")
                    else:
                        if status == "indexing_stopped":
                            tracker.end_session("stopped")
                        else:
                            tracker.end_session("error")
                except Exception as e:
                    if "indexing_stopped" in str(e):
                        logger.info("Индексация остановлена во время выполнения")
                        tracker.end_session("stopped")
                        success, status = False, "indexing_stopped"
                    else:
                        logger.exception(f"Ошибка индексации: {e}")
                        tracker.end_session("error")
                        success, status = False, f"indexing_error: {str(e)}"
        
        # Обновляем конфигурацию (только если не используются предварительно нарезанные документы)
        if success and pre_chunked_documents is None:
            # Не обновляем config.is_indexed если поле существует (оно могло быть удалено)
            if hasattr(config, 'is_indexed'):
                config.is_indexed = True
            config_manager.save(config)
        
        return success, status

    except Exception as e:
        logger.exception(f"Неожиданная ошибка при индексации: {e}")
        try:
            tracker.end_session("error")
        except:
            pass  # Игнорируем ошибки при завершении сессии
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
        folder_path_resolved = folder_path_resolved.resolve()  # Ensure both paths are resolved for Windows compatibility
        relative_source_path = abs_filepath.relative_to(folder_path_resolved)
    except ValueError:
        # Если файл не в корневой папке, используем только имя файла
        logger.warning(f"Файл {filepath} не находится внутри {folder_path_resolved}. Используется только имя файла.")
        relative_source_path = Path(abs_filepath.name)
    
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
        folder_path_resolved = folder_path_resolved.resolve()  # Ensure both paths are resolved for Windows compatibility
        relative_source_path = abs_filepath.relative_to(folder_path_resolved)
    except ValueError:
        # Если файл не в корневой папке, используем только имя файла
        logger.warning(f"Файл {filepath} не находится внутри {folder_path_resolved}. Используется только имя файла.")
        relative_source_path = Path(abs_filepath.name)
    
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


def run_indexing_logic_sync(client=None, pre_chunked_documents=None):
    """
    Синхронная обертка для запуска асинхронной логики индексации.
    Используется для запуска в фоновых задачах FastAPI.
    
    Args:
        client: Клиент Qdrant (опционально)
        pre_chunked_documents: Предварительно нарезанные документы (опционально)
    
    Returns:
        Результат выполнения индексации
    """
    import sys
    import asyncio
    
    # Проверяем, есть ли уже запущенный event loop
    try:
        loop = asyncio.get_running_loop()
        # Если уже есть запущенный loop, создаем задачу
        import concurrent.futures
        import threading
        
        def run_in_new_loop():
            return asyncio.run(run_indexing_logic(client=client, pre_chunked_documents=pre_chunked_documents))
        
        # Запускаем в отдельном потоке с новым event loop
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            try:
                return future.result()
            except Exception as e:
                logger.exception(f"Ошибка при выполнении индексации в фоновом потоке: {e}")
                return False, f"indexing_error: {str(e)}"
                
    except RuntimeError:
        # Нет запущенного loop, можем использовать asyncio.run
        try:
            return asyncio.run(run_indexing_logic(client=client, pre_chunked_documents=pre_chunked_documents))
        except Exception as e:
            logger.exception(f"Ошибка при выполнении индексации: {e}")
            return False, f"indexing_error: {str(e)}"