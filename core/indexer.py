"""Модуль для управления индексацией документов в Qdrant."""

import time
import traceback
from pathlib import Path
from typing import List, Tuple, Any

from langchain_community.document_loaders import TextLoader
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from config.settings import load_config, save_config
from core.embeddings import get_dense_embedder, get_device
from core.chunker import get_text_splitter


def get_qdrant_client() -> QdrantClient:
    """Создает и возвращает клиент Qdrant."""
    config = load_config()
    return QdrantClient(url=config["qdrant_url"])


def run_indexing_logic() -> Tuple[bool, str]:
    """
    Основная логика индексации документов.
    
    Returns:
        Tuple[bool, str]: (успех, статус)
    """
    config = load_config()
    if not config.get("use_dense_vectors", False):
        return False, "no_index_type"
    
    folder_path = Path(config["folder_path"])
    folder_path_resolved = folder_path.resolve() # Для получения относительных путей
    if not folder_path.is_dir():
        return False, "folder_not_found"
    
    text_splitter = get_text_splitter()
    
    # Загрузить модель эмбеддинга один раз
    device = get_device(config['device'])
    dense_embedder = get_dense_embedder(config, device)
    
    # Параметры для пакетной обработки
    batch_size = config.get("indexing_batch_size", 50)  # Получаем из конфигурации, по умолчанию 50

    docs_batch = []
    collection_created = False
    docs_processed = 0
    
    def process_file(filepath, file_extension):
        """Общая функция для обработки .txt и .md файлов."""
        nonlocal docs_batch, collection_created, docs_processed
        try:
            loaded_docs = TextLoader(str(filepath), encoding="utf-8").load()
            chunks = text_splitter.split_documents(loaded_docs)
            try:
                # Получение относительного пути файла от корневой папки
                abs_filepath = filepath.resolve()
                relative_source_path = abs_filepath.relative_to(folder_path_resolved)
            except ValueError:
                # Если файл не в корневой папке, используем полный путь
                print(f"Предупреждение: Файл {filepath} не находится внутри {folder_path_resolved}. Используется полный путь.")
                relative_source_path = abs_filepath
            
            # Добавляем метаданные к чанкам
            for chunk in chunks:
                chunk.metadata["source"] = str(relative_source_path)
            
            docs_batch.extend(chunks)
            docs_processed += len(chunks)
            
            # Если набралась полная партия, обрабатываем её
            if len(docs_batch) >= batch_size:
                if not collection_created:
                    # Создаем новую коллекцию для первой партии
                    QdrantVectorStore.from_documents(
                        documents=docs_batch,
                        url=config["qdrant_url"],
                        collection_name=config["collection_name"],
                        embedding=dense_embedder,
                        force_recreate=True,
                        vector_name="dense_vector",
                        # Используем batch_size из конфига для Qdrant
                        batch_size=config.get("indexing_batch_size", 50),
                    )
                    collection_created = True
                else:
                    # Добавляем к существующей коллекции
                    client = get_qdrant_client()
                    qdrant_store = QdrantVectorStore(
                        client=client,
                        collection_name=config["collection_name"],
                        embedding=dense_embedder,
                        vector_name="dense_vector",
                    )
                    qdrant_store.add_documents(docs_batch)
                
                # Очищаем партию
                docs_batch = []
        except Exception as e:
            print(f"Ошибка при обработке файла {filepath}: {e}")
            # traceback.print_exc() # Опционально, для отладки
    
    # Обрабатываем .txt файлы
    for filepath in folder_path.rglob("*.txt"):
        process_file(filepath, ".txt")
    
    # Обрабатываем .md файлы
    for filepath in folder_path.rglob("*.md"):
        process_file(filepath, ".md")
    
    # Обрабатываем оставшиеся документы в последней партии
    if docs_batch:
        try:
            if not collection_created:
                # Создаем новую коллекцию для последней партии
                QdrantVectorStore.from_documents(
                    documents=docs_batch,
                    url=config["qdrant_url"],
                    collection_name=config["collection_name"],
                    embedding=dense_embedder,
                    force_recreate=True,
                    vector_name="dense_vector",
                    batch_size=config.get("indexing_batch_size", 50),
                )
            else:
                # Добавляем к существующей коллекции
                client = get_qdrant_client()
                qdrant_store = QdrantVectorStore(
                    client=client,
                    collection_name=config["collection_name"],
                    embedding=dense_embedder,
                    vector_name="dense_vector",
                )
                qdrant_store.add_documents(docs_batch)
        except Exception as e:
            print(f"Ошибка при обработке последней партии документов: {e}")
            # traceback.print_exc() # Опционально, для отладки
    
    # Проверка наличия документов для индексации
    # Исправлено: теперь возвращает успех даже если документов нет, но операция завершена
    if docs_processed == 0:
        config["is_indexed"] = False
        save_config(config)
        return True, "indexed_successfully_no_docs"
    
    config["is_indexed"] = True
    save_config(config)
    return True, "indexed_successfully"