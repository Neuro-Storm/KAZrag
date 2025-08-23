"""Модуль для управления индексацией документов в Qdrant."""

import logging
from pathlib import Path
from typing import Tuple
import psutil

from langchain_community.document_loaders import TextLoader
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from core.qdrant_client import get_qdrant_client, aget_qdrant_client

from config.settings import load_config, save_config, Config
from core.embeddings import get_dense_embedder, get_device
from core.chunker import get_text_splitter

logger = logging.getLogger(__name__)


# Используем централизованные функции get_qdrant_client / aget_qdrant_client


async def run_indexing_logic(client = None) -> Tuple[bool, str]:
    """
    Основная логика индексации документов.
    
    Args:
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, создается новый.
        
    Returns:
        Tuple[bool, str]: (успех, статус)
    """
    config: Config = load_config()
    # Determine indexing modes
    index_dense = getattr(config, 'index_dense', True)
    index_bm25 = getattr(config, 'index_bm25', False)
    index_hybrid = getattr(config, 'index_hybrid', False)

    if not (index_dense or index_bm25 or index_hybrid):
        return False, "no_index_type"
    
    # Проверка для гибридного режима
    if index_hybrid and not index_dense:
        logger.error("Hybrid mode requires index_dense=True")
        return False, "hybrid_requires_dense"
    
    folder_path = Path(config.folder_path)
    folder_path_resolved = folder_path.resolve() # Для получения относительных путей
    if not folder_path.is_dir():
        return False, "folder_not_found"
    
    text_splitter = get_text_splitter(config.chunk_size, config.chunk_overlap)
    
    # Загрузить модель эмбеддинга один раз
    device = get_device(config.device)
    dense_embedder = get_dense_embedder(config, device)
    
    # Параметры для пакетной обработки
    # Используем memory_threshold для определения размера батча
    mem = psutil.virtual_memory().available
    batch_size = config.indexing_batch_size if mem > config.memory_threshold else config.indexing_batch_size // 2

    docs_batch = []
    collection_created = False
    docs_processed = 0
    
    # Если клиент не передан, создаем новый (асинхронно чтобы не блокировать loop)
    if client is None:
        client = await aget_qdrant_client(config)
    
    # Проверяем наличие коллекции и удаляем её, если force_recreate=True или если включён гибридный поиск
    try:
        client.get_collection(config.collection_name)
        # Если коллекция существует и force_recreate=True или включён гибридный режим (новый флаг), удаляем её
        if config.force_recreate or index_hybrid:
            client.delete_collection(config.collection_name)
    except Exception as e:
        # Коллекция не существует или произошла ошибка при проверке — логируем для отладки и продолжаем
        logger.debug(f"Проверка коллекции завершилась с ошибкой или коллекция отсутствует: {e}")
    
    # Подготовка sparse embedding для BM25/hybrid
    sparse_emb = None
    if index_bm25 or index_hybrid:
        try:
            from core.sparse_embedding_adapter import SparseEmbeddingAdapter
            sparse_emb = SparseEmbeddingAdapter(config.sparse_embedding)
            logger.info(f"Sparse embedding adapter initialized: model={config.sparse_embedding}, type={type(sparse_emb)}")
        except ImportError:
            logger.warning("fastembed недоступен: индексация BM25/hybrid будет пропущена либо выполнится только dense часть.")
            sparse_emb = None
        except Exception as e:
            logger.exception(f"Ошибка при инициализации sparse embedding adapter ({config.sparse_embedding}): {e}")
            sparse_emb = None
    
    async def process_file(filepath, file_extension, config: Config):
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
                # Если файл не в корневой папке, используем только имя файла
                logger.warning(f"Файл {filepath} не находится внутри {folder_path_resolved}. Используется только имя файла.")
                relative_source_path = abs_filepath.name
            
            # Добавляем метаданные к чанкам
            for chunk in chunks:
                chunk.metadata["source"] = str(relative_source_path)
            
            docs_batch.extend(chunks)
            docs_processed += len(chunks)
            
            # Если набралась полная партия, обрабатываем её
            if len(docs_batch) >= batch_size:
                if not collection_created:
                    # Создаем новую коллекцию для первой партии
                    if index_hybrid and sparse_emb:
                        logger.info(f"Creating hybrid Qdrant collection '{config.collection_name}' with dense and sparse embeddings")
                        QdrantVectorStore.from_documents(
                            documents=docs_batch,
                            url=config.qdrant_url,
                            collection_name=config.collection_name,
                            embedding=dense_embedder if index_dense else None,
                            force_recreate=True,
                            vector_name="dense_vector",
                            sparse_embedding=sparse_emb if index_bm25 or index_hybrid else None,
                            retrieval_mode=RetrievalMode.HYBRID,
                            batch_size=config.indexing_batch_size,
                        )
                    elif index_bm25 and sparse_emb and not index_dense:
                        logger.info(f"Creating sparse-only Qdrant collection '{config.collection_name}' using fastembed sparse embeddings")
                        QdrantVectorStore.from_documents(
                            documents=docs_batch,
                            url=config.qdrant_url,
                            collection_name=config.collection_name,
                            embedding=None,
                            force_recreate=True,
                            vector_name=None,
                            sparse_embedding=sparse_emb,
                            retrieval_mode=RetrievalMode.SPARSE,
                            batch_size=config.indexing_batch_size,
                        )
                    elif index_dense:
                        logger.info(f"Creating dense-only Qdrant collection '{config.collection_name}'")
                        QdrantVectorStore.from_documents(
                            documents=docs_batch,
                            url=config.qdrant_url,
                            collection_name=config.collection_name,
                            embedding=dense_embedder,
                            force_recreate=True,
                            vector_name="dense_vector",
                            batch_size=config.indexing_batch_size,
                        )
                    else:
                        # Нечего индексировать
                        logger.warning("Нет подходящего режима индексации для первой партии (пропускаем)")
                    collection_created = True
                else:
                    # Добавляем к существующей коллекции
                    client = get_qdrant_client(config)
                    logger.info(f"Adding documents to collection '{config.collection_name}' (hybrid={index_hybrid and bool(sparse_emb)})")
                    if index_hybrid and sparse_emb:
                        qdrant_store = QdrantVectorStore(
                            client=client,
                            collection_name=config.collection_name,
                            embedding=dense_embedder if index_dense else None,
                            vector_name="dense_vector",
                            sparse_embedding=sparse_emb,
                            retrieval_mode=RetrievalMode.HYBRID
                        )
                    elif index_bm25 and sparse_emb and not index_dense:
                        qdrant_store = QdrantVectorStore(
                            client=client,
                            collection_name=config.collection_name,
                            embedding=None,
                            vector_name=None,
                            sparse_embedding=sparse_emb,
                            retrieval_mode=RetrievalMode.SPARSE
                        )
                    else:
                        qdrant_store = QdrantVectorStore(
                            client=client,
                            collection_name=config.collection_name,
                            embedding=dense_embedder if index_dense else None,
                            vector_name="dense_vector"
                        )
                    try:
                        await qdrant_store.aadd_documents(docs_batch)
                    except Exception as e:
                        # Если Qdrant жалуется на отсутствие sparse векторов, пробуем пересоздать коллекцию с force_recreate=True
                        err_msg = str(e)
                        logger.exception(f"Ошибка при добавлении документов в коллекцию: {err_msg}")
                        if 'does not contain sparse vectors' in err_msg or 'does not contain sparse' in err_msg:
                            logger.info("Похоже, в коллекции отсутствуют sparse-векторы — пересоздаём коллекцию с force_recreate=True и повторяем добавление")
                            # Пересоздаём коллекцию используя from_documents с force_recreate
                            try:
                                if index_hybrid and sparse_emb:
                                    QdrantVectorStore.from_documents(
                                        documents=docs_batch,
                                        url=config.qdrant_url,
                                        collection_name=config.collection_name,
                                        embedding=dense_embedder if index_dense else None,
                                        force_recreate=True,
                                        vector_name="dense_vector",
                                        sparse_embedding=sparse_emb,
                                        retrieval_mode=RetrievalMode.HYBRID,
                                        batch_size=config.indexing_batch_size,
                                    )
                                elif index_bm25 and sparse_emb and not index_dense:
                                    QdrantVectorStore.from_documents(
                                        documents=docs_batch,
                                        url=config.qdrant_url,
                                        collection_name=config.collection_name,
                                        embedding=None,
                                        force_recreate=True,
                                        vector_name=None,
                                        sparse_embedding=sparse_emb,
                                        retrieval_mode=RetrievalMode.SPARSE,
                                        batch_size=config.indexing_batch_size,
                                    )
                                else:
                                    QdrantVectorStore.from_documents(
                                        documents=docs_batch,
                                        url=config.qdrant_url,
                                        collection_name=config.collection_name,
                                        embedding=dense_embedder if index_dense else None,
                                        force_recreate=True,
                                        vector_name="dense_vector",
                                        batch_size=config.indexing_batch_size,
                                    )
                                collection_created = True
                            except Exception as recreate_error:
                                logger.exception("Не удалось пересоздать коллекцию для добавления sparse-векторов")
                                raise recreate_error
                        else:
                            # Для других ошибок просто пробрасываем исключение
                            raise e
                
                # Очищаем партию
                docs_batch = []
        except Exception as e:
            logger.exception(f"Ошибка при обработке файла {filepath}: {e}")
            # traceback.print_exc() # Опционально, для отладки
    
    # Обрабатываем .txt файлы
    for filepath in folder_path.rglob("*.txt"):
        await process_file(filepath, ".txt", config)
    
    # Обрабатываем .md файлы
    for filepath in folder_path.rglob("*.md"):
        await process_file(filepath, ".md", config)
    
    # Обрабатываем оставшиеся документы в последней партии
    if docs_batch:
        try:
            if not collection_created:
                # Создаем новую коллекцию для последней партии (см. логику выше)
                if index_hybrid and sparse_emb:
                    QdrantVectorStore.from_documents(
                        documents=docs_batch,
                        url=config.qdrant_url,
                        collection_name=config.collection_name,
                        embedding=dense_embedder if index_dense else None,
                        force_recreate=True,
                        vector_name="dense_vector",
                        sparse_embedding=sparse_emb if index_bm25 or index_hybrid else None,
                        retrieval_mode=RetrievalMode.HYBRID,
                        batch_size=config.indexing_batch_size,
                    )
                elif index_bm25 and sparse_emb and not index_dense:
                    QdrantVectorStore.from_documents(
                        documents=docs_batch,
                        url=config.qdrant_url,
                        collection_name=config.collection_name,
                        embedding=None,
                        force_recreate=True,
                        vector_name=None,
                        sparse_embedding=sparse_emb,
                        retrieval_mode=RetrievalMode.SPARSE,
                        batch_size=config.indexing_batch_size,
                    )
                elif index_dense:
                    QdrantVectorStore.from_documents(
                        documents=docs_batch,
                        url=config.qdrant_url,
                        collection_name=config.collection_name,
                        embedding=dense_embedder,
                        force_recreate=True,
                        vector_name="dense_vector",
                        batch_size=config.indexing_batch_size,
                    )
                else:
                    logger.warning("Нет подходящего режима индексации для последней партии (пропускаем)")
            else:
                # Добавляем к существующей коллекции
                client = get_qdrant_client(config)
                if index_hybrid and sparse_emb:
                    qdrant_store = QdrantVectorStore(
                        client=client,
                        collection_name=config.collection_name,
                        embedding=dense_embedder if index_dense else None,
                        vector_name="dense_vector",
                        sparse_embedding=sparse_emb,
                        retrieval_mode=RetrievalMode.HYBRID
                    )
                elif index_bm25 and sparse_emb and not index_dense:
                    qdrant_store = QdrantVectorStore(
                        client=client,
                        collection_name=config.collection_name,
                        embedding=None,
                        vector_name=None,
                        sparse_embedding=sparse_emb,
                        retrieval_mode=RetrievalMode.SPARSE
                    )
                else:
                    qdrant_store = QdrantVectorStore(
                        client=client,
                        collection_name=config.collection_name,
                        embedding=dense_embedder if index_dense else None,
                        vector_name="dense_vector"
                    )
                try:
                    await qdrant_store.aadd_documents(docs_batch)
                except Exception as e:
                    err_msg = str(e)
                    logger.exception(f"Ошибка при добавлении последней партии документов: {err_msg}")
                    if 'does not contain sparse vectors' in err_msg or 'does not contain sparse' in err_msg:
                        logger.info("Обнаружено отсутствие sparse-векторов в коллекции: пересоздаём коллекцию с force_recreate=True и повторяем добавление для последней партии")
                        try:
                            if index_hybrid and sparse_emb:
                                QdrantVectorStore.from_documents(
                                    documents=docs_batch,
                                    url=config.qdrant_url,
                                    collection_name=config.collection_name,
                                    embedding=dense_embedder if index_dense else None,
                                    force_recreate=True,
                                    vector_name="dense_vector",
                                    sparse_embedding=sparse_emb,
                                    retrieval_mode=RetrievalMode.HYBRID,
                                    batch_size=config.indexing_batch_size,
                                )
                            elif index_bm25 and sparse_emb and not index_dense:
                                QdrantVectorStore.from_documents(
                                    documents=docs_batch,
                                    url=config.qdrant_url,
                                    collection_name=config.collection_name,
                                    embedding=None,
                                    force_recreate=True,
                                    vector_name=None,
                                    sparse_embedding=sparse_emb,
                                    retrieval_mode=RetrievalMode.SPARSE,
                                    batch_size=config.indexing_batch_size,
                                )
                            else:
                                QdrantVectorStore.from_documents(
                                    documents=docs_batch,
                                    url=config.qdrant_url,
                                    collection_name=config.collection_name,
                                    embedding=dense_embedder if index_dense else None,
                                    force_recreate=True,
                                    vector_name="dense_vector",
                                    batch_size=config.indexing_batch_size,
                                )
                            collection_created = True
                        except Exception as recreate_error:
                            logger.exception("Не удалось пересоздать коллекцию при добавлении последней партии")
                            raise recreate_error
                    else:
                        # Для других ошибок просто пробрасываем исключение
                        raise e
        except Exception as e:
            logger.exception(f"Ошибка при обработке последней партии документов: {e}")
    
    # Проверка наличия документов для индексации
    # Исправлено: теперь возвращает успех даже если документов нет, но операция завершена
    if docs_processed == 0:
        config.is_indexed = True  # Устанавливаем True, так как операция завершена успешно
        save_config(config)
        return True, "indexed_successfully_no_docs"
    
    config.is_indexed = True
    save_config(config)
    return True, "indexed_successfully"


def run_indexing_from_config():
    """
    Запускает индексацию, используя настройки из config.json.
    
    Returns:
        Tuple[bool, str]: (успех, статус)
    """
    import asyncio
    try:
        # Вызов асинхронной функции индексации
        result = asyncio.run(run_indexing_logic())
        return result
    except Exception as e:
        logger.exception(f"Ошибка при индексации: {e}")
        return False, f"indexing_error: {str(e)}"