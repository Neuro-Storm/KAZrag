"""Модуль для индексации документов в Qdrant."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import psutil

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from core.qdrant_client import get_qdrant_client, aget_qdrant_client
from config.settings import Config
from core.embeddings import get_dense_embedder, get_device
from core.sparse_embedding_adapter import SparseEmbeddingAdapter
from core.metadata_manager import metadata_manager

logger = logging.getLogger(__name__)


class Indexer:
    """Класс для индексации документов в Qdrant."""

    def __init__(self, config: Config):
        """
        Инициализация индексатора.
        
        Args:
            config (Config): Конфигурация приложения.
        """
        self.config = config
        self.index_dense = getattr(config, 'index_dense', True)
        self.index_bm25 = getattr(config, 'index_bm25', False)
        self.index_hybrid = getattr(config, 'index_hybrid', False)
        
        # Проверка для гибридного режима
        if self.index_hybrid and not self.index_dense:
            raise IndexingError("Hybrid mode requires index_dense=True")
        
        # Определение размера батча
        mem = psutil.virtual_memory().available
        self.batch_size = (
            config.indexing_batch_size 
            if mem > config.memory_threshold 
            else config.indexing_batch_size // 2
        )
        
        # Инициализация sparse embedding если нужно
        self.sparse_emb = None
        if self.index_bm25 or self.index_hybrid:
            try:
                self.sparse_emb = SparseEmbeddingAdapter(config.sparse_embedding)
                logger.info(f"Sparse embedding adapter initialized: model={config.sparse_embedding}, type={type(self.sparse_emb)}")
            except ImportError:
                logger.warning("fastembed недоступен: индексация BM25/hybrid будет пропущена либо выполнится только dense часть.")
                self.sparse_emb = None
            except Exception as e:
                logger.exception(f"Ошибка при инициализации sparse embedding adapter ({config.sparse_embedding}): {e}")
                self.sparse_emb = None

    async def index_documents(
        self, 
        documents: List[Document], 
        client: Optional[QdrantClient] = None,
        folder_path_resolved: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """
        Индексация списка документов.
        
        Args:
            documents (List[Document]): Список документов для индексации.
            client (QdrantClient, optional): Клиент Qdrant. Если не указан, создается новый.
            folder_path_resolved (Path, optional): Путь к корневой папке для вычисления относительных путей.
            
        Returns:
            Tuple[bool, str]: (успех, статус)
        """
        if not documents:
            return True, "indexed_successfully_no_docs"
        
        # Если клиент не передан, создаем новый
        if client is None:
            client = await aget_qdrant_client(self.config)
        
        # Проверяем наличие коллекции и удаляем её, если force_recreate=True или если включён гибридный поиск
        try:
            client.get_collection(self.config.collection_name)
            # Если коллекция существует и force_recreate=True или включён гибридный режим, удаляем её
            if self.config.force_recreate or self.index_hybrid:
                client.delete_collection(self.config.collection_name)
        except Exception as e:
            # Коллекция не существует или произошла ошибка при проверке — логируем для отладки и продолжаем
            logger.debug(f"Проверка коллекции завершилась с ошибкой или коллекция отсутствует: {e}")
        
        # Подготовка эмбеддера
        device = get_device(self.config.device)
        dense_embedder = get_dense_embedder(self.config, device)
        
        # Обработка документов партиями
        docs_batch = []
        collection_created = False
        docs_processed = 0
        
        for doc in documents:
            docs_batch.append(doc)
            docs_processed += 1
            
            # Если набралась полная партия, обрабатываем её
            if len(docs_batch) >= self.batch_size:
                try:
                    await self._process_batch(
                        docs_batch, 
                        dense_embedder, 
                        client, 
                        collection_created,
                        folder_path_resolved
                    )
                    collection_created = True
                    docs_batch = []
                except Exception as e:
                    logger.exception(f"Ошибка при обработке партии документов: {e}")
                    raise IndexingError(f"Ошибка при обработке партии документов: {e}")
        
        # Обрабатываем оставшиеся документы в последней партии
        if docs_batch:
            try:
                await self._process_batch(
                    docs_batch, 
                    dense_embedder, 
                    client, 
                    collection_created,
                    folder_path_resolved
                )
            except Exception as e:
                logger.exception(f"Ошибка при обработке последней партии документов: {e}")
                raise IndexingError(f"Ошибка при обработке последней партии документов: {e}")
        
        return True, "indexed_successfully"

    async def _process_batch(
        self, 
        docs_batch: List[Document], 
        dense_embedder, 
        client: QdrantClient, 
        collection_created: bool,
        folder_path_resolved: Optional[Path] = None
    ):
        """
        Обработка партии документов.
        
        Args:
            docs_batch (List[Document]): Партия документов.
            dense_embedder: Эмбеддер для dense векторов.
            client (QdrantClient): Клиент Qdrant.
            collection_created (bool): Флаг, указывающий, была ли коллекция уже создана.
            folder_path_resolved (Path, optional): Путь к корневой папке для вычисления относительных путей.
        """
        if not collection_created:
            # Создаем новую коллекцию для первой партии
            await self._create_collection(docs_batch, dense_embedder, client)
        else:
            # Добавляем к существующей коллекции
            await self._add_to_collection(docs_batch, dense_embedder, client)

    async def _create_collection(self, docs_batch: List[Document], dense_embedder, client: QdrantClient):
        """
        Создание коллекции и добавление первой партии документов.
        
        Args:
            docs_batch (List[Document]): Партия документов.
            dense_embedder: Эмбеддер для dense векторов.
            client (QdrantClient): Клиент Qdrant.
        """
        if self.index_hybrid and self.sparse_emb:
            logger.info(f"Creating hybrid Qdrant collection '{self.config.collection_name}' with dense and sparse embeddings")
            QdrantVectorStore.from_documents(
                documents=docs_batch,
                url=self.config.qdrant_url,
                collection_name=self.config.collection_name,
                embedding=dense_embedder if self.index_dense else None,
                force_recreate=True,
                vector_name="dense_vector",
                sparse_embedding=self.sparse_emb if self.index_bm25 or self.index_hybrid else None,
                retrieval_mode=RetrievalMode.HYBRID,
                batch_size=self.config.indexing_batch_size,
            )
        elif self.index_bm25 and self.sparse_emb and not self.index_dense:
            logger.info(f"Creating sparse-only Qdrant collection '{self.config.collection_name}' using fastembed sparse embeddings")
            QdrantVectorStore.from_documents(
                documents=docs_batch,
                url=self.config.qdrant_url,
                collection_name=self.config.collection_name,
                embedding=None,
                force_recreate=True,
                vector_name=None,
                sparse_embedding=self.sparse_emb,
                retrieval_mode=RetrievalMode.SPARSE,
                batch_size=self.config.indexing_batch_size,
            )
        elif self.index_dense:
            logger.info(f"Creating dense-only Qdrant collection '{self.config.collection_name}'")
            QdrantVectorStore.from_documents(
                documents=docs_batch,
                url=self.config.qdrant_url,
                collection_name=self.config.collection_name,
                embedding=dense_embedder,
                force_recreate=True,
                vector_name="dense_vector",
                batch_size=self.config.indexing_batch_size,
            )
        else:
            # Нечего индексировать
            logger.warning("Нет подходящего режима индексации для первой партии (пропускаем)")

    async def _add_to_collection(self, docs_batch: List[Document], dense_embedder, client: QdrantClient):
        """
        Добавление партии документов к существующей коллекции.
        
        Args:
            docs_batch (List[Document]): Партия документов.
            dense_embedder: Эмбеддер для dense векторов.
            client (QdrantClient): Клиент Qdrant.
        """
        logger.info(f"Adding documents to collection '{self.config.collection_name}' (hybrid={self.index_hybrid and bool(self.sparse_emb)})")
        if self.index_hybrid and self.sparse_emb:
            qdrant_store = QdrantVectorStore(
                client=client,
                collection_name=self.config.collection_name,
                embedding=dense_embedder if self.index_dense else None,
                vector_name="dense_vector",
                sparse_embedding=self.sparse_emb,
                retrieval_mode=RetrievalMode.HYBRID
            )
        elif self.index_bm25 and self.sparse_emb and not self.index_dense:
            qdrant_store = QdrantVectorStore(
                client=client,
                collection_name=self.config.collection_name,
                embedding=None,
                vector_name=None,
                sparse_embedding=self.sparse_emb,
                retrieval_mode=RetrievalMode.SPARSE
            )
        else:
            qdrant_store = QdrantVectorStore(
                client=client,
                collection_name=self.config.collection_name,
                embedding=dense_embedder if self.index_dense else None,
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
                    if self.index_hybrid and self.sparse_emb:
                        QdrantVectorStore.from_documents(
                            documents=docs_batch,
                            url=self.config.qdrant_url,
                            collection_name=self.config.collection_name,
                            embedding=dense_embedder if self.index_dense else None,
                            force_recreate=True,
                            vector_name="dense_vector",
                            sparse_embedding=self.sparse_emb,
                            retrieval_mode=RetrievalMode.HYBRID,
                            batch_size=self.config.indexing_batch_size,
                        )
                    elif self.index_bm25 and self.sparse_emb and not self.index_dense:
                        QdrantVectorStore.from_documents(
                            documents=docs_batch,
                            url=self.config.qdrant_url,
                            collection_name=self.config.collection_name,
                            embedding=None,
                            force_recreate=True,
                            vector_name=None,
                            sparse_embedding=self.sparse_emb,
                            retrieval_mode=RetrievalMode.SPARSE,
                            batch_size=self.config.indexing_batch_size,
                        )
                    else:
                        QdrantVectorStore.from_documents(
                            documents=docs_batch,
                            url=self.config.qdrant_url,
                            collection_name=self.config.collection_name,
                            embedding=dense_embedder if self.index_dense else None,
                            force_recreate=True,
                            vector_name="dense_vector",
                            batch_size=self.config.indexing_batch_size,
                        )
                except Exception as recreate_error:
                    logger.exception("Не удалось пересоздать коллекцию для добавления sparse-векторов")
                    raise IndexingError(f"Не удалось пересоздать коллекцию: {recreate_error}")
            else:
                # Для других ошибок просто пробрасываем исключение
                raise


class IndexingError(Exception):
    """Пользовательское исключение для ошибок индексации."""
    pass