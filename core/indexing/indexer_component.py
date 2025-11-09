"""Модуль для индексации документов в Qdrant."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import psutil
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from config.settings import Config
from core.embedding.embeddings import get_dense_embedder, get_device
from core.embedding.sparse_embedding_adapter import SparseEmbeddingAdapter
from core.indexing.indexing_tracker import IndexingTracker
from core.qdrant.qdrant_client import aget_qdrant_client
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

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
        if config.use_bm25 and (self.index_bm25 or self.index_hybrid):
            try:
                from core.embedding.sparse_embedding_adapter import SparseEmbeddingAdapter
                self.sparse_emb = SparseEmbeddingAdapter(config)
                logger.info(f"Native BM25 sparse embedding adapter initialized")
            except ImportError:
                logger.warning("Не удалось инициализировать native BM25: индексация BM25/hybrid будет пропущена либо выполнится только dense часть.")
                self.sparse_emb = None
            except Exception as e:
                logger.exception(f"Ошибка при инициализации native BM25 sparse embedding adapter: {e}")
                self.sparse_emb = None

    async def index_documents(
        self,
        documents: List[Document],
        client: Optional[QdrantClient] = None,
        folder_path_resolved: Optional[Path] = None,
        file_paths: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Индексация списка документов.

        Args:
            documents (List[Document]): Список документов для индексации.
            client (QdrantClient, optional): Клиент Qdrant. Если не указан, создается новый.
            folder_path_resolved (Path, optional): Путь к корневой папке для вычисления относительных путей.
            file_paths (List[str], optional): Список путей файлов для отслеживания прогресса.

        Returns:
            Tuple[bool, str]: (успех, статус)
        """
        if not documents:
            return True, "indexed_successfully_no_docs"
        
        # Если клиент не передан, создаем новый
        if client is None:
            client = await aget_qdrant_client(self.config)
        
        # Проверяем наличие коллекции и удаляем её, если включена перезапись
        try:
            client.get_collection(self.config.collection_name)
            # Если коллекция существует и включена перезапись, удаляем её
            if self.config.force_recreate:
                client.delete_collection(self.config.collection_name)
        except Exception as e:
            # Коллекция не существует или произошла ошибка при проверке — логируем для отладки и продолжаем
            logger.debug(f"Проверка коллекции завершилась с ошибкой или коллекция отсутствует: {e}")
        
        # Подготовка эмбеддера
        device = get_device(self.config.device)
        dense_embedder = get_dense_embedder(self.config, device)

        # Инициализация трекера для проверки остановки
        tracker = IndexingTracker()

        # Группируем документы по файлам для отслеживания
        docs_by_file = {}
        for doc in documents:
            source_path = doc.metadata.get("source", "")
            if source_path and file_paths:
                # Находим полный путь к файлу
                for file_path in file_paths:
                    if source_path in file_path or file_path.endswith(source_path):
                        if file_path not in docs_by_file:
                            docs_by_file[file_path] = []
                        docs_by_file[file_path].append(doc)
                        break
                else:
                    # Если файл не найден в списке, используем source_path как ключ
                    if source_path not in docs_by_file:
                        docs_by_file[source_path] = []
                    docs_by_file[source_path].append(doc)
            else:
                # Если нет информации о файле, группируем по source
                if source_path not in docs_by_file:
                    docs_by_file[source_path] = []
                docs_by_file[source_path].append(doc)

        logger.info(f"Документы сгруппированы по {len(docs_by_file)} файлам")

        # Обработка документов по файлам
        collection_created = False
        for file_path, file_docs in docs_by_file.items():
            # Проверяем остановку перед обработкой файла
            if tracker.is_indexing_stopped():
                logger.info("Индексация остановлена во время обработки документов")
                raise IndexingError("indexing_stopped")

            # Обновляем статус файла на индексацию в Qdrant
            try:
                tracker.update_file(Path(file_path), "indexing")
                logger.info(f"Начало индексации в Qdrant файла: {Path(file_path).name}")
            except Exception as e:
                logger.warning(f"Не удалось обновить статус файла {file_path}: {e}")

            # Обрабатываем документы файла партиями
            docs_batch = []
            for doc in file_docs:
                # Проверяем остановку перед обработкой каждого документа
                if tracker.is_indexing_stopped():
                    logger.info("Индексация остановлена во время обработки документов")
                    raise IndexingError("indexing_stopped")

                docs_batch.append(doc)

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
                        if "indexing_stopped" in str(e):
                            logger.info("Индексация остановлена в процессе обработки партии")
                            raise IndexingError("indexing_stopped")
                        logger.exception(f"Ошибка при обработке партии документов: {e}")
                        raise IndexingError(f"Ошибка при обработке партии документов: {e}")

            # Обрабатываем оставшиеся документы файла
            if docs_batch and not tracker.is_indexing_stopped():
                try:
                    await self._process_batch(
                        docs_batch,
                        dense_embedder,
                        client,
                        collection_created,
                        folder_path_resolved
                    )
                except Exception as e:
                    if "indexing_stopped" in str(e):
                        logger.info("Индексация остановлена в процессе обработки последней партии файла")
                        raise IndexingError("indexing_stopped")
                    logger.exception(f"Ошибка при обработке последней партии документов файла: {e}")
                    raise IndexingError(f"Ошибка при обработке последней партии документов файла: {e}")

            # Отмечаем файл как проиндексированный в Qdrant
            try:
                tracker.update_file(Path(file_path), "indexed")
                logger.info(f"Файл {Path(file_path).name} успешно проиндексирован в Qdrant")
            except Exception as e:
                logger.warning(f"Не удалось обновить статус файла {file_path} на indexed: {e}")

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
        # Инициализация трекера для проверки остановки
        tracker = IndexingTracker()

        # Проверяем остановку перед обработкой партии
        if tracker.is_indexing_stopped():
            logger.info("Индексация остановлена перед обработкой партии")
            raise IndexingError("indexing_stopped")

        if not collection_created:
            # Создаем новую коллекцию для первой партии (только создаем коллекцию, не добавляем документы)
            await self._create_collection(docs_batch, dense_embedder, client)
            # Затем добавляем документы в созданную коллекцию
            await self._add_to_collection(docs_batch, dense_embedder, client)
        else:
            # Добавляем к существующей коллекции
            await self._add_to_collection(docs_batch, dense_embedder, client)

    async def _create_collection(self, docs_batch: List[Document], dense_embedder, client: QdrantClient):
        """
        Создание коллекции без добавления документов.

        Args:
            docs_batch (List[Document]): Партия документов (используется только для определения типа индексации).
            dense_embedder: Эмбеддер для dense векторов.
            client (QdrantClient): Клиент Qdrant.
        """
        # Инициализация трекера для проверки остановки
        tracker = IndexingTracker()

        # Проверяем остановку перед созданием коллекции
        if tracker.is_indexing_stopped():
            logger.info("Индексация остановлена перед созданием коллекции")
            raise IndexingError("indexing_stopped")

        if self.index_hybrid and self.sparse_emb:
            logger.info(f"Creating hybrid Qdrant collection '{self.config.collection_name}' with dense and sparse embeddings")

            # Проверяем, существует ли коллекция
            collection_exists = False
            try:
                collection_info = client.get_collection(self.config.collection_name)
                collection_exists = True
                logger.info(f"Коллекция '{self.config.collection_name}' уже существует")
            except Exception:
                logger.info(f"Коллекция '{self.config.collection_name}' не существует, создаем новую")

            # Проверяем остановку еще раз перед созданием
            if tracker.is_indexing_stopped():
                logger.info("Индексация остановлена перед созданием коллекции")
                raise IndexingError("indexing_stopped")

            if collection_exists:
                # Коллекция существует, просто выходим - документы будут добавлены через _add_to_collection
                logger.info(f"Коллекция '{self.config.collection_name}' уже существует")
                return
            else:
                # Если коллекция не существует, создаем её вручную
                logger.info(f"Создаем новую коллекцию '{self.config.collection_name}'")

                # Определяем конфигурацию векторов
                vectors_config = {}
                if self.index_dense:
                    # Получаем размерность вектора
                    try:
                        # Пробуем получить размерность через embed_query
                        test_embedding = dense_embedder.embed_query("test")
                        vector_size = len(test_embedding)
                    except Exception:
                        # Если не получилось, используем стандартное значение
                        vector_size = 768  # Стандартный размер для ru-en-RoSBERTa

                    vectors_config["dense_vector"] = VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )

                # Определяем конфигурацию sparse векторов
                sparse_vectors_config = None
                if self.config.use_bm25 and (self.index_bm25 or self.index_hybrid):
                    sparse_vectors_config = {
                        self.config.sparse_vector_name or "sparse_vector": SparseVectorParams()
                    }

                # Создаем коллекцию
                client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=vectors_config if vectors_config else None,
                    sparse_vectors_config=sparse_vectors_config
                )
        elif self.index_bm25 and self.sparse_emb and not self.index_dense:
            logger.info(f"Creating sparse-only Qdrant collection '{self.config.collection_name}' using native BM25 sparse embeddings")

            # Проверяем, существует ли коллекция
            collection_exists = False
            try:
                collection_info = client.get_collection(self.config.collection_name)
                collection_exists = True
                logger.info(f"Коллекция '{self.config.collection_name}' уже существует")
            except Exception:
                logger.info(f"Коллекция '{self.config.collection_name}' не существует, создаем новую")

            # Проверяем остановку еще раз перед созданием
            if tracker.is_indexing_stopped():
                logger.info("Индексация остановлена перед созданием коллекции")
                raise IndexingError("indexing_stopped")

            if collection_exists:
                # Коллекция существует, просто выходим - документы будут добавлены через _add_to_collection
                logger.info(f"Коллекция '{self.config.collection_name}' уже существует")
                return
            else:
                # Если коллекция не существует, создаем её вручную
                logger.info(f"Создаем новую sparse коллекцию '{self.config.collection_name}'")

                # Создаем sparse коллекцию
                sparse_vectors_config = {
                    self.config.sparse_vector_name or "sparse_vector": SparseVectorParams()
                }

                client.create_collection(
                    collection_name=self.config.collection_name,
                    sparse_vectors_config=sparse_vectors_config
                )
        elif self.index_dense:
            logger.info(f"Creating dense-only Qdrant collection '{self.config.collection_name}'")

            # Проверяем, существует ли коллекция
            collection_exists = False
            try:
                collection_info = client.get_collection(self.config.collection_name)
                collection_exists = True
                logger.info(f"Коллекция '{self.config.collection_name}' уже существует")
            except Exception:
                logger.info(f"Коллекция '{self.config.collection_name}' не существует, создаем новую")

            # Проверяем остановку еще раз перед созданием
            if tracker.is_indexing_stopped():
                logger.info("Индексация остановлена перед созданием коллекции")
                raise IndexingError("indexing_stopped")

            if collection_exists:
                # Коллекция существует, просто выходим - документы будут добавлены через _add_to_collection
                logger.info(f"Коллекция '{self.config.collection_name}' уже существует")
                return
            else:
                # Если коллекция не существует, создаем её вручную
                logger.info(f"Создаем новую dense коллекцию '{self.config.collection_name}'")

                # Получаем размерность вектора
                try:
                    # Пробуем получить размерность через embed_query
                    test_embedding = dense_embedder.embed_query("test")
                    vector_size = len(test_embedding)
                except Exception:
                    # Если не получилось, используем стандартное значение
                    vector_size = 768  # Стандартный размер для ru-en-RoSBERTa

                # Создаем dense коллекцию
                vectors_config = {
                    "dense_vector": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                }

                client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=vectors_config
                )
        else:
            # Нечего индексировать
            logger.warning("Нет подходящего режима индексации для первой партии (пропускаем)")

    async def _add_to_collection(self, docs_batch: List[Document], dense_embedder, client: QdrantClient):
        """
        Добавление партии документов к существующей коллекции.
        Использует прямое Qdrant API вместо QdrantVectorStore для избежания автоматического удаления коллекции.

        Args:
            docs_batch (List[Document]): Партия документов.
            dense_embedder: Эмбеддер для dense векторов.
            client (QdrantClient): Клиент Qdrant.
        """
        import uuid
        from qdrant_client.models import PointStruct

        # Инициализация трекера для проверки остановки
        tracker = IndexingTracker()

        # Проверяем остановку перед добавлением в коллекцию
        if tracker.is_indexing_stopped():
            logger.info("Индексация остановлена перед добавлением в коллекцию")
            raise IndexingError("indexing_stopped")

        logger.info(f"Adding {len(docs_batch)} documents to collection '{self.config.collection_name}'")

        # Проверяем существование коллекции
        try:
            collection_info = client.get_collection(self.config.collection_name)
            logger.info(f"Collection '{self.config.collection_name}' exists, adding documents...")
        except Exception as e:
            logger.error(f"Collection '{self.config.collection_name}' does not exist: {e}")
            logger.error("This should not happen! Collection should be created in _create_collection method first.")
            raise IndexingError(f"Collection {self.config.collection_name} not found")

        # Готовим точки для добавления
        points = []
        for i, doc in enumerate(docs_batch):
            # Проверяем остановку в цикле обработки документов
            if tracker.is_indexing_stopped():
                logger.info("Индексация остановлена во время подготовки документов")
                raise IndexingError("indexing_stopped")

            point_id = str(uuid.uuid4())
            payload = doc.metadata.copy()
            payload["page_content"] = doc.page_content

            # Определяем векторы в зависимости от режима индексации
            vector_dict = {}

            if self.index_dense and dense_embedder:
                # Добавляем dense вектор
                dense_vector = dense_embedder.embed_query(doc.page_content)
                vector_dict["dense_vector"] = dense_vector

            if self.config.use_bm25 and (self.index_bm25 or self.index_hybrid) and self.sparse_emb:
                # Добавляем sparse вектор
                sparse_vector = self.sparse_emb.embed_query(doc.page_content)
                vector_dict[self.config.sparse_vector_name or "sparse_vector"] = sparse_vector

            # Создаем точку
            point = PointStruct(
                id=point_id,
                vector=vector_dict,
                payload=payload
            )
            points.append(point)

        # Проверяем остановку перед upsert
        if tracker.is_indexing_stopped():
            logger.info("Индексация остановлена перед добавлением точек в коллекцию")
            raise IndexingError("indexing_stopped")

        # Добавляем точки в коллекцию
        try:
            client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True
            )
            logger.info(f"Successfully added {len(points)} points to collection '{self.config.collection_name}'")
        except Exception as e:
            logger.exception(f"Error upserting points to collection '{self.config.collection_name}': {e}")
            # НЕ удаляем коллекцию при ошибке! Вместо этого логируем ошибку и пробрасываем исключение
            raise IndexingError(f"Failed to upsert points to collection {self.config.collection_name}: {str(e)}")


class IndexingError(Exception):
    """Пользовательское исключение для ошибок индексации."""
    pass