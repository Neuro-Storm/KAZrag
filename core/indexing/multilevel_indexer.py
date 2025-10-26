"""Модуль для многоуровневой индексации документов в Qdrant."""

import logging
import uuid  # Добавляем импорт uuid
from typing import Any, Dict, List

from langchain_core.documents import Document
from qdrant_client.http.models import SparseVectorParams
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

from config.settings import Config
from core.embedding.embedding_manager import EmbeddingManager
from core.embedding.sparse_embedding_adapter import SparseEmbeddingAdapter
from core.indexing.multilevel_chunker import (
    MultiLevelChunker,
)
from core.qdrant.qdrant_client import get_qdrant_client
from core.utils.constants import DEFAULT_COLLECTION_NAME
from .indexing_models import IndexingParams, MultiLevelChunkData

# Импортируем SparseEmbeddingAdapter
try:
    from core.embedding.sparse_embedding_adapter import SparseEmbeddingAdapter
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Sparse embedding adapter не доступен.")

logger = logging.getLogger(__name__)


class MultiLevelIndexer:
    """Индексатор с поддержкой многоуровневого чанкинга."""
    
    def __init__(self, config: Config):
        """
        Инициализация многоуровневого индексатора.
        
        Args:
            config (Config): Конфигурация приложения
        """
        self.config = config
        self.client = get_qdrant_client()
        self.embedding_manager = EmbeddingManager.get_instance()
        
        # Инициализация sparse embedding если нужно
        self.sparse_emb = None
        if getattr(config, 'use_bm25', False) and (getattr(config, 'index_bm25', False) or getattr(config, 'index_hybrid', False)):
            if SPARSE_AVAILABLE:
                try:
                    self.sparse_emb = SparseEmbeddingAdapter(config)
                    logger.info(f"Sparse embedding adapter initialized with native BM25")
                except Exception as e:
                    logger.exception(f"Ошибка при инициализации native sparse embedding adapter: {e}")
                    self.sparse_emb = None
            else:
                logger.warning("Native BM25 sparse embedding недоступен: sparse embedding будет пропущен.")
        
        # Создаем многоуровневый чанкер
        self.multilevel_chunker = self._create_multilevel_chunker()
    
    def _ensure_collection_exists(self):
        """Убедиться, что коллекция существует с правильной конфигурацией."""
        collection_name = getattr(self.config, 'collection_name', DEFAULT_COLLECTION_NAME)
        
        # Получаем размер вектора от эмбеддера
        sample_vector = self.embedding_manager.embed_query("test")
        vector_size = len(sample_vector)
        
        # Проверяем существование коллекции
        try:
            self.client.get_collection(collection_name)
            logger.info(f"Коллекция {collection_name} уже существует")
            # TODO: Проверить, соответствует ли конфигурация коллекции нашим требованиям
        except Exception:
            # Коллекция не существует, создаем новую
            logger.info(f"Создание коллекции {collection_name}")
            
            # Определяем конфигурацию векторов в зависимости от режима
            vectors_config = {}
            
            # Проверяем, нужно ли индексировать dense векторы
            index_dense = getattr(self.config, 'index_dense', True)
            # Проверяем, нужно ли индексировать sparse векторы
            index_sparse = (
                self.sparse_emb and 
                (getattr(self.config, 'index_bm25', False) or getattr(self.config, 'index_hybrid', False))
            )
            
            # Dense векторы с поддержкой мультивекторов
            if index_dense:
                vectors_config["dense_vector"] = VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    )
                )
                
            # Sparse векторы (если нужны)
            sparse_vectors_config = {}
            if index_sparse:
                from qdrant_client.models import SparseVectorParams, Modifier, SparseIndexParams
                sparse_vectors_config[getattr(self.config, 'sparse_vector_name', 'sparse_vector')] = SparseVectorParams(
                    modifier=Modifier.IDF,  # Enable BM25-like IDF
                    index=SparseIndexParams(on_disk=True)  # For large collections
                )
            
            logger.debug(f"Vectors config: {vectors_config}")
            logger.debug(f"Sparse vectors config: {sparse_vectors_config}")
            
            # Попробуем передать конфигурацию в правильном формате
            try:
                if sparse_vectors_config:
                    # Создаем коллекцию с dense и sparse векторами
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=vectors_config,
                        sparse_vectors_config=sparse_vectors_config
                    )
                else:
                    # Создаем коллекцию только с dense векторами (если они нужны) или без векторов
                    if vectors_config:
                        self.client.create_collection(
                            collection_name=collection_name,
                            vectors_config=vectors_config
                        )
                    else:
                        # Если ни один тип векторов не нужен, создаем коллекцию без векторов
                        # Это маловероятный случай, но обработаем его для полноты
                        self.client.create_collection(
                            collection_name=collection_name
                        )
                        
            except Exception as e:
                logger.error(f"Failed to create collection {collection_name}: {e}")
                raise
    
    def _create_multilevel_chunker(self) -> MultiLevelChunker:
        """Создание многоуровневого чанкера на основе конфигурации."""
        # Определяем стратегию микро-чанкинга
        micro_strategy = getattr(self.config, 'multilevel_micro_strategy', 'character')
        
        if micro_strategy == 'paragraph':
            return MultiLevelChunker(
                macro_chunk_strategy=getattr(self.config, 'multilevel_macro_strategy', 'character'),
                macro_chunk_size=getattr(self.config, 'multilevel_macro_chunk_size', 10000),
                macro_chunk_overlap=getattr(self.config, 'multilevel_macro_chunk_overlap', 1000),
                macro_paragraphs_per_chunk=getattr(self.config, 'multilevel_macro_paragraphs_per_chunk', 5),
                macro_paragraph_overlap=getattr(self.config, 'multilevel_macro_paragraph_overlap', 1),
                micro_chunk_strategy='paragraph',
                micro_paragraphs_per_chunk=getattr(self.config, 'multilevel_micro_paragraphs_per_chunk', 3),
                micro_paragraph_overlap=getattr(self.config, 'multilevel_micro_paragraph_overlap', 1)
            )
        elif micro_strategy == 'sentence':
            return MultiLevelChunker(
                macro_chunk_strategy=getattr(self.config, 'multilevel_macro_strategy', 'character'),
                macro_chunk_size=getattr(self.config, 'multilevel_macro_chunk_size', 10000),
                macro_chunk_overlap=getattr(self.config, 'multilevel_macro_chunk_overlap', 1000),
                macro_sentences_per_chunk=getattr(self.config, 'multilevel_macro_sentences_per_chunk', 10),
                macro_sentence_overlap=getattr(self.config, 'multilevel_macro_sentence_overlap', 1),
                micro_chunk_strategy='sentence',
                micro_sentences_per_chunk=getattr(self.config, 'multilevel_micro_sentences_per_chunk', 5),
                micro_sentence_overlap=getattr(self.config, 'multilevel_micro_sentence_overlap', 1)
            )
        else:  # character
            return MultiLevelChunker(
                macro_chunk_strategy=getattr(self.config, 'multilevel_macro_strategy', 'character'),
                macro_chunk_size=getattr(self.config, 'multilevel_macro_chunk_size', 10000),
                macro_chunk_overlap=getattr(self.config, 'multilevel_macro_chunk_overlap', 1000),
                micro_chunk_strategy='character',
                micro_chunk_size=getattr(self.config, 'multilevel_micro_chunk_size', 1000),
                micro_chunk_overlap=getattr(self.config, 'multilevel_micro_chunk_overlap', 100)
            )
    
    def index_documents_multilevel(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Многоуровневая индексация документов.
        
        Args:
            documents (List[Document]): Список документов для индексации
            
        Returns:
            Dict[str, Any]: Статистика индексации
        """
        logger.info(f"Начало многоуровневой индексации {len(documents)} документов")
        
        # Убедиться, что коллекция существует с правильной конфигурацией
        self._ensure_collection_exists()
        
        # Создаем многоуровневые чанки
        multilevel_chunks = self.multilevel_chunker.create_multilevel_chunks(documents)
        
        logger.info(f"После создания многоуровневых чанков: {len(multilevel_chunks)} чанков")
        
        # Подготавливаем данные для индексации
        points_to_upload = []
        total_vectors = 0
        
        logger.info(f"Начало обработки {len(multilevel_chunks)} чанков")
        
        for i, chunk_data in enumerate(multilevel_chunks):
            logger.info(f"Обработка чанка {i+1}/{len(multilevel_chunks)}")
            macro_chunk = chunk_data["macro_chunk"]
            micro_chunks = chunk_data["micro_chunks"]
            chunk_index = chunk_data["chunk_index"]
            
            logger.info(f"Макро-чанк {i+1}: {len(micro_chunks)} микро-чанков")
            
            # Создаем точку для Qdrant без явного указания ID (Qdrant сгенерирует UUID автоматически)
            
            # Получаем все dense векторы для макро-чанка (макро + микро)
            all_texts = [macro_chunk.page_content] + micro_chunks
            dense_vectors = self.embedding_manager.embed_texts(all_texts)
            
            # Получаем sparse вектор для макро-чанка
            sparse_vector = None
            if self.sparse_emb:
                try:
                    sparse_vector = self.sparse_emb.embed_query(macro_chunk.page_content)
                    # sparse_vector уже в формате {"indices": [...], "values": [...]} от адаптера
                except Exception as e:
                    logger.warning(f"Ошибка при генерации sparse вектора: {e}")
            
            # Определяем вектор для точки в зависимости от режима
            point_vector = {}  # Инициализируем пустой словарь для векторов
            
            # Проверяем, нужно ли индексировать dense векторы
            index_dense = getattr(self.config, 'index_dense', True)
            # Проверяем, нужно ли индексировать sparse векторы
            index_sparse = (
                self.sparse_emb and 
                (getattr(self.config, 'index_bm25', False) or getattr(self.config, 'index_hybrid', False))
            )
            
            if index_dense:
                # Добавляем dense векторы (мультивекторы)
                point_vector["dense_vector"] = dense_vectors
                
            # Добавляем sparse векторы в основной вектор, если они есть
            if index_sparse and sparse_vector:
                point_vector[getattr(self.config, 'sparse_vector_name', 'sparse_vector')] = sparse_vector
            
            # Создаем точку для Qdrant с автоматически сгенерированным UUID
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=point_vector,
                payload={
                    "content": macro_chunk.page_content,
                    "metadata": macro_chunk.metadata,
                    "chunk_index": chunk_index,
                    "total_micro_chunks": len(micro_chunks),
                    "micro_contents": micro_chunks,  # Сохраняем содержимое микро-чанков для отладки
                    "source": macro_chunk.metadata.get("source", "unknown")  # Добавляем source в payload для отладки
                }
            )
        
            # Добавляем точку в список для загрузки
            points_to_upload.append(point)
            total_vectors += len(dense_vectors) if index_dense else 0
            
            logger.info(f"Добавлена точка {i+1}, всего точек: {len(points_to_upload)}")
        
        # Загружаем точки в Qdrant
        collection_name = getattr(self.config, 'collection_name', DEFAULT_COLLECTION_NAME)
        
        # Если включена перезапись, удаляем существующую коллекцию
        if getattr(self.config, 'force_recreate', False):
            try:
                self.client.delete_collection(collection_name)
                self._ensure_collection_exists()
            except Exception as e:
                logger.warning(f"Не удалось удалить коллекцию {collection_name}: {e}")
        
        self.client.upload_points(
            collection_name=collection_name,
            points=points_to_upload,
            wait=True
        )
        
        logger.info(f"Успешно проиндексировано {len(points_to_upload)} макро-чанков с {total_vectors} dense векторами")
        
        return {
            "macro_chunks": len(points_to_upload),
            "total_vectors": total_vectors,
            "micro_vectors_per_chunk": total_vectors - len(points_to_upload) if len(points_to_upload) > 0 and index_dense else 0
        }
    
    def search_multilevel(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Многоуровневый поиск.
        
        Args:
            query (str): Поисковый запрос
            k (int): Количество результатов
            
        Returns:
            List[Dict[str, Any]]: Результаты поиска
        """
        # Определяем режим поиска
        index_dense = getattr(self.config, 'index_dense', True)
        index_bm25 = getattr(self.config, 'index_bm25', False)
        index_hybrid = getattr(self.config, 'index_hybrid', False)
        
        # Проверяем, нужно ли использовать sparse векторы
        use_sparse = self.sparse_emb and (index_bm25 or index_hybrid)
        
        logger.info(f"Search mode: index_dense={index_dense}, index_bm25={index_bm25}, index_hybrid={index_hybrid}")
        logger.info(f"Use sparse: {use_sparse}")
        
        # Определяем вектор для поиска в зависимости от режима
        search_vector = None
        
        if index_hybrid and index_dense and use_sparse:
            # Гибридный поиск - используем оба вектора
            # Векторизуем запрос для dense поиска
            dense_query_vector = self.embedding_manager.embed_query(query)
            
            # Векторизуем запрос для sparse поиска
            sparse_vector = self.sparse_emb.embed_query(query)
            # sparse_vector уже в формате {"indices": [...], "values": [...]} от адаптера
            
            # Для гибридного поиска передаем оба вектора
            search_vector = {
                "dense_vector": dense_query_vector,
                getattr(self.config, 'sparse_vector_name', 'sparse_vector'): sparse_vector
            }
            logger.info("Using hybrid search mode")
        elif index_dense:
            # Только dense поиск
            query_vector = self.embedding_manager.embed_query(query)
            search_vector = {"dense_vector": query_vector}
            logger.info("Using dense search mode")
        elif use_sparse:
            # Только sparse поиск
            sparse_vector = self.sparse_emb.embed_query(query)
            # sparse_vector уже в формате {"indices": [...], "values": [...]} от адаптера
            search_vector = {getattr(self.config, 'sparse_vector_name', 'sparse_vector'): sparse_vector}
            logger.info("Using sparse search mode")
        else:
            logger.warning("No valid search mode determined")
            return []
        
        logger.info(f"Search vector: {search_vector}")
        
        # Выполняем поиск в Qdrant с именованным вектором
        collection_name = getattr(self.config, 'collection_name', DEFAULT_COLLECTION_NAME)
        
        # Для гибридного поиска используем метод query_points или раздельный подход
        if index_hybrid and index_dense and use_sparse:
            # Гибридный поиск - выполним отдельные поиски и объединим результаты
            logger.info("Выполнение гибридного поиска")
            
            # Dense поиск
            dense_results = self.client.search(
                collection_name=collection_name,
                query_vector=("dense_vector", dense_query_vector),
                limit=k * 2,  # Берем больше для reranking
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Sparse поиск
            sparse_results = self.client.search(
                collection_name=collection_name,
                query_vector=None,  # Для sparse поиска query_vector=None
                sparse_vector={getattr(self.config, 'sparse_vector_name', 'sparse_vector'): sparse_vector},  # {indices: [...], values: [...]}
                vector_name=getattr(self.config, 'sparse_vector_name', 'sparse_vector'),  # Имя sparse named vector
                limit=k * 2,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Объединяем результаты (простое объединение с reranking)
            # Создаем словарь для объединения результатов по ID
            combined_dict = {}
            
            # Добавляем dense результаты (с коэффициентом alpha)
            config = ConfigManager.get_instance().get()
            alpha = getattr(config, "hybrid_alpha", 0.7)  # Вес для dense результатов (повышенный вес для семантики)
            for result in dense_results:
                result_id = result.id
                combined_dict[result_id] = {
                    'point': result,
                    'score': alpha * result.score
                }
            
            # Добавляем sparse результаты (с коэффициентом 1-alpha)
            beta = 1 - alpha
            for result in sparse_results:
                result_id = result.id
                if result_id in combined_dict:
                    # Если результат уже есть, добавляем к существующему скору
                    combined_dict[result_id]['score'] += beta * result.score
                else:
                    # Если результат новый, добавляем его
                    combined_dict[result_id] = {
                        'point': result,
                        'score': beta * result.score
                    }
            
            # Сортируем по объединенному скору и берем top-k
            sorted_results = sorted(combined_dict.items(), key=lambda x: x[1]['score'], reverse=True)[:k]
            
            # Возвращаем только объекты PointStruct
            search_results = [item[1]['point'] for item in sorted_results]
        elif index_dense:
            # Только dense поиск
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=("dense_vector", search_vector["dense_vector"]),
                limit=k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
        elif use_sparse:
            # Только sparse поиск
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=None,
                sparse_vector=search_vector,
                vector_name=getattr(self.config, 'sparse_vector_name', 'sparse_vector'),
                limit=k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
        else:
            logger.warning("No valid search mode determined")
            return []
        
        logger.info(f"Search returned {len(search_results)} results")
        
        # Обрабатываем результаты
        results = []
        for result in search_results:
            processed_result = {
                "content": result.payload.get("content", ""),
                "metadata": result.payload.get("metadata", {}),
                "score": result.score,
                "chunk_index": result.payload.get("chunk_index", 0),
                "total_micro_chunks": result.payload.get("total_micro_chunks", 0),
                "micro_contents": result.payload.get("micro_contents", []),
                "source": result.payload.get("source", "unknown")
            }
            results.append(processed_result)
        
        logger.info(f"Processed {len(results)} results for return")
        return results


def create_multilevel_indexer(config: Config) -> MultiLevelIndexer:
    """Создание многоуровневого индексатора."""
    return MultiLevelIndexer(config)