"""Модуль для выполнения поиска в Qdrant."""

import logging
from typing import List, Tuple, Any, Optional
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from config.settings import load_config, Config
from core.embeddings import get_dense_embedder, get_search_device

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Создает и возвращает клиент Qdrant."""
    config: Config = load_config()
    return QdrantClient(url=config.qdrant_url)


def search_in_collection(
    query: str, 
    collection_name: str, 
    search_device: str = "cpu", 
    k: int = 5
) -> Tuple[List[Tuple[Any, float]], Optional[str]]:
    """
    Выполняет семантический поиск в указанной коллекции Qdrant.
    
    Args:
        query (str): Поисковый запрос.
        collection_name (str): Имя коллекции для поиска.
        search_device (str): Устройство для поиска ("cpu" или "cuda").
        k (int): Количество результатов.
        
    Returns:
        Tuple[List[Tuple[Any, float]], Optional[str]]: Список кортежей (документ, оценка схожести) и сообщение об ошибке (если есть).
    """
    logger.info(f"Starting search for query: {query}, collection: {collection_name}, device: {search_device}, k: {k}")
    
    client = get_qdrant_client()
    config: Config = load_config()
    results: List[Tuple[Any, float]] = []
    
    # Проверяем наличие коллекции
    logger.info(f"Checking if collection {collection_name} exists")
    try:
        client.get_collection(collection_name)
        collection_exists = True
    except Exception:
        collection_exists = False
    
    if not collection_exists:
        logger.warning(f"Collection {collection_name} not found")
        return [], "collection_not_found"
    
    try:
        if config.use_dense_vectors:
            logger.info("Using dense vectors for search")
            device = get_search_device(search_device)
            logger.info(f"Search device determined: {device}")
            dense_embedder = get_dense_embedder(config, device)
            logger.info("Dense embedder created successfully")
            
            qdrant_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=dense_embedder,
                vector_name="dense_vector",
            )
            logger.info("QdrantVectorStore created successfully")
            
            results = qdrant_store.similarity_search_with_score(query=query, k=k)
            logger.info(f"Search completed, found {len(results)} results")
        else:
            logger.info("Dense vectors are not enabled")
    except Exception as e:
        logger.error(f"Error during search: {e}")
        import traceback
        traceback.print_exc()
        results = []
        return results, str(e)
        
    return results, None