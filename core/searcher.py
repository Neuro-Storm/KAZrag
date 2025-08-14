"""Модуль для выполнения поиска в Qdrant."""

import logging
from typing import List, Tuple, Any, Optional
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from config.settings import load_config, Config
from core.embeddings import get_dense_embedder, get_search_device

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Создает и возвращает клиент Qdrant."""
    config: Config = load_config()
    return QdrantClient(url=config.qdrant_url)


def search_in_collection(query: str, collection_name: str, device: str, k: int, client: QdrantClient = None) -> Tuple[List[Tuple[Any, float]], Optional[str]]:
    """
    Выполняет поиск в указанной коллекции Qdrant.
    
    Args:
        query (str): Поисковый запрос.
        collection_name (str): Название коллекции.
        device (str): Устройство для поиска ("cpu" или "cuda").
        k (int): Количество результатов.
        client (QdrantClient, optional): Клиент Qdrant. Если не указан, создается новый.
        
    Returns:
        Tuple[List[Tuple[Any, float]], Optional[str]]: (результаты поиска, ошибка)
    """
    try:
        config: Config = load_config()
        
        # Если клиент не передан, создаем новый
        if client is None:
            client = get_qdrant_client()
            
        # Получаем эмбеддер для поиска
        search_device = get_search_device(device)
        embedder = get_dense_embedder(config, search_device)
        
        # Создаем QdrantVectorStore для поиска
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedder,
            vector_name="dense_vector"
        )
        
        # Выполняем поиск
        results = qdrant.similarity_search_with_score(query, k=k)
            
        return results, None
        
    except Exception as e:
        logger.error(f"Ошибка при поиске: {e}")
        return [], f"Ошибка при поиске: {str(e)}"