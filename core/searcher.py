"""Модуль для выполнения поиска в Qdrant."""

from typing import List, Tuple, Any
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from config.settings import load_config
from core.embeddings import get_dense_embedder, get_search_device


def get_qdrant_client() -> QdrantClient:
    """Создает и возвращает клиент Qdrant."""
    config = load_config()
    return QdrantClient(url=config["qdrant_url"])


def search_in_collection(
    query: str, 
    collection_name: str, 
    search_device: str = "cpu", 
    k: int = 5
) -> List[Tuple[Any, float]]:
    """
    Выполняет семантический поиск в указанной коллекции Qdrant.
    
    Args:
        query (str): Поисковый запрос.
        collection_name (str): Имя коллекции для поиска.
        search_device (str): Устройство для поиска ("cpu" или "cuda").
        k (int): Количество результатов.
        
    Returns:
        List[Tuple[Any, float]]: Список кортежей (документ, оценка схожести).
    """
    client = get_qdrant_client()
    config = load_config()
    results: List[Tuple[Any, float]] = []
    
    try:
        if config.get("use_dense_vectors"):
            device = get_search_device(search_device)
            dense_embedder = get_dense_embedder(config, device)
            qdrant_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=dense_embedder,
                vector_name="dense_vector",
            )
            results = qdrant_store.similarity_search_with_score(query=query, k=k)
    except Exception as e:
        print(f"Ошибка во время поиска: {e}")
        import traceback
        traceback.print_exc()
        results = []
        
    return results