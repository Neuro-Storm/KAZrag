"""Адаптер для SparseTextEmbedding из fastembed для обеспечения совместимости с langchain-qdrant."""

from typing import List

from fastembed import SparseTextEmbedding


class SparseVector:
    """Класс для представления sparse вектора с атрибутами values и indices."""
    
    def __init__(self, indices: List[int], values: List[float]):
        """Инициализация sparse вектора.
        
        Args:
            indices: Список индексов ненулевых значений
            values: Список соответствующих значений
        """
        self.indices = indices
        self.values = values


class SparseEmbeddingAdapter:
    """Адаптер для SparseTextEmbedding, предоставляющий метод embed_documents."""
    
    def __init__(self, model_name: str):
        """Инициализация адаптера.
        
        Args:
            model_name: Название модели для SparseTextEmbedding
        """
        self.sparse_embedding = SparseTextEmbedding(model_name=model_name)
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        """Генерация sparse эмбеддингов для списка документов.
        
        Args:
            texts: Список текстов для обработки
            
        Returns:
            Список sparse векторов
        """
        # Используем метод embed вместо embed_documents
        embeddings = list(self.sparse_embedding.embed(texts))
        # Преобразуем sparse эмбеддинги в формат, совместимый с langchain-qdrant
        # Возвращаем объекты SparseVector с атрибутами values и indices
        return [SparseVector(emb.indices.tolist(), emb.values.tolist()) for emb in embeddings]
    
    def embed_query(self, text: str) -> SparseVector:
        """Генерация sparse эмбеддинга для одного запроса.
        
        Args:
            text: Текст запроса
            
        Returns:
            Sparse вектор запроса
        """
        # Используем метод embed для одного текста
        embedding = list(self.sparse_embedding.embed([text]))[0]
        return SparseVector(embedding.indices.tolist(), embedding.values.tolist())