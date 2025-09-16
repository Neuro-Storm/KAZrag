"""Module for centralized reranker management."""

import logging
from collections import OrderedDict
from typing import List, Tuple, Optional, Any

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config.settings import Config
from core.search.gguf_reranker import get_local_reranker, LocalReranker

logger = logging.getLogger(__name__)


class RerankerError(Exception):
    """Custom exception for reranker management errors."""
    pass


class RerankerManager:
    """Centralized manager for rerankers with caching."""
    
    _instance: Optional['RerankerManager'] = None
    MAX_CACHE_SIZE = 2  # Максимум 2 модели в кэше одновременно
    
    def __init__(self):
        """Initialize RerankerManager."""
        self._reranker_cache: OrderedDict = OrderedDict()
        
    @classmethod
    def get_instance(cls) -> 'RerankerManager':
        """Get singleton instance of RerankerManager.
        
        Returns:
            RerankerManager: Singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_reranker(self, config: Config) -> CrossEncoder:
        """Получает или создает кэшированный экземпляр reranker.
        
        Args:
            config: Configuration object
            
        Returns:
            CrossEncoder or LocalReranker: Reranker instance
            
        Raises:
            RerankerError: If there's an error creating the reranker
        """
        try:
            model_name = config.reranker_model
            
            # Проверяем, является ли модель локальной (по расширению .gguf)
            is_local = model_name.endswith('.gguf')
            
            # Ключ кэша - кортеж (модель, is_local)
            cache_key = (model_name, is_local)
            
            # Проверяем, есть ли в кэше модель с такими параметрами
            if cache_key in self._reranker_cache:
                # Перемещаем модель в начало OrderedDict (обновляем время последнего использования)
                self._reranker_cache.move_to_end(cache_key, last=True)
                return self._reranker_cache[cache_key]
            
            # Если кэш переполнен, удаляем самую старую запись
            if len(self._reranker_cache) >= self.MAX_CACHE_SIZE:
                # Удаляем самую старую запись (первую в OrderedDict)
                oldest_key, _ = self._reranker_cache.popitem(last=False)
                logger.info(f"Удалена старая модель из кэша: {oldest_key}")
            
            # Создаем новый reranker
            if is_local:
                reranker = get_local_reranker(config)
            else:
                reranker = CrossEncoder(model_name)
            
            # Сохраняем модель в кэш
            self._reranker_cache[cache_key] = reranker
            
            return reranker
            
        except Exception as e:
            logger.exception(f"Error creating reranker: {e}")
            raise RerankerError(f"Failed to create reranker: {e}")
    
    def clear_cache(self) -> None:
        """Очищает кэш rerankers."""
        self._reranker_cache.clear()
        logger.info("Кэш rerankers очищен")
    
    def get_cache_info(self) -> dict:
        """Получает информацию о кэше rerankers.
        
        Returns:
            dict: Cache information
        """
        return {
            "cache_size": len(self._reranker_cache),
            "max_cache_size": self.MAX_CACHE_SIZE,
            "cached_models": list(self._reranker_cache.keys())
        }
    
    def rerank_documents(self, query: str, documents: List[Tuple[Any, float]], config: Config) -> List[Tuple[Any, float]]:
        """Переранжирует список документов.
        
        Args:
            query: Поисковый запрос
            documents: Список документов с оценками (Document, score)
            config: Configuration object
            
        Returns:
            List[Tuple[Any, float]]: Переранжированные документы с новыми оценками
        """
        try:
            if not documents:
                return []
            
            # Получаем reranker
            reranker = self.get_reranker(config)
            
            # Подготавливаем пары (запрос, содержимое)
            pairs = []
            doc_objects = []
            
            for doc, score in documents:
                # Извлекаем содержимое
                if isinstance(doc, dict):
                    content = doc.get('content', '')
                elif hasattr(doc, 'page_content'):
                    content = doc.page_content
                else:
                    content = str(doc)
                
                if content:
                    pairs.append([query, content])
                    doc_objects.append((doc, score))
                else:
                    logger.debug(f"Document has no content: {type(doc)}, keys: {list(doc.keys()) if isinstance(doc, dict) else 'not dict'}")
            
            logger.debug(f"Prepared {len(pairs)} pairs for reranking, {len(doc_objects)} doc objects")
            
            if not pairs:
                logger.debug("No pairs to rerank, returning original documents")
                return documents
            
            # Получаем оценки от reranker
            scores = reranker.predict(pairs)
            
            # Создаем пары (документ, новая_оценка) и сортируем по убыванию оценок
            reranked_docs = []
            for i, (doc, original_score) in enumerate(doc_objects):
                # Если документ уже является словарем, обновляем его, иначе создаем новый словарь
                if isinstance(doc, dict):
                    # Добавляем оригинальную оценку к документу
                    doc_with_scores = doc.copy()
                    doc_with_scores['original_score'] = original_score
                    doc_with_scores['reranker_score'] = float(scores[i])
                    reranked_docs.append((doc_with_scores, float(scores[i])))
                else:
                    # Создаем словарь с содержимым и оценками
                    doc_with_scores = {
                        'content': getattr(doc, 'page_content', str(doc)),
                        'metadata': getattr(doc, 'metadata', {}),
                        'original_score': original_score,
                        'reranker_score': float(scores[i])
                    }
                    reranked_docs.append((doc_with_scores, float(scores[i])))
            
            # Сортируем документы по оценкам reranker (по убыванию)
            reranked_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Возвращаем топ-K
            top_k = min(config.reranker_top_k, len(reranked_docs))
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.exception(f"Error reranking documents: {e}")
            # Возвращаем оригинальные документы в случае ошибки
            return documents