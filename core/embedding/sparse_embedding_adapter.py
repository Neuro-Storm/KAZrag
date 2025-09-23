"""Адаптер для разреженных эмбеддингов с native BM25."""

import logging
from collections import Counter
from typing import Dict, List, Tuple

from qdrant_client.models import SparseVector
from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class SparseEmbeddingAdapter:
    """Адаптер для sparse эмбеддингов, заменяющий Fastembed на native BM25."""
    
    def __init__(self, config=None):
        """Инициализирует адаптер."""
        self.config = config or ConfigManager.get_instance().get()
        if not self.config.use_bm25:
            raise ValueError("BM25 not enabled in config")

    def encode(self, texts: List[str], return_sparse: bool = True) -> List[SparseVector]:
        """
        Генерирует sparse vectors для текстов (native BM25-like: токены + TF weights).
        
        Args:
            texts: List of texts.
            return_sparse: Always True for sparse.
            
        Returns:
            List of SparseVector (indices: token hashes, values: TF).
        """
        sparse_vectors = []
        for text in texts:
            # Simple tokenization (adapt to config.bm25_tokenizer)
            tokens = text.lower().split()  # Word tokenizer
            tokens = [t for t in tokens if len(t) >= self.config.bm25_min_token_len]
            
            # TF weights (frequency)
            token_counts = Counter(tokens)
            total_terms = len(tokens)
            indices = [hash(token) for token in token_counts.keys()]  # Use hash for indices (positive ints)
            values = [count / total_terms for count in token_counts.values()]  # Normalized TF
            
            # Ensure indices are unique and sorted (Qdrant requirement)
            indexed = sorted(zip(indices, values))
            indices = [idx for idx, _ in indexed]
            values = [val for _, val in indexed]
            
            sparse_vectors.append(SparseVector(indices=indices, values=values))
        
        logger.debug(f"Generated {len(sparse_vectors)} sparse vectors for BM25")
        return sparse_vectors

    def embed_query(self, query: str) -> SparseVector:
        """Single query embedding."""
        return self.encode([query], return_sparse=True)[0]