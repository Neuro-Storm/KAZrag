"""Адаптер для разреженных эмбеддингов с native BM25."""

import logging
from collections import Counter
from typing import Dict, List, Tuple, Any

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

    def encode(self, texts: List[str], return_sparse: bool = True) -> List[Dict[str, Any]]:
        """
        Генерирует sparse vectors для текстов (native BM25-like: токены + TF weights).
        Returns raw dict for Qdrant JSON: {"indices": [int], "values": [float]}.
        
        Args:
            texts: List of texts.
            return_sparse: Always True for sparse.
            
        Returns:
            List of dict (for SparseVector).
        """
        sparse_vectors = []
        for text in texts:
            # Simple tokenization (adapt to config.bm25_tokenizer)
            tokens = text.lower().split()  # Word tokenizer
            tokens = [t for t in tokens if len(t) >= self.config.bm25_min_token_len]
            
            # TF weights (frequency)
            token_counts = Counter(tokens)
            total_terms = len(tokens) or 1  # Avoid div/0
            indices = [abs(hash(token)) % (2**31 - 1) for token in token_counts.keys()]  # Positive unique-ish ints
            values = [count / total_terms for count in token_counts.values()]
            
            # Ensure unique/sorted (Qdrant req)
            unique_pairs = sorted(set(zip(indices, values)), key=lambda x: x[0])
            indices = [idx for idx, _ in unique_pairs]
            values = [val for _, val in unique_pairs]
            
            sparse_vectors.append({"indices": indices, "values": values})
        
        logger.debug(f"Generated {len(sparse_vectors)} sparse vectors for BM25")
        return sparse_vectors

    def embed_query(self, query: str) -> Dict[str, Any]:
        """Single query embedding."""
        return self.encode([query], return_sparse=True)[0]