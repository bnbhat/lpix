"""
lpix — Embedding module

Wraps sentence-transformers for encoding bug chunks.

Model choice: BAAI/bge-small-en-v1.5
- 33MB on disk, 384-dim embeddings
- MTEB score ~62.x (outperforms all-MiniLM-L6-v2 at 56.x on retrieval tasks)
- Supports BGE instruction prefix for queries (improves recall)
- Fast CPU inference (~5ms/chunk on modern CPU)
- No API key needed, works fully offline

Alternatives considered:
- all-MiniLM-L6-v2: 22MB, MTEB 56.x — smaller but lower quality
- all-mpnet-base-v2: 420MB, MTEB 57.x — larger, not worth it
- bge-base-en-v1.5: 110MB, MTEB 63.x — good upgrade if quality matters
- OpenAI text-embedding-3-small: best quality, but requires API key + cost
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Union

logger = logging.getLogger(__name__)

# BGE models benefit from an instruction prefix on QUERIES (not on documents)
# This is a key gotcha with BGE — applying it to documents hurts performance
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """
    Thin wrapper around sentence-transformers for lpix.
    
    Usage:
        model = EmbeddingModel()
        
        # Encode documents (no prefix)
        doc_embeddings = model.encode_documents(["text1", "text2"])
        
        # Encode a search query (with BGE prefix)
        query_embedding = model.encode_query("crashes on startup")
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        batch_size: int = 64,
        normalize: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model ID
            device: "cpu" or "cuda" or "mps"
            batch_size: Encoding batch size (tune for memory vs speed)
            normalize: L2-normalize embeddings (required for cosine similarity 
                      via dot product, which is faster than scipy cosine)
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
    
    def _load(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: uv pip install sentence-transformers"
            )
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Model loaded. Embedding dim: {self._model.get_sentence_embedding_dimension()}")
    
    @property
    def dim(self) -> int:
        """Embedding dimension."""
        self._load()
        return self._model.get_sentence_embedding_dimension()
    
    def encode_documents(
        self, texts: list[str], show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode document texts (no instruction prefix).
        
        Returns:
            np.ndarray of shape (n, dim), float32, L2-normalized if normalize=True
        """
        self._load()
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single search query with BGE instruction prefix.
        
        Returns:
            np.ndarray of shape (dim,), float32, L2-normalized if normalize=True
        """
        self._load()
        
        # Apply BGE prefix only for BGE models
        if "bge" in self.model_name.lower():
            query = BGE_QUERY_PREFIX + query
        
        return self._model.encode(
            query,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
    
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Batch-encode multiple queries."""
        self._load()
        if "bge" in self.model_name.lower():
            queries = [BGE_QUERY_PREFIX + q for q in queries]
        return self._model.encode(
            queries,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )


# Module-level singleton for convenience
_default_model: EmbeddingModel | None = None


def get_default_model() -> EmbeddingModel:
    global _default_model
    if _default_model is None:
        _default_model = EmbeddingModel()
    return _default_model
