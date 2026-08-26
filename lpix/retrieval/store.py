"""
lpix — Vector Store module

Uses ChromaDB for persistent local storage of bug embeddings.

Why ChromaDB:
- Pure Python, zero infra (no Docker, no server process)
- Persistent by default (SQLite + HNSW index on disk)
- Native metadata filtering (status, importance, tags, project)
- Built-in cosine similarity
- At 10k vectors: <100ms query latency, ~50MB on disk
- Simple Python API, well-maintained

Why not:
- FAISS: No persistence, no metadata filtering, manual serialization
- Qdrant: Needs a server process or separate library for embedded mode
- sqlite-vec: Promising but very new (2024), fewer features
- Pinecone/Weaviate: Cloud-only or heavy infra

ChromaDB gotchas (2024/2025):
- v0.4+ changed the API significantly from v0.3 — use get_or_create_collection()
- embeddings must be list[list[float]], not numpy arrays — convert explicitly
- metadatas cannot contain None values — use empty string instead
- tags (list) must be stored as comma-joined string for metadata filtering
- ChromaDB's built-in embedding function is slow — always pass embeddings directly
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from lpix.ingestion.launchpad import BugChunk

logger = logging.getLogger(__name__)

# Default storage location
DEFAULT_DB_PATH = Path.home() / ".lpix" / "chroma_db"


class BugVectorStore:
    """
    ChromaDB-backed vector store for Launchpad bug chunks.
    
    Usage:
        store = BugVectorStore()
        store.upsert_chunks(chunks, embeddings)
        results = store.query("memory leak in nova compute", n_results=10)
    """
    
    COLLECTION_NAME = "launchpad_bugs"
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
    
    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb not installed. Run: uv pip install chromadb")
        
        logger.info(f"Opening ChromaDB at {self.db_path}")
        self._client = chromadb.PersistentClient(path=str(self.db_path))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # use cosine distance
        )
        count = self._collection.count()
        logger.info(f"Collection '{self.COLLECTION_NAME}' has {count} vectors")
        return self._collection
    
    def upsert_chunks(
        self,
        chunks: list[BugChunk],
        embeddings: np.ndarray,
    ) -> int:
        """
        Upsert bug chunks with their embeddings.
        Uses upsert (not add) so re-ingestion is idempotent.
        
        Args:
            chunks: List of BugChunk objects
            embeddings: np.ndarray shape (n, dim)
        
        Returns:
            Number of chunks upserted
        """
        collection = self._get_collection()
        
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        
        # ChromaDB requires list[list[float]], not numpy
        emb_list = embeddings.tolist()
        
        # Metadata: ChromaDB cannot store None, list, or nested dict
        # Convert tags list → comma-separated string
        metadatas = []
        for c in chunks:
            metadatas.append({
                "bug_id": c.bug_id,
                "bug_url": c.bug_url,
                "project": c.project,
                "title": c.title,
                "status": c.status,
                "importance": c.importance,
                "tags": ",".join(c.tags) if c.tags else "",
                "chunk_type": c.chunk_type,
                "comment_index": c.comment_index,
                "author": c.author or "",
                "date_created": c.date_created,
                "date_last_updated": c.date_last_updated,
            })
        
        # Batch upsert in chunks of 500 (ChromaDB recommend < 1000/batch)
        BATCH = 500
        for i in range(0, len(ids), BATCH):
            collection.upsert(
                ids=ids[i:i+BATCH],
                documents=documents[i:i+BATCH],
                embeddings=emb_list[i:i+BATCH],
                metadatas=metadatas[i:i+BATCH],
            )
        
        logger.info(f"Upserted {len(ids)} chunks")
        return len(ids)
    
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query the vector store for similar chunks.
        
        Args:
            query_embedding: np.ndarray shape (dim,)
            n_results: How many results to return (fetch more for reranking)
            where: ChromaDB metadata filter, e.g.:
                   {"status": {"$in": ["Confirmed", "Triaged"]}}
                   {"importance": {"$in": ["Critical", "High"]}}
                   {"project": "nova"}
        
        Returns:
            List of dicts with: chunk_id, text, metadata, distance, score
        
        ChromaDB distance metric:
            With hnsw:space=cosine, lower distance = more similar
            score = 1 - distance (range 0-1, higher = better)
        """
        collection = self._get_collection()
        
        query_kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(n_results, collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where
        
        results = collection.query(**query_kwargs)
        
        hits = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "score": 1.0 - dist,  # cosine similarity
            })
        
        return hits
    
    def get_all_texts(self) -> tuple[list[str], list[str]]:
        """
        Return all document texts and IDs (for BM25 index building).
        
        Returns:
            (ids, texts) tuple
        """
        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            return [], []
        
        # Fetch all in batches
        all_ids, all_texts = [], []
        BATCH = 1000
        offset = 0
        while offset < count:
            result = collection.get(
                limit=BATCH,
                offset=offset,
                include=["documents"],
            )
            all_ids.extend(result["ids"])
            all_texts.extend(result["documents"])
            offset += BATCH
        
        return all_ids, all_texts
    
    def count(self) -> int:
        return self._get_collection().count()
    
    def delete_by_bug_id(self, bug_id: int):
        """Delete all chunks for a specific bug (e.g. on re-ingest)."""
        collection = self._get_collection()
        collection.delete(where={"bug_id": bug_id})
