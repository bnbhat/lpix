"""
lpix — FastAPI service layer

Exposes lpix search and sync over HTTP so Hermes (or any client)
can call it without direct access to Launchpad credentials, ChromaDB
files, or the embedding model.

Endpoints:
    POST /search          — hybrid RAG search
    POST /sync            — ingest/update a Launchpad project
    GET  /status          — health check + index stats
    GET  /bug/{bug_id}    — fetch full stored content for one bug

Auth:
    All endpoints (except GET /health) require:
        Authorization: Bearer <API_KEY>
    Set API_KEY env var in the container. Hermes only needs this one secret.

Usage:
    uvicorn lpix.serve:app --host 0.0.0.0 --port 8080

    # or via CLI:
    python -m lpix.serve
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from lpix.sync.ingest import ingest_project
from lpix.retrieval.store import BugVectorStore
from lpix.sync.state import SyncState

logger = logging.getLogger(__name__)

# ── Auth ──────────────────────────────────────────────────────────────────────

_security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(_security)):
    """Bearer token auth. Reads API_KEY from environment."""
    expected = os.environ.get("API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="API_KEY not set in server environment"
        )
    if credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


# ── Lazy retriever singleton ──────────────────────────────────────────────────

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    from lpix.embedding.model import EmbeddingModel
    from lpix.retrieval.store import BugVectorStore
    from lpix.retrieval.hybrid import BM25Index, CrossEncoderReranker, HybridRetriever

    store = BugVectorStore()
    embedding_model = EmbeddingModel()
    bm25 = BM25Index()
    if not bm25.load():
        logger.warning("No BM25 index on disk — will build on first sync")
    reranker = CrossEncoderReranker()

    _retriever = HybridRetriever(
        store=store,
        embedding_model=embedding_model,
        bm25_index=bm25,
        reranker=reranker,
    )
    return _retriever


def invalidate_retriever():
    """Call after sync to force re-init with fresh BM25 index."""
    global _retriever
    _retriever = None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the retriever at startup if the index already exists."""
    try:
        r = get_retriever()
        count = r.store.count()
        logger.info(f"lpix ready — {count} vectors in store")
    except Exception as e:
        logger.warning(f"Retriever warm-up skipped: {e}")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="lpix",
    description="Launchpad Bug Tracker RAG API",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    n_results: int = Field(5, ge=1, le=20, description="Number of results")
    status_filter: Optional[str] = Field(
        None,
        description="Comma-separated statuses: New, Confirmed, Triaged, In Progress, Fix Released, ..."
    )
    importance_filter: Optional[str] = Field(
        None,
        description="Comma-separated importances: Critical, High, Medium, Low"
    )
    project_filter: Optional[str] = Field(
        None,
        description="Filter by project name e.g. 'nova'"
    )


class BugResult(BaseModel):
    bug_id: int
    url: str
    title: str
    chunk_type: str       # "description" | "comment"
    status: str
    importance: str
    tags: str
    author: str
    date: str
    text_preview: str     # first 600 chars of the chunk


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[BugResult]


class SyncRequest(BaseModel):
    project: str = Field(..., description="Launchpad project slug e.g. 'ubuntu', 'nova'")
    full_resync: bool = Field(False, description="Re-ingest all bugs, ignoring last sync time")
    limit: Optional[int] = Field(None, description="Max bugs to fetch (omit for all)")


class SyncResponse(BaseModel):
    ok: bool
    project: str
    bugs_ingested: int
    chunks_indexed: int
    total_chunks: int


class StatusResponse(BaseModel):
    ok: bool
    total_vectors: int
    projects: dict        # {project: {last_sync, bug_count}}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Unauthenticated health check — for Docker/k8s liveness probes."""
    return {"ok": True}


@app.get("/status", response_model=StatusResponse)
def status(_: str = Depends(verify_api_key)):
    """Index stats and per-project sync state."""
    store = BugVectorStore()
    state = SyncState()

    projects = {}
    for project, info in state._data.items():
        projects[project] = {
            "last_sync": info.get("last_sync"),
            "bug_count": info.get("bug_count", 0),
        }

    return StatusResponse(
        ok=True,
        total_vectors=store.count(),
        projects=projects,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, _: str = Depends(verify_api_key)):
    """Hybrid BM25 + dense search with cross-encoder reranking."""
    from lpix.tools import _deduplicate_by_bug

    status_filter = None
    if req.status_filter:
        status_filter = [s.strip() for s in req.status_filter.split(",") if s.strip()]

    importance_filter = None
    if req.importance_filter:
        importance_filter = [i.strip() for i in req.importance_filter.split(",") if i.strip()]

    try:
        retriever = get_retriever()

        if retriever.store.count() == 0:
            raise HTTPException(
                status_code=503,
                detail="Index is empty — run POST /sync first"
            )

        results = retriever.search(
            query=req.query,
            n_final=req.n_results * 2,
            filter_status=status_filter,
            filter_importance=importance_filter,
            filter_project=req.project_filter,
            use_reranker=True,
        )
        results = _deduplicate_by_bug(results, max_per_bug=2)[:req.n_results]

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))

    bug_results = []
    for r in results:
        m = r.get("metadata", {})
        bug_results.append(BugResult(
            bug_id=int(m.get("bug_id", 0)),
            url=m.get("bug_url", m.get("url", "")),
            title=m.get("title", ""),
            chunk_type=m.get("chunk_type", ""),
            status=m.get("status", ""),
            importance=m.get("importance", ""),
            tags=m.get("tags", ""),
            author=m.get("author", ""),
            date=m.get("date_last_updated", m.get("date", "")),
            text_preview=r.get("text", "")[:600],
        ))

    return SearchResponse(query=req.query, count=len(bug_results), results=bug_results)


@app.get("/bug/{bug_id}")
def get_bug(bug_id: int, _: str = Depends(verify_api_key)):
    """Fetch full indexed content (description + all comments) for one bug."""
    from lpix.tools.search_tool import get_launchpad_bug
    try:
        content = get_launchpad_bug(bug_id)
        return {"bug_id": bug_id, "content": content}
    except Exception as e:
        logger.exception("get_bug failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync", response_model=SyncResponse)
def sync(req: SyncRequest, _: str = Depends(verify_api_key)):
    """
    Ingest or incrementally update a Launchpad project.
    
    Uses Launchpad credentials from the container environment only.
    Hermes never needs to supply or see these credentials.
    
    On first run: set full_resync=true
    Subsequent runs: full_resync=false (default) for incremental updates
    """
    try:
        bugs, chunks = ingest_project(
            project_name=req.project,
            full_resync=req.full_resync,
            limit=req.limit,
        )
        invalidate_retriever()

        store = BugVectorStore()
        return SyncResponse(
            ok=True,
            project=req.project,
            bugs_ingested=bugs,
            chunks_indexed=chunks,
            total_chunks=store.count(),
        )
    except Exception as e:
        logger.exception("Sync failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "lpix.serve:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        reload=False,
    )
