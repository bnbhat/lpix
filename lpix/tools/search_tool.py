"""
lpix — Hermes Agent Tool

Exposes the lpix search as a Hermes tool.

Installation:
    Copy or symlink this file to your Hermes plugins directory:
    ~/.hermes/plugins/lpix_tool.py

    Then restart Hermes (or use: hermes plugins reload)

How Hermes tools work:
    - Each public function with a docstring becomes a tool
    - The docstring IS the tool description shown to the LLM
    - Type annotations are used to generate the JSON schema
    - Return a string (Hermes will include it in the conversation)

Design choices:
    - Lazy initialization (model + store loaded on first call)
    - Returns structured markdown for readable LLM consumption
    - Includes metadata (status, importance, URL) for actionable responses
    - n_results default=5 is good for most queries (reranker picks best)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# --- Lazy globals (initialized on first tool call) ---
_retriever = None


def _get_retriever():
    """Lazy-initialize the retriever (loads model + opens DB)."""
    global _retriever
    if _retriever is not None:
        return _retriever
    
    from lpix.embedding.model import EmbeddingModel
    from lpix.retrieval.store import BugVectorStore
    from lpix.retrieval.hybrid import BM25Index, CrossEncoderReranker, HybridRetriever
    
    store = BugVectorStore()
    embedding_model = EmbeddingModel()
    bm25 = BM25Index()
    
    # Try to load cached BM25 index
    loaded = bm25.load()
    if not loaded:
        logger.warning("No BM25 index found — building from store (may be slow on first use)")
        bm25.build(*store.get_all_texts())
    
    reranker = CrossEncoderReranker()
    
    _retriever = HybridRetriever(
        store=store,
        embedding_model=embedding_model,
        bm25_index=bm25,
        reranker=reranker,
    )
    return _retriever


def search_launchpad_bugs(
    query: str,
    n_results: int = 5,
    status_filter: Optional[str] = None,
    importance_filter: Optional[str] = None,
    project_filter: Optional[str] = None,
) -> str:
    """
    Search Launchpad bug tracker tickets using natural language.
    
    Searches through bug titles, descriptions, and comments using semantic
    similarity + keyword matching (hybrid BM25 + dense retrieval).
    
    Args:
        query: Natural language query, e.g.:
               "memory leak in nova compute when live migrating"
               "crash on startup after upgrading to 22.04"
               "bug 1234567"
               "OOM killer triggered"
        n_results: Number of results to return (1-10, default 5)
        status_filter: Comma-separated statuses to include, e.g.:
                       "Confirmed,Triaged" or "Fix Released"
                       Options: New, Incomplete, Confirmed, Triaged,
                                In Progress, Fix Committed, Fix Released
        importance_filter: Comma-separated importances, e.g.: "Critical,High"
                           Options: Critical, High, Medium, Low, Undecided
        project_filter: Filter by project name, e.g.: "nova" or "ubuntu"
    
    Returns:
        Formatted results with bug URLs, status, and relevant excerpts.
    """
    if not query or not query.strip():
        return "Error: query cannot be empty."
    
    n_results = max(1, min(10, n_results))
    
    # Parse filter args
    filter_status = None
    if status_filter:
        filter_status = [s.strip() for s in status_filter.split(",") if s.strip()]
    
    filter_importance = None
    if importance_filter:
        filter_importance = [i.strip() for i in importance_filter.split(",") if i.strip()]
    
    try:
        retriever = _get_retriever()
        
        if retriever.store.count() == 0:
            return (
                "The lpix vector store is empty. "
                "Run the ingest command first:\n"
                "  python -m lpix.sync.ingest --project <project-name>"
            )
        
        results = retriever.search(
            query=query,
            n_final=n_results,
            filter_status=filter_status,
            filter_importance=filter_importance,
            filter_project=project_filter,
            use_reranker=True,
        )
        
    except Exception as e:
        logger.exception("Search failed")
        return f"Search error: {e}"
    
    if not results:
        return f"No results found for: '{query}'"
    
    # Format results as readable markdown
    lines = [f"## Launchpad Bug Search: '{query}'\n"]
    lines.append(f"Found {len(results)} relevant results:\n")
    
    seen_bugs = set()  # deduplicate by bug ID (multiple chunks per bug)
    
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        bug_id = meta.get("bug_id", "?")
        
        # Show best chunk per bug (results are already sorted by relevance)
        is_first_for_bug = bug_id not in seen_bugs
        seen_bugs.add(bug_id)
        
        score = r.get("rerank_score") or r.get("rrf_score") or r.get("score", 0)
        chunk_type = meta.get("chunk_type", "")
        comment_idx = meta.get("comment_index", 0)
        location = "description" if comment_idx == 0 else f"comment #{comment_idx}"
        
        lines.append(f"### {i}. Bug #{bug_id}: {meta.get('title', 'Unknown')}")
        lines.append(
            f"**Status:** {meta.get('status', '?')} | "
            f"**Importance:** {meta.get('importance', '?')} | "
            f"**Match in:** {location}"
        )
        
        tags = meta.get("tags", "")
        if tags:
            lines.append(f"**Tags:** {tags}")
        
        lines.append(f"**URL:** {meta.get('bug_url', '')}")
        
        # Show the relevant text excerpt (first 400 chars)
        text = r.get("text", "")
        # Skip the header line (already shown above)
        text_lines = text.split("\n")
        body = "\n".join(l for l in text_lines if not l.startswith("Bug #")).strip()
        excerpt = body[:400] + ("..." if len(body) > 400 else "")
        if excerpt:
            lines.append(f"\n> {excerpt.replace(chr(10), chr(10) + '> ')}")
        
        lines.append("")
    
    return "\n".join(lines)


def get_launchpad_bug(bug_id: int) -> str:
    """
    Retrieve all indexed content for a specific Launchpad bug by ID.
    
    Fetches all stored chunks (description + all comments) for a bug
    from the local vector store index.
    
    Args:
        bug_id: The numeric Launchpad bug ID (e.g. 1234567)
    
    Returns:
        Full bug content from the index, or a message if not found.
    """
    try:
        retriever = _get_retriever()
        collection = retriever.store._get_collection()
        
        results = collection.get(
            where={"bug_id": bug_id},
            include=["documents", "metadatas"],
        )
        
        if not results["ids"]:
            return (
                f"Bug #{bug_id} not found in the index. "
                f"It may not have been ingested yet, or may not exist.\n"
                f"Direct URL: https://bugs.launchpad.net/bugs/{bug_id}"
            )
        
        # Sort chunks by comment index
        chunks = sorted(
            zip(results["ids"], results["documents"], results["metadatas"]),
            key=lambda x: x[2].get("comment_index", 0),
        )
        
        meta = chunks[0][2]
        lines = [
            f"# Bug #{bug_id}: {meta.get('title', 'Unknown')}",
            f"**Status:** {meta.get('status')} | **Importance:** {meta.get('importance')}",
            f"**URL:** {meta.get('bug_url')}",
            f"**Tags:** {meta.get('tags', '')}",
            "",
        ]
        
        for chunk_id, text, m in chunks:
            idx = m.get("comment_index", 0)
            author = m.get("author", "")
            label = "**Description**" if idx == 0 else f"**Comment #{idx}**{' by ' + author if author else ''}"
            lines.append(f"### {label}")
            # Extract just the body (skip the "Bug #..." header line)
            body_lines = [l for l in text.split("\n") if not l.startswith("Bug #")]
            lines.append("\n".join(body_lines).strip())
            lines.append("")
        
        return "\n".join(lines)
    
    except Exception as e:
        logger.exception("get_launchpad_bug failed")
        return f"Error fetching bug #{bug_id}: {e}"


def sync_launchpad_project(
    project: str,
    full_resync: bool = False,
    limit: Optional[int] = None,
) -> str:
    """
    Sync/ingest a Launchpad project's bugs into the local search index.
    
    Fetches bug reports and comments from Launchpad and stores them
    in the local vector store for semantic search. Automatically does
    incremental sync (only fetches bugs changed since last sync).
    
    Args:
        project: Launchpad project name, e.g. "ubuntu", "nova", "neutron"
        full_resync: If True, re-ingests all bugs (ignores last sync time).
                     Use when you want to refresh stale data.
        limit: Max number of bugs to fetch (None = all, use for testing)
    
    Returns:
        Summary of ingest results.
    """
    try:
        from lpix.sync.ingest import ingest_project
        
        bugs, chunks = ingest_project(
            project_name=project,
            full_resync=full_resync,
            limit=limit,
        )
        
        return (
            f"✓ Sync complete for project '{project}'\n"
            f"  Bugs ingested: {bugs}\n"
            f"  Chunks indexed: {chunks}\n"
            f"\nYou can now search with: search_launchpad_bugs(query='...')"
        )
    except Exception as e:
        logger.exception("sync_launchpad_project failed")
        return f"Sync failed for '{project}': {e}"
