"""
lpix Hermes Plugin — register(ctx) entry point.

Installation:
    ln -s /path/to/lpix/lpix/tools ~/.hermes/plugins/lpix
    hermes plugins enable lpix

Or copy the whole tools/ directory to ~/.hermes/plugins/lpix/

The register(ctx) function is called by Hermes at startup.
It wires three tools:
  - search_launchpad_bugs  : hybrid RAG search over indexed bug chunks
  - get_launchpad_bug      : fetch full stored content for a specific bug ID
  - sync_launchpad_project : ingest/update a project's bugs into the local index

Design notes:
  - All heavy imports (chromadb, sentence-transformers) are deferred to first call
  - Handlers return JSON strings (Hermes requirement)
  - _retriever is a module-level singleton (lazy init on first search)
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_retriever = None


def _get_retriever():
    """Lazy-initialize the full retrieval stack on first tool call."""
    global _retriever
    if _retriever is not None:
        return _retriever

    from lpix.embedding.model import EmbeddingModel
    from lpix.retrieval.store import BugVectorStore
    from lpix.retrieval.hybrid import BM25Index, CrossEncoderReranker, HybridRetriever

    store = BugVectorStore()
    embedding_model = EmbeddingModel()
    bm25 = BM25Index()
    loaded = bm25.load()
    if not loaded:
        logger.warning("No BM25 index found — building from store")
        bm25.build(*store.get_all_texts())

    reranker = CrossEncoderReranker()
    _retriever = HybridRetriever(
        store=store,
        embedding_model=embedding_model,
        bm25_index=bm25,
        reranker=reranker,
    )
    return _retriever


def _deduplicate_by_bug(results: list[dict], max_per_bug: int = 2) -> list[dict]:
    """
    Limit results to max_per_bug chunks per bug_id.
    Prevents one verbose bug dominating the context window.
    """
    seen: dict[int, int] = {}
    out = []
    for r in results:
        bug_id = r.get("metadata", {}).get("bug_id", 0)
        if seen.get(bug_id, 0) < max_per_bug:
            out.append(r)
            seen[bug_id] = seen.get(bug_id, 0) + 1
    return out


def _format_results(query: str, results: list[dict]) -> str:
    """Format search results as a JSON string for Hermes."""
    output = []
    for r in results:
        m = r.get("metadata", {})
        output.append({
            "bug_id": m.get("bug_id"),
            "url": m.get("bug_url", m.get("url", "")),
            "title": m.get("title", ""),
            "chunk_type": m.get("chunk_type", ""),
            "status": m.get("status", ""),
            "importance": m.get("importance", ""),
            "tags": m.get("tags", ""),
            "author": m.get("author", ""),
            "date": m.get("date_last_updated", m.get("date", "")),
            "text_preview": r.get("text", "")[:600],
        })
    return json.dumps({"query": query, "results": output, "count": len(output)}, indent=2)


def register(ctx):
    """
    Register lpix tools with Hermes Agent.
    Called automatically by Hermes when the plugin is enabled.
    """

    # ── Tool 1: search_launchpad_bugs ─────────────────────────────────────────

    ctx.register_tool(
        name="search_launchpad_bugs",
        toolset="lpix",
        schema={
            "name": "search_launchpad_bugs",
            "description": (
                "Search Launchpad bug tracker tickets using semantic + keyword hybrid search. "
                "Returns the most relevant bug reports, comments, and status information. "
                "Use for: finding bugs by symptom, error message, component, package, or keyword. "
                "Searches bug titles, descriptions, and all comments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language or keyword search query. "
                            "Examples: 'snapd crash on ARM', 'network-manager DNS leak', "
                            "'OOM killer nova-compute', 'bug 1234567'"
                        ),
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (1–10, default: 5)",
                        "default": 5,
                    },
                    "status_filter": {
                        "type": "string",
                        "description": (
                            "Optional: comma-separated statuses to include. "
                            "Options: New, Incomplete, Confirmed, Triaged, In Progress, "
                            "Fix Committed, Fix Released"
                        ),
                    },
                    "importance_filter": {
                        "type": "string",
                        "description": (
                            "Optional: comma-separated importances. "
                            "Options: Critical, High, Medium, Low, Undecided"
                        ),
                    },
                    "project_filter": {
                        "type": "string",
                        "description": "Optional: filter by project name (e.g. 'nova', 'ubuntu')",
                    },
                },
                "required": ["query"],
            },
        },
        handler=_handle_search,
    )

    # ── Tool 2: get_launchpad_bug ─────────────────────────────────────────────

    ctx.register_tool(
        name="get_launchpad_bug",
        toolset="lpix",
        schema={
            "name": "get_launchpad_bug",
            "description": (
                "Retrieve the full indexed content (description + all comments) "
                "for a specific Launchpad bug by its numeric ID. "
                "Use when you already know the bug number and want full details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bug_id": {
                        "type": "integer",
                        "description": "The numeric Launchpad bug ID (e.g. 1234567)",
                    },
                },
                "required": ["bug_id"],
            },
        },
        handler=_handle_get_bug,
    )

    # ── Tool 3: sync_launchpad_project ────────────────────────────────────────

    ctx.register_tool(
        name="sync_launchpad_project",
        toolset="lpix",
        schema={
            "name": "sync_launchpad_project",
            "description": (
                "Sync/ingest Launchpad bug reports into the local search index. "
                "Run with full_resync=false (default) for incremental updates. "
                "Run with full_resync=true for first-time setup or to refresh stale data. "
                "The index persists between Hermes sessions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Launchpad project slug (e.g. 'ubuntu', 'nova', 'snapd')",
                    },
                    "full_resync": {
                        "type": "boolean",
                        "description": "Re-ingest all bugs (ignores last sync time). Default: false.",
                        "default": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max bugs to fetch (omit for all, use small number for testing)",
                    },
                },
                "required": ["project"],
            },
        },
        handler=_handle_sync,
    )

    logger.info("lpix plugin registered: search_launchpad_bugs, get_launchpad_bug, sync_launchpad_project")


# ── Handlers (module-level so they're importable and testable) ────────────────

def _handle_search(params: dict, **_) -> str:
    query = (params.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"})

    n = max(1, min(10, int(params.get("n_results", 5))))

    status_filter = None
    if params.get("status_filter"):
        status_filter = [s.strip() for s in params["status_filter"].split(",") if s.strip()]

    importance_filter = None
    if params.get("importance_filter"):
        importance_filter = [i.strip() for i in params["importance_filter"].split(",") if i.strip()]

    project_filter = params.get("project_filter") or None

    try:
        retriever = _get_retriever()
        if retriever.store.count() == 0:
            return json.dumps({
                "error": "Index is empty. Run sync_launchpad_project first.",
                "hint": "sync_launchpad_project(project='ubuntu', limit=100) to test",
            })

        results = retriever.search(
            query=query,
            n_final=n * 2,        # over-fetch then deduplicate
            filter_status=status_filter,
            filter_importance=importance_filter,
            filter_project=project_filter,
            use_reranker=True,
        )
        results = _deduplicate_by_bug(results, max_per_bug=2)[:n]
        return _format_results(query, results)

    except Exception as e:
        logger.exception("search_launchpad_bugs failed")
        return json.dumps({"error": str(e)})


def _handle_get_bug(params: dict, **_) -> str:
    bug_id = params.get("bug_id")
    if not bug_id:
        return json.dumps({"error": "bug_id is required"})
    try:
        from lpix.tools.search_tool import get_launchpad_bug
        return json.dumps({"content": get_launchpad_bug(int(bug_id))})
    except Exception as e:
        logger.exception("get_launchpad_bug failed")
        return json.dumps({"error": str(e)})


def _handle_sync(params: dict, **_) -> str:
    project = (params.get("project") or "").strip()
    if not project:
        return json.dumps({"error": "project is required"})
    try:
        from lpix.sync.ingest import ingest_project
        bugs, chunks = ingest_project(
            project_name=project,
            full_resync=bool(params.get("full_resync", False)),
            limit=params.get("limit"),
        )
        # Invalidate cached retriever so next search uses fresh index
        global _retriever
        _retriever = None
        return json.dumps({
            "ok": True,
            "project": project,
            "bugs_ingested": bugs,
            "chunks_indexed": chunks,
        })
    except Exception as e:
        logger.exception("sync_launchpad_project failed")
        return json.dumps({"error": str(e)})
