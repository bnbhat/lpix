"""
lpix Hermes plugin — HTTP client mode

When lpix runs as a separate container, this replaces the in-process plugin.
Hermes only needs:
  - LPIX_URL  e.g. http://lpix:8080
  - LPIX_API_KEY  the shared secret

Nothing else. No Launchpad credentials, no ChromaDB, no embedding model.

Installation:
    # In Hermes .env:
    LPIX_URL=http://lpix:8080
    LPIX_API_KEY=your-secret-key

    # Link this as the Hermes plugin:
    ln -s /path/to/lpix/lpix/tools ~/.hermes/plugins/lpix
    hermes plugins enable lpix
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── HTTP client ───────────────────────────────────────────────────────────────

def _lpix_post(endpoint: str, payload: dict) -> dict:
    """
    POST to the lpix service.
    Reads LPIX_URL and LPIX_API_KEY from environment.
    """
    import urllib.request
    import urllib.error

    base_url = os.environ.get("LPIX_URL", "http://lpix:8080").rstrip("/")
    api_key = os.environ.get("LPIX_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "LPIX_API_KEY not set. Add it to Hermes .env:\n"
            "  LPIX_API_KEY=your-secret-key"
        )

    url = f"{base_url}{endpoint}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"lpix API error {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach lpix at {base_url}: {e.reason}\n"
            f"Is the lpix container running?"
        )


def _lpix_get(endpoint: str) -> dict:
    """GET from the lpix service."""
    import urllib.request
    import urllib.error

    base_url = os.environ.get("LPIX_URL", "http://lpix:8080").rstrip("/")
    api_key = os.environ.get("LPIX_API_KEY", "")

    url = f"{base_url}{endpoint}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"lpix API error {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach lpix at {base_url}: {e.reason}")


# ── Deduplication (client-side, same logic as server) ────────────────────────

def _deduplicate_by_bug(results: list[dict], max_per_bug: int = 2) -> list[dict]:
    seen: dict = {}
    out = []
    for r in results:
        bug_id = r.get("bug_id", 0)
        if seen.get(bug_id, 0) < max_per_bug:
            out.append(r)
            seen[bug_id] = seen.get(bug_id, 0) + 1
    return out


# ── Hermes plugin registration ────────────────────────────────────────────────

def register(ctx):
    """Register lpix HTTP tools with Hermes Agent."""

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
                            "Natural language or keyword query. "
                            "Examples: 'snapd crash on ARM', 'OOM killer nova-compute', "
                            "'network-manager DNS leak', 'bug 1234567'"
                        ),
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results (1–10, default 5)",
                        "default": 5,
                    },
                    "status_filter": {
                        "type": "string",
                        "description": (
                            "Optional comma-separated statuses: "
                            "New, Incomplete, Confirmed, Triaged, In Progress, "
                            "Fix Committed, Fix Released"
                        ),
                    },
                    "importance_filter": {
                        "type": "string",
                        "description": "Optional comma-separated importances: Critical, High, Medium, Low",
                    },
                    "project_filter": {
                        "type": "string",
                        "description": "Optional project name e.g. 'nova', 'ubuntu'",
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
                "Fetch the full indexed content (description + all comments) "
                "for a specific Launchpad bug by numeric ID. "
                "Use when you already know the bug number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bug_id": {
                        "type": "integer",
                        "description": "Numeric Launchpad bug ID e.g. 1234567",
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
                "Trigger a Launchpad bug ingest/sync on the lpix service. "
                "The service handles all Launchpad authentication internally. "
                "Use full_resync=false (default) for incremental updates, "
                "full_resync=true for first-time setup."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Launchpad project slug e.g. 'ubuntu', 'nova', 'snapd'",
                    },
                    "full_resync": {
                        "type": "boolean",
                        "description": "Re-ingest all bugs. Default false (incremental).",
                        "default": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max bugs to fetch (omit for all; use small number for testing)",
                    },
                },
                "required": ["project"],
            },
        },
        handler=_handle_sync,
    )

    # ── Tool 4: lpix_status ───────────────────────────────────────────────────

    ctx.register_tool(
        name="lpix_status",
        toolset="lpix",
        schema={
            "name": "lpix_status",
            "description": (
                "Check the lpix service status: total indexed vectors, "
                "per-project sync times, and whether the service is reachable."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        handler=_handle_status,
    )

    logger.info("lpix HTTP plugin registered (4 tools) — service: %s",
                os.environ.get("LPIX_URL", "http://lpix:8080"))


# ── Handlers ──────────────────────────────────────────────────────────────────

def _handle_search(params: dict, **_) -> str:
    query = (params.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"})

    payload = {
        "query": query,
        "n_results": max(1, min(10, int(params.get("n_results", 5)))),
    }
    if params.get("status_filter"):
        payload["status_filter"] = params["status_filter"]
    if params.get("importance_filter"):
        payload["importance_filter"] = params["importance_filter"]
    if params.get("project_filter"):
        payload["project_filter"] = params["project_filter"]

    try:
        return json.dumps(_lpix_post("/search", payload), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_get_bug(params: dict, **_) -> str:
    bug_id = params.get("bug_id")
    if not bug_id:
        return json.dumps({"error": "bug_id is required"})
    try:
        return json.dumps(_lpix_get(f"/bug/{int(bug_id)}"), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_sync(params: dict, **_) -> str:
    project = (params.get("project") or "").strip()
    if not project:
        return json.dumps({"error": "project is required"})

    payload: dict = {"project": project, "full_resync": bool(params.get("full_resync", False))}
    if params.get("limit"):
        payload["limit"] = int(params["limit"])

    try:
        return json.dumps(_lpix_post("/sync", payload), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_status(params: dict, **_) -> str:
    try:
        return json.dumps(_lpix_get("/status"), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
