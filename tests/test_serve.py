"""
Tests for lpix FastAPI service — uses TestClient (no running server needed).
Run with: pytest tests/test_serve.py -v
"""
import json
import os
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault("API_KEY", "test-secret-key")

from fastapi.testclient import TestClient
from lpix.serve import app

client = TestClient(app, raise_server_exceptions=False)

HEADERS = {"Authorization": "Bearer test-secret-key"}
BAD_HEADERS = {"Authorization": "Bearer wrong-key"}


# ── Auth tests ────────────────────────────────────────────────────────────────

def test_health_no_auth():
    """GET /health should work without auth."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_search_requires_auth():
    """POST /search without auth returns 401."""
    resp = client.post("/search", json={"query": "test"})
    assert resp.status_code == 401


def test_search_wrong_key():
    """POST /search with wrong key returns 401."""
    resp = client.post("/search", json={"query": "test"}, headers=BAD_HEADERS)
    assert resp.status_code == 401


def test_status_requires_auth():
    resp = client.get("/status")
    assert resp.status_code == 401


def test_sync_requires_auth():
    resp = client.post("/sync", json={"project": "nova"})
    assert resp.status_code == 401


# ── Request validation ────────────────────────────────────────────────────────

def test_search_missing_query():
    """POST /search with no query body returns 422."""
    resp = client.post("/search", json={}, headers=HEADERS)
    assert resp.status_code == 422


def test_search_n_results_clamped():
    """n_results > 20 should be rejected (ge=1, le=20 in schema)."""
    resp = client.post("/search", json={"query": "test", "n_results": 100}, headers=HEADERS)
    assert resp.status_code == 422


def test_sync_missing_project():
    resp = client.post("/sync", json={}, headers=HEADERS)
    assert resp.status_code == 422


# ── Search with mocked retriever ──────────────────────────────────────────────

def make_mock_result(bug_id=1001, chunk_type="description"):
    return {
        "chunk_id": f"bug-{bug_id}-comment-0",
        "text": f"Bug #{bug_id}: Test crash\nThis is a test bug description.",
        "metadata": {
            "bug_id": bug_id,
            "bug_url": f"https://bugs.launchpad.net/bugs/{bug_id}",
            "title": "Test crash",
            "chunk_type": chunk_type,
            "status": "Confirmed",
            "importance": "High",
            "tags": "crash,regression",
            "author": "testuser",
            "date_last_updated": "2024-06-01T00:00:00+00:00",
        },
        "score": 0.92,
        "rerank_score": 0.87,
    }


def test_search_returns_results():
    """POST /search with a valid query returns structured results."""
    mock_results = [make_mock_result(1001), make_mock_result(1002)]

    with patch("lpix.serve.get_retriever") as mock_get_ret:
        mock_retriever = MagicMock()
        mock_retriever.store.count.return_value = 500
        mock_retriever.search.return_value = mock_results
        mock_get_ret.return_value = mock_retriever

        resp = client.post(
            "/search",
            json={"query": "crash on startup", "n_results": 5},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "crash on startup"
    assert data["count"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["bug_id"] == 1001
    assert data["results"][0]["status"] == "Confirmed"
    assert data["results"][0]["importance"] == "High"
    assert len(data["results"][0]["text_preview"]) <= 600


def test_search_empty_store_returns_503():
    """POST /search when index is empty returns 503."""
    with patch("lpix.serve.get_retriever") as mock_get_ret:
        mock_retriever = MagicMock()
        mock_retriever.store.count.return_value = 0
        mock_get_ret.return_value = mock_retriever

        resp = client.post(
            "/search",
            json={"query": "anything"},
            headers=HEADERS,
        )

    assert resp.status_code == 503
    assert "empty" in resp.json()["detail"].lower()


def test_search_deduplication():
    """Multiple chunks from the same bug should be deduplicated to max 2."""
    # 3 chunks from bug 1001, 1 from bug 1002
    mock_results = [
        make_mock_result(1001, "description"),
        make_mock_result(1001, "comment"),
        make_mock_result(1001, "comment"),   # third — should be dropped
        make_mock_result(1002, "description"),
    ]

    with patch("lpix.serve.get_retriever") as mock_get_ret:
        mock_retriever = MagicMock()
        mock_retriever.store.count.return_value = 500
        mock_retriever.search.return_value = mock_results
        mock_get_ret.return_value = mock_retriever

        resp = client.post(
            "/search",
            json={"query": "test", "n_results": 10},
            headers=HEADERS,
        )

    data = resp.json()
    assert data["count"] == 3  # 2 from bug 1001 + 1 from bug 1002
    bug1001_chunks = [r for r in data["results"] if r["bug_id"] == 1001]
    assert len(bug1001_chunks) == 2


def test_search_filter_params_passed_through():
    """status_filter and importance_filter should reach the retriever."""
    with patch("lpix.serve.get_retriever") as mock_get_ret:
        mock_retriever = MagicMock()
        mock_retriever.store.count.return_value = 100
        mock_retriever.search.return_value = []
        mock_get_ret.return_value = mock_retriever

        resp = client.post(
            "/search",
            json={
                "query": "nova crash",
                "status_filter": "Confirmed,Triaged",
                "importance_filter": "Critical",
                "project_filter": "nova",
            },
            headers=HEADERS,
        )

    assert resp.status_code == 200
    call_kwargs = mock_retriever.search.call_args[1]
    assert call_kwargs["filter_status"] == ["Confirmed", "Triaged"]
    assert call_kwargs["filter_importance"] == ["Critical"]
    assert call_kwargs["filter_project"] == "nova"


# ── Sync endpoint ─────────────────────────────────────────────────────────────

def test_sync_calls_ingest_project():
    """POST /sync should call ingest_project with correct args."""
    with patch("lpix.serve.ingest_project") as mock_ingest, \
         patch("lpix.serve.BugVectorStore") as mock_store_cls:

        mock_ingest.return_value = (42, 380)
        mock_store = MagicMock()
        mock_store.count.return_value = 380
        mock_store_cls.return_value = mock_store

        resp = client.post(
            "/sync",
            json={"project": "nova", "full_resync": True, "limit": 50},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["project"] == "nova"
    assert data["bugs_ingested"] == 42
    assert data["chunks_indexed"] == 380

    mock_ingest.assert_called_once_with(
        project_name="nova",
        full_resync=True,
        limit=50,
    )


def test_sync_invalidates_retriever_cache():
    """After sync, the retriever singleton should be reset."""
    import lpix.serve as serve_module

    # Plant a fake retriever
    serve_module._retriever = MagicMock()

    with patch("lpix.serve.ingest_project", return_value=(10, 80)), \
         patch("lpix.serve.BugVectorStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store.count.return_value = 80
        mock_store_cls.return_value = mock_store

        client.post("/sync", json={"project": "ubuntu"}, headers=HEADERS)

    assert serve_module._retriever is None


# ── Status endpoint ───────────────────────────────────────────────────────────

def test_status_returns_vector_count():
    with patch("lpix.serve.BugVectorStore") as mock_store_cls, \
         patch("lpix.serve.SyncState") as mock_state_cls:

        mock_store = MagicMock()
        mock_store.count.return_value = 1234
        mock_store_cls.return_value = mock_store

        mock_state = MagicMock()
        mock_state._data = {
            "ubuntu": {"last_sync": "2024-06-01T00:00:00+00:00", "bug_count": 500}
        }
        mock_state_cls.return_value = mock_state

        resp = client.get("/status", headers=HEADERS)

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["total_vectors"] == 1234
    assert "ubuntu" in data["projects"]
