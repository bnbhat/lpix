"""
Tests for lpix — run with: pytest tests/ -v
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from lpix.ingestion.launchpad import BugChunk, bug_to_chunks
from lpix.embedding.model import EmbeddingModel, BGE_QUERY_PREFIX
from lpix.retrieval.hybrid import BM25Index, reciprocal_rank_fusion
from lpix.tools import _deduplicate_by_bug


# ── Chunking tests ──────────────────────────────────────────────────────────

def make_bug_data(**kwargs):
    base = {
        "bug_id": 12345,
        "bug_url": "https://bugs.launchpad.net/bugs/12345",
        "title": "Test bug: crash on startup",
        "description": "The application crashes immediately on startup.",
        "status": "Confirmed",
        "importance": "High",
        "tags": ["crash", "regression"],
        "date_created": "2024-01-01T00:00:00+00:00",
        "date_last_updated": "2024-06-01T00:00:00+00:00",
        "project": "ubuntu",
        "messages": [],
    }
    base.update(kwargs)
    return base


def test_bug_to_chunks_description_only():
    """A bug with no comments produces one chunk (the description)."""
    bug = make_bug_data()
    chunks = bug_to_chunks(bug)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "description"
    assert chunks[0].chunk_id == "bug-12345-comment-0"
    assert "crash on startup" in chunks[0].text
    assert "Confirmed" in chunks[0].text
    assert "crash" in chunks[0].text  # tags included


def test_bug_to_chunks_with_comments():
    """A bug with 2 comments produces 3 chunks (description + 2 comments)."""
    msg1 = MagicMock()
    msg1.content = "I can reproduce this on Ubuntu 22.04."
    msg1.owner_link = "https://api.launchpad.net/1.0/~testuser"
    msg1.date_created = MagicMock()
    msg1.date_created.isoformat.return_value = "2024-01-02T00:00:00+00:00"

    msg2 = MagicMock()
    msg2.content = "Fixed in commit abc123."
    msg2.owner_link = "https://api.launchpad.net/1.0/~devuser"
    msg2.date_created = MagicMock()
    msg2.date_created.isoformat.return_value = "2024-01-03T00:00:00+00:00"

    bug = make_bug_data(messages=[MagicMock(), msg1, msg2])
    chunks = bug_to_chunks(bug)

    assert len(chunks) == 3
    assert chunks[0].chunk_type == "description"
    assert chunks[1].chunk_type == "comment"
    assert chunks[1].comment_index == 1
    assert "testuser" in chunks[1].author
    assert chunks[2].chunk_type == "comment"
    assert chunks[2].comment_index == 2


def test_chunk_text_truncation():
    """Long descriptions are truncated to max_comment_length."""
    long_desc = "X" * 5000
    bug = make_bug_data(description=long_desc)
    chunks = bug_to_chunks(bug, max_comment_length=2000)
    assert len(chunks[0].text) < 3000


def test_chunk_id_uniqueness():
    """Chunk IDs must be unique across a bug."""
    msg = MagicMock()
    msg.content = "A comment"
    msg.owner_link = "https://api.launchpad.net/1.0/~user"
    msg.date_created = MagicMock()
    msg.date_created.isoformat.return_value = "2024-01-02T00:00:00+00:00"

    bug = make_bug_data(messages=[MagicMock(), msg])
    chunks = bug_to_chunks(bug)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_metadata_completeness():
    """Each chunk must have all required metadata fields."""
    bug = make_bug_data()
    chunks = bug_to_chunks(bug)
    for chunk in chunks:
        assert chunk.bug_id == 12345
        assert chunk.project == "ubuntu"
        assert chunk.status == "Confirmed"
        assert chunk.importance == "High"
        assert chunk.tags == ["crash", "regression"]


# ── BM25 tests ──────────────────────────────────────────────────────────────

def test_bm25_build_and_search():
    texts = [
        "nova compute crash memory leak OOM",
        "neutron network timeout connectivity",
        "cinder volume attach fails",
        "nova live migration failure",
    ]
    ids = [f"chunk-{i}" for i in range(len(texts))]

    bm25 = BM25Index()
    bm25.build(ids, texts)

    results = bm25.search("nova crash", n=4)
    assert len(results) > 0
    top_ids = [r[0] for r in results]
    assert "chunk-0" in top_ids
    assert results[0][1] != 0


def test_bm25_empty_returns_empty():
    bm25 = BM25Index()
    results = bm25.search("anything")
    assert results == []


def test_bm25_snake_case_tokenizer():
    """BM25 should split snake_case and match partial tokens."""
    texts = ["nova_compute memory_leak bug_report"]
    ids = ["chunk-0"]
    bm25 = BM25Index()
    bm25.build(ids, texts)
    results = bm25.search("memory leak compute")
    assert len(results) > 0


# ── RRF tests ────────────────────────────────────────────────────────────────

def test_rrf_merges_lists():
    list1 = ["a", "b", "c"]
    list2 = ["b", "d", "a"]
    result = reciprocal_rank_fusion([list1, list2])
    ids = [r[0] for r in result]
    top2 = set(ids[:2])
    assert top2 == {"a", "b"}


def test_rrf_single_list():
    lst = ["x", "y", "z"]
    result = reciprocal_rank_fusion([lst])
    assert [r[0] for r in result] == ["x", "y", "z"]


def test_rrf_empty():
    result = reciprocal_rank_fusion([])
    assert result == []


# ── Embedding model tests ────────────────────────────────────────────────────

def test_embedding_model_query_prefix():
    model = EmbeddingModel(model_name="BAAI/bge-small-en-v1.5")
    with patch.object(model, '_load'), \
         patch.object(model, '_model') as mock_model:
        mock_model.encode.return_value = np.zeros(384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        model._model = mock_model
        model.encode_query("test query")
        call_args = mock_model.encode.call_args[0][0]
        assert BGE_QUERY_PREFIX in call_args


def test_embedding_model_no_prefix_for_documents():
    model = EmbeddingModel(model_name="BAAI/bge-small-en-v1.5")
    with patch.object(model, '_load'), \
         patch.object(model, '_model') as mock_model:
        mock_model.encode.return_value = np.zeros((2, 384))
        model._model = mock_model
        model.encode_documents(["doc1", "doc2"], show_progress=False)
        call_args = mock_model.encode.call_args[0][0]
        assert BGE_QUERY_PREFIX not in call_args[0]


# ── Sync state tests ──────────────────────────────────────────────────────────

def test_sync_state_roundtrip(tmp_path):
    from lpix.sync.state import SyncState
    from datetime import datetime, timezone

    state = SyncState(path=tmp_path / "sync.json")
    assert state.get_last_sync("ubuntu") is None

    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    state.set_last_sync("ubuntu", dt)
    state.save()

    state2 = SyncState(path=tmp_path / "sync.json")
    loaded = state2.get_last_sync("ubuntu")
    assert loaded is not None
    assert loaded.year == 2024
    assert loaded.month == 6


def test_sync_state_reset(tmp_path):
    from lpix.sync.state import SyncState

    state = SyncState(path=tmp_path / "sync.json")
    state.set_last_sync("nova")
    state.set_last_sync("ubuntu")
    state.save()

    state.reset("nova")
    assert state.get_last_sync("nova") is None
    assert state.get_last_sync("ubuntu") is not None


# ── Deduplication test ───────────────────────────────────────────────────────

def test_deduplicate_by_bug():
    """Max 2 chunks per bug_id should be enforced."""
    results = [
        {"metadata": {"bug_id": 1}, "text": "a"},
        {"metadata": {"bug_id": 1}, "text": "b"},
        {"metadata": {"bug_id": 1}, "text": "c"},  # should be dropped
        {"metadata": {"bug_id": 2}, "text": "d"},
        {"metadata": {"bug_id": 2}, "text": "e"},
        {"metadata": {"bug_id": 2}, "text": "f"},  # should be dropped
    ]
    deduped = _deduplicate_by_bug(results, max_per_bug=2)
    assert len(deduped) == 4
    bug1_chunks = [r for r in deduped if r["metadata"]["bug_id"] == 1]
    bug2_chunks = [r for r in deduped if r["metadata"]["bug_id"] == 2]
    assert len(bug1_chunks) == 2
    assert len(bug2_chunks) == 2


def test_deduplicate_preserves_order():
    """Deduplication should preserve result order."""
    results = [
        {"metadata": {"bug_id": 3}, "text": "first"},
        {"metadata": {"bug_id": 1}, "text": "second"},
        {"metadata": {"bug_id": 2}, "text": "third"},
    ]
    deduped = _deduplicate_by_bug(results, max_per_bug=1)
    texts = [r["text"] for r in deduped]
    assert texts == ["first", "second", "third"]
