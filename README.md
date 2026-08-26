# lpix — Launchpad Bug RAG System

A minimal, high-impact RAG (Retrieval-Augmented Generation) system for querying Launchpad bug tracker tickets via natural language, designed to integrate with Hermes Agent.

## Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Data source** | `launchpadlib` | Official Launchpad Python client |
| **Chunking** | Per-comment + ticket header | Preserves context, best recall |
| **Embeddings** | `sentence-transformers/bge-small-en-v1.5` | Best quality/size ratio, MTEB top-performer, 33MB |
| **Vector store** | `ChromaDB` (persistent) | Zero-infra, Python-native, hybrid search ready |
| **BM25** | `rank_bm25` | Lightweight hybrid retrieval |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast, accurate reranking |
| **LLM** | Hermes Agent (via tool) | Already running |

## Quick Start

```bash
# Install dependencies
uv pip install launchpadlib sentence-transformers chromadb rank-bm25 tqdm

# Ingest a Launchpad project
python -m lpix.sync.ingest --project ubuntu --limit 500

# Start querying via Hermes tool
# (register tools/search_tool.py as a Hermes plugin)
```

## Architecture

```
Launchpad API
     │
     ▼
 ingestion/          ← fetch bugs + comments
     │
     ▼
 chunking            ← ticket header + per-comment chunks
     │
     ▼
 embedding           ← bge-small-en-v1.5
     │
     ▼
 ChromaDB            ← persistent vector store
     │
     ▼
 retrieval           ← hybrid BM25 + dense → rerank
     │
     ▼
 Hermes Tool         ← search_launchpad_bugs()
```

## Modules

- `lpix/ingestion/` — Launchpad API fetching
- `lpix/embedding/` — embedding model wrapper
- `lpix/retrieval/` — hybrid search + reranking
- `lpix/sync/` — incremental sync state
- `lpix/tools/` — Hermes tool definitions
