# lpix — Launchpad Bug RAG Service
# Multi-stage build: keeps the final image lean

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency spec first (layer cache — only rebuilds if pyproject.toml changes)
COPY pyproject.toml .
COPY lpix/__init__.py lpix/

# Install all runtime deps into /build/.venv
RUN uv venv .venv && \
    uv pip install \
        --python .venv/bin/python \
        launchpadlib \
        "sentence-transformers>=3.0.0" \
        "chromadb>=0.5.0" \
        "rank-bm25>=0.2.2" \
        "fastapi>=0.111.0" \
        "uvicorn[standard]>=0.29.0" \
        tqdm \
        click


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user — Launchpad creds and DB files are owned by this user
RUN useradd --uid 1000 --create-home lpix
WORKDIR /home/lpix

# Copy venv from builder
COPY --from=builder /build/.venv /home/lpix/.venv

# Copy application code
COPY lpix/ ./lpix/

# Data volumes
# /data/chroma_db  — ChromaDB persistent files (vector store + HNSW index)
# /data/lp_creds   — Launchpad OAuth credentials (written by launchpadlib)
# /data/sync       — sync state JSON
RUN mkdir -p /data/chroma_db /data/lp_creds /data/sync && \
    chown -R lpix:lpix /data

VOLUME ["/data/chroma_db", "/data/lp_creds", "/data/sync"]

USER lpix

ENV PATH="/home/lpix/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Override default storage paths to use the mounted volumes
    LPIX_CHROMA_PATH=/data/chroma_db \
    LPIX_CREDS_PATH=/data/lp_creds \
    LPIX_SYNC_PATH=/data/sync/state.json \
    LPIX_BM25_PATH=/data/sync/bm25_index.pkl

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "-m", "lpix.serve"]
