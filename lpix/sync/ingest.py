"""
lpix — Main ingest pipeline

Ties together: Launchpad fetch → chunk → embed → store → sync state.

Usage:
    python -m lpix.sync.ingest --project ubuntu --limit 200
    python -m lpix.sync.ingest --project nova --full-resync
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from lpix.ingestion.launchpad import get_launchpad, iter_project_bugs, bug_to_chunks
from lpix.embedding.model import EmbeddingModel
from lpix.retrieval.store import BugVectorStore
from lpix.retrieval.hybrid import BM25Index
from lpix.sync.state import SyncState

logger = logging.getLogger(__name__)


def ingest_project(
    project_name: str,
    authenticated: bool = False,
    full_resync: bool = False,
    limit: Optional[int] = None,
    statuses: Optional[list[str]] = None,
    db_path: Optional[Path] = None,
    credentials_file: Optional[str] = None,
):
    """
    Full ingest pipeline for a Launchpad project.
    
    Args:
        project_name: Launchpad project slug (e.g. "ubuntu", "nova")
        authenticated: Use OAuth for higher rate limits
        full_resync: Ignore sync state and re-fetch everything
        limit: Max bugs to fetch (None = all)
        statuses: Bug statuses to include. None = all active statuses
        db_path: Override ChromaDB path
        credentials_file: OAuth credentials file path
    """
    # Default to active statuses only
    if statuses is None:
        statuses = [
            "New", "Incomplete", "Confirmed", "Triaged",
            "In Progress", "Fix Committed",
        ]
    
    sync_state = SyncState()
    store = BugVectorStore(db_path=db_path)
    embedding_model = EmbeddingModel()
    
    # Determine since time for incremental sync
    since = None
    if not full_resync:
        since = sync_state.get_last_sync(project_name)
        if since:
            logger.info(f"Incremental sync for '{project_name}' since {since.isoformat()}")
        else:
            logger.info(f"First-time full sync for '{project_name}'")
    else:
        logger.info(f"Full re-sync for '{project_name}' (--full-resync)")
    
    # Mark sync start time BEFORE fetching (avoid missing bugs updated during ingest)
    sync_start = datetime.now(timezone.utc)
    
    # Connect to Launchpad
    lp = get_launchpad(authenticated=authenticated, credentials_file=credentials_file)
    
    # Fetch bugs and process
    bug_iter = iter_project_bugs(
        lp, project_name, since=since, statuses=statuses, limit=limit
    )
    
    total_chunks = 0
    total_bugs = 0
    batch_chunks = []
    EMBED_BATCH = 256  # embed this many chunks at once
    
    for bug_data in tqdm(bug_iter, desc=f"Ingesting {project_name}", unit="bug"):
        chunks = bug_to_chunks(bug_data)
        batch_chunks.extend(chunks)
        total_bugs += 1
        
        # Process in batches to control memory
        if len(batch_chunks) >= EMBED_BATCH:
            _embed_and_store(batch_chunks, embedding_model, store)
            total_chunks += len(batch_chunks)
            batch_chunks = []
    
    # Process remaining
    if batch_chunks:
        _embed_and_store(batch_chunks, embedding_model, store)
        total_chunks += len(batch_chunks)
    
    logger.info(f"Ingested {total_bugs} bugs → {total_chunks} chunks")
    
    # Rebuild BM25 index
    logger.info("Rebuilding BM25 index...")
    bm25 = BM25Index()
    ids, texts = store.get_all_texts()
    if ids:
        bm25.build(ids, texts)
    
    # Save sync state
    sync_state.set_last_sync(project_name, sync_start)
    sync_state.set_bug_count(project_name, store.count())
    sync_state.save()
    
    logger.info(
        f"Sync complete. Vector store: {store.count()} total chunks. "
        f"Next sync will be incremental from {sync_start.isoformat()}"
    )
    return total_bugs, total_chunks


def _embed_and_store(chunks, embedding_model, store):
    """Embed a batch of chunks and upsert into the vector store."""
    texts = [c.text for c in chunks]
    embeddings = embedding_model.encode_documents(texts, show_progress=False)
    store.upsert_chunks(chunks, embeddings)


@click.command()
@click.option("--project", required=True, help="Launchpad project name (e.g. ubuntu, nova)")
@click.option("--limit", default=None, type=int, help="Max bugs to fetch")
@click.option("--full-resync", is_flag=True, help="Ignore sync state, re-fetch all")
@click.option("--authenticated", is_flag=True, help="Use OAuth (higher rate limits)")
@click.option("--db-path", default=None, help="ChromaDB storage path")
@click.option("--verbose", "-v", is_flag=True)
def main(project, limit, full_resync, authenticated, db_path, verbose):
    """Ingest Launchpad bugs into the lpix vector store."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    t0 = time.time()
    bugs, chunks = ingest_project(
        project_name=project,
        authenticated=authenticated,
        full_resync=full_resync,
        limit=limit,
        db_path=Path(db_path) if db_path else None,
    )
    elapsed = time.time() - t0
    click.echo(
        f"\n✓ Done in {elapsed:.1f}s: {bugs} bugs, {chunks} chunks indexed"
    )


if __name__ == "__main__":
    main()
