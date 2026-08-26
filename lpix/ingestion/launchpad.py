"""
lpix — Launchpad Bug RAG System

Ingestion module: fetches bugs and comments from Launchpad API using launchpadlib.

Key design decisions:
- Anonymous access for public projects (no OAuth needed)
- Authenticated access for private projects or higher rate limits
- Fetches: bug title, description, tags, status, importance, comments
- Pagination handled via launchpadlib's lazr.restfulclient collections
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class BugChunk:
    """A single chunk of text extracted from a Launchpad bug."""
    
    # Identifiers
    chunk_id: str           # e.g. "bug-12345-comment-0"  (comment-0 = description)
    bug_id: int
    bug_url: str
    
    # Text content
    text: str               # The actual chunk text
    chunk_type: str         # "description" | "comment" | "header"
    
    # Metadata for filtering
    project: str
    title: str
    status: str             # "New", "Confirmed", "Fix Released", etc.
    importance: str         # "Critical", "High", "Medium", "Low", "Undecided"
    tags: list[str]
    
    # For incremental sync
    date_created: str       # ISO8601
    date_last_updated: str  # ISO8601
    
    # Comment-specific
    comment_index: int = 0  # 0 = description, 1+ = comments
    author: str = ""


def get_launchpad(authenticated: bool = False, credentials_file: Optional[str] = None):
    """
    Get a Launchpad API connection.
    
    Args:
        authenticated: If True, use OAuth flow (needed for private bugs or 
                       higher rate limits). Opens browser on first use.
        credentials_file: Path to store OAuth credentials between sessions.
                         Defaults to ~/.launchpadlib/credentials/
    
    Returns:
        launchpadlib Launchpad instance
    
    Rate limits:
        - Anonymous: ~3600 requests/hour (1/sec average)
        - Authenticated: ~50,000 requests/hour
        - launchpadlib handles back-off automatically on 503s
    """
    try:
        from launchpadlib.launchpad import Launchpad
    except ImportError:
        raise ImportError(
            "launchpadlib not installed. Run: uv pip install launchpadlib"
        )
    
    app_name = "lpix-rag"
    
    if authenticated:
        lp = Launchpad.login_with(
            app_name,
            "production",
            credentials_file=credentials_file,
            version="devel",
        )
    else:
        lp = Launchpad.login_anonymously(
            app_name,
            "production",
            version="devel",
        )
    
    return lp


def iter_project_bugs(
    lp,
    project_name: str,
    since: Optional[datetime] = None,
    statuses: Optional[list[str]] = None,
    limit: Optional[int] = None,
    batch_size: int = 75,
) -> Iterator[dict]:
    """
    Iterate over bugs in a Launchpad project.
    
    Args:
        lp: Launchpad connection from get_launchpad()
        project_name: e.g. "ubuntu", "nova", "neutron"
        since: Only return bugs modified after this datetime (for incremental sync)
        statuses: Filter by status. None = all statuses.
                  Default active: ["New", "Confirmed", "Triaged", "In Progress"]
        limit: Max bugs to fetch (None = all)
        batch_size: Bugs per API page (max 300, sweet spot ~75 for stability)
    
    Yields:
        dict with bug fields
    
    Gotchas:
        - launchpadlib returns lazr.restfulclient Entry objects, not plain dicts
        - Accessing .bug_tasks[0] triggers another HTTP request (lazy loading)
        - date_last_updated on the BugTask, not the Bug itself
        - Collections auto-paginate but are not true iterators — they fetch pages lazily
        - Use .lp_attributes to inspect available fields on an object
    """
    project = lp.projects[project_name]
    
    search_kwargs = {
        "order_by": "date_last_updated",  # newest first for incremental sync
    }
    
    if since is not None:
        # Launchpad API: modified_since filters by date_last_updated
        search_kwargs["modified_since"] = since
    
    if statuses is not None:
        search_kwargs["status"] = statuses
    
    # searchTasks returns BugTask objects (bug + project context)
    bug_tasks = project.searchTasks(**search_kwargs)
    
    count = 0
    for task in bug_tasks:
        if limit is not None and count >= limit:
            break
        
        try:
            bug = task.bug
            
            yield {
                "bug_id": bug.id,
                "bug_url": bug.web_link,
                "title": bug.title,
                "description": bug.description or "",
                "status": task.status,
                "importance": task.importance,
                "tags": list(bug.tags),
                "date_created": bug.date_created.isoformat(),
                "date_last_updated": bug.date_last_updated.isoformat(),
                "messages": bug.messages,  # lazr collection, iterate separately
                "project": project_name,
            }
            count += 1
            
            if count % 50 == 0:
                logger.info(f"Fetched {count} bugs from {project_name}")
                
        except Exception as e:
            logger.warning(f"Error fetching bug task: {e}")
            continue


def bug_to_chunks(bug_data: dict, max_comment_length: int = 2000) -> list[BugChunk]:
    """
    Convert a bug dict (from iter_project_bugs) into BugChunk objects.
    
    Chunking strategy: "ticket header + per-comment"
    - Chunk 0 (type="description"): title + tags + description text
      (gives dense context for the whole bug)
    - Chunks 1..N (type="comment"): each message body individually
      (allows pinpointing specific comments)
    
    Why NOT sliding window over all text:
    - Comments are already natural semantic units
    - Attribution (who said what) is preserved
    - Incremental updates only re-embed changed comments
    
    Why NOT one chunk per bug:
    - Long bugs (100+ comments) exceed embedding model context windows
    - Relevant info is often in a specific comment, not the whole thread
    
    Args:
        bug_data: dict from iter_project_bugs()
        max_comment_length: Truncate individual comments to this length
                           (bge-small max is 512 tokens ≈ 380 words)
    
    Returns:
        List of BugChunk objects ready for embedding
    """
    chunks = []
    base = {
        "bug_id": bug_data["bug_id"],
        "bug_url": bug_data["bug_url"],
        "project": bug_data["project"],
        "title": bug_data["title"],
        "status": bug_data["status"],
        "importance": bug_data["importance"],
        "tags": bug_data["tags"],
        "date_created": bug_data["date_created"],
        "date_last_updated": bug_data["date_last_updated"],
    }
    
    # Chunk 0: Header + Description
    # Include tags as text — they're semantic signals (e.g. "regression", "crash")
    tags_str = ", ".join(bug_data["tags"]) if bug_data["tags"] else ""
    header = (
        f"Bug #{bug_data['bug_id']}: {bug_data['title']}\n"
        f"Status: {bug_data['status']} | Importance: {bug_data['importance']}"
    )
    if tags_str:
        header += f" | Tags: {tags_str}"
    
    desc_text = bug_data["description"].strip()
    if desc_text:
        description_chunk_text = f"{header}\n\n{desc_text[:max_comment_length]}"
    else:
        description_chunk_text = header
    
    chunks.append(BugChunk(
        chunk_id=f"bug-{bug_data['bug_id']}-comment-0",
        text=description_chunk_text,
        chunk_type="description",
        comment_index=0,
        author="",
        **base,
    ))
    
    # Chunks 1..N: Each comment
    # messages[0] is the description (already handled above), skip it
    try:
        messages = bug_data["messages"]
        for i, msg in enumerate(messages):
            if i == 0:
                continue  # description already in chunk 0
            
            try:
                body = (msg.content or "").strip()
                author = str(msg.owner_link).split("~")[-1] if msg.owner_link else ""
                date = msg.date_created.isoformat() if msg.date_created else ""
            except Exception:
                continue
            
            if not body or body == "-- \n":  # skip empty/system messages
                continue
            
            # Prepend compact header so each comment chunk is self-contained
            comment_text = (
                f"Bug #{bug_data['bug_id']}: {bug_data['title']}\n"
                f"Comment #{i} by {author}:\n"
                f"{body[:max_comment_length]}"
            )
            
            chunks.append(BugChunk(
                chunk_id=f"bug-{bug_data['bug_id']}-comment-{i}",
                text=comment_text,
                chunk_type="comment",
                comment_index=i,
                author=author,
                date_created=date or bug_data["date_created"],
                **{k: v for k, v in base.items() if k != "date_created"},
            ))
    except Exception as e:
        logger.warning(f"Error processing messages for bug {bug_data['bug_id']}: {e}")
    
    return chunks
