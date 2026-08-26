"""
lpix — Incremental Sync

Tracks ingest state so we only re-fetch changed/new bugs.

Strategy:
- Store last_sync_time per project in a simple JSON file (~/.lpix/sync_state.json)
- Use Launchpad's modified_since filter to only get bugs updated after last sync
- Compare date_last_updated on individual bugs to detect actual changes
- Re-embed and upsert changed chunks (ChromaDB upsert is idempotent)

Why not a database:
- JSON state file is dead simple and sufficient
- ~10k bugs means sync completes in minutes even full re-sync
- SQLite would be overkill for this metadata

Gotcha:
- Launchpad's modified_since filter uses UTC timestamps
- launchpadlib returns datetime objects with UTC timezone
- Store timestamps as ISO8601 with Z suffix for clarity
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path(os.environ.get("LPIX_SYNC_PATH", Path.home() / ".lpix" / "sync_state.json"))


class SyncState:
    """
    Manages per-project sync timestamps.
    
    Usage:
        state = SyncState()
        
        last_sync = state.get_last_sync("ubuntu")  # None on first run
        
        # ... ingest bugs since last_sync ...
        
        state.set_last_sync("ubuntu")  # update to now
        state.save()
    """
    
    def __init__(self, path: Path = DEFAULT_STATE_PATH):
        self.path = path
        self._data: dict = {}
        self.load()
    
    def load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
                logger.debug(f"Loaded sync state from {self.path}")
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
                self._data = {}
        else:
            self._data = {}
    
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)
        logger.debug(f"Saved sync state to {self.path}")
    
    def get_last_sync(self, project: str) -> Optional[datetime]:
        """Returns last sync time for project, or None if never synced."""
        ts = self._data.get(project, {}).get("last_sync")
        if ts is None:
            return None
        # Parse ISO8601, ensure UTC
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt
    
    def set_last_sync(self, project: str, dt: Optional[datetime] = None):
        """Set last sync time for project (defaults to now UTC)."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        if project not in self._data:
            self._data[project] = {}
        self._data[project]["last_sync"] = dt.isoformat()
    
    def get_bug_count(self, project: str) -> int:
        return self._data.get(project, {}).get("bug_count", 0)
    
    def set_bug_count(self, project: str, count: int):
        if project not in self._data:
            self._data[project] = {}
        self._data[project]["bug_count"] = count
    
    def reset(self, project: Optional[str] = None):
        """Reset sync state (force full re-sync)."""
        if project:
            self._data.pop(project, None)
        else:
            self._data = {}
