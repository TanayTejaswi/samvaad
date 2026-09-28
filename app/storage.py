"""SQLite storage manager for Samvaad transcripts.

Provides thread-safe async access to local transcript history.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

from inference.qnn_session import load_config

logger = logging.getLogger("samvaad.storage")


class StorageManager:
    """Manages persistent SQLite storage for transcript history."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_config()
        self.db_path = Path(cfg.get("storage", {}).get("db_path", "data/transcripts.db"))
        self.max_history = int(cfg.get("storage", {}).get("max_history_entries", 500))
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the database schema if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS transcripts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        text TEXT NOT NULL,
                        latency_ms INTEGER NOT NULL,
                        device TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            logger.info("Storage manager initialized at %s", self.db_path)
        except sqlite3.Error as e:
            logger.error("Failed to initialize database: %s", e)

    def save_transcript(
        self, timestamp: str, text: str, latency_ms: int, device: str
    ) -> int | None:
        """Saves a transcription record to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO transcripts (timestamp, text, latency_ms, device)
                    VALUES (?, ?, ?, ?)
                    """,
                    (timestamp, text, latency_ms, device),
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error("Failed to save transcript: %s", e)
            return None

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Retrieves recent transcript history, ordered by most recent first."""
        limit = limit or self.max_history
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Return rows as dictionaries
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, timestamp, text, latency_ms, device 
                    FROM transcripts 
                    ORDER BY id DESC 
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                # Convert to dict and reverse to chronological order for UI if desired
                # But UI typically expects newest first or chronological. Let's return chronological.
                return [dict(row) for row in rows][::-1]
        except sqlite3.Error as e:
            logger.error("Failed to retrieve history: %s", e)
            return []
