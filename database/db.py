"""SQLite database initialization and query facade."""

from pathlib import Path
import sqlite3
import threading

import numpy as np

from .models import EventRecord, VisitorIdentity
from .queries import VisitorQueries


class VisitorDatabase:
    """SQLite persistence for enrolled visitors and one entry/exit per track."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS visitors (
                    visitor_id TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    visitor_id TEXT NOT NULL REFERENCES visitors(visitor_id),
                    event_type TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
                    timestamp TEXT NOT NULL,
                    image_path TEXT NOT NULL UNIQUE,
                    bbox TEXT NOT NULL,
                    UNIQUE(session_id, camera_id, track_id, event_type)
                );
                CREATE INDEX IF NOT EXISTS idx_events_visitor ON events(visitor_id, timestamp);
                """
            )
            self.connection.commit()
        self.queries = VisitorQueries(self.connection, self.lock)

    def identify_or_register(self, embedding: np.ndarray, threshold: float) -> VisitorIdentity:
        return self.queries.identify_or_register(embedding, threshold)

    def record_event(self, event: EventRecord) -> bool:
        return self.queries.record_event(event)

    def unique_visitor_count(self) -> int:
        return self.queries.unique_visitor_count()

    def close(self) -> None:
        with self.lock:
            self.connection.close()
