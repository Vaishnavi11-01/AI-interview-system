"""SQLite database initialization and query facade."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import threading

import numpy as np

from .models import EventRecord, VisitorIdentity
from .queries import VisitorQueries
from utils.helpers import utc_now


class VisitorDatabase:
    """SQLite persistence for visitor identity, events, and audit logs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self._closed = False
        self._initialize_schema()
        self.queries = VisitorQueries(self.connection, self.lock)

    def _initialize_schema(self) -> None:
        with self.lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA busy_timeout = 5000")
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS visitors (
                    visitor_id TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    image_path TEXT,
                    visit_count INTEGER NOT NULL DEFAULT 0
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
                    entry_time TEXT,
                    exit_time TEXT,
                    duration REAL,
                    frame_number INTEGER,
                    confidence REAL,
                    snapshot_path TEXT,
                    UNIQUE(session_id, camera_id, track_id, event_type)
                );
                CREATE TABLE IF NOT EXISTS logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_visitor_time ON events(visitor_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_session_track ON events(session_id, camera_id, track_id);
                """
            )
            for table, column, datatype in (
                ("visitors", "last_seen", "TEXT"),
                ("visitors", "image_path", "TEXT"),
                ("visitors", "visit_count", "INTEGER NOT NULL DEFAULT 0"),
                ("events", "entry_time", "TEXT"),
                ("events", "exit_time", "TEXT"),
                ("events", "duration", "REAL"),
                ("events", "frame_number", "INTEGER"),
                ("events", "confidence", "REAL"),
                ("events", "snapshot_path", "TEXT"),
            ):
                existing = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
                if not any(row[1] == column for row in existing):
                    self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {datatype}")
            self.connection.commit()

    def identify_or_register(self, embedding: np.ndarray, threshold: float) -> VisitorIdentity:
        return self.queries.identify_or_register(embedding, threshold)

    def record_event(self, event: EventRecord) -> bool:
        return self.queries.record_event(event)

    def unique_visitor_count(self) -> int:
        return self.queries.unique_visitor_count()

    def log_message(self, level: str, message: str) -> None:
        timestamp = utc_now()
        with self.lock:
            self.connection.execute(
                "INSERT INTO logs(timestamp, level, message) VALUES (?, ?, ?)",
                (timestamp, level, message),
            )
            self.connection.commit()

    def close(self) -> None:
        if self._closed:
            return
        with self.lock:
            if self.connection:
                self.connection.close()
            self._closed = True
