"""SQLite query operations for visitor identity and event persistence."""

import json
import sqlite3
import threading
import uuid

import numpy as np

from .models import EventRecord, VisitorIdentity
from utils.helpers import utc_now


class VisitorQueries:
    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self.connection = connection
        self.lock = lock

    def identify_or_register(self, embedding: np.ndarray, threshold: float) -> VisitorIdentity:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        with self.lock:
            rows = self.connection.execute("SELECT visitor_id, embedding, embedding_dim FROM visitors").fetchall()
            best_id: str | None = None
            best_similarity = -1.0
            for row in rows:
                if row["embedding_dim"] != vector.size:
                    continue
                stored = np.frombuffer(row["embedding"], dtype=np.float32)
                similarity = float(np.dot(vector, stored))
                if similarity > best_similarity:
                    best_id, best_similarity = row["visitor_id"], similarity
            if best_id is not None and best_similarity >= threshold:
                return VisitorIdentity(best_id, False, best_similarity)

            visitor_id = f"visitor_{uuid.uuid4().hex[:12]}"
            self.connection.execute(
                "INSERT INTO visitors(visitor_id, first_seen, embedding, embedding_dim) VALUES (?, ?, ?, ?)",
                (visitor_id, utc_now(), vector.tobytes(), int(vector.size)),
            )
            self.connection.commit()
            return VisitorIdentity(visitor_id, True, best_similarity)

    def record_event(self, event: EventRecord) -> bool:
        with self.lock:
            cursor = self.connection.execute(
                """INSERT OR IGNORE INTO events
                (session_id, camera_id, track_id, visitor_id, event_type, timestamp, image_path, bbox)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.session_id, event.camera_id, event.track_id, event.visitor_id,
                    event.event_type, event.timestamp, event.image_path, json.dumps(event.bbox),
                ),
            )
            self.connection.commit()
            return cursor.rowcount == 1

    def unique_visitor_count(self) -> int:
        with self.lock:
            return int(self.connection.execute("SELECT COUNT(*) FROM visitors").fetchone()[0])
