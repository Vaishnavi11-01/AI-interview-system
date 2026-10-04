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
        self._gallery: dict[int, tuple[list[str], np.ndarray]] | None = None

    def _load_gallery(self) -> dict[int, tuple[list[str], np.ndarray]]:
        if self._gallery is None:
            grouped: dict[int, tuple[list[str], list[np.ndarray]]] = {}
            rows = self.connection.execute(
                "SELECT visitor_id, embedding, embedding_dim FROM visitors"
            ).fetchall()
            for row in rows:
                dimension = int(row["embedding_dim"])
                vector = np.frombuffer(row["embedding"], dtype=np.float32)
                if vector.size != dimension or not np.all(np.isfinite(vector)):
                    continue
                ids, vectors = grouped.setdefault(dimension, ([], []))
                ids.append(row["visitor_id"])
                vectors.append(vector)
            self._gallery = {
                dimension: (ids, np.vstack(vectors))
                for dimension, (ids, vectors) in grouped.items()
            }
        return self._gallery

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("Embedding must be finite and non-empty")
        norm = float(np.linalg.norm(array))
        if norm < 1e-12:
            raise ValueError("Embedding norm is too small")
        return array / norm

    def identify_or_register(self, embedding: np.ndarray, threshold: float) -> VisitorIdentity:
        vector = self._normalize(embedding)
        with self.lock:
            gallery_cache = self._load_gallery()
            best_id: str | None = None
            best_similarity = -1.0
            gallery = gallery_cache.get(vector.size)
            if gallery is not None:
                visitor_ids, vectors = gallery
                similarities = vectors @ vector
                best_index = int(np.argmax(similarities))
                best_similarity = float(similarities[best_index])
                best_id = visitor_ids[best_index]

            now = utc_now()
            if best_id is not None and best_similarity >= threshold:
                self.connection.execute(
                    "UPDATE visitors SET last_seen = ? WHERE visitor_id = ?",
                    (now, best_id),
                )
                self.connection.commit()
                return VisitorIdentity(best_id, False, float(best_similarity), int(vector.size))

            visitor_id = f"visitor_{uuid.uuid4().hex[:12]}"
            self.connection.execute(
                """
                INSERT INTO visitors(visitor_id, first_seen, last_seen, embedding, embedding_dim, image_path, visit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (visitor_id, now, now, vector.tobytes(), int(vector.size), None, 1),
            )
            self.connection.commit()
            if vector.size in gallery_cache:
                visitor_ids, vectors = gallery_cache[vector.size]
                gallery_cache[vector.size] = (visitor_ids + [visitor_id], np.vstack((vectors, vector)))
            else:
                gallery_cache[vector.size] = ([visitor_id], vector.reshape(1, -1))
            return VisitorIdentity(visitor_id, True, float(best_similarity), int(vector.size))

    def record_event(self, event: EventRecord) -> bool:
        with self.lock:
            cursor = self.connection.execute(
                """INSERT OR IGNORE INTO events
                (session_id, camera_id, track_id, visitor_id, event_type, timestamp, image_path, bbox)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.session_id,
                    event.camera_id,
                    event.track_id,
                    event.visitor_id,
                    event.event_type,
                    event.timestamp,
                    event.image_path,
                    json.dumps(event.bbox),
                ),
            )
            self.connection.commit()
            return cursor.rowcount == 1

    def unique_visitor_count(self) -> int:
        with self.lock:
            return int(self.connection.execute("SELECT COUNT(*) FROM visitors").fetchone()[0])
