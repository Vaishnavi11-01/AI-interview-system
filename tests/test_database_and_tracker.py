import numpy as np
from collections import OrderedDict
from types import SimpleNamespace
import logging

from database.db import VisitorDatabase
from database.models import EventRecord, VisitorIdentity
from detector.yolo_detector import FaceDetection
from pipeline import FaceTrackingPipeline
from tracker.tracker_manager import IoUAppearanceTracker, Track


def test_identify_or_register_reuses_existing_identity(tmp_path):
    db = VisitorDatabase(tmp_path / "visitors.sqlite3")
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    first = db.identify_or_register(embedding, threshold=0.45)
    second = db.identify_or_register(embedding, threshold=0.45)

    assert first.is_new is True
    assert second.is_new is False
    assert first.visitor_id == second.visitor_id
    assert len(db.queries._gallery[4][0]) == 1
    db.close()


def test_duplicate_event_insertion_is_ignored(tmp_path):
    db = VisitorDatabase(tmp_path / "events.sqlite3")
    db.connection.execute(
        "INSERT INTO visitors(visitor_id, first_seen, last_seen, embedding, embedding_dim, image_path, visit_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("visitor_123", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00", np.zeros(4, dtype=np.float32).tobytes(), 4, "/tmp/face.jpg", 1),
    )
    db.connection.commit()

    event = EventRecord(
        session_id="session-1",
        camera_id="camera-1",
        track_id="track-1",
        visitor_id="visitor_123",
        event_type="entry",
        timestamp="2026-01-01T00:00:00+00:00",
        image_path="/tmp/entry.jpg",
        bbox=(0, 0, 10, 10),
    )

    assert db.record_event(event) is True
    assert db.record_event(event) is False
    db.close()


def test_tracker_reuses_existing_track_for_same_detection(tmp_path):
    tracker = IoUAppearanceTracker(max_missed_cycles=2, iou_threshold=0.15, similarity_threshold=0.45)
    detection = FaceDetection(
        bbox=(0, 0, 20, 20),
        crop=np.zeros((20, 20, 3), dtype=np.uint8),
        confidence=0.95,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    tracker.active["existing-track"] = Track(
        track_id="existing-track",
        bbox=(0, 0, 20, 20),
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        confidence=0.95,
        crop=detection.crop,
    )

    updates, exited = tracker.update([detection])

    assert len(updates) >= 1
    assert updates[0].is_new is False
    assert len(exited) == 0


def test_pipeline_embeds_only_new_tracks_and_reuses_crop_cache():
    class CountingEmbedder:
        calls = 0

        def embed(self, crop):
            self.calls += 1
            return np.array([1.0, 0.0], dtype=np.float32)

    pipeline = FaceTrackingPipeline.__new__(FaceTrackingPipeline)
    pipeline.tracker = IoUAppearanceTracker(max_missed_cycles=3, iou_threshold=0.15, similarity_threshold=0.45)
    pipeline.embedder = CountingEmbedder()
    pipeline.matcher = SimpleNamespace(
        identify=lambda embedding: VisitorIdentity("visitor_cached", False, 1.0, 2)
    )
    pipeline.logger = logging.getLogger("test-performance-pipeline")
    pipeline.frame_number = 0
    pipeline._embedding_cache = OrderedDict()
    pipeline._embedding_cache_limit = 128
    pipeline._unique_visitor_count = 1
    recorded_events = []
    pipeline._record_event = lambda track, event_type: recorded_events.append((track.track_id, event_type))

    crop = np.full((12, 12, 3), 127, dtype=np.uint8)
    first = FaceDetection((0, 0, 20, 20), crop, 0.9)
    updates, exited = pipeline.tracker.update([first])
    pipeline._handle_updates(updates, exited)
    assert pipeline.embedder.calls == 1
    assert first.embedding is not None

    overlapping = FaceDetection((1, 1, 21, 21), np.zeros_like(crop), 0.9)
    updates, exited = pipeline.tracker.update([overlapping])
    pipeline._handle_updates(updates, exited)
    assert updates[0].is_new is False
    assert overlapping.embedding is None
    assert pipeline.embedder.calls == 1

    # An exact crop seen as a different new track can reuse the cached embedding.
    separate = FaceDetection((100, 100, 120, 120), crop.copy(), 0.9)
    updates, exited = pipeline.tracker.update([separate])
    pipeline._handle_updates(updates, exited)
    assert updates[0].is_new is True
    assert separate.embedding is not None
    assert pipeline.embedder.calls == 1
