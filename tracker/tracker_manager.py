from dataclasses import dataclass
import uuid

import numpy as np

from detector.yolo_detector import FaceDetection


@dataclass
class Track:
    track_id: str
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray | None
    confidence: float
    crop: np.ndarray
    visitor_id: str | None = None
    missed_cycles: int = 0


@dataclass
class TrackUpdate:
    track: Track
    detection: FaceDetection | None
    is_new: bool


class IoUAppearanceTracker:
    """Greedy IoU association without per-frame face embeddings."""

    def __init__(self, max_missed_cycles: int, iou_threshold: float, similarity_threshold: float) -> None:
        self.max_missed_cycles = max_missed_cycles
        self.iou_threshold = iou_threshold
        self.similarity_threshold = similarity_threshold
        self.active: dict[str, Track] = {}

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        left, top = max(a[0], b[0]), max(a[1], b[1])
        right, bottom = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def _score(self, track: Track, detection: FaceDetection) -> float:
        return self._iou(track.bbox, detection.bbox)

    def update(self, detections: list[FaceDetection]) -> tuple[list[TrackUpdate], list[Track]]:
        for track in self.active.values():
            track.missed_cycles += 1

        candidates: list[tuple[float, str, int]] = []
        for track_id, track in self.active.items():
            for index, detection in enumerate(detections):
                iou = self._iou(track.bbox, detection.bbox)
                if iou >= self.iou_threshold:
                    score = self._score(track, detection)
                    candidates.append((score, track_id, index))

        candidates.sort(key=lambda item: item[0], reverse=True)
        assigned_tracks: set[str] = set()
        assigned_detections: set[int] = set()
        updates: list[TrackUpdate] = []

        for _, track_id, index in candidates:
            if track_id in assigned_tracks or index in assigned_detections:
                continue
            track = self.active[track_id]
            detection = detections[index]
            track.bbox = detection.bbox
            # Keep the first embedding; subsequent frames are tracked by box overlap.
            track.confidence = detection.confidence
            track.crop = detection.crop
            track.missed_cycles = 0
            assigned_tracks.add(track_id)
            assigned_detections.add(index)
            updates.append(TrackUpdate(track, detection, False))

        for index, detection in enumerate(detections):
            if index in assigned_detections:
                continue
            track = Track(
                track_id=uuid.uuid4().hex[:12],
                bbox=detection.bbox,
                embedding=detection.embedding,
                confidence=detection.confidence,
                crop=detection.crop,
            )
            self.active[track.track_id] = track
            updates.append(TrackUpdate(track, detection, True))

        exited = [track for track in self.active.values() if track.missed_cycles >= self.max_missed_cycles]
        for track in exited:
            self.active.pop(track.track_id, None)
        return updates, exited

    def flush(self) -> list[Track]:
        remaining = list(self.active.values())
        self.active.clear()
        return remaining
