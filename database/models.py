"""Lightweight persistence and recognition result models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VisitorIdentity:
    visitor_id: str
    is_new: bool
    similarity: float


@dataclass(frozen=True)
class EventRecord:
    session_id: str
    camera_id: str
    track_id: str
    visitor_id: str
    event_type: str
    timestamp: str
    image_path: str
    bbox: tuple[int, int, int, int]
