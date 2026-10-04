"""Persistence and recognition models used by the face tracker."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VisitorIdentity:
    visitor_id: str
    is_new: bool
    similarity: float
    embedding_dim: int = 0


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


@dataclass(frozen=True)
class LogRecord:
    timestamp: str
    level: str
    message: str
