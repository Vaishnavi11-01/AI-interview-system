from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off", ""}:
            return False
    if value is None:
        return default
    return bool(value)


@dataclass(frozen=True)
class Settings:
    source: Any
    camera_id: str
    detector_model: str
    detection_confidence: float
    detection_image_size: int
    detection_skip_frames: int
    max_missed_cycles: int
    track_iou_threshold: float
    track_similarity_threshold: float
    recognition_similarity_threshold: float
    database_path: Path
    logs_dir: Path
    show_preview: bool


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    base_dir = config_path.parent

    def local_path(key: str) -> Path:
        value = Path(raw[key]).expanduser()
        return value if value.is_absolute() else (base_dir / value).resolve()

    source = raw.get("source", 0)
    if isinstance(source, str) and source.isdecimal():
        source = int(source)

    settings = Settings(
        source=source,
        camera_id=str(raw.get("camera_id", "camera-1")),
        detector_model=str(local_path("detector_model")),
        detection_confidence=float(raw.get("detection_confidence", 0.45)),
        detection_image_size=int(raw.get("detection_image_size", 416)),
        detection_skip_frames=int(raw.get("detection_skip_frames", 2)),
        max_missed_cycles=int(raw.get("max_missed_cycles", 4)),
        track_iou_threshold=float(raw.get("track_iou_threshold", 0.15)),
        track_similarity_threshold=float(raw.get("track_similarity_threshold", 0.45)),
        recognition_similarity_threshold=float(raw.get("recognition_similarity_threshold", 0.45)),
        database_path=local_path("database_path"),
        logs_dir=local_path("logs_dir"),
        show_preview=_coerce_bool(raw.get("show_preview", True), default=True),
    )
    if settings.detection_skip_frames < 0:
        raise ValueError("detection_skip_frames must be zero or greater")
    if settings.max_missed_cycles < 1:
        raise ValueError("max_missed_cycles must be at least one")
    for name in (
        "detection_confidence",
        "track_iou_threshold",
        "track_similarity_threshold",
        "recognition_similarity_threshold",
    ):
        value = getattr(settings, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if settings.detection_image_size < 64:
        raise ValueError("detection_image_size must be at least 64")
    return settings
