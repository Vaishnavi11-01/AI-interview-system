import json
from pathlib import Path

from utils.config import load_settings


def test_show_preview_string_false_is_parsed_correctly(tmp_path):
    config = {
        "source": 0,
        "camera_id": "camera-1",
        "detector_model": "models/yolov8n-face.pt",
        "detection_confidence": 0.45,
        "detection_image_size": 640,
        "detection_skip_frames": 2,
        "max_missed_cycles": 4,
        "track_iou_threshold": 0.15,
        "track_similarity_threshold": 0.45,
        "recognition_similarity_threshold": 0.45,
        "database_path": "data/visitors.sqlite3",
        "logs_dir": "logs",
        "show_preview": "false",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    settings = load_settings(config_path)

    assert settings.show_preview is False


def test_show_preview_string_true_is_parsed_correctly(tmp_path):
    config = {
        "source": 0,
        "camera_id": "camera-1",
        "detector_model": "models/yolov8n-face.pt",
        "detection_confidence": 0.45,
        "detection_image_size": 640,
        "detection_skip_frames": 2,
        "max_missed_cycles": 4,
        "track_iou_threshold": 0.15,
        "track_similarity_threshold": 0.45,
        "recognition_similarity_threshold": 0.45,
        "database_path": "data/visitors.sqlite3",
        "logs_dir": "logs",
        "show_preview": "true",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    settings = load_settings(config_path)

    assert settings.show_preview is True
