"""Atomic JPEG snapshots for entry/exit event records."""

import os
from pathlib import Path

import cv2
import numpy as np


def save_face_snapshot(crop: np.ndarray, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError(f"Could not encode snapshot: {image_path}")
    temporary_path = image_path.with_name(image_path.stem + ".tmp.jpg")
    try:
        temporary_path.write_bytes(encoded.tobytes())
        os.replace(temporary_path, image_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
