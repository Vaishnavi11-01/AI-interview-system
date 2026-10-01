"""Image crop preparation for recognition after YOLO localization."""

import numpy as np


def padded_face_crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return a face crop with context so a landmark detector can align it."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    padding_x = max(8, int((x2 - x1) * 0.35))
    padding_y = max(8, int((y2 - y1) * 0.35))
    return frame[
        max(0, y1 - padding_y):min(height, y2 + padding_y),
        max(0, x1 - padding_x):min(width, x2 + padding_x),
    ].copy()
