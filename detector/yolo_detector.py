from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .preprocess import padded_face_crop


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    confidence: float
    embedding: np.ndarray | None = None
    visitor_id: str | None = None


class YoloFaceDetector:
    """YOLO detector wrapper; configure a face-trained YOLO weights file."""

    def __init__(self, model_path: str, confidence: float, image_size: int) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"YOLO face weights not found: {model_path}. Set detector_model in config.json."
            )
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        height, width = frame.shape[:2]
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )[0]
        detections: list[FaceDetection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = (x1, y1, x2, y2)
            crop = padded_face_crop(frame, bbox)
            if crop.size:
                detections.append(
                    FaceDetection(
                        bbox=bbox,
                        crop=crop,
                        confidence=float(box.conf[0].item()),
                    )
                )
        return detections
