"""Compute normalized ArcFace embeddings from detected face crops."""

import numpy as np

from .insightface_model import load_face_analysis


class InsightFaceEmbedder:
    """InsightFace ArcFace embedding generator (CPU by default)."""

    def __init__(self, model_name: str = "buffalo_l", use_gpu: bool = False) -> None:
        self.app = load_face_analysis(model_name, use_gpu)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            raise ValueError("Cannot create an embedding from an empty crop")
        faces = self.app.get(crop, max_num=1)
        if not faces:
            raise ValueError("InsightFace could not find a face inside the YOLO crop")
        face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
        vector = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.ndim != 1 or norm == 0.0:
            raise ValueError("InsightFace returned an invalid embedding")
        return vector / norm
