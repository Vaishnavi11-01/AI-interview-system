"""InsightFace ArcFace model initialization."""

from insightface.app import FaceAnalysis


def load_face_analysis(model_name: str = "buffalo_l", use_gpu: bool = False) -> FaceAnalysis:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(name=model_name, providers=providers)
    # YOLO already localizes the face; relaxed detection threshold helps the
    # internal landmark detector align small face crops for recognition.
    # YOLO has already cropped the face; 320px is enough for landmark alignment
    # in most scenes and substantially cheaper than re-detecting at 640px.
    app.prepare(ctx_id=0 if use_gpu else -1, det_thresh=0.25, det_size=(320, 320))
    return app
