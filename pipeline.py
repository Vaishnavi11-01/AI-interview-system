from datetime import datetime
import hashlib
from collections import OrderedDict
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from database.db import VisitorDatabase
from database.models import EventRecord
from detector.yolo_detector import FaceDetection, YoloFaceDetector
from logger.event_logger import configure_logging
from logger.image_logger import save_face_snapshot
from recognition.embedding import InsightFaceEmbedder
from recognition.matcher import IdentityMatcher
from tracker.tracker_manager import IoUAppearanceTracker, Track, TrackUpdate
from utils.config import Settings
from utils.helpers import utc_now


class FaceTrackingPipeline:
    def __init__(self, settings: Settings, use_gpu: bool = False) -> None:
        self.settings = settings
        self.logger = configure_logging(settings.logs_dir)
        self.database = VisitorDatabase(settings.database_path)
        self.detector = YoloFaceDetector(
            settings.detector_model,
            settings.detection_confidence,
            settings.detection_image_size,
            use_gpu=use_gpu,
        )
        self.embedder = InsightFaceEmbedder(use_gpu=use_gpu)
        self.matcher = IdentityMatcher(self.database, settings.recognition_similarity_threshold)
        self.tracker = IoUAppearanceTracker(
            settings.max_missed_cycles,
            settings.track_iou_threshold,
            settings.track_similarity_threshold,
        )
        self.session_id = uuid.uuid4().hex
        self.frame_number = 0
        self.closed = False
        self._embedding_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._embedding_cache_limit = 128
        self._unique_visitor_count = self.database.unique_visitor_count()
        self.logger.info(
            "session_started session_id=%s camera_id=%s source=%s detection_skip_frames=%d",
            self.session_id,
            settings.camera_id,
            settings.source,
            settings.detection_skip_frames,
        )

    def _prepare_detections(self, frame: np.ndarray) -> list[FaceDetection]:
        # Keep detector output unembedded so the tracker can first decide whether
        # each detection belongs to a known track.
        return self.detector.detect(frame)

    def _embedding_for_new_track(self, crop: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(crop)
        digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()
        key = f"{contiguous.shape}:{contiguous.dtype}".encode() + digest
        cached = self._embedding_cache.get(key)
        if cached is not None:
            self._embedding_cache.move_to_end(key)
            return cached.copy()

        embedding = self.embedder.embed(contiguous)
        self._embedding_cache[key] = embedding.copy()
        if len(self._embedding_cache) > self._embedding_cache_limit:
            self._embedding_cache.popitem(last=False)
        return embedding

    def _record_event(self, track: Track, event_type: str) -> None:
        if not track.visitor_id:
            self.logger.error("event_skipped_missing_identity track_id=%s type=%s", track.track_id, event_type)
            return

        timestamp = utc_now()
        date_folder = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
        event_folder = "entries" if event_type == "entry" else "exits"
        event_dir = self.settings.logs_dir / event_folder / date_folder
        event_dir.mkdir(parents=True, exist_ok=True)
        safe_timestamp = timestamp.replace(":", "-").replace("+", "_")
        image_path = event_dir / f"{safe_timestamp}_{track.visitor_id}_{track.track_id}.jpg"
        save_face_snapshot(track.crop, image_path)

        event = EventRecord(
            session_id=self.session_id,
            camera_id=self.settings.camera_id,
            track_id=track.track_id,
            visitor_id=track.visitor_id,
            event_type=event_type,
            timestamp=timestamp,
            image_path=str(image_path.resolve()),
            bbox=track.bbox,
        )

        try:
            inserted = self.database.record_event(event)
            if not inserted:
                try:
                    image_path.unlink(missing_ok=True)
                except OSError:
                    pass
                self.logger.warning(
                    "duplicate_event_suppressed session_id=%s track_id=%s type=%s",
                    self.session_id,
                    track.track_id,
                    event_type,
                )
                return
        except Exception:
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        self.logger.info(
            "face_%s visitor_id=%s track_id=%s camera_id=%s timestamp=%s image=%s unique_visitors=%d",
            event_type,
            track.visitor_id,
            track.track_id,
            self.settings.camera_id,
            timestamp,
            image_path,
            self.database.unique_visitor_count(),
        )

    def _handle_updates(self, updates: list[TrackUpdate], exited: list[Track]) -> None:
        for update in updates:
            track = update.track
            detection = update.detection
            if detection is None:
                continue
            if update.is_new:
                try:
                    detection.embedding = self._embedding_for_new_track(detection.crop)
                    track.embedding = detection.embedding
                except Exception as error:
                    self.logger.warning(
                        "embedding_failed frame=%d track_id=%s bbox=%s error=%s",
                        self.frame_number,
                        track.track_id,
                        detection.bbox,
                        error,
                    )
                    continue
                self.logger.info(
                    "embedding_generated frame=%d track_id=%s bbox=%s",
                    self.frame_number,
                    track.track_id,
                    detection.bbox,
                )
                identity = self.matcher.identify(detection.embedding)
                track.visitor_id = identity.visitor_id
                if identity.is_new:
                    self._unique_visitor_count += 1
                    self.logger.info(
                        "face_registered visitor_id=%s track_id=%s similarity=%.4f",
                        identity.visitor_id,
                        track.track_id,
                        identity.similarity,
                    )
                else:
                    self.logger.info(
                        "face_recognized visitor_id=%s track_id=%s similarity=%.4f",
                        identity.visitor_id,
                        track.track_id,
                        identity.similarity,
                    )
                self._record_event(track, "entry")
            else:
                self.logger.debug(
                    "face_tracked visitor_id=%s track_id=%s frame=%d bbox=%s",
                    track.visitor_id,
                    track.track_id,
                    self.frame_number,
                    track.bbox,
                )

        for track in exited:
            self._record_event(track, "exit")

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        for track in self.tracker.active.values():
            x1, y1, x2, y2 = track.bbox
            label = track.visitor_id or track.track_id
            cv2.rectangle(output, (x1, y1), (x2, y2), (40, 210, 100), 2)
            cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 210, 100), 2)
        cv2.putText(
            output,
            f"Unique visitors: {self._unique_visitor_count}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
        )
        return output

    def run(self, source: int | str | None = None, output_video: str | Path | None = None) -> int:
        if source in (None, ""):
            raise ValueError("A valid video or camera source is required")

        selected_source = self.settings.source if source is None else source
        final_count = 0
        self.logger.info("video_source_opening session_id=%s source=%s", self.session_id, selected_source)
        capture = cv2.VideoCapture(selected_source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {selected_source}")
        # Reduce latency from stale queued frames on live camera/RTSP backends.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        writer = None
        fps_window_started = time.perf_counter()
        fps_window_frames = 0
        detection_interval = self.settings.detection_skip_frames + 1
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if output_video is not None and writer is None:
                    output_path = Path(output_video).expanduser().resolve()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if not np.isfinite(fps) or fps <= 0:
                        fps = 25.0
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Unable to create output video: {output_path}")
                    self.logger.info("annotated_video_started path=%s fps=%.2f size=%dx%d", output_path, fps, width, height)

                if self.frame_number % detection_interval == 0:
                    detections = self._prepare_detections(frame)
                    updates, exited = self.tracker.update(detections)
                    self._handle_updates(updates, exited)

                annotated = self._annotate(frame) if self.settings.show_preview or writer is not None else None
                if writer is not None:
                    writer.write(annotated)
                if self.settings.show_preview:
                    cv2.imshow("Intelligent Face Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self.frame_number += 1
                fps_window_frames += 1
                now = time.perf_counter()
                elapsed = now - fps_window_started
                if elapsed >= 5.0:
                    self.logger.info(
                        "processing_fps session_id=%s fps=%.2f frames=%d elapsed=%.2f",
                        self.session_id,
                        fps_window_frames / elapsed,
                        fps_window_frames,
                        elapsed,
                    )
                    fps_window_started = now
                    fps_window_frames = 0
        except KeyboardInterrupt:
            self.logger.warning("session_interrupted session_id=%s", self.session_id)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if self.settings.show_preview:
                cv2.destroyAllWindows()

            final_count = self._unique_visitor_count
            try:
                for track in self.tracker.flush():
                    try:
                        self._record_event(track, "exit")
                    except Exception:
                        self.logger.exception("exit_event_failed track_id=%s", track.track_id)
            finally:
                final_count = self.database.unique_visitor_count()
                self.logger.info(
                    "session_finished session_id=%s frames=%d unique_visitors=%d",
                    self.session_id,
                    self.frame_number,
                    final_count,
                )
                self.database.close()
                self.closed = True

        return final_count

    def close(self) -> None:
        if not self.closed:
            self.database.close()
            self.closed = True
