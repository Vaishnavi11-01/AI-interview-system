# Intelligent Face Tracker with Auto-Registration

A Python application for detecting faces in a video file, webcam, or RTSP stream; assigning persistent visitor IDs using InsightFace embeddings; tracking appearances; and recording timestamped entry/exit snapshots in SQLite and a rotating log file.

> **Submission item:** Record and add your own Loom/YouTube demonstration URL below before submitting. The task requires an explanatory video; a placeholder is not a valid submission.
>
> **Sample evidence:** A sample video was processed. Genuine event logs, four snapshot images, and the SQLite database are included under `sample_output/`. `python main.py --report` generates a browsable gallery from the active database.

## Features

- YOLO face detection, with adjustable confidence, image size, and frame-skip interval.
- InsightFace `buffalo_l` ArcFace embeddings; cosine-similarity re-identification of previously registered visitors.
- Greedy IoU + embedding association to maintain tracks across detection cycles.
- Automatic registration on the first observation of a new identity. Returning visitors retain the same visitor ID and do not increase the visitor count.
- One database-protected entry and exit event per track/session, timestamped JPEG crops, ISO-8601 UTC timestamps, and structured `events.log` messages.
- SQLite persistence with WAL mode, full synchronous commits, foreign keys, unique constraints, and unique visitor count query.
- Webcam index, local video file, and RTSP input; optional annotated OpenCV preview.
- CLI count retrieval: `python main.py --count`.
- Visual event gallery: `python main.py --report` creates `visitor_report.html`, which can be opened in a web browser and regenerated after each run.
- Optional annotated MP4 output: pass `--output-video output/annotated.mp4` to save the processed stream with visitor boxes and IDs. Entry/exit snapshots remain separate images for the audit log.

## Project structure

```text
AI-Interview-System/
├── main.py                 # CLI entry point
├── config.json
├── requirements.txt
├── pipeline.py             # Processing orchestration
├── report.py               # HTML event gallery
├── detector/               # YOLO inference and crop preprocessing
├── recognition/            # InsightFace, embeddings, identity matcher
├── tracker/                # Track manager and optional ByteTrack factory
├── database/               # SQLite setup, models, and queries
├── logger/                 # Event log and atomic image snapshots
├── utils/                  # Configuration, timestamp helpers, timer
├── logs/                   # Runtime entries, exits, and events.log
├── output/                 # Optional annotated video output
└── sample_output/          # Sample-run evidence for submission
```

The active `tracker_manager.py` uses the project's IoU + ArcFace appearance tracker. `tracker/bytetrack.py` provides a factory for Ultralytics' real ByteTrack backend if you choose to swap trackers; it is not the active tracker by default.

## AI planning and workflow

1. **Plan:** Separate source/configuration, YOLO detection, InsightFace embedding, appearance/IoU tracking, persistence, and event handling so models can be swapped without changing the database contract.
2. **Build:** Add a JSON configuration surface; load face-specific YOLO weights; compute normalized ArcFace embeddings; match tracks on detection cycles; re-identify newly-created tracks against the visitor gallery.
3. **Persist:** Commit a visitor record before its entry event. Write an event JPEG atomically, insert event metadata, and remove the image if insertion fails. Database uniqueness constraints suppress duplicate event writes. On normal stream end, remaining active tracks receive an exit event.
4. **Verify:** Run the sample video, inspect `logs/events.log`, check entry/exit image folders, query the SQLite tables, then replay the same stream to verify visitor IDs are re-used. Tune recognition and track thresholds against the camera and lighting conditions.
5. **Demonstrate:** Show setup, a first registration, recognition of a returning identity, entry/exit images and rows, the final count, and RTSP configuration in the required video.

## Architecture

```mermaid
flowchart LR
    A[Video / Webcam / RTSP] --> B[OpenCV frame reader]
    B -->|every N+1 frames| C[YOLO face detector]
    C --> D[InsightFace ArcFace embeddings]
    D --> E[IoU + cosine track association]
    E --> F{New track?}
    F -->|yes| G[SQLite visitor re-identification / registration]
    F -->|existing| E
    G --> H[Entry event + JPEG crop]
    E --> I[Track timeout / stream end]
    I --> J[Exit event + JPEG crop]
    H --> K[(SQLite events + visitors)]
    J --> K
    H --> L[logs/entries/YYYY-MM-DD]
    J --> M[logs/exits/YYYY-MM-DD]
    C --> N[logs/events.log]
    G --> N
    E --> N
    I --> N
```

The tracking component advances its missed-detection age only on detection cycles. With `detection_skip_frames = N`, a detection runs every `N + 1` frames. An exit is generated after `max_missed_cycles` consecutive detection cycles without a match (or when the stream ends). Consequently, occlusion and threshold tuning affect entry/exit accuracy; validate on the target camera before production use.

## Setup

1. Use Python 3.12 (tested) and create/activate a virtual environment.
2. Install packages with `pip install -r requirements.txt`. CPU ONNX Runtime is installed by default. For CUDA, replace `onnxruntime` with a compatible `onnxruntime-gpu` build and install the matching CUDA/cuDNN runtime; then pass `--gpu`.
3. Obtain a **face-trained Ultralytics YOLO weights file** at `models/yolov8n-face.pt`, or set its location in `config.json`. Generic COCO YOLO weights do not detect faces. The development copy used here is git-ignored; re-download it from [akanametov/yolo-face](https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt) as needed. Review the GPL-3.0 license and use weights whose license permits your use.
4. InsightFace downloads its `buffalo_l` model pack on first initialization. Ensure network access for this initial setup, or pre-cache the model pack in the InsightFace model directory.
5. Obtain the sample video from the task's [Google Drive folder](https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing), then run it with `python main.py --source "path/to/video.mp4" --output-video "output/annotated.mp4"` to save a viewable annotated MP4. Use `--source 0` for a webcam or `--source "rtsp://user:password@host:554/stream"` for RTSP. Avoid committing credentials.
6. Press `q` to stop preview. Set `show_preview` to `false` in headless environments. Generated events are stored under `logs/`; the SQLite database is at `data/visitors.sqlite3`.
7. Query cumulative registered identities with `python main.py --count`. Create a visual gallery with `python main.py --report`, then open `visitor_report.html`; regenerate it after later runs to refresh. The count is global to the configured database, not limited to one run. Use a fresh database when evaluating a standalone sample.

### Configuration

`detection_skip_frames` is the required frame skip setting. Relative model/database/log paths resolve from the directory containing `config.json`.

```json
{
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
  "show_preview": true
}
```

The recognition threshold is a cosine similarity in `[0, 1]`; larger thresholds are more conservative and may split the same visitor into multiple IDs. Calibrate with representative face crops. Embeddings and face crops are biometric data: restrict access, obtain appropriate consent, and establish retention/deletion policies before real-world use. SQLite and file storage here are local and are not encrypted.

## Data and event contract

- Visitor gallery: `visitors(visitor_id, first_seen, embedding, embedding_dim)`.
- Events: `events(session_id, camera_id, track_id, visitor_id, event_type, timestamp, image_path, bbox)`.
- Entry images: `logs/entries/YYYY-MM-DD/`; exit images: `logs/exits/YYYY-MM-DD/`.
- Critical lifecycle records (embedding generation, registration, recognition, tracking, entry, exit, failures, and session summary): `logs/events.log` with rotation.
- `UNIQUE(session_id, camera_id, track_id, event_type)` prevents duplicate entry/exit rows for a track. A camera session ending normally closes active tracks as exits. A hard power loss cannot reliably infer physical exits; the next session receives a new session ID.

## Compute estimate

Planning estimate for one 640-pixel stream, subject to the selected weights, hardware, scene density, and detection skip interval:

| Resource | Approximate budget | Notes |
| --- | --- | --- |
| CPU-only | 4+ modern cores; 1–2 GB process/model RAM; roughly 3–12 processed frames/sec | YOLO and InsightFace both run on CPU; throughput falls with crowded frames. |
| NVIDIA GPU | 4–6 GB VRAM recommended; roughly 15–30 processed frames/sec | Requires compatible CUDA, cuDNN, ONNX Runtime GPU, and Ultralytics support. |
| Storage | About 50–200 KB per event JPEG, plus SQLite and rotating logs | Depends on crop dimensions/JPEG quality and number of tracked visits. |

These are rough capacity estimates, not benchmark claims. Measure the actual camera/model pair; frame skipping reduces inference load approximately in proportion to `1 / (detection_skip_frames + 1)`, but can reduce tracking accuracy for fast motion.

## Assumptions and limitations

- One configured source/camera is processed per process. Use a separate process/config/database or add a multi-camera service layer for multiple streams.
- A visitor is a face identity, not a person count inferred from body detections. Similarity errors, profile views, masks, blur, and lighting can cause false matches or duplicate identities.
- Entry/exit mean the start/end of a continuous in-frame track, not crossing a user-defined doorway line. Add a virtual line/zone if physical ingress/egress semantics are required.
- Tracking association is the small in-process IoU/cosine tracker by default; it does not provide motion compensation or the robustness of a tuned ByteTrack/DeepSORT deployment. An optional ByteTrack factory is available in `tracker/bytetrack.py` but is not wired into the default pipeline.
- Local SQLite WAL is appropriate for a single writer. Use a server database and queue-backed event writer for a multi-worker deployment.
- A `visitor_id` is created for each new unmatched embedding, and the unique visitor count is the number of rows in `visitors` across the configured database.

## Demonstration video

**Loom/YouTube URL:** TODO — replace with the submitted public/unlisted demonstration link.
See [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for a short recording walkthrough. The recording and upload require your own Loom/YouTube account.

## Sample artifacts

See [sample_output/README.md](sample_output/README.md) for the packaged sample run's counts, event evidence, and artifact paths.

This project is a part of a hackathon run by https://katomaran.com
