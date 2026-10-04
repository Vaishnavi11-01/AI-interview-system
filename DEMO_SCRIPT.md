# Hackathon presentation script (5–8 minutes)

Use this as a guide, not as a claim that every deployment condition has been tested. Show the live upload UI if models are installed and warmed up; otherwise show the saved sample artifacts. Do not claim sample counts or FPS unless you have verified them on the exact run and hardware you present.

## 1. Introduction (about 30 seconds)

“Hello everyone. My project is **Intelligent Face Tracker with Auto-Registration and Visitor Counting**. It processes a local video, webcam, or RTSP stream. It detects faces, matches returning visitors using face embeddings, automatically registers unmatched faces, tracks them through the stream, records entry and exit events, and persists visitor data in SQLite with images and logs.”

## 2. Problem and goal (about 30 seconds)

“Face detection alone cannot tell whether someone has visited before. The project maintains a persistent gallery of face embeddings so a returning face can reuse its visitor ID. The unique visitor count comes from registered identities, while entry and exit records represent observed track sessions. Those are different counts.”

## 3. Technologies (about 30 seconds)

“The application is written in Python. Ultralytics YOLO uses face-trained weights for face detection; InsightFace with ArcFace generates recognition embeddings; OpenCV reads and optionally annotates video; a lightweight in-project IoU tracker associates detections between detection cycles; SQLite stores visitors and events; Python logging writes the event log. The configuration is JSON, and I added a local Flask page for video uploads.”

Be accurate: the active tracker is the project’s `IoUAppearanceTracker`, not ByteTrack or DeepSORT. An optional ByteTrack factory exists, but it is not used by the default pipeline.

## 4. Architecture and data flow (about 60 seconds)

Show the architecture section in `README.md` or explain this flow:

“OpenCV reads frames from a file, camera, or RTSP source. On configured detection cycles, YOLO produces face boxes and crops. The tracker associates boxes with active tracks. When it creates a new track, InsightFace generates an embedding and the database matcher compares it with stored normalized embeddings. A match reuses that visitor ID; otherwise the system registers a new visitor. The first identified track observation creates an entry event. When a track has been missing for the configured number of detection cycles—or when a stream ends—the system records an exit. Event metadata and visitor embeddings persist in SQLite; event crops go into dated entry/exit folders, and lifecycle messages go into `logs/events.log`.”

Clarify that face recognition answers “who is this?” and tracking answers “is this the same currently visible detection from the previous cycle?” Track IDs are temporary; visitor IDs are persistent. An exit means a track disappeared from this stream, not proof of crossing a physical doorway.

## 5. Project structure (about 35 seconds)

Show the repository in VS Code. Point out:

- `detector/`: YOLO wrapper and face crop preprocessing.
- `recognition/`: InsightFace embedding and visitor matcher.
- `tracker/`: IoU tracking and optional ByteTrack integration.
- `database/`: SQLite schema and identity/event queries.
- `logger/`: structured logs and image snapshots.
- `pipeline.py`: connects the processing steps.
- `web_app.py` and `templates/`: local browser upload experience.
- `config.json`, `logs/`, `data/`, and `output/`: settings and generated artifacts.

## 6. Browser upload and live demo (about 2 minutes)

Start the local UI with `python web_app.py`, then open `http://127.0.0.1:5000` in a browser. Select or drag in a video and choose **Upload and process video**.

Say: “The upload is processed locally by the existing pipeline. The page reports the job status, then provides the annotated MP4, the unique visitor total, and an event report. On the annotated result, boxes and labels show active tracks; the visitor count is displayed on the frame.”

If you do not have a fresh video/model setup ready, open a previously generated annotated MP4 and the event report instead. Do not imply that a browser upload is a cloud service: the Flask development server defaults to local-only access at `127.0.0.1`.

## 7. Configuration and performance (about 40 seconds)

Show `config.json`. “`detection_skip_frames` controls detection cadence: with value $N$, YOLO runs every $N+1$ frames. Skipped frames are still read and written/displayed, but this implementation does not run a separate tracker update on each skipped frame. `max_missed_cycles` controls when a track is considered gone. Model path, similarity thresholds, database path, and log directory are configurable.”

“Performance depends on resolution, scene, model, CPU/GPU, and storage. The project logs measured processing FPS periodically. I will only report a number after checking the actual run log; it is not a guaranteed rate.”

## 8. Persistence and results (about 50 seconds)

Show the generated artifacts:

- `logs/events.log`
- `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`
- `data/visitors.sqlite3`
- the annotated MP4 and event report under `output/`

“The visitors table stores each registered identity and its embedding, so registrations persist across application restarts. The events table records entry/exit metadata and prevents duplicate event types for the same track within a session. For a normal end of stream or Ctrl+C, remaining active tracks are flushed as exits. A hard crash cannot reliably infer exits, but already committed visitor and event data remains in SQLite.”

Before recording, run the count/report commands and quote the values they actually show. The packaged sample evidence documents two visitor IDs and four event rows; do not substitute unrelated numbers from another run.

## 9. Limitations and next steps (about 30 seconds)

“Recognition can be affected by image quality, pose, lighting, and threshold choice. IoU association can lose or switch tracks during fast movement or occlusion. An exit is a track timeout, not a doorway crossing. The next improvements could include motion-aware tracking, multi-camera coordination, richer upload/job management, and deployment behind a production web server. Face images and embeddings are sensitive biometric data, so consent, access control, retention, and model licensing matter.”

## 10. Conclusion (about 20 seconds)

“This project connects face detection, automatic identity registration and re-identification, track-level entry/exit logging, local image storage, and persistent unique visitor counting. I’ve shown how a video is uploaded locally, processed through the pipeline, and reviewed through its annotated video and event records. Thank you.”

## Likely questions

**Why use tracking if you have face recognition?** Recognition assigns a persistent identity; tracking associates current detections across nearby frames and avoids recomputing an embedding on every detection cycle.

**Why YOLO and InsightFace?** YOLO localizes faces; InsightFace/ArcFace creates embeddings suitable for similarity matching. They perform different jobs.

**How does the unique count avoid counting a returning visitor again?** New track embeddings are compared with the persistent SQLite gallery. If the best cosine similarity passes the configured threshold, the stored visitor ID is reused.

**Does every visit produce exactly one entry and one exit?** The database uniqueness constraint enforces at most one of each event type per track and session. Normal stream completion flushes active tracks. Track fragmentation, missed detections, or process crashes can still affect correspondence to a real-world visit, so this is a system limitation rather than a guarantee of perfect doorway semantics.

**Why SQLite?** It provides durable local storage without a separate database server and is appropriate for this single-process demonstration.

After recording, upload the presentation video as public or unlisted, replace the Loom/YouTube placeholder in `README.md` with its URL, and verify that the published link opens.
