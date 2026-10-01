# Demo recording script (about 2 minutes)

Record your screen with Loom or another recorder. Show the actual application artifacts; do not claim that the sample run proves production accuracy.

1. **Opening (10 sec):** “This is my Intelligent Face Tracker hackathon project. It processes a video using a face-trained YOLO model for detection, InsightFace embeddings for re-identification, and SQLite for visitor and event persistence.”
2. **Show annotated video (30 sec):** Open `output/annotated_sample.mp4` in a video player. Point out the face boxes, visitor labels and on-screen unique count. Explain that processed playback is an optional visual aid, separate from the required event snapshots.
3. **Show gallery (25 sec):** Open `visitor_report.html` in a browser. Show the unique visitor, entry and exit totals; try the Entries/Exits filters. Explain that each card is an event snapshot and timestamp, as required by the logging specification.
4. **Show evidence (25 sec):** Show `sample_output/logs/events.log`, the dated entry/exit JPEG folders, and the copied SQLite database in `sample_output/data/visitors.sqlite3`. Point out the session's entry and exit records.
5. **Show configuration (20 sec):** Open `config.json` and identify `detection_skip_frames`, model weights, and the database/log paths. Mention that `--source` accepts a video, webcam index, or RTSP URL.
6. **Run commands (20 sec):** In a terminal, show `python main.py --count` and `python main.py --report`. For a fresh run, use `python main.py --source "sample_video.mp4" --output-video "output/annotated.mp4"`.
7. **Close (10 sec):** Mention known limitations: recognition thresholds and face detection need validation against the target camera, and biometric model licensing/consent must be considered.

After recording, upload the video as public or unlisted, replace the Loom/YouTube `TODO` in `README.md` with its URL, and watch the published link once to verify access.
