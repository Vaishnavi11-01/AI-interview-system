"""Local browser interface for uploading and processing visitor videos."""

import argparse
from pathlib import Path
import threading
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from report import generate_report
from utils.config import load_settings


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def create_app(config_path: str | Path = "config.json", use_gpu: bool = False, pipeline_factory=None) -> Flask:
    settings = load_settings(config_path)
    root_dir = settings.logs_dir.parent
    upload_dir = root_dir / "uploads"
    output_dir = root_dir / "output"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
    state_lock = threading.Lock()
    current_job: dict | None = None

    def process_video(job_id: str, source_path: Path, output_path: Path) -> None:
        nonlocal current_job
        with state_lock:
            if current_job and current_job["id"] == job_id:
                current_job["status"] = "processing"
                current_job["message"] = "Analyzing frames. This can take a while for long videos."

        pipeline = None
        try:
            factory = pipeline_factory
            if factory is None:
                from pipeline import FaceTrackingPipeline

                factory = FaceTrackingPipeline
            pipeline = factory(settings, use_gpu=use_gpu)
            visitor_count = pipeline.run(source=source_path, output_video=output_path)
            report_path = output_dir / "visitor_report.html"
            generate_report(settings.database_path, report_path, video_path=output_path)
            with state_lock:
                if current_job and current_job["id"] == job_id:
                    current_job.update(
                        status="completed",
                        message="Processing finished.",
                        visitor_count=visitor_count,
                        output_filename=output_path.name,
                        report_filename=report_path.name,
                    )
        except Exception as error:
            app.logger.exception("Video processing failed for job %s", job_id)
            with state_lock:
                if current_job and current_job["id"] == job_id:
                    current_job.update(status="failed", message=str(error))
            output_path.unlink(missing_ok=True)
        finally:
            if pipeline is not None:
                pipeline.close()
            source_path.unlink(missing_ok=True)

    @app.get("/")
    def index():
        return render_template("upload.html")

    @app.post("/upload")
    def upload_video():
        nonlocal current_job
        uploaded_file = request.files.get("video")
        if uploaded_file is None or not uploaded_file.filename:
            return jsonify(error="Choose a video file first."), 400

        safe_name = secure_filename(uploaded_file.filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            return jsonify(error="Unsupported file type. Choose MP4, AVI, MOV, MKV, M4V, or WebM."), 400

        with state_lock:
            if current_job and current_job["status"] in {"queued", "processing"}:
                return jsonify(error="A video is already being processed. Wait for it to finish."), 409

            job_id = uuid.uuid4().hex
            source_path = upload_dir / f"{job_id}{suffix}"
            output_path = output_dir / f"annotated_{job_id}.mp4"
            uploaded_file.save(source_path)
            current_job = {
                "id": job_id,
                "status": "queued",
                "message": "Upload received. Preparing the processing pipeline…",
                "filename": safe_name,
            }

        worker = threading.Thread(
            target=process_video,
            args=(job_id, source_path, output_path),
            name=f"video-{job_id[:8]}",
            daemon=True,
        )
        worker.start()
        return jsonify(job_id=job_id), 202

    @app.get("/api/status/<job_id>")
    def job_status(job_id: str):
        with state_lock:
            if current_job is None or current_job["id"] != job_id:
                return jsonify(error="Processing job not found."), 404
            return jsonify(dict(current_job))

    @app.get("/outputs/<path:filename>")
    def output_file(filename: str):
        return send_from_directory(output_dir, filename, as_attachment=False)

    @app.get("/logs/<path:filename>")
    def log_file(filename: str):
        return send_from_directory(settings.logs_dir, filename, as_attachment=False)

    @app.errorhandler(413)
    def file_too_large(_error):
        return jsonify(error="The file is too large. The upload limit is 2 GB."), 413

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Local video upload interface for the visitor tracker")
    parser.add_argument("--config", default="config.json", help="Path to JSON settings")
    parser.add_argument("--host", default="127.0.0.1", help="Interface address (default: local machine only)")
    parser.add_argument("--port", type=int, default=5000, help="Web interface port")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA-enabled inference")
    args = parser.parse_args()
    app = create_app(args.config, use_gpu=args.gpu)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())