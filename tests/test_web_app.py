import io
import json
import threading

from database.db import VisitorDatabase
from web_app import create_app


def test_upload_rejects_unsupported_file_type(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "source": "",
        "detector_model": "models/missing.pt",
        "database_path": "data/visitors.sqlite3",
        "logs_dir": "logs",
    }), encoding="utf-8")
    app = create_app(config_path)
    app.testing = True

    response = app.test_client().post(
        "/upload",
        data={"video": (io.BytesIO(b"not a video"), "notes.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json()["error"]


def test_uploaded_video_runs_pipeline_and_serves_result(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "source": "",
        "detector_model": "models/missing.pt",
        "database_path": "data/visitors.sqlite3",
        "logs_dir": "logs",
    }), encoding="utf-8")
    finished = threading.Event()

    class FakePipeline:
        def __init__(self, settings, use_gpu=False):
            self.settings = settings

        def run(self, source, output_video):
            output_video.write_bytes(b"annotated-video")
            database = VisitorDatabase(self.settings.database_path)
            database.close()
            return 2

        def close(self):
            finished.set()

    app = create_app(config_path, pipeline_factory=FakePipeline)
    app.testing = True
    client = app.test_client()
    response = client.post(
        "/upload",
        data={"video": (io.BytesIO(b"source-video"), "interview.mp4")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    assert finished.wait(timeout=5)

    status = client.get(f"/api/status/{job_id}").get_json()
    assert status["status"] == "completed"
    assert status["visitor_count"] == 2
    assert client.get(f"/outputs/{status['output_filename']}").data == b"annotated-video"
    assert client.get(f"/outputs/{status['report_filename']}").status_code == 200