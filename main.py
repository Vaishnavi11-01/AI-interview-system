import argparse
from pathlib import Path

from database.db import VisitorDatabase
from report import generate_report
from utils.config import load_settings


def parse_source(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO + InsightFace unique visitor tracker")
    parser.add_argument("--config", default="config.json", help="Path to JSON settings")
    parser.add_argument("--source", help="Video file, camera index, or RTSP URL (overrides config)")
    parser.add_argument("--output-video", help="Optional path to save an annotated MP4 of the processed stream")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA-enabled ONNX Runtime for InsightFace")
    parser.add_argument("--count", action="store_true", help="Print registered unique visitor count and exit")
    parser.add_argument("--report", action="store_true", help="Build an HTML gallery of visitor snapshots and exit")
    args = parser.parse_args()
    settings = load_settings(args.config)

    if args.report:
        report_path = generate_report(settings.database_path, settings.logs_dir.parent / "visitor_report.html")
        print(f"Visitor gallery created: {report_path}")
        return 0

    if args.count:
        database = VisitorDatabase(settings.database_path)
        try:
            print(database.unique_visitor_count())
        finally:
            database.close()
        return 0

    selected_source = args.source if args.source is not None else settings.source
    if selected_source in (None, "", 0):
        raise SystemExit("No source configured. Pass --source 0 for webcam or --source 'video.mp4' for a file.")

    # Import model-dependent packages only when starting the pipeline.
    from pipeline import FaceTrackingPipeline

    pipeline = FaceTrackingPipeline(settings, use_gpu=args.gpu)
    count = pipeline.run(
        source=parse_source(args.source) if args.source is not None else selected_source,
        output_video=args.output_video,
    )
    print(f"Unique registered visitors: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
