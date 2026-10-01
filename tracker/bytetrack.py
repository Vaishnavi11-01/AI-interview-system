"""Factory for Ultralytics' built-in ByteTrack tracker.

The current pipeline uses ``IoUAppearanceTracker`` in ``tracker_manager.py``
because it associates both spatial boxes and ArcFace embeddings. This factory
is provided for deployments that want to replace it with actual ByteTrack.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any


def create_bytetrack(track_buffer: int = 30) -> Any:
    """Construct Ultralytics BYTETracker using its packaged default settings."""
    # Keep this optional backend lazy: importing the module does not load the
    # tracker package's optional LAP dependency.
    from ultralytics import __file__ as ultralytics_init
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML

    config_path = Path(ultralytics_init).parent / "cfg" / "trackers" / "bytetrack.yaml"
    settings = YAML.load(str(config_path))
    settings["track_buffer"] = max(1, int(track_buffer))
    return BYTETracker(SimpleNamespace(**settings))
