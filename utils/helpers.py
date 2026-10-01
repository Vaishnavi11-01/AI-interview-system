"""Common timestamp and filesystem helpers."""

from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
