"""Reusable elapsed-time context manager for profiling pipeline stages."""

from time import perf_counter
from types import TracebackType
from typing import Self


class Timer:
    def __enter__(self) -> Self:
        self.started_at = perf_counter()
        self.elapsed_seconds = 0.0
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.elapsed_seconds = perf_counter() - self.started_at
