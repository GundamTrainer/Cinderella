from __future__ import annotations

import queue
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AudioFrame:
    sequence_number: int
    frame_count: int
    samples: Any
    input_overflow: bool


class AudioInputBuffer:
    """Bounded callback-to-worker handoff with aggregated, media-free diagnostics."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Audio input queue capacity must be positive.")
        self._queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._sequence_number = 0
        self._diagnostics: Counter[str] = Counter()

    @property
    def capacity(self) -> int:
        return self._queue.maxsize

    def submit(
        self,
        samples: Any,
        *,
        frame_count: int,
        status: Any = None,
    ) -> bool:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
            raise ValueError("Audio callback frame_count must be a nonnegative integer.")
        with self._lock:
            self._sequence_number += 1
            sequence_number = self._sequence_number
        self._record("audio.frame_received", frame_count)
        input_overflow = self.record_input_status(status, frame_count)
        try:
            self._queue.put_nowait(
                AudioFrame(
                    sequence_number=sequence_number,
                    frame_count=frame_count,
                    samples=samples.copy(),
                    input_overflow=input_overflow,
                )
            )
        except queue.Full:
            self._record("audio.frame_dropped", frame_count)
            return False
        if self._queue.qsize() * 4 >= self.capacity * 3:
            self._record("audio.queue_pressure", self._queue.qsize())
        return True

    def record_input_status(self, status: Any, frame_count: int) -> bool:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
            raise ValueError("Audio callback frame_count must be a nonnegative integer.")
        input_overflow = bool(status) and "overflow" in str(status).casefold()
        if input_overflow:
            self._record("audio.input_overflow", frame_count)
        return input_overflow

    def discard(self, *, frame_count: int, status: Any = None, reason: str) -> None:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
            raise ValueError("Audio callback frame_count must be a nonnegative integer.")
        if not reason:
            raise ValueError("Discard reason must not be empty.")
        with self._lock:
            self._sequence_number += 1
        self._record("audio.frame_received", frame_count)
        self.record_input_status(status, frame_count)
        self._record(f"audio.segment_rejected:{reason}", 1)

    def record_sequence_gap(self, missing_chunks: int) -> None:
        if (
            isinstance(missing_chunks, bool)
            or not isinstance(missing_chunks, int)
            or missing_chunks <= 0
        ):
            raise ValueError("missing_chunks must be a positive integer.")
        self._record("audio.sequence_gap", missing_chunks)

    def record_segment_started(self) -> None:
        self._record("audio.segment_started", 1)

    def record_segment(self, *, accepted: bool, reason: str = "unknown") -> None:
        key = "audio.segment_accepted" if accepted else f"audio.segment_rejected:{reason}"
        self._record(key, 1)

    def record_segment_ended(self) -> None:
        self._record("audio.segment_ended", 1)

    def get(self, timeout: float | None = None) -> AudioFrame:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> AudioFrame:
        return self._queue.get_nowait()

    def drain_diagnostics(self) -> dict[str, int]:
        with self._lock:
            snapshot = dict(self._diagnostics)
            self._diagnostics.clear()
        return snapshot

    def _record(self, key: str, count: int) -> None:
        with self._lock:
            self._diagnostics[key] += count
