from __future__ import annotations

import threading
from queue import Empty, Full, Queue

from .event_model import Event


class EventBusFullError(Full):
    """Raised when a bounded event queue is full."""


class EventBusClosedError(RuntimeError):
    """Raised when publishing to a closed event bus."""


class EventBus:
    """Thread-safe bounded event bus for runtime events."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Event bus capacity must be greater than zero.")
        self._queue: Queue[Event] = Queue(maxsize=capacity)
        self._lock = threading.RLock()
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def publish(self, event: Event, *, block: bool = True, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                raise EventBusClosedError("Event bus is closed and cannot accept new events.")
        try:
            if block:
                self._queue.put(event, block=True, timeout=timeout)
            else:
                self._queue.put_nowait(event)
        except Full as exc:
            raise EventBusFullError(f"Event bus is at capacity ({self._queue.maxsize}).") from exc

    def poll(self, *, block: bool = True, timeout: float | None = None) -> Event | None:
        try:
            if block:
                return self._queue.get(timeout=timeout)
            return self._queue.get_nowait()
        except Empty:
            return None

    def drain(self, *, max_events: int | None = None, block: bool = False, timeout: float | None = None) -> list[Event]:
        drained: list[Event] = []
        while True:
            if max_events is not None and len(drained) >= max_events:
                break
            event = self.poll(block=block, timeout=timeout)
            if event is None:
                break
            drained.append(event)
        return drained

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

    def size(self) -> int:
        return self._queue.qsize()

    def __len__(self) -> int:
        return self.size()
