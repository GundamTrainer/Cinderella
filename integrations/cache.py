from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


def cache_key(provider_id: str, arguments: dict[str, object]) -> str:
    encoded = json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(f"{provider_id}:{encoded}".encode()).hexdigest()


@dataclass(frozen=True)
class _Entry[T]:
    value: T
    expires_at: float


class MemoryCache[T]:
    """Thread-safe bounded TTL cache; values are evicted in least-recently-used order."""

    def __init__(
        self, *, capacity: int = 256, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive.")
        self._capacity = capacity
        self._clock = clock
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= self._clock():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.value

    def put(self, key: str, value: T, *, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._entries[key] = _Entry(value, self._clock() + ttl_seconds)
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
