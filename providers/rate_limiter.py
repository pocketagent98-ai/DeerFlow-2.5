"""Token-bucket rate limiter — process-local requests-per-minute pacing."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    capacity: int = 40
    window_seconds: float = 60.0
    tokens: float = field(init=False)
    updated: float = field(init=False, default_factory=time.monotonic)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.updated
        if elapsed > 0:
            rate = self.capacity / self.window_seconds
            self.tokens = min(float(self.capacity), self.tokens + elapsed * rate)
            self.updated = now

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    def time_until_token(self) -> float:
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                return 0.0
            rate = self.capacity / self.window_seconds
            return (1.0 - self.tokens) / rate
