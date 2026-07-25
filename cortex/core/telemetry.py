"""Inference telemetry — latency and cache accounting for the latency claims in the UI.

Everything is in-process and offline; no metrics leave the host.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional


@dataclass
class CallRecord:
    label: str  # e.g. "stage1", "harness", "agent-step", "critique"
    model: str
    role: str  # fast | deep
    ms: float
    cached: bool = False
    ok: bool = True


@dataclass
class Telemetry:
    """Rolling latency/cache stats. Thread-safe: the autonomy loop writes from a worker."""

    records: Deque[CallRecord] = field(default_factory=lambda: deque(maxlen=400))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    total_calls: int = 0
    cache_hits: int = 0
    total_ms: float = 0.0
    saved_ms: float = 0.0  # estimated wall-clock avoided by cache hits

    def record(
        self,
        label: str,
        *,
        model: str,
        role: str,
        ms: float,
        cached: bool = False,
        ok: bool = True,
    ) -> None:
        with self._lock:
            self.records.append(
                CallRecord(label=label, model=model, role=role, ms=ms, cached=cached, ok=ok)
            )
            self.total_calls += 1
            self.total_ms += ms
            if cached:
                self.cache_hits += 1
                # Credit the cache with the current average cost of an uncached call.
                self.saved_ms += max(self._avg_uncached_locked() - ms, 0.0)

    def _avg_uncached_locked(self) -> float:
        live = [r.ms for r in self.records if not r.cached]
        return sum(live) / len(live) if live else 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            live = [r for r in self.records if not r.cached]
            by_role: dict[str, list[float]] = {}
            for r in live:
                by_role.setdefault(r.role, []).append(r.ms)
            hit_rate = (self.cache_hits / self.total_calls) if self.total_calls else 0.0
            return {
                "total_calls": self.total_calls,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": hit_rate,
                "avg_ms": (sum(x.ms for x in live) / len(live)) if live else 0.0,
                "p50_ms": _percentile([x.ms for x in live], 0.50),
                "p95_ms": _percentile([x.ms for x in live], 0.95),
                "fast_avg_ms": _mean(by_role.get("fast", [])),
                "deep_avg_ms": _mean(by_role.get("deep", [])),
                "saved_s": self.saved_ms / 1000.0,
                "recent": [
                    {
                        "label": r.label,
                        "role": r.role,
                        "ms": round(r.ms),
                        "cached": r.cached,
                        "ok": r.ok,
                    }
                    for r in list(self.records)[-12:][::-1]
                ],
            }

    def reset(self) -> None:
        with self._lock:
            self.records.clear()
            self.total_calls = 0
            self.cache_hits = 0
            self.total_ms = 0.0
            self.saved_ms = 0.0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


class Stopwatch:
    """Context manager that reports elapsed ms into a Telemetry instance."""

    def __init__(
        self,
        telemetry: Optional[Telemetry],
        label: str,
        *,
        model: str = "",
        role: str = "deep",
        cached: bool = False,
    ):
        self.telemetry = telemetry
        self.label = label
        self.model = model
        self.role = role
        self.cached = cached
        self.ok = True
        self.ms = 0.0
        self._t0 = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
        self.ok = exc_type is None
        if self.telemetry is not None:
            self.telemetry.record(
                self.label,
                model=self.model,
                role=self.role,
                ms=self.ms,
                cached=self.cached,
                ok=self.ok,
            )
        return False


TELEMETRY = Telemetry()
