"""Content-addressed inference cache — the cheapest way to cut wall-clock latency.

Identical (model, role, prompt, sampling) requests are answered from disk instead of
re-running the model. SOC streams repeat heavily (same log line, same file, replayed
demo), so hit rates are high in practice. Fully offline: a JSON file under data/cache.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "data" / "cache" / "inference.json"


def make_key(**parts: Any) -> str:
    """Stable hash over any JSON-serializable request description."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class InferenceCache:
    """Thread-safe write-through cache with a bounded, persisted store."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        enabled: bool = True,
        max_entries: int = 2000,
    ):
        self.path = Path(path or DEFAULT_CACHE)
        self.enabled = enabled
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._store: dict[str, str] = {}
        self._dirty = 0
        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if isinstance(data, dict):
                    self._store = {str(k): str(v) for k, v in data.items()}
        except Exception:
            self._store = {}

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        if not self.enabled or value is None:
            return
        with self._lock:
            self._store[key] = value
            if len(self._store) > self.max_entries:
                # Drop oldest insertions; dicts preserve order.
                for k in list(self._store.keys())[: len(self._store) - self.max_entries]:
                    self._store.pop(k, None)
            self._dirty += 1
            should_flush = self._dirty >= 5
        if should_flush:
            self.flush()

    def flush(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            snapshot = dict(self._store)
            self._dirty = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(snapshot))
            tmp.replace(self.path)
        except Exception:
            pass

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._dirty = 0
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception:
            pass

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
