"""Offline Hugging Face / embedding helpers.

After weights are vendored under data/models/, runtime needs no Hub API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_EMBED = ROOT / "data" / "models" / "all-MiniLM-L6-v2"
HUB_ID = "sentence-transformers/all-MiniLM-L6-v2"


def enable_hf_offline() -> None:
    """Block Hub network calls; use local cache / vendored files only."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def resolve_embedding_model(configured: Optional[str] = None) -> str:
    """
    Prefer a local directory of weights so embedding never hits the network.

    Order:
      1. Explicit local path in config (if it exists)
      2. Vendored data/models/all-MiniLM-L6-v2
      3. Configured Hub id (only if offline flags are off / first-time download)
    """
    candidates: list[Path] = []
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            candidates.append((ROOT / p).resolve())
            candidates.append(p.expanduser().resolve())
        else:
            candidates.append(p)
    candidates.append(DEFAULT_LOCAL_EMBED.resolve())

    for path in candidates:
        if path.is_dir() and (path / "config.json").exists():
            enable_hf_offline()
            return str(path)

    # Fall back to Hub id — caller may still download once if online.
    return configured or HUB_ID
