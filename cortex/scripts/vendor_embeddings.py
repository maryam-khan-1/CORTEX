#!/usr/bin/env python3
"""One-time: vendor MiniLM weights into data/models/ for fully offline RAG.

Prefers copying from the local Hugging Face cache. If missing, downloads once
(network) then writes a self-contained directory CORTEX can load with
HF_HUB_OFFLINE=1 forever after.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "models" / "all-MiniLM-L6-v2"
HUB_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sentence-transformers--all-MiniLM-L6-v2"
)


def copy_from_hub_cache() -> bool:
    snaps = HUB_CACHE / "snapshots"
    if not snaps.is_dir():
        return False
    dirs = sorted(snaps.iterdir())
    if not dirs:
        return False
    src = dirs[-1]
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    # Follow symlinks so the vendor dir is self-contained
    shutil.copytree(src, DEST, symlinks=False, ignore_dangling_symlinks=False)
    # copytree with symlinks=False still may copy link targets on some platforms;
    # ensure real files:
    for p in DEST.rglob("*"):
        if p.is_symlink():
            target = p.resolve()
            p.unlink()
            if target.is_dir():
                shutil.copytree(target, p)
            else:
                shutil.copy2(target, p)
    return (DEST / "config.json").exists()


def download_once() -> None:
    from huggingface_hub import snapshot_download

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        shutil.rmtree(DEST)
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=str(DEST),
        local_dir_use_symlinks=False,
    )


def main() -> None:
    if copy_from_hub_cache():
        print(f"Vendored from HF cache → {DEST}")
    else:
        print("No local HF cache hit; downloading once from Hub…")
        download_once()
        print(f"Downloaded → {DEST}")
    size = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    print(f"Ready ({size / 1e6:.1f} MB). Runtime can set HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"vendor_embeddings failed: {e}", file=sys.stderr)
        sys.exit(1)
