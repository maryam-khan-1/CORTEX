"""Ollama wrapper: verify tags, generate, retry."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import ollama


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.json"


def load_config(path: Optional[Path] = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    with open(cfg_path) as f:
        return json.load(f)


class Engine:
    """Local Ollama engine with startup tag verification and graceful fallback."""

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        client: Optional[ollama.Client] = None,
    ):
        self.config = config or load_config()
        self.client = client or ollama.Client()
        self._loaded: set[str] = set()
        self.available = self._list_models()
        self.fast_model = self._resolve(
            self.config["models"]["fast"],
            self.config["models"].get("fallback_fast"),
        )
        self.deep_model = self._resolve(
            self.config["models"]["deep"],
            self.config["models"].get("fallback_deep"),
        )

    def _list_models(self) -> list[str]:
        """Current ollama python client: .list().models, m.model (NOT ['name'])."""
        resp = self.client.list()
        return [m.model for m in resp.models if getattr(m, "model", None)]

    def _prefix_match(self, wanted: str) -> Optional[str]:
        if not wanted:
            return None
        # Exact first, then prefix (tags carry :suffix).
        for name in self.available:
            if name == wanted:
                return name
        for name in self.available:
            if name.startswith(wanted) or wanted.startswith(name.split(":")[0]):
                # Prefer prefix match where installed name starts with wanted prefix
                if name.startswith(wanted.split(":")[0]) or name.startswith(wanted):
                    return name
        # Broader: wanted is a prefix of installed, or installed starts with wanted
        for name in self.available:
            base_wanted = wanted.split(":")[0]
            base_name = name.split(":")[0]
            if name.startswith(wanted) or wanted.startswith(name) or base_wanted == base_name:
                return name
        return None

    def _resolve(self, primary: str, fallback: Optional[str] = None) -> str:
        hit = self._prefix_match(primary)
        if hit:
            return hit
        if fallback:
            hit = self._prefix_match(fallback)
            if hit:
                return hit
        raise RuntimeError(
            f"No Ollama model matching {primary!r}"
            + (f" or fallback {fallback!r}" if fallback else "")
            + f". Available: {self.available}"
        )

    def ensure_loaded(self, model: str) -> None:
        """Load model on demand (no-op if already warmed)."""
        if model in self._loaded:
            return
        # A tiny generate warms the model into memory.
        try:
            self.client.chat(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                options={"num_predict": 1, "temperature": 0.2},
            )
        except Exception:
            # Still mark attempted; subsequent calls will surface real errors.
            pass
        self._loaded.add(model)

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 64,
        role: str = "deep",
        max_retries: int = 2,
    ) -> str:
        """Chat generate with bounded transient retries. Returns assistant text."""
        if model is None:
            model = self.fast_model if role == "fast" else self.deep_model
        self.ensure_loaded(model)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    },
                )
                content = resp.message.content if resp.message else None
                if content is None:
                    raise RuntimeError("empty response from ollama")
                return content
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"ollama generate failed after retries: {e}") from e
        raise RuntimeError(f"ollama generate failed: {last_err}")
