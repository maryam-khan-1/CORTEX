"""Ollama wrapper: verify tags, generate, retry — tuned for laptop latency.

Latency levers used here (all offline):
  * `think=False` — these Gemma 4 builds are reasoning models. Left on, the whole
    num_predict budget is spent on `message.thinking` and `message.content` comes back
    empty, which is both the slowest and the most dangerous failure mode. Disabling
    reasoning took a deep structured report from ~49s to ~8s on an M1.
  * hard context/prediction caps so Gemma's 128K KV never gets allocated
  * grammar-constrained JSON via Ollama `format` → valid output first try, no retry round-trips
  * content-addressed cache so repeated log lines / replayed demos skip inference entirely
  * keep_alive so the weights stay resident between stages
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import ollama

from core.cache import InferenceCache, make_key
from core.telemetry import TELEMETRY, Stopwatch, Telemetry


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
        *,
        cache: Optional[InferenceCache] = None,
        telemetry: Optional[Telemetry] = None,
    ):
        self.config = config or load_config()
        self.client = client or ollama.Client()
        self._loaded: set[str] = set()
        self.perf = self.config.get("performance", {})
        self.telemetry = telemetry if telemetry is not None else TELEMETRY
        cache_cfg = self.config.get("cache", {})
        self.cache = cache if cache is not None else InferenceCache(
            enabled=bool(cache_cfg.get("enabled", True)),
            max_entries=int(cache_cfg.get("max_entries", 2000)),
        )
        self.available = self._list_models()
        self.fast_model = self._resolve(
            self.config["models"]["fast"],
            self.config["models"].get("fallback_fast"),
        )
        self.deep_model = self._resolve(
            self.config["models"]["deep"],
            self.config["models"].get("fallback_deep"),
        )
        self.keep_alive = self.perf.get("keep_alive", "30m")
        self.think = bool(self.perf.get("think", False))
        # Some builds reject the `think` argument outright; probe once, then stop asking.
        self._think_supported = True

    def _list_models(self) -> list[str]:
        """Current ollama python client: .list().models, m.model (NOT ['name'])."""
        resp = self.client.list()
        return [m.model for m in resp.models if getattr(m, "model", None)]

    def _prefix_match(self, wanted: str) -> Optional[str]:
        if not wanted:
            return None
        for name in self.available:
            if name == wanted:
                return name
        for name in self.available:
            if name.startswith(wanted) or wanted.startswith(name.split(":")[0]):
                if name.startswith(wanted.split(":")[0]) or name.startswith(wanted):
                    return name
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

    def model_for(self, role: str) -> str:
        return self.fast_model if role == "fast" else self.deep_model

    def _ctx_for(self, role: str) -> int:
        if role == "fast":
            return int(self.perf.get("num_ctx_fast", 4096))
        return int(self.perf.get("num_ctx_deep", 8192))

    def _predict_for(self, role: str, override: Optional[int] = None) -> int:
        if override is not None:
            return override
        if role == "fast":
            return int(self.perf.get("num_predict_fast", 160))
        return int(self.perf.get("num_predict_deep", 700))

    def _options(
        self,
        *,
        role: str,
        temperature: float,
        top_p: float,
        top_k: int,
        num_predict: Optional[int] = None,
    ) -> dict[str, Any]:
        # Cap context hard — default Gemma/Ollama 128K KV is what made M1 feel stuck.
        opts: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_ctx": self._ctx_for(role),
            "num_predict": self._predict_for(role, num_predict),
        }
        threads = self.perf.get("num_thread")
        if threads:
            opts["num_thread"] = int(threads)
        return opts

    def _call(self, kwargs: dict[str, Any]) -> Any:
        """client.chat with reasoning disabled, degrading gracefully on older backends."""
        if self._think_supported and not self.think:
            try:
                return self.client.chat(think=False, **kwargs)
            except TypeError:
                self._think_supported = False
            except Exception as e:
                # Ollama raises for models that don't advertise thinking support.
                if "think" not in str(e).lower():
                    raise
                self._think_supported = False
        return self.client.chat(**kwargs)

    @staticmethod
    def _content_of(resp: Any) -> str:
        """Prefer content; if a reasoning model gave us only `thinking`, say so loudly.

        Returning "" here would let a classifier fall through to its default label, which
        for a security tool means silently calling a critical alert benign.
        """
        msg = getattr(resp, "message", None)
        content = (getattr(msg, "content", None) or "").strip() if msg else ""
        if content:
            return content
        thinking = (getattr(msg, "thinking", None) or "") if msg else ""
        if thinking:
            raise RuntimeError(
                "model returned reasoning only (empty content) — raise num_predict or keep think disabled"
            )
        raise RuntimeError("empty response from ollama")

    def ensure_loaded(self, model: str, *, role: str = "deep") -> None:
        """Mark model resident. Optional tiny warmup (off by default — first real call loads it)."""
        if model in self._loaded:
            return
        if self.perf.get("warmup", False):
            try:
                self._call(
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": "ok"}],
                        "options": {
                            "num_predict": 1,
                            "temperature": 0.2,
                            "num_ctx": self._ctx_for(role),
                        },
                        "keep_alive": self.keep_alive,
                    }
                )
            except Exception:
                pass
        self._loaded.add(model)

    def prewarm(self, roles: tuple[str, ...] = ("fast", "deep")) -> None:
        """Load weights before the operator clicks anything — removes cold-start from demos."""
        for role in roles:
            model = self.model_for(role)
            try:
                self._call(
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": "ready"}],
                        "options": {
                            "num_predict": 1,
                            "temperature": 0.2,
                            "num_ctx": self._ctx_for(role),
                        },
                        "keep_alive": self.keep_alive,
                    }
                )
                self._loaded.add(model)
            except Exception:
                pass

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
        num_predict: Optional[int] = None,
        format: Optional[Any] = None,
        label: str = "generate",
        use_cache: bool = True,
    ) -> str:
        """Chat generate with bounded transient retries. Returns assistant text."""
        if model is None:
            model = self.model_for(role)
        options = self._options(
            role=role,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_predict=num_predict,
        )

        # Cache only deterministic-ish requests; high temperature is meant to vary.
        cacheable = use_cache and temperature <= 0.35
        key = ""
        if cacheable:
            key = make_key(
                kind="generate",
                model=model,
                system=system or "",
                prompt=prompt,
                options=options,
                format=format,
            )
            hit = self.cache.get(key)
            if hit is not None:
                self.telemetry.record(
                    label, model=model, role=role, ms=0.4, cached=True, ok=True
                )
                return hit

        self.ensure_loaded(model, role=role)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                with Stopwatch(self.telemetry, label, model=model, role=role):
                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "options": options,
                        "keep_alive": self.keep_alive,
                    }
                    if format is not None:
                        kwargs["format"] = format
                    resp = self._call(kwargs)
                content = self._content_of(resp)
                if cacheable and key:
                    self.cache.set(key, content)
                return content
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(0.35 * (attempt + 1))
                    continue
                raise RuntimeError(f"ollama generate failed after retries: {e}") from e
        raise RuntimeError(f"ollama generate failed: {last_err}")

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: Optional[str] = None,
        tools: Optional[list[Any]] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 64,
        role: str = "deep",
        format: Optional[Any] = None,
        num_predict: Optional[int] = None,
        label: str = "chat",
    ) -> Any:
        """Native chat (supports Gemma 4 tool calling). Returns ChatResponse."""
        if model is None:
            model = self.model_for(role)
        self.ensure_loaded(model, role=role)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": self._options(
                role=role,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_predict=num_predict,
            ),
            "keep_alive": self.keep_alive,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if format is not None:
            kwargs["format"] = format
        with Stopwatch(self.telemetry, label, model=model, role=role):
            return self._call(kwargs)
