"""Model router: live-discovered chain, NVIDIA-first, z.ai fallback.

Chain construction (automatic, no hardcoded model ids):
1. Discover every model available on NVIDIA NIM with your key (live /v1/models).
2. Rank them by power (see ranker.py) and take the top `nvidia_depth` (default 4).
3. Append z.ai's free models (glm-4.7-flash, glm-4.5-flash, glm-4.6v-flash).
4. On every `chat()` the catalog is refreshed when stale (TTL), so a newly
   launched NVIDIA model that outranks the current primary is promoted
   automatically on the next call.

Rate limits:
- NVIDIA NIM: one shared 40 requests/minute bucket (the free tier is
  account-wide, not per model).
- z.ai: one shared 40 requests/minute bucket across its free models.
- Combined, that is ~80 requests/minute — roughly one request every 0.75s;
  staying conservative (one per ~1.5s per provider) is the default pacing.

Retries: HTTP 429/5xx/timeouts are retried per model with exponential backoff
(honoring Retry-After), then the chain falls through to the next model.
Hard errors (401/400) skip retries for that model and fall through directly.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .discovery import CatalogEntry, ModelCatalog, NVIDIA_BASE_URL, ZAI_BASE_URL, ZAI_FREE_MODELS
from .ranker import rank_models
from .rate_limiter import TokenBucket

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
NVIDIA_RPM = 40        # NVIDIA NIM free tier: 40 requests per minute (shared)
ZAI_RPM = 40           # z.ai free models: paced likewise


class RouterError(RuntimeError):
    """All providers in the chain failed."""


@dataclass
class ModelSpec:
    provider: str
    model: str
    api_base: str
    api_key_env: str
    role: str = ""

    @property
    def api_key(self) -> Optional[str]:
        import os

        return os.environ.get(self.api_key_env) or None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class Attempt:
    spec: "ModelSpec"
    ok: bool
    status: Optional[int] = None
    error: Optional[str] = None
    retries: int = 0
    elapsed: float = 0.0


@dataclass
class ModelRouter:
    nvidia_depth: int = 4                     # how many top NVIDIA models to keep
    max_retries: int = 2                      # retries per model before fallback
    backoff_base: float = 1.5
    max_backoff: float = 30.0
    sleep: Callable[[float], None] = time.sleep
    client_factory: Optional[Callable[[ModelSpec], Any]] = None
    catalog: ModelCatalog = field(default_factory=ModelCatalog)
    attempt_log: List[Attempt] = field(default_factory=list)

    # ------------------------------------------------------------------ chain
    def build_chain(self, force_refresh: bool = False) -> List[ModelSpec]:
        """Live-discovered, power-ranked fallback chain (NVIDIA first)."""
        entries = self.catalog.discover(force=force_refresh)
        if not entries:
            return []
        ranked = rank_models(entries)
        nvidia = [e for e in ranked if e.provider == "nvidia"][: self.nvidia_depth]
        zai = [e for e in ranked if e.provider == "zai"]
        chain = nvidia + zai
        return [
            ModelSpec(provider=e.provider, model=e.model, api_base=e.api_base,
                      api_key_env=e.api_key_env)
            for e in chain
        ]

    def available_chain(self) -> List[ModelSpec]:
        chain = self.build_chain()
        return [s for s in chain if s.api_key]

    # -------------------------------------------------------------- internals
    def _client(self, spec: ModelSpec) -> Any:
        if self.client_factory is not None:
            return self.client_factory(spec)
        from openai import OpenAI

        key = spec.api_key
        if not key:
            raise RouterError(f"Missing API key env var {spec.api_key_env}")
        return OpenAI(base_url=spec.api_base, api_key=key)

    def _bucket_for(self, provider: str, buckets: Dict[str, TokenBucket]) -> TokenBucket:
        if provider not in buckets:
            rpm = NVIDIA_RPM if provider == "nvidia" else ZAI_RPM
            buckets[provider] = TokenBucket(capacity=rpm, window_seconds=60.0)
        return buckets[provider]

    def _backoff_seconds(self, retry_no: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.max_backoff)
        return min(self.backoff_base * (2 ** max(retry_no - 1, 0)), self.max_backoff) * (
            0.75 + random.random() / 2
        )

    @staticmethod
    def _parse_retry_after(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ----------------------------------------------------------------- public
    def chat(self, messages: List[Dict[str, str]], stream: bool = False,
             **kwargs: Any) -> Dict[str, Any]:
        candidates = self.available_chain()
        if not candidates:
            raise RouterError("No models available (check API keys / discovery)")

        buckets: Dict[str, TokenBucket] = {}
        last_error: Optional[str] = None

        for spec in candidates:
            bucket = self._bucket_for(spec.provider, buckets)
            try:
                client = self._client(spec)
            except Exception as exc:
                self.attempt_log.append(Attempt(spec, False, error=str(exc)))
                last_error = f"{spec.label}: {exc}"
                continue

            retry_no = 0
            while True:
                wait = bucket.time_until_token()
                if wait > 0:
                    self.sleep(min(wait, self.max_backoff))
                bucket.try_acquire()

                started = time.monotonic()
                try:
                    resp = client.chat.completions.create(
                        model=spec.model, messages=messages, stream=stream, **kwargs
                    )
                    self.attempt_log.append(
                        Attempt(spec, True, elapsed=time.monotonic() - started, retries=retry_no)
                    )
                    return resp.model_dump() if hasattr(resp, "model_dump") else resp
                except Exception as exc:  # noqa: BLE001
                    status = getattr(exc, "status_code", None)
                    retry_after = None
                    if hasattr(exc, "headers"):
                        retry_after = self._parse_retry_after(
                            exc.headers.get("retry-after") if callable(
                                getattr(exc.headers, "get", None)
                            ) else None
                        )
                    msg = f"{type(exc).__name__}: {exc}"

                    if status is not None and status not in RETRYABLE_STATUS:
                        self.attempt_log.append(
                            Attempt(spec, False, status=status, error=msg, retries=retry_no,
                                    elapsed=time.monotonic() - started)
                        )
                        last_error = f"{spec.label} [{status}]: {msg}"
                        break

                    retry_no += 1
                    if retry_no > self.max_retries:
                        self.attempt_log.append(
                            Attempt(spec, False, status=status, error=msg, retries=retry_no - 1,
                                    elapsed=time.monotonic() - started)
                        )
                        last_error = f"{spec.label} [{status}] after {retry_no - 1} retries: {msg}"
                        break

                    self.sleep(self._backoff_seconds(retry_no, retry_after))

        raise RouterError(f"All models failed. Last error: {last_error}")

    def ask(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.chat(messages, **kwargs)
        return resp["choices"][0]["message"]["content"]

    @property
    def current_chain(self) -> List[str]:
        return [s.label for s in self.build_chain()]
