"""Live model discovery for NVIDIA NIM and z.ai.

NVIDIA NIM exposes an OpenAI-compatible `GET /v1/models` endpoint that lists
EVERY model available to your account — including models released after this
code was written. That is the auto-discovery mechanism: nothing is hardcoded
on the NVIDIA side; the catalog is fetched live.

z.ai does not reliably expose a public models-list endpoint, and only three
of its models are fully free on the API (input, output and cached input all
$0): glm-4.7-flash, glm-4.5-flash (text) and glm-4.6v-flash (vision-capable).
Those three are ALWAYS the z.ai side of the catalog. The models listing
(when reachable) is informational only: it never adds, and never removes,
models — live evidence (2026-09-16): a valid key can generate with
glm-4.7-flash while /v1/models does not list it at all. If a free model ever
stops working, the router's hard-error path skips it automatically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/"

# The three fully-free z.ai API models (verify current status at docs.z.ai).
ZAI_FREE_MODELS: List[str] = ["glm-4.7-flash", "glm-4.5-flash", "glm-4.6v-flash"]


@dataclass
class CatalogEntry:
    provider: str          # "nvidia" | "zai"
    model: str             # model id as passed to the API
    api_base: str
    api_key_env: str
    discovered_at: float = field(default_factory=time.time)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def fetch_nvidia_models(api_key: str, client_factory: Optional[Callable] = None) -> List[str]:
    """Fetch the LIVE list of model ids from NVIDIA NIM. Nothing hardcoded."""
    if client_factory is not None:
        client = client_factory(NVIDIA_BASE_URL, api_key)
    else:
        from openai import OpenAI

        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    models = client.models.list()
    return sorted(m.id for m in models)


def fetch_zai_models(api_key: str, client_factory: Optional[Callable] = None) -> List[str]:
    """Try z.ai's models endpoint; return whatever it offers (may be empty)."""
    try:
        if client_factory is not None:
            client = client_factory(ZAI_BASE_URL, api_key)
        else:
            from openai import OpenAI

            client = OpenAI(base_url=ZAI_BASE_URL, api_key=api_key)
        models = client.models.list()
        return sorted(m.id for m in models)
    except Exception:
        return []


class ModelCatalog:
    """Discovers and caches provider models, refreshing on a TTL.

    A new model launched on NVIDIA NIM shows up automatically on the next
    refresh (default TTL 10 minutes, or force=True).
    """

    def __init__(self, ttl_seconds: float = 600.0, client_factory: Optional[Callable] = None):
        self.ttl_seconds = ttl_seconds
        self.client_factory = client_factory
        self._entries: List[CatalogEntry] = []
        self._fetched_at: float = 0.0
        self.nvidia_error: Optional[str] = None
        self.zai_error: Optional[str] = None
        self.zai_confirmed: List[str] = []  # free models seen in the live listing (informational)

    @property
    def entries(self) -> List[CatalogEntry]:
        return list(self._entries)

    def _nvidia_entries(self, api_key: str) -> List[CatalogEntry]:
        try:
            ids = fetch_nvidia_models(api_key, self.client_factory)
        except Exception as exc:
            self.nvidia_error = str(exc)
            return []
        return [
            CatalogEntry(provider="nvidia", model=mid, api_base=NVIDIA_BASE_URL,
                          api_key_env="NVIDIA_API_KEY")
            for mid in ids
        ]

    def _zai_entries(self, api_key: str) -> List[CatalogEntry]:
        listed = fetch_zai_models(api_key, self.client_factory)
        # SAFETY RULE: only the three documented free models are ever used.
        # Arbitrary listed models (which may be PAID) are never picked.
        #
        # The listing is INFORMATIONAL ONLY — it does not add or remove
        # models. Live evidence (2026-09-16): a valid key can generate with
        # glm-4.7-flash while /v1/models does not list it, so trusting the
        # listing would silently delete the whole free fallback layer. If a
        # documented free model ever stops working, the router skips it via
        # its hard-error path.
        if listed:
            confirmed = sorted(set(ZAI_FREE_MODELS) & set(listed))
            self.zai_error = None  # listing reachable; informational only
            if confirmed:
                self.zai_confirmed = confirmed
        keep = list(ZAI_FREE_MODELS)
        return [
            CatalogEntry(provider="zai", model=mid, api_base=ZAI_BASE_URL,
                          api_key_env="ZAI_API_KEY")
            for mid in keep
        ]

    def discover(self, nvidia_key: Optional[str] = None, zai_key: Optional[str] = None,
                 force: bool = False) -> List[CatalogEntry]:
        import os

        nvidia_key = nvidia_key or os.environ.get("NVIDIA_API_KEY")
        zai_key = zai_key or os.environ.get("ZAI_API_KEY")

        fresh = (time.monotonic() - self._fetched_at) < self.ttl_seconds
        if self._entries and not force and fresh:
            return self.entries

        entries: List[CatalogEntry] = []
        if nvidia_key:
            entries += self._nvidia_entries(nvidia_key)
        if zai_key:
            entries += self._zai_entries(zai_key)
        if entries:
            self._entries = entries
            self._fetched_at = time.monotonic()
        return self.entries


def discover_models(nvidia_key: Optional[str] = None, zai_key: Optional[str] = None,
                    client_factory: Optional[Callable] = None) -> List[CatalogEntry]:
    """One-shot discovery helper."""
    return ModelCatalog(client_factory=client_factory).discover(nvidia_key, zai_key)
