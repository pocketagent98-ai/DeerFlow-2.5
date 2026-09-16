"""Auto-discovered model providers (DeerFlow 2.5 evolution).

Wired in via ``deerflow.config.get_app_config``: every time DeerFlow asks for
its configuration, the NVIDIA NIM catalog and the z.ai free models are
discovered LIVE (through the repo-root ``providers`` package) and appended to
the model list before the config is handed out.

What this means in practice:

- Set ``NVIDIA_API_KEY`` and/or ``ZAI_API_KEY`` in the environment and every
  model available to those accounts appears in DeerFlow's model picker —
  no config.yaml editing, no ``providers.sync`` run needed.
- When NVIDIA launches a new model, it shows up on the next config load
  (config hot-reloads included), ranked by power (most powerful first).
- Models the user explicitly configured in config.yaml always win: an
  auto-discovered model that collides with an existing ``model`` id or name
  is skipped, and user models keep their position (the first entry remains
  the default model).
- Set ``DEERFLOW_AUTO_PROVIDERS=0`` (or "false"/"no") to disable the bridge
  entirely. Discovery failures are logged and silently ignored, so DeerFlow
  never fails to boot because of it.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)

AUTO_PROVIDERS_ENV = "DEERFLOW_AUTO_PROVIDERS"
NVIDIA_DEPTH = 4           # how many top-ranked NVIDIA models to inject
DEFAULT_RPM = 40           # provider rate-limit pacing for injected models

_slug_re = re.compile(r"[^a-z0-9-]+")


def _import_providers():
    """Import the repo-root ``providers`` package (light deps: openai only)."""
    try:
        import providers  # already importable (repo root on sys.path / cwd)

        return providers
    except ImportError:
        # backend/packages/harness/deerflow/config/auto_providers.py
        # -> repo root is five parents up.
        repo_root = Path(__file__).resolve().parents[5]
        sys.path.insert(0, str(repo_root))
        import providers

        return providers


def _slug(model_id: str) -> str:
    tail = model_id.split("/")[-1].lower()
    slug = _slug_re.sub("-", tail).strip("-")
    return f"auto-{slug or 'model'}"


def _select_models(entries: List[Any]) -> List[Any]:
    """Power-ranked selection: top NVIDIA models first, then z.ai free set."""
    providers = _import_providers()
    ranked = providers.rank_models(entries)
    nvidia = [e for e in ranked if e.provider == "nvidia"][:NVIDIA_DEPTH]
    zai = [e for e in ranked if e.provider == "zai"]
    return nvidia + zai


def inject_into_app_config(config: Any) -> List[str]:
    """Append live-discovered provider models to an AppConfig instance.

    Returns the names of the injected models (empty when disabled/failed).
    Never raises: any problem is logged and ignored, and the config object
    is always returned to the caller in a usable state.
    """
    flag = os.environ.get(AUTO_PROVIDERS_ENV, "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return []

    try:
        providers = _import_providers()
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.info("Auto-providers disabled (providers package unavailable: %s)", exc)
        return []

    try:
        catalog = providers.ModelCatalog()
        entries = catalog.discover()
    except Exception as exc:
        logger.warning("Auto-provider discovery failed (ignored): %s", exc)
        return []

    if not entries:
        # Expected when no NVIDIA_API_KEY / ZAI_API_KEY are configured.
        logger.debug("Auto-provider discovery returned no models.")
        return []

    try:
        selected = _select_models(entries)
    except Exception as exc:
        logger.warning("Auto-provider ranking failed (ignored): %s", exc)
        return []

    from deerflow.config.model_config import ModelConfig

    existing_names = {m.name for m in config.models}
    existing_model_ids = {getattr(m, "model", None) for m in config.models}

    injected: List[str] = []
    for entry in selected:
        if entry.model in existing_model_ids:
            continue
        name = _slug(entry.model)
        if name in existing_names:
            name = f"{name}-{entry.provider}"
        api_key = os.environ.get(entry.api_key_env) or ""
        model_config = ModelConfig(
            name=name,
            display_name=f"{entry.model} (auto, {entry.provider})",
            description=(
                f"Auto-discovered from {entry.provider}. Ranked by the DeerFlow "
                "2.5 power ranker; newly launched stronger models are promoted "
                "automatically."
            ),
            use="langchain_openai:ChatOpenAI",
            model=entry.model,
            api_base=entry.api_base,
            api_key=api_key,
            timeout=600.0,
            max_retries=2,
            request_admission={
                "requests_per_minute": DEFAULT_RPM,
                "max_wait_seconds": 300.0,
                "max_queue_size": 256,
            },
        )
        existing_names.add(name)
        existing_model_ids.add(entry.model)
        config.models.append(model_config)
        if getattr(config, "_models_by_name", None) is not None:
            config._models_by_name.setdefault(name, model_config)
        injected.append(name)

    if injected:
        logger.info(
            "Auto-providers injected %d live models: %s",
            len(injected),
            ", ".join(injected),
        )
    return injected
