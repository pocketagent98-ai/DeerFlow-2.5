from .app_config import get_app_config as _upstream_get_app_config
from .extensions_config import ExtensionsConfig, get_extensions_config
from .loop_detection_config import LoopDetectionConfig
from .memory_config import MemoryConfig, get_memory_config
from .paths import Paths, get_paths
from .skill_evolution_config import SkillEvolutionConfig
from .skills_config import SkillsConfig
from .tracing_config import (
    get_enabled_tracing_providers,
    get_explicitly_enabled_tracing_providers,
    get_tracing_config,
    is_monocle_tracing_enabled,
    is_tracing_enabled,
    validate_enabled_tracing_providers,
)
from .auto_providers import inject_into_app_config

__all__ = [
    "get_app_config",
    "SkillEvolutionConfig",
    "Paths",
    "get_paths",
    "SkillsConfig",
    "ExtensionsConfig",
    "get_extensions_config",
    "LoopDetectionConfig",
    "MemoryConfig",
    "get_memory_config",
    "get_tracing_config",
    "get_explicitly_enabled_tracing_providers",
    "get_enabled_tracing_providers",
    "is_monocle_tracing_enabled",
    "is_tracing_enabled",
    "validate_enabled_tracing_providers",
]


def get_app_config():
    """DeerFlow 2.5: upstream config plus auto-discovered provider models.

    Wraps the upstream getter so that NVIDIA NIM and z.ai models discovered
    live (see ``auto_providers.py``) are appended to the model list before the
    config is handed out. Set ``NVIDIA_API_KEY`` / ``ZAI_API_KEY`` in the
    environment to activate; set ``DEERFLOW_AUTO_PROVIDERS=0`` to disable.
    User-configured models always keep priority.
    """
    config = _upstream_get_app_config()
    if not getattr(config, "_auto_providers_applied", False):
        try:
            inject_into_app_config(config)
        except Exception:  # never break config retrieval
            pass
        finally:
            try:
                config._auto_providers_applied = True
            except Exception:
                pass
    return config
