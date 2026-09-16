"""DeerFlow evolved 2.5 — auto-discovering multi-provider LLM brain.

Built on top of DeerFlow (https://github.com/bytedance/deer-flow), MIT licensed,
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates,
Copyright (c) 2025-2026 DeerFlow Authors.

What this adds to DeerFlow:
- LIVE model discovery from NVIDIA NIM (every model in your account's catalog,
  including brand-new ones the moment they appear) plus z.ai's free models.
- A power ranker that always puts the most powerful model first, and
  automatically promotes a newly launched model when it outranks the current
  primary — no code changes needed.
- A rate-limit aware router: NVIDIA NIM paced at 40 requests/minute,
  z.ai paced likewise, so combined you can safely send roughly one request
  every ~1.5 seconds or faster, with retry + fallback on 429/5xx.
- `python -m providers.sync` regenerates the DeerFlow `config.yaml` models
  section from the live catalogs, so new models appear inside DeerFlow's UI too.
"""

from .discovery import ModelCatalog, discover_models  # noqa: F401
from .model_router import ModelRouter, ModelSpec, RouterError  # noqa: F401
from .ranker import rank_models  # noqa: F401

__version__ = "2.5.0"
