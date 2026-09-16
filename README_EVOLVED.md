# DeerFlow Evolved 2.5 — Auto-Discovering NVIDIA NIM + z.ai Brain

> A fork of [DeerFlow](https://github.com/bytedance/deer-flow) (MIT licensed,
> Copyright (c) 2025 Bytedance Ltd. and/or its affiliates,
> Copyright (c) 2025-2026 DeerFlow Authors — original LICENSE preserved).
> This fork keeps DeerFlow exactly as it is, and adds a live, self-updating
> multi-provider LLM brain on top of it — wired INTO DeerFlow itself.

## What this adds

| Addition | File | What it does |
|---|---|---|
| **Live model discovery** | `providers/discovery.py` | Fetches the FULL, live model list from NVIDIA NIM (`/v1/models`) with your key — nothing hardcoded. Any model NVIDIA launches shows up automatically on the next refresh (TTL 10 min, or forced). z.ai side uses its three fully-free API models: `glm-4.7-flash`, `glm-4.5-flash`, `glm-4.6v-flash`. |
| **Tier-first power ranking with auto-upgrade** | `providers/ranker.py` | Ranking is tier-first (lightning/ultra/frontier = top tier compared absolutely, then generation, then parameters). Today `nvidia/nemotron-3.5-lightning-30b-a3b` ranks #1. The day NVIDIA launches a bigger/newer top-tier model (e.g. a `nemotron-4-*`), it scores higher and is promoted automatically — no code change, no config change. |
| **Rate-limit aware router** | `providers/model_router.py` | Chain = top NVIDIA models first, then z.ai free models. NVIDIA NIM paced at **40 requests/minute** (account-wide shared bucket); z.ai likewise — combined roughly one request per ~0.75–1.5 s. On 429/5xx: exponential-backoff retries honoring `Retry-After`, then automatic fallback to the next model. Hard errors (401/400) fall through immediately. |
| **Deep DeerFlow integration** | `backend/.../config/auto_providers.py` + `config/__init__.py` | `deerflow.config.get_app_config()` is wrapped: with the env keys set, discovered models are appended to DeerFlow's model list on every config load — they simply appear in the model picker. User-configured models keep priority; `DEERFLOW_AUTO_PROVIDERS=0` disables the bridge. |
| **DeerFlow config sync** | `providers/sync.py` | `python -m providers.sync --write config.yaml` regenerates DeerFlow's `models:` section from the LIVE catalogs, for those who prefer explicit config. |
| **Circuit breaker + self-healing** | `providers/model_router.py` | 3 consecutive failures open the breaker for 60 s (configurable), after which the model is probed again; any success resets it. Per-model `stats` counters for observability. |
| **z.ai paid-model safety** | `providers/discovery.py` | Only the three documented free models are ever used. Endpoint up + free models missing → no z.ai models at all (never arbitrary/paid ones). |
| **Free stack guide** | `docs/FREE_STACK.md` | Zero-cost replacements for every paid tool: SearXNG/DuckDuckGo (Tavily/Exa/Parallel Search), Crawl4AI (Firecrawl), Semantic Scholar/OpenAlex/arXiv (Consensus), Qdrant for long-term memory. |
| **Tests (no keys needed)** | `providers/tests/` | 29 tests: discovery, z.ai paid-model safety, auto-promotion of newly launched models, tier-first ranking safety, retry, fallback, Retry-After, provider rate buckets, circuit breaker, and the real `deerflow.config.get_app_config()` pipeline. |

## Setup

### 1. Keys

```bash
cp .env.example .env
# NVIDIA NIM: sign up at https://build.nvidia.com -> key starts with nvapi-
#   Every model in the catalog can be used with the free serverless credits
#   (40 requests/minute account-wide).
# z.ai: key from https://z.ai -> the three free models are used automatically.
NVIDIA_API_KEY=nvapi-...
ZAI_API_KEY=...
```

### 2. Standalone use (the brain, without the full DeerFlow server)

```bash
pip install -r providers/requirements.txt
export NVIDIA_API_KEY=nvapi-... ZAI_API_KEY=...

python - << 'EOF'
from providers import ModelRouter

router = ModelRouter()
print("Live chain (most powerful first):", router.current_chain)
print(router.ask("Explain the difference between retry and fallback in 3 bullets."))
EOF
```

The chain is rebuilt from the live catalog on every call (TTL-cached), so a
model NVIDIA launched an hour ago is already in it.

### 3. Inside DeerFlow (automatic)

The bridge is wired into `deerflow.config.get_app_config` (see
`backend/packages/harness/deerflow/config/auto_providers.py`). With the env
keys set, discovered models are appended to DeerFlow's model list on every
config load — they simply appear in the model picker. User-configured models
always keep priority; `DEERFLOW_AUTO_PROVIDERS=0` disables the bridge.
`python -m providers.sync --write config.yaml` remains available if you
prefer the models written into the config file explicitly.

### 4. Tests

```bash
python -m pytest providers/tests -v
```

## Rate-limit design (why 40 RPM everywhere)

- NVIDIA NIM's free serverless tier is limited per account (not per model),
  so all NIM models share ONE 40 req/min token bucket.
- z.ai's free models share one 40 req/min bucket.
- Combined throughput: ~80 req/min (about one request every 0.75 s). The
  token bucket smooths bursts; the router additionally backs off on any 429
  using the provider's own `Retry-After` hint, then retries, then falls back.

## How "automatic new model" works, exactly

1. `ModelCatalog.discover()` calls `GET {NVIDIA_BASE_URL}/models` with your key.
2. The response contains every model your account can use — today and in the
   future. There is no hardcoded NVIDIA model list in this codebase.
3. `rank_models()` scores each model (tier keyword + params + generation).
   A `nemotron-4-lightning` or `nemotron-4-ultra` launched tomorrow scores
   above today's `nemotron-3.5-lightning` and takes slot #1 on its own.
4. An unrecognized name (e.g. `acme/mystery-model`) never outranks a known
   top-tier model; it appears in discovery output so you can pin it manually
   with `POWER_OVERRIDES="acme/mystery-model"` if you want it first.

## License & credits

MIT. Original DeerFlow code: Copyright (c) 2025 Bytedance Ltd. and/or its
affiliates, Copyright (c) 2025-2026 DeerFlow Authors (see `LICENSE`).
This fork's additions (`providers/`, the auto-providers bridge, the docs) are MIT as well. The license file and all upstream
credits are preserved exactly as required — that is what makes this fork safe
to use, share and build on.
