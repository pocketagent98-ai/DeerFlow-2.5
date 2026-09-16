# 🦌 DeerFlow 2.5 — The Evolved DeerFlow

> **This is [DeerFlow](https://github.com/bytedance/deer-flow), evolved.**
> A fork that keeps the entire upstream harness untouched and adds a
> **live, self-updating, multi-provider AI brain** inside it: NVIDIA NIM
> first, z.ai's free models as fallback, with rate-limit retry, automatic
> model fallback, and **auto-discovery of newly launched models**.
>
> Upstream DeerFlow is MIT licensed, Copyright (c) 2025 Bytedance Ltd. and/or
> its affiliates, Copyright (c) 2025-2026 DeerFlow Authors. This fork preserves
> the upstream LICENSE and all credits — see [Credits](#credits). The full
> upstream README (setup of the base harness, all features) is available in
> the [upstream repository](https://github.com/bytedance/deer-flow/blob/main/README.md)
> and in this repo's git history.

---

## What makes this DeerFlow 2.5

| Evolution | Where | What it does |
|---|---|---|
| **🧠 Auto-discovering model brain** | `providers/` + `backend/.../config/auto_providers.py` | Set `NVIDIA_API_KEY` and/or `ZAI_API_KEY` in `.env` and **every model available on NVIDIA NIM appears in DeerFlow's model picker automatically** — no config editing, no sync runs. When NVIDIA launches a new, stronger model tomorrow, it shows up on the next config load and — if it outranks today's models — takes the top slot by itself. |
| **⚡ Rate-limit aware fallback** | `providers/model_router.py` | NVIDIA NIM paced at **40 requests/minute** (account-wide), z.ai likewise. On 429/5xx: exponential backoff honoring `Retry-After`, then automatic fallback down the chain (top NVIDIA models → z.ai free models). Hard errors (401/400) fall through immediately. |
| **🏆 Transparent power ranking** | `providers/ranker.py` | Every discovered model is scored (`TIER + 0.005×PARAMS + 10×VERSION`): today `nemotron-3.5-lightning` ranks #1; a hypothetical `nemotron-4-ultra` would outrank it automatically. Unknown names never jump the queue — pin them with `POWER_OVERRIDES` if you want them first. |
| **🎮 Game Factory skill** | `skills/public/game-factory/SKILL.md` | The research-to-plan methodology: objective-driven research missions (20/45/60-min budget with completion criteria — not just a timer), source verification, Game Bible, and an Atomic Task DAG where every task carries dependencies, allowed/do-not-touch files and acceptance criteria, ending at a mandatory user approval gate before any code is written. |
| **🆓 Free stack guide** | `docs/FREE_STACK.md` | Zero-cost replacements for every paid tool: SearXNG/DuckDuckGo (Tavily/Exa/Parallel Search), Crawl4AI (Firecrawl), Semantic Scholar/OpenAlex/arXiv (Consensus), Qdrant for long-term memory. |
| **✅ 21 tests, no keys needed** | `providers/tests/` | Unit + integration tests: discovery, auto-promotion of newly launched models, retry/fallback, Retry-After, provider rate buckets, and the real `deerflow.config.get_app_config()` pipeline (skips cleanly when backend deps are absent). |

## Quick start

```bash
git clone https://github.com/pocketagent98-ai/DeerFlow-2.5.git
cd DeerFlow-2.5

# 1. your free keys:
#    NVIDIA NIM (every model, free serverless credits, 40 req/min):
#      sign up at https://build.nvidia.com -> key starts with nvapi-
#    z.ai (3 fully-free models: glm-4.7-flash, glm-4.5-flash, glm-4.6v-flash):
#      key from https://z.ai
cp .env.example .env   # fill NVIDIA_API_KEY / ZAI_API_KEY

# 2. that's it for the brain. Start DeerFlow exactly like upstream
#    (see the upstream README / Install.md for `make setup`, `make dev`,
#    Docker options). Auto-discovered models appear in the model picker.
```

### The brain, standalone (without the full harness)

```bash
pip install -r providers/requirements.txt
export NVIDIA_API_KEY=nvapi-... ZAI_API_KEY=...

python - << 'EOF'
from providers import ModelRouter

router = ModelRouter()
print("Live chain (most powerful first):", router.current_chain)
print(router.ask("Explain the core loop of an endless runner game in 3 bullets."))
EOF
```

### Run the tests

```bash
pip install -r providers/requirements.txt pydantic pyyaml python-dotenv sqlalchemy
python -m pytest providers/tests -v   # 21 tests
```

## How the auto-discovery works, exactly

1. `providers/discovery.py` calls `GET https://integrate.api.nvidia.com/v1/models`
   with your key — nothing about NVIDIA is hardcoded; you get every model your
   account can use, today and in the future.
2. z.ai side uses its three fully-free API models
   (`glm-4.7-flash`, `glm-4.5-flash`, `glm-4.6v-flash`).
3. `providers/ranker.py` ranks everything most-powerful-first.
4. `deerflow.config.get_app_config()` (wrapped in
   `backend/packages/harness/deerflow/config/__init__.py`, logic in
   `auto_providers.py`) appends the ranked models to DeerFlow's model list on
   every config load — user-configured models always keep priority, and
   `DEERFLOW_AUTO_PROVIDERS=0` turns the whole thing off.

`python -m providers.sync --write config.yaml` remains available if you
prefer the models written into your config file explicitly.

## Credits

- **DeerFlow** — the entire base harness, its design and its community:
  https://github.com/bytedance/deer-flow — MIT, Copyright (c) 2025 Bytedance
  Ltd. and/or its affiliates, Copyright (c) 2025-2026 DeerFlow Authors.
- **This fork's additions** (`providers/`, the auto-providers bridge, the
  game-factory skill, the free-stack docs) — MIT, same license file.

## License

MIT — see [LICENSE](./LICENSE).
