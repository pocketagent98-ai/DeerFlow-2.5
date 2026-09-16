# The Free Stack — running DeerFlow Evolved at zero tool cost

This guide wires DeerFlow to a fully free research/planning stack. Every paid
tool has a self-hosted or genuinely-free replacement.

## 1. LLM brain (already built in)

The `providers/` package handles this — no paid API required:

| Concern | Solution |
|---|---|
| Primary reasoning | **NVIDIA NIM** — every model in the catalog, free serverless credits, 40 req/min (account-wide). Auto-discovered live; new models appear automatically. |
| Fallback brain | **z.ai** — `glm-4.7-flash`, `glm-4.5-flash`, `glm-4.6v-flash` are fully free models, used automatically when NVIDIA rate-limits or fails. |
| Rate limits | Built-in token buckets (40 RPM per provider) + 429 retry with `Retry-After` + automatic model fallback. |

Keys: `NVIDIA_API_KEY` (https://build.nvidia.com), `ZAI_API_KEY` (https://z.ai).

## 2. Web search (replaces Tavily / Exa / Parallel Search)

- **DuckDuckGo** — no API key at all; supported natively by DeerFlow 1.x
  (`SEARCH_API=duckduckgo`).
- **SearXNG** — self-hosted metasearch (Docker, one container): queries many
  engines in parallel per query, no keys, no per-query cost.

  ```bash
  docker run -d --name searxng -p 8888:8080 searxng/searxng
  ```

- **Semantic reranking (Exa-style "search by meaning", free)** — take broad
  results from SearXNG/DuckDuckGo, embed query + results locally with a free
  embedding model (e.g. BGE via `sentence-transformers`, runs on CPU or a T4),
  re-rank by cosine similarity.

## 3. Crawling (replaces Firecrawl)

- **Crawl4AI** — open source (Apache-2.0), self-hosted, Playwright-based:
  renders JavaScript, outputs clean LLM-ready markdown/JSON, CSS/XPath/LLM
  extraction, caching, Docker deployment. No API keys, no per-page fees.

  ```bash
  pip install crawl4ai
  crawl4ai-setup
  ```

  Or run the Docker server and point DeerFlow's crawler at it.
- **Jina Reader (free tier)** — DeerFlow's default crawler, works with no key.

## 4. Academic research (replaces Consensus)

All three are free, no credit card:

- **Semantic Scholar API** — 226M+ papers, abstracts, citation counts, AI
  TLDR summaries; most endpoints work without any key (rate-limited), a free
  key raises limits. https://api.semanticscholar.org
- **OpenAlex** — free scholarly catalog API with free keys. https://openalex.org
- **arXiv** — native DeerFlow search option (`SEARCH_API=arxiv`), no cost.

## 5. Memory / long-term project memory storage

- **Qdrant** — free, self-hostable vector database for the long-term project memory and
  research memory:

  ```bash
  docker run -d -p 6333:6333 qdrant/qdrant
  ```

## 6. Recommended .env summary

```bash
NVIDIA_API_KEY=nvapi-...     # primary brain (free credits)
ZAI_API_KEY=...              # fallback brain (3 free models)
# optional self-hosted extras:
SEARXNG_URL=http://localhost:8888
QDRANT_URL=http://localhost:6333
```

With this stack the entire research-and-planning brain of the pipeline runs
at zero tool cost; money is spent only on LLM tokens you choose to buy.
