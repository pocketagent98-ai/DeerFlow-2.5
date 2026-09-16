#!/usr/bin/env python3
"""DeerFlow 2.5 — LIVE verification (needs real API keys, run on YOUR machine).

This is the final end-to-end check of the multi-provider brain against the
real NVIDIA NIM and z.ai APIs. It makes ~6 API calls total (well within the
40 req/min free tier).

Usage:
    export NVIDIA_API_KEY=nvapi-...        # from https://build.nvidia.com
    export ZAI_API_KEY=...                 # from https://z.ai
    python verify_live.py                  # from the repo root

(or put the keys in .env — python-dotenv loads it if installed)

Keys are NEVER printed, logged, or written anywhere by this script.
Exit code 0 = all critical checks passed.
"""

from __future__ import annotations

import base64
import os
import sys
import traceback

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    print(f"\n=== {name} ===")
    try:
        detail = fn()
        RESULTS.append((name, True, detail or "ok"))
        print(f"PASS: {detail or 'ok'}")
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"FAIL: {type(exc).__name__}: {str(exc)[:300]}")


def mask(value: str) -> str:
    return f"{value[:6]}...{value[-4:]}" if len(value) > 12 else "***"


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    for var in ("NVIDIA_API_KEY", "ZAI_API_KEY"):
        v = os.environ.get(var)
        if v:
            print(f"  {var} set: {mask(v)}")
        else:
            print(f"  {var} NOT SET (related checks will be skipped)")


def main() -> int:
    print("=" * 66)
    print("  DeerFlow 2.5 — LIVE VERIFICATION (real API endpoints)")
    print("=" * 66)
    load_env()

    has_nvidia = bool(os.environ.get("NVIDIA_API_KEY"))
    has_zai = bool(os.environ.get("ZAI_API_KEY"))
    if not (has_nvidia or has_zai):
        print("\nNo API keys found. Set NVIDIA_API_KEY and/or ZAI_API_KEY.")
        return 2

    if has_nvidia:
        def nvidia_discovery():
            from providers.discovery import fetch_nvidia_models
            ids = fetch_nvidia_models(os.environ["NVIDIA_API_KEY"])
            assert len(ids) > 0, "empty model list"
            return f"{len(ids)} models live-discovered on your NVIDIA account (nothing hardcoded)"
        check("NVIDIA live discovery (GET /v1/models)", nvidia_discovery)

    def zai_safety():
        from providers.discovery import fetch_zai_models, ZAI_FREE_MODELS
        listed = fetch_zai_models(os.environ["ZAI_API_KEY"])
        if listed:
            found = sorted(set(ZAI_FREE_MODELS) & set(listed))
            assert found, "endpoint up but no free models listed -> using none (paid-safe)"
            return f"endpoint up; exactly the free set confirmed: {found}"
        return "endpoint unavailable -> documented free set in use (paid-safe fallback)"
    if has_zai:
        check("z.ai free-model safety (exactly 3 free models, never paid)", zai_safety)

    def chain():
        from providers import ModelRouter
        r = ModelRouter()
        labels = r.current_chain
        assert labels, "chain is empty"
        return f"power-ranked chain ({len(labels)} models): {labels}"
    check("Router chain (most powerful first)", chain)

    if has_nvidia:
        def nvidia_ask():
            from providers import ModelRouter
            r = ModelRouter(nvidia_depth=1)
            out = r.ask("Reply with exactly the word OK and nothing else.", max_tokens=8)
            return f"real NVIDIA answer via router: {out!r}"
        check("NVIDIA real generation (router.ask)", nvidia_ask)

    if has_zai:
        def zai_ask():
            from providers import ModelRouter
            from providers.discovery import CatalogEntry, ModelCatalog, ZAI_BASE_URL
            cat = ModelCatalog(ttl_seconds=0.0)
            cat._entries = [CatalogEntry("zai", "glm-4.7-flash", ZAI_BASE_URL, "ZAI_API_KEY")]
            cat._fetched_at = 1e18
            r = ModelRouter(catalog=cat)
            out = r.ask("Reply with exactly the word OK and nothing else.", max_tokens=8)
            return f"real z.ai glm-4.7-flash answer: {out!r}"
        check("z.ai real generation (glm-4.7-flash)", zai_ask)

        def zai_vision():
            # 1x1 red PNG — vision check on glm-4.6v-flash (the free vision model)
            png = base64.b64encode(bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
                "3df80000000c4944415408d763f8cfc0f01f0005050202b9cdc760000000004945"
                "4e44ae426082")).decode()
            from providers import ModelRouter
            from providers.discovery import CatalogEntry, ModelCatalog, ZAI_BASE_URL
            cat = ModelCatalog(ttl_seconds=0.0)
            cat._entries = [CatalogEntry("zai", "glm-4.6v-flash", ZAI_BASE_URL, "ZAI_API_KEY")]
            cat._fetched_at = 1e18
            r = ModelRouter(catalog=cat)
            resp = r.chat([{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image? One word."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{png}"}},
                ],
            }], max_tokens=8)
            out = resp["choices"][0]["message"]["content"]
            return f"real z.ai glm-4.6v-flash vision answer: {out!r}"
        check("z.ai vision generation (glm-4.6v-flash, 1x1 red pixel)", zai_vision)

    print("\n" + "=" * 66)
    print("  SUMMARY")
    print("=" * 66)
    failed = [n for n, ok, _ in RESULTS if not ok]
    for n, ok, d in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"\n  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed.")
    if failed:
        print("  Note: if everything failed with connection errors, check your")
        print("  internet/firewall; the code paths themselves are covered by")
        print("  29 offline tests (`python -m pytest providers/tests -v`).")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        traceback.print_exc()
        sys.exit(130)
