#!/usr/bin/env python3
"""Deep key diagnosis for DeerFlow 2.5 — why is a key rejected?

Checks, with REAL API calls (run on a network-unrestricted machine or the
GitHub Actions runner via .github/workflows/deep-diagnosis.yml):

  NVIDIA (integrate.api.nvidia.com):
    - live model catalog + presence of the top-ranked models
    - three REAL generations on the top chain models

  z.ai key — finds out WHY it is rejected:
    - GET  https://api.z.ai/api/paas/v4/models
    - POST https://api.z.ai/api/paas/v4/chat/completions  (glm-4.7-flash)
    - POST https://open.bigmodel.cn/api/paas/v4/chat/completions (glm-4-flash)
      ^ if the key works HERE but not on api.z.ai, the key was created on
        Zhipu BigModel (the China platform) instead of z.ai (international)

Keys are read from NVIDIA_API_KEY / ZAI_API_KEY and NEVER printed.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

ZAI = "https://api.z.ai/api/paas/v4"
BIGMODEL = "https://open.bigmodel.cn/api/paas/v4"
NVIDIA = "https://integrate.api.nvidia.com/v1"


def http(method: str, url: str, headers: dict, payload: dict | None = None,
         timeout: int = 90):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode(errors="replace")
            return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def short(text: str, n: int = 400) -> str:
    return text[:n].replace("\n", " ") + ("..." if len(text) > n else "")


def nvidia_section() -> None:
    key = os.environ.get("NVIDIA_API_KEY")
    print("\n" + "=" * 66)
    print("  NVIDIA NIM — deep check")
    print("=" * 66)
    if not key:
        print("  NVIDIA_API_KEY not set; skipping.")
        return
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    st, body = http("GET", f"{NVIDIA}/models", hdr)
    if st != 200:
        print(f"  models list: HTTP {st} — {short(body)}")
        return
    ids = sorted(m["id"] for m in json.loads(body)["data"])
    print(f"  models list: HTTP 200 — {len(ids)} models on your account")
    tops = ["nvidia/nemotron-3.5-lightning-30b-a3b",
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "nvidia/nemotron-3-ultra-550b-a55b",
            "nvidia/nemotron-3-super-120b-a12b"]
    for t in tops:
        print(f"    {'PRESENT ' if t in ids else 'absent  '} {t}")

    print("\n  three REAL generations (top of the power-ranked chain):")
    for model in tops[:3]:
        payload = {"model": model,
                   "messages": [{"role": "user", "content": "Say OK"}],
                   "max_tokens": 8, "temperature": 0}
        st, body = http("POST", f"{NVIDIA}/chat/completions", hdr, payload)
        if st == 200:
            content = json.loads(body)["choices"][0]["message"].get("content", "")
            print(f"    HTTP 200  {model}\n              -> real answer: {content!r}")
        else:
            print(f"    HTTP {st}  {model} — {short(body, 200)}")
        time.sleep(1.6)  # stay well under the 40 req/min account limit


def zai_section() -> None:
    key = os.environ.get("ZAI_API_KEY")
    print("\n" + "=" * 66)
    print("  z.ai key — WHY is it rejected? (deep diagnosis)")
    print("=" * 66)
    if not key:
        print("  ZAI_API_KEY not set; skipping.")
        return
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    print("\n  [A] GET api.z.ai/api/paas/v4/models (the endpoint our code uses):")
    st, body = http("GET", f"{ZAI}/models", hdr, timeout=60)
    print(f"      HTTP {st} — {short(body, 250)}")

    print("\n  [B] POST api.z.ai chat/completions, glm-4.7-flash (free model):")
    payload = {"model": "glm-4.7-flash",
               "messages": [{"role": "user", "content": "Say OK"}],
               "max_tokens": 8}
    st, body = http("POST", f"{ZAI}/chat/completions", hdr, payload, timeout=90)
    print(f"      HTTP {st} — {short(body, 250)}")

    print("\n  [C] same key against open.bigmodel.cn (Zhipu's China platform):")
    payload = {"model": "glm-4-flash",
               "messages": [{"role": "user", "content": "Say OK"}],
               "max_tokens": 8}
    st, body = http("POST", f"{BIGMODEL}/chat/completions", hdr, payload, timeout=90)
    print(f"      HTTP {st} — {short(body, 250)}")

    print("\n  interpretation guide:")
    print("    - [A] 401 + [B] 401 + [C] 200  -> key is a BIGMODEL.CN key, not z.ai;")
    print("         create the key at z.ai (international platform) instead.")
    print("    - [A]/[B]/[C] all 401          -> key is wrong, expired or revoked;")
    print("         create a fresh one.")
    print("    - [B] 200                       -> key works; the earlier failure")
    print("         was transient (re-run verify_live.py).")


def main() -> None:
    print("DeerFlow 2.5 — DEEP KEY DIAGNOSIS")
    print(f"keys: NVIDIA={'set' if os.environ.get('NVIDIA_API_KEY') else 'NOT SET'}"
          f" zai={'set' if os.environ.get('ZAI_API_KEY') else 'NOT SET'}")
    nvidia_section()
    zai_section()
    print("\nDone.")


if __name__ == "__main__":
    main()
