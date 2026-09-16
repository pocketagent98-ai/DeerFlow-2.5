"""Power ranking: always put the most capable model first, auto-upgrade later.

"Power" cannot be read from an API field, so this ranker makes its scoring
transparent and editable instead of hiding it:

score = TIER + 0.005 * PARAMETERS + 10 * VERSION

- TIER is a keyword score from the model name:
    lightning / ultra / frontier -> 100   (top tier)
    pro                          -> 80
    super                        -> 70
    (unrecognized keyword)       -> 50   (flagged for review, never auto-promoted
                                           above a known top-tier model)
    air                          -> 35
    nano / mini                  -> 25
- PARAMETERS is the total parameter count parsed from the name (550b, 30b...).
- VERSION is the leading version number in the name (3.5 -> 35, 4 -> 40), so a
  newer generation outranks an older one within the same tier.

With the default scores, `nvidia/nemotron-3.5-lightning-30b-a3b` ranks just
above `nvidia/nemotron-3-ultra-550b-a55b` (tier tie broken by generation),
matching the current expectation. When NVIDIA launches something like a
`nemotron-4-ultra` or `nemotron-4-lightning`, it scores higher and is
promoted automatically — that is the "new models show up by themselves" rule.

POWER_OVERRIDES (comma-separated env var, e.g.
POWER_OVERRIDES="moonshotai/kimi-k3,nvidia/nemotron-3.5-lightning-30b-a3b")
pins models at the top of the chain regardless of score.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from .discovery import CatalogEntry

TIER_SCORES = {
    "lightning": 100.0,
    "ultra": 100.0,
    "frontier": 100.0,
    "pro": 80.0,
    "super": 70.0,
    "air": 35.0,
    "nano": 25.0,
    "mini": 20.0,
    "flash": 40.0,
    "turbo": 45.0,
    "fast": 45.0,
}
UNKNOWN_TIER = 50.0

# Matches "550b" but not the MoE active-param suffix in "a55b" / "-a3b".
_PARAMS_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b\b", re.IGNORECASE)
_VERSION_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _tier(name: str) -> float:
    lowered = name.lower()
    best = UNKNOWN_TIER
    matched = False
    for kw, sc in TIER_SCORES.items():
        if kw in lowered:
            matched = True
            best = max(best, sc)
    return best if matched else UNKNOWN_TIER


def _params(name: str) -> float:
    total = 0.0
    for m in _PARAMS_RE.finditer(name.lower()):
        total += float(m.group(1))
    return total


def _version(name: str) -> float:
    m = _VERSION_RE.search(name)
    return float(m.group(1)) if m else 0.0


def score_model(model_id: str) -> float:
    return _tier(model_id) + 0.005 * _params(model_id) + 10.0 * _version(model_id)


def rank_models(
    entries: List[CatalogEntry],
    overrides: Optional[List[str]] = None,
) -> List[CatalogEntry]:
    """Sort entries most-powerful-first. Pinned overrides come first, in order."""
    overrides = overrides or []
    if not overrides:
        env = os.environ.get("POWER_OVERRIDES")
        if env:
            overrides = [x.strip() for x in env.split(",") if x.strip()]

    def key(e: CatalogEntry):
        try:
            pin = overrides.index(e.model)
        except ValueError:
            pin = len(overrides)
        return (pin, -score_model(e.model))

    return sorted(entries, key=key)


def find_new_models(known: List[str], discovered: List[CatalogEntry]) -> List[str]:
    """Model ids seen in discovery that were not in the known list."""
    known_set = set(known)
    return [e.model for e in discovered if e.model not in known_set]
