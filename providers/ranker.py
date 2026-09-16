"""Power ranking: always put the most capable model first, auto-upgrade later.

"Power" cannot be read from an API field, so this ranker makes its scoring
transparent and editable instead of hiding it.

Ranking is TIER-FIRST (lexicographic on a tuple), which guarantees the
documented safety property:

    an unrecognized model NEVER outranks a known top-tier model, no matter
    what numbers appear in its name.

Sort key (descending on each component):

    1. TIER    — keyword score from the model name:
                 lightning / ultra / frontier -> 100   (top tier)
                 pro                          -> 80
                 super                        -> 70
                 (unrecognized keyword)       -> 50
                 air                          -> 35
                 nano / mini                  -> 25
    2. VERSION — the leading version number in the name (4 > 3.5 > 3), so a
                 newer generation outranks an older one within the same tier.
    3. PARAMS  — total parameter count parsed from the name (550b, 30b...).

Consequences (all covered by tests):

- `nvidia/nemotron-3.5-lightning-30b-a3b` ranks just above
  `nvidia/nemotron-3-ultra-550b-a55b` (tier tie broken by generation 3.5 > 3).
- A `nemotron-4-ultra` or `nemotron-4-lightning` launched tomorrow outranks
  today's 3.5 generation automatically — that is the "new models show up by
  themselves" rule.
- A name like `acme-model-999-10000b` stays in the unknown tier (50) and can
  never jump above any lightning/ultra/frontier/pro/super model, regardless
  of the 999 or 10000b in its name.

POWER_OVERRIDES (comma-separated env var, e.g.
POWER_OVERRIDES="moonshotai/kimi-k3,nvidia/nemotron-3.5-lightning-30b-a3b")
pins models at the top of the chain regardless of score.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

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


def rank_key(model_id: str) -> Tuple[float, float, float]:
    """Tier-first sort key: (tier, version, params), descending."""
    return (_tier(model_id), _version(model_id), _params(model_id))


def score_model(model_id: str) -> float:
    """Convenience single-number score for display purposes.

    NOTE: ranking uses the tier-first tuple (`rank_key`), not this number,
    so that no amount of version/parameter numerics can overcome a lower
    tier. Kept for backwards compatibility and logs.
    """
    tier, version, params = rank_key(model_id)
    return tier + 0.005 * params + 0.1 * version


def rank_models(
    entries: List[CatalogEntry],
    overrides: Optional[List[str]] = None,
) -> List[CatalogEntry]:
    """Sort entries most-powerful-first. Pinned overrides come first, in order.

    Tier is compared absolutely first, so a model with an unrecognized
    (unknown-tier) name can never outrank a known top-tier model.
    """
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
        tier, version, params = rank_key(e.model)
        return (pin, -tier, -version, -params)

    return sorted(entries, key=key)


def find_new_models(known: List[str], discovered: List[CatalogEntry]) -> List[str]:
    """Model ids seen in discovery that were not in the known list."""
    known_set = set(known)
    return [e.model for e in discovered if e.model not in known_set]
