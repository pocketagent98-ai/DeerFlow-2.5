"""Tests for the providers package — run WITHOUT any API keys.

Run:  python -m pytest providers/tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from providers.discovery import (  # noqa: E402
    CatalogEntry,
    ModelCatalog,
    NVIDIA_BASE_URL,
    ZAI_BASE_URL,
    ZAI_FREE_MODELS,
)
from providers.model_router import ModelRouter, ModelSpec, RouterError  # noqa: E402
from providers.ranker import rank_models, score_model  # noqa: E402


def entry(provider, model):
    base = NVIDIA_BASE_URL if provider == "nvidia" else ZAI_BASE_URL
    env = "NVIDIA_API_KEY" if provider == "nvidia" else "ZAI_API_KEY"
    return CatalogEntry(provider=provider, model=model, api_base=base, api_key_env=env)


class TestRanker(unittest.TestCase):
    def test_lightning_ranks_top_by_default(self):
        ranked = rank_models([
            entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            entry("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
            entry("nvidia", "moonshotai/kimi-k3"),
            entry("nvidia", "deepseek-ai/deepseek-v4-pro-0813"),
            entry("zai", "glm-4.5-flash"),
        ])
        self.assertEqual(ranked[0].model, "nvidia/nemotron-3.5-lightning-30b-a3b")

    def test_new_more_powerful_model_auto_promotes(self):
        # NVIDIA launches a next-generation ultra -> must beat 3.5-lightning
        ranked = rank_models([
            entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            entry("nvidia", "nvidia/nemotron-4-ultra-550b-a55b"),
        ])
        self.assertEqual(ranked[0].model, "nvidia/nemotron-4-ultra-550b-a55b")

    def test_next_gen_lightning_also_promotes(self):
        ranked = rank_models([
            entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            entry("nvidia", "nvidia/nemotron-4-lightning-30b-a3b"),
        ])
        self.assertEqual(ranked[0].model, "nvidia/nemotron-4-lightning-30b-a3b")

    def test_unknown_model_does_not_beat_top_tier(self):
        # An unrecognized new model scores mid-tier and is not auto-promoted
        ranked = rank_models([
            entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            entry("nvidia", "acme/mystery-model"),
        ])
        self.assertEqual(ranked[0].model, "nvidia/nemotron-3.5-lightning-30b-a3b")

    def test_overrides_pin_first(self):
        ranked = rank_models(
            [
                entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
                entry("nvidia", "moonshotai/kimi-k3"),
            ],
            overrides=["moonshotai/kimi-k3"],
        )
        self.assertEqual(ranked[0].model, "moonshotai/kimi-k3")

    def test_zai_free_models_rank_below_nvidia_top(self):
        ranked = rank_models([
            entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            entry("zai", "glm-4.7-flash"),
            entry("zai", "glm-4.5-flash"),
            entry("zai", "glm-4.6v-flash"),
        ])
        self.assertEqual(ranked[0].provider, "nvidia")
        self.assertEqual(len([e for e in ranked if e.provider == "zai"]), 3)


class FakeModelsClient:
    def __init__(self, ids):
        self.ids = ids

    class _M:
        def __init__(self, id):
            self.id = id

    class _Models:
        def __init__(self, ids):
            self._ids = ids

        def list(self):
            return [FakeModelsClient._M(i) for i in self._ids]

    @property
    def models(self):
        return self._Models(self.ids)


class TestDiscovery(unittest.TestCase):
    def test_nvidia_discovery_is_live(self):
        def factory(base, key):
            if "nvidia" in base:
                return FakeModelsClient(
                    ["nvidia/nemotron-3.5-lightning-30b-a3b", "moonshotai/kimi-k3", "acme/brand-new"]
                )
            raise RuntimeError("z.ai models endpoint unavailable")  # -> static free set

        catalog = ModelCatalog(client_factory=factory)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-x", "ZAI_API_KEY": "z-x"}):
            entries = catalog.discover()
        models = [e.model for e in entries]
        self.assertIn("acme/brand-new", models)      # unknown new model shows up
        self.assertIn("nvidia/nemotron-3.5-lightning-30b-a3b", models)
        zai_models = [e.model for e in entries if e.provider == "zai"]
        self.assertEqual(sorted(zai_models), sorted(ZAI_FREE_MODELS))

    def test_new_model_appears_after_refresh(self):
        # Simulates: catalog fetched, then NVIDIA launches a new model,
        # then the TTL expires and the new model is discovered automatically.
        live_ids = {"ids": ["nvidia/nemotron-3.5-lightning-30b-a3b"]}

        def factory(base, key):
            return FakeModelsClient(live_ids["ids"])

        catalog = ModelCatalog(ttl_seconds=0.0, client_factory=factory)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-x", "ZAI_API_KEY": "z-x"}):
            first = catalog.discover()
            live_ids["ids"].append("nvidia/nemotron-4-lightning-30b-a3b")  # launch!
            second = catalog.discover(force=True)
        self.assertEqual(len([e for e in first if e.provider == "nvidia"]), 1)
        self.assertEqual(len([e for e in second if e.provider == "nvidia"]), 2)

    def test_missing_keys_give_empty_catalog(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            entries = ModelCatalog().discover()
        self.assertEqual(entries, [])


class FakeResponse:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

    def model_dump(self):
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class FakeStatusError(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.headers = {"retry-after": str(retry_after)} if retry_after is not None else {}


class FakeChatClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    class _Completions:
        def __init__(self, parent):
            self._p = parent

        def create(self, **kw):
            self._p.calls += 1
            outcome = self._p.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    @property
    def chat(self):
        return type("Chat", (), {"completions": self._Completions(self)})()


class TestRouterChain(unittest.TestCase):
    def make_router(self, entries, clients_by_label, **kw):
        catalog = ModelCatalog(ttl_seconds=0.0)
        catalog._entries = entries
        catalog._fetched_at = 1e18  # never stale
        return ModelRouter(
            catalog=catalog,
            client_factory=lambda spec: clients_by_label[spec.label],
            sleep=lambda _s: None,
            **kw,
        )

    def test_chain_is_nvidia_first_then_zai(self):
        router = self.make_router(
            [
                entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
                entry("nvidia", "moonshotai/kimi-k3"),
                entry("zai", "glm-4.5-flash"),
                entry("zai", "glm-4.7-flash"),
            ],
            {},
        )
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            labels = router.current_chain
        self.assertEqual(labels[0], "nvidia:nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertTrue(labels[-1].startswith("zai:"))
        self.assertEqual(len(labels), 4)

    def test_new_model_promotes_in_chain_automatically(self):
        entries = [
            entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            entry("zai", "glm-4.5-flash"),
        ]
        router = self.make_router(entries, {})
        # NVIDIA launches a more powerful model; discovery now returns it
        router.catalog._entries.append(entry("nvidia", "nvidia/nemotron-4-ultra-550b-a55b"))
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            labels = router.current_chain
        self.assertEqual(labels[0], "nvidia:nvidia/nemotron-4-ultra-550b-a55b")

    def test_429_retries_then_succeeds(self):
        spec_entries = [entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b")]
        client = FakeChatClient([FakeStatusError(429, retry_after=1), FakeResponse("ok")])
        router = self.make_router(spec_entries, {"nvidia:nvidia/nemotron-3.5-lightning-30b-a3b": client})
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            out = router.ask("hi")
        self.assertEqual(out, "ok")
        self.assertEqual(client.calls, 2)

    def test_429_exhausted_falls_back_to_zai_free(self):
        nvidia_client = FakeChatClient(
            [FakeStatusError(429), FakeStatusError(429), FakeStatusError(429)]
        )
        zai_client = FakeChatClient([FakeResponse("from zai free")])
        router = self.make_router(
            [
                entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
                entry("zai", "glm-4.5-flash"),
            ],
            {
                "nvidia:nvidia/nemotron-3.5-lightning-30b-a3b": nvidia_client,
                "zai:glm-4.5-flash": zai_client,
            },
        )
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            out = router.ask("hi")
        self.assertEqual(out, "from zai free")
        self.assertEqual(nvidia_client.calls, 3)   # 1 + 2 retries
        self.assertEqual(zai_client.calls, 1)

    def test_hard_error_falls_through_without_retry(self):
        nvidia_client = FakeChatClient([FakeStatusError(401)])
        zai_client = FakeChatClient([FakeResponse("ok")])
        router = self.make_router(
            [
                entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
                entry("zai", "glm-4.5-flash"),
            ],
            {
                "nvidia:nvidia/nemotron-3.5-lightning-30b-a3b": nvidia_client,
                "zai:glm-4.5-flash": zai_client,
            },
        )
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            out = router.ask("hi")
        self.assertEqual(out, "ok")
        self.assertEqual(nvidia_client.calls, 1)

    def test_all_fail_raises(self):
        c1 = FakeChatClient([FakeStatusError(503)] * 3)
        c2 = FakeChatClient([FakeStatusError(500)] * 3)
        router = self.make_router(
            [
                entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
                entry("zai", "glm-4.5-flash"),
            ],
            {
                "nvidia:nvidia/nemotron-3.5-lightning-30b-a3b": c1,
                "zai:glm-4.5-flash": c2,
            },
        )
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            with self.assertRaises(RouterError):
                router.ask("hi")

    def test_missing_key_models_skipped(self):
        router = self.make_router(
            [entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
             entry("zai", "glm-4.5-flash")],
            {},
        )
        with mock.patch.dict(os.environ, {"ZAI_API_KEY": "k"}, clear=True):
            os.environ.pop("NVIDIA_API_KEY", None)
            labels = [s.label for s in router.available_chain()]
        self.assertEqual(labels, ["zai:glm-4.5-flash"])


class AlwaysFailsClient:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    @property
    def chat(self):
        parent = self

        class _Completions:
            def create(self, **kw):
                parent.calls += 1
                raise parent.exc

        return type("Chat", (), {"completions": _Completions()})()


class TestZaiSafety(unittest.TestCase):
    def test_zai_never_falls_back_to_arbitrary_models(self):
        # Endpoint UP but none of the three known free models are listed:
        # NOTHING must be used — never arbitrary (possibly paid) models.
        def factory(base, key):
            if "z.ai" in base:
                return FakeModelsClient(["glm-4.6", "some-paid-model", "another-paid-model"])
            return FakeModelsClient(["nvidia/nemotron-3.5-lightning-30b-a3b"])

        catalog = ModelCatalog(ttl_seconds=0.0, client_factory=factory)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            entries = catalog.discover()
        zai = [e.model for e in entries if e.provider == "zai"]
        self.assertEqual(zai, [])

    def test_zai_uses_exactly_the_three_free_models(self):
        # Endpoint UP and the free models are listed: exactly those three,
        # even if other (paid) models are also listed.
        def factory(base, key):
            if "z.ai" in base:
                return FakeModelsClient(list(ZAI_FREE_MODELS) + ["glm-4.6-plus", "glm-4.7"])
            return FakeModelsClient(["nvidia/nemotron-3.5-lightning-30b-a3b"])

        catalog = ModelCatalog(ttl_seconds=0.0, client_factory=factory)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            entries = catalog.discover()
        zai = sorted(e.model for e in entries if e.provider == "zai")
        self.assertEqual(zai, sorted(ZAI_FREE_MODELS))


class TestRankerSafety(unittest.TestCase):
    def test_ridiculous_numbers_never_outrank_top_tier(self):
        # A name full of huge numbers but no tier keyword must stay at the
        # BOTTOM, below every known top-tier model (tier is compared first).
        ranked = rank_models([
            entry("nvidia", "acme-model-999-10000b"),
            entry("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
            entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
        ])
        models = [e.model for e in ranked]
        self.assertEqual(models[0], "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertEqual(models[-1], "acme-model-999-10000b")

    def test_newer_generation_still_promotes_within_tier(self):
        # The auto-upgrade rule must survive the tier-first fix.
        ranked = rank_models([
            entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            entry("nvidia", "nvidia/nemotron-4-ultra-550b-a55b"),
        ])
        self.assertEqual(ranked[0].model, "nvidia/nemotron-4-ultra-550b-a55b")


class TestCircuitBreaker(unittest.TestCase):
    def _router(self, nvidia_client, zai_client, **kw):
        catalog = ModelCatalog(ttl_seconds=0.0)
        catalog._entries = [
            entry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            entry("zai", "glm-4.5-flash"),
        ]
        catalog._fetched_at = 1e18  # never stale
        clients = {
            "nvidia:nvidia/nemotron-3.5-lightning-30b-a3b": nvidia_client,
            "zai:glm-4.5-flash": zai_client,
        }
        return ModelRouter(
            catalog=catalog,
            client_factory=lambda spec: clients[spec.label],
            sleep=lambda _s: None,
            **kw,
        )

    def test_opens_and_skips_failing_model(self):
        nvidia = AlwaysFailsClient(FakeStatusError(401))
        zai = FakeChatClient([FakeResponse("zai ok"), FakeResponse("zai ok again")])
        router = self._router(nvidia, zai, failure_threshold=1, recovery_seconds=3600)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            self.assertEqual(router.ask("hi"), "zai ok")
            # nvidia has now failed once -> breaker open -> second call skips it
            self.assertEqual(router.ask("hi"), "zai ok again")
        self.assertEqual(nvidia.calls, 1)  # never retried after opening

    def test_recovers_after_recovery_window(self):
        nvidia = AlwaysFailsClient(FakeStatusError(401))
        zai = FakeChatClient([FakeResponse("zai ok"), FakeResponse("zai ok 2")])
        # recovery_seconds=0 -> the breaker immediately allows a new probe
        router = self._router(nvidia, zai, failure_threshold=1, recovery_seconds=0.0)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            router.ask("hi")
            router.ask("hi again")  # nvidia is probed again (half-open)
        self.assertEqual(nvidia.calls, 2)
        self.assertEqual(zai.calls, 2)

    def test_success_resets_the_breaker(self):
        nvidia = FakeChatClient([FakeStatusError(401), FakeResponse("n ok"), FakeResponse("n ok 2")])
        zai = FakeChatClient([FakeResponse("z ok")])
        # threshold 2: one hard failure is recorded but does NOT open the breaker
        router = self._router(nvidia, zai, failure_threshold=2, recovery_seconds=3600)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            self.assertEqual(router.ask("hi"), "z ok")    # nvidia hard-failed once
            self.assertEqual(router.ask("hi"), "n ok")     # success -> breaker reset
            self.assertEqual(router.ask("hi"), "n ok 2")   # still in chain, not open
        self.assertEqual(router.stats["nvidia:nvidia/nemotron-3.5-lightning-30b-a3b"]["failure"], 1)
        self.assertEqual(router.stats["nvidia:nvidia/nemotron-3.5-lightning-30b-a3b"]["success"], 2)
        self.assertNotIn("nvidia:nvidia/nemotron-3.5-lightning-30b-a3b", router._breaker)

    def test_stats_recorded_for_all_models(self):
        nvidia = FakeChatClient([FakeStatusError(429), FakeResponse("n"), FakeResponse("n2")])
        zai = FakeChatClient([FakeResponse("z")])
        router = self._router(nvidia, zai)
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "k", "ZAI_API_KEY": "k"}):
            router.ask("1")   # 429 then success on nvidia (after retry)
        self.assertEqual(router.stats["nvidia:nvidia/nemotron-3.5-lightning-30b-a3b"]["success"], 1)
        # zai was never attempted, so it has no stats entry yet
        self.assertNotIn("zai:glm-4.5-flash", router.stats)


if __name__ == "__main__":
    unittest.main()
