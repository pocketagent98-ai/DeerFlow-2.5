"""Integration test: auto-discovered provider models inside real DeerFlow.

Requires the backend package to be importable (it adds the paths itself).
Skips automatically when backend dependencies are missing, so `pytest
providers/tests` still works in the standalone layout.

Run:  python -m pytest providers/tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

try:
    sys.path.insert(0, os.path.join(REPO_ROOT, "backend", "packages", "harness"))
    sys.path.insert(0, os.path.join(REPO_ROOT, "backend", "packages", "extension-api"))
    import deerflow.config  # noqa: F401
    import pydantic  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    BACKEND_AVAILABLE = False

import providers
from providers.discovery import CatalogEntry, NVIDIA_BASE_URL, ZAI_BASE_URL

MINIMAL_CONFIG = (
    "config_version: 44\n"
    "log_level: info\n"
    "sandbox:\n"
    "  use: deerflow.sandbox.local:LocalSandboxProvider\n"
)


class FakeCatalog:
    def discover(self, *a, **kw):
        return [
            CatalogEntry("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b",
                         NVIDIA_BASE_URL, "NVIDIA_API_KEY"),
            CatalogEntry("nvidia", "nvidia/nemotron-3-ultra-550b-a55b",
                         NVIDIA_BASE_URL, "NVIDIA_API_KEY"),
            CatalogEntry("zai", "glm-4.7-flash", ZAI_BASE_URL, "ZAI_API_KEY"),
            CatalogEntry("zai", "glm-4.5-flash", ZAI_BASE_URL, "ZAI_API_KEY"),
        ]


@unittest.skipUnless(BACKEND_AVAILABLE, "DeerFlow backend not importable in this environment")
class TestAutoProvidersIntegration(unittest.TestCase):
    def setUp(self):
        self._orig_catalog = providers.ModelCatalog
        providers.ModelCatalog = FakeCatalog
        self._tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        self._tmp.write(MINIMAL_CONFIG)
        self._tmp.close()
        self._env = mock.patch.dict(os.environ, {
            "NVIDIA_API_KEY": "nvapi-fake",
            "ZAI_API_KEY": "zai-fake",
            "DEER_FLOW_CONFIG_PATH": self._tmp.name,
        })
        self._env.start()
        # fresh config cache for every test
        from deerflow.config.app_config import reset_app_config
        reset_app_config()

    def tearDown(self):
        from deerflow.config.app_config import reset_app_config
        reset_app_config()
        self._env.stop()
        providers.ModelCatalog = self._orig_catalog
        os.unlink(self._tmp.name)

    def test_models_auto_injected_via_get_app_config(self):
        import deerflow.config as dc

        cfg = dc.get_app_config()
        names = [m.name for m in cfg.models]
        self.assertIn("auto-nemotron-3-5-lightning-30b-a3b", names)
        self.assertIn("auto-glm-4-7-flash", names)
        self.assertIn("auto-glm-4-5-flash", names)
        # power ranking: lightning before ultra
        self.assertLess(
            names.index("auto-nemotron-3-5-lightning-30b-a3b"),
            names.index("auto-nemotron-3-ultra-550b-a55b"),
        )
        # config lookup works for injected models
        self.assertIsNotNone(cfg.get_model_config("auto-glm-4-5-flash"))

    def test_no_double_injection_on_cached_config(self):
        import deerflow.config as dc

        first = dc.get_app_config()
        count = len(first.models)
        second = dc.get_app_config()
        self.assertEqual(len(second.models), count)

    def test_user_models_keep_priority(self):
        import deerflow.config as dc

        with open(self._tmp.name, "a") as fh:
            fh.write(
                "models:\n  - name: my-own-model\n"
                "    use: langchain_openai:ChatOpenAI\n"
                "    model: gpt-test\n    api_key: $NVIDIA_API_KEY\n"
            )
        from deerflow.config.app_config import reset_app_config
        reset_app_config()
        cfg = dc.get_app_config()
        names = [m.name for m in cfg.models]
        self.assertEqual(names[0], "my-own-model")
        self.assertTrue(any(n.startswith("auto-") for n in names))

    def test_disable_flag(self):
        import deerflow.config as dc

        with open(self._tmp.name, "a") as fh:
            fh.write(
                "models:\n  - name: my-own-model\n"
                "    use: langchain_openai:ChatOpenAI\n"
                "    model: gpt-test\n    api_key: $NVIDIA_API_KEY\n"
            )
        from deerflow.config.app_config import reset_app_config
        with mock.patch.dict(os.environ, {"DEERFLOW_AUTO_PROVIDERS": "0"}):
            reset_app_config()
            cfg = dc.get_app_config()
        self.assertEqual([m.name for m in cfg.models], ["my-own-model"])

    def test_injection_survives_discovery_failure(self):
        """A broken discovery must never break config retrieval."""
        import deerflow.config as dc

        class BrokenCatalog:
            def discover(self, *a, **kw):
                raise RuntimeError("network down")

        providers.ModelCatalog = BrokenCatalog
        from deerflow.config.app_config import reset_app_config
        reset_app_config()
        cfg = dc.get_app_config()  # must not raise
        self.assertEqual([m.name for m in cfg.models], [])


if __name__ == "__main__":
    unittest.main()
