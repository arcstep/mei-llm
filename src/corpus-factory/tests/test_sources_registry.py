from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


registry = load_module(
    "mei_51m_source_registry", "src/corpus-factory/sources/registry.py"
)
sources = load_module(
    "mei_51m_source_manager", "src/corpus-factory/sources/source_manager.py"
)


class RegistryTests(unittest.TestCase):
    def test_registry_loads_and_validates(self) -> None:
        value = registry.load_registry()
        self.assertEqual("mei-51m-source-registry-v1", value["schema"])
        self.assertEqual(
            ["fineweb2_hq", "wiki_zh", "wiki_en", "dialogue", "structured", "code"],
            registry.roles(value),
        )

    def test_entry_for_resolves_sources(self) -> None:
        value = registry.load_registry()
        hq = registry.entry_for("fineweb2-hq-cmn-hani", value)
        self.assertEqual("epfml/FineWeb2-HQ", hq["repository"])
        self.assertEqual("fineweb2_hq", hq["role"])
        wikidata = registry.entry_for("wikidata-json", value)
        self.assertEqual("record", wikidata["admit"]["dedup_mode"])
        with self.assertRaisesRegex(registry.RegistryError, "unknown source_id"):
            registry.entry_for("not-a-source", value)

    def test_validation_fails_closed(self) -> None:
        base = registry.load_registry()
        broken = copy.deepcopy(base)
        broken["schema"] = "something-else"
        self.assertTrue(registry.validate_registry(broken))
        broken = copy.deepcopy(base)
        broken["sources"][0]["source_id"] = broken["sources"][1]["source_id"]
        self.assertTrue(registry.validate_registry(broken))
        broken = copy.deepcopy(base)
        broken["sources"][0]["band"] = "D"
        self.assertTrue(registry.validate_registry(broken))
        broken = copy.deepcopy(base)
        broken["sources"][0]["manifest_source"]["type"] = "torrent"
        self.assertTrue(registry.validate_registry(broken))
        broken = copy.deepcopy(base)
        del broken["sources"][0]["license"]
        self.assertTrue(registry.validate_registry(broken))
        with self.assertRaisesRegex(registry.RegistryError, "registry not found"):
            registry.load_registry(
                Path(__file__).with_name("_missing_registry.json")
            )

    def test_roles_require_known_band_and_dedup(self) -> None:
        base = registry.load_registry()
        broken = copy.deepcopy(base)
        broken["roles"]["dialogue"]["band_policy"] = "Z"
        self.assertTrue(registry.validate_registry(broken))
        broken = copy.deepcopy(base)
        broken["roles"]["structured"]["dedup_mode"] = "whole-file"
        self.assertTrue(registry.validate_registry(broken))


class LegacyCompatibilityTests(unittest.TestCase):
    def test_source_roles_legacy_kept_for_v1_reads(self) -> None:
        self.assertEqual(("wiki", "fineweb2_hq"), sources.SOURCE_ROLES_LEGACY)
        self.assertFalse(hasattr(sources, "SOURCE_ROLES"))

    def test_v1_plan_mix_behavior_unchanged(self) -> None:
        result = sources.plan_mix(
            300,
            {"wiki": 200, "fineweb2_hq": 100},
            {"wiki": 0, "fineweb2_hq": 0},
            hq_fraction=0.65,
        )
        self.assertEqual("blocked_insufficient_pool", result["status"])
        self.assertEqual({"fineweb2_hq": 95}, result["shortages"])


if __name__ == "__main__":
    unittest.main()
