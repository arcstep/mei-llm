from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mei_llm.registry import Registry  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FiveDomainLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry.open(ROOT)

    def test_current_is_byte_identical_to_migration_baseline(self) -> None:
        baseline = json.loads(
            (ROOT / ".internal/registry/migrations/2026-09-four-domain-baseline.json").read_text()
        )
        actual = hashlib.sha256((ROOT / "CURRENT.json").read_bytes()).hexdigest()
        self.assertEqual(baseline["current_json_sha256"], actual)

    def test_five_business_domains_and_thin_source_control_are_visible(self) -> None:
        visible = sorted(
            path.name
            for path in ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        self.assertEqual(
            ["corpus", "cycles", "docs", "models", "src"],
            visible,
        )

    def test_internal_contains_no_unique_python_implementation(self) -> None:
        hidden_sources = list((ROOT / ".internal").rglob("*.py"))
        self.assertEqual([], hidden_sources)

    def test_executed_cycles_bind_model_factory_pipeline_locks(self) -> None:
        for cycle_id in ("exp-000300m", "exp-000600m"):
            cycle = self.registry.cycle(cycle_id)
            lock_path = ROOT / cycle["pipeline_lock"]
            self.assertTrue(lock_path.is_file())
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(lock["cycle_id"], cycle_id)
            self.assertEqual(lock["model_id"], "mei-1.1-51m")
            self.assertEqual(lock["source_capture_mode"], "reconstructed")

    def test_platform_has_no_in_tree_build_cache(self) -> None:
        shared = ROOT / "src/platform" / "_shared"
        self.assertFalse((shared / "target").exists())
        self.assertFalse((shared / "target-check").exists())
        self.assertTrue((ROOT / "src/platform/python-sdk").is_dir())
        self.assertTrue((ROOT / "src/platform/browser-sdk").is_dir())
        self.assertFalse((ROOT / "src/platform/sdk").exists())
        self.assertFalse((ROOT / "src/platform/runtime").exists())
        self.assertFalse((ROOT / "src/platform/packaging").exists())

    def test_legacy_platform_paths_resolve_to_shallow_current_layout(self) -> None:
        cases = {
            "runtime/_shared/byte_grammar.py": "src/platform/_shared/runtime/byte_grammar.py",
            "sdk/python/mei_sdk/engine.py": "src/platform/python-sdk/mei_sdk/engine.py",
            "sdk/js/index.mjs": "src/platform/browser-sdk/index.mjs",
            "sdk/Cargo.toml": "src/platform/_shared/Cargo.toml",
            "sdk/c/mei_sdk.h": "cycles/mei-1.1-51m/_legacy/platform-experimental-c/c/mei_sdk.h",
        }
        for old, new in cases.items():
            self.assertEqual(self.registry.resolve(old), (ROOT / new).resolve())
            self.assertTrue((ROOT / new).is_file(), old)

    def test_every_legacy_training_path_resolves_to_visible_model_factory(self) -> None:
        mapping = json.loads(
            (ROOT / "src/model-factory/contracts/LEGACY_PATH_MAP.json").read_text()
        )
        self.assertEqual(mapping["mapping_count"], len(mapping["exact"]))
        for old, new in mapping["exact"].items():
            self.assertTrue(old.startswith("training/mei-1.0-51m-train-v1/"))
            self.assertTrue(new.startswith("src/model-factory/"))
            self.assertEqual(self.registry.resolve(old), (ROOT / new).resolve())
            self.assertTrue((ROOT / new).is_file(), old)

    def test_intermediate_hidden_training_paths_resolve_to_model_factory(self) -> None:
        mapping = json.loads(
            (ROOT / "src/model-factory/contracts/LEGACY_PATH_MAP.json").read_text()
        )
        cases = dict(mapping["intermediate_hidden_exact"])
        cases.update(
            {
                ".internal/src/mei_llm/training/qat_cq2_v2_51m.py": (
                    "src/model-factory/training/qat/qat_cq2_v2_51m.py"
                ),
                ".internal/src/mei_llm/training/recipes/cpt-lifecycle-v1.json": (
                    "src/model-factory/recipes/cpt-lifecycle-v1.json"
                ),
            }
        )
        for old, new in cases.items():
            self.assertEqual(self.registry.resolve(old), (ROOT / new).resolve())
            self.assertTrue((ROOT / new).is_file(), old)

    def test_executed_cycles_have_contracts(self) -> None:
        for item in self.registry.cycles()["cycles"]:
            if "path" not in item:
                continue
            cycle = self.registry.cycle(item["cycle_id"])
            self.assertEqual(cycle["cycle_id"], item["cycle_id"])
            self.assertTrue((ROOT / cycle["corpus_contract"]).is_file())

    def test_artifact_uris_are_unique_and_cycle_bound(self) -> None:
        entries = self.registry.artifacts()["entries"]
        self.assertEqual(len({entry["uri"] for entry in entries}), len(entries))
        self.assertTrue(all(entry.get("cycle_id") for entry in entries))
        production_cycles = {item["cycle_id"] for item in self.registry.cycles()["cycles"]}
        for entry in entries:
            self.assertIn(entry["cycle_id"], production_cycles)
            self.assertTrue((ROOT / entry["path"]).exists(), entry["uri"])

    def test_current_legacy_base_resolves(self) -> None:
        current = json.loads((ROOT / "CURRENT.json").read_text())
        resolved = self.registry.resolve(current["base"])
        self.assertTrue(str(resolved).endswith("models/mei-1.1-51m/exp-00300m/base/mei-1.0-51m-base-scratch300m-v1"))
        self.assertEqual(
            self.registry.resolve(current["training"]),
            (ROOT / "src/model-factory").resolve(),
        )

    def test_legacy_cycle_and_comparison_runs_resolve(self) -> None:
        cycle_run = self.registry.resolve(
            "training/runs/mei-1.0-51m/"
            "productize-cpt600m-adaptive-v5-cq2-v2-43bdca400ac1/STATUS.json"
        )
        comparison = self.registry.resolve(
            "training/runs/mei-1.0-51m/"
            "adaptive-v5-paired-final-gates-v4/paired-comparison.json"
        )
        self.assertTrue(cycle_run.is_file())
        self.assertTrue(comparison.is_file())
        self.assertIn("exp-00600m/runs", cycle_run.as_posix())
        self.assertIn("comparisons/runs", comparison.as_posix())

    def test_exposure_and_parent_invariants(self) -> None:
        first = self.registry.cycle("exp-000300m")
        second = self.registry.cycle("exp-000600m")
        self.assertIsNone(first["parent_cycle"])
        self.assertEqual(first["actual_exposure_tokens"], first["delta_exposure_tokens"])
        self.assertEqual(second["parent_cycle"], first["cycle_id"])
        self.assertEqual(second["parent_exposure_tokens"], first["actual_exposure_tokens"])
        self.assertEqual(
            second["actual_exposure_tokens"],
            second["parent_exposure_tokens"] + second["delta_exposure_tokens"],
        )

    def test_immutable_base_hashes_survived_migration(self) -> None:
        baseline = json.loads(
            (ROOT / ".internal/registry/migrations/2026-09-four-domain-baseline.json").read_text()
        )["immutable_artifacts"]
        cases = {
            "scratch300m_weights_sha256": ROOT / "cycles/mei-1.1-51m/exp-00300m/models/base/mei-1.0-51m-base-scratch300m-v1/mei-1.0-51m-base-scratch300m-v1.npz",
            "scratch300m_state_sha256": ROOT / "cycles/mei-1.1-51m/exp-00300m/models/base/mei-1.0-51m-base-scratch300m-v1/mei-1.0-51m-base-scratch300m-v1-state.npz",
            "cpt600m_weights_sha256": ROOT / "cycles/mei-1.1-51m/exp-00600m/models/base/mei-1.0-51m-base-cpt600m-clean-source-v3-v1/mei-1.0-51m-base-cpt600m-clean-source-v3-v1.npz",
            "cpt600m_state_sha256": ROOT / "cycles/mei-1.1-51m/exp-00600m/models/base/mei-1.0-51m-base-cpt600m-clean-source-v3-v1/mei-1.0-51m-base-cpt600m-clean-source-v3-v1-state.npz",
        }
        for key, path in cases.items():
            self.assertEqual(sha256_file(path), baseline[key])

    def test_canonical_model_assets_match_hash_locked_inventory(self) -> None:
        releases = ROOT / "models/mei-1.1-51m"
        for cycle_id, legacy_id in (("exp-00300m", "exp-000300m"), ("exp-00600m", "exp-000600m")):
            cycle_root = releases / cycle_id
            manifest = json.loads((cycle_root / "ASSETS.json").read_text())
            self.assertEqual(manifest["cycle_id"], legacy_id)
            self.assertEqual(manifest["off_device_backup_status"], "open")
            for item in manifest["assets"]:
                path = cycle_root / item["path"]
                self.assertTrue(path.is_file(), item["path"])
                self.assertEqual(path.stat().st_size, item["bytes"])
                self.assertEqual(sha256_file(path), item["sha256"])

    def test_legacy_roots_are_not_active(self) -> None:
        for name in (
            "architecture",
            "artifacts",
            "base",
            "corpus-factory",
            "model-factory",
            "notebook",
            "platform",
            "runtime",
            "sdk",
            "sft",
            "tokenizer",
            "training",
        ):
            self.assertFalse((ROOT / name).exists(), name)
        self.assertFalse(any((ROOT / name).is_symlink() for name in (
            "architecture", "base", "corpus", "notebook", "runtime", "sdk", "training"
        )))

    def test_planned_cycles_are_registry_only(self) -> None:
        for item in self.registry.cycles()["cycles"]:
            if item["status"].startswith("planned"):
                self.assertNotIn("path", item)
                import re as _re
                normalized = _re.sub(r"exp-0*(\d+)m", lambda m: f"exp-{int(m.group(1)):05d}m", item["cycle_id"].replace("-v2", ""))
                self.assertFalse((ROOT / "cycles/mei-1.2-51m" / normalized).exists())

    def test_public_docs_do_not_link_private_monorepo_docs(self) -> None:
        for path in ROOT.rglob("*.md"):
            if path.parts[0] in ("corpus", "cycles") or ".git" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("](../docs/", text, str(path))
            self.assertNotIn("](../../docs/", text, str(path))


if __name__ == "__main__":
    unittest.main()
