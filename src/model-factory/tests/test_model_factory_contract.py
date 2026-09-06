from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FACTORY = ROOT / "src/model-factory"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected object: {path}")
    return value


class ModelFactoryContractTest(unittest.TestCase):
    def test_training_and_evaluation_subdomains_are_self_describing(self) -> None:
        catalog = load_json(FACTORY / "contracts/CODE_CATALOG.json")
        declared = {row["path"] for row in catalog["directory_contracts"]}
        expected = {
            "src/model-factory/training/cpt",
            "src/model-factory/training/qat",
            "src/model-factory/training/tool_use",
            "src/model-factory/training/heads",
            "src/model-factory/evaluation/base",
            "src/model-factory/evaluation/tool_use",
            "src/model-factory/evaluation/heads",
            "src/model-factory/evaluation/resources",
            "src/model-factory/evaluation/alignment",
        }
        self.assertTrue(expected.issubset(declared))
        for path in expected:
            self.assertTrue((ROOT / path / "README.md").is_file(), path)

    def test_every_python_source_has_a_directory_contract(self) -> None:
        catalog = load_json(FACTORY / "contracts/CODE_CATALOG.json")
        roots = {
            Path(row["path"]).relative_to("src/model-factory").parts[0]
            for row in catalog["directory_contracts"]
        }
        sources = [
            path
            for path in FACTORY.rglob("*.py")
            if "__pycache__" not in path.parts
        ]
        uncovered = sorted(
            path.relative_to(FACTORY).as_posix()
            for path in sources
            if path.relative_to(FACTORY).parts[0] not in roots
        )
        self.assertEqual([], uncovered)

    def test_current_entrypoints_resolve_and_are_registered(self) -> None:
        factory = load_json(FACTORY / "FACTORY.json")
        pipelines = load_json(FACTORY / "contracts/PIPELINES.json")["pipelines"]
        current = {
            row["entrypoint"]
            for row in pipelines
            if row["status"] == "current"
        }
        declared = {
            value
            for key, value in factory["current_entrypoints"].items()
            if key != "source_recovery_audit"
        }
        self.assertEqual(current, declared)
        for module in {*declared, factory["current_entrypoints"]["source_recovery_audit"]}:
            self.assertTrue((FACTORY / (module.replace(".", "/") + ".py")).is_file())

    def test_executed_cycle_locks_cover_current_productization_stages(self) -> None:
        pipeline = next(
            row
            for row in load_json(FACTORY / "contracts/PIPELINES.json")["pipelines"]
            if row["pipeline_id"] == "mei-51m-adaptive-productization-v5"
        )
        for cycle_id, dir_name in (("exp-000300m", "exp-00300m"), ("exp-000600m", "exp-00600m")):
            lock = load_json(
                ROOT / f"cycles/mei-1.1-51m/{dir_name}/pipeline/PIPELINE.lock.json"
            )
            self.assertEqual(lock["pipeline_id"], pipeline["pipeline_id"])
            self.assertTrue(set(pipeline["stages"]).issubset(lock["stages"]))
            self.assertEqual(lock["source_capture_mode"], "reconstructed")
            self.assertFalse(lock["exact_reproducible"])
            self.assertGreater(lock["source_recovery"]["exact_unavailable"], 0)
            self.assertEqual(len(lock["run_components"]), 3)

    def test_compatibility_snapshot_receipt_is_hash_shaped(self) -> None:
        receipt = load_json(FACTORY / "compatibility/PRE_REFACTOR_SOURCE_SNAPSHOT.json")
        self.assertEqual(len(receipt["sha256"]), 64)
        self.assertGreater(receipt["bytes"], 0)
        self.assertTrue(receipt["artifact"].startswith(".local/recovery/"))


if __name__ == "__main__":
    unittest.main()
