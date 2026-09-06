from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from common._repo import phase_binding_identity
from mei_llm import phase_binding
from mei_llm.registry import Registry


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PhaseBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry(ROOT)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temp = Path(self.temporary.name)
        self.inputs: dict[str, Path] = {}
        for name in ("architecture", "base-release", "base-weights", "anchor"):
            path = self.temp / name
            path.write_text(name, encoding="utf-8")
            self.inputs[name] = path
        replay = self.temp / "replay"
        replay.mkdir()
        replay_manifest = replay / "MANIFEST.json"
        replay_manifest.write_text("{}", encoding="utf-8")
        self.inputs["replay"] = replay
        self.source_manifest = self.temp / "source-manifest.json"
        self.source_manifest.write_text("{}", encoding="utf-8")
        self.source_closure = self.temp / "source-closure.json"
        recipe = ROOT / "src/model-factory/recipes/phase-qat-cq2-v1.json"
        self.source_closure.write_text(
            json.dumps(
                {
                    "schema": "mei-51m-stage-source-closure-v1",
                    "entries": [
                        {
                            "path": "src/model-factory/recipes/phase-qat-cq2-v1.json",
                            "sha256": sha(recipe),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def binding(self) -> dict:
        recipe = ROOT / "src/model-factory/recipes/phase-qat-cq2-v1.json"
        return {
            "schema": "mei-51m-phase-binding-v1",
            "binding_id": "test-exp-000900m-v2-qat-v1",
            "model_id": "mei-1.0-51m",
            "cycle_id": "exp-000900m-v2",
            "phase": "qat",
            "pipeline_id": "mei-51m-qat-cq2-v1",
            "pipeline_recipe_sha256": sha(recipe),
            "source": {
                "git_revision": "test-revision",
                "dirty_patch_sha256": None,
                "source_manifest": {
                    "ref": str(self.source_manifest),
                    "sha256": sha(self.source_manifest),
                },
                "stage_source_closure": {
                    "ref": str(self.source_closure),
                    "sha256": sha(self.source_closure),
                },
            },
            "inputs": {
                "architecture_contract": {
                    "ref": str(self.inputs["architecture"]),
                    "sha256": sha(self.inputs["architecture"]),
                },
                "base_release": {
                    "ref": str(self.inputs["base-release"]),
                    "sha256": sha(self.inputs["base-release"]),
                },
                "base_weights": {
                    "ref": str(self.inputs["base-weights"]),
                    "sha256": sha(self.inputs["base-weights"]),
                },
                "float_anchor": {
                    "ref": str(self.inputs["anchor"]),
                    "sha256": sha(self.inputs["anchor"]),
                },
                "replay_corpus": {
                    "ref": str(self.inputs["replay"]),
                    "target": "MANIFEST.json",
                    "pass_ref": True,
                    "sha256": sha(self.inputs["replay"] / "MANIFEST.json"),
                },
            },
            "outputs": {
                "run_dir": "cycles/mei-1.1-51m/exp-000900m-v2/runs/test-qat"
            },
            "parameters": {
                "target_tokens": 5000000,
                "seq_len": 512,
                "batch_size": 4,
                "grad_accum": 1,
                "lr": 0.00005,
                "checkpoint_every_steps": 100,
            },
            "current_sha256": self.registry.current_sha256(),
        }

    def write_binding(self, value: dict) -> Path:
        path = self.temp / "binding.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_qat_plan_uses_only_explicit_bound_inputs(self) -> None:
        path = self.write_binding(self.binding())
        plan = phase_binding.invocation(
            self.registry, path, resume=False, dry_run=True
        )
        command = plan["command"]
        self.assertIn(str(self.inputs["base-release"].resolve()), command)
        self.assertIn(str(self.inputs["base-weights"].resolve()), command)
        self.assertIn("--target-tokens", command)
        self.assertIn("--dry-run", command)
        self.assertEqual(
            plan["environment"]["MEI_PHASE_CYCLE_ID"], "exp-000900m-v2"
        )
        self.assertEqual(
            plan["environment"]["MEI_PHASE_BINDING_SHA256"],
            plan["binding_verification"]["binding_sha256"],
        )
        self.assertFalse(plan["binding_verification"]["errors"])
        with mock.patch.dict(os.environ, plan["environment"], clear=False):
            identity = phase_binding_identity()
        self.assertEqual(identity["binding_id"], "test-exp-000900m-v2-qat-v1")
        self.assertEqual(identity["cycle_id"], "exp-000900m-v2")

    def test_input_hash_drift_is_rejected(self) -> None:
        binding = self.binding()
        binding["inputs"]["base_weights"]["sha256"] = "0" * 64
        report = phase_binding.verify(
            self.registry, self.write_binding(binding), require_inputs=True
        )
        self.assertFalse(report["ok"])
        self.assertTrue(any("hash drifted" in error for error in report["errors"]))

    def test_output_cannot_escape_cycle_root(self) -> None:
        binding = self.binding()
        binding["outputs"]["run_dir"] = "cycles/mei-1.1-51m/exp-00600m/runs/wrong"
        report = phase_binding.verify(
            self.registry, self.write_binding(binding), require_inputs=True
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("must stay inside cycle artifact root" in error for error in report["errors"])
        )

    def test_recipe_hash_drift_is_rejected(self) -> None:
        binding = self.binding()
        binding["pipeline_recipe_sha256"] = "0" * 64
        report = phase_binding.verify(
            self.registry, self.write_binding(binding), require_inputs=True
        )
        self.assertFalse(report["ok"])
        self.assertIn("pipeline recipe hash drifted", report["errors"])

    def test_source_closure_drift_is_rejected(self) -> None:
        binding = self.binding()
        self.source_closure.write_text(
            json.dumps(
                {
                    "schema": "mei-51m-stage-source-closure-v1",
                    "entries": [
                        {"path": "CURRENT.json", "sha256": "0" * 64}
                    ],
                }
            ),
            encoding="utf-8",
        )
        binding["source"]["stage_source_closure"]["sha256"] = sha(
            self.source_closure
        )
        report = phase_binding.verify(
            self.registry, self.write_binding(binding), require_inputs=True
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("source closure drifted" in error for error in report["errors"])
        )

    def test_phase_recipes_do_not_contain_exposure_paths(self) -> None:
        for path in (ROOT / "src/model-factory/recipes").glob("phase-*.json"):
            value = path.read_text(encoding="utf-8")
            self.assertNotIn("exp-000300m", value)
            self.assertNotIn("exp-000600m", value)

    def test_template_derives_inputs_and_recipe_hash_from_registry(self) -> None:
        value = phase_binding.template(
            self.registry,
            cycle_id="exp-000900m-v2",
            pipeline_id="mei-51m-qat-cq2-v1",
        )
        self.assertEqual(value["phase"], "qat")
        self.assertIn("base_release", value["inputs"])
        self.assertIn("target_tokens", value["parameters"])
        self.assertEqual(
            value["pipeline_recipe_sha256"],
            sha(ROOT / "src/model-factory/recipes/phase-qat-cq2-v1.json"),
        )

    def test_split_recipes_enforce_true_phase_boundaries(self) -> None:
        bootstrap = phase_binding.load_json(
            ROOT / "src/model-factory/recipes/phase-sft-bootstrap-v4.json"
        )
        sft = phase_binding.load_json(
            ROOT / "src/model-factory/recipes/phase-sft-adaptive-v5.json"
        )
        evaluation = phase_binding.load_json(
            ROOT / "src/model-factory/recipes/phase-model-evaluation-v5.json"
        )
        runtime = phase_binding.load_json(
            ROOT / "src/model-factory/recipes/phase-runtime-release-v5.json"
        )
        self.assertEqual(
            bootstrap["fixed_arguments"][-1], "tool_index_v4"
        )
        self.assertIn("sft-alignment", sft["fixed_arguments"])
        self.assertIn("model-evaluation", evaluation["fixed_arguments"])
        self.assertNotIn(
            "adaptive_generation_eval_v5",
            evaluation["requires_existing_stages"],
        )
        self.assertIn("runtime-release", runtime["fixed_arguments"])


if __name__ == "__main__":
    unittest.main()
