from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mei_llm import cycle_control
from mei_llm.cli import (
    _product_action,
    _product_doctor,
    _training_arguments,
    parse_cli_args,
)
from mei_llm.registry import Registry


SPECIALIZED_SKILLS = {
    "mei-51m-cycle-orchestrator": {"doctor.py", "plan.py", "status.py", "verify.py"},
    "mei-51m-corpus-sourcing": {
        "inventory.py",
        "plan_mix.py",
        "download_hq.py",
        "admit.py",
        "freeze_pool.py",
    },
    "mei-51m-corpus-factory": {
        "doctor.py",
        "build.py",
        "validate.py",
        "build_eval_lock.py",
        "register_targets.py",
    },
    "mei-51m-corpus-quality": {
        "audit_source.py",
        "audit_synthetic.py",
        "audit_sft.py",
        "diagnose_signal.py",
        "compare.py",
        "decide_reuse.py",
    },
    "mei-51m-cpt-training": {
        "doctor.py",
        "plan.py",
        "status.py",
        "run.py",
        "resume.py",
        "register_base.py",
        "propose_freeze.py",
    },
    "mei-51m-productization": {
        "doctor.py",
        "plan.py",
        "run.py",
        "resume.py",
        "status.py",
        "adopt.py",
        "compare.py",
        "final_audit.py",
    },
    "mei-51m-qat-training": {
        "template.py",
        "doctor.py",
        "plan.py",
        "run.py",
        "resume.py",
        "status.py",
    },
    "mei-51m-sft-alignment": {"phase.py"},
    "mei-51m-model-evaluation": {"phase.py"},
    "mei-51m-runtime-release": {"phase.py"},
}


class SkillControlPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry.open(ROOT)

    def test_specialized_skills_have_declared_thin_entrypoints(self) -> None:
        self.assertEqual(10, len(SPECIALIZED_SKILLS))
        for name, expected in SPECIALIZED_SKILLS.items():
            skill = ROOT / "skills" / name
            self.assertTrue((skill / "SKILL.md").is_file(), name)
            actual = {
                path.name
                for path in (skill / "scripts").glob("*.py")
                if not path.name.startswith("_")
            }
            self.assertTrue(expected <= actual, f"{name}: {expected - actual}")

    def test_compatibility_skill_is_routing_only(self) -> None:
        skill = ROOT / "skills/mei-51m-cpt-lifecycle"
        self.assertTrue((skill / "scripts/route.py").is_file())
        files = {
            path.name
            for path in (skill / "scripts").glob("*.py")
            if not path.name.startswith("_")
        }
        self.assertEqual({"route.py"}, files)

    def test_current_pipelines_bind_separate_recipes(self) -> None:
        registry = json.loads(
            (ROOT / "model-factory/contracts/PIPELINES.json").read_text(
                encoding="utf-8"
            )
        )
        current = {
            row["pipeline_id"]: row
            for row in registry["pipelines"]
            if row["status"] == "current"
        }
        cpt = current["mei-51m-cpt-lifecycle-v1"]
        product = current["mei-51m-adaptive-productization-v5"]
        self.assertEqual("cpt", cpt["track"])
        self.assertEqual(
            "model-factory/recipes/cpt-training-v1.json", cpt["recipe"]
        )
        self.assertEqual(
            "model-factory/recipes/productization-adaptive-v5.json",
            product["recipe"],
        )
        self.assertNotEqual(cpt["entrypoint"], product["entrypoint"])

    def test_direct_productization_run_is_compatibility_only(self) -> None:
        with self.assertRaisesRegex(SystemExit, "compatibility-only"):
            _product_action(self.registry, "run", [])

    def test_cycle_plan_does_not_materialize_planned_cycle(self) -> None:
        path = ROOT / "cycles/mei-1.0-51m/exp-000900m-v2"
        self.assertFalse(path.exists())
        result = cycle_control.plan(self.registry, "exp-000900m-v2")
        self.assertFalse(result["materialize_cycle_directory"])
        self.assertEqual(300_000_000, result["required_increment_tokens"])
        self.assertFalse(path.exists())

    def test_cycle_init_proposal_is_write_once_outside_cycle_tree(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "proposal.json"
            first = cycle_control.init_proposal(
                self.registry, "exp-000900m-v2", out=out
            )
            second = cycle_control.init_proposal(
                self.registry, "exp-000900m-v2", out=out
            )
            self.assertEqual(first, second)
            self.assertEqual("ready_for_sourcing", first["status"])

    def test_cycle_registry_and_product_contracts_verify(self) -> None:
        self.assertTrue(cycle_control.verify(self.registry)["ok"])
        self.assertEqual(0, _product_doctor(self.registry))

    def test_cli_exposes_required_control_surfaces(self) -> None:
        cases = (
            ["cycle", "init", "--cycle-id", "exp-000900m-v2"],
            ["cycle", "status", "--cycle-id", "exp-000600m"],
            ["cycle", "resume", "--cycle-id", "exp-000900m-v2"],
            ["cycle", "verify"],
            ["corpus", "source", "inventory", "--role", "wiki"],
            ["corpus", "evaluate", "audit-source", "--manifest", "x"],
            ["cpt", "status", "--run-id", "run"],
            ["base", "register", "--run-id", "run"],
            ["productization", "status", "--run-dir", "run"],
            ["productization", "plan", "--base-release", "release"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertIsNotNone(parse_cli_args(list(argv)))

    def test_training_actions_require_explicit_confirmation(self) -> None:
        with self.assertRaises(SystemExit):
            _training_arguments([], action="fixture")
        self.assertEqual(
            ["--run-dir", "x"],
            _training_arguments(
                ["--confirm-training", "--run-dir", "x"], action="fixture"
            ),
        )

    def test_300m_600m_regression_fixtures_preserve_failure_lessons(self) -> None:
        c300 = self.registry.cycle("exp-000300m")
        c600 = self.registry.cycle("exp-000600m")
        self.assertLess(
            c600["metrics"]["base_valid_loss"], c300["metrics"]["base_valid_loss"]
        )
        self.assertGreater(
            c600["metrics"]["float_task_balanced_accuracy"],
            c300["metrics"]["float_task_balanced_accuracy"],
        )
        self.assertIn("hybrid_recovery", c600["confounds"])
        self.assertIn("corpus_diversity_degraded", c600["confounds"])
        self.assertGreater(
            c600["metrics"]["retrieval_no_match_false_selection_rate"], 0.8
        )
        self.assertLess(c600["metrics"]["mw_dev_macro_f1"], 0.05)
        self.assertGreater(c600["metrics"]["confidence_test_ece"], 0.19)
        self.assertLess(c600["metrics"]["narration_learned_acceptance"], 0.02)
        self.assertFalse(c600["release_eligible"])
        self.assertFalse(c600["eligibility"]["corpus_reuse_eligible"])


if __name__ == "__main__":
    unittest.main()
