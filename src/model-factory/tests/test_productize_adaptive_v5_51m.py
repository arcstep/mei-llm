from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import orchestration.productize_adaptive_v5_51m as productizer
from mei_sdk.protocol import normalize_request


class AdaptiveV5ProductizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.args = productizer.parse_args([])
        cls.plan = productizer.build_plan(cls.args)

    def test_plan_freezes_exact_replay_budgets_and_runtime_policy(self) -> None:
        recipe = self.plan["immutable"]["recipe"]
        self.assertEqual(recipe["fullcall_alignment_steps"], 2_000)
        self.assertEqual(recipe["agent_alignment_steps"], 1_000)
        self.assertEqual(recipe["retrieval_r2_steps"], 1_600)
        self.assertEqual(recipe["mw_steps"], 2_000)
        self.assertEqual(recipe["confidence_steps"], 800)
        self.assertEqual(recipe["narration_steps"], 1_200)
        policy = self.plan["immutable"]["runtime_policy"]
        self.assertEqual(policy["candidate_batch_size"], 5)
        self.assertEqual(policy["max_context_tokens"], 2048)
        self.assertEqual(policy["default_output_reserve"], 128)
        self.assertEqual(policy["profiles"], {"compact": 1024, "standard": 1536})

    def test_phase_boundary_is_an_explicit_stage(self) -> None:
        args = productizer.parse_args(
            [
                "--phase-scope",
                "model-evaluation",
                "--stop-after-stage",
                "sidecar_runtime_eval_v5",
                "--dry-run",
            ]
        )
        self.assertEqual(args.stop_after_stage, "sidecar_runtime_eval_v5")
        self.assertEqual(args.phase_scope, "model-evaluation")

    def test_phase_scope_never_executes_another_phase(self) -> None:
        self.assertEqual(
            productizer.phase_stage_mode(
                "sft-alignment", "adaptive_generation_eval_v5"
            ),
            "skip",
        )
        self.assertEqual(
            productizer.phase_stage_mode(
                "model-evaluation", "confidence_head_v5"
            ),
            "reuse_only",
        )
        self.assertEqual(
            productizer.phase_stage_mode(
                "runtime-release", "python_runtime_gate_v5"
            ),
            "execute",
        )

    def test_stage_graph_reuses_cpt_qat_and_retrains_mw_from_zero(self) -> None:
        stage_ids = [row["stage_id"] for row in self.plan["stages"]]
        self.assertEqual(tuple(stage_ids), productizer.STAGES)
        self.assertNotIn("cpt", stage_ids)
        self.assertNotIn("qat", stage_ids)
        self.assertIn("adopt_training_prefix_v4", stage_ids)
        self.assertIn("mw_disposition_v5", stage_ids)
        self.assertLess(
            stage_ids.index("agent_alignment_replay_v5"),
            stage_ids.index("retrieval_r2_v5"),
        )
        self.assertLess(
            stage_ids.index("final_adaptive_artifacts_v5"),
            stage_ids.index("mw_disposition_v5"),
        )

    def test_preflight_validates_all_frozen_inventory_and_tool_profiles(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mei-adaptive-v5-test-") as raw:
            receipt = productizer.preflight(self.args, self.plan, Path(raw))
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["counts"]["mw_train"], 13_763)
        self.assertEqual(receipt["profile_eligibility"]["compact"]["ineligible"], 0)
        self.assertEqual(receipt["profile_eligibility"]["standard"]["ineligible"], 0)
        self.assertTrue(receipt["joint_context_contract"]["assistant_suffix_included"])

    def test_eval_inventory_has_separate_locked_test_rows(self) -> None:
        rows = productizer._data_rows(self.args)
        self.assertEqual(len(rows["confidence_dev"]), 716)
        self.assertEqual(len(rows["confidence_test"]), 716)
        self.assertIsNot(rows["confidence_dev"], rows["confidence_test"])

    def test_browser_gate_records_real_heap_and_warm_decode(self) -> None:
        measured = {
            "wasm_heap_peak_bytes": 82_968_576,
            "steady_decode_tok_s": 63.65,
            "steady_decode_ran": True,
            "measurement_profile": "split-prefill-plus-steady-decode-v2",
        }
        slow = productizer._browser_gate_metrics(
            measured,
            functional=True,
            build_returncode=0,
            smoke_returncode=0,
        )
        self.assertTrue(slow["heap_within_96_mib"])
        self.assertFalse(slow["warm_steady_decode_100_tok_s_validated"])
        self.assertTrue(slow["degraded"])
        fast = productizer._browser_gate_metrics(
            {**measured, "steady_decode_tok_s": 100.0},
            functional=True,
            build_returncode=0,
            smoke_returncode=0,
        )
        self.assertFalse(fast["degraded"])

    def test_joint_budget_stress_uses_product_output_reserve(self) -> None:
        request = {"query": "probe", "max_new": 1, "history": []}
        stressed = productizer._joint_budget_stress_request(
            request, output_reserve=128
        )
        self.assertEqual(request["max_new"], 1)
        self.assertEqual(stressed["max_new"], 128)
        self.assertNotEqual(stressed["query"], request["query"])
        self.assertEqual(len(stressed["history"]), 2)

    def test_joint_budget_stress_rejects_invalid_output_reserve(self) -> None:
        with self.assertRaises(ValueError):
            productizer._joint_budget_stress_request({}, output_reserve=0)

    def test_adaptive_eval_mw_override_conforms_to_wire_v2(self) -> None:
        normalized = normalize_request(
            {
                "wire_version": "mei-runtime-wire-v2",
                "query": "评测请求",
                "mw": dict(productizer.EVALUATION_MW_OVERRIDE),
            }
        )
        self.assertEqual(normalized["mw"]["source"], "protocol-test")

    def test_recovery_plan_adopts_hash_bound_prefix_and_starts_at_eval(self) -> None:
        source = (
            productizer.ROOT
            / "cycles/mei-1.1-51m/exp-00300m/runs/"
            "productize-scratch300m-adaptive-v5-cq2-v2-164574857928"
        )
        args = productizer.parse_args(
            ["--adopt-adaptive-prefix-run", str(source)]
        )
        plan = productizer.build_plan(args)
        stage_ids = [row["stage_id"] for row in plan["stages"]]
        self.assertEqual(
            stage_ids,
            [
                productizer.ADAPTIVE_PREFIX_ADOPTION_STAGE,
                *productizer.ADAPTIVE_DOWNSTREAM_STAGES,
            ],
        )
        self.assertNotIn("fullcall_alignment_replay_v5", stage_ids)
        self.assertNotIn("retrieval_r2_v5", stage_ids)
        evidence = plan["immutable"]["adaptive_prefix_adoption"]
        self.assertEqual(evidence["blocked_boundary"]["terminal_status"], "blocked")
        self.assertFalse(evidence["source_transition"]["lm_or_head_weights_changed"])
        self.assertEqual(len(evidence["verified_outputs"]), 38)

    def test_packaged_recovery_starts_at_python_gate_without_retraining(self) -> None:
        source = (
            productizer.ROOT
            / "cycles/mei-1.1-51m/exp-00300m/runs/"
            "productize-scratch300m-adaptive-v5-cq2-v2-40ba9754076e"
        )
        args = productizer.parse_args(["--adopt-packaged-run", str(source)])
        plan = productizer.build_plan(args)
        stage_ids = [row["stage_id"] for row in plan["stages"]]
        self.assertEqual(
            stage_ids,
            [
                productizer.PACKAGED_RUN_ADOPTION_STAGE,
                *productizer.PACKAGED_RUN_DOWNSTREAM_STAGES,
            ],
        )
        for stage in (
            "fullcall_alignment_replay_v5",
            "agent_alignment_replay_v5",
            "retrieval_r2_v5",
            "mw_disposition_v5",
            "confidence_head_v5",
            "narration_adapter_v5",
            "package_v2_cq2_v5",
        ):
            self.assertNotIn(stage, stage_ids)
        evidence = plan["immutable"]["packaged_run_adoption"]
        self.assertEqual(evidence["blocked_boundary"]["stage_id"], "python_runtime_gate_v5")
        self.assertEqual(evidence["runtime_boundary"]["terminal_status"], "blocked")
        self.assertFalse(evidence["source_transition"]["lm_or_head_weights_changed"])
        self.assertFalse(evidence["source_transition"]["package_bytes_changed"])
        self.assertTrue(Path(evidence["package"]["path"]).is_dir())

    def test_two_adoption_boundaries_are_mutually_exclusive(self) -> None:
        prefix = (
            productizer.ROOT
            / "cycles/mei-1.1-51m/exp-00300m/runs/"
            "productize-scratch300m-adaptive-v5-cq2-v2-164574857928"
        )
        packaged = (
            productizer.ROOT
            / "cycles/mei-1.1-51m/exp-00300m/runs/"
            "productize-scratch300m-adaptive-v5-cq2-v2-40ba9754076e"
        )
        args = productizer.parse_args(
            [
                "--adopt-adaptive-prefix-run",
                str(prefix),
                "--adopt-packaged-run",
                str(packaged),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
            productizer.build_plan(args)


if __name__ == "__main__":
    unittest.main()
