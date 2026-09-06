from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import orchestration.supervise_paired_adaptive_v5_51m as supervisor


class PairedAdaptiveV5SupervisorTests(unittest.TestCase):
    def _packaged_fixture_run(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        spec = supervisor._specs(
            adopt_completed_prefixes=True,
            adopt_packaged_300m=True,
        )[0]
        args = supervisor.productizer.parse_args(supervisor._productizer_args(spec))
        plan = supervisor.productizer.build_plan(args)
        (root / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        local = {
            "python_runtime_gate_v5": {
                "warm_decode": {"tokens_per_second": 300.0, "target_validated": True},
                "process_rss": {"rss_bytes": 1024},
            },
            "browser_wasm_gate_v5": {
                "functional_complete": True,
                "heap_peak_bytes": 82_000_000,
                "heap_within_96_mib": True,
                "warm_steady_decode_tok_s": 101.0,
                "warm_steady_decode_100_tok_s_validated": True,
            },
            "final_audit_v5": {
                "process_complete": True,
                "release_eligible": False,
                "release_ineligible_reasons": ["fixture_quality"],
            },
        }
        for stage, metrics in local.items():
            receipt = root / "stages" / stage / "receipt.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(
                json.dumps(
                    {
                        "schema": "mei-productization-stage-receipt-v2",
                        "stage_id": stage,
                        "terminal_status": (
                            "degraded" if stage == "browser_wasm_gate_v5" else "passed"
                        ),
                        "metrics": metrics,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    def test_fixed_pair_and_child_commands_are_explicit(self) -> None:
        specs = supervisor._specs()
        self.assertEqual([row["label"] for row in specs], ["300m", "600m"])
        for spec in specs:
            command = supervisor._productizer_args(spec)
            self.assertIn("--base-release", command)
            self.assertIn("--base-weights", command)
            self.assertIn("--qat-import-receipt", command)
            self.assertIn("--seed-run", command)
            self.assertIn("--package-id", command)

    def test_progress_parser_keeps_stage_and_training_payload(self) -> None:
        value = {"mw_step": 600, "loss": 1.25}
        event = supervisor._progress_payload("600m", "mw_disposition_v5", value)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["chain"], "600m")
        self.assertEqual(event["stage"], "mw_disposition_v5")
        self.assertEqual(event["payload"]["mw_step"], 600)

    def test_receipt_state_reports_first_blocker_without_hiding_prior_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mei-supervisor-test-") as raw:
            root = Path(raw)
            for stage, status in (
                ("adopt_training_prefix_v4", "passed"),
                ("seed_adaptive_views_v5", "degraded"),
                ("fullcall_alignment_replay_v5", "blocked"),
            ):
                path = root / "stages" / stage / "receipt.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "terminal_status": status,
                            "error": {"type": "Example"} if status == "blocked" else None,
                        }
                    ),
                    encoding="utf-8",
                )
            state = supervisor._receipt_state(root)
        self.assertEqual(state["last_passed_stage"], "seed_adaptive_views_v5")
        self.assertEqual(
            state["first_blocker"]["stage"], "fullcall_alignment_replay_v5"
        )

    def test_supervisor_dry_plan_is_sequential_and_current_read_only(self) -> None:
        args = supervisor.parse_args(["--dry-run"])
        plan = supervisor.build_plan(args)
        self.assertEqual(plan["sequence"], ["300m", "600m"])
        self.assertEqual(plan["metal_concurrency"], 1)
        self.assertIn("CURRENT.json", plan["mutations_forbidden"])
        self.assertEqual(len(plan["chains"]), 2)

    def test_recovery_plan_binds_both_blocked_prefixes(self) -> None:
        args = supervisor.parse_args(
            ["--dry-run", "--adopt-completed-prefixes"]
        )
        plan = supervisor.build_plan(args)
        self.assertEqual(plan["recovery_mode"], "adopt-verified-adaptive-prefix")
        for chain in plan["chains"]:
            self.assertIn("adaptive_prefix_run", chain)
            self.assertEqual(
                chain["planned_stage_ids"][0],
                supervisor.productizer.ADAPTIVE_PREFIX_ADOPTION_STAGE,
            )
            self.assertNotIn(
                "agent_alignment_replay_v5", chain["planned_stage_ids"]
            )
            self.assertEqual(len(chain["adaptive_prefix_plan_sha256"]), 64)
            self.assertEqual(
                len(chain["adaptive_prefix_blocked_receipt_sha256"]), 64
            )

    def test_mixed_recovery_adopts_packaged_300m_and_prefix_600m(self) -> None:
        args = supervisor.parse_args(
            [
                "--dry-run",
                "--adopt-completed-prefixes",
                "--adopt-packaged-300m",
            ]
        )
        plan = supervisor.build_plan(args)
        self.assertEqual(
            plan["recovery_mode"],
            "adopt-packaged-300m-and-adaptive-prefix-600m",
        )
        by_label = {row["label"]: row for row in plan["chains"]}
        self.assertIn("packaged_run", by_label["300m"])
        self.assertNotIn("adaptive_prefix_run", by_label["300m"])
        self.assertEqual(
            by_label["300m"]["planned_stage_ids"],
            [
                supervisor.productizer.PACKAGED_RUN_ADOPTION_STAGE,
                *supervisor.productizer.PACKAGED_RUN_DOWNSTREAM_STAGES,
            ],
        )
        self.assertIn("adaptive_prefix_run", by_label["600m"])
        self.assertEqual(
            by_label["600m"]["planned_stage_ids"][0],
            supervisor.productizer.ADAPTIVE_PREFIX_ADOPTION_STAGE,
        )

    def test_packaged_recovery_requires_prefix_recovery_mode(self) -> None:
        with self.assertRaises(SystemExit):
            supervisor.parse_args(["--adopt-packaged-300m"])

    def test_explicit_paired_packaged_reevaluation_uses_both_sources(self) -> None:
        run_300 = supervisor.PACKAGED_300_RUN
        run_600 = (
            supervisor.ROOT
            / "cycles/mei-1.1-51m/exp-00600m/runs/"
            "productize-cpt600m-adaptive-v5-cq2-v2-43bdca400ac1"
        )
        args = supervisor.parse_args(
            [
                "--dry-run",
                "--adopt-packaged-300m-run",
                str(run_300),
                "--adopt-packaged-600m-run",
                str(run_600),
            ]
        )
        plan = supervisor.build_plan(args)
        self.assertEqual(plan["recovery_mode"], "adopt-packaged-selected-chains")
        for chain in plan["chains"]:
            self.assertIn("packaged_run", chain)
            self.assertEqual(
                chain["planned_stage_ids"],
                [
                    supervisor.productizer.PACKAGED_RUN_ADOPTION_STAGE,
                    *supervisor.productizer.PACKAGED_RUN_DOWNSTREAM_STAGES,
                ],
            )

    def test_paired_packaged_reevaluation_requires_two_sources(self) -> None:
        with self.assertRaises(SystemExit):
            supervisor.parse_args(
                [
                    "--adopt-packaged-300m-run",
                    str(supervisor.PACKAGED_300_RUN),
                ]
            )

    def test_comparison_and_finalizers_resolve_transitive_adoption(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mei-paired-finalizer-") as raw:
            root = Path(raw)
            run_300 = root / "run-300"
            run_600 = root / "run-600"
            self._packaged_fixture_run(run_300)
            self._packaged_fixture_run(run_600)
            chains = [
                {"label": "300m", "run_dir": str(run_300), "process_complete": True},
                {"label": "600m", "run_dir": str(run_600), "process_complete": True},
            ]
            entry = supervisor._comparison_entry(chains[0])
            self.assertEqual(entry["base"]["exposure_tokens"], 300_000_485)
            self.assertEqual(entry["cq2_qat"]["tokens_seen_qat"], 5_001_216)
            self.assertAlmostEqual(entry["cq2_qat"]["valid_loss"], 3.1912143416702747)
            self.assertIsNotNone(entry["cq2_qat"]["relative_loss_tax"])
            self.assertEqual(entry["fullcall_alignment"]["steps"], 2_000)
            self.assertEqual(entry["agent_alignment"]["steps"], 1_000)
            self.assertEqual(entry["retrieval_r2"]["steps"], 1_600)
            self.assertEqual(entry["narration"]["adapter"]["rank"], 16)
            self.assertEqual(
                entry["adaptive_generation"]["splits"]["test"]["structural"]["n"],
                196,
            )
            comparison = supervisor._write_comparison(root, chains)
            self.assertIsNotNone(comparison)
            assert comparison is not None
            deliverables = supervisor._write_final_deliverables(
                root, chains, comparison
            )
            self.assertTrue(deliverables["process_complete"])
            receipt = json.loads(Path(deliverables["receipt"]).read_text())
            self.assertEqual(len(receipt["artifacts"]), 8)
            audit = json.loads(
                Path(deliverables["artifacts"]["paired_final_audit"]).read_text()
            )
            self.assertEqual(
                [row["id"] for row in audit["f01_f14"]],
                [f"F{index:02d}" for index in range(1, 15)],
            )
            alignment = json.loads(
                Path(deliverables["artifacts"]["alignment_json"]).read_text()
            )
            confidence = next(
                row
                for row in alignment["matrix"]
                if row["feature"] == "execution confidence"
            )
            self.assertEqual(confidence["validation_status"], "validated")


if __name__ == "__main__":
    unittest.main()
