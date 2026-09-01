from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import productize_sft_v3_300m as productizer
import sft_v4_contract_51m as contract


class SftV4ProductizerTests(unittest.TestCase):
    def _schema_only_future_base(
        self, root: Path, *, exposure: int
    ) -> tuple[productizer.argparse.Namespace, Path]:
        """Create identity receipts, never a usable model-quality artifact."""

        fixture = root / str(exposure)
        fixture.mkdir(parents=True)
        base_id = f"mei-1.0-51m-base-cpt{exposure}-schema-fixture"
        weights = fixture / "weights.npz"
        weights.write_bytes(f"schema-only:{exposure}".encode("utf-8"))
        contracts = productizer.architecture_contracts()
        release = fixture / "RELEASE.json"
        release.write_bytes(
            contract.canonical_bytes(
                {
                    "model_id": base_id,
                    "architecture_id": "mei-1.0-51m-arch-v1",
                    "params": 51_463_797,
                    "tokens_seen_exposure": exposure,
                    "weights_sha256": contract.sha_file(weights),
                    "weight_contract_sha256": contracts[
                        "weight_contract_sha256"
                    ],
                    "tokenizer_sha256": contract.sha_file(
                        productizer.TOKENIZER_ZH_V1
                    ),
                    "fixture_scope": "schema_only_not_a_real_base",
                }
            )
            + b"\n"
        )
        master = fixture / "qat-master.npz"
        master.write_bytes(f"schema-only-qat:{exposure}".encode("utf-8"))
        master_sha = contract.sha_file(master)
        worker = fixture / "qat-worker-receipt.json"
        worker.write_bytes(
            contract.canonical_bytes(
                {
                    "terminal_status": "passed",
                    "parent_id": base_id,
                    "quant_math_id": "mei-cq-v2-g128-wht-codebook",
                    "contracts": {
                        name: str(contracts[name])
                        for name in (
                            "weight_contract_sha256",
                            "runtime_profile_sha256",
                            "training_aux_sha256",
                        )
                    },
                    "outputs": {"master_sha256": master_sha},
                    "stage_fingerprint_sha256": contract.sha_bytes(
                        f"worker:{exposure}".encode("utf-8")
                    ),
                    "fixture_scope": "schema_only_not_a_real_qat_run",
                }
            )
            + b"\n"
        )
        candidate = fixture / "qat-import-candidate-receipt.json"
        candidate.write_bytes(
            contract.canonical_bytes(
                {
                    "schema": "mei-cq2-qat-import-candidate-receipt-v1",
                    "status": "passed",
                    "base": {
                        "model_id": base_id,
                        "tokens_seen_exposure": exposure,
                        "weights_sha256": contract.sha_file(weights),
                    },
                    "master": {
                        "path": str(master),
                        "sha256": master_sha,
                    },
                    "worker_receipt": {
                        "path": str(worker),
                        "sha256": contract.sha_file(worker),
                    },
                    "quant_math_id": "mei-cq-v2-g128-wht-codebook",
                    "fixture_scope": "schema_only_not_quality_evidence",
                }
            )
            + b"\n"
        )
        args = productizer.parse_args(
            [
                "--base-release",
                str(release),
                "--base-weights",
                str(weights),
                "--qat-import-receipt",
                str(candidate),
                "--package-id",
                f"schema-fixture-{exposure}",
                "--run-dir",
                str(fixture / "run"),
            ]
        )
        return args, candidate

    def test_plan_freezes_every_input_and_correct_stage_order(self):
        args = productizer.parse_args([])
        with mock.patch.object(productizer.lifecycle, "live_cpt_workers", return_value=[]):
            plan = productizer.build_plan(args)
        self.assertFalse(plan["heavy_execution_deferred"])
        self.assertEqual(
            [row["stage_id"] for row in plan["stages"]],
            list(productizer.STAGES),
        )
        self.assertEqual(
            plan["immutable"]["data_release"]["release_id"], contract.RELEASE_ID
        )
        self.assertEqual(
            plan["immutable"]["linguistic_augmentation"]["release_id"],
            contract.LINGUISTIC_AUGMENTATION_ID,
        )
        self.assertEqual(
            plan["immutable"]["linguistic_augmentation"]["manifest_sha256"],
            contract.sha_file(args.linguistic_augmentation / "manifest.json"),
        )
        self.assertEqual(
            plan["immutable"]["validation_scope"]["profile"],
            "python-browser-wasm",
        )
        self.assertIn("crossgen_schema_eval_v4", productizer.STAGES)
        self.assertEqual(plan["immutable"]["eval_lock"]["id"], contract.EVAL_ID)
        self.assertIn("sft-v4-quality-schema", str(args.run_dir))
        self.assertEqual(args.agent_steps, contract.AGENT_TRAIN_STEPS)
        self.assertEqual(args.fullcall_steps, 7_200)
        self.assertEqual(args.float_control_steps, 7_200)
        self.assertEqual(args.retrieval_r0_steps, 1_600)
        self.assertEqual(args.retrieval_r1_steps, 1_600)
        self.assertEqual(
            plan["immutable"]["narration_release"]["manifest_sha256"],
            contract.sha_file(args.narration_release / "manifest.json"),
        )
        self.assertEqual(
            len({row["stage_fingerprint_sha256"] for row in plan["stages"]}),
            len(productizer.STAGES),
        )

    def test_recipe_change_produces_a_new_run_fingerprint(self):
        first_args = productizer.parse_args([])
        second_args = productizer.parse_args(["--fullcall-steps", "4001"])
        with mock.patch.object(productizer.lifecycle, "live_cpt_workers", return_value=[]):
            first = productizer.build_plan(first_args)
            second = productizer.build_plan(second_args)
        self.assertNotEqual(
            first["run_fingerprint_sha256"], second["run_fingerprint_sha256"]
        )

    def test_live_cpt_guard_fires_before_mlx_or_run_directory_mutation(self):
        args = productizer.parse_args([])
        with mock.patch.object(
            productizer.lifecycle,
            "live_cpt_workers",
            return_value=[{"pid": 7, "tokens_seen": 550_000_000}],
        ):
            plan = productizer.build_plan(args)
        self.assertTrue(plan["heavy_execution_deferred"])
        with tempfile.TemporaryDirectory() as temp_name:
            target = Path(temp_name) / "must-not-exist"
            with self.assertRaisesRegex(RuntimeError, "live CPT owns Metal"):
                productizer.execute(args, plan, target)
            self.assertFalse(target.exists())

    def test_selected_base_requires_its_own_qat_receipt(self):
        with tempfile.TemporaryDirectory() as temp_name:
            args, _candidate = self._schema_only_future_base(
                Path(temp_name), exposure=600_000_000
            )
            args.qat_import_receipt = productizer.DEFAULT_QAT_IMPORT_RECEIPT
            with self.assertRaisesRegex(RuntimeError, "another Base"):
                productizer.build_plan(args)

    def test_future_exposures_reuse_one_stage_graph_without_an_allowlist(self):
        exposures = (
            600_000_000,
            900_000_000,
            1_200_000_000,
            1_500_000_000,
            2_100_000_000,
        )
        plans = []
        with tempfile.TemporaryDirectory() as temp_name, mock.patch.object(
            productizer.lifecycle, "live_cpt_workers", return_value=[]
        ):
            for exposure in exposures:
                args, candidate = self._schema_only_future_base(
                    Path(temp_name), exposure=exposure
                )
                plan = productizer.build_plan(args)
                self.assertEqual(
                    plan["immutable"]["base"]["tokens_seen_exposure"], exposure
                )
                self.assertEqual(
                    plan["immutable"]["qat_import"]["candidate_receipt_sha256"],
                    contract.sha_file(candidate),
                )
                self.assertEqual(
                    [row["stage_id"] for row in plan["stages"]],
                    list(productizer.STAGES),
                )
                plans.append(plan)
        self.assertEqual(
            len({plan["run_fingerprint_sha256"] for plan in plans}),
            len(exposures),
        )


if __name__ == "__main__":
    unittest.main()
