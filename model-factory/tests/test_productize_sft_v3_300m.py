from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orchestration.productize_sft_v3_300m as productizer
import contracts.sft_v4_contract_51m as contract


class SftV4ProductizerTests(unittest.TestCase):
    def _verified_prefix_fixture(self, root: Path, plan: dict) -> Path:
        prefix = root / "prefix"
        prefix.mkdir(parents=True)
        preflight_path = root / "preflight.json"
        preflight_path.write_bytes(
            contract.canonical_bytes(
                {
                    "status": "passed",
                    "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
                }
            )
            + b"\n"
        )
        parent = {
            **plan,
            "execution_preflight": {
                "path": str(preflight_path),
                "sha256": contract.sha_file(preflight_path),
                "preflight_fingerprint_sha256": contract.sha_bytes(b"preflight"),
                "status": "passed",
            },
        }
        productizer.lifecycle.write_json(prefix / "plan.json", parent)
        required_names = {
            "oracle_top5_agent_v4": "agent-master.npz",
            "retrieval_r1_v4": "retrieval-r1.npz",
            "tool_index_v4": "tool-index.json",
        }
        for stage_id in productizer.TRAINING_PREFIX_STAGES:
            directory = prefix / "stages" / stage_id
            directory.mkdir(parents=True)
            output = directory / required_names.get(stage_id, "output.bin")
            output.write_bytes(f"{stage_id}:verified".encode("utf-8"))
            input_fingerprint, input_evidence = (
                productizer.lifecycle._stage_input_fingerprint(prefix, parent, stage_id)
            )
            stage = productizer.lifecycle.stage_map(parent)[stage_id]
            productizer.lifecycle.write_json(
                directory / "receipt.json",
                {
                    "schema": "mei-productization-stage-receipt-v2",
                    "stage_id": stage_id,
                    "stage_fingerprint_sha256": stage["stage_fingerprint_sha256"],
                    "stage_input_fingerprint_sha256": input_fingerprint,
                    "input_artifacts": input_evidence,
                    "run_fingerprint_sha256": parent["run_fingerprint_sha256"],
                    "terminal_status": "passed",
                    "output_hashes": {str(output.resolve()): contract.sha_file(output)},
                },
            )
        return prefix

    def _verified_productization_prefix_fixture(
        self, root: Path, full_plan: dict
    ) -> Path:
        training_prefix = self._verified_prefix_fixture(
            root / "training-prefix-root", full_plan
        )
        adopted_args = productizer.parse_args(
            ["--adopt-training-prefix-run", str(training_prefix)]
        )
        with mock.patch.object(
            productizer.lifecycle, "live_cpt_workers", return_value=[]
        ):
            continuation_plan = productizer.build_plan(adopted_args)

        continuation = root / "productization-prefix"
        continuation.mkdir(parents=True)
        preflight_path = root / "productization-preflight.json"
        preflight_path.write_bytes(
            contract.canonical_bytes(
                {
                    "status": "passed",
                    "run_fingerprint_sha256": continuation_plan[
                        "run_fingerprint_sha256"
                    ],
                }
            )
            + b"\n"
        )
        parent = {
            **continuation_plan,
            "execution_preflight": {
                "path": str(preflight_path),
                "sha256": contract.sha_file(preflight_path),
                "preflight_fingerprint_sha256": contract.sha_bytes(
                    b"productization-preflight"
                ),
                "status": "passed",
            },
        }
        productizer.lifecycle.write_json(continuation / "plan.json", parent)
        paths = productizer._stage_paths(continuation, training_prefix)
        confidence_root = continuation / "stages/confidence_harvest_v4"

        for stage_id in productizer.PRODUCTIZATION_PREFIX_STAGES:
            directory = continuation / "stages" / stage_id
            directory.mkdir(parents=True, exist_ok=True)
            outputs: list[Path]
            if stage_id == productizer.TRAINING_PREFIX_ADOPTION_STAGE:
                report = directory / "training-prefix-adoption.json"
                report.write_bytes(b"verified-adoption")
                outputs = [report, paths["agent"], paths["r1"], paths["index"]]
            elif stage_id == "mw_disposition_v4":
                paths["mw"].write_bytes(b"verified-mw")
                outputs = [paths["mw"]]
            elif stage_id == "confidence_harvest_v4":
                outputs = []
                for name in (
                    "train-outcomes.jsonl",
                    "valid-outcomes.jsonl",
                    "dev-outcomes.jsonl",
                ):
                    output = confidence_root / name
                    output.write_bytes((name + ":verified").encode("utf-8"))
                    outputs.append(output)
            elif stage_id == "confidence_head_v4":
                paths["confidence"].write_bytes(b"verified-confidence")
                paths["calibration"].write_bytes(b'{"scale":1,"bias":0}\n')
                outputs = [paths["confidence"], paths["calibration"]]
            elif stage_id == "narration_adapter_v4":
                paths["narration"].write_bytes(b"verified-narration")
                outputs = [paths["narration"]]
            else:
                output = directory / "output.bin"
                output.write_bytes(f"{stage_id}:verified".encode("utf-8"))
                outputs = [output]

            input_fingerprint, input_evidence = (
                productizer.lifecycle._stage_input_fingerprint(
                    continuation, parent, stage_id
                )
            )
            stage = productizer.lifecycle.stage_map(parent)[stage_id]
            productizer.lifecycle.write_json(
                directory / "receipt.json",
                {
                    "schema": "mei-productization-stage-receipt-v2",
                    "stage_id": stage_id,
                    "stage_fingerprint_sha256": stage["stage_fingerprint_sha256"],
                    "stage_input_fingerprint_sha256": input_fingerprint,
                    "input_artifacts": input_evidence,
                    "run_fingerprint_sha256": parent["run_fingerprint_sha256"],
                    "terminal_status": "passed",
                    "output_hashes": {
                        str(output.resolve()): contract.sha_file(output)
                        for output in outputs
                    },
                },
            )
        return continuation

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
                    "weight_contract_sha256": contracts["weight_contract_sha256"],
                    "tokenizer_sha256": contract.sha_file(productizer.TOKENIZER_ZH_V1),
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
        with mock.patch.object(
            productizer.lifecycle, "live_cpt_workers", return_value=[]
        ):
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
        with mock.patch.object(
            productizer.lifecycle, "live_cpt_workers", return_value=[]
        ):
            first = productizer.build_plan(first_args)
            second = productizer.build_plan(second_args)
        self.assertNotEqual(
            first["run_fingerprint_sha256"], second["run_fingerprint_sha256"]
        )

    def test_training_prefix_adoption_is_hash_bound_and_uses_a_new_stage_graph(self):
        args = productizer.parse_args([])
        with mock.patch.object(
            productizer.lifecycle, "live_cpt_workers", return_value=[]
        ):
            full_plan = productizer.build_plan(args)
        with tempfile.TemporaryDirectory() as temp_name:
            prefix = self._verified_prefix_fixture(Path(temp_name), full_plan)
            evidence = productizer._verify_training_prefix_run(
                prefix, full_plan["immutable"]
            )
            self.assertEqual(
                evidence["stage_ids"], list(productizer.TRAINING_PREFIX_STAGES)
            )
            agent = Path(evidence["artifacts"]["final_lm_master"]["path"])
            original = agent.read_bytes()
            agent.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "not reusable"):
                productizer._verify_training_prefix_run(prefix, full_plan["immutable"])
            agent.write_bytes(original)

            adopted_args = productizer.parse_args(
                ["--adopt-training-prefix-run", str(prefix)]
            )
            with mock.patch.object(
                productizer.lifecycle, "live_cpt_workers", return_value=[]
            ):
                adopted_plan = productizer.build_plan(adopted_args)
            self.assertEqual(
                [row["stage_id"] for row in adopted_plan["stages"]],
                [
                    productizer.TRAINING_PREFIX_ADOPTION_STAGE,
                    *productizer.DOWNSTREAM_STAGES,
                ],
            )
            self.assertNotEqual(
                adopted_plan["run_fingerprint_sha256"],
                full_plan["run_fingerprint_sha256"],
            )

    def test_productization_prefix_adoption_is_recursive_and_hash_bound(self):
        args = productizer.parse_args([])
        with mock.patch.object(
            productizer.lifecycle, "live_cpt_workers", return_value=[]
        ):
            full_plan = productizer.build_plan(args)
        with tempfile.TemporaryDirectory() as temp_name:
            prefix = self._verified_productization_prefix_fixture(
                Path(temp_name), full_plan
            )
            evidence = productizer._verify_productization_prefix_run(
                prefix, full_plan["immutable"]
            )
            self.assertEqual(
                evidence["stage_ids"],
                list(productizer.PRODUCTIZATION_PREFIX_STAGES),
            )
            narration = Path(evidence["artifacts"]["narration"]["path"])
            original = narration.read_bytes()
            narration.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "not reusable"):
                productizer._verify_productization_prefix_run(
                    prefix, full_plan["immutable"]
                )
            narration.write_bytes(original)

            adopted_args = productizer.parse_args(
                ["--adopt-productization-prefix-run", str(prefix)]
            )
            with mock.patch.object(
                productizer.lifecycle, "live_cpt_workers", return_value=[]
            ):
                adopted_plan = productizer.build_plan(adopted_args)
            self.assertEqual(
                [row["stage_id"] for row in adopted_plan["stages"]],
                [
                    productizer.PRODUCTIZATION_PREFIX_ADOPTION_STAGE,
                    *productizer.FINALIZATION_STAGES,
                ],
            )
            self.assertNotEqual(
                adopted_plan["run_fingerprint_sha256"],
                full_plan["run_fingerprint_sha256"],
            )

    def test_prefix_adoption_modes_are_mutually_exclusive(self):
        args = productizer.parse_args(
            [
                "--adopt-training-prefix-run",
                "training",
                "--adopt-productization-prefix-run",
                "productization",
            ]
        )
        with mock.patch.object(
            productizer.lifecycle, "live_cpt_workers", return_value=[]
        ):
            with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
                productizer.build_plan(args)

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

    def test_degraded_hybrid_base_is_productizable_but_not_promotable(self):
        with tempfile.TemporaryDirectory() as temp_name:
            args, _candidate = self._schema_only_future_base(
                Path(temp_name), exposure=600_000_000
            )
            release = contract.load_json(args.base_release)
            release.update(
                {
                    "kind": "base-cpt-candidate",
                    "release_eligible": False,
                    "promotion_prohibited": True,
                    "productization_experiment_eligible": True,
                    "lineage_assurance": "hybrid_recovery",
                    "corpus_quality": {"corpus_diversity_degraded": True},
                    "release_blockers": ["corpus_diversity_degraded"],
                    "numeric_integrity": {
                        "schema": "mei-cpt-numeric-integrity-v1",
                        "status": "passed",
                        "all_numeric_tensors_finite": True,
                        "parameter_names_and_order_exact": True,
                        "parameter_shapes_exact": True,
                        "parameter_tensor_count": 400,
                        "parameter_count": 51_463_797,
                        "train_state_tokens_seen": 600_000_000,
                    },
                }
            )
            args.base_release.write_bytes(contract.canonical_bytes(release) + b"\n")
            qat_candidate = contract.load_json(args.qat_import_receipt)
            qat_candidate["base"]["numeric_integrity"] = release["numeric_integrity"]
            args.qat_import_receipt.write_bytes(
                contract.canonical_bytes(qat_candidate) + b"\n"
            )
            with mock.patch.object(
                productizer.lifecycle, "live_cpt_workers", return_value=[]
            ):
                plan = productizer.build_plan(args)
            base = plan["immutable"]["base"]
            self.assertFalse(base["base_release_eligible"])
            self.assertTrue(base["productization_experiment_eligible"])
            self.assertEqual(base["lineage_assurance"], "hybrid_recovery")
            self.assertTrue(base["corpus_diversity_degraded"])
            self.assertEqual(base["numeric_integrity"]["status"], "passed")

            release["numeric_integrity"]["parameter_count"] = 1
            args.base_release.write_bytes(contract.canonical_bytes(release) + b"\n")
            with self.assertRaisesRegex(RuntimeError, "numeric-integrity"):
                productizer.lifecycle.validate_base(
                    args.base_release, args.base_weights
                )
            release["numeric_integrity"]["parameter_count"] = 51_463_797

            release["productization_experiment_eligible"] = False
            args.base_release.write_bytes(contract.canonical_bytes(release) + b"\n")
            with self.assertRaisesRegex(RuntimeError, "not eligible"):
                productizer.lifecycle.validate_base(
                    args.base_release, args.base_weights
                )

    def test_future_exposures_reuse_one_stage_graph_without_an_allowlist(self):
        exposures = (
            600_000_000,
            900_000_000,
            1_200_000_000,
            1_500_000_000,
            2_100_000_000,
        )
        plans = []
        with (
            tempfile.TemporaryDirectory() as temp_name,
            mock.patch.object(
                productizer.lifecycle, "live_cpt_workers", return_value=[]
            ),
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
