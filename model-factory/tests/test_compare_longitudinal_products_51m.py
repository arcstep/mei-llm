from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import evaluation.alignment.compare_longitudinal_products_51m as compare


def record(*, exposure: int, weights: str, value: float) -> dict:
    source = {
        ".internal/src/mei_llm/training/pipelines/sft_v3_training_51m.py": "same-training",
        ".internal/src/mei_llm/training/pipelines/productize_51m.py": "same-control",
    }
    return {
        "run_dir": f"/run/{exposure}",
        "base": {
            "id": f"base-{exposure}",
            "exposure_tokens": exposure,
            "release_sha256": f"release-{exposure}",
            "weights_sha256": weights,
        },
        "contracts": {"weight_contract_sha256": "weight"},
        "tokenizer_sha256": "tokenizer",
        "data_release": {"manifest_sha256": "data", "release_fingerprint": "data-fp"},
        "linguistic_augmentation": {
            "manifest_sha256": "linguistic",
            "release_fingerprint": "linguistic-fp",
        },
        "narration_release": {"manifest_sha256": "narration"},
        "eval_lock": {"evaluation_fingerprint": "eval", "lock_sha256": "eval-lock"},
        "recipe": {"seed": 51, "fullcall_steps": 7200},
        "validation_scope": {"profile": "python-browser-wasm"},
        "source_manifest": source,
        "semantic_source_manifest": compare._semantic_source(source),
        "qat_recipe": {"target_tokens": 5_000_000, "seq_len": 512, "seed": 51},
        "qat_corpus_files": {"manifest.json": "corpus"},
        "qat_group_policy": {"group_size": 128},
        "qat_source_files": {
            ".internal/src/mei_llm/training/pipelines/qat_cq2_v2_51m.py": "same-launcher",
            ".internal/src/mei_llm/training/pipelines/cq2_qat_51m.py": "same-worker",
        },
        "qat_numerical_source_files": {
            ".internal/src/mei_llm/training/pipelines/cq2_qat_51m.py": "same-worker",
        },
        "runtime_source_manifest": {
            "platform/python-sdk/mei_sdk/runtime_51m.py": "same-runtime",
            "platform/_shared/rust/mei-sdk-wasm/src/lib.rs": "same-wasm",
        },
        "quant_math_id": "mei-cq-v2-g128-wht-codebook",
        "base_anchor": {"validation_nll": value + 2.0},
        "float_task_control": {"task_control_slice": {"balanced_accuracy": value}},
        "qat": {"metrics": {"valid_loss": value + 1.0}},
        "scorecard": {
            "metrics": {
                "retrieval": {"recall_at_5": value},
                "fullcall": {"false_execute_rate": 1.0 - value},
            }
        },
        "package": {"package_bytes": 100, "tensor_count": 410},
        "resources": {"measurements": {"wasm_heap_peak_bytes": 200}},
        "gates": {"retrieval": {"passed": value > 0.5}},
        "process_complete": True,
        "release_eligible": False,
        "artifact_hashes": {},
        "confounds": [],
    }


class LongitudinalComparisonTest(unittest.TestCase):
    def test_cpt_base_numeric_integrity_is_a_material_boundary(self) -> None:
        release = {
            "kind": "base-cpt-candidate",
            "tokens_seen_exposure": 600_000_000,
        }
        self.assertIn(
            "base_numeric_integrity_unverified",
            {item["code"] for item in compare._base_confounds(release)},
        )
        release["numeric_integrity"] = {
            "schema": "mei-cpt-numeric-integrity-v1",
            "status": "passed",
            "all_numeric_tensors_finite": True,
            "parameter_names_and_order_exact": True,
            "parameter_shapes_exact": True,
            "parameter_tensor_count": 400,
            "parameter_count": 51_463_797,
            "train_state_tokens_seen": 600_000_000,
        }
        self.assertNotIn(
            "base_numeric_integrity_unverified",
            {item["code"] for item in compare._base_confounds(release)},
        )

    def test_exact_pair_is_main_and_causal_curve_eligible(self) -> None:
        baseline = record(exposure=300_000_000, weights="a", value=0.4)
        candidate = record(exposure=600_000_000, weights="b", value=0.7)
        report = compare.build_comparison(baseline, candidate)
        self.assertTrue(report["paired_experiment_complete"])
        self.assertTrue(report["main_scale_curve_eligible"])
        self.assertTrue(report["causal_exposure_attribution_eligible"])
        rows = {row["metric"]: row for row in report["metric_deltas"]}
        self.assertEqual(
            rows["scorecard.metrics.retrieval.recall_at_5"]["outcome"], "improved"
        )
        self.assertEqual(
            rows["scorecard.metrics.fullcall.false_execute_rate"]["outcome"], "improved"
        )
        self.assertEqual(rows["base_anchor.validation_nll"]["outcome"], "regressed")

    def test_control_plane_source_drift_is_visible_not_causal(self) -> None:
        baseline = record(exposure=300_000_000, weights="a", value=0.4)
        candidate = record(exposure=600_000_000, weights="b", value=0.5)
        candidate["source_manifest"] = dict(candidate["source_manifest"])
        candidate["source_manifest"][
            ".internal/src/mei_llm/training/pipelines/productize_51m.py"
        ] = "control-fix"
        candidate["semantic_source_manifest"] = compare._semantic_source(
            candidate["source_manifest"]
        )
        report = compare.build_comparison(baseline, candidate)
        self.assertTrue(report["main_scale_curve_eligible"])
        self.assertFalse(report["causal_exposure_attribution_eligible"])
        self.assertFalse(report["comparability"]["source_manifest_exact"])
        self.assertTrue(report["comparability"]["semantic_source_manifest_exact"])

    def test_material_lineage_confound_excludes_main_curve(self) -> None:
        baseline = record(exposure=300_000_000, weights="a", value=0.4)
        candidate = record(exposure=600_000_000, weights="b", value=0.5)
        candidate["confounds"] = [
            {"code": "lineage_hybrid_recovery", "severity": "material"}
        ]
        report = compare.build_comparison(baseline, candidate)
        self.assertTrue(report["paired_experiment_complete"])
        self.assertFalse(report["main_scale_curve_eligible"])
        self.assertFalse(report["causal_exposure_attribution_eligible"])

    def test_qat_launcher_drift_is_visible_but_not_material(self) -> None:
        baseline = record(exposure=300_000_000, weights="a", value=0.4)
        candidate = record(exposure=600_000_000, weights="b", value=0.5)
        candidate["qat_source_files"] = dict(candidate["qat_source_files"])
        candidate["qat_source_files"][
            ".internal/src/mei_llm/training/pipelines/qat_cq2_v2_51m.py"
        ] = "changed-launcher"
        report = compare.build_comparison(baseline, candidate)
        self.assertFalse(report["comparability"]["qat_source_files_exact"])
        self.assertTrue(report["comparability"]["qat_numerical_source_files_exact"])
        self.assertTrue(report["main_scale_curve_eligible"])
        self.assertFalse(report["causal_exposure_attribution_eligible"])

    def test_qat_numerical_source_drift_is_material_even_when_recipe_matches(
        self,
    ) -> None:
        baseline = record(exposure=300_000_000, weights="a", value=0.4)
        candidate = record(exposure=600_000_000, weights="b", value=0.5)
        candidate["qat_source_files"] = dict(candidate["qat_source_files"])
        candidate["qat_source_files"][
            ".internal/src/mei_llm/training/pipelines/cq2_qat_51m.py"
        ] = "changed-worker"
        candidate["qat_numerical_source_files"] = compare._qat_numerical_source(
            candidate["qat_source_files"]
        )
        report = compare.build_comparison(baseline, candidate)
        self.assertFalse(report["comparability"]["qat_source_files_exact"])
        self.assertFalse(report["comparability"]["qat_numerical_source_files_exact"])
        self.assertFalse(report["main_scale_curve_eligible"])
        self.assertFalse(report["causal_exposure_attribution_eligible"])

    def test_runtime_evaluation_source_drift_is_material(self) -> None:
        baseline = record(exposure=300_000_000, weights="a", value=0.4)
        candidate = record(exposure=600_000_000, weights="b", value=0.5)
        candidate["runtime_source_manifest"] = dict(
            candidate["runtime_source_manifest"]
        )
        candidate["runtime_source_manifest"][
            "platform/python-sdk/mei_sdk/runtime_51m.py"
        ] = "changed-runtime"
        report = compare.build_comparison(baseline, candidate)
        self.assertFalse(
            report["comparability"]["runtime_evaluation_source_exact"]
        )
        self.assertFalse(report["main_scale_curve_eligible"])
        self.assertIn(
            "runtime_evaluation_source_not_exact",
            {item["code"] for item in report["confounds"]},
        )

    def test_missing_qat_numerical_source_evidence_fails_closed(self) -> None:
        baseline = record(exposure=300_000_000, weights="a", value=0.4)
        candidate = record(exposure=600_000_000, weights="b", value=0.5)
        candidate["qat_numerical_source_files"] = {}
        report = compare.build_comparison(baseline, candidate)
        self.assertFalse(report["comparability"]["qat_numerical_source_files_exact"])
        self.assertFalse(report["main_scale_curve_eligible"])
        self.assertIn(
            "qat_numerical_source_not_exact",
            {item["code"] for item in report["confounds"]},
        )

    def test_contract_drift_is_fail_closed_for_curve(self) -> None:
        baseline = record(exposure=300_000_000, weights="a", value=0.4)
        candidate = record(exposure=600_000_000, weights="b", value=0.5)
        candidate["recipe"] = {"seed": 52, "fullcall_steps": 7200}
        report = compare.build_comparison(baseline, candidate)
        self.assertFalse(
            report["comparability"]["frozen_productization_contract_exact"]
        )
        self.assertFalse(report["main_scale_curve_eligible"])

    def test_same_base_is_not_a_scale_pair(self) -> None:
        baseline = record(exposure=300_000_000, weights="a", value=0.4)
        candidate = record(exposure=300_000_000, weights="a", value=0.5)
        report = compare.build_comparison(baseline, candidate)
        self.assertFalse(report["comparability"]["distinct_base_weights"])
        self.assertFalse(report["comparability"]["candidate_exposure_greater"])
        self.assertFalse(report["main_scale_curve_eligible"])

    def test_write_once_is_idempotent_and_refuses_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "comparison.json"
            compare.write_once(path, {"a": 1})
            compare.write_once(path, {"a": 1})
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                compare.write_once(path, {"a": 2})


if __name__ == "__main__":
    unittest.main()
