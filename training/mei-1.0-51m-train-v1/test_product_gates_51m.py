from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import eval_lock_v2_51m as lock
import hard_gate_summary_51m as hard
import qat_q4_three_runtime_parity_51m as parity
import resource_baseline_51m as resource
from identity_51m import qat_checkpoint_improved


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


LOCK_LAYERS = {
    "recall_at_5_learned_min": 0.2,
    "recall_at_5_must_exceed_lexical": True,
    "oracle_top5_fullcall_exact_min": 0.12,
    "oracle_top5_execute_refuse_ok_min": 0.5,
    "learned_top5_e2e_exact_min": 0.04,
    "unsupported_accepted_max": 0,
    "unprovenanced_argument_accepted_max": 0,
}


class ProductGates51MTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.jobs = self.root / "jobs"
        self.pkg = self.root / "package"
        self.jobs.mkdir()
        self.pkg.mkdir()
        dump(self.pkg / "mei-model.json", {"package_id": "test-q4"})
        (self.pkg / "weights.q4").write_bytes(b"MEIQPK01payload")
        common = {
            "package_id": "test-q4",
            "prompt_token_ids": [2, 3],
            "generated_token_ids": [4, 5],
            "prefill_topk_ids": [7, 8, 9, 10, 11],
        }
        dump(
            self.jobs / "mlx-qat-q4-golden.json",
            {"package_id": "test-q4", "token_ids": [2, 3], "greedy_ids": [4, 5], "prefill_topk_ids": [7, 8, 9, 10, 11], "greedy_text": "ok"},
        )
        dump(self.jobs / "rust-q4-prefill.json", {"ok": True, "package_id": "test-q4"})
        dump(self.jobs / "rust-q4-short-greedy.json", {"ok": True, "max_abs_logit_vs_mlx": 0.1, **common})
        dump(self.jobs / "wasm-q4-smoke.json", {"ok": True, "loaded": True, "refuse_float": True, "complete_ran": True, "n_new": 8, **common})
        dump(self.jobs / "wasm-q4-browser-smoke.json", {"ok": True, "n_new": 8, **common})
        dump(self.jobs / "apple-q4-smoke.json", {"ok": True, "package_id": "test-q4"})

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_parity(self) -> int:
        argv = ["parity", "--jobs-dir", str(self.jobs), "--package-dir", str(self.pkg)]
        with patch.object(sys, "argv", argv):
            return parity.main()

    def test_parity_requires_exact_tokens_topk_and_prompt(self) -> None:
        self.assertEqual(self.run_parity(), 0)
        report = json.loads((self.jobs / "qat-q4-three-runtime-parity.json").read_text())
        self.assertTrue(report["ok"])
        row = json.loads((self.jobs / "wasm-q4-browser-smoke.json").read_text())
        row["generated_token_ids"] = [99]
        dump(self.jobs / "wasm-q4-browser-smoke.json", row)
        self.assertEqual(self.run_parity(), 2)

    def test_hard_gate_accepts_declared_q4_fallback(self) -> None:
        dump(self.jobs / "qat-q4-rung-0p5m.json", {"product_ok": True, "quality_ok": True, "rung": "0p5m"})
        dump(self.jobs / "qat-q4-package-receipt.json", {"ok": True, "within_budget": True, "within_raw_budget": True})
        dump(
            self.jobs / "qat-q4-three-runtime-parity.json",
            {
                "ok": True,
                "same_package_ok": True,
                "same_prompt_tokens_ok": True,
                "generated_token_ids_exact": True,
                "prefill_topk_ids_exact": True,
            },
        )
        dump(self.jobs / "qat-cq2-blocked.json", {"quality_blocked": True})
        dump(
            self.jobs / "sft-ondisk-qat-51m.json",
            {
                "fullcall_sft": {"steps": 1},
                "isolation": {"ok": True},
                "used_51m_weights": True,
                "n_loaded": 400,
                "weight_qat_ste": True,
                "activation_kv_int8_ste": True,
                "training_order": ["qat_fullcall_sft", "freeze_lm", "retrieval_head", "confidence_head"],
            },
        )
        dump(
            self.jobs / "sft-qat-q4-package-receipt.json",
            {"ok": True, "within_budget": True, "within_raw_budget": True},
        )
        dump(self.jobs / "toolcall-qat-lock-v2.json", {"hard_ok": True})
        dump(self.jobs / "resource-baseline-51m.json", {"hard_ok": True})
        report = hard.layer(self.jobs)
        self.assertTrue(report["hard_ok"])
        self.assertEqual(report["gates"]["mixed_q2q4"]["status"], "quality_blocked_qat_q4_only")

    def test_hard_gate_stops_on_lock_safety_failure(self) -> None:
        dump(self.jobs / "qat-q4-rung-0p5m.json", {"product_ok": True, "rung": "0p5m"})
        dump(
            self.jobs / "qat-q4-package-receipt.json",
            {"ok": True, "within_budget": True, "within_raw_budget": True},
        )
        dump(
            self.jobs / "qat-q4-three-runtime-parity.json",
            {
                "ok": True,
                "same_package_ok": True,
                "same_prompt_tokens_ok": True,
                "generated_token_ids_exact": True,
                "prefill_topk_ids_exact": True,
            },
        )
        dump(self.jobs / "qat-cq2-blocked.json", {"quality_blocked": True})
        dump(
            self.jobs / "sft-ondisk-qat-51m.json",
            {
                "fullcall_sft": {"steps": 1},
                "isolation": {"ok": True},
                "used_51m_weights": True,
                "n_loaded": 400,
                "weight_qat_ste": True,
                "activation_kv_int8_ste": True,
                "training_order": [
                    "qat_fullcall_sft",
                    "freeze_lm",
                    "retrieval_head",
                    "confidence_head",
                ],
            },
        )
        dump(
            self.jobs / "sft-qat-q4-package-receipt.json",
            {"ok": True, "within_budget": True, "within_raw_budget": True},
        )
        dump(self.jobs / "toolcall-qat-lock-v2.json", {"hard_ok": False})
        dump(self.jobs / "resource-baseline-51m.json", {"hard_ok": True})
        report = hard.layer(self.jobs)
        self.assertFalse(report["hard_ok"])
        self.assertFalse(report["gates"]["lock_v2"]["ok"])

    def test_lock_shards_default_to_serial_batches(self) -> None:
        self.assertEqual(lock.shard_batches([0, 1, 2, 3], 1), [[0], [1], [2], [3]])
        self.assertEqual(lock.shard_batches([0, 1, 2, 3], 4), [[0, 1, 2, 3]])
        with self.assertRaises(ValueError):
            lock.shard_batches([0], 0)

    def _passing_product_jobs(self, *, lock_ok: bool) -> None:
        dump(self.jobs / "qat-q4-rung-0p5m.json", {"product_ok": True, "quality_ok": True, "rung": "0p5m"})
        dump(
            self.jobs / "qat-q4-package-receipt.json",
            {"ok": True, "within_budget": True, "within_raw_budget": True},
        )
        dump(
            self.jobs / "qat-q4-three-runtime-parity.json",
            {
                "ok": True,
                "same_package_ok": True,
                "same_prompt_tokens_ok": True,
                "generated_token_ids_exact": True,
                "prefill_topk_ids_exact": True,
            },
        )
        dump(self.jobs / "qat-cq2-blocked.json", {"quality_blocked": True})
        dump(
            self.jobs / "sft-ondisk-qat-51m.json",
            {
                "fullcall_sft": {"steps": 1},
                "isolation": {"ok": True},
                "used_51m_weights": True,
                "n_loaded": 400,
                "weight_qat_ste": True,
                "activation_kv_int8_ste": True,
                "training_order": ["qat_fullcall_sft", "freeze_lm", "retrieval_head", "confidence_head"],
            },
        )
        dump(
            self.jobs / "sft-qat-q4-package-receipt.json",
            {"ok": True, "within_budget": True, "within_raw_budget": True},
        )
        dump(self.jobs / "toolcall-qat-lock-v2.json", {"hard_ok": lock_ok})
        dump(self.jobs / "resource-baseline-51m.json", {"hard_ok": True})

    def test_qat_keeps_best_passing_score_not_later_regression(self) -> None:
        self.assertTrue(qat_checkpoint_improved({}, 1.2, product_ok=True))
        self.assertFalse(qat_checkpoint_improved({"selection_score": 1.0}, 1.4, product_ok=True))
        self.assertTrue(qat_checkpoint_improved({"selection_score": 1.4}, 1.0, product_ok=True))
        self.assertFalse(qat_checkpoint_improved({}, 0.1, product_ok=False))

    def test_resource_paths_are_absolute_for_rust_wasm_cwd(self) -> None:
        pkg = self.root / "rel" / "pkg"
        jobs = self.root / "rel" / "jobs"
        pkg.mkdir(parents=True)
        jobs.mkdir(parents=True)
        got_pkg, got_jobs = resource.resolve_package_and_jobs(pkg, jobs)
        self.assertTrue(got_pkg.is_absolute())
        self.assertTrue(got_jobs.is_absolute())
        self.assertEqual(got_pkg, pkg.resolve())

    def test_lock_resume_skips_complete_shards_only(self) -> None:
        shard_dir = self.root / "shards"
        shard_dir.mkdir()
        fingerprint = "fp-1"
        complete = {
            "kind": "toolcall-qat-lock-v2",
            "input_fingerprint": fingerprint,
            "counts": {"n_retrieval": 10, "n_fullcall": 10},
            "layers": {"recall_at_5_learned": {"ok": False}},
        }
        dump(shard_dir / "shard-000.json", complete)
        dump(shard_dir / "shard-001.json", {"input_fingerprint": fingerprint})
        (shard_dir / "shard-002.json").write_text("{not-json", encoding="utf-8")
        pending = lock.pending_lock_shards(shard_dir, 4, fingerprint, resume=True)
        self.assertEqual(pending, [1, 2, 3])
        self.assertFalse(lock.shard_is_complete({"kind": "other"}, fingerprint))

    def test_lock_safety_failure_is_recorded_but_receipt_exit_is_zero(self) -> None:
        shard_dir = self.root / "lock-shards"
        shard_dir.mkdir()
        fingerprint = "safety-fp"
        counts = {
            "n_retrieval": 10,
            "learned_hits": 10,
            "lexical_hits": 1,
            "n_fullcall": 10,
            "oracle_exact": 10,
            "oracle_execute_refuse": 10,
            "learned_exact": 10,
            "unsupported_accepted": 1,
            "unprovenanced_argument_accepted": 0,
        }
        for index in range(4):
            dump(
                shard_dir / f"shard-{index:03d}.json",
                {
                    "kind": "toolcall-qat-lock-v2",
                    "input_fingerprint": fingerprint,
                    "counts": {key: (value if key != "n_retrieval" else 10) for key, value in counts.items()},
                    "layers": {},
                },
            )
            # Split counts across shards: aggregate sums them. Use zeros on 1-3.
            if index:
                dump(
                    shard_dir / f"shard-{index:03d}.json",
                    {
                        "kind": "toolcall-qat-lock-v2",
                        "input_fingerprint": fingerprint,
                        "counts": {key: 0 for key in counts},
                        "layers": {},
                    },
                )
        with patch.object(lock, "shard_fingerprint", return_value=fingerprint), patch.object(
            lock, "THRESHOLDS", self.root / "thresholds.json"
        ):
            dump(self.root / "thresholds.json", {"registered_before_eval": True})
            code = lock.aggregate_shards(shard_dir, 4, LOCK_LAYERS, self.pkg, self.jobs)
        self.assertEqual(code, 0)
        report = json.loads((self.jobs / "toolcall-qat-lock-v2.json").read_text())
        self.assertFalse(report["hard_ok"])
        self.assertFalse(report["layers"]["unsupported_accepted"]["ok"])
        self.assertTrue(report["evaluation_complete"])
        self.assertEqual(lock.lock_receipt_exit_code(evaluation_complete=True), 0)
        self.assertEqual(lock.lock_receipt_exit_code(evaluation_complete=False), 2)

    def test_lock_fail_writes_ineligible_freeze_proposal(self) -> None:
        self._passing_product_jobs(lock_ok=False)
        with patch.object(sys, "argv", ["hard", "--jobs-dir", str(self.jobs)]):
            code = hard.main()
        self.assertEqual(code, 2)
        proposal = json.loads((self.jobs / "freeze-proposal.json").read_text())
        self.assertFalse(proposal["eligible"])
        self.assertFalse(proposal["writes_current"])
        self.assertTrue(proposal["requires_explicit_user_freeze_command"])


if __name__ == "__main__":
    unittest.main()