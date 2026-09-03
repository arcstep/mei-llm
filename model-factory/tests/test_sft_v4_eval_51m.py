from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluation.tool_use.freeze_longitudinal_eval_v7_51m as freezer
import contracts.sft_v4_contract_51m as contract


class SftV4EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.args = freezer.parse_args([])
        cls.payloads, cls.lock = freezer.build_payloads(cls.args)
        cls.isolation = json.loads(cls.payloads["isolation-receipt.json"])

    def test_lock_has_three_separate_quality_banks(self) -> None:
        self.assertEqual(self.lock["schema"], "mei-51m-longitudinal-eval-lock-v4")
        self.assertEqual(self.lock["id"], contract.EVAL_ID)
        self.assertEqual(self.lock["status"], "frozen")
        self.assertEqual(
            set(self.lock["banks"]),
            {
                "structural_seen",
                "natural_cross_generator",
                "whole_schema_holdout",
                "clean_structured_mw",
                "agent",
                "narration",
                "base_language",
            },
        )

    def test_whole_schema_holdout_is_disjoint_and_prefix_safe(self) -> None:
        self.assertEqual(self.isolation["schema_train_tool_count"], 64)
        self.assertEqual(self.isolation["schema_holdout_tool_count"], 32)
        self.assertEqual(self.isolation["schema_train_holdout_tool_overlap"], 0)
        self.assertEqual(self.isolation["schema_token_budget"]["status"], "passed")
        self.assertLessEqual(
            self.isolation["schema_token_budget"]["selected_schema_sink_tokens"]["max"],
            contract.STABLE_PREFIX_TOKENS_MAX,
        )
        for split in ("dev", "test"):
            audit = self.isolation["schema_audits"][split]
            self.assertEqual(audit["status"], "passed", audit["errors"][:3])
            self.assertEqual(audit["tools"], 32)
            self.assertGreater(audit["array_tools"], 0)
            self.assertGreater(audit["const_tools"], 0)
            self.assertGreater(audit["format_tools"], 0)
            self.assertGreater(audit["multiple_of_tools"], 0)
            self.assertGreater(audit["null_tools"], 0)

    def test_mw_bank_is_clean_structured_balanced_and_isolated(self) -> None:
        self.assertEqual(self.isolation["mw_synthetic_shortcut_rows"], 0)
        self.assertEqual(self.isolation["known_train_eval_query_overlap"], 0)
        for split in ("dev", "test"):
            audit = self.isolation["mw_audits"][split]
            self.assertEqual(audit["status"], "passed", audit["errors"][:3])
            self.assertEqual(audit["rows"], 1000)
            self.assertEqual(set(audit["class_counts"]), {str(value) for value in range(20)})
            self.assertEqual(set(audit["class_counts"].values()), {50})
            rows = [
                json.loads(line)
                for line in self.payloads[f"mw.{split}.jsonl"].decode("utf-8").splitlines()
            ]
            self.assertTrue(
                all(
                    all(
                        field in row
                        for field in (
                            "context",
                            "evidence",
                            "permissions",
                            "state",
                            "history",
                            "tool_results",
                        )
                    )
                    for row in rows
                )
            )

    def test_natural_crossgen_is_admitted_and_global_overlap_is_removed(self) -> None:
        exclusions = {
            name: audit["historical_training_overlap_excluded"]
            for name, audit in self.isolation["natural_audits"].items()
        }
        self.assertEqual(sum(exclusions.values()), 1)
        for audit in self.isolation["natural_audits"].values():
            self.assertEqual(audit["status"], "passed", audit["errors"][:3])
            self.assertEqual(audit["synthetic_shortcut_rows"], 0)
            self.assertGreater(audit["rows"], 0)

    def test_freeze_is_atomic_hash_complete_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args = freezer.parse_args(["--eval-root", name])
            first = freezer.freeze(args)
            second = freezer.freeze(args)
            self.assertTrue(first["ok"])
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            target = Path(name) / contract.EVAL_ID
            lock = json.loads((target / "lock.json").read_text(encoding="utf-8"))
            for filename, spec in lock["artifacts"].items():
                self.assertEqual(contract.sha_file(target / filename), spec["sha256"])


if __name__ == "__main__":
    unittest.main()
