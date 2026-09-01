from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import freeze_sft_linguistic_augmentation_v2_51m as freezer
import sft_v4_contract_51m as contract


class SftV4LinguisticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payloads, cls.manifest = freezer.build_payloads(freezer.parse_args([]))
        cls.quality = json.loads(cls.payloads["quality-audit-receipt.json"])
        cls.budget = json.loads(cls.payloads["token-budget-receipt.json"])

    def test_release_separates_natural_and_offline_claims(self) -> None:
        self.assertEqual(
            self.manifest["schema"], "mei-sft-linguistic-augmentation-v2"
        )
        self.assertTrue(
            self.manifest["claims"]["all_147_tools_have_offline_linguistic_coverage"]
        )
        self.assertFalse(
            self.manifest["claims"]["offline_rows_are_natural_user_data"]
        )
        self.assertEqual(self.quality["offline_train_tool_coverage"], 147)
        self.assertFalse(self.quality["offline_is_natural_user_data"])

    def test_rows_cover_every_tool_without_split_or_eval_leakage(self) -> None:
        self.assertEqual(self.quality["status"], "passed", self.quality["errors"][:3])
        self.assertEqual(self.quality["train_valid_query_overlap"], 0)
        self.assertEqual(self.quality["eval_query_overlap"], 0)
        for name, stats in self.quality["tasks"].items():
            self.assertEqual(stats["tools"], 147, name)
            self.assertEqual(stats["query_eval_overlap"], 0, name)
            self.assertEqual(stats["synthetic_shortcut_rows"], 0, name)
        train_fullcall = self.quality["tasks"]["linguistic-full-call.train.jsonl"]
        self.assertGreater(train_fullcall["execute"], 0)
        self.assertGreater(train_fullcall["refuse"], 0)
        self.assertEqual(
            set(train_fullcall["source_roles"]),
            {
                "admitted-multiteacher-natural-query",
                "offline-linguistic-program-not-natural-user",
            },
        )

    def test_top5_and_token_budgets_are_deployable(self) -> None:
        self.assertEqual(self.budget["status"], "passed", self.budget["errors"])
        self.assertLessEqual(
            self.budget["selected_schema_sink_tokens"]["max"],
            contract.STABLE_PREFIX_TOKENS_MAX,
        )
        self.assertLessEqual(
            self.budget["retrieval_tool_tokens"]["max"],
            contract.RETRIEVAL_MAX_TOKENS,
        )

    def test_freeze_is_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args = freezer.parse_args(["--release-root", name])
            first = freezer.freeze(args)
            second = freezer.freeze(args)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            target = Path(name) / contract.LINGUISTIC_AUGMENTATION_ID
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            for filename, spec in manifest["artifacts"].items():
                self.assertEqual(contract.sha_file(target / filename), spec["sha256"])


if __name__ == "__main__":
    unittest.main()
