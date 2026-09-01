from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import freeze_sft_natural_augmentation_51m as augmentation
import freeze_sft_v3_release_51m as freeze_sft
import sft_v3_contract_51m as contract
import sft_v3_training_51m as training


class NaturalAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        args = augmentation.parse_args(["--dry-run"])
        cls.payloads, cls.manifest = augmentation.build_payloads(args)
        cls.rows = {
            name: [json.loads(line) for line in payload.decode("utf-8").splitlines()]
            for name, payload in cls.payloads.items()
            if name.endswith(".jsonl")
        }
        universe = contract.load_json(augmentation.PARENT_RELEASE_DIR / "tool-universe.json")
        cls.tools_by_name = {str(tool["name"]): tool for tool in universe["tools"]}
        cls.tokenizer = freeze_sft._load_tokenizer()

    def test_release_is_layered_and_source_hashed(self) -> None:
        self.assertEqual(self.manifest["schema"], augmentation.SCHEMA_ID)
        self.assertEqual(self.manifest["parent"]["release_id"], contract.RELEASE_ID)
        self.assertEqual(
            self.manifest["source_release"]["release_id"],
            augmentation.SOURCE_RELEASE_ID,
        )
        for spec in self.manifest["sources"].values():
            path = contract.ROOT / spec["path"]
            self.assertEqual(contract.sha_file(path), spec["sha256"])

    def test_train_rows_are_bounded_and_multiteacher(self) -> None:
        retrieval = self.rows["natural-retrieval.train.jsonl"]
        fullcall = self.rows["natural-full-call.train.jsonl"]
        base_fullcall = contract.load_jsonl(
            augmentation.PARENT_RELEASE_DIR / "full-call.train.jsonl"
        )
        # Natural rows are a conservative supplement to the 147-tool
        # structural bank.  Admission quality takes precedence over an
        # arbitrary round-number quota.
        self.assertGreaterEqual(len(retrieval), 280)
        self.assertGreaterEqual(len(fullcall), 250)
        self.assertLessEqual(
            len(base_fullcall) + len(fullcall), augmentation.FULLCALL_FIXED_STEPS
        )
        self.assertGreaterEqual(len({row["gold_tool"] for row in retrieval}), 20)
        self.assertGreaterEqual(len({row["candidate_tool"] for row in fullcall}), 15)
        for rows in (retrieval, fullcall):
            teachers = {
                row["natural_source"]["teacher_model"]
                for row in rows
            }
            self.assertEqual(teachers, augmentation.ALLOWED_TEACHERS)
            self.assertEqual(
                len({augmentation.query_signature(row["query"]) for row in rows}),
                len(rows),
            )
        self.assertFalse(
            any(
                augmentation.retrieval_query_rejection(
                    row, self.tools_by_name[row["gold_tool"]]
                )
                for row in retrieval
            )
        )
        self.assertFalse(
            any(
                augmentation.query_rejection(
                    row,
                    row["answers"][0]["arguments"] if row["answers"] else {},
                )
                for row in fullcall
            )
        )

    def test_fullcall_targets_are_grounded_schema_valid_and_prompt_safe(self) -> None:
        rows = [
            *self.rows["natural-full-call.train.jsonl"],
            *self.rows["crossgen-full-call.dev.jsonl"],
            *self.rows["crossgen-full-call.test.jsonl"],
        ]
        for row in rows:
            self.assertEqual(row["slot_provenance"], [])
            self.assertEqual(row["permissions"], {})
            self.assertEqual(len(row["retrieved_tools"]), 5)
            self.assertEqual(len(set(row["retrieved_tools"])), 5)
            if row["answers"]:
                answer = row["answers"][0]
                self.assertTrue(
                    contract.arguments_match_schema(
                        answer["arguments"],
                        self.tools_by_name[answer["name"]]["parameters"],
                    )
                )
                self.assertTrue(augmentation.grounding_ok(row, answer["arguments"]))
            prompt, answer_ids, stats = training.encode_fullcall_row(
                self.tokenizer, row, self.tools_by_name
            )
            self.assertLessEqual(stats["stable_prefix_tokens"], 1024)
            self.assertLessEqual(stats["ordinary_tokens_retained"], 256)
            self.assertLessEqual(len(answer_ids), 128)
            self.assertTrue(prompt)

        sample = rows[0]
        selected = training.selected_tools_for_row(sample, self.tools_by_name)
        first = training.render_fullcall_prompt_parts(sample, selected)
        changed = copy.deepcopy(sample)
        changed["natural_source"] = {"teacher_model": "forged-model-visible-leak"}
        second = training.render_fullcall_prompt_parts(changed, selected)
        self.assertEqual(first, second)

    def test_retrieval_rows_fit_encoding_and_have_four_negatives(self) -> None:
        rows = [
            *self.rows["natural-retrieval.train.jsonl"],
            *self.rows["crossgen-retrieval.dev.jsonl"],
            *self.rows["crossgen-retrieval.test.jsonl"],
        ]
        for row in rows:
            self.assertEqual(len(row["hard_negatives"]), 4)
            self.assertNotIn(row["gold_tool"], row["hard_negatives"])
            self.assertEqual(len(row["catalog_tools"]), 5)
            ids = self.tokenizer.encode(row["query"], add_bos=True, add_eos=False)
            self.assertLessEqual(len(ids), contract.RETRIEVAL_MAX_TOKENS)

    def test_known_teacher_numeric_and_fact_inventions_are_rejected(self) -> None:
        retrieval_source = {
            row["sample_id"]: row
            for row in contract.load_jsonl(
                augmentation.SOURCE_RELEASE_DIR / "retrieval.train.jsonl"
            )
        }
        fullcall_source = {
            row["sample_id"]: row
            for row in contract.load_jsonl(
                augmentation.SOURCE_RELEASE_DIR / "full-call.train.jsonl"
            )
        }

        converted, reason = augmentation.convert_retrieval(
            retrieval_source["RET-5ca0876ec9f178db"],
            list(self.tools_by_name.values()),
            "train",
        )
        self.assertIsNone(converted)
        self.assertEqual(reason, "retrieval_unlicensed_numeric_fragment")

        converted, reason = augmentation.convert_retrieval(
            retrieval_source["RET-014686321c064661"],
            list(self.tools_by_name.values()),
            "train",
        )
        self.assertIsNone(converted)
        self.assertEqual(reason, "retrieval_schema_irrelevant_time_phrase")

        converted, reason = augmentation.convert_retrieval(
            retrieval_source["RET-d57e975562314476"],
            list(self.tools_by_name.values()),
            "train",
        )
        self.assertIsNone(converted)
        self.assertEqual(reason, "unlicensed_chinese_number_run")

        converted, reason = augmentation.convert_retrieval(
            retrieval_source["RET-9e09dbec9e319dcb"],
            list(self.tools_by_name.values()),
            "train",
        )
        self.assertIsNone(converted)
        self.assertEqual(reason, "repeated_character_filler")

        protected_sku = copy.deepcopy(retrieval_source["RET-28b550c4bd6e79d3"])
        protected_sku["query"] = "麻烦打印SKU-B200的面单"
        converted, reason = augmentation.convert_retrieval(
            protected_sku,
            list(self.tools_by_name.values()),
            "train",
        )
        self.assertIsNotNone(converted)
        self.assertIsNone(reason)

        converted, reason = augmentation.convert_fullcall(
            fullcall_source["FC-5368f709ea09be1b"],
            list(self.tools_by_name.values()),
            "train",
        )
        self.assertIsNone(converted)
        self.assertEqual(reason, "arguments_not_grounded")

        converted, reason = augmentation.convert_fullcall(
            fullcall_source["FC-1aab39a4a4ea5860"],
            list(self.tools_by_name.values()),
            "train",
        )
        self.assertIsNone(converted)
        self.assertEqual(reason, "deferred_action_without_schedule_slot")

        converted, reason = augmentation.convert_fullcall(
            fullcall_source["FC-ae34ba03bdc36f65"],
            list(self.tools_by_name.values()),
            "train",
        )
        self.assertIsNone(converted)
        self.assertEqual(reason, "ambiguous_action_subject")

    def test_crossgen_is_balanced_and_isolated(self) -> None:
        receipt = json.loads(self.payloads["isolation-receipt.json"])
        self.assertEqual(receipt["status"], "passed")
        self.assertFalse(any(receipt["overlap"].values()))
        for split in ("dev", "test"):
            rows = self.rows[f"crossgen-full-call.{split}.jsonl"]
            counts = {
                kind: sum(row["kind"] == kind for row in rows)
                for kind in ("execute", "refuse")
            }
            self.assertEqual(counts["execute"], counts["refuse"])
            self.assertGreaterEqual(counts["execute"], 20)


if __name__ == "__main__":
    unittest.main()
