from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import freeze_sft_v4_release_51m as freezer
import sft_v4_contract_51m as contract
import sft_v3_training_51m as training


class SftV4ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payloads, cls.manifest = freezer.build_payloads(freezer.parse_args([]))
        cls.coverage = json.loads(cls.payloads["coverage-receipt.json"])
        cls.isolation = json.loads(cls.payloads["isolation-receipt.json"])

    def test_deploy_and_training_universes_are_separate(self) -> None:
        deploy = json.loads(self.payloads["tool-universe.json"])
        training = json.loads(self.payloads["training-tool-universe.json"])
        self.assertEqual(len(deploy["tools"]), 147)
        self.assertEqual(len(training["tools"]), 211)
        self.assertEqual(training["training_only_schema_tool_count"], 64)
        self.assertEqual(
            training["final_product_index_policy"],
            "deploy tools only; training-only schema tools excluded",
        )
        self.assertEqual(self.manifest["tool_universes"]["final_index_policy"], "deploy universe only")

    def test_mw_contamination_is_rejected_and_replaced(self) -> None:
        self.assertEqual(self.coverage["mw_cleaning"]["train"]["rejected_synthetic_shortcut"], 800)
        self.assertEqual(self.coverage["mw_cleaning"]["valid"]["rejected_synthetic_shortcut"], 80)
        for split in ("train", "valid"):
            audit = self.coverage["mw_audits"][split]
            self.assertEqual(audit["status"], "passed", audit["errors"][:3])
            self.assertEqual(audit["synthetic_shortcut_rows"], 0)
            self.assertEqual(set(audit["class_counts"]), {str(value) for value in range(20)})
            rows = [
                json.loads(line)
                for line in self.payloads[f"mw-disposition.{split}.jsonl"].decode("utf-8").splitlines()
            ]
            self.assertTrue(
                all(
                    all(
                        key in row
                        for key in (
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

    def test_agent_validation_no_longer_reuses_eval_dev(self) -> None:
        self.assertEqual(self.coverage["agent_audits"]["train"]["filtered_eval_overlap_rows"], 0)
        self.assertEqual(self.coverage["agent_audits"]["valid"]["filtered_eval_overlap_rows"], 320)
        valid = self.coverage["agent_audits"]["valid"]
        self.assertEqual(valid["status"], "passed", valid["errors"][:3])
        self.assertEqual(valid["unique_call_tools"], 147)
        self.assertEqual(valid["unique_terminal_tools"], 147)

    def test_confidence_candidates_cover_all_banks_but_have_no_labels(self) -> None:
        expected = {
            "train": {"structural": 3528, "schema": 256, "linguistic": 1471},
            "valid": {"structural": 1176, "schema": 256, "linguistic": 588},
        }
        for split in ("train", "valid"):
            audit = self.coverage["confidence_candidates"][split]
            self.assertEqual(audit["labels_embedded"], 0)
            self.assertEqual(audit["source_banks"], expected[split])
            rows = [
                json.loads(line)
                for line in self.payloads[f"confidence-harvest.{split}.jsonl"].decode("utf-8").splitlines()
            ]
            self.assertTrue(all(row["label"] is None for row in rows))

    def test_isolation_and_token_budget_are_hard_passes(self) -> None:
        self.assertEqual(self.isolation["status"], "passed")
        self.assertEqual(self.isolation["train_valid_eval_query_overlap"], 0)
        self.assertEqual(self.isolation["train_valid_query_overlap"], 0)
        self.assertEqual(self.isolation["synthetic_shortcut_rows"], 0)
        self.assertEqual(self.coverage["token_budget"]["status"], "passed")
        self.assertLessEqual(
            self.coverage["token_budget"]["selected_schema_sink_tokens"]["max"],
            contract.STABLE_PREFIX_TOKENS_MAX,
        )

    def test_fullcall_and_float_control_budget_expose_every_training_row(self) -> None:
        schedule = json.loads(self.payloads["training-schedule.json"])
        stages = {row["stage"]: row for row in schedule["ordering"]}
        train_rows = sum(
            self.coverage["row_counts"][name]
            for name in (
                "full-call.train.jsonl",
                "schema-full-call.train.jsonl",
                "linguistic-full-call.train.jsonl",
            )
        )
        self.assertEqual(train_rows, 5255)
        self.assertGreaterEqual(stages["float_task_control"]["steps"], train_rows)
        self.assertGreaterEqual(
            stages["oracle_top5_quant_aware_fullcall"]["steps"], train_rows
        )
        self.assertEqual(stages["float_task_control"]["steps"], 7_200)
        self.assertEqual(
            stages["oracle_top5_quant_aware_fullcall"]["steps"], 7_200
        )

    def test_v4_bank_schedules_are_weighted_complete_and_deterministic(self) -> None:
        def rows(name: str, bank: str | None = None) -> list[dict]:
            values = [
                json.loads(line)
                for line in self.payloads[name].decode("utf-8").splitlines()
            ]
            for row in values:
                row["_training_bank"] = bank or (
                    "linguistic_natural"
                    if row.get("linguistic_layer")
                    == "historical_admitted_natural"
                    else "linguistic_offline"
                )
            return values

        fullcall = [
            *rows("full-call.train.jsonl", "structural"),
            *rows("schema-full-call.train.jsonl", "schema"),
            *rows("linguistic-full-call.train.jsonl"),
        ]
        fullcall_schedule = training.fullcall_training_schedule(fullcall, 7_200)
        self.assertEqual(
            fullcall_schedule,
            training.fullcall_training_schedule(fullcall, 7_200),
        )
        self.assertEqual(set(fullcall_schedule), set(range(len(fullcall))))
        self.assertEqual(
            Counter(fullcall[index]["_training_bank"] for index in fullcall_schedule),
            {
                "structural": 3_600,
                "schema": 1_080,
                "linguistic_natural": 1_260,
                "linguistic_offline": 1_260,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "cannot expose every bank row"):
            training.fullcall_training_schedule(fullcall, 6_000)

        retrieval = [
            *rows("retrieval.train.jsonl", "structural"),
            *rows("schema-retrieval.train.jsonl", "schema"),
            *rows("linguistic-retrieval.train.jsonl"),
        ]
        retrieval_schedule = training.retrieval_training_schedule(
            retrieval, 1_600, 8
        )
        flattened = [index for batch in retrieval_schedule for index in batch]
        self.assertEqual(set(flattened), set(range(len(retrieval))))
        self.assertTrue(
            all(
                len({retrieval[index]["gold_tool"] for index in batch}) == 8
                for batch in retrieval_schedule
            )
        )
        self.assertEqual(
            Counter(retrieval[index]["_training_bank"] for index in flattened),
            {
                "structural": 6_400,
                "schema": 1_920,
                "linguistic_natural": 2_240,
                "linguistic_offline": 2_240,
            },
        )

    def test_mw_prompt_consumes_structured_evidence_but_not_gold_label(self) -> None:
        rows = [
            json.loads(line)
            for line in self.payloads["mw-disposition.train.jsonl"].decode("utf-8").splitlines()
        ]
        row = next(item for item in rows if item.get("evidence"))
        universe = json.loads(self.payloads["tool-universe.json"])
        by_name = {tool["name"]: tool for tool in universe["tools"]}
        tools = [by_name[name] for name in row["retrieved_tools"]]
        first = training.render_mw_prompt_parts(row, tools)
        changed_evidence = dict(row)
        changed_evidence["evidence"] = [
            {"kind": "required_fact", "status": "verified", "fact": "replacement"}
        ]
        second = training.render_mw_prompt_parts(changed_evidence, tools)
        self.assertNotEqual(first["prompt"], second["prompt"])
        changed_gold = dict(row)
        changed_gold["reason_class_id"] = (int(row["reason_class_id"]) + 1) % 20
        changed_gold["reason_code"] = "forged_gold_must_stay_hidden"
        third = training.render_mw_prompt_parts(changed_gold, tools)
        self.assertEqual(first["prompt"], third["prompt"])
        self.assertEqual(first["prompt_id"], contract.MW_PROMPT_ID)

    def test_release_is_hash_complete_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args = freezer.parse_args(["--release-root", name])
            first = freezer.freeze(args)
            second = freezer.freeze(args)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            target = Path(name) / freezer.RELEASE_ID
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            for filename, spec in manifest["artifacts"].items():
                self.assertEqual(contract.sha_file(target / filename), spec["sha256"])


if __name__ == "__main__":
    unittest.main()
