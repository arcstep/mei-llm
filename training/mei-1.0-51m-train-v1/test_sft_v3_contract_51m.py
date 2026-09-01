from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import freeze_longitudinal_eval_51m as freeze_eval
import freeze_sft_v3_release_51m as freeze_sft
import sft_v3_contract_51m as contract


class SftV3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = contract.universe_tools()

    def test_generated_training_rows_cover_all_tools_and_are_schema_valid(self):
        rows = contract.build_single_step_rows(
            self.tools,
            split="train",
            retrieval_per_tool=contract.TRAIN_RETRIEVAL_PER_TOOL,
            execute_per_tool=contract.TRAIN_EXECUTE_PER_TOOL,
            refuse_per_tool=contract.TRAIN_REFUSE_PER_TOOL,
        )
        audit = contract.audit_single_step_rows(
            rows,
            self.tools,
            expected_split="train",
            min_retrieval_per_tool=contract.TRAIN_RETRIEVAL_PER_TOOL,
            min_execute_per_tool=contract.TRAIN_EXECUTE_PER_TOOL,
            min_refuse_per_tool=contract.TRAIN_REFUSE_PER_TOOL,
        )
        self.assertEqual(audit["status"], "passed", audit["errors"][:3])
        self.assertEqual(audit["retrieval_unique_gold_tools"], 147)
        self.assertEqual(audit["fullcall_unique_execute_tools"], 147)
        self.assertEqual(audit["fullcall_unique_refusal_candidates"], 147)
        self.assertEqual(audit["retrieval_rows"], 147 * 16)
        self.assertEqual(audit["execute_rows"], 147 * 12)
        self.assertEqual(audit["refuse_rows"], 147 * 12)
        self.assertEqual(audit["negative_distinct_min_per_tool"], 27)
        self.assertEqual(audit["gold_slot_provenance_rows"], 0)
        self.assertEqual(audit["unrestricted_permission_rows"], 147 * 24)
        self.assertEqual(audit["configurable_empty_execute_rows"], 0)
        self.assertEqual(audit["inapplicable_refusal_reason_rows"], 0)
        self.assertEqual(audit["mechanical_query_defect_rows"], 0)
        self.assertEqual(audit["incomplete_counterfactual_groups"], 0)
        self.assertEqual(audit["counterfactual_pair_groups"], 147 * 12)
        self.assertEqual(
            set(audit["refusal_reasons"]),
            {
                "negation_cancels",
                "missing_slot",
                "ambiguous_scope",
                "unknown_slot_value",
                "mixed_intent",
                "offtopic",
            },
        )
        for row in rows["fullcall"]:
            self.assertEqual(row["permissions"], {})
            self.assertEqual(row["slot_provenance"], [])
            self.assertEqual(
                row["target_text"], contract.serialize_tool_target(row["answers"])
            )
            if row["kind"] == "execute":
                schema = next(
                    tool["parameters"]
                    for tool in self.tools
                    if tool["name"] == row["candidate_tool"]
                )
                if schema.get("properties"):
                    self.assertTrue(row["gold_args"])

    def test_portable_projection_and_exact_target_order(self):
        document, receipt = contract.portable_universe_document()
        patterns = {
            spec["pattern"]
            for tool in document["tools"]
            for spec in (tool.get("parameters") or {}).get("properties", {}).values()
            if "pattern" in spec
        }
        self.assertEqual(patterns, {r"^[0-9]{2}:[0-9]{2}$"})
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["pattern_rewrites"][0]["count"], 2)
        target = contract.serialize_tool_target(
            [{"arguments": {"z": 1, "a": 2}, "name": "demo"}]
        )
        self.assertEqual(
            target, '[{"name":"demo","arguments":{"a":2,"z":1}}]'
        )

    def test_exact_token_budgets_cover_all_training_variants(self):
        rows = contract.build_single_step_rows(
            self.tools,
            split="train",
            retrieval_per_tool=contract.TRAIN_RETRIEVAL_PER_TOOL,
            execute_per_tool=contract.TRAIN_EXECUTE_PER_TOOL,
            refuse_per_tool=contract.TRAIN_REFUSE_PER_TOOL,
        )
        tokenizer = freeze_sft._load_tokenizer()
        audit = contract.token_budget_audit(rows, self.tools, tokenizer)
        self.assertEqual(audit["status"], "passed", audit["errors"])
        self.assertLessEqual(
            audit["selected_schema_sink_tokens"]["max"],
            contract.STABLE_PREFIX_TOKENS_MAX,
        )
        self.assertLessEqual(
            audit["retrieval_tool_tokens"]["max"],
            contract.RETRIEVAL_MAX_TOKENS,
        )

    def test_train_valid_dev_test_queries_are_disjoint(self):
        seen: set[str] = set()
        specs = (
            ("train", 16, 12, 12),
            ("valid", 4, 4, 4),
            ("dev", 4, 2, 2),
            ("test", 4, 2, 2),
        )
        for split, retrieval_n, execute_n, refuse_n in specs:
            rows = contract.build_single_step_rows(
                self.tools,
                split=split,
                retrieval_per_tool=retrieval_n,
                execute_per_tool=execute_n,
                refuse_per_tool=refuse_n,
            )
            audit = contract.audit_single_step_rows(
                rows,
                self.tools,
                expected_split=split,
                min_retrieval_per_tool=retrieval_n,
                min_execute_per_tool=execute_n,
                min_refuse_per_tool=refuse_n,
                forbidden_query_hashes=seen,
            )
            self.assertEqual(audit["status"], "passed", audit["errors"][:3])
            seen.update(audit["query_hashes"])

    def test_agent_terminal_supplement_covers_every_tool_without_fake_chains(self):
        rows = contract.build_agent_terminal_rows(
            self.tools,
            split="train",
            variants_per_tool=contract.TRAIN_AGENT_TERMINAL_PER_TOOL,
        )
        audit = contract.audit_agent_rows(
            rows,
            self.tools,
            expected_split="train",
            minimum_unique_call_tools=147,
            minimum_terminal_tools=147,
        )
        self.assertEqual(audit["status"], "passed", audit["errors"][:3])
        self.assertEqual(audit["rows"], 147 * 4 * 2)
        self.assertEqual(audit["trajectories"], 147 * 4)
        self.assertEqual(audit["unique_call_tools"], 147)
        self.assertEqual(audit["unique_terminal_tools"], 147)
        self.assertTrue(all(row["trajectory_length"] == 2 for row in rows))

        tampered = copy.deepcopy(rows)
        terminal = next(row for row in tampered if row.get("tool_results"))
        terminal["tool_results"][0]["call_id"] = "call-tampered"
        failed = contract.audit_agent_rows(
            tampered,
            self.tools,
            expected_split="train",
            minimum_unique_call_tools=147,
            minimum_terminal_tools=147,
        )
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(
            any("invalid trusted ToolResultV2" in error for error in failed["errors"])
        )

    def test_longitudinal_lock_is_hash_complete_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_name:
            args = type(
                "Args",
                (),
                {
                    "eval_root": Path(temp_name),
                    "eval_id": contract.EVAL_ID,
                    "universe": contract.TOOL_UNIVERSE_PATH,
                    "historical_eval": contract.HISTORICAL_EVAL_DIR,
                    "agent_release": contract.PARENT_RELEASE_DIR,
                    "narration_release": contract.NARRATION_RELEASE_DIR,
                },
            )()
            first = freeze_eval.freeze(args)
            second = freeze_eval.freeze(args)
            self.assertTrue(first["ok"])
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            target = Path(temp_name) / contract.EVAL_ID
            lock = json.loads((target / "lock.json").read_text(encoding="utf-8"))
            isolation = json.loads(
                (target / "isolation-receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(lock["status"], "frozen")
            self.assertEqual(lock["schema"], "mei-51m-longitudinal-eval-lock-v3")
            self.assertEqual(isolation["status"], "passed")
            self.assertEqual(isolation["historical_train_eval_query_overlap"], 0)
            self.assertEqual(lock["artifacts"]["mw.dev.jsonl"]["rows"], 1000)
            self.assertEqual(lock["artifacts"]["mw.test.jsonl"]["rows"], 1000)
            self.assertEqual(lock["artifacts"]["confidence.dev.jsonl"]["rows"], 588)
            self.assertEqual(lock["artifacts"]["confidence.test.jsonl"]["rows"], 588)
            historical_dev = len(
                contract.load_jsonl(
                    contract.PARENT_RELEASE_DIR / "agent-continuation.valid.jsonl"
                )
            )
            historical_test = len(
                contract.load_jsonl(
                    contract.PARENT_RELEASE_DIR / "agent-continuation.eval.jsonl"
                )
            )
            self.assertEqual(
                lock["artifacts"]["multistep.dev.jsonl"]["rows"],
                historical_dev + 147 * 2,
            )
            self.assertEqual(
                lock["artifacts"]["multistep.test.jsonl"]["rows"],
                historical_test + 147 * 2,
            )
            self.assertEqual(isolation["multistep_unique_call_tools"]["dev"], 147)
            self.assertEqual(isolation["multistep_unique_call_tools"]["test"], 147)
            mw_rows = contract.load_jsonl(target / "mw.test.jsonl")
            self.assertEqual({row["reason_class_id"] for row in mw_rows}, set(range(20)))
            self.assertTrue(all(len(row["oracle_top5"]) == 5 for row in mw_rows))
            universe = json.loads(
                (target / "tool-universe.json").read_text(encoding="utf-8")
            )
            self.assertEqual(universe["universe_id"], "mei-51m-portable-tool-universe-v2")
            budget = json.loads(
                (target / "token-budget-receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(budget["status"], "passed")

    def test_sft_release_is_balanced_hash_complete_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            eval_args = type(
                "EvalArgs",
                (),
                {
                    "eval_root": root / "eval",
                    "eval_id": contract.EVAL_ID,
                    "universe": contract.TOOL_UNIVERSE_PATH,
                    "historical_eval": contract.HISTORICAL_EVAL_DIR,
                    "agent_release": contract.PARENT_RELEASE_DIR,
                    "narration_release": contract.NARRATION_RELEASE_DIR,
                },
            )()
            freeze_eval.freeze(eval_args)
            args = type(
                "Args",
                (),
                {
                    "release_root": root / "releases",
                    "release_id": contract.RELEASE_ID,
                    "parent_release": contract.PARENT_RELEASE_DIR,
                    "eval_dir": eval_args.eval_root / contract.EVAL_ID,
                    "universe": contract.TOOL_UNIVERSE_PATH,
                    "narration_release": contract.NARRATION_RELEASE_DIR,
                    "base_release": freeze_sft.BASE_RELEASE_PATH,
                },
            )()
            first = freeze_sft.freeze(args)
            second = freeze_sft.freeze(args)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            target = args.release_root / contract.RELEASE_ID
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            coverage = json.loads(
                (target / "coverage-receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "frozen")
            self.assertEqual(manifest["base_compatibility"]["exposure_allowlist"], None)
            self.assertEqual(coverage["historical_v2"]["retrieval_unique_gold_tools"], 25)
            self.assertEqual(
                coverage["v3"]["train"]["retrieval_unique_gold_tools"], 147
            )
            self.assertEqual(
                coverage["v3"]["train"]["fullcall_unique_execute_tools"], 147
            )
            self.assertEqual(
                coverage["v3"]["mw_generated_train_unique_candidate_tools"], 147
            )
            self.assertEqual(
                coverage["v3"]["agent_input_audits"]["train"]["status"], "passed"
            )
            self.assertEqual(coverage["v3"]["agent_train_unique_call_tools"], 147)
            self.assertEqual(coverage["v3"]["agent_train_unique_terminal_tools"], 147)
            self.assertEqual(
                coverage["v3"]["agent_train_rows"],
                len(
                    contract.load_jsonl(
                        contract.PARENT_RELEASE_DIR / "agent-continuation.train.jsonl"
                    )
                )
                + 147 * contract.TRAIN_AGENT_TERMINAL_PER_TOOL * 2,
            )
            self.assertEqual(
                coverage["v3"]["train"]["negative_distinct_min_per_tool"], 27
            )
            mw_row = json.loads(
                next(
                    line
                    for line in (target / "mw-disposition.train.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line
                )
            )
            self.assertNotIn("teacher_model", mw_row)
            self.assertNotIn("prompt_text", mw_row)
            confidence = json.loads(
                next(
                    line
                    for line in (target / "confidence-harvest.train.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line
                )
            )
            self.assertIsNone(confidence["label"])
            self.assertEqual(confidence["permissions"], {})
            qat = json.loads(
                (target / "qat-import-candidate-receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(qat["status"], "passed")


if __name__ == "__main__":
    unittest.main()
