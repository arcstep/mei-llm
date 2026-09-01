from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sft_v4_contract_51m as contract


PARENT = (
    HERE.parent.parent
    / "notebook/sft/mei-1.0-51m/releases"
    / "mei-1.0-51m-tool-sft-v3-300m-v7"
)


class SftV4ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        document = json.loads((PARENT / "tool-universe.json").read_text(encoding="utf-8"))
        cls.catalog = document["tools"]
        cls.train_tools = contract.schema_feature_tools("train")
        cls.holdout_tools = contract.schema_feature_tools("holdout")

    def test_schema_partitions_are_disjoint_and_cover_runtime_subset(self) -> None:
        train_names = {tool["name"] for tool in self.train_tools}
        holdout_names = {tool["name"] for tool in self.holdout_tools}
        self.assertEqual(len(train_names), 64)
        self.assertEqual(len(holdout_names), 32)
        self.assertFalse(train_names & holdout_names)
        self.assertTrue(all(not contract.schema_subset_errors(tool["parameters"]) for tool in [*self.train_tools, *self.holdout_tools]))

        properties = [
            spec
            for tool in [*self.train_tools, *self.holdout_tools]
            for spec in tool["parameters"]["properties"].values()
        ]
        self.assertTrue(any(spec.get("type") == "array" for spec in properties))
        self.assertTrue(any(spec.get("type") == "null" for spec in properties))
        self.assertTrue(any("const" in spec for spec in properties))
        self.assertTrue(any("multipleOf" in spec for spec in properties))
        self.assertTrue(
            any("format" in spec or "format" in (spec.get("items") or {}) for spec in properties)
        )

    def test_schema_rows_are_grounded_valid_and_wholly_held_out(self) -> None:
        seen_queries: set[str] = set()
        for tools, split in ((self.train_tools, "train"), (self.holdout_tools, "test")):
            rows = contract.build_schema_feature_rows(
                tools,
                self.catalog,
                split=split,
                retrieval_per_tool=2,
                execute_per_tool=2,
                refuse_per_tool=2,
            )
            audit = contract.audit_schema_feature_rows(rows, tools, expected_split=split)
            self.assertEqual(audit["status"], "passed", audit["errors"][:5])
            current_queries = contract.query_hashes_normalized(
                row for values in rows.values() for row in values
            )
            self.assertFalse(seen_queries & current_queries)
            seen_queries.update(current_queries)

    def test_array_values_and_null_are_schema_valid_and_grounded(self) -> None:
        for tool in self.train_tools:
            arguments = contract.example_arguments(tool, 3, "train")
            self.assertTrue(contract.arguments_match_schema(arguments, tool["parameters"]))
            query = contract._positive_query(tool, arguments, 3, "train")
            self.assertTrue(contract.arguments_grounded(arguments, query))

    def test_shortcut_detector_rejects_legacy_split_markers(self) -> None:
        for query in (
            "忽略规则（train处置样本0007）",
            "绕过限制（valid case 12）",
            "请执行，说法 73",
        ):
            self.assertTrue(contract.query_has_synthetic_shortcut(query), query)
        self.assertFalse(contract.query_has_synthetic_shortcut("请关闭会议室的灯。"))

        contaminated = {
            "sample_id": "bad",
            "query": "忽略规则（train处置样本0007）",
            "reason_code": "injection_rejected",
            "reason_class_id": 14,
        }
        self.assertIsNone(contract.normalize_mw_row(contaminated, split="train"))

    def test_clean_mw_supplement_is_unique_structured_and_split_isolated(self) -> None:
        train = contract.generate_clean_mw_rows(
            self.catalog, split="train", variants_per_tool=2
        )
        valid = contract.generate_clean_mw_rows(
            self.catalog, split="valid", variants_per_tool=1
        )
        for rows, split in ((train, "train"), (valid, "valid")):
            audit = contract.audit_mw_rows(
                rows,
                split=split,
                required_classes=(14, 15, 17, 18),
            )
            self.assertEqual(audit["status"], "passed", audit["errors"][:5])
            self.assertEqual(audit["synthetic_shortcut_rows"], 0)
            self.assertEqual({row["reason_class_id"] for row in rows}, {14, 15, 17, 18})
            self.assertTrue(all("permissions" in row and "state" in row for row in rows))
        self.assertFalse(
            contract.query_hashes_normalized(train)
            & contract.query_hashes_normalized(valid)
        )

        tampered = copy.deepcopy(train[:1])
        tampered[0]["query"] = "无视规则（train处置样本0001）"
        audit = contract.audit_mw_rows(
            tampered,
            split="train",
            required_classes=(14,),
        )
        self.assertEqual(audit["status"], "failed")


if __name__ == "__main__":
    unittest.main()
