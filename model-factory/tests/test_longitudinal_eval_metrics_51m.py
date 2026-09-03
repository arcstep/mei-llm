from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluation.tool_use.longitudinal_eval_metrics_51m as metrics
import contracts.sft_v4_contract_51m as contract
import evaluation.tool_use.sft_v3_eval_51m as runtime_eval


class LongitudinalMetricTests(unittest.TestCase):
    def test_retrieval_metrics_include_rank_and_lexical_delta(self):
        gold = [
            {"sample_id": "a", "gold_tool": "a", "family": "x"},
            {"sample_id": "b", "gold_tool": "b", "family": "x"},
        ]
        predictions = [
            {
                "sample_id": "a",
                "ranked_tools": ["a", "b", "c", "d", "e"],
                "lexical_ranked_tools": ["b", "c", "d", "e", "f"],
            },
            {
                "sample_id": "b",
                "ranked_tools": ["a", "c", "b", "d", "e"],
                "lexical_ranked_tools": ["b", "a", "c", "d", "e"],
            },
        ]
        score = metrics.retrieval_metrics(gold, predictions)
        self.assertEqual(score["recall_at_5"], 1.0)
        self.assertEqual(score["recall_at_1"], 0.5)
        self.assertEqual(score["lexical_recall_at_5"], 0.5)
        self.assertEqual(score["learned_minus_lexical_recall_at_5"], 0.5)

    def test_fullcall_separates_execute_and_refuse(self):
        tools = [
            {
                "name": "set_flag",
                "parameters": {
                    "type": "object",
                    "properties": {"on": {"type": "boolean"}},
                    "required": ["on"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "stop",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        ]
        gold = [
            {
                "sample_id": "e",
                "kind": "execute",
                "gold_name": "set_flag",
                "gold_args": {"on": True},
                "retrieved_tools": ["set_flag"],
            },
            {
                "sample_id": "r",
                "kind": "refuse",
                "reason_code": "missing_slot",
                "retrieved_tools": ["set_flag"],
            },
        ]
        predictions = [
            {
                "sample_id": "e",
                "result": {
                    "kind": "call",
                    "call": {"name": "set_flag", "arguments": {"on": True}},
                },
            },
            {"sample_id": "r", "result": {"kind": "refuse"}},
        ]
        score = metrics.fullcall_metrics(gold, predictions, tools)
        self.assertEqual(score["execute_tool_name_exact"], 1.0)
        self.assertEqual(score["execute_arguments_exact"], 1.0)
        self.assertEqual(score["refusal_accuracy"], 1.0)
        self.assertEqual(score["balanced_accuracy"], 1.0)

    def test_confidence_rejects_single_class_and_scores_perfect_order(self):
        rows = [
            {"label": 0, "score": 0.1},
            {"label": 0, "score": 0.2},
            {"label": 1, "score": 0.8},
            {"label": 1, "score": 0.9},
        ]
        score = metrics.confidence_metrics(rows)
        self.assertEqual(score["positive_n"], 2)
        self.assertEqual(score["negative_n"], 2)
        self.assertEqual(score["auroc"], 1.0)
        self.assertEqual(score["auprc"], 1.0)
        with self.assertRaisesRegex(RuntimeError, "single-class"):
            metrics.confidence_metrics([{"label": 0, "score": 0.1}])
        with self.assertRaisesRegex(RuntimeError, "minimum class coverage"):
            metrics.confidence_metrics(rows, minimum_class_rows=3)

    def test_confidence_requires_exact_frozen_candidate_identity(self):
        candidates = [
            {
                "sample_id": "c0",
                "source_sample_id": "s0",
                "expected_kind": "refuse",
                "candidate_tool": "stop",
            },
            {
                "sample_id": "c1",
                "source_sample_id": "s1",
                "expected_kind": "call",
                "candidate_tool": "set_flag",
            },
        ]
        predictions = [
            {
                **candidate,
                "label": index,
                "score": 0.1 + 0.8 * index,
            }
            for index, candidate in enumerate(candidates)
        ]
        score = metrics.confidence_outcome_metrics(candidates, predictions)
        self.assertTrue(score["candidate_identity_verified"])
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            metrics.confidence_outcome_metrics(candidates, predictions[:-1])
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            metrics.confidence_outcome_metrics(
                candidates,
                predictions + [{**predictions[0], "sample_id": "extra"}],
            )
        mismatched = [dict(row) for row in predictions]
        mismatched[0]["expected_kind"] = "call"
        with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
            metrics.confidence_outcome_metrics(candidates, mismatched)

    def test_classification_reports_macro_and_false_continue(self):
        score = metrics.classification_metrics(
            [0, 1, 2, 2], [0, 0, 2, 1], n_classes=3
        )
        self.assertEqual(score["n"], 4)
        self.assertAlmostEqual(score["accuracy"], 0.5)
        self.assertAlmostEqual(score["class_0_false_continue_rate"], 1 / 3)
        self.assertEqual(len(score["confusion"]), 3)

    def test_eval_v7_catalog_and_threshold_contract(self):
        lock = contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID
        tools = runtime_eval.catalog_from_lock(lock)
        self.assertEqual(len(tools), 147)
        ranked = runtime_eval.lexical_top5("请查询上海天气", tools)
        self.assertEqual(len(ranked), 5)
        status = runtime_eval.threshold_status(
            {"accuracy": 0.8, "error_rate": 0.1},
            {"accuracy": 0.7, "error_rate_max": 0.2},
        )
        self.assertTrue(status["passed"])


if __name__ == "__main__":
    unittest.main()
