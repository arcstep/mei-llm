from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import contracts.sft_v3_contract_51m as contract
import evaluation.tool_use.sft_v3_eval_51m as evaluation
import training.tool_use.sft_v3_training_51m as training


class _FallbackNarrationRuntime:
    narration_adapter = object()

    @staticmethod
    def generate_narration(prompt: str, *, max_new: int):
        del prompt, max_new
        return {"text": "不受信任的自由生成", "n_out": 8}


class _OversizedTokenizer:
    eos_id = 2

    @staticmethod
    def encode(text: str, *, add_bos: bool, add_eos: bool):
        del text, add_bos, add_eos
        return list(range(evaluation.contract.STABLE_PREFIX_TOKENS_MAX + 1))


class _BudgetFailRuntime:
    tokenizer = _OversizedTokenizer()
    mw_disposition_head = object()

    @staticmethod
    def search_top_k(query: str, catalog, *, k: int):
        del query
        return list(catalog[:k])

    @staticmethod
    def greedy(*args, **kwargs):
        del args, kwargs
        raise AssertionError("deterministic budget rejection must precede decode")

    @staticmethod
    def model(*args, **kwargs):
        del args, kwargs
        raise AssertionError("deterministic budget rejection must precede heads")


class _TinyTokenizer:
    eos_id = 2

    @staticmethod
    def encode(text: str, *, add_bos: bool, add_eos: bool):
        del text, add_bos, add_eos
        return [1]


class _ScriptedRuntime:
    tokenizer = _TinyTokenizer()

    def __init__(self, outputs):
        self.outputs = list(outputs)

    @staticmethod
    def search_top_k(query: str, catalog, *, k: int):
        del query
        return list(catalog[:k])

    def greedy(self, *args, **kwargs):
        del args, kwargs
        text = self.outputs.pop(0)
        return {"text": text, "n_out": 1, "decode_ms": 1.0, "logprob_sum": -0.1}


def _five_tools():
    return [
        {
            "name": f"tool_{index}",
            "description": f"tool {index}",
            "parameters": {"type": "object", "properties": {}},
        }
        for index in range(5)
    ]


def _five_tools_with_grounded_argument():
    tools = _five_tools()
    tools[0]["parameters"] = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    return tools


class SftV3EvaluationTests(unittest.TestCase):
    def test_all_v7_fullcall_rows_project_to_strict_request_v2(self):
        evaluation._ensure_sdk_path()
        from mei_sdk.protocol import normalize_request

        for name in (
            "fullcall.dev.jsonl",
            "fullcall.test.jsonl",
            "natural-fullcall.dev.jsonl",
            "natural-fullcall.test.jsonl",
            "schema-fullcall.dev.jsonl",
            "schema-fullcall.test.jsonl",
        ):
            with self.subTest(bank=name):
                rows = evaluation.contract.load_jsonl(
                    evaluation.contract.DEFAULT_EVAL_ROOT
                    / evaluation.contract.EVAL_ID
                    / name
                )
                for row in rows:
                    request = normalize_request(training.deployment_request_v3(row))
                    self.assertEqual(request["wire_version"], evaluation.contract.WIRE_ID)
                    self.assertTrue(
                        all(isinstance(fact, dict) for fact in request["context"].get("facts", []))
                    )

    def test_control_slice_is_balanced_across_all_tools(self):
        rows = contract.load_jsonl(
            contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID / "fullcall.dev.jsonl"
        )
        selected = evaluation.balanced_fullcall_sample(rows)
        self.assertEqual(len(selected), 147 * 2)
        self.assertEqual(
            {(row["candidate_tool"], row["kind"]) for row in selected},
            {
                (tool, kind)
                for tool in {row["candidate_tool"] for row in rows}
                for kind in ("execute", "refuse")
            },
        )

    def test_evaluation_call_ids_are_stable_unique_and_wire_valid(self):
        call = {"name": "get_weather", "arguments": {"city": "上海"}}
        first = evaluation._evaluation_call_id("trajectory-a", 1, call)
        again = evaluation._evaluation_call_id("trajectory-a", 1, call)
        second = evaluation._evaluation_call_id("trajectory-a", 2, call)
        self.assertEqual(first, again)
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^call-s[0-9a-f]{8}-[1-8]-[0-9a-f]{12}$")

    def test_fixture_result_is_rebound_to_generated_call(self):
        rows = [
            {
                "prior_tool_results": [
                    {
                        "wire_version": contract.WIRE_ID,
                        "call_id": "old-call",
                        "status": "ok",
                        "payload": {"pnr": "PNR-1"},
                        "provenance": {"source": "old", "verified": True},
                    }
                ]
            }
        ]
        rebound = evaluation._trajectory_fixture_result(rows, 0, "call-s00000001-1-000000000001")
        self.assertIsNotNone(rebound)
        self.assertEqual(rebound["call_id"], "call-s00000001-1-000000000001")
        self.assertTrue(rebound["provenance"]["verified"])
        self.assertIn("frozen-eval-host-simulator", rebound["provenance"]["source"])

    def test_fullcall_budget_rejection_is_scored_as_error_without_decode(self):
        catalog = _five_tools()
        rows = [
            {
                "sample_id": "execute-over-budget",
                "query": "执行",
                "kind": "execute",
                "gold_name": "tool_0",
                "gold_args": {},
            },
            {
                "sample_id": "refuse-over-budget",
                "query": "拒绝",
                "kind": "refuse",
                "reason_code": "permission_denied",
            },
        ]
        report, predictions = evaluation.evaluate_fullcall(
            _BudgetFailRuntime(), rows, catalog, retrieval_mode="learned_top5"
        )
        self.assertEqual(report["tool_schema_budget_error_count"], 2)
        self.assertEqual(report["error_rate"], 1.0)
        self.assertTrue(
            all(row["result"]["error"] == "tool_schema_budget_exceeded" for row in predictions)
        )

    def test_fullcall_model_quality_is_not_conflated_with_missing_host_evidence(self):
        catalog = _five_tools_with_grounded_argument()
        runtime = _ScriptedRuntime(
            ['[{"name":"tool_0","arguments":{"value":7}}]', "[]"]
        )
        rows = [
            {
                "sample_id": "exact-call-without-host-evidence",
                "query": "把值设为七",
                "kind": "execute",
                "gold_name": "tool_0",
                "gold_args": {"value": 7},
                "answers": [{"name": "tool_0", "arguments": {"value": 7}}],
                "oracle_top5": catalog,
            },
            {
                "sample_id": "correct-refusal",
                "query": "不要执行",
                "kind": "refuse",
                "reason_code": "negation_cancels",
                "oracle_top5": catalog,
            },
        ]
        report, predictions = evaluation.evaluate_fullcall(
            runtime, rows, catalog, retrieval_mode="oracle_top5"
        )
        self.assertEqual(report["balanced_accuracy"], 1.0)
        self.assertEqual(report["execute_arguments_exact"], 1.0)
        self.assertEqual(predictions[0]["result"]["kind"], "call")
        self.assertEqual(predictions[0]["pipeline_result"]["kind"], "error")
        self.assertEqual(
            predictions[0]["pipeline_validation_error"], "provenance_missing"
        )
        self.assertEqual(
            report["pipeline_validation_error_counts"], {"provenance_missing": 1}
        )

    def test_confidence_label_uses_actual_model_correctness_not_a_later_gate(self):
        catalog = _five_tools_with_grounded_argument()
        runtime = _ScriptedRuntime(
            ['[{"name":"tool_0","arguments":{"value":7}}]']
        )
        outcomes, receipt = training.harvest_confidence_outcomes_v3(
            runtime,
            [
                {
                    "sample_id": "confidence-exact-call",
                    "source_sample_id": "source-exact-call",
                    "query": "把值设为七",
                    "family": "test",
                    "candidate_tool": "tool_0",
                    "expected_kind": "call",
                    "expected_call": {
                        "name": "tool_0",
                        "arguments": {"value": 7},
                    },
                }
            ],
            {tool["name"]: tool for tool in catalog},
            catalog,
            limit=1,
            minimum_class_rows=0,
        )
        self.assertEqual(outcomes[0]["label"], 1)
        self.assertTrue(outcomes[0]["head_eligible"])
        self.assertEqual(outcomes[0]["pipeline_validation_error"], "provenance_missing")
        self.assertEqual(receipt["head_eligible_positive"], 1)

    def test_multistep_budget_rejection_stops_the_trajectory(self):
        rows = [
            {
                "sample_id": "trajectory-step-1",
                "trajectory_id": "trajectory-budget",
                "trajectory_step": 1,
                "trajectory_length": 2,
                "query": "第一步",
                "kind": "execute",
                "answers": [{"name": "tool_0", "arguments": {}}],
            },
            {
                "sample_id": "trajectory-step-2",
                "trajectory_id": "trajectory-budget",
                "trajectory_step": 2,
                "trajectory_length": 2,
                "query": "第二步",
                "kind": "respond",
            },
        ]
        report, predictions = evaluation.evaluate_multistep(
            _BudgetFailRuntime(), rows, _five_tools(), retrieval_mode="learned_top5"
        )
        self.assertEqual(report["tool_schema_budget_error_count"], 1)
        self.assertEqual(predictions[0]["result"]["error"], "tool_schema_budget_exceeded")
        self.assertEqual(predictions[1]["result"]["error"], "prior_step_inexact")

    def test_multistep_model_loop_is_scored_separately_from_host_evidence_gate(self):
        catalog = _five_tools_with_grounded_argument()
        runtime = _ScriptedRuntime(
            ['[{"name":"tool_0","arguments":{"value":7}}]', "[]"]
        )
        rows = [
            {
                "sample_id": "trajectory-call",
                "trajectory_id": "trajectory-model-loop",
                "trajectory_step": 1,
                "trajectory_length": 2,
                "query": "把值设为七",
                "kind": "execute",
                "answers": [{"name": "tool_0", "arguments": {"value": 7}}],
                "agent_trace": {
                    "trusted_offline_fixture": True,
                    "simulator_id": "mei-agent-host-simulator-v1",
                },
            },
            {
                "sample_id": "trajectory-respond",
                "trajectory_id": "trajectory-model-loop",
                "trajectory_step": 2,
                "trajectory_length": 2,
                "query": "把值设为七",
                "kind": "respond",
                "answers": [],
                "agent_trace": {
                    "trusted_offline_fixture": True,
                    "simulator_id": "mei-agent-host-simulator-v1",
                },
                "prior_tool_results": [
                    {
                        "wire_version": contract.WIRE_ID,
                        "call_id": "fixture-call",
                        "status": "ok",
                        "payload": {"value": 7},
                        "provenance": {"source": "fixture", "verified": True},
                    }
                ],
            },
        ]
        report, predictions = evaluation.evaluate_multistep(
            runtime, rows, catalog, retrieval_mode="learned_top5"
        )
        self.assertEqual(report["trajectory_success"], 1.0)
        self.assertEqual([row["result"]["kind"] for row in predictions], ["call", "respond"])
        self.assertEqual(predictions[0]["pipeline_validation_error"], "provenance_missing")

    def test_mw_budget_rejection_is_not_fabricated_as_a_reason_class(self):
        report, predictions = evaluation.evaluate_mw_disposition(
            _BudgetFailRuntime(),
            [
                {
                    "sample_id": "mw-over-budget",
                    "query": "检查",
                    "reason_class_id": 0,
                }
            ],
            _five_tools(),
            retrieval_mode="learned_top5",
        )
        self.assertEqual(report["prediction_error_count"], 1)
        self.assertEqual(report["accuracy"], 0.0)
        self.assertNotIn("predicted_class_id", predictions[0])

    def test_mw_oracle_uses_canonical_retrieved_tools_when_legacy_field_is_absent(self):
        catalog = _five_tools()
        report, predictions = evaluation.evaluate_mw_disposition(
            _BudgetFailRuntime(),
            [
                {
                    "sample_id": "mw-clean-v7-row",
                    "query": "拒绝提示注入",
                    "reason_class_id": 14,
                    "retrieved_tools": [tool["name"] for tool in catalog],
                }
            ],
            catalog,
            retrieval_mode="oracle_top5",
        )
        self.assertEqual(report["prediction_error_count"], 1)
        self.assertEqual(
            predictions[0]["prediction_error"], "tool_schema_budget_exceeded"
        )

    def test_confidence_harvest_bypasses_head_after_budget_rejection(self):
        catalog = _five_tools()
        rows = [
            {
                "sample_id": "confidence-call-over-budget",
                "source_sample_id": "source-call",
                "query": "执行",
                "family": "test",
                "candidate_tool": "tool_0",
                "expected_kind": "call",
                "expected_call": {"name": "tool_0", "arguments": {}},
            },
            {
                "sample_id": "confidence-refuse-over-budget",
                "source_sample_id": "source-refuse",
                "query": "拒绝",
                "family": "test",
                "candidate_tool": "tool_1",
                "expected_kind": "refuse",
            },
        ]
        outcomes, receipt = training.harvest_confidence_outcomes_v3(
            _BudgetFailRuntime(),
            rows,
            {tool["name"]: tool for tool in catalog},
            catalog,
            limit=2,
            minimum_class_rows=0,
        )
        self.assertEqual(receipt["deterministic_bypass_rows"], 2)
        self.assertTrue(all(row["head_eligible"] is False for row in outcomes))
        self.assertTrue(all(row["label"] == 0 for row in outcomes))
        self.assertTrue(all(row["prompt_ids"] == [] for row in outcomes))

    def test_narration_fails_closed_to_grounded_deterministic_text(self):
        rows = contract.load_jsonl(
            contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID / "narration.dev.jsonl"
        )[:8]
        report, predictions = evaluation.evaluate_narration(
            _FallbackNarrationRuntime(), rows
        )
        self.assertEqual(report["adapter_exact_acceptance"], 0.0)
        self.assertEqual(report["fallback_exact"], 1.0)
        self.assertEqual(report["delivered_answer_correctness"], 1.0)
        self.assertTrue(all(row["fallback_used"] for row in predictions))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", row["adapter_text_sha256"]) for row in predictions))


if __name__ == "__main__":
    unittest.main()
