from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sft_v3_contract_51m as contract
import sft_v3_eval_51m as evaluation


class _FallbackNarrationRuntime:
    narration_adapter = object()

    @staticmethod
    def generate_narration(prompt: str, *, max_new: int):
        del prompt, max_new
        return {"text": "不受信任的自由生成", "n_out": 8}


class SftV3EvaluationTests(unittest.TestCase):
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
