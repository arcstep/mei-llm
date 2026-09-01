from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "evaluate_narration_adapter_51m",
    HERE / "evaluate_narration_adapter_51m.py",
)
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


class NarrationGenerationEvaluationTest(unittest.TestCase):
    def test_number_and_polarity_metrics_are_semantic(self) -> None:
        self.assertEqual(evaluator.extract_numbers("已调到27℃，原来是26℃"), ["27", "26"])
        self.assertTrue(evaluator.polarity_matches("空调启动失败：设备离线。", "failure_on", []))
        self.assertFalse(evaluator.polarity_matches("空调已启动。", "failure_on", []))
        self.assertTrue(evaluator.polarity_matches("空调已关闭。", "success_off", []))
        self.assertTrue(evaluator.polarity_matches("操作已取消。", "cancelled", []))

    def test_summary_distinguishes_adapter_quality_from_safe_delivery(self) -> None:
        base = {
            "semantic_family": "temperature_query",
            "required_facts": ["26", "温度"],
            "target_numbers": ["26"],
            "adapter_nonempty": True,
            "required_facts_match": True,
            "numeric_exact": True,
            "polarity_match": True,
            "bounded": True,
            "max_new_reached": False,
            "delivered_exact": True,
        }
        exact = {**base, "adapter_exact": True, "fallback_used": False}
        fallback = {
            **base,
            "adapter_exact": False,
            "fallback_used": True,
            "required_facts_match": False,
            "numeric_exact": False,
            "polarity_match": False,
        }
        report = evaluator.summarize_cases([exact, fallback])
        self.assertEqual(report["adapter_exact_rate"], 0.5)
        self.assertEqual(report["adapter_fallback_rate"], 0.5)
        self.assertEqual(report["deterministic_fallback_exact_rate"], 1.0)
        self.assertEqual(report["mechanism_status"], "passed")
        self.assertEqual(report["learned_adapter_quality"], "degraded")


if __name__ == "__main__":
    unittest.main()
