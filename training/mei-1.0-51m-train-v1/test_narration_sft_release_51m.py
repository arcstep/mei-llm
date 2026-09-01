from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location(
    "freeze_narration_sft_release_51m", HERE / "freeze_narration_sft_release_51m.py"
)
assert spec and spec.loader
freeze_narration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_narration)


class NarrationSftReleaseTests(unittest.TestCase):
    def test_verified_agent_results_become_grounded_narration_only(self):
        rows = freeze_narration.build_rows(freeze_narration.PARENT_DIR)
        self.assertEqual({key: len(value) for key, value in rows.items()}, {"train": 800, "valid": 100, "eval": 100})
        row = rows["train"][0]
        self.assertTrue(row["verified_result_views"])
        self.assertFalse(row["can_execute_tools"])
        self.assertIn("已执行完成", row["target"])
        self.assertIn("<verified_results>", row["prompt"])
        self.assertEqual(row["grounding_target"], "exact-deterministic-template")


if __name__ == "__main__":
    unittest.main()
