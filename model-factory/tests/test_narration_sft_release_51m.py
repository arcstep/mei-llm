from __future__ import annotations

import unittest

from release import freeze_narration_sft_release_51m as freeze_narration


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
