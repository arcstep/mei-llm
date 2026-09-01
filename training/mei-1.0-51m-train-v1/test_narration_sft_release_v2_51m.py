from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "freeze_narration_v2", HERE / "freeze_narration_sft_release_v2_51m.py"
)
assert SPEC and SPEC.loader
freeze_narration_v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze_narration_v2)


class NarrationV2ReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = freeze_narration_v2.build_rows(
            freeze_narration_v2.RELEASE_ROOT / freeze_narration_v2.PARENT_NARRATION_ID
        )
        cls.coverage = freeze_narration_v2.validate(cls.rows)

    def test_counts_and_isolation(self) -> None:
        self.assertEqual({key: len(value) for key, value in self.rows.items()}, {
            "train": 4_800,
            "valid": 600,
            "eval": 600,
        })
        self.assertEqual(self.coverage["group_cross_split"], 0)
        self.assertEqual(self.coverage["prompt_overlap"], 0)

    def test_temperature_power_failure_and_multistep_are_explicit(self) -> None:
        families = {row["semantic_family"] for row in self.rows["train"]}
        self.assertTrue({
            "temperature_query", "temperature_set", "temperature_raise",
            "device_start", "device_stop", "device_start_error",
            "device_cancelled", "multi_step_climate",
        } <= families)
        targets = "\n".join(row["target"] for row in self.rows["train"])
        for phrase in ("当前温度为", "已启动", "已关闭", "启动失败", "已取消", "已将客厅温度设为"):
            self.assertIn(phrase, targets)

    def test_every_row_is_bounded_and_executor_free(self) -> None:
        for split in self.rows.values():
            for row in split:
                self.assertFalse(row["can_execute_tools"])
                self.assertLessEqual(len(row["target"]), 160)
                self.assertTrue(all(view["provenance"]["verified"] for view in row["verified_result_views"]))


if __name__ == "__main__":
    unittest.main()
