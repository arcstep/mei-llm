from __future__ import annotations

import unittest

import evaluation.heads.evaluate_narration_adapter_51m as narration
import orchestration.run_downstream_mtp_ablation_51m as mtp
import orchestration.run_downstream_portable_gates_51m as portable
import orchestration.run_downstream_resource_gates_51m as resources
import orchestration.run_paired_product_runtime_gates_51m as driver
import orchestration.verify_arbitrary_base_entry_51m as base_entry
import orchestration.verify_downstream_package_51m as package


class PairedRuntimeGateDriverTests(unittest.TestCase):
    def test_command_order_and_current_source_mode_are_explicit(self) -> None:
        args = driver.parse_args(
            [
                "--productization-run",
                "product",
                "--package",
                "package",
                "--master",
                "master.npz",
                "--base-release",
                "RELEASE.json",
                "--base-weights",
                "base.npz",
                "--out-root",
                "out",
                "--current-source-reevaluation",
            ]
        )
        commands = driver.command_plan(args)
        self.assertEqual(
            [row["stage"] for row in commands],
            [
                "package_verification",
                "portable_runtime_gates",
                "narration_generation_eval",
                "mtp_ablation",
                "resource_measurement",
                "arbitrary_base_entry",
            ],
        )
        self.assertIn("--current-source-reevaluation", commands[0]["command"])
        self.assertLess(
            [row["stage"] for row in commands].index("portable_runtime_gates"),
            [row["stage"] for row in commands].index("resource_measurement"),
        )

        parsers = {
            "package_verification": package.parse_args,
            "portable_runtime_gates": portable.parse_args,
            "narration_generation_eval": narration.parse_args,
            "mtp_ablation": mtp.parse_args,
            "resource_measurement": resources.parse_args,
            "arbitrary_base_entry": base_entry.parse_args,
        }
        for row in commands:
            with self.subTest(stage=row["stage"]):
                parsed = parsers[row["stage"]](row["command"][2:])
                self.assertIsNotNone(parsed)


if __name__ == "__main__":
    unittest.main()
