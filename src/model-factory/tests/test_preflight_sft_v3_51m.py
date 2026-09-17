from __future__ import annotations

import unittest

import training.tool_use.preflight_sft_v3_51m as preflight
import orchestration.productize_sft_bootstrap as productizer


class PreflightRunIdentityTests(unittest.TestCase):
    def test_package_and_recipe_are_forwarded_to_productizer_identity(self) -> None:
        args = preflight.parse_args(
            [
                "--package-id",
                "mei-1.0-51m-cpt600m-test",
                "--float-control-steps",
                "7201",
                "--agent-steps",
                "4001",
                "--eval-limit",
                "1175",
            ]
        )
        product_args = productizer.parse_args(preflight._productizer_argv(args))
        self.assertEqual(product_args.package_id, "mei-1.0-51m-cpt600m-test")
        self.assertEqual(product_args.float_control_steps, 7201)
        self.assertEqual(product_args.agent_steps, 4001)
        self.assertEqual(product_args.eval_limit, 1175)


if __name__ == "__main__":
    unittest.main()
