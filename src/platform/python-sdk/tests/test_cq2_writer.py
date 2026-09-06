from __future__ import annotations

import unittest

import numpy as np

from mei_sdk import SdkError, TensorContainer, TensorToPack, build_tensor_container


class Cq2WriterTest(unittest.TestCase):
    def test_builds_mixed_quant_and_safe_heads_in_one_container(self) -> None:
        values = np.linspace(-2.0, 2.0, 200, dtype=np.float32)
        blob, directory = build_tensor_container(
            [
                TensorToPack(
                    "lm.matrix",
                    (20, 10),
                    values,
                    "lm",
                    "cq2",
                    (2, 4),
                ),
                TensorToPack(
                    "heads.mw_disposition.proj.bias",
                    (20,),
                    np.zeros((20,), dtype=np.float32),
                    "mw_disposition",
                    "f16",
                ),
                TensorToPack(
                    "lm.scalar",
                    (),
                    np.array(0.25, dtype=np.float32),
                    "lm",
                    "f32",
                ),
            ]
        )
        parsed = TensorContainer.parse(blob)
        self.assertEqual([row["name"] for row in directory], list(parsed.entries))
        self.assertEqual(parsed.entries["lm.matrix"].dtype, "cq2")
        self.assertEqual(parsed.entries["lm.scalar"].shape, ())
        self.assertEqual(len(parsed.dequantize("lm.matrix")), 200)
        self.assertEqual(parsed.dequantize("heads.mw_disposition.proj.bias"), [0.0] * 20)
        self.assertAlmostEqual(parsed.dequantize("lm.scalar")[0], 0.25)

    def test_refuses_duplicate_and_all_q4_masquerading_as_cq2(self) -> None:
        row = TensorToPack("lm.w", (1,), [1.0], "lm", "f16")
        with self.assertRaisesRegex(SdkError, "lm.w"):
            build_tensor_container([row, row])
        with self.assertRaisesRegex(SdkError, "all-q4"):
            build_tensor_container(
                [TensorToPack("lm.w", (129,), np.ones(129), "lm", "cq2", (4, 4))]
            )


if __name__ == "__main__":
    unittest.main()
