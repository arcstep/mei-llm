from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "CURRENT.json").is_file())
ARCH = ROOT / "src/architecture/mei-1.2-51m"
sys.path.insert(0, str(ARCH))

from heads import (  # noqa: E402
    MW_DISPOSITION_CLASSES,
    NARRATION_ADAPTER_RANK,
    MWDispositionHead,
    NarrationAdapterHead,
)


class ProductHeads51MTest(unittest.TestCase):
    def test_mw_disposition_has_locked_20_by_512_geometry(self) -> None:
        head = MWDispositionHead(512)
        cells = [mx.ones((2, 3, 512), dtype=mx.float16) for _ in range(28)]
        logits = head(cells)
        mx.eval(logits, head.parameters())
        self.assertEqual(MW_DISPOSITION_CLASSES, 20)
        self.assertEqual(tuple(logits.shape), (2, 20))
        self.assertEqual(tuple(head.proj.weight.shape), (20, 512))
        self.assertEqual(tuple(head.proj.bias.shape), (20,))
        self.assertEqual(int(head.proj.weight.size + head.proj.bias.size), 10_260)

    def test_mw_disposition_rejects_class_count_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            MWDispositionHead(512, n_classes=16)

    def test_narration_adapter_is_rank16_and_outside_backbone(self) -> None:
        head = NarrationAdapterHead(512, 24_000)
        hidden = mx.ones((1, 2, 512), dtype=mx.float16)
        logits = head(hidden)
        mx.eval(logits, head.parameters())
        self.assertEqual(NARRATION_ADAPTER_RANK, 16)
        self.assertEqual(tuple(head.down.weight.shape), (16, 512))
        self.assertEqual(tuple(head.up.weight.shape), (24_000, 16))
        self.assertEqual(tuple(logits.shape), (1, 2, 24_000))
        self.assertEqual(int(head.down.weight.size + head.up.weight.size), 392_192)


if __name__ == "__main__":
    unittest.main()
