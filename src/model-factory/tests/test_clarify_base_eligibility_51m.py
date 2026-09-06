from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import release.clarify_base_eligibility_51m as clarify


class ClarifyBaseEligibilityTests(unittest.TestCase):
    def test_checkpoint_continuation_is_separate_from_corpus_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            weights = root / "base.npz"
            state = root / "base-state.npz"
            np.savez(weights, weight=np.asarray([1.0], dtype=np.float32))
            np.savez(
                state,
                **{
                    "p.weight": np.asarray([1.0], dtype=np.float32),
                    "o.step": np.asarray(1, dtype=np.int64),
                },
            )
            release = root / "RELEASE.json"
            release.write_text(
                json.dumps(
                    {
                        "model_id": "base-test",
                        "tokens_seen_exposure": 600,
                        "weights": weights.name,
                        "weights_sha256": clarify.sha256_file(weights),
                        "train_state": state.name,
                        "state_sha256": clarify.sha256_file(state),
                    }
                ),
                encoding="utf-8",
            )
            diversity = root / "diversity.json"
            diversity.write_text(
                json.dumps(
                    {
                        "schema": "mei-synthetic-corpus-diversity-receipt-v1",
                        "terminal_status": "degraded",
                        "corpus_diversity_degraded": True,
                    }
                ),
                encoding="utf-8",
            )
            scan = {
                "status": "passed",
                "parameter_tensor_count": 400,
                "parameter_count": 51_463_797,
                "optimizer_tensor_count": 1,
                "parameter_names_and_order_exact": True,
                "parameter_shapes_exact": True,
                "all_numeric_tensors_finite": True,
            }
            with (
                mock.patch.object(clarify, "_scan_checkpoint", return_value=scan),
                mock.patch.object(clarify, "CURRENT_PATH", release),
            ):
                receipt = clarify.build_receipt(release, diversity)
            eligibility = receipt["eligibility"]
            self.assertTrue(eligibility["continuation_checkpoint_eligible"])
            self.assertFalse(eligibility["automatic_parent_promotion_eligible"])
            self.assertFalse(eligibility["corpus_reuse_eligible"])

    def test_clarification_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "receipt.json"
            clarify.write_once(path, {"status": "passed"})
            clarify.write_once(path, {"status": "passed"})
            with self.assertRaises(FileExistsError):
                clarify.write_once(path, {"status": "degraded"})


if __name__ == "__main__":
    unittest.main()
