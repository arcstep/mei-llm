from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import base_candidate_51m as candidate
import lifecycle_51m as lifecycle


class BaseCandidateBoundaryTest(unittest.TestCase):
    def test_register_and_propose_never_touch_current(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runs = root / "runs"
            directory = runs / "run-600m"
            cpt = directory / "checkpoints/cpt"
            (directory / "stages/cpt_gate").mkdir(parents=True)
            cpt.mkdir(parents=True)
            schedule = root / "corpus/schedule-cpt-600m.json"
            schedule.parent.mkdir(parents=True)
            schedule.write_text(json.dumps({"parent_checkpoint": "base/parent-state.npz"}), encoding="utf-8")
            config = {
                "run_id": "run-600m",
                "rung": "600m",
                "architecture_id": "mei-1.0-51m-arch-v1",
                "weight_contract_sha256": lifecycle.architecture_contracts()["weight_contract_sha256"],
                "target_exposure": 600_000_000,
                "target_exposure_tokens": 600_000_000,
                "corpus": {
                    "schedule": "corpus/schedule-cpt-600m.json",
                    "parent_exposure": 300_000_000,
                    "incremental_exposure": 300_000_000,
                    "snapshot_sha256": "snapshot",
                },
            }
            (directory / "run.json").write_text(json.dumps(config), encoding="utf-8")
            (directory / "stages/cpt_gate/receipt.json").write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )
            summary = {
                "tokens_seen": 600_000_000,
                "params": 51_463_797,
                "architecture_id": "mei-1.0-51m-arch-v1",
                "architecture_sha256": "f02aebda393c7fca181af7b7225581577f689531245dec00ac549de1482c5ea7",
            }
            (cpt / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            for name in ("pretrain-cpt.npz", "pretrain-cpt-state.npz", "pretrain-cpt-state.meta.json"):
                (cpt / name).write_bytes(name.encode())
            with patch.object(lifecycle, "RUN_ROOT", runs), patch.object(candidate, "ROOT", root), patch.object(
                candidate, "BASE_ROOT", root / "base"
            ), patch.object(candidate, "current_hash", return_value="unchanged"):
                registered = candidate.register_base_candidate("run-600m")
                dest = root / "base/mei-1.0-51m-base-cpt600m-v1"
                self.assertTrue((dest / "RELEASE.json").is_file())
                self.assertTrue(registered["current_unchanged"])
                proposal = candidate.propose_freeze(dest)
                self.assertTrue(proposal["eligible"])
                self.assertEqual(proposal["current_sha256_observed"], "unchanged")

    def test_finalize_refuses_without_explicit_confirmation_before_any_write(self) -> None:
        with patch.object(candidate, "current_hash", side_effect=AssertionError("must not read CURRENT")):
            with self.assertRaises(PermissionError):
                candidate.finalize_current(
                    Path("base/mei-1.0-51m-base-cpt600m-v1"),
                    expected_current_sha256="unused",
                    confirmation="not authorized",
                )

    def test_legacy_promoter_contains_no_current_write(self) -> None:
        here = Path(__file__).resolve().parent
        for name in (
            "promote_cpt1b_base.py",
            "promote_scratch_base.py",
            "promote_scratch_base_51m.py",
        ):
            text = (here / name).read_text(encoding="utf-8")
            self.assertIn("retired", text)
            self.assertNotIn("activate_current", text)
            self.assertNotIn(".replace(current_path)", text)
            self.assertNotIn("atomic_json(CURRENT_PATH", text)


if __name__ == "__main__":
    unittest.main()
