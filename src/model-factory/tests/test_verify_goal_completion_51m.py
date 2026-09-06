from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import orchestration.verify_goal_completion_51m as audit


class GoalCompletionAuditTests(unittest.TestCase):
    def test_hash_map_verifies_declared_files(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=audit.ROOT.parent, prefix="mei-goal-audit-hash-"
        ) as raw:
            path = Path(raw) / "artifact.bin"
            path.write_bytes(b"frozen")
            expected = hashlib.sha256(b"frozen").hexdigest()
            rows = audit._verify_hash_map(
                {str(path): expected}, cache={}, label="fixture"
            )
            self.assertEqual(rows[0]["sha256"], expected)
            with self.assertRaises(RuntimeError):
                audit._verify_hash_map(
                    {str(path): "0" * 64}, cache={}, label="fixture"
                )

    def test_receipt_verifies_outputs_and_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=audit.ROOT.parent, prefix="mei-goal-audit-receipt-"
        ) as raw:
            root = Path(raw)
            output = root / "output.json"
            output.write_text("{}\n", encoding="utf-8")
            receipt = root / "stages/example/receipt.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(
                json.dumps(
                    {
                        "stage_id": "example",
                        "terminal_status": "degraded",
                        "output_hashes": {
                            str(output): hashlib.sha256(output.read_bytes()).hexdigest()
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = audit._verify_receipt(receipt, cache={})
            self.assertEqual(row["terminal_status"], "degraded")
            self.assertEqual(row["verified_output_count"], 1)

    def test_skill_mirror_is_byte_exact_and_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=audit.ROOT.parent, prefix="mei-goal-audit-skill-"
        ) as raw:
            root = Path(raw)
            canonical = root / "canonical"
            cursor = root / "cursor"
            for target in (canonical, cursor):
                (target / "references").mkdir(parents=True)
                (target / "SKILL.md").write_text("same\n", encoding="utf-8")
                (target / "references/x.md").write_text("same ref\n", encoding="utf-8")
            report = audit._verify_skill_mirror(canonical, cursor, cache={})
            self.assertTrue(report["byte_identical"])
            self.assertEqual(report["file_count"], 2)
            (cursor / "SKILL.md").write_text("drift\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                audit._verify_skill_mirror(canonical, cursor, cache={})


if __name__ == "__main__":
    unittest.main()
