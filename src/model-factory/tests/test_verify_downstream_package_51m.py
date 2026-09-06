from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import orchestration.verify_downstream_package_51m as verifier


class CurrentSourceReevaluationTests(unittest.TestCase):
    def test_source_drift_is_explicit_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "runtime.py"
            source.write_text("current\n", encoding="utf-8")
            with mock.patch.object(verifier, "ROOT", root):
                drift = verifier._current_source_drift(
                    {"runtime.py": "0" * 64, "missing.py": "1" * 64}
                )
            self.assertEqual([row["path"] for row in drift], ["missing.py", "runtime.py"])
            self.assertIsNone(drift[0]["current_sha256"])
            self.assertEqual(drift[1]["current_sha256"], verifier.sha_file(source))

    def test_reevaluation_flag_requires_productization_run(self) -> None:
        args = verifier.parse_args(
            [
                "--package",
                "package",
                "--adoption-receipt",
                "adoption.json",
                "--current-source-reevaluation",
                "--out",
                "out.json",
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "requires --productization-run"):
            verifier.verify(
                args.package,
                args.adoption_receipt,
                args.out,
                current_source_reevaluation=args.current_source_reevaluation,
            )


if __name__ == "__main__":
    unittest.main()
