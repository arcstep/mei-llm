from __future__ import annotations

import unittest

from orchestration.audit_source_recovery import classify_expected


class SourceRecoveryAuditTest(unittest.TestCase):
    def test_classifies_exact_bytes_by_strongest_available_source(self) -> None:
        report = classify_expected(
            {"a.py": "a", "b.py": "b", "c.py": "c", "d.py": "d", "e.py": "e"},
            current={"a": ["moved/a.py"]},
            archives={},
            head={"a": ["a.py"], "b": ["b.py"]},
            index={"c": ["c.py"]},
            git_objects={"d": ["deadbeef"]},
        )
        self.assertEqual(report["manifest_entries"], 5)
        self.assertEqual(report["exact_recoverable"], 4)
        self.assertEqual(report["exact_unavailable"], 1)
        self.assertEqual(
            report["recovery_method_counts"],
            {
                "current_tree": 1,
                "git_head": 1,
                "git_index": 1,
                "git_object": 1,
                "unavailable": 1,
            },
        )
        self.assertEqual(report["unavailable"][0]["historical_path"], "e.py")


if __name__ == "__main__":
    unittest.main()
