from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_downstream_portable_gates_51m as portable


class PortableGateSummaryTests(unittest.TestCase):
    def test_release_browser_wasm_field_is_the_authoritative_build_gate(self):
        self.assertTrue(
            portable.portable_report_passed(
                {
                    "all_gates_passed": True,
                    "browser_wasm_build": {"profile": "release"},
                }
            )
        )
        self.assertFalse(
            portable.portable_report_passed(
                {
                    "all_gates_passed": True,
                    "node_wasm_build": {"profile": "release"},
                }
            )
        )
        self.assertFalse(
            portable.portable_report_passed(
                {
                    "all_gates_passed": False,
                    "browser_wasm_build": {"profile": "release"},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
