from __future__ import annotations

import math
import unittest

from evaluation.resources.measure_resources_51m import qualification_gates


class ResourceQualificationTests(unittest.TestCase):
    def test_browser_scope_does_not_promote_native_rust_diagnostic_to_hard_gate(self):
        gates, diagnostics, rate = qualification_gates(
            {
                "package_bytes": 18 * 1024 * 1024,
                "rust_session_peak_bytes": 400 * 1024 * 1024,
                "wasm_heap_peak_bytes": 80 * 1024 * 1024,
            },
            {"steady_decode_tok_s": 25.0},
        )
        self.assertTrue(all(gates.values()))
        self.assertFalse(diagnostics["rust_session"])
        self.assertEqual(rate, 25.0)

    def test_decode_floor_and_non_finite_rate_fail_closed(self):
        measurements = {
            "package_bytes": 17 * 1024 * 1024,
            "rust_session_peak_bytes": 1,
            "wasm_heap_peak_bytes": 80 * 1024 * 1024,
        }
        for value in (9.99, math.nan, math.inf, "not-a-rate"):
            gates, _diagnostics, rate = qualification_gates(
                measurements, {"steady_decode_tok_s": value}
            )
            self.assertFalse(gates["wasm_steady_decode"])
            self.assertTrue(math.isfinite(rate))


if __name__ == "__main__":
    unittest.main()
