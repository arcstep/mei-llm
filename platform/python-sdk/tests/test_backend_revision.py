from __future__ import annotations

import unittest
from pathlib import Path

from mei_sdk.mlx_backend import backend_file_fingerprints, backend_revision
from mei_sdk.version import SDK_ROOT


class BackendRevisionTests(unittest.TestCase):
    def test_revision_is_stable_hex(self):
        a = backend_revision()
        b = backend_revision()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        files = backend_file_fingerprints()
        self.assertTrue(files)
        self.assertTrue(all(len(v) == 64 or v == "missing" for v in files.values()))
        arch = SDK_ROOT.parents[1] / "models/mei-1.0-51m/architecture/architecture.py"
        self.assertTrue(arch.is_file())

    def test_fused_revision_is_distinct_and_includes_fused_ops(self):
        reference = backend_revision("mlx-reference")
        fused = backend_revision("mlx-fused")
        self.assertNotEqual(reference, fused)
        files = backend_file_fingerprints("mlx-fused")
        self.assertTrue(any(path.endswith("/fused_ops.py") for path in files))

    def test_cq2_revision_includes_packed_metal_source(self):
        fused = backend_revision("mlx-fused")
        cq2 = backend_revision("mlx-cq2")
        self.assertNotEqual(fused, cq2)
        files = backend_file_fingerprints("mlx-cq2")
        self.assertTrue(any(path.endswith("/cq2_metal.py") for path in files))


if __name__ == "__main__":
    unittest.main()
