from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "measure_resources_51m", HERE / "measure_resources_51m.py"
)
assert SPEC and SPEC.loader
resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resources)


class ResourceReceiptTest(unittest.TestCase):
    def test_attach_receipt_converges_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            source = temp / "draft"
            destination = temp / "measured"
            source.mkdir()
            payload = b"portable tensor fixture"
            (source / "tensors.bin").write_bytes(payload)
            manifest = {
                "package_format": "mei-model-package-v2",
                "package_id": "fixture-v2",
                "tensor_container": {
                    "file": "tensors.bin",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "files": [
                    {
                        "path": "tensors.bin",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "nbytes": len(payload),
                        "role": "tensor_container",
                    }
                ],
                "resources": {"package_bytes": 0},
            }
            resources.write_json(source / "mei-model.json", manifest)
            source_before = resources.sha_file(source / "mei-model.json")
            final_manifest, receipt = resources.attach_receipt(
                source,
                destination,
                rust_peak=1234,
                wasm_peak=5678,
                rust_runner={"runner_id": "rust-fixture"},
                wasm_runner={"runner_id": "wasm-fixture"},
            )
            measured = sum(path.stat().st_size for path in destination.iterdir() if path.is_file())
            self.assertEqual(receipt["measurements"]["package_bytes"], measured)
            self.assertEqual(final_manifest["resources"]["package_bytes"], measured)
            receipt_row = next(
                row for row in final_manifest["files"] if row["role"] == "resource_receipt"
            )
            self.assertEqual(
                receipt_row["sha256"],
                resources.sha_file(destination / "resource-measurement-receipt.json"),
            )
            self.assertEqual(resources.sha_file(source / "mei-model.json"), source_before)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                resources.attach_receipt(
                    source,
                    destination,
                    rust_peak=1,
                    wasm_peak=1,
                    rust_runner={},
                    wasm_runner={},
                )


if __name__ == "__main__":
    unittest.main()
