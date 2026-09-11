import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import source_capture


class SourceCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "src/model-factory/training/cpt/train.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("VALUE = 1\n")

    def tearDown(self):
        self.temporary.cleanup()

    def capture(self):
        manifest = source_capture.manifest(self.root)
        with patch.object(source_capture.subprocess, "check_output", side_effect=[b"patch", "revision\n"]):
            binding = source_capture.capture(self.root, self.root / "capture", manifest, ["cpt", "cpt_gate"])
        return manifest, binding

    def test_training_and_untracked_bytes_are_recoverable(self):
        manifest, binding = self.capture()
        self.source.unlink()
        self.assertEqual(source_capture.verify(manifest, binding), [])
        self.assertIn("src/model-factory/training/cpt/train.py", manifest["files"])
        self.assertEqual(set(binding["phase_closures"]), {"cpt", "cpt_gate"})

    def test_source_drift_during_capture_is_rejected(self):
        manifest = source_capture.manifest(self.root)
        self.source.write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "changed during capture"):
            source_capture.capture(self.root, self.root / "capture", manifest, ["cpt"])

    def test_missing_or_modified_archive_is_rejected(self):
        manifest, binding = self.capture()
        archive = Path(binding["source_archive"])
        archive.write_bytes(b"broken")
        self.assertIn("source_archive missing or changed", source_capture.verify(manifest, binding))

    def test_invalid_archive_with_updated_outer_hash_is_rejected(self):
        manifest, binding = self.capture()
        archive = Path(binding["source_archive"])
        archive.write_bytes(b"broken")
        binding["source_archive_sha256"] = hashlib.sha256(b"broken").hexdigest()
        self.assertTrue(source_capture.verify(manifest, binding))

    def test_missing_manifest_is_not_recoverable(self):
        _, binding = self.capture()
        self.assertIn("source manifest binding mismatch", source_capture.verify({}, binding))


if __name__ == "__main__":
    unittest.main()
