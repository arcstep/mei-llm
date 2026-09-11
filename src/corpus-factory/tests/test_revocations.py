import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "quality/revocations.py"
spec = importlib.util.spec_from_file_location("revocations", module_path)
revocations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(revocations)


class RevocationTests(unittest.TestCase):
    def test_renaming_manifest_does_not_bypass_revocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b'{"artifacts": {}}'
            manifest = root / "renamed.json"
            manifest.write_bytes(original)
            receipt = root / "corpus/pools/example/quality/REVOKED-v1.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({"status": "do_not_adopt", "artifacts": {
                "old-name.json": hashlib.sha256(original).hexdigest()}}))
            self.assertTrue(revocations.manifest_revocation_errors(manifest, {}, root))

    def test_rewrapping_revoked_tokens_does_not_bypass_revocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            receipt = root / "corpus/pools/example/quality/REVOKED-v1.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({"status": "do_not_adopt", "artifacts": {"old.bin": "a" * 64}}))
            self.assertTrue(revocations.manifest_revocation_errors(manifest, {"artifacts": {"new.bin": "a" * 64}}, root))
