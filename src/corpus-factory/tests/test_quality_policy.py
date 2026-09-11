from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sources = load_module(
    "mei_51m_source_manager", "src/corpus-factory/sources/source_manager.py"
)
quality = load_module("mei_51m_corpus_quality", "src/corpus-factory/quality/audit.py")


class FakeTokenizer:
    model_sha256 = "a" * 64
    tokenizer_id = "zh-24k-v1"

    def encode_document(self, text: str) -> list[int]:
        return [2, *[ord(character) % 1000 for character in text], 1]


def admit_dialogue(root: Path, *, with_clearance: bool) -> Path:
    source = root / "dialogue.jsonl"
    source.write_text(
        json.dumps({"text": "把客厅的灯打开"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = root / "admitted"
    clearance = None
    if with_clearance:
        clearance = root / "clearance.json"
        clearance.write_text(json.dumps({"source_id": "opensubtitles-zh", "status": "passed",
                                         "reviewer": "fixture reviewer", "reviewed_at": "2026-09-04"}), encoding="utf-8")
    with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
        sources.admit(
            [source],
            out,
            role="dialogue",
            license_id="subtitle-rights-uncleared",
            license_reviewed=True,
            seen_ledger=None,
            source_id="opensubtitles-zh",
            source_url="https://opus.nlpl.eu/OpenSubtitles.php",
            clearance_receipt=clearance,
        )
    return out


def admit_structured(root: Path) -> Path:
    source = root / "records.jsonl"
    source.write_text(
        json.dumps({"id": "Q1", "name": "地球"}) + "\n"
        + json.dumps({"id": "Q2", "name": "月球"}) + "\n",
        encoding="utf-8",
    )
    out = root / "admitted"
    with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
        sources.admit(
            [source],
            out,
            role="structured",
            license_id="CC0-1.0",
            license_reviewed=True,
            seen_ledger=None,
            source_id="wikidata-json",
            source_url="https://example.com/records.jsonl",
            mode="structured",
        )
    return out


class PolicyGateTests(unittest.TestCase):
    def test_unsigned_or_wrong_source_clearance_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            admitted = admit_dialogue(root, with_clearance=True)
            clearance = root / "clearance.json"
            clearance.write_text(json.dumps({"source_id": "other-source", "status": "pending_named_review"}))
            manifest_path = admitted / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["clearance_receipt"]["sha256"] = quality.sha256_file(clearance)
            manifest_path.write_text(json.dumps(manifest))
            receipt = quality.audit_source(manifest_path)
            self.assertEqual(receipt["status"], "blocked")
            self.assertIn("clearance source_id mismatch", receipt["errors"])
            self.assertIn("clearance named review missing", receipt["errors"])

    def test_dialogue_without_clearance_is_blocked_by_policy(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            admitted = admit_dialogue(root, with_clearance=False)
            receipt = quality.audit_source(admitted / "manifest.json")
            self.assertEqual("blocked", receipt["status"])
            self.assertIn("clearance receipt required by policy", receipt["errors"])

    def test_dialogue_with_clearance_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            admitted = admit_dialogue(root, with_clearance=True)
            receipt = quality.audit_source(admitted / "manifest.json")
            self.assertEqual("passed", receipt["status"])

    def test_band_violation_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            admitted = admit_structured(root)
            fake_entry = {
                "source_id": "wikidata-json",
                "role": "structured",
                "band": "C",
            }
            with patch.object(
                quality, "_registry_entry", return_value=fake_entry
            ):
                receipt = quality.audit_source(admitted / "manifest.json")
            self.assertEqual("blocked", receipt["status"])
            self.assertIn(
                "band 'C' not allowed for role 'structured'", receipt["errors"]
            )


class AuditStructuredTests(unittest.TestCase):
    def test_token_layout_and_rehash_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            admitted = admit_structured(root)
            receipt = quality.audit_structured(admitted / "manifest.json")
            self.assertEqual("passed", receipt["status"])
            self.assertEqual("structured_source", receipt["kind"])

    def test_tampered_record_sha256_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            admitted = admit_structured(root)
            documents = admitted / "documents.jsonl"
            rows = [json.loads(line) for line in documents.read_text(encoding="utf-8").splitlines()]
            rows[0]["record_sha256"] = "f" * 64
            documents.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            receipt = quality.audit_structured(admitted / "manifest.json")
            self.assertEqual("blocked", receipt["status"])
            self.assertTrue(
                any("record_sha256 mismatch" in error for error in receipt["errors"])
            )

    def test_broken_token_layout_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            admitted = admit_structured(root)
            documents = admitted / "documents.jsonl"
            rows = [json.loads(line) for line in documents.read_text(encoding="utf-8").splitlines()]
            rows[1]["token_offset"] += 7
            documents.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            receipt = quality.audit_structured(admitted / "manifest.json")
            self.assertEqual("blocked", receipt["status"])
            self.assertIn("token_offset layout broken", receipt["errors"])


if __name__ == "__main__":
    unittest.main()
