from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sources = load_module(
    "mei_51m_source_manager", "corpus-factory/sources/source_manager.py"
)
structured = load_module(
    "mei_51m_source_structured", "corpus-factory/sources/structured.py"
)
quality = load_module("mei_51m_corpus_quality", "corpus-factory/quality/audit.py")


class FakeTokenizer:
    model_sha256 = "a" * 64
    tokenizer_id = "zh-24k-v1"

    def encode_document(self, text: str) -> list[int]:
        return [2, *[ord(character) % 1000 for character in text], 1]


def admit_structured(
    source: Path, out: Path, *, source_id: str = "wikidata-json",
    max_invalid_ratio: float | None = None, seen_ledger: Path | None = None,
):
    with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
        return sources.admit(
            [source],
            out,
            role="structured",
            license_id="CC0-1.0",
            license_reviewed=True,
            seen_ledger=seen_ledger,
            source_id=source_id,
            source_url="https://example.com/data.jsonl",
            dataset_version="2024-08-01",
            mode="structured",
            max_invalid_ratio=max_invalid_ratio,
        )


class StructuredRecordizerTests(unittest.TestCase):
    def test_json_records_use_canonical_key_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "records.jsonl"
            path.write_text(
                json.dumps({"b": 1, "a": 2}) + "\n"
                + json.dumps({"a": 2, "b": 1}) + "\n",
                encoding="utf-8",
            )
            records = list(structured.iter_records(path))
            self.assertEqual(2, len(records))
            self.assertTrue(all(record["valid"] for record in records))
            self.assertEqual(records[0]["canonical_bytes"], records[1]["canonical_bytes"])
            self.assertIn('"a":2', records[0]["text"])

    def test_yaml_records_preserve_indentation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "workflow.yml"
            path.write_text(
                "name: ci\non:\n  push:\n    branches: [main]\n---\nname: cd\n",
                encoding="utf-8",
            )
            records = list(structured.iter_records(path))
            self.assertEqual(2, len(records))
            self.assertIn("    branches:", records[0]["text"])

    def test_invalid_json_lines_are_counted_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "broken.jsonl"
            path.write_text(
                json.dumps({"a": 1}) + "\n" + "{not json\n", encoding="utf-8"
            )
            records = list(structured.iter_records(path))
            self.assertEqual(2, len(records))
            self.assertFalse(records[1]["valid"])


class StructuredAdmitTests(unittest.TestCase):
    def test_record_level_dedup_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "records.jsonl"
            source.write_text(
                json.dumps({"id": "Q1", "name": "地球"}) + "\n"
                + json.dumps({"name": "地球", "id": "Q1"}) + "\n"  # dup via canonical
                + json.dumps({"id": "Q2", "name": "月球"}) + "\n",
                encoding="utf-8",
            )
            result = admit_structured(source, root / "admitted")
            self.assertEqual("mei-51m-admitted-natural-source-v2", result["schema"])
            self.assertEqual("record", result["dedup_mode"])
            self.assertEqual(2, result["documents"])
            self.assertEqual(1, result["duplicates_skipped"])
            self.assertEqual(0.0, result["structure_check"]["invalid_ratio"])
            rows = [
                json.loads(line)
                for line in (root / "admitted/documents.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertTrue(all("record_sha256" in row for row in rows))
            self.assertTrue(all("normalized_sha256" not in row for row in rows))
            receipt = quality.audit_source(root / "admitted/manifest.json")
            self.assertEqual("passed", receipt["status"])

    def test_structure_check_fails_closed_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "broken.jsonl"
            source.write_text(
                json.dumps({"id": "Q1"}) + "\n" + "{broken\n" + "{broken2\n",
                encoding="utf-8",
            )
            out = root / "blocked"
            with self.assertRaisesRegex(sources.SourceError, "structure check failed"):
                admit_structured(source, out)
            self.assertFalse(out.exists())

    def test_threshold_can_only_tighten(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "records.jsonl"
            source.write_text(json.dumps({"id": "Q1"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(sources.SourceError, "only tighten"):
                admit_structured(
                    source, root / "blocked", max_invalid_ratio=0.5
                )

    def test_structured_mode_requires_source_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "records.jsonl"
            source.write_text(json.dumps({"id": "Q1"}) + "\n", encoding="utf-8")
            with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
                with self.assertRaisesRegex(
                    sources.SourceError, "requires --source-id"
                ):
                    sources.admit(
                        [source],
                        root / "blocked",
                        role="structured",
                        license_id="CC0-1.0",
                        license_reviewed=True,
                        seen_ledger=None,
                        mode="structured",
                    )

    def test_xml_requires_record_element(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "dblp.xml"
            source.write_text("<article>t</article>\n", encoding="utf-8")
            with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
                with self.assertRaisesRegex(sources.SourceError, "record_element"):
                    sources.admit(
                        [source],
                        root / "blocked",
                        role="structured",
                        license_id="CC0-1.0",
                        license_reviewed=True,
                        seen_ledger=None,
                        source_id="wikidata-json",
                        mode="structured",
                    )

    def test_seen_ledger_reuse_works_with_record_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "records.jsonl"
            source.write_text(
                json.dumps({"id": "Q1", "name": "地球"}) + "\n", encoding="utf-8"
            )
            first = admit_structured(source, root / "first")
            first_row = json.loads(
                (root / "first/documents.jsonl").read_text(encoding="utf-8")
            )
            ledger = root / "ledger.jsonl"
            ledger.write_text(
                json.dumps(
                    {"record_sha256": first_row["record_sha256"]}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(sources.SourceError, "no unseen documents"):
                admit_structured(source, root / "second", seen_ledger=ledger)
            self.assertEqual(1, first["documents"])


if __name__ == "__main__":
    unittest.main()
