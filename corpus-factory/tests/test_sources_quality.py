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
quality = load_module("mei_51m_corpus_quality", "corpus-factory/quality/audit.py")


class FakeTokenizer:
    model_sha256 = "a" * 64
    tokenizer_id = "zh-24k-v1"

    def encode_document(self, text: str) -> list[int]:
        return [2, *[ord(character) % 1000 for character in text], 1]


class SourceManagerTests(unittest.TestCase):
    def test_mix_blocks_when_pool_is_short(self) -> None:
        result = sources.plan_mix(
            300,
            {"wiki": 200, "fineweb2_hq": 100},
            {"wiki": 0, "fineweb2_hq": 0},
            hq_fraction=0.65,
        )
        self.assertEqual("blocked_insufficient_pool", result["status"])
        self.assertEqual({"fineweb2_hq": 95}, result["shortages"])
        self.assertFalse(result["allow_repeat"])

    def test_download_requires_exact_batch_authorization_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(sources.SourceError, "download refused"):
                sources.download_hq(
                    ["data/cmn_Hani/train-000.parquet"],
                    Path(raw),
                    "not-authorized",
                )
            with self.assertRaisesRegex(sources.SourceError, "must belong"):
                sources.download_hq(
                    ["data/not-cmn_Hani/train-000.parquet"],
                    Path(raw),
                    sources.DOWNLOAD_AUTHORIZATION,
                )

    def test_admission_requires_license_and_deduplicates_documents(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"text": "第一篇高质量文档"}, ensure_ascii=False),
                        json.dumps({"text": "第一篇高质量文档  "}, ensure_ascii=False),
                        json.dumps({"text": "第二篇高质量文档"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(sources.SourceError, "license"):
                sources.admit(
                    [source],
                    root / "blocked",
                    role="fineweb2_hq",
                    license_id="ODC-By-1.0",
                    license_reviewed=False,
                    seen_ledger=None,
                )
            with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
                result = sources.admit(
                    [source],
                    root / "admitted",
                    role="fineweb2_hq",
                    license_id="ODC-By-1.0",
                    license_reviewed=True,
                    seen_ledger=None,
                )
            self.assertEqual(2, result["documents"])
            self.assertEqual(1, result["duplicates_skipped"])
            receipt = quality.audit_source(root / "admitted/manifest.json")
            self.assertEqual("passed", receipt["status"])

    def test_pool_release_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            admitted = root / "admitted"
            admitted.mkdir()
            (admitted / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "mei-51m-admitted-natural-source-v1",
                        "source_role": "wiki",
                        "documents": 2,
                        "tokens": 10,
                    }
                ),
                encoding="utf-8",
            )
            sources.freeze_pool([admitted], root / "pool", "pool-v1")
            with self.assertRaises(FileExistsError):
                sources.freeze_pool([admitted], root / "pool", "pool-v2")


class ProvenanceV2Tests(unittest.TestCase):
    def write_dialogue_source(self, root: Path) -> Path:
        source = root / "dialogue.jsonl"
        source.write_text(
            "\n".join(
                [
                    json.dumps({"text": "把客厅的灯打开"}, ensure_ascii=False),
                    json.dumps({"text": "帮我查一下明天北京的天气"}, ensure_ascii=False),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return source

    def test_admit_v2_records_provenance_and_passes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self.write_dialogue_source(root)
            clearance = root / "clearance.json"
            clearance.write_text(json.dumps({"reviewed": "2026-09-04"}), encoding="utf-8")
            with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
                result = sources.admit(
                    [source],
                    root / "admitted",
                    role="dialogue",
                    license_id="subtitle-rights-uncleared",
                    license_reviewed=True,
                    seen_ledger=None,
                    source_id="opensubtitles-zh",
                    source_url="https://opus.nlpl.eu/OpenSubtitles.php",
                    dataset_version="v2018",
                    clearance_receipt=clearance,
                )
            self.assertEqual("mei-51m-admitted-natural-source-v2", result["schema"])
            self.assertEqual(2, result["provenance_version"])
            self.assertEqual("opensubtitles-zh", result["source_id"])
            self.assertEqual("text", result["dedup_mode"])
            self.assertIsNotNone(result["registry_entry_sha256"])
            self.assertIsNotNone(result["acquired_at"])
            self.assertEqual("subtitle-rights-uncleared", result["license_id"])
            self.assertIsNotNone(result["clearance_receipt"])
            rows = [
                json.loads(line)
                for line in (root / "admitted/documents.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertTrue(all(row["source_id"] == "opensubtitles-zh" for row in rows))
            self.assertTrue(all(row["record_key"] is None for row in rows))
            receipt = quality.audit_source(root / "admitted/manifest.json")
            self.assertEqual("passed", receipt["status"])

    def test_admit_v2_rejects_unknown_or_mismatched_source_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self.write_dialogue_source(root)
            with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
                with self.assertRaisesRegex(sources.SourceError, "unknown source_id"):
                    sources.admit(
                        [source],
                        root / "blocked-1",
                        role="dialogue",
                        license_id="x",
                        license_reviewed=True,
                        seen_ledger=None,
                        source_id="not-a-source",
                    )
                with self.assertRaisesRegex(sources.SourceError, "registered as"):
                    sources.admit(
                        [source],
                        root / "blocked-2",
                        role="dialogue",
                        license_id="x",
                        license_reviewed=True,
                        seen_ledger=None,
                        source_id="wikidata-json",
                    )

    def test_admit_v2_requires_clearance_receipt_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self.write_dialogue_source(root)
            with self.assertRaisesRegex(sources.SourceError, "missing clearance receipt"):
                sources.admit(
                    [source],
                    root / "blocked",
                    role="dialogue",
                    license_id="x",
                    license_reviewed=True,
                    seen_ledger=None,
                    source_id="opensubtitles-zh",
                    clearance_receipt=root / "does-not-exist.json",
                )
            self.assertFalse((root / "blocked").exists())

    def test_admit_appends_to_seen_ledger_and_missing_ledger_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ledger = root / "ledger.jsonl"
            source = self.write_dialogue_source(root)
            with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
                first = sources.admit(
                    [source],
                    root / "admitted-1",
                    role="dialogue",
                    license_id="x",
                    license_reviewed=True,
                    seen_ledger=ledger,
                    source_id="opensubtitles-zh",
                )
            self.assertEqual(2, first["documents"])
            self.assertEqual(2, len(ledger.read_text(encoding="utf-8").splitlines()))
            # A second admit of the same raw sees the ledger and finds no unseen docs.
            with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
                with self.assertRaisesRegex(sources.SourceError, "no unseen documents"):
                    sources.admit(
                        [source],
                        root / "admitted-2",
                        role="dialogue",
                        license_id="x",
                        license_reviewed=True,
                        seen_ledger=ledger,
                        source_id="opensubtitles-zh",
                    )

    def test_admit_without_source_id_keeps_v1_schema(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "wiki.jsonl"
            source.write_text(
                json.dumps({"text": "维基百科条目正文"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
                result = sources.admit(
                    [source],
                    root / "admitted",
                    role="wiki",
                    license_id="CC BY-SA 3.0",
                    license_reviewed=True,
                    seen_ledger=None,
                )
            self.assertEqual("mei-51m-admitted-natural-source-v1", result["schema"])
            receipt = quality.audit_source(root / "admitted/manifest.json")
            self.assertEqual("passed", receipt["status"])

    def test_audit_blocks_v2_manifest_without_source_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "tokens.bin").write_bytes(b"\x00")
            documents = root / "documents.jsonl"
            documents.write_text("{}\n", encoding="utf-8")
            manifest = {
                "schema": "mei-51m-admitted-natural-source-v2",
                "provenance_version": 2,
                "source_role": "dialogue",
                "license_id": "x",
                "license_reviewed": True,
                "dedup_mode": "text",
                "registry_entry_sha256": "a" * 64,
                "documents": 1,
                "tokens": 1,
                "artifacts": {
                    "documents.jsonl": quality.sha256_file(documents),
                    "tokens.bin": quality.sha256_file(root / "tokens.bin"),
                },
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            receipt = quality.audit_source(root / "manifest.json")
            self.assertEqual("blocked", receipt["status"])
            self.assertIn("source_id missing", receipt["errors"])


class CorpusQualityTests(unittest.TestCase):
    def write_review(self, root: Path) -> Path:
        path = root / "review.json"
        path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "semantic_consistency": "passed",
                    "reviewer": "fixture-reviewer",
                    "sample_size": 20,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_unique_ids_do_not_hide_template_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            corpus = root / "synthetic.jsonl"
            corpus.write_text(
                "".join(
                    json.dumps({"text": f"请查询订单编号 {index}"}, ensure_ascii=False)
                    + "\n"
                    for index in range(30)
                ),
                encoding="utf-8",
            )
            result = quality.audit_synthetic(
                [corpus],
                self.write_review(root),
                eval_markers=None,
                min_template_ratio=0.2,
            )
            self.assertEqual("blocked", result["status"])
            self.assertEqual(1.0, result["exact_unique_ratio"])
            self.assertIn("template_diversity_degraded", result["reasons"])

    def test_semantic_review_and_leakage_are_hard_gates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            corpus = root / "synthetic.jsonl"
            corpus.write_text(
                "".join(
                    json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    for text in (
                        "查询北京天气",
                        "把摄氏温度换成华氏",
                        "检查明天的航班",
                        "取消刚才的预约",
                    )
                ),
                encoding="utf-8",
            )
            markers = root / "markers.txt"
            markers.write_text("刚才的预约\n", encoding="utf-8")
            result = quality.audit_synthetic(
                [corpus],
                self.write_review(root),
                eval_markers=markers,
                min_template_ratio=0.5,
            )
            self.assertEqual("blocked", result["status"])
            self.assertIn("eval_leakage", result["reasons"])

    def test_reuse_decision_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            passed = root / "passed.json"
            blocked = root / "blocked.json"
            passed.write_text(
                json.dumps({"status": "passed", "corpus_reuse_eligible": True}),
                encoding="utf-8",
            )
            blocked.write_text(
                json.dumps({"status": "blocked", "corpus_reuse_eligible": False}),
                encoding="utf-8",
            )
            result = quality.decide_reuse([passed, blocked], "candidate-v1")
            self.assertEqual("retire", result["action"])
            self.assertFalse(result["corpus_reuse_eligible"])


if __name__ == "__main__":
    unittest.main()
