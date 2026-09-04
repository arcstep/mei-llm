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
