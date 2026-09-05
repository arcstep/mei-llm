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


class FakeTokenizer:
    model_sha256 = "a" * 64
    tokenizer_id = "zh-24k-v3"

    def encode_document(self, text: str) -> list[int]:
        return [2, 3, 1]


class PointerTests(unittest.TestCase):
    def test_real_pointer_is_frozen_v2(self) -> None:
        pointer = sources.tokenizer_pointer()
        self.assertEqual("mei-51m-tokenizer-pointer-v1", pointer["schema"])
        self.assertEqual("frozen", pointer["status"])
        self.assertEqual("zh-24k-v3", pointer["tokenizer_id"])
        self.assertEqual("zh-32k-v2", pointer["supersedes"])

    def test_missing_pointer_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "TOKENIZER.json"
            with self.assertRaisesRegex(sources.SourceError, "pointer missing"):
                sources.tokenizer_pointer(missing)

    def test_draft_pointer_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            draft = Path(raw) / "TOKENIZER.json"
            draft.write_text(
                json.dumps(
                    {
                        "schema": "mei-51m-tokenizer-pointer-v1",
                        "tokenizer_id": "zh-32k-v2",
                        "status": "draft",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(sources.SourceError, "not frozen"):
                sources.tokenizer_pointer(draft)

    def test_load_tokenizer_returns_frozen_v3_with_id(self) -> None:
        tokenizer = sources.load_tokenizer()
        self.assertEqual("zh-24k-v3", tokenizer.tokenizer_id)
        self.assertEqual(24000, tokenizer.vocab_size)


class ExpectedTokenizerBindingTests(unittest.TestCase):
    def write_source(self, root: Path) -> Path:
        source = root / "source.jsonl"
        source.write_text(
            json.dumps({"text": "一篇文档"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return source

    def test_mismatched_expected_tokenizer_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
                with self.assertRaisesRegex(sources.SourceError, "expected tokenizer"):
                    sources.admit(
                        [self.write_source(root)],
                        root / "blocked",
                        role="dialogue",
                        license_id="x",
                        license_reviewed=True,
                        seen_ledger=None,
                        source_id="opensubtitles-zh",
                        expected_tokenizer_id="zh-32k-v2",
                    )

    def test_matching_expected_tokenizer_admits(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with patch.object(sources, "load_tokenizer", return_value=FakeTokenizer()):
                result = sources.admit(
                    [self.write_source(root)],
                    root / "admitted",
                    role="dialogue",
                    license_id="x",
                    license_reviewed=True,
                    seen_ledger=None,
                    source_id="opensubtitles-zh",
                    expected_tokenizer_id="zh-24k-v3",
                )
            self.assertEqual("zh-24k-v3", result["tokenizer_id"])


if __name__ == "__main__":
    unittest.main()
