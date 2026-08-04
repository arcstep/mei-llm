from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SKILL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from corpus_stats import stats  # noqa: E402
from family_snapshot import snapshot  # noqa: E402


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class CorpusScriptTests(unittest.TestCase):
    def test_stats_and_family_snapshot_use_explicit_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            sample = {
                "sample_id": "sample.metric.v1",
                "family_id": "sample.metric",
                "route_mode": "build",
                "task_type": "build.metric.scalar",
                "source_kind": "example",
                "truth_status": "verified",
            }
            manifest = {"family_id": "sample.metric", "split": "train"}
            write(corpus / "samples" / "build.jsonl", json.dumps(sample) + "\n")
            write(corpus / "manifests" / "families.jsonl", json.dumps(manifest) + "\n")
            write(
                corpus / "eval" / "train.index.json",
                json.dumps({"family_ids": ["sample.metric"], "sample_ids": ["sample.metric.v1"]}),
            )
            self.assertEqual(1, stats(corpus)["total_samples"])
            result = snapshot(corpus, "sample.metric")
            self.assertEqual(1, result["sample_count"])
            self.assertEqual(manifest, result["family"])

    def test_public_skill_has_no_monorepo_docs_or_absolute_siblings(self) -> None:
        forbidden = (
            "docs/" + "mei-llm/",
            "docs/" + "mei-lang/",
            "../" + "docs/",
            "/" + "Users/",
            "workspaces/" + "ws-",
        )
        for path in SKILL_DIR.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".py", ".sh"}:
                text = path.read_text(encoding="utf-8")
                for value in forbidden:
                    self.assertNotIn(value, text, str(path))


if __name__ == "__main__":
    unittest.main()
