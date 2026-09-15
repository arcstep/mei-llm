from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sqlite3
from unittest.mock import patch


PATH = Path(__file__).parents[1] / "sources/v12_inputs_freeze.py"
SPEC = importlib.util.spec_from_file_location("v12_inputs_freeze", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class V12InputFreezeTest(unittest.TestCase):
    def test_rounded_stages_preserve_capacity_and_report_rounding(self):
        capacity = {"large":2_400_000_001,"small":150_204_073}
        stages = MODULE.rounded_stage_quotas(capacity,[800_000_000,800_000_000,900_000_000])
        self.assertEqual([sum(v[p] for v in stages.values()) for p in (1,2,3)],
                         [800_000_000,800_000_000,900_001_792])
        for source, phases in stages.items():
            self.assertLessEqual(sum(phases.values()),capacity[source]-1)
            self.assertTrue(all(n%2048==0 for n in phases.values()))
        with self.assertRaises(ValueError):
            MODULE.rounded_stage_quotas({"short":2049},[2048,2048,2048])

    def test_explicit_language_quota_does_not_backfill_with_reserved_script(self):
        db = sqlite3.connect(":memory:")
        MODULE._schema(db)
        for index, language in enumerate(("zh_hans", "en", "zh_hant")):
            db.execute("INSERT INTO records(file_index,row_number,source,domain,language,group_key,split,phase,priority,text_sha,token_offset,token_length,known_eval_overlap) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (0,index,language,"foundation",language,language,"train",1,str(index),str(index),0,1,0))
        db.commit()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "plan.json").write_text(json.dumps({"language_weights":{"foundation":{"zh_hans":8,"en":1}},
                "phases":[{"phase":1,"domain_target_tokens":{"foundation":9}}]}))
            with patch.object(MODULE, "ROOT", path):
                MODULE.select_records({"corpus_plan":"plan.json"}, db)
        selected = db.execute("SELECT language FROM records JOIN selected ON records.id=selected.record_id").fetchall()
        self.assertEqual(set(selected), {("zh_hans",), ("en",)})
        db.close()

    def test_group_split_is_stable_and_related_items_stay_together(self):
        row = {"group_id": "conversation-1", "metadata": {}}
        group = MODULE.group_key(row, "lccc-dialogue", "a" * 64)
        self.assertEqual(group, "lccc-dialogue:conversation-1")
        self.assertEqual(MODULE.split_phase(group, set(), set()), MODULE.split_phase(group, set(), set()))

    def test_gutenberg_chapter_ranges_share_work(self):
        a = {"metadata": {"title": "粉妝樓1-10回"}, "origin": {}}
        b = {"metadata": {"title": "粉妝樓11-20回"}, "origin": {}}
        self.assertEqual(MODULE.group_key(a, "project-gutenberg", "a"), MODULE.group_key(b, "project-gutenberg", "b"))

    def test_allocate_never_exceeds_capacity(self):
        got = MODULE._allocate({"a": 3, "b": 7}, 8)
        self.assertEqual(sum(got.values()), 8)
        self.assertLessEqual(got["a"], 3)
        self.assertLessEqual(got["b"], 7)

    def test_invalid_utf8_fails_closed(self):
        with self.assertRaises(ValueError):
            MODULE._strict_row(b'{"text":"\xff"}\n', Path("bad.jsonl"), 1)

    def test_known_eval_overlap_is_exact_normalized_line(self):
        protected = {MODULE.hashlib.sha256("请查询北京市今日天气".encode()).hexdigest()}
        self.assertTrue(MODULE._known_eval_overlap("header\n请查询北京市今日天气\ntail", protected))
        self.assertFalse(MODULE._known_eval_overlap("请查询上海天气", protected))


if __name__ == "__main__":
    unittest.main()
