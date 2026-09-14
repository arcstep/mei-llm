from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).parents[1] / "sources/v12_inputs_freeze.py"
SPEC = importlib.util.spec_from_file_location("v12_inputs_freeze", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class V12InputFreezeTest(unittest.TestCase):
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
