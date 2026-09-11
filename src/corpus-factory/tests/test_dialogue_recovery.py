import importlib.util
import unittest
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "quality/dialogue_recovery.py"
spec = importlib.util.spec_from_file_location("dialogue_recovery", module_path)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class DialogueRecoveryTests(unittest.TestCase):
    def test_old_colon_prefix_and_tool_citation_do_not_hide_overlap(self):
        self.assertEqual(recovery.text_hash(": 你好 世界"),
                         recovery.text_hash("<|MOSS|>: 你好  世界<sup><|1|></sup><eom>"))

    def test_non_chinese_turns_are_not_counted_as_retained(self):
        row = {"chat": {"turn_1": {"Human": "hello", "MOSS": "你好"}}}
        self.assertEqual(list(recovery.role_turns(row)), [("MOSS", "你好")])
