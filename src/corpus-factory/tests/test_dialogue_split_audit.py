import importlib.util
import unittest
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "quality/dialogue_split_audit.py"
spec = importlib.util.spec_from_file_location("dialogue_split_audit", module_path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class DialogueSplitAuditTests(unittest.TestCase):
    def test_spacing_and_punctuation_variants_are_identical(self):
        left = audit.grams("今天 天气 真不错，我们出去走走吧。")
        right = audit.grams("今天天气真不错！我们出去走走吧")
        self.assertEqual(audit.jaccard(left, right), 1.0)

    def test_unrelated_dialogue_is_not_near_duplicate(self):
        self.assertLess(audit.jaccard(audit.grams("今天去公园散步怎么样"), audit.grams("饭做好了可以过来吃饭")), 0.8)
