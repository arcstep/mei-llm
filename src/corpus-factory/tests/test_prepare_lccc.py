import importlib.util
import unittest
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "generators/prepare_lccc.py"
spec = importlib.util.spec_from_file_location("prepare_lccc", module_path)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PrepareLcccTests(unittest.TestCase):
    def test_chinese_detokenization_preserves_english_word_boundaries(self):
        self.assertEqual(prepare.clean_turn("我 用 New York 的 3 1 个 示例 。"), "我用New York的31个示例。")

    def test_normalized_group_detects_punctuation_and_spacing_variants(self):
        self.assertEqual(prepare.fingerprint("今天 天气 好！"), prepare.fingerprint("今天天气好。"))

    def test_filters_repeated_turns_and_urls(self):
        self.assertEqual(prepare.quality_reason(["今天吃什么？", "今天吃什么！"]), "repeated_turn")
        self.assertIsNotNone(prepare.quality_reason(["这个网站怎么样啊", "请访问https://example.com查看详细介绍"]))
