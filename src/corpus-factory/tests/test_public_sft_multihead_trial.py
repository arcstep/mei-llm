"""Unit checks for public multi-head source parsing and deterministic selection."""
import unittest

from public_sft_multihead_trial import parse_python_calls, stable_sample


class PublicSftMultiheadTrialTest(unittest.TestCase):
    def test_moss_positional_call_is_literal(self):
        calls = parse_python_calls('<|Commands|>: Search("飞行汽车 技术进步")<eoc>', {"Search": ["query"]})
        self.assertEqual(calls, [{"name": "Search", "arguments": {"query": "飞行汽车 技术进步"}}])

    def test_api_keyword_and_multiple_calls(self):
        calls = parse_python_calls("API-Request: [Find(city='苏州'), Reserve(id=12)]")
        self.assertEqual(calls[0]["arguments"]["city"], "苏州")
        self.assertEqual(calls[1]["arguments"]["id"], 12)

    def test_executable_expression_is_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            parse_python_calls("Danger(value=get_secret())")

    def test_stable_sample_ignores_input_order(self):
        items = [{"id": str(i)} for i in range(20)]
        a = stable_sample(items, 5, lambda x: x["id"])
        b = stable_sample(list(reversed(items)), 5, lambda x: x["id"])
        self.assertEqual({x["id"] for x in a}, {x["id"] for x in b})


if __name__ == "__main__":
    unittest.main()
