"""Protect against target-derived evidence and silent value changes in the pilot."""
import copy
import unittest

from public_five_head_trial import source_evidence
from grammar import dump_calls
from provenance import validate_generated_call


class PublicFiveHeadTrialTest(unittest.TestCase):
    def setUp(self):
        self.tools = [{"name":"verify_code", "parameters":{"type":"object", "required":["code"],
                     "properties":{"code":{"type":"string"}}}}]

    def test_literal_leading_zero_is_preserved(self):
        query = 'Verify code "001234".'
        spans, evidence = source_evidence(query, self.tools)
        self.assertTrue(any(s["value"] == "001234" for s in spans))
        call = [{"name":"verify_code", "arguments":{"code":"001234"}}]
        got = validate_generated_call(dump_calls(call), tools=self.tools,
              request={"query":query,"evidence":evidence}, enforce_confidence=False)
        self.assertTrue(got["ok"])

    def test_fabricated_value_cannot_get_evidence(self):
        query = 'Please verify my code.'
        _, evidence = source_evidence(query, self.tools)
        got = validate_generated_call(dump_calls([{"name":"verify_code", "arguments":{"code":"123456"}}]),
              tools=self.tools, request={"query":query,"evidence":evidence}, enforce_confidence=False)
        self.assertFalse(got["ok"])
        self.assertEqual(got["error"], "provenance_missing")

    def test_slot_candidates_do_not_choose_a_gold_tool(self):
        other = copy.deepcopy(self.tools[0]); other["name"] = "second_tool"
        _, evidence = source_evidence('Code "123456".', self.tools+[other])
        self.assertEqual({e["tool"] for e in evidence if e["value"]=="123456"}, {"verify_code","second_tool"})


if __name__ == "__main__":
    unittest.main()
