import unittest
from sft_candidate_triage import make_review


def row(calls):
    return {'query': '订单号查重', 'heads': {'lm': {'target': calls, 'status': 'source_backed_candidate'},
            'disposition': {'action': 'complete'}}, 'visible_tools': [], 'source_result': {}, 'source_narration': ''}


class TriageTest(unittest.TestCase):
    def test_old_candidate_cannot_turn_into_approved_gold(self):
        r = make_review(row([]), 'MOSS', 'input', 1)
        self.assertFalse(r['formal_admitted'])
        self.assertTrue(all(h['label_mask'] == 0 for h in r['heads'].values()))
        self.assertEqual(r['heads']['disposition']['status'], 'pending_policy_evidence')

    def test_missing_entity_not_silently_accepted_or_injected(self):
        original = row([{'name': 'lookup', 'arguments': {'selected_entities': ['未来结果']}}])
        r = make_review(original, 'CrossWOZ', 'input', 1)
        self.assertIn('selected_entity_absent_from_prior_visible_text', r['issues'])
        self.assertEqual(original['query'], '订单号查重')
        self.assertEqual(r['selected_entity_findings'][0]['finding'], 'literal_absence_not_final_semantic_judgment')

    def test_tool_discovery_is_not_business_lm_gold(self):
        r = make_review(row([{'name': 'public_apibank_toolsearcher', 'arguments': {}}]), 'API-Bank', 'input', 1)
        self.assertEqual(r['heads']['lm']['status'], 'not_applicable_discovery')
        self.assertEqual(r['heads']['retrieval']['status'], 'discovery_query_candidate')

    def test_multi_call_retained_for_review(self):
        calls = [{'name': 'check', 'arguments': {'column': c}} for c in ['数量', '金额']]
        r = make_review(row(calls), 'ToolACE', 'input', 1)
        self.assertEqual(r['calls'], calls)
        self.assertIn('multiple_calls_require_dependency_review', r['issues'])


if __name__ == '__main__':
    unittest.main()
