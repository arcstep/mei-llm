import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sources'))
from qualified_sft_specimens import (evidence_check, make_case, schema_check,
    validate_case, verify_result)


class TinyTokenizer:
    def __init__(self):
        self.chars = {}
    def encode(self, text):
        return [self.chars.setdefault(c, len(self.chars)+4) for c in text]
    def decode(self, ids):
        inverse = {v:k for k,v in self.chars.items()}
        return ''.join(inverse[i] for i in ids)


def specimen():
    row = {'source':'toolace', 'association_group':'test:one',
           'messages':[{'role':'user','content':'Find quotes about hope'}],
           'gold_lm_target':[{'name':'quotes', 'arguments':{'word':'hope'}}],
           'visible_tools':[{'name':'quotes', 'description':'find quotes by keyword',
             'parameters':{'type':'object', 'properties':{'word':{'type':'string'}},
                           'required':['word'], 'additionalProperties':False}}]}
    spec = {'id':'test-one', 'review':'keyword is hope, not the topic of another task',
            'evidence':[{'call_index':0,'argument_path':['word'],'value':'hope',
                         'quotes':[{'event_id':'query','text':'hope'}], 'reason':'exact keyword'}]}
    return make_case(spec, row, {}, TinyTokenizer())


class SpecimenTests(unittest.TestCase):
    def test_valid_and_language(self):
        case = specimen()
        validate_case(case)
        self.assertEqual(case['language'], 'en')
        self.assertEqual(case['heads']['lm']['semantic_label_mask'], 1)
        self.assertEqual(case['heads']['confidence']['semantic_label_mask'], 0)

    def test_target_swap_invalidates_review(self):
        case = specimen()
        case['heads']['lm']['target'][0]['arguments']['word'] = 'fear'
        with self.assertRaisesRegex(ValueError, 'binding'):
            validate_case(case)

    def test_future_evidence_rejected(self):
        case = specimen()
        case['events']['query']['phase'] = 'after_call'
        with self.assertRaisesRegex(ValueError, 'future'):
            validate_case(case)

    def test_missing_evidence_rejected(self):
        case = specimen()
        case['parameter_evidence'] = []
        with self.assertRaisesRegex(ValueError, 'uncovered'):
            validate_case(case)

    def test_parameter_role_swap_rejected(self):
        call = [{'name':'range','arguments':{'min':1,'max':100}}]
        ev = [{'call_index':0,'argument_path':['min'],'value':100,'quotes':[],'reason':'swapped'}]
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            evidence_check(call, ev, {})

    def test_schema_types_and_unknown_assertions(self):
        for value, schema in [(True, {'type':'integer'}),
                              (1, {'type':'number','multipleOf':2}),
                              ({}, {'type':'object','required':['word']})]:
            with self.assertRaises(ValueError):
                schema_check(value, schema)

    def test_result_must_match_constraint(self):
        call = {'arguments':{'constraints':{'星级':'1'},'requested_fields':['房费']}}
        verify_result(call, [{'星级':'1','房费':'70元'}])
        with self.assertRaises(ValueError):
            verify_result(call, [{'星级':'5','房费':'70元'}])

    def test_prompt_not_in_loss(self):
        case = specimen()
        case['encoding']['loss_mask'][0] = 1
        with self.assertRaisesRegex(ValueError, 'mask'):
            validate_case(case)

    def test_static_confidence_forbidden(self):
        case = specimen()
        case['heads']['confidence']['semantic_label_mask'] = 1
        with self.assertRaisesRegex(ValueError, 'confidence'):
            validate_case(case)

    def test_valid_mask_contains_eos(self):
        case = specimen()
        e = case['encoding']
        self.assertEqual(sum(e['loss_mask']), e['target_tokens']+1)
        self.assertEqual(e['input_ids'][-1], 1)


if __name__ == '__main__':
    unittest.main()
