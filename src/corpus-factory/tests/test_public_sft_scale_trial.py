import unittest

from public_sft_scale_trial import convert_group, moss_calls, sample_records


class CharTokenizer:
    def encode(self, value):
        return list(map(ord,value))
    def decode(self, ids):
        return ''.join(map(chr,ids))


class ScaleTrialTests(unittest.TestCase):
    def test_sampling_order_and_replay(self):
        records=[({'id':i},{'text':str(i)}) for i in range(50)]
        a,n=sample_records(iter(records),10,7)
        b,m=sample_records(iter(reversed(records)),10,7)
        self.assertEqual((a,n),(b,m))

    def test_moss_rejects_statements_and_duplicate_parameters(self):
        for text in ['x=1; Search("a")','Search("a",query="b")',
                     'Search(query="a",query="b")','Search(secret())']:
            with self.assertRaises((ValueError,SyntaxError)):
                moss_calls(text)

    def test_multiple_calls_and_prior_results_are_preserved_without_future_results(self):
        row={'system':'Here is a list of functions in JSON format that you can invoke: [{"name":"find","parameters":{"type":"dict","properties":{"q":{"type":"string"}},"required":["q"]}}]',
             'conversations':[{'from':'user','value':'查询甲、乙'},
               {'from':'assistant','value':'[find(q="甲"),find(q="乙")]'},
               {'from':'tool','value':'历史结果 abc'},
               {'from':'user','value':'再查询abc'},
               {'from':'assistant','value':'[find(q="abc")]'},
               {'from':'tool','value':'未来结果 xyz'}]}
        group=convert_group('ToolACE',{'row_index':1},row,CharTokenizer())
        first,second=group['views']
        self.assertEqual(len(first['calls']),2)
        self.assertEqual(second['behavior']['arguments_with_prior_result_literal'],1)
        self.assertFalse(second['behavior']['verified_result_dependency'])
        self.assertIn('历史结果 abc',str(second['input']))
        self.assertNotIn('未来结果 xyz',str(second['input']))
        self.assertEqual(second['heads']['confidence']['status'],'pending_model')
        self.assertEqual(sum(second['production_label_masks'].values()),0)

    def test_no_call_is_not_refusal_and_bad_schema_is_not_gold(self):
        row={'system':'Here is a list of functions in JSON format that you can invoke: [{"name":"f","parameters":{"type":"dict","properties":{"n":{"type":"integer"}},"required":["n"]}}]',
             'conversations':[{'from':'user','value':'试试'},
                 {'from':'assistant','value':'[]'},
                 {'from':'assistant','value':'[f(n="wrong")]'}]}
        group=convert_group('ToolACE',{'row_index':1},row,CharTokenizer())
        self.assertIsNone(group['views'][0]['heads']['disposition']['action'])
        self.assertTrue(group['views'][1]['schema_errors'])
        self.assertNotEqual(group['views'][1]['heads']['lm']['status'],'schema_checked_semantics_pending')

    def test_turn_ten_does_not_precede_turn_two_after_json_roundtrip(self):
        from public_sft_scale_trial import events_for
        row={'chat':{'turn_10':{'Human':'ten'},'turn_2':{'Human':'two'},'turn_1':{'Human':'one'}}}
        _,events=events_for('MOSS',row)
        self.assertEqual([e['content'] for e in events],['one','two','ten'])


if __name__=='__main__':
    unittest.main()
