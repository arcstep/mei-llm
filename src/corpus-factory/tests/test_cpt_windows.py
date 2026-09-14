import sys
import json
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sources'))
from cpt_windows import split_record

class Characters:
    def encode_document(self,t): return [0]+list(t)+[1]
    def encode(self,t): return list(t)


def record(messages, tools=None):
    text='Tools: '+json.dumps(tools or [],ensure_ascii=False,separators=(',',':'))+'\n'+ '\n'.join(json.dumps(m,ensure_ascii=False,separators=(',',':')) for m in messages)
    return {'text':text,'text_sha256':'test','source_id':'test','group_id':'same-family','origin':{},'split':'candidate-unassigned','metadata':{}}


class WindowTests(unittest.TestCase):
    def check(self,row,limit=256):
        ws=list(split_record(row,Characters(),limit,32,64))
        self.assertTrue(ws)
        self.assertEqual(''.join(row['text'][w['metadata']['primary_span'][0]:w['metadata']['primary_span'][1]] for w in ws),row['text'])
        for w in ws:
            self.assertLessEqual(w['metadata']['tokens'],limit)
            self.assertFalse(w['metadata']['sft_eligible'])
            self.assertEqual(w['group_id'],'same-family')
            for c in w['metadata']['context_spans']:
                self.assertLessEqual(c['span'][1],w['metadata']['primary_span'][0])
        return ws

    def test_long_unicode_string_no_loss_and_partial_labels(self):
        r=record([{'role':'tool','content':{'id':'001','value':'汉字😀\\"'*200}}])
        ws=self.check(r)
        self.assertTrue(any(w['metadata']['partial_message'] for w in ws))
        for w in ws:
            if w['metadata']['partial_message']:
                self.assertIn('Source continuation (partial message)',w['text'])

    def test_complete_short_dialogue_kept_together(self):
        r=record([{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello'}])
        ws=self.check(r)
        self.assertEqual(len(ws),1);self.assertEqual(ws[0]['text'],r['text'])

    def test_exact_context_or_explicit_uncovered_no_fabrication(self):
        tools=[{'name':'f','description':'too long '*100}]
        r=record([{'role':'user','content':'Check 001'}, {'role':'assistant','content':'x'*400},
                  {'role':'assistant','tool_calls':[{'function':{'name':'f','arguments':{'id':'001'}}}]}],tools)
        ws=self.check(r)
        self.assertTrue(any(w['metadata']['uncovered_context'] for w in ws))
        self.assertTrue(any(w['metadata']['context_spans'] for w in ws))

    def test_invalid_budget_fails(self):
        with self.assertRaises(ValueError): list(split_record(record([]),Characters(),100,90,50))

    def test_overlapping_windows_do_not_double_count_primary_spans(self):
        r=record([{'role':'user','content':'a'*i} for i in range(1,20)])
        self.check(r)
