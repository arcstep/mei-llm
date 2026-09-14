import sys
import unittest
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sources'))
from nemotron_cpt import convert, check_value, strict_loads


def fixture():
    return {'tools':[{'type':'function','function':{'name':'read','strict':True,'parameters':{
        'type':'object','properties':{'id':{'type':'string'}},'required':['id']}}}],
        'messages':[{'role':'user','content':'Read 001'},
            {'role':'assistant','tool_calls':[{'id':'a','function':{'name':'read','arguments':'{"id":"001"}'}}],
             'reasoning_content':'PRIVATE_TEACHER_REASONING'},
            {'role':'tool','content':{'ok':False,'value':None,'n':-1.25}}]}


class CPTTests(unittest.TestCase):
    def test_structured_result_strict_and_values_preserved(self):
        row=fixture(); original=deepcopy(row)
        text,_,meta=convert(row)
        self.assertFalse(meta['issues']); self.assertEqual(row,original)
        self.assertIn('"ok":false,"value":null,"n":-1.25',text)
        self.assertIn('001',text); self.assertNotIn('PRIVATE_TEACHER_REASONING',text)
        self.assertNotIn('tool_call_id',text)
        self.assertEqual(meta['result_links'][0]['basis'],'single_pending_inferred')

    def test_parallel_explicit_ids_and_ambiguous_missing_ids(self):
        r=fixture();r['messages'][1]['tool_calls'].append({'id':'b','function':{'name':'read','arguments':{'id':'002'}}})
        r['messages'][2]['tool_call_id']='b'
        r['messages'].append({'role':'tool','tool_call_id':'a','content':'ok'})
        self.assertFalse(convert(r)[2]['issues'])
        del r['messages'][2]['tool_call_id']; del r['messages'][3]['tool_call_id']
        self.assertIn('unresolved_result_association',convert(r)[2]['issues'])

    def test_wrong_arguments_go_to_review_not_repaired(self):
        r=fixture();r['messages'][1]['tool_calls'][0]['function']['arguments']={}
        text,_,m=convert(r)
        self.assertTrue(any('missing_required' in x for x in m['issues']))
        self.assertIn('"arguments":{}',text)

    def test_unused_conflict_does_not_invalidate_valid_called_tool(self):
        r=fixture()
        r['tools'] += [{'name':'unused','description':'one'},{'name':'unused','description':'two'}]
        self.assertFalse(convert(r)[2]['issues'])
        r['tools'].append({'name':'read','parameters':{}})
        self.assertIn('conflicting_called_tool:read',convert(r)[2]['issues'])

    def test_open_call_and_no_call_context_not_forced_refuse(self):
        r=fixture();r['messages'].pop()
        self.assertFalse(convert(r)[2]['issues'])
        r['messages'].pop()
        text,_,m=convert(r)
        self.assertFalse(m['issues']); self.assertEqual(m['features']['no_call_context'],1)
        self.assertNotIn('assistant',text)

    def test_unknown_constraints_and_duplicate_json_keys(self):
        self.assertTrue(check_value('aaa',{'type':'string','pattern':'^b'})[1])
        self.assertTrue(check_value(True,{'enum':[1]})[0])
        with self.assertRaises(ValueError): strict_loads('{"x":1,"x":2}')

    def test_json_enum_object_order_and_numeric_equality(self):
        self.assertFalse(check_value({'b':2,'a':1},{'enum':[{'a':1,'b':2}]})[0])
        self.assertFalse(check_value(1.0,{'type':'integer','enum':[1]})[0])
