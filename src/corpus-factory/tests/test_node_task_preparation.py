"""Contract fixtures protect dependency semantics; these are not model quality tests."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('node_tasks',ROOT/'src/corpus-factory/sources/node_task_preparation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=json.loads((ROOT/'src/corpus-factory/fixtures/node-task-seeds-20260914-v1.json').read_text())
CAT={'check_required':{},'check_unique':{},'check_number':{},'check_enum':{},'check_product':{}}
class TaskContracts(unittest.TestCase):
 def test_all_state_gold(self):
  for c in D['states']:
   with self.subTest(c=c['id']):self.assertEqual(m.evaluate_state(c),c['expected'])
 def test_issue_does_not_release_pass_dependency(self):
  p=copy.deepcopy(D['plans'][2]['plan'])
  self.assertEqual(m.ready(p,{'required':'issues'}),[])
  p['tasks'][1]['gate']='all_succeeded'
  self.assertEqual(m.ready(p,{'required':'issues'}),['unique'])
 def test_parallel_order_is_not_gold_order(self):
  p=copy.deepcopy(D['plans'][1]['plan']);q=copy.deepcopy(p);q['tasks'].reverse()
  self.assertEqual(m.ready(p,{}),m.ready(q,{}))
 def test_cycle_and_unknown_dependency(self):
  p=copy.deepcopy(D['plans'][1]['plan']);p['tasks'][0]['after']=['unique'];p['tasks'][1]['after']=['required']
  with self.assertRaises(AssertionError):m.validate_plan(p,CAT)
  p['tasks'][0]['after']=['missing']
  with self.assertRaises(AssertionError):m.validate_plan(p,CAT)
 def test_untrusted_state_cannot_complete(self):
  c=copy.deepcopy(D['states'][7]);c['states']['product']='unknown'
  self.assertEqual(m.evaluate_state(c)['decision'],'incomplete')
 def test_duplicate_task_and_unsupported_tool(self):
  p=copy.deepcopy(D['plans'][1]['plan']);p['tasks'][1]['id']=p['tasks'][0]['id']
  with self.assertRaises(AssertionError):m.validate_plan(p,CAT)
  p=copy.deepcopy(D['plans'][0]['plan']);p['tasks'][0]['tool']='upload'
  with self.assertRaises(AssertionError):m.validate_plan(p,CAT)
if __name__=='__main__':unittest.main()
