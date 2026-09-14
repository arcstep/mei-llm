"""Tests for v1.2 task preparation contracts; no model-quality claim."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "node_task_v12", ROOT / "src/corpus-factory/sources/node_task_v12.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class NodeTaskV12Tests(unittest.TestCase):
    def test_engineering_quota_and_groups(self):
        rows = MODULE.engineering_cases()
        self.assertEqual(len(rows), 120)
        self.assertEqual(set(row["behavior"] for row in rows), set(MODULE.BEHAVIORS))
        for behavior in MODULE.BEHAVIORS:
            self.assertEqual(sum(row["behavior"] == behavior for row in rows), 10)
        self.assertTrue(all(row["independent"] is False for row in rows))

    def test_real_node_runtime_replay(self):
        result = MODULE.validate_engineering(MODULE.engineering_cases())
        self.assertEqual(result["cases"], 120)

    def test_plan_runtime_rejects_cycle_and_unverified_narration(self):
        script = r"""
import {validatePlan,narrate} from './src/demos/data-check/node-runtime.mjs';
let cycle=false,narration=false;
try { validatePlan({schema:'mei-node-plan-v1',revision:1,status:'draft',tasks:[
 {id:'a',tool:'check_required',args:{column:'A'},after:['b'],gate:'all_passed',required:true},
 {id:'b',tool:'check_unique',args:{column:'A'},after:['a'],gate:'all_passed',required:true}]}); } catch { cycle=true; }
try { narrate({verified:false,results:[]}); } catch { narration=true; }
process.stdout.write(JSON.stringify({cycle,narration}));
"""
        process = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT,
                                 text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(process.stdout), {"cycle": True, "narration": True})

    def test_every_nonempty_lm_target_is_single_call(self):
        for row in MODULE.engineering_cases():
            target = row["gold_lm_target"]
            self.assertTrue(target == [] or (isinstance(target, dict) and set(target) == {"name", "arguments"}))

    def test_toolace_pilot_compiles_one_call_views(self):
        row = {"text": 'Tools: [{"name":"f","parameters":{"type":"object"}}]\n'
                       'user: do it\nassistant: [{"name":"f","arguments":{}}]\ntool: ok'}
        views = MODULE._toolace_views(row, 8)
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["target"], [{"name": "f", "arguments": {}}])

    def test_nemotron_parallel_call_is_not_flattened(self):
        row = {"text": 'Tools: [{"type":"function","function":{"name":"f"}}]\n'
                       '{"role":"user","content":"do it"}\n'
                       '{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{}}},'
                       '{"function":{"name":"f","arguments":{}}}]}'}
        self.assertEqual(MODULE._nemotron_views(row, 8), [])


if __name__ == "__main__":
    unittest.main()
