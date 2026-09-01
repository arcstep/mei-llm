from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "sdk/python"))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freeze_agent = load_module("freeze_agent_sft_release_51m", "freeze_agent_sft_release_51m.py")
train_sft = load_module("train_sft_ondisk_51m", "train_sft_ondisk_51m.py")


class AgentSftReleaseTests(unittest.TestCase):
    def test_rows_are_group_isolated_and_have_real_continuations(self):
        rows = freeze_agent.build_rows()
        universe = json.loads(freeze_agent.UNIVERSE_PATH.read_text(encoding="utf-8"))
        receipt = freeze_agent.validate_rows(
            rows, {str(tool["name"]) for tool in universe["tools"]}
        )
        self.assertTrue(receipt["ok"])
        self.assertGreater(receipt["nonempty_tool_result_rows"], 0)
        self.assertGreater(receipt["target_counts"]["respond"], 0)
        logistics = next(
            row
            for row in rows["train"]
            if row["family"] == "agent_logistics" and row["kind"] == "respond"
        )
        self.assertEqual(len(logistics["prior_calls"]), 3)
        self.assertEqual(len(logistics["tool_results"]), 3)

    def test_deployment_prompt_matches_runtime_trusted_history_shape(self):
        row = next(
            row
            for row in freeze_agent.build_rows()["train"]
            if row["family"] == "agent_travel" and row["trajectory_step"] == 2
        )
        request = train_sft.deployment_request(row)
        self.assertEqual(len(request["history"]), 1)
        self.assertEqual(len(request["tool_results"]), 1)
        history_call = json.loads(request["history"][0]["content"])
        self.assertEqual(history_call["name"], "book_flight")
        self.assertEqual(
            history_call["call_id"], request["tool_results"][0]["call_id"]
        )
        self.assertIn("pnr", request["tool_results"][0]["payload"])

    def test_freeze_is_non_destructive_and_hash_complete(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name) / "releases"
            root.mkdir()
            parent = root / freeze_agent.PARENT_ID
            parent.mkdir()
            source_parent = freeze_agent.PARENT_DIR
            for path in source_parent.iterdir():
                if path.is_file():
                    (parent / path.name).write_bytes(path.read_bytes())
            args = type(
                "Args",
                (),
                {
                    "release_root": root,
                    "release_id": "agent-test-v1",
                    "parent_id": freeze_agent.PARENT_ID,
                    "universe": freeze_agent.UNIVERSE_PATH,
                },
            )()
            result = freeze_agent.freeze(args)
            self.assertTrue(result["ok"])
            manifest = json.loads(
                (root / "agent-test-v1/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["parent_release"]["release_id"], freeze_agent.PARENT_ID)
            self.assertTrue((root / freeze_agent.PARENT_ID / "manifest.json").is_file())
            self.assertGreater(
                manifest["outputs"]["full-call.train.jsonl"]["rows"],
                json.loads((parent / "manifest.json").read_text(encoding="utf-8"))["outputs"]["full-call.train.jsonl"]["rows"],
            )


if __name__ == "__main__":
    unittest.main()
