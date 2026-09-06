from __future__ import annotations

import unittest
from pathlib import Path

from mei_sdk import SdkError
from mei_sdk.ffi import NativeEngine


SDK_ROOT = Path(__file__).resolve().parents[2] / "_shared"
FIXTURE = SDK_ROOT / "fixtures" / "packages" / "tiny-protocol-v1"


class NativeFfiV2Test(unittest.TestCase):
    def _engine(self) -> NativeEngine:
        try:
            return NativeEngine()
        except SdkError as exc:
            if exc.id in {"engine_unavailable", "abi_version_mismatch"}:
                self.skipTest(str(exc))
            raise

    def test_stepwise_native_session(self) -> None:
        engine = self._engine()
        self.assertEqual(engine.abi_version(), 2)
        self.assertEqual(engine.version()["wire_version"], "mei-runtime-wire-v2")
        engine.open(str(FIXTURE))
        tool = {
            "name": "clock.read",
            "description": "clock",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
        try:
            self.assertEqual(engine.register_tools([tool])["registered"], 1)
            with self.assertRaises(SdkError) as one_shot:
                engine.complete_json(
                    {
                        "wire_version": "mei-runtime-wire-v2",
                        "query": "time",
                        "oracle_tools": [tool],
                        "candidate_text": '[{"name":"clock.read","arguments":{}}]',
                        "confidence": {"value": 1.0, "source": "protocol-test"},
                    }
                )
            self.assertEqual(one_shot.exception.id, "tool_result_required")
            with engine.create_session() as session:
                turn = session.complete(
                    {
                        "wire_version": "mei-runtime-wire-v2",
                        "query": "time",
                        "oracle_tools": [tool],
                        "candidate_text": '[{"name":"clock.read","arguments":{}}]',
                        "confidence": {"value": 1.0, "source": "protocol-test"},
                    }
                )
                self.assertEqual(turn["kind"], "call")
                call_id = turn["call"]["call_id"]
                ack = session.submit_tool_result(
                    {
                        "wire_version": "mei-runtime-wire-v2",
                        "call_id": call_id,
                        "status": "ok",
                        "payload": {"hour": 12},
                        "provenance": {"source": "host_executor", "verified": True},
                    }
                )
                self.assertTrue(ack["accepted"])
                terminal = session.complete(
                    {
                        "wire_version": "mei-runtime-wire-v2",
                        "query": "time",
                        "oracle_tools": [tool],
                        "candidate_text": "[]",
                    }
                )
                self.assertEqual(terminal["kind"], "respond")
                narration = session.narrate({"mode": "adapter"})
                self.assertEqual(narration["mode"], "deterministic")
                self.assertEqual(narration["text"], "clock.read已执行完成，hour为12。")
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
