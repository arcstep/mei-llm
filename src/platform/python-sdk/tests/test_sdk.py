from __future__ import annotations

import json
import unittest
from pathlib import Path

from mei_sdk import Engine, load_package, parse_v2_text, schema_fingerprint, sdk_versions
from mei_sdk.canonical import dumps_canonical
from mei_sdk.errors import SdkError
from mei_sdk.protocol import leak_markers, normalize_request, render_request
from mei_sdk.version import SDK_ROOT, SPEC_DIR

TINY = SDK_ROOT / "fixtures" / "packages" / "tiny-protocol-v1"
GOLDEN = SPEC_DIR / "golden"


class VersionTests(unittest.TestCase):
    def test_experimental_and_no_needle2(self):
        v = sdk_versions()
        self.assertTrue(v["sdk_semver"].endswith("experimental"))
        blob = json.dumps(v)
        self.assertNotIn("needle", blob.lower())
        self.assertEqual(v["product"], "mei-1.2-51m Runtime")
        self.assertEqual(v["sdk_semver"], "0.2.0-experimental")
        self.assertEqual(v["wire_version"], "mei-runtime-wire-v2")
        self.assertEqual(v["model_package_version"], "mei-model-package-v2")
        self.assertEqual(v["runtime_abi_version"], "mei-runtime-abi-2")
        self.assertEqual(Engine.version(), v)


class ProtocolTests(unittest.TestCase):
    def test_fingerprint_stable(self):
        expected = GOLDEN / "schema_fingerprint.json"
        gold = json.loads(expected.read_text(encoding="utf-8"))
        self.assertEqual(schema_fingerprint(gold["tools"]), gold["sha256"])
        self.assertEqual(dumps_canonical(gold["compact"]), gold["canonical"])

    def test_canonical_json_ignores_nested_object_insertion_order(self):
        left = {
            "工具": {"z": 1, "a": {"后": True, "前": False}},
            "array": [{"b": 2, "a": 1}],
        }
        right = {
            "array": [{"a": 1, "b": 2}],
            "工具": {"a": {"前": False, "后": True}, "z": 1},
        }
        self.assertEqual(dumps_canonical(left), dumps_canonical(right))
        with self.assertRaises(ValueError):
            dumps_canonical({"value": float("nan")})

    def test_canonical_number_golden_and_safe_domain(self):
        golden = json.loads(
            (GOLDEN / "canonical_numbers_v2.json").read_text(encoding="utf-8")
        )
        for case in golden["cases"]:
            self.assertEqual(dumps_canonical(case["input"]), case["canonical"])
        self.assertEqual(dumps_canonical(-0.0), "0")
        self.assertEqual(dumps_canonical(1e-6), "0.000001")
        self.assertEqual(dumps_canonical(1e-7), "1e-7")
        for literal in golden["invalid_json_literals"]:
            with self.assertRaisesRegex(ValueError, "safe domain"):
                normalize_request(
                    json.loads(
                        '{"wire_version":"mei-runtime-wire-v2","query":"x",'
                        '"state":{"unsafe":' + literal + "}}"
                    )
                )

    def test_parse_empty_array_is_refuse(self):
        parsed = parse_v2_text("[]")
        self.assertTrue(parsed["ok"])
        self.assertTrue(parsed["refuse"])
        self.assertEqual(parsed["function_calls"], [])

    def test_parse_one_call(self):
        parsed = parse_v2_text('[{"name":"light.set","arguments":{"on":true}}]')
        self.assertTrue(parsed["ok"])
        self.assertFalse(parsed["refuse"])
        self.assertEqual(parsed["function_calls"][0]["name"], "light.set")

    def test_parse_two_calls_illegal(self):
        parsed = parse_v2_text('[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]')
        self.assertFalse(parsed["ok"])
        self.assertEqual(parsed["error"], "illegal_shape")

    def test_parse_rejects_duplicate_keys_and_nonfinite_numbers(self):
        duplicate = parse_v2_text(
            '[{"name":"a","name":"b","arguments":{}}]'
        )
        nonfinite = parse_v2_text(
            '[{"name":"a","arguments":{"value":NaN}}]'
        )
        self.assertEqual(duplicate["error"], "json")
        self.assertEqual(nonfinite["error"], "json")

    def test_leaks(self):
        self.assertEqual(leak_markers("gold_route_id=1"), ["gold_route_id"])


class PackageTests(unittest.TestCase):
    def test_tiny_package_reports_missing_heads(self):
        pkg = load_package(TINY)
        caps = pkg.capabilities()
        self.assertTrue(pkg.verified_hashes)
        self.assertIn("contrastive", caps["missing_or_untrained_heads"])
        self.assertIn("mw_disposition", caps["missing_or_untrained_heads"])
        self.assertIn("confidence", caps["missing_or_untrained_heads"])
        self.assertEqual(caps["heads"]["contrastive"]["status"], "missing")

    def test_hash_mismatch(self):
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "pkg"
            shutil.copytree(TINY, dest)
            manifest = json.loads((dest / "mei-model.json").read_text(encoding="utf-8"))
            manifest["weights"]["sha256"] = "0" * 64
            (dest / "mei-model.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(SdkError) as ctx:
                load_package(dest)
            self.assertEqual(ctx.exception.id, "package_hash_mismatch")


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine.load(str(TINY))
        self.session = self.engine.create_session()
        self.tool = {
            "name": "light.set",
            "description": "开灯",
            "parameters": {"type": "object", "properties": {}},
        }

    def test_candidate_refuse(self):
        result = self.session.complete(
            {"query": "开灯", "oracle_tools": [self.tool], "candidate_text": "[]"}
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["refuse"])
        self.assertEqual(result["function_calls"], [])
        self.assertEqual(result["stats"]["backend"], "protocol")

    def test_candidate_call(self):
        result = self.session.complete(
            {
                "query": "开灯",
                "oracle_tools": [self.tool],
                "candidate_text": '[{"name":"light.set","arguments":{}}]',
            }
        )
        self.assertTrue(result["ok"])
        self.assertFalse(result["refuse"])
        self.assertEqual(result["function_calls"][0]["name"], "light.set")

    def test_unknown_tool(self):
        result = self.session.complete(
            {
                "query": "x",
                "oracle_tools": [self.tool],
                "candidate_text": '[{"name":"other.tool","arguments":{}}]',
            }
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["id"], "protocol_violation")

    def test_too_many_tools(self):
        tools = [{"name": f"t{i}", "parameters": {"type": "object", "properties": {}}} for i in range(6)]
        result = self.session.complete({"query": "x", "oracle_tools": tools, "candidate_text": "[]"})
        self.assertEqual(result["error"]["id"], "too_many_tools")

    def test_gold_leak(self):
        result = self.session.complete(
            {"query": "gold_route_id=9", "oracle_tools": [self.tool], "candidate_text": "[]"}
        )
        self.assertEqual(result["error"]["id"], "gold_leak")

    def test_engine_unavailable_without_candidate(self):
        result = self.session.complete({"query": "开灯", "oracle_tools": [self.tool]})
        self.assertEqual(result["error"]["id"], "engine_unavailable")
        self.assertIn("contrastive", result["capabilities"]["missing_or_untrained_heads"])

    def test_incomplete_package_cannot_force_mlx_inference(self):
        with self.assertRaises(SdkError) as caught:
            Engine.load(str(TINY), backend="mlx-reference")
        self.assertEqual(caught.exception.id, "capability_missing")

    def test_run_stops_on_refuse(self):
        loop = self.session.run({"query": "开灯", "oracle_tools": [self.tool], "candidate_text": "[]"})
        self.assertEqual(loop["stopped_reason"], "refuse")
        self.assertEqual(len(loop["turns"]), 1)

    def test_cancel(self):
        self.session.cancel()
        result = self.session.complete({"query": "开灯", "oracle_tools": [self.tool], "candidate_text": "[]"})
        self.assertEqual(result["error"]["id"], "cancelled")

    def test_render_matches_golden(self):
        path = GOLDEN / "render_request.json"
        if not path.is_file():
            self.skipTest("golden not generated")
        gold = json.loads(path.read_text(encoding="utf-8"))
        got = render_request(gold["request"], gold["request"]["oracle_tools"])
        self.assertEqual(got["prompt"], gold["prompt"])
        self.assertEqual(got["schema_fingerprint"], gold["schema_fingerprint"])


if __name__ == "__main__":
    unittest.main()
