from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import tempfile
import unittest
import re
from pathlib import Path

from mei_sdk import Engine, SdkError, UnsupportedSchemaError, load_package
from mei_sdk.cq2 import CqTensor, TensorContainer, dequantize, quantize
from mei_sdk.engine import _trusted_call_history
from mei_sdk.protocol import normalize_request, render_request
from mei_sdk.runtime_51m import mw_disposition_from_probabilities
from mei_sdk.shared import (
    BoundedKVManager,
    ToolIndex,
    allowed_token_ids,
    compile_byte_grammar_cached,
    is_accept_bytes,
    is_legal_byte_prefix,
    validate_generated_call,
    validate_tools,
    catalog_fingerprint,
    token_to_bytes,
)
from mei_sdk.version import SDK_ROOT

TINY = SDK_ROOT / "fixtures" / "packages" / "tiny-protocol-v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _container_sections(entries: list[dict], sections: list[dict[str, bytes]]) -> bytes:
    materialized = [
        {key: value for key, value in entry.items() if key not in {"data", "scales", "bit_map"}}
        for entry in entries
    ]
    for _ in range(32):
        header = json.dumps(
            {"quant_math_id": "mei-cq-v2-g128-wht-codebook", "tensors": materialized},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        cursor = 16 + len(header)
        updated = []
        for entry, blobs in zip(entries, sections):
            row = {
                key: value
                for key, value in entry.items()
                if key not in {"data", "scales", "bit_map"}
            }
            for name in ("data", "scales", "bit_map"):
                if name in blobs:
                    payload = blobs[name]
                    row[name] = {"offset": cursor, "nbytes": len(payload)}
                    cursor += len(payload)
            updated.append(row)
        if updated == materialized:
            payload = b"".join(
                blobs[name]
                for blobs in sections
                for name in ("data", "scales", "bit_map")
                if name in blobs
            )
            return b"MEICQ201" + struct.pack("<II", 2, len(header)) + header + payload
        materialized = updated
    raise AssertionError("test container header offsets did not converge")


def _container_bytes(entries: list[dict], payloads: list[bytes]) -> bytes:
    return _container_sections(entries, [{"data": payload} for payload in payloads])


def _write_v2_manifest(root: Path, manifest: dict) -> None:
    """Write a package manifest whose self-sized resource total is stable."""

    payload_bytes = sum(int(row["nbytes"]) for row in manifest["files"])
    for _ in range(32):
        encoded = json.dumps(
            manifest, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        measured = payload_bytes + len(encoded)
        if manifest["resources"]["package_bytes"] == measured:
            (root / "mei-model.json").write_bytes(encoded)
            return
        manifest["resources"]["package_bytes"] = measured
    raise AssertionError("test manifest package_bytes did not converge")


class AgentContinuationHistoryTests(unittest.TestCase):
    def test_trusted_result_is_paired_with_its_session_call(self):
        call_id = "call-s00000001-1-0123456789ab"
        history = _trusted_call_history(
            [{"call_id": call_id, "status": "ok", "payload": {"pnr": "PNR-7"}}],
            {
                call_id: {
                    "call_id": call_id,
                    "name": "book_flight",
                    "arguments": {"date": "2026-09-01", "from_city": "北京", "to_city": "上海"},
                }
            },
        )
        self.assertEqual(history[0]["role"], "assistant")
        self.assertEqual(history[0]["call_id"], call_id)
        self.assertEqual(
            history[0]["content"],
            '{"arguments":{"date":"2026-09-01","from_city":"北京","to_city":"上海"},'
            '"call_id":"call-s00000001-1-0123456789ab","name":"book_flight"}',
        )


class SchemaAndGrammarTests(unittest.TestCase):
    def setUp(self):
        self.tool = {
            "name": "天气.查询",
            "description": "查询天气",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "城市": {"type": "string", "minLength": 1, "maxLength": 12},
                    "天数": {"type": "integer", "minimum": 1, "maximum": 7},
                    "提醒": {"type": ["boolean", "null"]},
                    "标签": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "[A-Za-z0-9一-龥]+"},
                        "minItems": 1,
                        "maxItems": 2,
                    },
                },
                "required": ["城市"],
            },
        }

    def test_utf8_scalar_array_and_escape_are_accepted(self):
        grammar = compile_byte_grammar_cached([self.tool])
        text = '[{"name":"天气.查询","arguments":{"城市":"成\\u90fd","天数":3,"提醒":null,"标签":["出行","雨"]}}]'
        encoded = text.encode("utf-8")
        for length in range(len(encoded) + 1):
            # A prefix ending inside a multi-byte code point remains legal.
            self.assertTrue(is_legal_byte_prefix(encoded[:length], grammar), length)
        self.assertTrue(is_accept_bytes(encoded, grammar))
        escaped_emoji = '[{"name":"天气.查询","arguments":{"城市":"\\ud83d\\ude00"}}]'
        lone_surrogate = '[{"name":"天气.查询","arguments":{"城市":"\\ud800"}}]'
        self.assertTrue(is_accept_bytes(escaped_emoji.encode(), grammar))
        self.assertFalse(is_accept_bytes(lone_surrogate.encode(), grammar))

    def test_locked_sentencepiece_dummy_prefix_transduces_training_answers(self):
        try:
            import sentencepiece as sentencepiece
        except ImportError:
            self.skipTest("sentencepiece is unavailable")
        model = SDK_ROOT.parent / "tokenizer" / "zh-24k-v1" / "zh-24k-v1.model"
        processor = sentencepiece.SentencePieceProcessor(model_file=str(model))

        class LockedTokenizer:
            sp = processor
            pad_id = 0
            eos_id = 1
            bos_id = 2
            unk_id = 3
            vocab_size = processor.get_piece_size()
            model_path = model

        tokenizer = LockedTokenizer()
        for text in (
            "[]",
            '[{"name":"天气.查询","arguments":{"城市":"成都"}}]',
        ):
            ids = processor.encode(text, out_type=int)
            restored = b"".join(
                token_to_bytes(tokenizer, token, at_start=index == 0)
                for index, token in enumerate(ids)
            ).decode("utf-8")
            self.assertEqual(restored, text)
        ids = processor.encode("[]", out_type=int)
        self.assertEqual(processor.id_to_piece(ids[0]), "▁")
        grammar = compile_byte_grammar_cached([self.tool], tokenizer)
        initial = allowed_token_ids(
            tokenizer, grammar, b"", generated_token_count=0
        )
        self.assertIn(ids[0], initial)
        self.assertNotIn(tokenizer.unk_id, initial)
        after_dummy = allowed_token_ids(
            tokenizer, grammar, b"", generated_token_count=1
        )
        self.assertNotIn(ids[0], after_dummy)

    def test_out_of_range_and_unknown_argument_are_not_accepting(self):
        grammar = compile_byte_grammar_cached([self.tool])
        bad_range = '[{"name":"天气.查询","arguments":{"城市":"成都","天数":8}}]'
        bad_key = '[{"name":"天气.查询","arguments":{"城市":"成都","秘密":1}}]'
        self.assertFalse(is_accept_bytes(bad_range.encode(), grammar))
        self.assertFalse(is_accept_bytes(bad_key.encode(), grammar))

    def test_number_fraction_and_exponent_keep_every_prefix_legal(self):
        tool = {
            "name": "number.check",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number", "multipleOf": 0.5}},
                "required": ["value"],
            },
        }
        grammar = compile_byte_grammar_cached([tool])
        text = '[{"name":"number.check","arguments":{"value":-12.5e+2}}]'
        encoded = text.encode()
        for length in range(len(encoded) + 1):
            self.assertTrue(is_legal_byte_prefix(encoded[:length], grammar), length)
        self.assertTrue(is_accept_bytes(encoded, grammar))

    def test_numeric_constraints_are_exact_and_enum_uses_longest_literal(self):
        tool = {
            "name": "number.check",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "enum": [1, 10],
                        "multipleOf": 0.1,
                    }
                },
                "required": ["value"],
            },
        }
        grammar = compile_byte_grammar_cached([tool])
        ten = b'[{"name":"number.check","arguments":{"value":10}}]'
        self.assertTrue(is_accept_bytes(ten, grammar))
        multiple = {
            **tool,
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number", "multipleOf": 0.1}},
                "required": ["value"],
            },
        }
        grammar = compile_byte_grammar_cached([multiple])
        self.assertFalse(
            is_accept_bytes(
                b'[{"name":"number.check","arguments":{"value":100000000.05}}]',
                grammar,
            )
        )
        self.assertTrue(
            is_accept_bytes(
                b'[{"name":"number.check","arguments":{"value":100000000.1}}]',
                grammar,
            )
        )
        unsafe_constraint = {
            **tool,
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer", "minimum": 9_007_199_254_740_992}
                },
            },
        }
        with self.assertRaises(UnsupportedSchemaError):
            validate_tools([unsafe_constraint])
        float_const = {
            "name": "number.const",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number", "const": 1.0}},
                "required": ["value"],
            },
        }
        grammar = compile_byte_grammar_cached([float_const])
        self.assertTrue(
            is_accept_bytes(
                b'[{"name":"number.const","arguments":{"value":1}}]', grammar
            )
        )
        self.assertFalse(
            is_accept_bytes(
                b'[{"name":"number.const","arguments":{"value":1.0}}]', grammar
            )
        )

    def test_canonical_and_compatibility_tool_contracts_cannot_be_dual_written(self):
        for canonical, compatibility, left, right in (
            (
                "required_permissions",
                "x-mei-permissions",
                [],
                ["admin"],
            ),
            (
                "required_state",
                "x-mei-state",
                {},
                {"armed": True},
            ),
        ):
            tool = {
                "name": "dual.contract",
                "parameters": {"type": "object", "properties": {}},
                canonical: left,
                compatibility: right,
            }
            with self.assertRaises(UnsupportedSchemaError):
                validate_tools([tool])

    def test_unsafe_numbers_and_non_json_state_fail_closed(self):
        unsafe = '[{"name":"天气.查询","arguments":{"城市":"成都","天数":' + (
            "9" * 400
        ) + "}}]"
        parsed = validate_generated_call(
            unsafe,
            tools=[self.tool],
            request={"query": "成都", "system_facts": "成都"},
            confidence=1.0,
        )
        self.assertFalse(parsed["ok"])
        tool = {
            "name": "bad.state",
            "parameters": {"type": "object", "properties": {}},
            "required_state": {1: "not-a-json-object-key"},
        }
        with self.assertRaises(UnsupportedSchemaError):
            validate_tools([tool])

    def test_nested_and_combinator_schemas_fail_at_registration(self):
        for child in (
            {"type": "object", "properties": {}},
            {"oneOf": [{"type": "string"}, {"type": "null"}]},
            {"type": "array", "items": {"type": "object", "properties": {}}},
        ):
            tool = {
                "name": "bad",
                "parameters": {"type": "object", "properties": {"x": child}},
            }
            with self.assertRaises(UnsupportedSchemaError):
                validate_tools([tool])

    def test_complete_request_v2_is_schema_strict(self):
        request = {
            "wire_version": "mei-runtime-wire-v2",
            "query": "开灯",
            "evidence": [
                {
                    "subject": "light.set",
                    "predicate": "on",
                    "value": True,
                    "source": "ui",
                    "verified": True,
                }
            ],
            "permissions": {"scopes": ["device.write"]},
            "state": {"online": True},
            "mw": {"decision": "continue", "source": "protocol-test"},
            "confidence": {"value": 0.9, "source": "protocol-test"},
            "enforce_confidence": True,
        }
        normalized = normalize_request(request)
        self.assertEqual(normalized["query"], "开灯")
        rendered = render_request(
            request,
            [{"name": "light.set", "parameters": {"type": "object", "properties": {}}}],
        )
        self.assertIn("<evidence>", rendered["ordinary"])
        self.assertIn("<permissions>", rendered["ordinary"])
        self.assertNotIn("权限：", rendered["sink"])
        with self.assertRaisesRegex(ValueError, "unexpected field"):
            normalize_request({**request, "catalog": []})
        with self.assertRaisesRegex(ValueError, "not unique"):
            normalize_request(
                {**request, "permissions": {"scopes": ["device.write", "device.write"]}}
            )
        with self.assertRaisesRegex(ValueError, "expected"):
            normalize_request({**request, "confidence": 0.9})
        with self.assertRaisesRegex(ValueError, "prohibited"):
            normalize_request(
                {
                    **request,
                    "mw_disposition": {
                        "decision": "continue",
                        "source": "deterministic-policy",
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "missing required"):
            normalize_request(
                {
                    **request,
                    "mw": {"decision": "constrain", "source": "protocol-test"},
                }
            )
        with self.assertRaisesRegex(TypeError, "not JSON-compatible"):
            normalize_request({**request, "state": {"opaque": {"not-json"}}})
        with self.assertRaisesRegex(ValueError, "both scopes and grants"):
            normalize_request(
                {
                    **request,
                    "permissions": {"scopes": [], "grants": ["admin"]},
                }
            )

    def test_portable_pattern_and_format_are_cross_runtime_strict(self):
        for pattern in (r"\w+", r"(?=x)x", r"[a&&b]", r"[a--b]"):
            with self.assertRaises(UnsupportedSchemaError):
                validate_tools(
                    [
                        {
                            "name": "pattern.test",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "string", "pattern": pattern}},
                            },
                        }
                    ]
                )
        tool = {
            "name": "time.test",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string", "format": "date-time"}},
                "required": ["value"],
            },
        }
        grammar = compile_byte_grammar_cached([tool])
        accepted = '[{"name":"time.test","arguments":{"value":"2026-08-30T12:34:56Z"}}]'
        loose = '[{"name":"time.test","arguments":{"value":"2026-08-30 12:34:56"}}]'
        self.assertTrue(is_accept_bytes(accepted.encode(), grammar))
        self.assertFalse(is_accept_bytes(loose.encode(), grammar))


class GateOrderTests(unittest.TestCase):
    def setUp(self):
        self.tool = {
            "name": "light.set",
            "parameters": {
                "type": "object",
                "properties": {"on": {"type": "boolean"}},
                "required": ["on"],
            },
            "required_permissions": ["device.write"],
            "required_state": {"online": True},
        }
        self.text = '[{"name":"light.set","arguments":{"on":true}}]'
        self.request = {
            "wire_version": "mei-runtime-wire-v2",
            "query": "打开灯",
            "evidence": [
                {"tool": "light.set", "argument": "on", "value": True, "source": "ui", "verified": True}
            ],
            "permissions": {"scopes": ["device.write"]},
            "state": {"online": True},
            "mw": {"decision": "continue", "source": "protocol-test"},
        }

    def test_full_deterministic_order(self):
        result = validate_generated_call(
            self.text, tools=[self.tool], request=self.request, confidence=0.9
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            [row["gate"] for row in result["gates"]],
            ["grammar", "schema", "provenance", "permission", "state", "mw", "confidence"],
        )

    def test_learned_confidence_cannot_override_permission(self):
        request = dict(self.request)
        request["permissions"] = {"scopes": ["device.write"], "denied_tools": ["light.set"]}
        result = validate_generated_call(self.text, tools=[self.tool], request=request, confidence=1.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "permission_denied")
        self.assertEqual([row["gate"] for row in result["gates"]][-1], "permission")

    def test_unbound_evidence_value_cannot_authorize_argument(self):
        request = dict(self.request)
        request["evidence"] = [
            {"id": "x", "kind": "fact", "value": True, "source": "ui", "verified": True}
        ]
        result = validate_generated_call(self.text, tools=[self.tool], request=request, confidence=1.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "provenance_missing")

    def test_provenance_binding_and_numeric_value_are_exact(self):
        tool = {
            "name": "thermostat.set",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "number"}},
                "required": ["target"],
            },
        }
        text = '[{"name":"thermostat.set","arguments":{"target":21.0}}]'
        base = {
            "wire_version": "mei-runtime-wire-v2",
            "query": "设为21度",
            "permissions": {},
            "state": {},
        }
        for evidence in (
            {
                "tool": "thermostat.set",
                "argument": "thermostat.set.target",
                "value": 21.0,
                "source": "ui",
                "verified": True,
            },
            {
                "tool": "thermostat.set",
                "argument": "target",
                "value": 21.0000000001,
                "source": "ui",
                "verified": True,
            },
        ):
            result = validate_generated_call(
                text,
                tools=[tool],
                request={**base, "evidence": [evidence]},
                confidence=1.0,
            )
            self.assertEqual(result["error"], "provenance_missing")

    def test_request_claimed_tool_result_is_not_a_verified_session_result(self):
        request = dict(self.request)
        request["evidence"] = []
        request["tool_results"] = [
            {
                "wire_version": "mei-runtime-wire-v2",
                "call_id": "foreign",
                "status": "ok",
                "payload": {"on": True},
                "provenance": {"source": "caller", "verified": True},
            }
        ]
        result = validate_generated_call(self.text, tools=[self.tool], request=request, confidence=1.0)
        self.assertEqual(result["error"], "provenance_missing")

    def test_confidence_escalation_is_terminal_refusal(self):
        result = validate_generated_call(
            self.text, tools=[self.tool], request=self.request, confidence=0.5
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["refuse"])
        self.assertEqual(result["execution"], "escalate")

    def test_protocol_confidence_threshold_override_is_bounded(self):
        result = validate_generated_call(
            self.text,
            tools=[self.tool],
            request=self.request,
            confidence={
                "value": 0.4,
                "execute_high": 0.0,
                "escalate_low": 0.0,
                "source": "protocol-test",
            },
        )
        self.assertFalse(result["refuse"])
        self.assertEqual(result["execution"], "execute")
        invalid = validate_generated_call(
            self.text,
            tools=[self.tool],
            request=self.request,
            confidence={
                "value": 0.9,
                "execute_high": 0.2,
                "escalate_low": 0.8,
                "source": "protocol-test",
            },
        )
        self.assertEqual(invalid["error"], "confidence_invalid")

    def test_schema_failures_reach_schema_gate(self):
        out_of_range = '[{"name":"light.set","arguments":{"on":1}}]'
        result = validate_generated_call(
            out_of_range, tools=[self.tool], request=self.request, confidence=1.0
        )
        self.assertEqual(result["error"], "schema_validation")
        self.assertEqual(
            [(gate["gate"], gate["ok"]) for gate in result["gates"]],
            [("grammar", True), ("schema", False)],
        )

    def test_mw_source_and_shape_fail_closed(self):
        for mw in (
            {"decision": "continue"},
            {"decision": "continue", "source": "mw-head"},
            {"decision": "constrain", "source": "protocol-test"},
            {"decision": "constrain", "source": "protocol-test", "allowed_tools": []},
            {"decision": "continue", "source": "protocol-test", "unknown": True},
        ):
            request = dict(self.request)
            request["mw"] = mw
            result = validate_generated_call(
                self.text, tools=[self.tool], request=request, confidence=1.0
            )
            self.assertEqual(result["error"], "mw_invalid")
        missing_receipt = dict(self.request)
        missing_receipt.pop("mw")
        missing_receipt["mw_disposition"] = {
            "decision": "continue",
            "source": "mw-head",
        }
        result = validate_generated_call(
            self.text, tools=[self.tool], request=missing_receipt, confidence=1.0
        )
        self.assertEqual(result["error"], "mw_invalid")

    def test_mw_reason_head_projects_only_calibrated_class_zero_to_continue(self):
        receipt = "a" * 64
        ready = [0.0] * 20
        ready[0], ready[1], ready[2] = 0.70, 0.15, 0.15
        wire, audit = mw_disposition_from_probabilities(
            ready, receipt_sha256=receipt
        )
        self.assertEqual(wire, {"decision": "continue", "source": "mw-head", "receipt_sha256": receipt})
        self.assertEqual(audit["reason_code"], 0)
        uncertain = [0.0] * 20
        uncertain[0], uncertain[1] = 0.55, 0.45
        self.assertEqual(
            mw_disposition_from_probabilities(uncertain, receipt_sha256=receipt)[0]["decision"],
            "stop",
        )
        unsafe = [0.0] * 20
        unsafe[0], unsafe[19] = 0.1, 0.9
        unsafe_wire, unsafe_audit = mw_disposition_from_probabilities(
            unsafe, receipt_sha256=receipt
        )
        self.assertEqual(unsafe_wire["decision"], "stop")
        self.assertEqual(unsafe_audit["reason_code"], 19)
        with self.assertRaises(SdkError):
            mw_disposition_from_probabilities([1.0], receipt_sha256=receipt)


class IndexAndWindowTests(unittest.TestCase):
    def test_cq2_matches_cross_runtime_golden_bytes(self):
        import math

        values = [math.sin(index * 0.173) * 0.8 + ((index % 11) - 5) * 0.03 for index in range(256)]
        packed = quantize(values, [2, 4])
        self.assertEqual(packed.scales_f16, (14482, 14482))
        self.assertEqual(packed.bit_map, b"\x02")
        self.assertEqual(
            packed.data.hex(),
            "6b9956a9856898a96556a6955ba555946996ab9557958694695a55566a565b96"
            "7387788886898987a57c7f868f796c7578778787768787777078887876898b78"
            "787676787386787780978889948c8e7a89787888787687887975848771878688",
        )
        decoded = dequantize(packed)
        mse = sum((left - right) ** 2 for left, right in zip(values, decoded)) / len(values)
        self.assertLess(mse, 0.14)
        with self.assertRaises(SdkError):
            quantize(values, [2])
        with self.assertRaises(SdkError):
            quantize([float("nan")] * 128, [2])

    def test_cq2_zero_group_uses_minimum_positive_f16_scale(self):
        packed = quantize([0.0] * 128, [2])
        self.assertEqual(packed.scales_f16, (1,))
        self.assertEqual(packed.data, b"\x55" * 32)
        decoded = dequantize(packed)
        self.assertEqual(len(decoded), 128)
        self.assertTrue(all(math.isfinite(value) for value in decoded))

    def test_cq2_reads_shared_rust_js_golden(self):
        golden = json.loads((SDK_ROOT / "spec" / "golden" / "cq2_v2.json").read_text(encoding="utf-8"))
        values = [((index % 17) - 8) / 8 for index in range(golden["n_values"])]
        packed = quantize(values, golden["group_bits"])
        self.assertEqual(len(packed.data), golden["data_nbytes"])
        self.assertEqual(_sha(packed.data), golden["data_sha256"])
        self.assertEqual(
            b"".join(struct.pack("<H", value) for value in packed.scales_f16).hex(),
            golden["scales_f16_le_hex"],
        )
        self.assertEqual(packed.bit_map.hex(), golden["bit_map_hex"])
        decoded = dequantize(packed)
        for actual, expected in zip(decoded, golden["decoded_head"]):
            self.assertAlmostEqual(actual, expected, delta=golden["decoded_abs_tolerance"])

    def test_cq2_container_rejects_all_q4_masquerade(self):
        packed = quantize([float(index % 7) for index in range(128)], [4])
        scales = b"".join(struct.pack("<H", value) for value in packed.scales_f16)
        entry = {
            "name": "matrix",
            "role": "lm",
            "shape": [128],
            "n_params": 128,
            "dtype": "cq2",
            "group_size": 128,
            "transform": "wht",
            "codebook": "gaussian-lloyd-q2-v1",
        }
        blob = _container_sections(
            [entry],
            [{"data": packed.data, "scales": scales, "bit_map": packed.bit_map}],
        )
        with self.assertRaisesRegex(SdkError, "all-q4"):
            TensorContainer.parse(blob)

    def test_cq2_container_requires_portable_tensor_role(self):
        entry = {
            "name": "safe",
            "role": None,
            "shape": [1],
            "n_params": 1,
            "dtype": "f16",
            "transform": "none",
            "codebook": "none",
        }
        blob = _container_bytes([entry], [struct.pack("<e", 1.0)])
        with self.assertRaisesRegex(SdkError, "role is invalid"):
            TensorContainer.parse(blob)

    def test_container_accepts_canonical_scalar_tensor_shape(self):
        entry = {
            "name": "blocks.0.attn_gate",
            "role": "lm",
            "shape": [],
            "n_params": 1,
            "dtype": "f16",
            "transform": "none",
            "codebook": "none",
        }
        blob = _container_bytes([entry], [struct.pack("<e", 0.0)])
        parsed = TensorContainer.parse(blob)
        self.assertEqual(parsed.entries[entry["name"]].shape, ())
        self.assertEqual(parsed.dequantize(entry["name"]), [0.0])

    def test_f16_index_roundtrip_and_stable_tie(self):
        tools = [
            {"name": "b", "parameters": {"type": "object", "properties": {}}},
            {"name": "a", "parameters": {"type": "object", "properties": {}}},
        ]
        index = ToolIndex(model_hash="1" * 64, head_hash="2" * 64, tokenizer_hash="3" * 64)
        index.build(tools, lambda _text: [1.0, 0.0])
        self.assertEqual([row.tool_id for row in index.topk([1.0, 0.0], k=2)], ["a", "b"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            index.save(path)
            loaded = ToolIndex.load(path, expected_fingerprint=index.fingerprint)
            self.assertEqual([row.tool_id for row in loaded.topk([1.0, 0.0], k=2)], ["a", "b"])
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["records"][0]["tool_id"] = "renamed"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ID/schema mismatch"):
                ToolIndex.load(path)

    def test_tool_index_rejects_noncanonical_serializer_and_malformed_embedding(self):
        tools = [{"name": "a", "parameters": {"type": "object", "properties": {}}}]
        index = ToolIndex(model_hash="1" * 64, head_hash="2" * 64, tokenizer_hash="3" * 64)
        index.build(tools, lambda _text: [1.0, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            raw = index.as_dict()
            raw["serializer_id"] = "evil-serializer"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical runtime serializer"):
                ToolIndex.load(path)
            malformed = index.as_dict()
            malformed["records"][0]["embedding_f16_base64"] = None
            path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "base64 text"):
                ToolIndex.load(path)

    def test_catalog_fingerprint_matches_cross_runtime_golden(self):
        golden = json.loads(
            (SDK_ROOT / "spec" / "golden" / "catalog_fingerprint_v2.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(catalog_fingerprint(golden["catalog"]), golden["sha256"])
        self.assertEqual(
            catalog_fingerprint(list(reversed(golden["catalog"]))), golden["sha256"]
        )

    def test_index_fingerprint_covers_permission_contract(self):
        base = {
            "name": "device.write",
            "parameters": {"type": "object", "properties": {}},
            "required_permissions": ["device.write"],
        }
        index = ToolIndex(model_hash="1" * 64, head_hash="2" * 64, tokenizer_hash="3" * 64)
        index.build([base], lambda _text: [1.0, 0.0])
        original = index.fingerprint
        changed = dict(base)
        changed["required_permissions"] = ["admin"]
        index.build([changed], lambda _text: [1.0, 0.0])
        self.assertNotEqual(index.fingerprint, original)

    def test_sink_and_ordinary_are_bounded(self):
        manager = BoundedKVManager()
        manager.prepare(range(16), range(1000))
        self.assertEqual(len(manager.sink_ids), 16)
        self.assertEqual(len(manager.ordinary_ids), 256)
        self.assertEqual(manager.ordinary_ids[0], 744)
        for token in range(1000, 2000):
            manager.append_ordinary(token)
        self.assertEqual(len(manager.ordinary_ids), 256)
        self.assertLess(max(manager.visible_positions), 2048)
        self.assertGreater(manager.snapshot()["total_tokens_seen"], 2000)
        self.assertTrue(manager.snapshot()["bounded"])
        self.assertEqual(manager.snapshot()["target_kv_dtype"], "int8")
        self.assertIsNone(manager.snapshot()["measured_cache_dtype"])
        self.assertFalse(manager.ram_bound_ok(1, 1, 1))
        with self.assertRaisesRegex(ValueError, "tool_schema_budget_exceeded"):
            manager.prepare(range(1025), [])


class SessionStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine.load(TINY)
        self.session = self.engine.create_session()
        self.tool = {"name": "clock.read", "parameters": {"type": "object", "properties": {}}}
        self.call = '[{"name":"clock.read","arguments":{}}]'

    def test_complete_submit_result_continue(self):
        first = self.session.complete(
            {"query": "几点", "oracle_tools": [self.tool], "candidate_text": self.call}
        )
        self.assertEqual(first["kind"], "call")
        self.assertRegex(
            first["call"]["call_id"], r"^call-s[0-9a-f]{8}-1-[0-9a-f]{12}$"
        )
        self.assertEqual(
            set(first),
            {
                "wire_version",
                "kind",
                "call",
                "refusal",
                "error",
                "state",
                "selected_tools",
                "schema_fingerprint",
                "raw_text",
                "confidence",
                "provenance",
                "capabilities",
                "stats",
                "ok",
                "refuse",
                "function_calls",
            },
        )
        self.assertEqual(set(first["state"]), {"step", "pending_call_id", "cancelled"})
        self.assertEqual(set(first["call"]), {"call_id", "name", "arguments"})
        blocked = self.session.complete(
            {"query": "几点", "oracle_tools": [self.tool], "candidate_text": "[]"}
        )
        self.assertEqual(blocked["kind"], "error")
        with self.assertRaises(SdkError):
            self.session.submit_tool_result("call_stale", {"hour": 12})
        with self.assertRaisesRegex(SdkError, "provenance must be verified"):
            self.session.submit_tool_result(
                {
                    "wire_version": "mei-runtime-wire-v2",
                    "call_id": first["call"]["call_id"],
                    "status": "ok",
                    "payload": {"hour": 12},
                    "provenance": {"source": "host_executor", "verified": False},
                }
            )
        with self.assertRaises(SdkError) as too_large:
            self.session.submit_tool_result(
                {
                    "wire_version": "mei-runtime-wire-v2",
                    "call_id": first["call"]["call_id"],
                    "status": "ok",
                    "payload": "x" * (64 * 1024),
                    "provenance": {"source": "host_executor", "verified": True},
                }
            )
        self.assertEqual(too_large.exception.id, "tool_result_too_large")
        accepted_payload = {"hour": 12}
        ack = self.session.submit_tool_result(
            {
                "wire_version": "mei-runtime-wire-v2",
                "call_id": first["call"]["call_id"],
                "status": "ok",
                "payload": accepted_payload,
                "provenance": {"source": "host_executor", "verified": True},
            }
        )
        self.assertTrue(ack["accepted"])
        self.assertEqual(set(ack), {"wire_version", "accepted", "call_id", "state"})
        accepted_payload["hour"] = 99
        with self.assertRaisesRegex(SdkError, "narration_requires_respond"):
            self.session.narrate()
        last = self.session.complete(
            {"query": "几点", "oracle_tools": [self.tool], "candidate_text": "[]"}
        )
        self.assertEqual(last["kind"], "respond")
        self.assertFalse(last["refuse"])
        self.assertIsNone(last["refusal"])
        narrated = self.session.narrate(
            {"call_ids": [first["call"]["call_id"]], "mode": "deterministic"}
        )
        self.assertEqual(narrated["text"], "clock.read已执行完成，hour为12。")
        self.assertTrue(narrated["grounded"])
        fallback = self.session.narrate({"mode": "adapter"})
        self.assertEqual(fallback["mode"], "deterministic")
        self.assertTrue(fallback["fallback_used"])

    def test_run_success_requires_respond_after_verified_result(self):
        result = self.session.run(
            {
                "query": "几点",
                "oracle_tools": [self.tool],
                "candidate_texts": [self.call, "[]"],
            },
            executors={"clock.read": lambda _call: {"hour": 12}},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["stopped_reason"], "respond")
        self.assertEqual([turn["kind"] for turn in result["turns"]], ["call", "respond"])

    def test_registered_catalog_accepts_strict_structured_v2_request(self):
        tool = {
            "name": "light.set",
            "parameters": {
                "type": "object",
                "properties": {"on": {"type": "boolean"}},
                "required": ["on"],
            },
            "required_permissions": ["device.write"],
            "required_state": {"online": True},
        }
        self.engine.register_tools([tool])
        session = self.engine.create_session()
        turn = session.complete(
            {
                "wire_version": "mei-runtime-wire-v2",
                "query": "开灯",
                "evidence": [
                    {
                        "subject": "light.set",
                        "predicate": "on",
                        "value": True,
                        "source": "ui",
                        "verified": True,
                    }
                ],
                "permissions": {"scopes": ["device.write"]},
                "state": {"online": True},
                "mw": {"decision": "continue", "source": "protocol-test"},
                "confidence": {"value": 0.9, "source": "protocol-test"},
                "enforce_confidence": True,
                "candidate_text": '[{"name":"light.set","arguments":{"on":true}}]',
            }
        )
        self.assertEqual(turn["kind"], "call")
        self.assertEqual(
            [gate["gate"] for gate in turn["provenance"]["gates"]],
            ["retrieval", "grammar", "schema", "provenance", "permission", "state", "mw", "confidence"],
        )

    def test_only_session_accepted_tool_result_can_supply_later_provenance(self):
        first = self.session.complete(
            {"query": "读取", "oracle_tools": [self.tool], "candidate_text": self.call}
        )
        self.session.submit_tool_result(first["call"]["call_id"], {"on": True})
        next_tool = {
            "name": "light.set",
            "parameters": {
                "type": "object",
                "properties": {"on": {"type": "boolean"}},
                "required": ["on"],
            },
        }
        second = self.session.complete(
            {
                "query": "应用刚才的结果",
                "oracle_tools": [next_tool],
                "candidate_text": '[{"name":"light.set","arguments":{"on":true}}]',
            }
        )
        self.assertEqual(second["kind"], "call")
        self.assertEqual(
            second["provenance"]["arguments"]["on"]["source"],
            "verified_tool_result",
        )

    def test_tool_result_call_id_is_bound_to_its_session(self):
        first_session = self.engine.create_session()
        second_session = self.engine.create_session()
        request = {
            "query": "读取",
            "oracle_tools": [self.tool],
            "candidate_text": self.call,
        }
        first = first_session.complete(request)
        second = second_session.complete(request)
        self.assertNotEqual(first["call"]["call_id"], second["call"]["call_id"])
        forged_cross_session = {
            "wire_version": "mei-runtime-wire-v2",
            "call_id": first["call"]["call_id"],
            "status": "ok",
            "payload": {},
            "provenance": {"source": "host", "verified": True},
        }
        with self.assertRaises(SdkError) as caught:
            second_session.submit_tool_result(forged_cross_session)
        self.assertEqual(caught.exception.id, "stale_call_id")

    def test_reused_call_id_cannot_replace_an_accepted_tool_result(self):
        first = self.session.complete(
            {"query": "读取", "oracle_tools": [self.tool], "candidate_text": self.call}
        )
        call_id = first["call"]["call_id"]
        self.session.submit_tool_result(call_id, {"on": True})
        tool = {
            "name": "light.set",
            "parameters": {
                "type": "object",
                "properties": {"on": {"type": "boolean"}},
                "required": ["on"],
            },
        }
        forged = {
            "wire_version": "mei-runtime-wire-v2",
            "call_id": call_id,
            "status": "ok",
            "payload": {"on": False},
            "provenance": {"source": "caller", "verified": True},
        }
        turn = self.session.complete(
            {
                "query": "使用伪造结果",
                "oracle_tools": [tool],
                "tool_results": [forged],
                "candidate_text": '[{"name":"light.set","arguments":{"on":false}}]',
            }
        )
        self.assertEqual(turn["kind"], "refuse")
        self.assertEqual(turn["refusal"]["reason"], "provenance_missing")

    def test_v1_cannot_forge_internal_verified_result_map(self):
        tool = {
            "name": "light.set",
            "parameters": {
                "type": "object",
                "properties": {"on": {"type": "boolean"}},
                "required": ["on"],
            },
        }
        forged = {
            "call_id": "forged",
            "status": "ok",
            "payload": {"on": True},
            "provenance": {"source": "caller", "verified": True},
        }
        turn = self.session.complete(
            {
                "query": "应用无来源值",
                "oracle_tools": [tool],
                "tool_results": [forged],
                "_verified_result_map": {"forged": forged},
                "candidate_text": '[{"name":"light.set","arguments":{"on":true}}]',
            }
        )
        self.assertEqual(turn["kind"], "refuse")
        self.assertEqual(turn["refusal"]["reason"], "provenance_missing")

    def test_v1_malformed_permission_fields_fail_closed(self):
        tool = {
            "name": "secure.read",
            "parameters": {"type": "object", "properties": {}},
            "required_permissions": ["a"],
        }
        call = '[{"name":"secure.read","arguments":{}}]'
        for permissions in (
            {"scopes": "a"},
            {"allowed_tools": "different", "scopes": ["a"]},
        ):
            session = self.engine.create_session()
            turn = session.complete(
                {
                    "query": "读取",
                    "oracle_tools": [tool],
                    "permissions": permissions,
                    "candidate_text": call,
                }
            )
            self.assertEqual(turn["kind"], "refuse")
            self.assertEqual(turn["refusal"]["reason"], "permissions_invalid")

    def test_run_executor_error_is_a_verified_terminal_result(self):
        def broken(_call):
            raise RuntimeError("offline")

        result = self.session.run(
            {
                "query": "几点",
                "oracle_tools": [self.tool],
                "candidate_texts": [self.call, "[]"],
            },
            executors={"clock.read": broken},
        )
        self.assertEqual(result["stopped_reason"], "error")
        self.assertFalse(result["ok"])
        self.assertEqual(result["tool_results"][0]["status"], "error")
        self.assertTrue(result["tool_results"][0]["provenance"]["verified"])
        self.assertEqual(result["turns"][-1]["kind"], "error")
        self.assertIsNone(self.session.pending_call)
        self.assertEqual(
            set(result), {"wire_version", "ok", "turns", "tool_results", "stopped_reason"}
        )
        self.assertEqual(
            set(result["tool_results"][0]),
            {"wire_version", "call_id", "status", "payload", "error", "provenance"},
        )

    def test_missing_executor_fails_closed(self):
        result = self.session.run(
            {"query": "几点", "oracle_tools": [self.tool], "candidate_text": self.call}
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["stopped_reason"], "error")
        self.assertEqual(result["turns"][-1]["error"]["id"], "executor_missing")
        self.assertIsNone(self.session.pending_call)
        self.assertEqual(result["tool_results"][-1]["error"]["code"], "executor_missing")
        self.assertTrue(result["tool_results"][-1]["provenance"]["verified"])

    def test_session_max_steps_is_a_distinct_fail_closed_terminal(self):
        session = self.engine.create_session({"max_steps": 1})
        result = session.run(
            {
                "query": "循环",
                "oracle_tools": [self.tool],
                "candidate_texts": [self.call, self.call],
            },
            executors={"clock.read": lambda _call: {"hour": 12}},
            max_steps=2,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["stopped_reason"], "max_steps")
        self.assertEqual(result["turns"][-1]["error"]["id"], "max_steps")

    def test_run_rejected_executor_payload_clears_pending_and_natural_limit_is_terminal(self):
        session = self.engine.create_session({"max_tool_result_bytes": 64})
        rejected = session.run(
            {
                "query": "读取",
                "oracle_tools": [self.tool],
                "candidate_text": self.call,
            },
            executors={"clock.read": lambda _call: {"blob": "x" * 256}},
        )
        self.assertEqual(rejected["stopped_reason"], "error")
        self.assertEqual(rejected["turns"][-1]["error"]["id"], "tool_result_too_large")
        self.assertIsNone(session.pending_call)
        limited = self.engine.create_session().run(
            {
                "query": "读取",
                "oracle_tools": [self.tool],
                "candidate_text": self.call,
            },
            executors={"clock.read": lambda _call: {"hour": 12}},
            max_steps=1,
        )
        self.assertEqual(limited["stopped_reason"], "max_steps")
        self.assertEqual(limited["turns"][-1]["error"]["id"], "max_steps")

    def test_cancel_pending_is_idempotent_and_closed_handles_fail(self):
        first = self.session.complete(
            {"query": "几点", "oracle_tools": [self.tool], "candidate_text": self.call}
        )
        self.assertIsNotNone(first["state"]["pending_call_id"])
        self.session.cancel()
        self.session.cancel()
        self.assertIsNone(self.session.pending_call)
        self.assertEqual(self.session.step_index, 1)
        terminal = self.session.complete(
            {"query": "几点", "oracle_tools": [self.tool], "candidate_text": "[]"}
        )
        self.assertEqual(terminal["error"]["id"], "cancelled")
        self.assertEqual(terminal["state"]["step"], 1)
        closed = self.engine.create_session()
        closed.close()
        for operation in (
            lambda: closed.register_tools([self.tool]),
            closed.cancel,
            lambda: closed.retrieve("x", [self.tool]),
            lambda: closed.embed("x"),
        ):
            with self.assertRaises(SdkError) as caught:
                operation()
            self.assertEqual(caught.exception.id, "session_closed")
        with self.assertRaises(SdkError):
            self.engine.create_session({"unknown": True})

    def test_tool_result_contract_and_cancelled_terminal_state(self):
        first = self.session.complete(
            {"query": "几点", "oracle_tools": [self.tool], "candidate_text": self.call}
        )
        call_id = first["call"]["call_id"]
        with self.assertRaisesRegex(SdkError, "missing required field error"):
            self.session.submit_tool_result(
                {
                    "wire_version": "mei-runtime-wire-v2",
                    "call_id": call_id,
                    "status": "error",
                    "payload": None,
                    "provenance": {"source": "host", "verified": True},
                }
            )
        ack = self.session.submit_tool_result(
            {
                "wire_version": "mei-runtime-wire-v2",
                "call_id": call_id,
                "status": "cancelled",
                "payload": None,
                "provenance": {"source": "host", "verified": True},
            }
        )
        self.assertTrue(ack["state"]["cancelled"])
        terminal = self.session.complete(
            {"query": "几点", "oracle_tools": [self.tool], "candidate_text": "[]"}
        )
        self.assertEqual(terminal["error"]["id"], "cancelled")

    def test_candidate_and_release_forbid_raw_decode(self):
        original = self.engine.package.manifest.get("release_class")
        try:
            for release_class in ("candidate", "release"):
                self.engine.package.manifest["release_class"] = release_class
                session = self.engine.create_session()
                turn = session.complete(
                    {
                        "query": "几点",
                        "oracle_tools": [self.tool],
                        "candidate_text": self.call,
                        "decode_mode": "raw",
                    }
                )
                self.assertEqual(turn["error"]["id"], "decode_mode_forbidden")
                self.assertEqual(turn["state"]["step"], 0)
                fixture_session = self.engine.create_session()
                fixture_turn = fixture_session.complete(
                    {
                        "query": "几点",
                        "oracle_tools": [self.tool],
                        "candidate_text": self.call,
                    }
                )
                self.assertEqual(
                    fixture_turn["error"]["id"], "decode_mode_forbidden"
                )
                self.assertIn("candidate_text", fixture_turn["error"]["message"])
        finally:
            self.engine.package.manifest["release_class"] = original


class NarrationAndPackageTests(unittest.TestCase):
    def test_semantic_narration_query_action_and_failure(self):
        from mei_sdk.shared import NarrationProvider, verified_result_view

        provider = NarrationProvider()
        cases = [
            ({"tool_name": "get_temperature", "arguments": {"zone": "客厅"}, "status": "ok", "payload": {"temperature_c": 26}}, "客厅当前温度为26℃。"),
            ({"tool_name": "set_temperature", "arguments": {"zone": "客厅", "target_c": 27}, "status": "ok", "payload": {"applied": True, "temperature_c": 27}}, "已将客厅温度设为27℃。"),
            ({"tool_name": "start_device", "arguments": {"device": "空调"}, "status": "error", "payload": None, "error": {"code": "offline", "message": "设备离线"}}, "空调启动失败：设备离线。"),
        ]
        for index, (value, expected) in enumerate(cases):
            value.update({
                "call_id": f"call-s00000001-{index + 1}-0123456789ab",
                "provenance": {"source": "test", "verified": True},
            })
            self.assertEqual(provider.narrate(verified_result_view(value)), expected)

    def test_canonical_51m_weight_contract_is_exact_not_parameter_sum_only(self):
        from mei_sdk import package as package_contract

        tensors = package_contract._CANONICAL_LM_TENSORS
        self.assertEqual(len(tensors), 400)
        self.assertEqual(
            sum(math.prod(shape) for _name, shape in tensors),
            51_463_797,
        )
        self.assertEqual(
            package_contract._CANONICAL_WEIGHT_CONTRACT_SHA256,
            "c468b96453f0a377b1ffbcfef00ed9e108c82b2c44847ee5345509a181581d9b",
        )
        self.assertEqual(
            package_contract._CANONICAL_RUNTIME_CONTRACT_SHA256,
            "74839b08155e624f14318ca8646166ddc68ee6496720dedac26aa91fdc8bdf43",
        )
        self.assertEqual(
            package_contract._CANONICAL_TRAINING_AUX_SHA256,
            "83849db3926693e49c0896a58c172ae15e4b203550cee0ef12a4c37a8c1d48ac",
        )

    def test_narration_only_accepts_verified_view(self):
        from mei_sdk.shared import NarrationProvider, verified_result_view

        provider = NarrationProvider()
        with self.assertRaises(ValueError):
            provider.narrate({"call_id": "x", "status": "ok", "payload": {}})
        text = provider.narrate(
            verified_result_view({
                "call_id": "x",
                "tool_name": "天气查询",
                "status": "ok",
                "payload": {"温度": 22, "天气": "晴"},
                "provenance": {"source": "host", "verified": True},
            })
        )
        self.assertEqual(text, "天气查询已执行完成，天气为晴；温度为22。")
        cancelled = provider.narrate(
            verified_result_view({
                "call_id": "y",
                "tool_name": "天气查询",
                "status": "cancelled",
                "payload": None,
                "provenance": {"source": "host", "verified": True},
            })
        )
        self.assertEqual(cancelled, "天气查询已取消，未继续执行。")
        self.assertFalse(provider.capabilities()["can_execute_tools"])

    def test_v1_is_read_only_degraded_and_rejects_traversal(self):
        package = load_package(TINY)
        self.assertEqual(package.compatibility["mode"], "read_only_degraded_adapter")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "pkg"
            shutil.copytree(TINY, target)
            manifest = json.loads((target / "mei-model.json").read_text(encoding="utf-8"))
            manifest["tokenizer"]["file"] = "../outside"
            (target / "mei-model.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(SdkError):
                load_package(target)

    def test_minimal_v2_package_and_overlap_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = b"tokenizer"
            (root / "tokenizer.model").write_bytes(tokenizer)
            entries = []
            payloads = []
            for index, role in enumerate(("lm", "contrastive", "mw_disposition", "confidence")):
                entries.append(
                    {
                        "name": role,
                        "role": role,
                        "shape": [1],
                        "n_params": 1,
                        "dtype": "f16",
                        "transform": "none",
                        "codebook": "none",
                    }
                )
                payloads.append(struct.pack("<e", float(index)))
            container = _container_bytes(entries, payloads)
            parsed_container = TensorContainer.parse(container)
            self.assertEqual(parsed_container.dequantize("confidence"), [3.0])
            entries = [
                {
                    **entry,
                    "data": {
                        "offset": parsed_container.entries[entry["name"]].data_offset,
                        "nbytes": parsed_container.entries[entry["name"]].data_nbytes,
                    },
                }
                for entry in entries
            ]
            (root / "tensors.bin").write_bytes(container)
            contrastive = parsed_container.entries["contrastive"]
            head_digest = hashlib.sha256()
            head_digest.update(b"contrastive")
            head_digest.update(
                parsed_container.blob[
                    contrastive.data_offset : contrastive.data_offset
                    + contrastive.data_nbytes
                ]
            )
            tool_index = ToolIndex(
                model_hash=_sha(container),
                head_hash=head_digest.hexdigest(),
                tokenizer_hash=_sha(tokenizer),
            )
            tool_index.build([], lambda _text: [1.0])
            tool_index.save(root / "tool-index.json")
            tool_index_bytes = (root / "tool-index.json").read_bytes()

            def head(name):
                return {
                    "present": True,
                    "trained": False,
                    "status": "untrained",
                    "tensor_prefixes": [name],
                }
            manifest = {
                "package_format": "mei-model-package-v2",
                "product": "mei-1.0-51m",
                "package_id": "tiny-v2",
                "runtime_min": "mei-runtime-abi-2",
                "parent_package_id": None,
                "contracts": {
                    "weight_contract_sha256": "c468b96453f0a377b1ffbcfef00ed9e108c82b2c44847ee5345509a181581d9b",
                    "runtime_profile_sha256": "74839b08155e624f14318ca8646166ddc68ee6496720dedac26aa91fdc8bdf43",
                    "training_aux_sha256": "83849db3926693e49c0896a58c172ae15e4b203550cee0ef12a4c37a8c1d48ac",
                },
                "architecture": {
                    "id": "mei-1.0-51m-arch-v1",
                    "d_model": 512,
                    "n_layers": 27,
                    "n_heads": 8,
                    "n_kv_heads": 4,
                    "head_dim": 64,
                    "vocab_size": 24000,
                    "max_seq_len": 2048,
                    "parameter_count": 51463797,
                    "rope_theta": 100000.0,
                    "engram_layers": [2, 15],
                    "engram_orders": [2, 3],
                    "engram_slots": 8192,
                    "engram_conv_taps": 4,
                    "mhc_lanes": 4,
                    "sinkhorn_iters": 20,
                    "tie_embeddings": True,
                    "rms_eps": 1e-6,
                    "conf_probes": 8,
                    "mlp": "FixedWalshHadamardMLP",
                    "confidence_head": True,
                },
                "runtime_profile": {
                    "max_context_tokens": 2048,
                    "stable_prefix_tokens": 1024,
                    "rolling_window_tokens": 256,
                    "default_output_tokens": 128,
                    "kv_dtype": "i8",
                    "activation_dtype": "i8",
                },
                "tokenizer": {
                    "id": "zh-24k-v1",
                    "file": "tokenizer.model",
                    "sha256": _sha(tokenizer),
                    "pad_id": 0,
                    "eos_id": 1,
                    "bos_id": 2,
                    "unk_id": 3,
                },
                "tensor_container": {
                    "file": "tensors.bin",
                    "format": "mei-cq-tensor-v2",
                    "sha256": _sha(container),
                    "payload_bytes": len(container),
                    "quant_math_id": "mei-cq-v2-g128-wht-codebook",
                    "directory": entries,
                },
                "files": [
                    {
                        "path": "tokenizer.model",
                        "sha256": _sha(tokenizer),
                        "nbytes": len(tokenizer),
                        "role": "tokenizer",
                    },
                    {
                        "path": "tensors.bin",
                        "sha256": _sha(container),
                        "nbytes": len(container),
                        "role": "tensor_container",
                    },
                    {
                        "path": "tool-index.json",
                        "sha256": _sha(tool_index_bytes),
                        "nbytes": len(tool_index_bytes),
                        "role": "tool_index",
                    },
                ],
                "heads": {name: head(name) for name in ("lm", "contrastive", "mw_disposition", "confidence")},
                "capabilities": {
                    "retrieval": True,
                    "full_call": True,
                    "mw_disposition": True,
                    "confidence": True,
                    "multi_step": True,
                },
                "training_receipts": [],
                "resources": {
                    "package_bytes": 0,
                    "rust_session_peak_bytes": 1,
                    "wasm_heap_peak_bytes": 1,
                },
                "release_class": "candidate",
            }
            _write_v2_manifest(root, manifest)
            package = load_package(root)
            self.assertEqual(package.compatibility["mode"], "native_v2")
            self.assertFalse(package.runtime_quantization_complete())
            self.assertFalse(package.tensor_identity_complete())
            self.assertFalse(package.packed_inference_ready())
            self.assertFalse(package.capabilities()["resource_eligible"])
            self.assertFalse(package.capabilities()["release_eligible"])
            from mei_sdk import package as package_contract

            manifest["runtime_quantization"] = dict(
                package_contract._CANONICAL_RUNTIME_QUANTIZATION
            )
            _write_v2_manifest(root, manifest)
            package = load_package(root)
            self.assertTrue(package.runtime_quantization_complete())
            manifest["runtime_quantization"]["activation_q_quantized"] = True
            _write_v2_manifest(root, manifest)
            with self.assertRaises(SdkError):
                load_package(root)
            manifest["runtime_quantization"] = dict(
                package_contract._CANONICAL_RUNTIME_QUANTIZATION
            )
            _write_v2_manifest(root, manifest)

            # A malicious directory with the right aggregate parameter count
            # is not the canonical 400-tensor weight identity.
            package.manifest["contracts"] = {
                "weight_contract_sha256": package_contract._CANONICAL_WEIGHT_CONTRACT_SHA256,
                "runtime_profile_sha256": package_contract._CANONICAL_RUNTIME_CONTRACT_SHA256,
                "training_aux_sha256": package_contract._CANONICAL_TRAINING_AUX_SHA256,
            }
            package.manifest["architecture"] = dict(package_contract._CANONICAL_ARCHITECTURE)
            package.manifest["runtime_profile"] = dict(
                package_contract._CANONICAL_RUNTIME_PROFILE
            )
            package.manifest["tensor_container"]["directory"] = [
                {
                    "name": "fake.weight",
                    "role": "lm",
                    "shape": [51_463_797],
                    "n_params": 51_463_797,
                }
            ]
            self.assertFalse(package.tensor_identity_complete())
            native_v2 = Engine(package).create_session().complete(
                {"query": "missing wire", "candidate_text": "[]"}
            )
            self.assertEqual(native_v2["error"]["id"], "invalid_argument")
            self.assertIn("native v2 sessions require", native_v2["error"]["message"])
            explicit_v1 = Engine(package).create_session().complete(
                {
                    "wire_version": "mei-runtime-wire-v1",
                    "query": "legacy wire on native v2",
                    "candidate_text": "[]",
                }
            )
            self.assertEqual(explicit_v1["error"]["id"], "invalid_argument")
            duplicate_receipt = b'{"kind":"duplicate-receipt"}'
            duplicate_sha = _sha(duplicate_receipt)
            for name in ("receipt-a.json", "receipt-b.json"):
                (root / name).write_bytes(duplicate_receipt)
                manifest["files"].append(
                    {
                        "path": name,
                        "sha256": duplicate_sha,
                        "nbytes": len(duplicate_receipt),
                        "role": "training_receipt",
                    }
                )
            manifest["training_receipts"].append(duplicate_sha)
            _write_v2_manifest(root, manifest)
            with self.assertRaisesRegex(SdkError, "every training_receipts"):
                load_package(root)
            manifest["training_receipts"].remove(duplicate_sha)
            manifest["files"] = [
                row
                for row in manifest["files"]
                if row["path"] not in {"receipt-a.json", "receipt-b.json"}
            ]
            (root / "receipt-a.json").unlink()
            (root / "receipt-b.json").unlink()
            receipt_payload = b"{}"
            receipt_sha = _sha(receipt_payload)
            (root / "mw-receipt.json").write_bytes(receipt_payload)
            manifest["files"].append(
                {
                    "path": "mw-receipt.json",
                    "sha256": receipt_sha,
                    "nbytes": len(receipt_payload),
                    "role": "training_receipt",
                }
            )
            manifest["training_receipts"].append(receipt_sha)
            manifest["heads"]["mw_disposition"].update(
                {
                    "trained": True,
                    "status": "ready",
                    "training_receipt_sha256": receipt_sha,
                }
            )
            _write_v2_manifest(root, manifest)
            with self.assertRaisesRegex(SdkError, "heads.mw_disposition"):
                load_package(root)
            manifest["heads"]["mw_disposition"] = head("mw_disposition")
            manifest["training_receipts"].remove(receipt_sha)
            manifest["files"] = [
                row for row in manifest["files"] if row["path"] != "mw-receipt.json"
            ]
            (root / "mw-receipt.json").unlink()
            _write_v2_manifest(root, manifest)
            (root / "undeclared.bin").write_bytes(b"x")
            with self.assertRaisesRegex(SdkError, "inventory mismatch"):
                load_package(root)
            (root / "undeclared.bin").unlink()
            manifest["unexpected"] = True
            _write_v2_manifest(root, manifest)
            with self.assertRaisesRegex(SdkError, "unexpected field"):
                load_package(root)
            manifest.pop("unexpected")
            manifest["tensor_container"]["directory"][1]["data"]["offset"] = manifest[
                "tensor_container"
            ]["directory"][0]["data"]["offset"]
            _write_v2_manifest(root, manifest)
            with self.assertRaisesRegex(SdkError, "overlapping"):
                load_package(root)


if __name__ == "__main__":
    unittest.main()
