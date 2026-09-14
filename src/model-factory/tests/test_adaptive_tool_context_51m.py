from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from training.tool_use.adaptive_tool_context_51m import (
    iter_alignment_views,
    iter_mw_visible_batches,
    select_alignment_replay_views,
    select_alignment_source_rows,
)
import release.freeze_sft_v3_release_51m as freeze_sft
import contracts.sft_v4_contract_51m as contract
import training.tool_use.sft_v3_training_51m as training
from mei_sdk.protocol import render_budgeted_request
from mei_sdk.shared import RankedCandidate


class _Tokenizer:
    eos_id = 1

    def encode(self, text, add_bos=False, add_eos=False):
        return ([2] if add_bos else []) + [ord(char) for char in str(text)] + (
            [1] if add_eos else []
        )

    def decode(self, ids):
        return "".join(chr(value) for value in ids if value not in {1, 2})


class _Runtime:
    tokenizer = _Tokenizer()

    def search_ranked(self, _query, catalog):
        # Raw scores are monotonically decreasing and remain above the fixture
        # expand threshold after sigmoid calibration.
        return [
            RankedCandidate(
                tool_id=tool["name"],
                schema=tool,
                raw_score=2.0 - index * 0.05,
                relevance=0.0,
                rank=index + 1,
            )
            for index, tool in enumerate(catalog)
        ]


class AdaptiveToolContext51MTest(unittest.TestCase):
    def setUp(self):
        self.catalog = [
            {
                "name": f"tool.{index:02d}",
                "description": "执行测试动作",
                "parameters": {"type": "object", "properties": {}},
            }
            for index in range(10)
        ]
        self.calibration = {
            "scale": 1.0,
            "bias": 0.0,
            "discard_threshold": 0.2,
            "expand_threshold": 0.6,
        }

    def test_ready_target_in_second_batch_relabels_first_only(self):
        source = {
            "sample_id": "mw-later-gold",
            "split": "train",
            "query": "执行第六项",
            "reason_class_id": 0,
            "reason_code": "ready_to_execute",
            "candidate_tool": "tool.05",
            "retrieved_tools": [f"tool.{index:02d}" for index in range(5)],
            "context": {},
            "evidence": [],
            "permissions": {},
            "state": {},
            "history": [],
            "tool_results": [],
        }
        views = list(
            iter_mw_visible_batches(
                _Runtime(),
                [source],
                self.catalog,
                self.calibration,
                retrieval_mode="learned",
                profile="compact",
            )
        )
        self.assertEqual([row["batch_index"] for row in views], [1, 2])
        self.assertEqual(
            [row["effective_reason_class_id"] for row in views], [10, 0]
        )
        self.assertEqual(views[0]["effective_reason_code"], "capability_insufficient")
        self.assertEqual(views[1]["visible_tools"][0], "tool.05")
        for view in views:
            self.assertLessEqual(view["prompt_tokens"] + 128, 2048)
            self.assertLessEqual(view["schema_budget"]["used"], 1024)

    def test_non_ready_mw_class_is_not_changed_by_batches(self):
        source = {
            "sample_id": "mw-permission",
            "split": "train",
            "query": "执行",
            "reason_class_id": 4,
            "reason_code": "permission_insufficient",
            "retrieved_tools": [f"tool.{index:02d}" for index in range(5)],
            "context": {},
            "evidence": [],
            "permissions": {},
            "state": {},
        }
        views = list(
            iter_mw_visible_batches(
                _Runtime(),
                [source],
                self.catalog,
                self.calibration,
                retrieval_mode="learned",
                profile="standard",
            )
        )
        self.assertEqual(len(views), 1)
        self.assertEqual({row["effective_reason_class_id"] for row in views}, {4})

    def test_all_below_discard_becomes_retrieval_terminal_not_mw_row(self):
        calibration = {
            **self.calibration,
            "scale": 1e-6,
            "bias": -20.0,
            "discard_threshold": 0.2,
            "expand_threshold": 0.6,
        }
        source = {
            "sample_id": "mw-no-match",
            "split": "train",
            "query": "没有工具",
            "reason_class_id": 10,
            "reason_code": "capability_insufficient",
            "retrieved_tools": [f"tool.{index:02d}" for index in range(5)],
        }
        views = list(
            iter_mw_visible_batches(
                _Runtime(),
                [source],
                self.catalog,
                calibration,
                retrieval_mode="learned",
                profile="compact",
            )
        )
        self.assertEqual(len(views), 1)
        self.assertFalse(views[0]["mw_eligible"])
        self.assertEqual(views[0]["retrieval_terminal"], "retrieval_no_match")

    def test_alignment_replay_materializes_later_batch_and_encoder_uses_it(self):
        sample_id = next(
            f"alignment-{index}"
            for index in range(100)
            if int(hashlib.sha256(f"alignment-{index}".encode()).hexdigest()[:8], 16)
            % 4
            in {2, 3}
        )
        source = {
            "sample_id": sample_id,
            "split": "train",
            "kind": "execute",
            "query": "执行第六项",
            "gold_name": "tool.05",
            "answers": [{"name": "tool.05", "arguments": {}}],
            "retrieved_tools": [f"tool.{index:02d}" for index in range(5)],
            "context": {},
            "evidence": [],
            "permissions": {},
            "state": {},
            "history": [],
            "tool_results": [],
        }
        views = list(
            iter_alignment_views(
                _Runtime(),
                [source],
                self.catalog,
                self.calibration,
                task="full_call",
            )
        )
        self.assertEqual(
            [row["batch_outcome"] for row in views],
            ["capability_insufficient"],
        )
        self.assertEqual(views[0]["answers"], [])
        self.assertEqual(views[0]["kind"], "capability_insufficient")
        for view in views:
            prompt, answer, stats = training.encode_fullcall_row(
                _Runtime.tokenizer, view, {}
            )
            self.assertTrue(stats["schema_budget_alignment"])
            self.assertEqual(len(prompt), view["_budgeted_prompt_tokens"])
            self.assertLessEqual(len(prompt) + 128, 2048)
            self.assertLessEqual(len(answer), 128)

    def test_alignment_call_keeps_source_kind(self):
        sample_id = next(
            f"alignment-{index}"
            for index in range(100)
            if int(hashlib.sha256(f"alignment-{index}".encode()).hexdigest()[:8], 16)
            % 4
            in {2, 3}
        )
        source = {
            "sample_id": sample_id,
            "split": "train",
            "kind": "execute",
            "query": "执行第一项",
            "gold_name": "tool.01",
            "answers": [{"name": "tool.01", "arguments": {}}],
            "retrieved_tools": [f"tool.{index:02d}" for index in range(5)],
            "context": {},
            "evidence": [],
            "permissions": {},
            "state": {},
            "history": [],
            "tool_results": [],
        }
        views = list(
            iter_alignment_views(
                _Runtime(),
                [source],
                self.catalog,
                self.calibration,
                task="full_call",
            )
        )
        self.assertEqual([row["batch_outcome"] for row in views], ["call"])
        self.assertEqual(views[0]["kind"], "execute")
        self.assertEqual(views[0]["answers"][0]["name"], "tool.01")

    def test_alignment_selector_freezes_exact_25_25_50_mix(self):
        rows = []
        for mode, profile in (
            ("anchor_top5", "standard"),
            ("adaptive", "standard"),
            ("adaptive", "compact"),
        ):
            for index in range(40):
                rows.append(
                    {
                        "view_id": f"{mode}-{profile}-{index}",
                        "retrieval_mode": mode,
                        "runtime_profile": profile,
                        "_training_bank": ("structural", "schema")[index % 2],
                        "kind": ("execute", "refuse")[index % 2],
                        "batch_outcome": ("call", "capability_insufficient")[index % 2],
                        "batch_index": index % 4 + 1,
                    }
                )
        selected = select_alignment_replay_views(rows, limit=40)
        counts = {}
        for row in selected:
            key = f"{row['retrieval_mode']}:{row['runtime_profile']}"
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(
            counts,
            {
                "anchor_top5:standard": 10,
                "adaptive:standard": 10,
                "adaptive:compact": 20,
            },
        )
        self.assertEqual(len({row["view_id"] for row in selected}), 40)

    def test_alignment_source_presample_reduces_expensive_retrieval(self):
        rows = [{"sample_id": f"source-{index}"} for index in range(2_000)]
        selected = select_alignment_source_rows(rows, view_limit=400)
        counts = {}
        for row in selected:
            bucket = int(
                hashlib.sha256(str(row["sample_id"]).encode()).hexdigest()[:8], 16
            ) % 4
            key = "anchor" if bucket == 0 else "standard" if bucket == 1 else "compact"
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts, {"anchor": 100, "standard": 200, "compact": 400})
        self.assertEqual(len(selected), 700)

    def test_python_budget_projection_matches_browser_core_golden(self):
        fixture = json.loads(
            (
                next(parent for parent in Path(__file__).resolve().parents if (parent / "CURRENT.json").is_file())
                / "src/platform/_shared/spec/golden/context_budget_v2.json"
            ).read_text(encoding="utf-8")
        )
        recipe = fixture["recipe"]
        tools = [
            {
                "name": f"device.action_{index}",
                "description": recipe["tool_description_unit"]
                * recipe["tool_description_repeat"],
                "parameters": {
                    "type": "object",
                    "title": {"description": "注释内部描述不得参与配额" * 30},
                    "examples": [
                        {"level": 3, "description": "示例内部描述同样整体删除" * 30}
                    ],
                    "properties": {
                        "level": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10,
                            "multipleOf": 1,
                            "description": recipe["parameter_description_unit"]
                            * recipe["parameter_description_repeat"],
                        },
                        "mode": {
                            "type": ["string", "null"],
                            "enum": ["auto", "eco", None],
                            "description": recipe["parameter_description_unit"]
                            * recipe["parameter_description_repeat"],
                        },
                    },
                    "required": ["level"],
                },
            }
            for index in range(recipe["tool_count"])
        ]
        history = recipe["history_unit"] * recipe["history_repeat"]
        request = {
            "wire_version": "mei-runtime-wire-v2",
            "query": recipe["query_unit"] * recipe["query_repeat"],
            "context": {
                "low": {"priority": 0, "text": "低优先上下文" * 200},
                "high": {"priority": 10, "text": "高优先上下文" * 200},
            },
            "evidence": [
                {"priority": 0, "text": "低优先证据" * 200},
                {"priority": 10, "text": "高优先证据" * 200},
            ],
            "permissions": {"scopes": ["device:write"]},
            "state": {"device": "online"},
            "history": [
                {"role": "user", "content": history},
                {"role": "assistant", "content": history},
                {"role": "user", "content": history},
            ],
            "tool_results": [
                {
                    "wire_version": "mei-runtime-wire-v2",
                    "call_id": call_id,
                    "status": "ok",
                    "payload": {"text": text * 500},
                    "provenance": {"source": "fixture", "verified": True},
                }
                for call_id, text in (("old-call", "旧结果"), ("latest-call", "最新结果"))
            ],
        }
        result = render_budgeted_request(
            request,
            tools,
            freeze_sft._load_tokenizer(),
            relevances=[0.95, 0.85, 0.75, 0.65, 0.55],
            runtime_profile=fixture["profile"],
            output_reserve=fixture["output_reserve"],
            already_normalized=True,
        )
        expected = fixture["expected"]
        self.assertEqual(result["selected_tools"], expected["selected_tools"])
        self.assertEqual(
            result["schema_projection_sha256"],
            expected["schema_projection_sha256"],
        )
        self.assertEqual(result["prompt_tokens"], expected["prompt_tokens"])
        self.assertEqual(result["schema_budget"]["used"], expected["schema_tokens"])
        self.assertEqual(
            result["prompt"].endswith(contract.ASSISTANT_SUFFIX),
            expected["assistant_suffix"],
        )
        self.assertEqual(
            result["input_budget"]["compacted_tool_results"],
            expected["compacted_tool_results"],
        )


if __name__ == "__main__":
    unittest.main()
