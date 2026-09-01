from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import freeze_sft_v3_release_51m as freeze_sft
import sft_v3_contract_51m as contract
import sft_v3_training_51m as training


class SftV3TrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.release = contract.DEFAULT_RELEASE_ROOT / contract.RELEASE_ID
        universe = contract.load_json(cls.release / "tool-universe.json")
        cls.tools_by_name = {str(tool["name"]): tool for tool in universe["tools"]}
        cls.tokenizer = freeze_sft._load_tokenizer()

    def test_release_hashes_and_identity_are_complete(self):
        manifest = training.verify_release_contract(self.release)
        self.assertEqual(manifest["release_id"], contract.RELEASE_ID)
        self.assertEqual(
            manifest["retrieval_encoding"]["max_tokens"],
            contract.RETRIEVAL_MAX_TOKENS,
        )

    def test_prompt_is_gold_independent_and_exactly_framed(self):
        row = contract.load_jsonl(self.release / "full-call.train.jsonl")[0]
        selected = training.selected_tools_for_row(row, self.tools_by_name)
        first = training.render_fullcall_prompt_parts(row, selected)
        changed = copy.deepcopy(row)
        changed["answers"] = []
        changed["gold_name"] = "forged_gold"
        changed["gold_args"] = {"forged": True}
        changed["target_text"] = "[]"
        second = training.render_fullcall_prompt_parts(changed, selected)
        self.assertEqual(first, second)
        self.assertTrue(first["prompt"].endswith(contract.ASSISTANT_SUFFIX))
        self.assertNotIn("gold_name", first["prompt"])
        changed["slot_provenance"] = [{"arg": "gold", "value": "leak"}]
        with self.assertRaisesRegex(RuntimeError, "slot provenance"):
            training.render_fullcall_prompt_parts(changed, selected)

    def test_encoding_obeys_sink_ring_and_target_contract(self):
        rows = contract.load_jsonl(self.release / "full-call.train.jsonl")
        for row in rows[:: max(1, len(rows) // 64)]:
            prompt, answer, stats = training.encode_fullcall_row(
                self.tokenizer, row, self.tools_by_name
            )
            self.assertLessEqual(
                stats["stable_prefix_tokens"], contract.STABLE_PREFIX_TOKENS_MAX
            )
            self.assertLessEqual(
                stats["ordinary_tokens_retained"], contract.ROLLING_WINDOW_TOKENS
            )
            self.assertLessEqual(stats["answer_tokens"], 128)
            self.assertEqual(len(prompt), stats["prompt_tokens"])
            self.assertEqual(len(answer), stats["answer_tokens"])

    def test_agent_call_and_result_are_interleaved(self):
        rows = contract.load_jsonl(self.release / "agent-continuation.train.jsonl")
        audit = contract.audit_agent_rows(
            rows,
            list(self.tools_by_name.values()),
            expected_split="train",
            minimum_unique_call_tools=147,
            minimum_terminal_tools=147,
        )
        self.assertEqual(audit["status"], "passed", audit["errors"][:3])
        row = next(value for value in rows if value.get("tool_results"))
        selected = training.selected_tools_for_row(row, self.tools_by_name)
        prompt = training.render_fullcall_prompt_parts(row, selected)["prompt"]
        self.assertLess(prompt.index("assistant："), prompt.index("tool："))
        self.assertLess(prompt.index("tool："), prompt.index("user："))

    def test_epoch_samplers_cover_every_row_once(self):
        fullcall = contract.load_jsonl(self.release / "full-call.train.jsonl")
        order = training.fullcall_epoch_order(fullcall, 0)
        self.assertEqual(len(order), len(fullcall))
        self.assertEqual(set(order), set(range(len(fullcall))))
        first_kinds = [fullcall[index]["kind"] for index in order[:8]]
        self.assertEqual(first_kinds, ["execute", "refuse"] * 4)

        agent = contract.load_jsonl(self.release / "agent-continuation.train.jsonl")
        agent_order = training.agent_epoch_order(agent, 0)
        self.assertEqual(len(agent_order), len(agent))
        self.assertEqual(set(agent_order), set(range(len(agent))))

    def test_retrieval_schedule_exposes_structural_and_natural_rows(self):
        natural_release = (
            contract.DEFAULT_RELEASE_ROOT
            / "mei-1.0-51m-tool-sft-natural-aug300m-v1"
        )
        structural = contract.load_jsonl(self.release / "retrieval.train.jsonl")
        natural = contract.load_jsonl(
            natural_release / "natural-retrieval.train.jsonl"
        )
        rows = [*structural, *natural]
        schedule = training.retrieval_training_schedule(rows, 1_200, 8)
        self.assertEqual(schedule, training.retrieval_training_schedule(rows, 1_200, 8))
        used = {index for batch in schedule for index in batch}
        self.assertEqual(used, set(range(len(rows))))
        self.assertTrue(
            all(
                len({rows[index]["gold_tool"] for index in batch}) == 8
                for batch in schedule
            )
        )
        natural_start = len(structural)
        self.assertTrue(set(range(natural_start, len(rows))) <= used)

    def test_confidence_uses_mean_token_log_probability(self):
        score = training.combined_confidence_score(10.0, -100.0, 100)
        self.assertAlmostEqual(score, 0.36787944117, places=6)
        self.assertGreater(score, 0.3)

    def test_confidence_sampler_is_stratified_and_unique(self):
        rows = contract.load_jsonl(self.release / "confidence-harvest.train.jsonl")
        sampled = training.confidence_candidate_sample(rows, 1200)
        self.assertEqual(len(sampled), 1200)
        self.assertEqual(len({row["sample_id"] for row in sampled}), 1200)
        self.assertEqual({row["expected_kind"] for row in sampled}, {"call", "refuse"})
        self.assertEqual(len({row["candidate_tool"] for row in sampled}), 147)

    def test_mw_learned_miss_has_explicit_capability_label(self):
        rows = contract.load_jsonl(self.release / "mw-disposition.train.jsonl")
        row = next(
            value
            for value in rows
            if value.get("reason_class_id") == 0 and value.get("candidate_tool")
        )
        oracle, label, reason = training.mw_training_view(row, self.tools_by_name)
        self.assertEqual(label, 0)
        self.assertEqual(reason, "ready_to_execute")
        learned_names = [
            name
            for name in self.tools_by_name
            if name != row["candidate_tool"]
        ][:5]
        learned, label, reason = training.mw_training_view(
            row, self.tools_by_name, learned_names
        )
        self.assertEqual(len(oracle), 5)
        self.assertEqual(len(learned), 5)
        self.assertEqual(label, 10)
        self.assertEqual(reason, "capability_insufficient")

    def test_all_mw_splits_fit_the_stable_prefix_budget(self):
        banks = {
            "train": self.release / "mw-disposition.train.jsonl",
            "valid": self.release / "mw-disposition.valid.jsonl",
            "dev": contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID / "mw.dev.jsonl",
            "test": contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID / "mw.test.jsonl",
        }
        for split, path in banks.items():
            with self.subTest(split=split):
                maximum = 0
                for row in contract.load_jsonl(path):
                    if row.get("oracle_top5"):
                        selected = list(row["oracle_top5"])
                    else:
                        selected = [
                            self.tools_by_name[name]
                            for name in row.get("retrieved_tools") or []
                        ][:5]
                    rendered = training.render_mw_prompt_parts(row, selected)
                    self.assertEqual(rendered["prompt_id"], training.MW_PROMPT_ID)
                    _ids, stats = training.encode_stable_ring_prompt(
                        self.tokenizer, rendered
                    )
                    maximum = max(maximum, stats["stable_prefix_tokens"])
                self.assertLessEqual(maximum, contract.STABLE_PREFIX_TOKENS_MAX)


if __name__ == "__main__":
    unittest.main()
