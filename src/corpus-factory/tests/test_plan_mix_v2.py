from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sources = load_module(
    "mei_51m_source_manager", "src/corpus-factory/sources/source_manager.py"
)

CAPACITIES = {
    "fineweb2_hq": 920_000_000,
    "wiki_zh": 560_000_000,
    "wiki_en": 70_000_000,
    "dialogue": 280_000_000,
    "structured": 175_000_000,
    "code": 140_000_000,
}
CONSUMED = {role: 0 for role in CAPACITIES}


class PlanMixV2Tests(unittest.TestCase):
    def test_quotas_must_match_actual_increment(self) -> None:
        with self.assertRaisesRegex(sources.SourceError, "sum to target"):
            sources.plan_mix(301, {"dialogue": 400}, {}, quotas={"dialogue": 300})

    def test_consumption_cannot_exceed_physical_capacity(self) -> None:
        with self.assertRaisesRegex(sources.SourceError, "physical capacity"):
            sources.plan_mix(10, {"dialogue": 100}, {"dialogue": 110}, quotas={"dialogue": 10})

    def test_capacity_pass_does_not_assert_naturalness_or_adoption(self) -> None:
        result = sources.plan_mix(10, {"dialogue": 100}, {}, quotas={"dialogue": 10})
        self.assertIsNone(result["synthetic_fraction"])
        self.assertFalse(result["training_adoption_eligible"])
        self.assertEqual(result["validation_scope"], "quota_capacity_only")

    def test_fraction_mode_assigns_quotas_with_floor_last_role(self) -> None:
        result = sources.plan_mix(
            300_000_000,
            CAPACITIES,
            CONSUMED,
            fractions={
                "fineweb2_hq": 0.41,
                "wiki_zh": 0.27,
                "dialogue": 0.13,
                "structured": 0.08,
                "code": 0.07,
                "wiki_en": 0.04,
            },
        )
        self.assertEqual("passed", result["status"])
        self.assertEqual(300_000_000, sum(result["quotas"].values()))
        self.assertEqual("floor_last_role", result["policy"]["rounding"])
        self.assertEqual({}, result["shortages"])
        self.assertTrue(result["policy"]["needs_reason"])

    def test_fraction_sum_must_equal_one(self) -> None:
        with self.assertRaisesRegex(sources.SourceError, "sum to 1"):
            sources.plan_mix(
                300,
                CAPACITIES,
                CONSUMED,
                fractions={"fineweb2_hq": 0.5, "wiki_zh": 0.4},
            )

    def test_quota_mode_blocks_when_pool_is_short(self) -> None:
        result = sources.plan_mix(
            300,
            {"fineweb2_hq": 100, "wiki_zh": 100},
            {"fineweb2_hq": 0, "wiki_zh": 0},
            quotas={"fineweb2_hq": 150, "wiki_zh": 150},
        )
        self.assertEqual("blocked_insufficient_pool", result["status"])
        self.assertEqual({"fineweb2_hq": 50, "wiki_zh": 50}, result["shortages"])
        self.assertFalse(result["allow_repeat"])

    def test_unknown_role_is_rejected(self) -> None:
        with self.assertRaisesRegex(sources.SourceError, "unknown roles"):
            sources.plan_mix(
                300,
                CAPACITIES,
                CONSUMED,
                quotas={"not_a_role": 300},
            )

    def test_reason_suppresses_needs_reason_flag(self) -> None:
        result = sources.plan_mix(
            300_000_000,
            CAPACITIES,
            CONSUMED,
            fractions={"fineweb2_hq": 0.5, "wiki_zh": 0.5},
            candidate_id="mix-zhv2-300m-c01-v1",
            supersedes=None,
            reasons=["初始 v2 比例，登记于 2026-09-04"],
        )
        self.assertFalse(result["policy"]["needs_reason"])
        self.assertEqual("mix-zhv2-300m-c01-v1", result["candidate_id"])

    def test_hq_fraction_alias_maps_legacy_wiki_to_wiki_zh(self) -> None:
        result = sources.plan_mix(
            300,
            {"wiki": 200, "fineweb2_hq": 100},
            {"wiki": 0, "fineweb2_hq": 0},
            hq_fraction=0.65,
        )
        self.assertEqual("blocked_insufficient_pool", result["status"])
        self.assertEqual({"fineweb2_hq": 95}, result["shortages"])
        self.assertEqual("legacy_hq_fraction", result["policy"]["rounding"])
        self.assertEqual("hq_fraction", result["deprecated_input"])
        self.assertEqual("mei-51m-cpt-natural-mix-candidate-v2", result["schema"])

    def test_exactly_one_input_mode_is_required(self) -> None:
        with self.assertRaisesRegex(sources.SourceError, "exactly one"):
            sources.plan_mix(300, CAPACITIES, CONSUMED)
        with self.assertRaisesRegex(sources.SourceError, "exactly one"):
            sources.plan_mix(
                300,
                CAPACITIES,
                CONSUMED,
                hq_fraction=0.5,
                fractions={"fineweb2_hq": 0.5, "wiki_zh": 0.5},
            )

    def test_parse_fraction_values_validates_range(self) -> None:
        with self.assertRaisesRegex(sources.SourceError, "in \\(0, 1\\]"):
            sources.parse_fraction_values(["wiki_zh=1.5"], name="--fraction")


if __name__ == "__main__":
    unittest.main()
