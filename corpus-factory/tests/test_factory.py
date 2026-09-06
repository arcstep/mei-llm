from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "corpus-factory/generators/factory_51m.py"
SPEC = importlib.util.spec_from_file_location("mei_corpus_factory_v3_tested", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
factory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = factory
SPEC.loader.exec_module(factory)

SFT_RELEASE = ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/sft-suite/historical-notebook-releases/releases/mei-1.0-51m-tool-sft-v4-300m-v4"
DEPLOY_UNIVERSE = SFT_RELEASE / "tool-universe.json"
TRAINING_UNIVERSE = SFT_RELEASE / "training-tool-universe.json"
MW_CODEBOOK = SFT_RELEASE / "mw-disposition-codebook-v1.json"
MW_DEFINITIONS = SFT_RELEASE / "mw-reason-definitions-v2-20class.json"
EVAL_LOCK = ROOT / "artifacts/mei-1.0-51m/legacy/_legacy/notebook/corpus/mei-1.0-51m/factory-v1/eval/phase1-lock/lock.json"
PILOT_RELEASE = ROOT / "artifacts/mei-1.0-51m/legacy/exp-000600m/corpus/sft-suite/factory-v3/releases/mei-1.0-51m-sft-gap-pilot-v1/release-manifest.json"
QAT_BINDING = ROOT / "artifacts/mei-1.0-51m/legacy/exp-000600m/corpus/sft-suite/factory-v3/manifests/qat-base300-sft-v4-binding-v1.json"
EVAL_V7_LOCK = ROOT / "artifacts/mei-1.0-51m/legacy/_legacy/notebook/evaluation/banks/mei-51m-longitudinal-eval-v7/lock.json"


def baseline_roles() -> dict[str, Path]:
    run = ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/runs/productize-scratch300m-sft-v4-quality-schema-cq2-v2-finalization-dcfa3bc208e0"
    return {
        "current": ROOT / "CURRENT.json",
        "base_300": ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/models/base/mei-1.0-51m-base-scratch300m-v1/RELEASE.json",
        "base_600": ROOT / "artifacts/mei-1.0-51m/legacy/exp-000600m/models/base/mei-1.0-51m-base-cpt600m-clean-source-v3-v1/RELEASE.json",
        "lm_release": ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/cpt-delta/lm-v1/RELEASE.json",
        "unique_ledger": ROOT / "artifacts/mei-1.0-51m/legacy/_legacy/notebook/corpus/lm-v1/assemble/work/zh-pretrain-v4/unique-ledger.json",
        "sft_v4_manifest": SFT_RELEASE / "manifest.json",
        "narration_manifest": ROOT / "artifacts/mei-1.0-51m/legacy/exp-000300m/corpus/sft-suite/historical-notebook-releases/releases/mei-1.0-51m-narration-sft-agent300m-v3/manifest.json",
        "qat_receipt": ROOT / "artifacts/mei-1.0-51m/legacy/_legacy/notebook/evaluation/jobs/mei-1.0-51m/qat-q4-rung-5m.json",
        "locked_eval_receipt": run / "stages/locked_test_eval_v4/receipt.json",
        "narration_eval_receipt": run / "downstream/narration-package-eval/receipt.json",
        "deploy_tool_universe": DEPLOY_UNIVERSE,
        "training_tool_universe": TRAINING_UNIVERSE,
        "mw_codebook": MW_CODEBOOK,
        "mw_definitions": MW_DEFINITIONS,
        "eval_lock": EVAL_LOCK,
        "factory_v2_index": ROOT / "artifacts/mei-1.0-51m/legacy/_legacy/notebook/corpus/mei-1.0-51m/factory-v2/indexes/sft-pilot-v1.json",
    }


class CorpusFactoryV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temp_parent = ROOT / "artifacts/mei-1.0-51m/legacy/exp-000600m/corpus/sft-suite"
        cls._temporary = tempfile.TemporaryDirectory(prefix="factory-v3-test-", dir=temp_parent)
        cls.temp = Path(cls._temporary.name)
        cls.baseline = cls.temp / "baseline.json"
        cls.ledger = cls.temp / "ledger.json"
        factory.inventory_baselines(baseline_roles(), ROOT, cls.baseline)
        factory.build_demand_ledger(cls.baseline, cls.ledger)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_baseline_inventory_uses_real_counts_and_zero_generation(self) -> None:
        inventory = factory.load_json(self.baseline)
        self.assertEqual(51_463_797, inventory["facts"]["product_contract"]["deployed_lm_params"])
        self.assertEqual(300_000_485, inventory["facts"]["base_300"]["tokens_seen_exposure"])
        self.assertFalse(inventory["facts"]["base_600"]["corpus_reuse_eligible"])
        self.assertEqual(147, inventory["facts"]["sft_v4"]["deploy_tool_count"])
        self.assertEqual(64, inventory["facts"]["sft_v4"]["training_only_schema_tool_count"])
        self.assertEqual(1_388, inventory["facts"]["sft_v4"]["train_trajectories"])
        self.assertEqual(39_973, inventory["facts"]["sft_v4"]["compiled_jsonl_rows"])
        self.assertEqual(5_001_216, inventory["facts"]["qat"]["tokens_seen"])
        self.assertEqual(0, inventory["generated_semantic_tasks"])
        self.assertEqual(0, inventory["provider_calls"])
        self.assertFalse(inventory["current_mutated"])

    def test_demand_ledger_is_complete_and_verified(self) -> None:
        verification = factory.verify_demand_ledger(self.ledger, self.baseline)
        self.assertEqual("passed", verification["status"])
        ledger = factory.load_json(self.ledger)
        self.assertEqual(160, ledger["sft_policy"]["pilot_semantic_task_target"])
        self.assertEqual([], ledger["high_priority_unclassified_cells"])
        self.assertEqual(
            {"fineweb2_hq_fraction": 0.65, "wiki_fraction": 0.35, "synthetic_fraction": 0.0},
            ledger["cpt_policy"]["default_increment_mix"],
        )
        self.assertEqual("blocked_on_document_level_index", ledger["cpt_policy"]["status"])

    def test_all_four_worklists_are_exactly_40_and_32_4_4(self) -> None:
        for cell_id in factory.PILOT_CELLS:
            out = self.temp / "worklists" / cell_id.replace(".", "-")
            receipt = factory.create_worklist(
                self.ledger,
                cell_id,
                DEPLOY_UNIVERSE,
                TRAINING_UNIVERSE,
                MW_CODEBOOK,
                MW_DEFINITIONS,
                out,
            )
            self.assertEqual(40, receipt["semantic_task_count"])
            rows = factory.load_jsonl(out / "worklist.jsonl")
            self.assertEqual({"train": 32, "dev": 4, "internal_holdout": 4}, dict(factory.Counter(row["split"] for row in rows)))

    def _build_pilot(self) -> tuple[list[Path], list[Path], list[Path]]:
        candidate_paths: list[Path] = []
        build_dirs: list[Path] = []
        audit_paths: list[Path] = []
        for cell_id in factory.PILOT_CELLS:
            slug = cell_id.replace(".", "-")
            worklist_dir = self.temp / "e2e/worklists" / slug
            candidate_path = self.temp / "e2e/candidates" / f"{slug}.jsonl"
            ingest_dir = self.temp / "e2e/samples" / slug / "ingest"
            build_dir = self.temp / "e2e/samples" / slug / "build"
            audit_path = self.temp / "e2e/samples" / slug / "audit.json"
            factory.create_worklist(self.ledger, cell_id, DEPLOY_UNIVERSE, TRAINING_UNIVERSE, MW_CODEBOOK, MW_DEFINITIONS, worklist_dir)
            factory.draft_candidates(worklist_dir / "worklist.jsonl", DEPLOY_UNIVERSE, TRAINING_UNIVERSE, MW_CODEBOOK, candidate_path)
            factory.ingest_candidates(
                worklist_dir / "worklist.jsonl",
                candidate_path,
                DEPLOY_UNIVERSE,
                TRAINING_UNIVERSE,
                MW_CODEBOOK,
                ROOT,
                ingest_dir,
            )
            factory.compile_shard(ingest_dir / "accepted.jsonl", DEPLOY_UNIVERSE, TRAINING_UNIVERSE, build_dir)
            audit = factory.audit_shard(ingest_dir / "accepted.jsonl", build_dir, EVAL_LOCK, audit_path)
            self.assertEqual("passed", audit["status"])
            self.assertEqual([], audit["duplicate_audit"]["exact_training_input_duplicates"])
            candidate_paths.append(ingest_dir / "accepted.jsonl")
            build_dirs.append(build_dir)
            audit_paths.append(audit_path)
        return candidate_paths, build_dirs, audit_paths

    def test_end_to_end_pilot_is_160_semantic_tasks_and_460_compiled_rows(self) -> None:
        candidate_paths, build_dirs, audit_paths = self._build_pilot()
        merged_path = self.temp / "e2e/merged-audit.json"
        merged = factory.merge_audits(audit_paths, merged_path)
        self.assertEqual("passed", merged["status"])
        self.assertEqual(160, merged["semantic_task_count"])
        self.assertEqual(460, merged["compiled_row_count"])

        bad_audit = factory.load_json(audit_paths[0])
        bad_audit["compiled_row_count"] = bad_audit["semantic_task_count"]
        bad_audit_path = self.temp / "e2e/compiled-count-masquerade.json"
        bad_audit_path.write_bytes(factory.pretty_bytes(bad_audit))
        blocked_merge = factory.merge_audits(
            [bad_audit_path, *audit_paths[1:]],
            self.temp / "e2e/compiled-count-masquerade-merged.json",
        )
        self.assertEqual("blocked", blocked_merge["status"])
        review_path = self.temp / "e2e/review-plan.json"
        review = factory.prepare_review(candidate_paths, merged_path, review_path)
        self.assertEqual(160, review["semantic_task_count"])
        self.assertEqual(1.0, review["policy"]["semantic_task_review_fraction"])
        self.assertEqual("pending_human_review", review["status"])

        verification_path = self.temp / "e2e/demand-verification.json"
        verification_path.write_bytes(factory.pretty_bytes(factory.verify_demand_ledger(self.ledger, self.baseline)))
        cpt_policy_path = self.temp / "e2e/cpt-policy.json"
        validation_path = self.temp / "e2e/sft-validation.json"
        factory.freeze_cpt_policy(self.ledger, cpt_policy_path)
        factory.freeze_sft_validation_contract(self.ledger, self.baseline, validation_path)
        named_paths = {
            "baseline_inventory": self.baseline,
            "demand_ledger": self.ledger,
            "demand_verification": verification_path,
            "cpt_policy": cpt_policy_path,
            "sft_validation_contract": validation_path,
            "merged_audit": merged_path,
            "review_plan": review_path,
            "factory_script": SCRIPT,
            "factory_tests": Path(__file__),
        }
        for index, path in enumerate(audit_paths):
            named_paths[f"audit_{index}"] = path
        for index, path in enumerate(candidate_paths):
            named_paths[f"accepted_{index}"] = path
        for index, path in enumerate(build_dirs):
            named_paths[f"build_manifest_{index}"] = path / "build-manifest.json"
        index_path = self.temp / "e2e/pilot-index.json"
        index_result = factory.freeze_pilot_index(named_paths, ROOT, index_path)
        self.assertEqual("pending_human_review", index_result["status"])
        self.assertEqual("passed", factory.verify_lineage(index_path, ROOT)["status"])

    def test_eval_path_is_rejected_during_ingest(self) -> None:
        cell_id = "sft.full_call.boundary"
        worklist_dir = self.temp / "negative/worklist"
        candidate_path = self.temp / "negative/candidates.jsonl"
        factory.create_worklist(self.ledger, cell_id, DEPLOY_UNIVERSE, TRAINING_UNIVERSE, MW_CODEBOOK, MW_DEFINITIONS, worklist_dir)
        factory.draft_candidates(worklist_dir / "worklist.jsonl", DEPLOY_UNIVERSE, TRAINING_UNIVERSE, MW_CODEBOOK, candidate_path)
        rows = factory.load_jsonl(candidate_path)
        rows[0]["source_paths"] = [factory.root_relative(EVAL_LOCK)]
        work = factory.load_jsonl(worklist_dir / "worklist.jsonl")
        work[0]["source_paths"] = [factory.root_relative(EVAL_LOCK)]
        escaped_worklist = self.temp / "negative/escaped-worklist.jsonl"
        escaped_candidates = self.temp / "negative/escaped-candidates.jsonl"
        escaped_worklist.write_bytes(factory.jsonl_bytes(work))
        escaped_candidates.write_bytes(factory.jsonl_bytes(rows))
        with self.assertRaises(factory.FactoryV3Error):
            factory.ingest_candidates(
                escaped_worklist,
                escaped_candidates,
                DEPLOY_UNIVERSE,
                TRAINING_UNIVERSE,
                MW_CODEBOOK,
                ROOT,
                self.temp / "negative/ingest",
            )

    def test_scale_campaign_is_zero_sample_and_requires_ab_authorization(self) -> None:
        campaign = self.temp / "scale/campaign.json"
        receipt = factory.freeze_scale_campaign(
            PILOT_RELEASE,
            self.ledger,
            self.baseline,
            QAT_BINDING,
            EVAL_V7_LOCK,
            SFT_RELEASE / "manifest.json",
            [SCRIPT, Path(__file__)],
            campaign,
        )
        self.assertEqual("passed", receipt["status"])
        self.assertEqual(0, factory.load_json(campaign)["generated_semantic_tasks"])
        ab = self.temp / "scale/ab-40.json"
        ab.write_bytes(factory.pretty_bytes({
            "status": "passed", "process_complete": True, "corpus_reuse_eligible": True,
            "model_release_eligible": False, "cell_id": "sft.full_call.boundary",
            "from_total": 40, "semantic_task_total": 40,
        }))
        authorization = self.temp / "scale/full-call-40-160.authorization.json"
        factory.register_scale_gate(campaign, PILOT_RELEASE, ab, "sft.full_call.boundary", 40, 160, authorization)
        out = self.temp / "scale/worklist"
        result = factory.create_scale_worklist(
            self.ledger, campaign, authorization, PILOT_RELEASE, "sft.full_call.boundary", 160, 1, 40,
            DEPLOY_UNIVERSE, TRAINING_UNIVERSE, MW_CODEBOOK, MW_DEFINITIONS, out,
        )
        self.assertEqual("passed", result["status"])
        rows = factory.load_jsonl(out / "worklist.jsonl")
        self.assertEqual(40, len(rows))
        self.assertEqual(factory.SCALE_BATCH_SPLITS, dict(factory.Counter(row["split"] for row in rows)))
        with self.assertRaises(factory.FactoryV3Error):
            factory.create_scale_worklist(
                self.ledger, campaign, authorization, PILOT_RELEASE, "sft.full_call.boundary", 640, 1, 40,
                DEPLOY_UNIVERSE, TRAINING_UNIVERSE, MW_CODEBOOK, MW_DEFINITIONS, self.temp / "scale/blocked",
            )

    def test_scale_schedulers_have_exact_counts_and_mw_degree_contract(self) -> None:
        deploy = factory.tool_index(DEPLOY_UNIVERSE)
        training = factory.tool_index(TRAINING_UNIVERSE)
        for target, expected in ((160, 120), (640, 480)):
            schedules = {
                "full": factory._fullcall_scale_rows(deploy, target),
                "schema": factory._schema_scale_rows(deploy, training, target, TRAINING_UNIVERSE, DEPLOY_UNIVERSE),
                "multi": factory._multistep_scale_rows(deploy, target, DEPLOY_UNIVERSE),
                "mw": factory._mw_scale_rows(deploy, MW_CODEBOOK, MW_DEFINITIONS, target, DEPLOY_UNIVERSE),
            }
            for rows in schedules.values():
                self.assertEqual(expected, len(rows))
                for index in range(0, len(rows), 40):
                    self.assertEqual(factory.SCALE_BATCH_SPLITS, dict(factory.Counter(row["split"] for row in rows[index:index + 40])))
            self.assertEqual(expected // 2, sum(row["trajectory_length"] == 3 for row in schedules["multi"]))
            self.assertEqual(expected // 2, sum(row["trajectory_length"] == 4 for row in schedules["multi"]))
            self.assertEqual(14, len({row["family_id"] for row in schedules["multi"]}))
        self.assertGreaterEqual(len({name for row in schedules["multi"] for name in row["tool_names"]}), 147)
        self.assertEqual(32, factory.Counter(row["reason_code"] for row in factory._fullcall_scale_rows(deploy, 160))["ambiguous_scope"])

    def test_degraded_600m_or_unregistered_source_is_rejected(self) -> None:
        cell_id = "sft.full_call.boundary"
        worklist_dir = self.temp / "negative-600/worklist"
        candidate_path = self.temp / "negative-600/candidates.jsonl"
        factory.create_worklist(self.ledger, cell_id, DEPLOY_UNIVERSE, TRAINING_UNIVERSE, MW_CODEBOOK, MW_DEFINITIONS, worklist_dir)
        factory.draft_candidates(worklist_dir / "worklist.jsonl", DEPLOY_UNIVERSE, TRAINING_UNIVERSE, MW_CODEBOOK, candidate_path)
        forbidden = "artifacts/mei-1.0-51m/legacy/exp-000600m/models/base/mei-1.0-51m-base-cpt600m-clean-source-v3-v1/RELEASE.json"
        work = factory.load_jsonl(worklist_dir / "worklist.jsonl")
        rows = factory.load_jsonl(candidate_path)
        work[0]["source_paths"] = [forbidden]
        rows[0]["source_paths"] = [forbidden]
        escaped_worklist = self.temp / "negative-600/escaped-worklist.jsonl"
        escaped_candidates = self.temp / "negative-600/escaped-candidates.jsonl"
        escaped_worklist.write_bytes(factory.jsonl_bytes(work))
        escaped_candidates.write_bytes(factory.jsonl_bytes(rows))
        with self.assertRaisesRegex(factory.FactoryV3Error, "unregistered or ineligible source"):
            factory.ingest_candidates(
                escaped_worklist,
                escaped_candidates,
                DEPLOY_UNIVERSE,
                TRAINING_UNIVERSE,
                MW_CODEBOOK,
                ROOT,
                self.temp / "negative-600/ingest",
            )

    def test_non_generative_demand_cell_cannot_create_worklist(self) -> None:
        with self.assertRaises(factory.FactoryV3Error):
            factory.create_worklist(
                self.ledger,
                "cpt.gap.structure",
                DEPLOY_UNIVERSE,
                TRAINING_UNIVERSE,
                MW_CODEBOOK,
                MW_DEFINITIONS,
                self.temp / "negative-cpt-worklist",
            )

    def test_cpt_mix_is_natural_only_without_trigger(self) -> None:
        out = self.temp / "cpt-natural.json"
        result = factory.compose_cpt_mixes(
            "frozen-300m-base",
            "a" * 64,
            300_000_485,
            600_000_000,
            "b" * 64,
            out,
        )
        self.assertEqual("passed", result["status"])
        manifest = factory.load_json(out)
        self.assertEqual(1, len(manifest["candidates"]))
        self.assertEqual(0.0, manifest["candidates"][0]["synthetic_fraction"])
        self.assertEqual(299_999_515, manifest["candidates"][0]["total_increment_tokens"])
        with self.assertRaises(factory.FactoryV3Error):
            factory.compose_cpt_mixes(
                "frozen-300m-base",
                "a" * 64,
                300_000_485,
                600_000_000,
                "b" * 64,
                self.temp / "forbidden.json",
                synthetic_release_sha256="c" * 64,
            )

    def test_cpt_trigger_enables_only_preregistered_grid(self) -> None:
        trigger_path = self.temp / "trigger.json"
        trigger = {
            "schema": "mei-51m-cpt-synthetic-trigger-v1",
            "status": "passed",
            "requirements": {
                "natural_control_misses_preregistered_holdout": True,
                "audited_natural_sources_cannot_fill_gap": True,
                "tokenizer_recipe_runtime_and_eval_causes_excluded": True,
            },
        }
        trigger_path.write_bytes(factory.pretty_bytes(trigger))
        out = self.temp / "cpt-triggered.json"
        factory.compose_cpt_mixes(
            "frozen-300m-base",
            "a" * 64,
            300_000_000,
            600_000_000,
            "b" * 64,
            out,
            trigger_path,
            "c" * 64,
        )
        fractions = [row["synthetic_fraction"] for row in factory.load_json(out)["candidates"]]
        self.assertEqual([0.0, 0.001, 0.0025, 0.005, 0.01], fractions)

    def test_ever_seen_ledger_is_balanced_and_requires_real_documents(self) -> None:
        index_path = self.temp / "documents.jsonl"
        rows = [
            {"document_id": "hq-a", "source_role": "fineweb2_hq", "sha256": "a" * 64, "tokens": 60, "eligible": True, "license_reviewed": True},
            {"document_id": "hq-b", "source_role": "fineweb2_hq", "sha256": "b" * 64, "tokens": 60, "eligible": True, "license_reviewed": True},
            {"document_id": "wiki-a", "source_role": "wiki", "sha256": "c" * 64, "tokens": 60, "eligible": True, "license_reviewed": True},
            {"document_id": "wiki-b", "source_role": "wiki", "sha256": "d" * 64, "tokens": 60, "eligible": True, "license_reviewed": True},
        ]
        index_path.write_bytes(factory.jsonl_bytes(rows))
        result = factory.build_ever_seen_ledger(index_path, self.temp / "ever-seen", 300)
        self.assertEqual("passed", result["status"])
        manifest = factory.load_json(self.temp / "ever-seen/manifest.json")
        self.assertEqual(300, manifest["actual_tokens"])
        self.assertTrue(all(value <= 1 for value in manifest["exposure_count_imbalance_by_source"].values()))

    def test_review_receipt_requires_all_160_decisions(self) -> None:
        candidate_paths, build_dirs, audit_paths = self._build_pilot()
        merged_path = self.temp / "review/merged.json"
        factory.merge_audits(audit_paths, merged_path)
        plan_path = self.temp / "review/plan.json"
        plan = factory.prepare_review(candidate_paths, merged_path, plan_path)
        attestation_path = self.temp / "review/attestation.json"
        attestation = {
            "schema": "mei-51m-human-review-attestation-v1",
            "product": factory.PRODUCT,
            "review_plan_sha256": factory.sha256_file(plan_path),
            "reviewer": "fixture-reviewer",
            "attested_on": "2026-09-02",
            "attested_semantic_tasks": 160,
            "verdict_scope": "all_160_pass",
            "failed_semantic_tasks": 0,
            "recording_mode": "unit_test_fixture",
            "codex_is_reviewer": False,
            "provider_calls": 0,
            "current_mutated": False,
        }
        attestation_path.write_bytes(factory.pretty_bytes(attestation))
        attestation_sha256 = factory.sha256_file(attestation_path)
        decisions = [
            {
                "candidate_id": sample_id,
                "reviewer": "fixture-reviewer",
                "verdict": "pass",
                "defect_class": "none",
                "note": "unit-test fixture",
                "provenance": "unit_test_human_review_attestation",
                "attestation_sha256": attestation_sha256,
            }
            for sample_id in plan["sample_ids"]
        ]
        decisions_path = self.temp / "review/decisions.jsonl"
        decisions_path.write_bytes(factory.jsonl_bytes(decisions))
        incomplete_path = self.temp / "review/incomplete-decisions.jsonl"
        incomplete_path.write_bytes(factory.jsonl_bytes(decisions[:-1]))
        with self.assertRaises(factory.FactoryV3Error):
            factory.record_review(plan_path, incomplete_path, self.temp / "review/incomplete-receipt.json")
        receipt_path = self.temp / "review/receipt.json"
        receipt = factory.record_review(plan_path, decisions_path, receipt_path)
        self.assertEqual("passed", receipt["status"])
        self.assertEqual(160, receipt["semantic_tasks_reviewed"])

        validation_path = self.temp / "review/sft-validation.json"
        factory.freeze_sft_validation_contract(self.ledger, self.baseline, validation_path)
        release_dir = self.temp / "review/release"
        result = factory.freeze_release(
            candidate_paths,
            build_dirs,
            merged_path,
            receipt_path,
            plan_path,
            decisions_path,
            attestation_path,
            self.baseline,
            self.ledger,
            validation_path,
            release_dir,
            "mei-1.0-51m-sft-gap-pilot-test-v1",
        )
        self.assertEqual("passed", result["status"])
        release_manifest_path = release_dir / "release-manifest.json"
        release_manifest = factory.load_json(release_manifest_path)
        self.assertTrue(release_manifest["training_eligible"])
        self.assertFalse(release_manifest["release_eligible"])
        self.assertTrue(release_manifest["corpus_reuse_eligible"])
        self.assertEqual(160, release_manifest["semantic_task_count"])
        self.assertEqual(460, release_manifest["compiled_row_count"])
        release_lineage = factory.verify_lineage(release_manifest_path, release_dir)
        self.assertEqual("passed", release_lineage["status"])

        verification_path = self.temp / "review/demand-verification.json"
        verification_path.write_bytes(factory.pretty_bytes(factory.verify_demand_ledger(self.ledger, self.baseline)))
        cpt_policy_path = self.temp / "review/cpt-policy.json"
        factory.freeze_cpt_policy(self.ledger, cpt_policy_path)
        release_lineage_path = self.temp / "review/release-lineage.json"
        release_lineage_path.write_bytes(factory.pretty_bytes(release_lineage))
        named_paths = {
            "baseline_inventory": self.baseline,
            "demand_ledger": self.ledger,
            "demand_verification": verification_path,
            "cpt_policy": cpt_policy_path,
            "sft_validation_contract": validation_path,
            "merged_audit": merged_path,
            "review_plan": plan_path,
            "review_receipt": receipt_path,
            "review_decisions": decisions_path,
            "review_attestation": attestation_path,
            "release_manifest": release_manifest_path,
            "release_lineage": release_lineage_path,
            "factory_script": SCRIPT,
            "factory_tests": Path(__file__),
        }
        for index, path in enumerate(audit_paths):
            named_paths[f"audit_{index}"] = path
        frozen_index_path = self.temp / "review/frozen-index.json"
        frozen = factory.freeze_pilot_index(named_paths, ROOT, frozen_index_path)
        self.assertEqual("frozen_training_delta", frozen["status"])
        frozen_index = factory.load_json(frozen_index_path)
        self.assertEqual(160, frozen_index["counts"]["human_reviewed_semantic_tasks"])
        self.assertTrue(frozen_index["training_eligible"])
        self.assertFalse(frozen_index["release_eligible"])
        self.assertEqual("passed", factory.verify_lineage(frozen_index_path, ROOT)["status"])

    def test_write_once_rejects_changed_content(self) -> None:
        path = self.temp / "immutable.json"
        factory.write_once(path, b"one\n")
        factory.write_once(path, b"one\n")
        with self.assertRaises(factory.FactoryV3Error):
            factory.write_once(path, b"two\n")

    def test_pending_human_review_is_a_successful_cli_transition(self) -> None:
        cell_id = "sft.full_call.boundary"
        worklist_dir = self.temp / "cli/worklist"
        candidate_path = self.temp / "cli/candidates.jsonl"
        factory.create_worklist(
            self.ledger,
            cell_id,
            DEPLOY_UNIVERSE,
            TRAINING_UNIVERSE,
            MW_CODEBOOK,
            MW_DEFINITIONS,
            worklist_dir,
        )
        with redirect_stdout(io.StringIO()):
            exit_code = factory.main(
                [
                    "draft-candidates",
                    "--worklist",
                    str(worklist_dir / "worklist.jsonl"),
                    "--tool-universe",
                    str(DEPLOY_UNIVERSE),
                    "--training-tool-universe",
                    str(TRAINING_UNIVERSE),
                    "--mw-codebook",
                    str(MW_CODEBOOK),
                    "--out",
                    str(candidate_path),
                ]
            )
        self.assertEqual(0, exit_code)
        self.assertTrue(candidate_path.is_file())


if __name__ == "__main__":
    unittest.main()
