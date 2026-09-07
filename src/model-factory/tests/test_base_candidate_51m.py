from __future__ import annotations

import unittest
import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import release.base_candidate_51m as candidate
import orchestration.lifecycle_51m as lifecycle


class BaseCandidateBoundaryTest(unittest.TestCase):
    def test_register_and_propose_never_touch_current(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runs = root / "runs"
            directory = runs / "run-600m"
            cpt = directory / "checkpoints/cpt"
            (directory / "stages/cpt_gate").mkdir(parents=True)
            cpt.mkdir(parents=True)
            schedule = root / "corpus/schedule-cpt-600m.json"
            schedule.parent.mkdir(parents=True)
            parent = root / "base/parent"
            parent.mkdir(parents=True)
            parent_state = parent / "parent-state.npz"
            parent_weights = parent / "parent.npz"
            parent_state.write_bytes(b"parent-state")
            parent_weights.write_bytes(b"parent-weights")
            tokenizer = root / "tokenizer.model"
            tokenizer.write_bytes(b"tokenizer")
            tokenizer_sha = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
            parent_release = {
                "model_id": "parent-300m",
                "tokens_seen_exposure": 300_000_000,
                "state_sha256": lifecycle.sha256_file(parent_state),
                "weights": parent_weights.name,
                "weights_sha256": lifecycle.sha256_file(parent_weights),
                "tokenizer_sha256": tokenizer_sha,
            }
            (parent / "RELEASE.json").write_text(
                json.dumps(parent_release), encoding="utf-8"
            )
            schedule.write_text(
                json.dumps({"parent_checkpoint": "base/parent/parent-state.npz"}),
                encoding="utf-8",
            )
            config = {
                "run_id": "run-600m",
                "rung": "600m",
                "architecture_id": "mei-1.0-51m-arch-v1",
                "weight_contract_sha256": lifecycle.architecture_contracts()[
                    "weight_contract_sha256"
                ],
                "target_exposure": 600_000_000,
                "target_exposure_tokens": 600_000_000,
                "source_capture_mode": "launch",
                "source_manifest_sha256": "source-manifest",
                "corpus": {
                    "schedule": "corpus/schedule-cpt-600m.json",
                    "parent_exposure": 300_000_000,
                    "incremental_exposure": 300_000_000,
                    "snapshot_sha256": "snapshot",
                },
            }
            (directory / "run.json").write_text(json.dumps(config), encoding="utf-8")
            (directory / "source-manifest.json").write_text(
                json.dumps({"manifest_sha256": "source-manifest"}), encoding="utf-8"
            )
            (directory / "jobs").mkdir()
            audited_corpus = directory / "jobs/audited-corpus.jsonl"
            audited_corpus.write_text('{"text":"可信语料"}\n', encoding="utf-8")
            (directory / "jobs/synthetic-diversity-receipt.json").write_text(
                json.dumps(
                    {
                        "schema": "mei-synthetic-corpus-diversity-receipt-v1",
                        "terminal_status": "passed",
                        "corpus_diversity_degraded": False,
                        "roles": {
                            "colloquial": {
                                "status": "passed",
                                "input": {
                                    "path": str(audited_corpus),
                                    "sha256": lifecycle.sha256_file(audited_corpus),
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = {
                "tokens_seen": 600_000_000,
                "params": 51_463_797,
                "architecture_id": "mei-1.0-51m-arch-v1",
                "architecture_sha256": "f02aebda393c7fca181af7b7225581577f689531245dec00ac549de1482c5ea7",
                "tokenizer_sha256": tokenizer_sha,
                "valid_loss": 3.1,
                "valid_loss_hq": 3.0,
                "valid_loss_structure": 0.1,
                "valid_loss_colloquial": 0.2,
                "probe_mean_nll": 4.4,
                "final_probe_mean_nll": 4.2,
            }
            (cpt / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            for name in (
                "pretrain-cpt.npz",
                "pretrain-cpt-state.npz",
                "pretrain-cpt-state.meta.json",
            ):
                (cpt / name).write_bytes(name.encode())
            gate_manifest = {
                str((cpt / "summary.json").resolve()): {
                    "sha256": lifecycle.sha256_file(cpt / "summary.json"),
                    "bytes": (cpt / "summary.json").stat().st_size,
                }
            }
            (directory / "stages/cpt_gate/receipt.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "fingerprint": "gate-fingerprint",
                        "output_manifest": gate_manifest,
                        "output_manifest_sha256": lifecycle.sha256_json(gate_manifest),
                        "current_sha256": "unchanged",
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(lifecycle, "RUN_ROOT", runs),
                patch.object(candidate, "ROOT", root),
                patch.object(
                    candidate,
                    "BASE_ROOT",
                    root / "cycles/mei-1.1-51m/exp-00600m/models/base",
                ),
                patch.object(candidate, "frozen_tokenizer_path", return_value=tokenizer),
                patch.object(candidate, "current_hash", return_value="unchanged"),
                patch.object(
                    candidate,
                    "recipe",
                    return_value={"stages": {"cpt_gate": {}}},
                ),
                patch.object(
                    candidate,
                    "stage_fingerprint",
                    return_value="gate-fingerprint",
                ),
                patch.object(
                    candidate,
                    "live_stage_state",
                    side_effect=(
                        {"lock_held": True, "pid_alive": True},
                        {"lock_held": False, "pid_alive": False},
                    ),
                ),
                patch.object(
                    candidate,
                    "_numeric_integrity",
                    return_value={
                        "schema": "mei-cpt-numeric-integrity-v1",
                        "status": "passed",
                    },
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "terminal cpt_gate"):
                    candidate._verified_inputs("run-600m")
                registered = candidate.register_base_candidate("run-600m")
                dest = root / "cycles/mei-1.1-51m/exp-00600m/models/base/mei-1.2-51m-base-cpt600m-v1"
                self.assertTrue((dest / "RELEASE.json").is_file())
                self.assertTrue(registered["current_unchanged"])
                release = json.loads((dest / "RELEASE.json").read_text())
                self.assertEqual(release["tokenizer_sha256"], tokenizer_sha)
                self.assertEqual(release["formal_parent"]["model_id"], "parent-300m")
                self.assertEqual(release["lineage_assurance"], "direct_parent")
                self.assertTrue(release["continuation_checkpoint_eligible"])
                self.assertTrue(release["automatic_parent_promotion_eligible"])
                self.assertTrue(release["corpus_reuse_eligible"])
                self.assertTrue(release["future_parent_eligible"])
                self.assertEqual(release["valid_loss"], 3.1)
                self.assertEqual(release["valid_loss_hq"], 3.0)
                self.assertEqual(release["valid_loss_structure"], 0.1)
                self.assertEqual(release["valid_loss_colloquial"], 0.2)
                self.assertEqual(release["final_probe_mean_nll"], 4.2)
                proposal = candidate.propose_freeze(dest)
                self.assertTrue(proposal["eligible"])
                self.assertTrue(proposal["continuation_checkpoint_eligible"])
                self.assertTrue(proposal["automatic_parent_promotion_eligible"])
                self.assertTrue(proposal["corpus_reuse_eligible"])
                self.assertEqual(proposal["current_sha256_observed"], "unchanged")
                self.assertFalse(proposal["current_mutated"])

    def test_numeric_integrity_checks_tensors_metrics_and_state(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            weights = directory / "weights.npz"
            state = directory / "state.npz"
            meta = directory / "state.meta.json"
            np.savez(
                weights,
                a=np.asarray([1.0, 2.0], dtype=np.float32),
                b=np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
            )
            np.savez(
                state,
                **{
                    "p.a": np.asarray([1.0, 2.0], dtype=np.float32),
                    "p.b": np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
                    "o.step": np.asarray(1, dtype=np.int64),
                },
            )
            summary = {
                "tokens_seen": 600,
                "tokenizer_sha256": "tokenizer",
                "valid_loss": 3.0,
                "valid_loss_hq": 2.9,
                "valid_loss_structure": 0.2,
                "valid_loss_colloquial": 2.1,
                "final_probe_mean_nll": 4.0,
            }
            config = {
                "architecture_id": "mei-1.0-51m-arch-v1",
                "weight_contract_sha256": "weight-contract",
            }
            meta.write_text(
                json.dumps(
                    {
                        "tokens_seen": 600,
                        "params": 5,
                        "architecture_id": "mei-1.0-51m-arch-v1",
                        "weight_contract_sha256": "weight-contract",
                        "tokenizer_sha256": "tokenizer",
                        "sampler_state": {"cursor": 1},
                    }
                ),
                encoding="utf-8",
            )
            report = candidate._numeric_integrity(
                weights,
                state,
                meta,
                summary,
                config,
                expected_parameter_tensors=2,
                expected_params=5,
            )
            self.assertEqual(report["parameter_tensor_count"], 2)
            self.assertTrue(report["all_numeric_tensors_finite"])

            np.savez(
                weights,
                a=np.asarray([1.0, np.nan], dtype=np.float32),
                b=np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
            )
            with self.assertRaisesRegex(RuntimeError, "non-finite tensor"):
                candidate._numeric_integrity(
                    weights,
                    state,
                    meta,
                    summary,
                    config,
                    expected_parameter_tensors=2,
                    expected_params=5,
                )

            np.savez(
                weights,
                a=np.asarray([1.0, 2.0], dtype=np.float32),
                b=np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
            )
            # hq 角色损失在新链为可选项；必选指标缺失才失败关闭
            summary["valid_loss"] = None
            with self.assertRaisesRegex(RuntimeError, "metrics are missing"):
                candidate._numeric_integrity(
                    weights,
                    state,
                    meta,
                    summary,
                    config,
                    expected_parameter_tensors=2,
                    expected_params=5,
                )

    def test_missing_diversity_audit_fails_closed_for_future_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            quality = candidate._corpus_quality(Path(raw))
        self.assertEqual(quality["status"], "not_audited")
        self.assertTrue(quality["corpus_diversity_degraded"])
        self.assertFalse(quality["corpus_reuse_eligible"])
        self.assertFalse(quality["automatic_parent_promotion_eligible"])
        self.assertFalse(quality["future_parent_eligible"])

    def test_hybrid_recovery_and_degraded_corpus_remain_productizable_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=candidate.ROOT) as raw:
            directory = Path(raw)
            jobs = directory / "jobs"
            jobs.mkdir()
            recovery_checkpoint = directory / "recovery-state.npz"
            recovery_checkpoint.write_bytes(b"recovery")
            source_receipt = directory / "historical-receipt.json"
            source_receipt.write_text(
                json.dumps({"status": "blocked"}), encoding="utf-8"
            )
            recovery = {
                "source_run_id": "historical-run",
                "checkpoint": str(recovery_checkpoint.relative_to(candidate.ROOT)),
                "checkpoint_sha256": lifecycle.sha256_file(recovery_checkpoint),
                "tokens_seen": 506_914_021,
                "source_receipt": str(source_receipt.relative_to(candidate.ROOT)),
                "source_receipt_sha256": lifecycle.sha256_file(source_receipt),
                "source_receipt_status": "blocked",
                "source_receipt_artifacts_match": True,
            }
            (jobs / "recovery-source.json").write_text(
                json.dumps(recovery), encoding="utf-8"
            )
            assurance, evidence = candidate._recovery_provenance(directory, {})
            self.assertEqual(assurance, "hybrid_recovery")
            self.assertEqual(evidence["tokens_seen"], 506_914_021)
            self.assertTrue(evidence["source_receipt_artifacts_match"])

            source_receipt.write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "missing or drifted"):
                candidate._recovery_provenance(directory, {})
            source_receipt.write_text(
                json.dumps({"status": "blocked"}), encoding="utf-8"
            )

            (jobs / "synthetic-diversity-receipt.json").write_text(
                json.dumps(
                    {
                        "schema": "mei-synthetic-corpus-diversity-receipt-v1",
                        "terminal_status": "degraded",
                        "corpus_diversity_degraded": True,
                        "roles": {
                            "colloquial": {
                                "status": "degraded",
                                "input": {
                                    "path": str(recovery_checkpoint),
                                    "sha256": lifecycle.sha256_file(
                                        recovery_checkpoint
                                    ),
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            quality = candidate._corpus_quality(directory)
            self.assertTrue(quality["corpus_diversity_degraded"])
            self.assertFalse(quality["corpus_reuse_eligible"])
            self.assertFalse(quality["automatic_parent_promotion_eligible"])
            self.assertFalse(quality["future_parent_eligible"])

    def test_diversity_audit_must_be_terminal_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            jobs = directory / "jobs"
            jobs.mkdir()
            source = directory / "source.jsonl"
            source.write_text('{"text":"样本"}\n', encoding="utf-8")
            receipt_path = jobs / "synthetic-diversity-receipt.json"
            receipt = {
                "schema": "mei-synthetic-corpus-diversity-receipt-v1",
                "terminal_status": "running",
                "corpus_diversity_degraded": False,
                "roles": {
                    "colloquial": {
                        "status": "passed",
                        "input": {
                            "path": str(source),
                            "sha256": lifecycle.sha256_file(source),
                        },
                    }
                },
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not terminal"):
                candidate._corpus_quality(directory)

            receipt["terminal_status"] = "passed"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            source.write_text('{"text":"已漂移"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing or drifted"):
                candidate._corpus_quality(directory)

            base_root = directory / "base"
            destination = base_root / "mei-1.0-51m-base-cpt600m-v1"
            destination.mkdir(parents=True)
            files = {}
            hashes = {}
            for role, hash_key in {
                "weights": "weights_sha256",
                "train_state": "state_sha256",
                "train_state_meta": "state_meta_sha256",
                "summary": "summary_sha256",
            }.items():
                path = destination / f"{role}.bin"
                path.write_bytes(role.encode())
                files[role] = path.name
                hashes[hash_key] = lifecycle.sha256_file(path)
            release = {
                "status": "registered_candidate",
                "process_complete": True,
                "release_eligible": False,
                "productization_experiment_eligible": True,
                "continuation_checkpoint_eligible": True,
                "automatic_parent_promotion_eligible": False,
                "corpus_reuse_eligible": False,
                "future_parent_eligible": False,
                "release_blockers": ["corpus_diversity_degraded"],
                "files": files,
                **hashes,
            }
            (destination / "RELEASE.json").write_text(
                json.dumps(release), encoding="utf-8"
            )
            with (
                patch.object(candidate, "BASE_ROOT", base_root),
                patch.object(candidate, "current_hash", return_value="current"),
            ):
                proposal = candidate.propose_freeze(destination)
            self.assertTrue(proposal["process_complete"])
            self.assertFalse(proposal["release_eligible"])
            self.assertFalse(proposal["eligible"])
            self.assertTrue(proposal["productization_experiment_eligible"])
            self.assertTrue(proposal["continuation_checkpoint_eligible"])
            self.assertFalse(proposal["automatic_parent_promotion_eligible"])
            self.assertFalse(proposal["corpus_reuse_eligible"])
            self.assertEqual(proposal["unmet_gates"], ["corpus_diversity_degraded"])

    def test_finalize_refuses_without_explicit_confirmation_before_any_write(
        self,
    ) -> None:
        with patch.object(
            candidate,
            "current_hash",
            side_effect=AssertionError("must not read CURRENT"),
        ):
            with self.assertRaises(PermissionError):
                candidate.finalize_current(
                    Path("cycles/mei-1.1-51m/exp-00600m/models/base/mei-1.0-51m-base-cpt600m-v1"),
                    expected_current_sha256="unused",
                    confirmation="not authorized",
                )

    def test_legacy_promoter_contains_no_current_write(self) -> None:
        here = Path(__file__).resolve().parent.parent / "compatibility"
        for name in (
            "promote_cpt1b_base.py",
            "promote_scratch_base.py",
            "promote_scratch_base_51m.py",
        ):
            text = (here / name).read_text(encoding="utf-8")
            self.assertIn("retired", text)
            self.assertNotIn("activate_current", text)
            self.assertNotIn(".replace(current_path)", text)
            self.assertNotIn("atomic_json(CURRENT_PATH", text)


if __name__ == "__main__":
    unittest.main()
