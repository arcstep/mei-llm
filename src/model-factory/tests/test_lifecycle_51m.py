from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import orchestration.lifecycle_51m as lc
from common.identity_51m import ARCHITECTURE_ID, assert_51m_architecture_id


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class Lifecycle51MTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus"
        self.runs = self.root / "runs"
        self.corpus.mkdir()
        dump(
            self.corpus / "RELEASE.json",
            {
                "training_mode": "cpt",
                "cumulative_exposure_tokens": 600_000_000,
                "license": "test-local-cleared",
                "contamination": {"checked": True, "hits": 0},
            },
        )
        dump(
            self.corpus / "manifest.json",
            {"immutable": True, "n_train_tokens": 300_000_000, "n_unique_train_tokens": 300_000_000},
        )
        dump(self.corpus / "hashes.json", {})
        dump(self.corpus / "mix.json", {"training_mode": "cpt"})
        sources = {
            name: {"token_quota": 75_000_000, "skip_tokens": 0, "max_epochs": 1.0}
            for name in ("wiki", "hq", "structure", "colloquial")
        }
        dump(
            self.corpus / "schedule-cpt-600m.json",
            {
                "kind": "cpt",
                "parent_tokens_seen": 300_000_000,
                "exposure_tokens": 300_000_000,
                "cumulative_exposure_tokens": 600_000_000,
                "parent_checkpoint": "base/parent-state.npz",
                "sampler": "quota_plan",
                "allow_repeat": False,
                "sources": sources,
            },
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_init_is_idempotent_and_isolated(self) -> None:
        with patch.object(lc, "RUN_ROOT", self.runs):
            first = lc.init_run("test-600m", self.corpus, 600_000_000)
            second = lc.init_run("test-600m", self.corpus, 600_000_000)
        self.assertEqual(first["corpus"], second["corpus"])
        self.assertEqual(first["params"], 51_463_797)
        self.assertTrue((self.runs / "test-600m/jobs").is_dir())
        self.assertTrue((self.runs / "test-600m/checkpoints").is_dir())
        self.assertEqual(first["source_manifest_schema_version"], 2)
        self.assertTrue(lc.source_capture_status(first)["unchanged"])
        Path(first["source_capture"]["source_archive"]).write_bytes(b"corrupted")
        self.assertFalse(lc.source_capture_status(first)["unchanged"])

    def test_changed_input_invalidates_existing_run(self) -> None:
        with patch.object(lc, "RUN_ROOT", self.runs):
            lc.init_run("test-drift", self.corpus, 600_000_000)
            dump(
                self.corpus / "manifest.json",
                {
                    "immutable": False,
                    "n_train_tokens": 300_000_000,
                    "n_unique_train_tokens": 300_000_000,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "different immutable inputs"):
                lc.init_run("test-drift", self.corpus, 600_000_000)

    def test_missing_local_corpus_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing local corpus directory"):
            lc.corpus_snapshot(self.root / "missing", 600_000_000)

    def test_legacy_corpus_snapshot_remains_adoptable(self) -> None:
        snapshot = lc.corpus_snapshot(self.corpus, 600_000_000)
        config = {
            "target_exposure": 600_000_000,
            "corpus": {
                "corpus_dir": str(self.corpus),
                "snapshot_sha256": snapshot["legacy_evidence_snapshot_sha256"],
            },
        }
        ok, detail = lc.internal_gate("corpus_freeze", self.runs / "legacy", config)
        self.assertTrue(ok, detail)

    def test_corpus_merkle_covers_resolved_shard_bytes(self) -> None:
        shard = self.corpus / "tokens/wiki-train-0000.bin"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"first")
        dump(
            self.corpus / "mix.json",
            {
                "training_mode": "cpt",
                "sources": {
                    name: (
                        {"train_shards": [str(shard)], "valid_shards": []}
                        if name == "wiki"
                        else {"train_shards": [], "valid_shards": []}
                    )
                    for name in ("wiki", "hq", "structure", "colloquial")
                },
            },
        )
        before = lc.corpus_snapshot(self.corpus, 600_000_000)
        shard.write_bytes(b"second")
        after = lc.corpus_snapshot(self.corpus, 600_000_000)
        self.assertNotEqual(before["artifact_merkle_sha256"], after["artifact_merkle_sha256"])
        self.assertNotEqual(before["snapshot_sha256"], after["snapshot_sha256"])

    def test_dependency_receipt_changes_fingerprint(self) -> None:
        directory = self.runs / "fp"
        dep = lc.stage_receipt(directory, "corpus_freeze")
        lc.atomic_json(dep, {"status": "passed", "value": 1})
        config = {
            "recipe_sha256": "recipe",
            "corpus": {"snapshot_sha256": "corpus"},
            "target_exposure": 600_000_000,
        }
        row = {"requires": ["corpus_freeze"], "command": ["x"]}
        before = lc.stage_fingerprint(directory, config, "cpt_readiness", row)
        lc.atomic_json(dep, {"status": "passed", "value": 2})
        after = lc.stage_fingerprint(directory, config, "cpt_readiness", row)
        self.assertNotEqual(before, after)

    def test_recipe_locks_identity_and_current_policy(self) -> None:
        data = lc.recipe()
        self.assertEqual(data["architecture_id"], "mei-1.0-51m-arch-v1")
        self.assertEqual(data["params"], 51_463_797)
        self.assertEqual(data["current_policy"], "read_only_until_explicit_user_freeze")
        self.assertEqual(
            ["corpus_freeze", "cpt_readiness", "cpt", "cpt_gate"],
            data["tracks"]["cpt"],
        )
        self.assertIn("source quality receipts", data["stages"]["corpus_freeze"]["gate"])
        self.assertIsNone(assert_51m_architecture_id(ARCHITECTURE_ID))
        self.assertIsNotNone(assert_51m_architecture_id("other-architecture"))

    def test_render_context_uses_actual_cpt_checkpoint_names(self) -> None:
        directory = self.runs / "render"
        context = lc.render_context(
            directory,
            {
                "corpus": {
                    "corpus_dir": str(self.corpus),
                    "schedule": str(self.corpus / "schedule-cpt-600m.json"),
                },
                "target_exposure": 600_000_000,
                "rung": "600m",
            },
        )
        self.assertTrue(context["cpt_weights"].endswith("/checkpoints/cpt/pretrain-cpt.npz"))
        self.assertTrue(context["parent_state"].endswith("/base/parent-state.npz"))

    def test_render_context_prefers_pinned_recovery_checkpoint(self) -> None:
        directory = self.runs / "render-recovery"
        recovery = self.runs / "source/checkpoints/cpt/pretrain-cpt-state.npz"
        context = lc.render_context(
            directory,
            {
                "corpus": {
                    "corpus_dir": str(self.corpus),
                    "schedule": str(self.corpus / "schedule-cpt-600m.json"),
                },
                "target_exposure": 600_000_000,
                "rung": "600m",
                "resume_checkpoint": {"checkpoint": str(recovery)},
            },
        )
        self.assertEqual(context["parent_state"], str(recovery))

    def test_offline_env_fail_closes_external_http(self) -> None:
        env = lc.offline_env(self.runs / "offline")
        self.assertEqual(env["MEI_OFFLINE"], "1")
        self.assertEqual(env["CARGO_NET_OFFLINE"], "true")
        self.assertEqual(env["HTTPS_PROXY"], "http://127.0.0.1:9")
        self.assertEqual(env["NO_PROXY"], "127.0.0.1,localhost")

    def test_stage_reuse_requires_unchanged_output_hash(self) -> None:
        directory = self.runs / "reuse"
        (directory / "jobs").mkdir(parents=True)
        config = {
            "recipe_sha256": "recipe",
            "corpus": {
                "snapshot_sha256": "corpus",
                "corpus_dir": str(self.corpus),
                "schedule": str(self.corpus / "schedule-cpt-600m.json"),
            },
            "target_exposure": 600_000_000,
            "rung": "600m",
        }
        row = {"kind": "command", "requires": [], "command": ["local-command"]}
        output = directory / "jobs/result.json"

        def execute(*_args, **_kwargs):
            dump(output, {"version": 1})
            return 0

        with patch.object(lc, "run_logged_process", side_effect=execute) as run, patch.object(
            lc, "current_hash", return_value="current"
        ):
            ok, first = lc.run_stage(directory, config, "runtime_parity", row, dry_run=False)
            self.assertTrue(ok)
            self.assertEqual(first["status"], "passed")
            ok, reused = lc.run_stage(directory, config, "runtime_parity", row, dry_run=False)
            self.assertTrue(ok)
            self.assertEqual(reused["status"], "reused")
            self.assertEqual(run.call_count, 1)
            dump(output, {"version": 2})
            ok, rerun = lc.run_stage(directory, config, "runtime_parity", row, dry_run=False)
            self.assertTrue(ok)
            self.assertEqual(rerun["status"], "passed")
            self.assertEqual(run.call_count, 2)

    def test_current_change_writes_blocker_receipt(self) -> None:
        directory = self.runs / "current"
        (directory / "jobs").mkdir(parents=True)
        config = {
            "recipe_sha256": "recipe",
            "corpus": {
                "snapshot_sha256": "corpus",
                "corpus_dir": str(self.corpus),
                "schedule": str(self.corpus / "schedule-cpt-600m.json"),
            },
            "target_exposure": 600_000_000,
            "rung": "600m",
        }
        row = {"kind": "command", "requires": [], "command": ["local-command"]}
        with patch.object(lc, "run_logged_process", return_value=0), patch.object(
            lc, "current_hash", side_effect=["before", "after"]
        ):
            ok, receipt = lc.run_stage(directory, config, "pack_q4", row, dry_run=False)
        self.assertFalse(ok)
        self.assertEqual(receipt["status"], "blocked")
        self.assertIn("CURRENT.json changed", receipt["failure_reason"])

    def test_cq2_degrades_only_with_quality_block_receipt(self) -> None:
        config = {
            "recipe_sha256": "recipe",
            "corpus": {
                "snapshot_sha256": "corpus",
                "corpus_dir": str(self.corpus),
                "schedule": str(self.corpus / "schedule-cpt-600m.json"),
            },
            "target_exposure": 600_000_000,
            "rung": "600m",
        }
        row = {
            "kind": "command_or_degrade",
            "requires": [],
            "command": ["cq2-command"],
            "degrade_receipt": "qat-cq2-blocked.json",
            "degrade_key": "quality_blocked",
            "failure_branch": "q4_only",
        }
        verified = self.runs / "cq2-verified"
        (verified / "jobs").mkdir(parents=True)

        def quality_block(*_args, **_kwargs):
            dump(verified / "jobs/qat-cq2-blocked.json", {"quality_blocked": True})
            return 2

        with patch.object(lc, "run_logged_process", side_effect=quality_block), patch.object(
            lc, "current_hash", return_value="current"
        ):
            ok, receipt = lc.run_stage(verified, config, "cq2", row, dry_run=False)
        self.assertTrue(ok)
        self.assertEqual(receipt["status"], "degraded")
        self.assertEqual(receipt["failure_branch"], "q4_only")

        crashed = self.runs / "cq2-crashed"
        (crashed / "jobs").mkdir(parents=True)
        with patch.object(lc, "run_logged_process", return_value=2), patch.object(
            lc, "current_hash", return_value="current"
        ):
            ok, receipt = lc.run_stage(crashed, config, "cq2", row, dry_run=False)
        self.assertFalse(ok)
        self.assertEqual(receipt["status"], "blocked")

    def test_identity_errors_reject_param_and_hash_drift(self) -> None:
        weight = lc.architecture_contracts()["weight_contract_sha256"]
        config = {"weight_contract_sha256": weight}
        ok = {
            "params": 51_463_797,
            "architecture_id": "mei-1.0-51m-arch-v1",
            "weight_contract_sha256": weight,
        }
        self.assertEqual(lc.identity_errors(ok, config), [])
        self.assertTrue(lc.identity_errors({**ok, "params": 58_000_000}, config))
        self.assertTrue(lc.identity_errors({**ok, "weight_contract_sha256": "other"}, config))
        legacy = {**ok, "weight_contract_sha256": None, "architecture_sha256": "f02aebda393c7fca181af7b7225581577f689531245dec00ac549de1482c5ea7"}
        self.assertEqual(lc.identity_errors(legacy, config), [])

    def test_recipe_closed_loop_keeps_continuation_and_serial_lock(self) -> None:
        data = lc.recipe()
        self.assertEqual(
            data["stage_order"],
            [
                "corpus_freeze",
                "cpt_readiness",
                "cpt",
                "cpt_gate",
            ],
        )
        self.assertIn("--resume-mode", data["stages"]["cpt"]["command"])
        self.assertEqual(
            data["stages"]["cpt"]["command"][data["stages"]["cpt"]["command"].index("--resume-mode") + 1],
            "continuation",
        )
        self.assertIn("sampler/cursor", data["hard_rules"]["resume"])
        self.assertIn(
            "separate booleans",
            data["hard_rules"]["promotion"],
        )

    def test_arbitrary_cumulative_exposure_has_deterministic_rung(self) -> None:
        target = 900_000_000
        self.assertEqual(lc.rung_name(target), "900m")
        sources = {
            name: {"token_quota": 75_000_000, "skip_tokens": 0, "max_epochs": 1.0}
            for name in ("wiki", "hq", "structure", "colloquial")
        }
        dump(
            self.corpus / "schedule-cpt-900m.json",
            {
                "kind": "cpt",
                "parent_tokens_seen": 600_000_000,
                "exposure_tokens": 300_000_000,
                "cumulative_exposure_tokens": target,
                "parent_checkpoint": "base/parent-state.npz",
                "sampler": "quota_plan",
                "allow_repeat": False,
                "sources": sources,
            },
        )
        snapshot = lc.corpus_snapshot(self.corpus, target)
        self.assertEqual(snapshot["target_exposure"], target)
        self.assertEqual(snapshot["parent_exposure"], 600_000_000)
        self.assertEqual(snapshot["incremental_exposure"], 300_000_000)

    def test_live_state_overrides_stale_blocked_ledger(self) -> None:
        directory = self.runs / "live"
        dump(directory / "checkpoints/cpt/heartbeat.json", {"pid": 123, "unix": lc.time.time(), "tokens_seen": 42})
        with patch.object(lc, "lock_is_held", return_value={"pid": 123}), patch.object(
            lc, "pid_alive_from_meta", return_value=True
        ):
            row = lc.live_stage_state(directory)
        self.assertEqual(row["state"], "live")
        self.assertEqual(row["tokens_seen"], 42)

    def test_resume_does_not_start_a_second_stage_while_heartbeat_is_live(self) -> None:
        directory = self.runs / "protected-live"
        dump(
            directory / "run.json",
            {
                "run_id": "protected-live",
                "status": "blocked",
                "current_sha256_at_init": "current",
            },
        )
        with patch.object(lc, "RUN_ROOT", self.runs), patch.object(
            lc, "current_hash", return_value="current"
        ), patch.object(lc, "live_stage_state", return_value={"state": "live", "pid": 123}), patch.object(
            lc, "run_stage"
        ) as run_stage:
            code = lc.execute("protected-live", dry_run=False, until=None)
        self.assertEqual(code, 0)
        run_stage.assert_not_called()

    def test_completed_detached_cpt_is_adopted_without_retraining(self) -> None:
        directory = self.runs / "adopt"
        cpt = directory / "checkpoints/cpt"
        cpt.mkdir(parents=True)
        dump(
            cpt / "summary.json",
            {"tokens_seen": 600_000_000, "paused": False, "exhausted": False},
        )
        for name in ("pretrain-cpt.npz", "pretrain-cpt-state.npz", "pretrain-cpt-state.meta.json"):
            (cpt / name).write_bytes(name.encode())
        config = {
            "recipe_sha256": "recipe",
            "corpus": {"snapshot_sha256": "corpus"},
            "target_exposure": 600_000_000,
            "target_exposure_tokens": 600_000_000,
            "source_capture_mode": "reconstructed",
        }
        row = {"kind": "command", "requires": [], "command": ["must-not-run"]}
        with patch.object(lc, "run_logged_process") as process, patch.object(
            lc, "current_hash", return_value="current"
        ):
            ok, receipt = lc.run_stage(directory, config, "cpt", row, dry_run=False)
        self.assertTrue(ok)
        self.assertEqual(receipt["completion_mode"], "adopted_detached_worker")
        process.assert_not_called()

    def test_plan_lists_every_stage_without_running_internal_gates(self) -> None:
        with patch.object(lc, "RUN_ROOT", self.runs):
            lc.init_run("test-plan", self.corpus, 600_000_000)
            code = lc.execute("test-plan", dry_run=True, until=None)
        self.assertEqual(code, 0)

    def test_skills_layer_removed_control_plane_is_the_entrypoint(self) -> None:
        self.assertFalse((lc.ROOT / "skills").exists())
        self.assertFalse((lc.ROOT / ".claude" / "skills").exists())
        self.assertTrue((lc.ROOT / "src/mei_llm/__main__.py").is_file())


if __name__ == "__main__":
    unittest.main()
