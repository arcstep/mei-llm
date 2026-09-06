from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestration import productize_51m as productize
from release import final_audit_51m as final_audit

HERE = Path(__file__).resolve().parent
PRODUCTIZE_SOURCE = HERE.parent / "orchestration" / "productize_51m.py"


class Productize51MTest(unittest.TestCase):
    def test_final_audit_accepts_complete_five_way_semantic_boundary(self) -> None:
        self.assertTrue(
            final_audit.semantic_boundaries_complete(
                dict(final_audit.EXPECTED_SEMANTIC_BOUNDARIES)
            )
        )
        missing_narration = dict(final_audit.EXPECTED_SEMANTIC_BOUNDARIES)
        missing_narration.pop("narration_adapter")
        self.assertFalse(final_audit.semantic_boundaries_complete(missing_narration))

    def test_default_plan_is_arbitrary_base_and_semantically_separated(self) -> None:
        args = productize.parse_args(["--dry-run"])
        with mock.patch.object(productize, "live_cpt_workers", return_value=[]):
            plan = productize.build_plan(args)
        self.assertEqual(plan["immutable"]["product"], "mei-1.0-51m")
        self.assertEqual(plan["immutable"]["base"]["tokens_seen_exposure"], 300_000_485)
        self.assertEqual(
            plan["immutable"]["semantic_boundaries"],
            {
                "retrieval": "independent-contrastive-head",
                "mw_disposition": "independent-20class-sidecar",
                "confidence": "independent-calibrated-binary-head",
                "narration_adapter": (
                    "independent-frozen-backbone-rank16-generation-sidecar"
                ),
                "mw_deviation": "deterministic-governance-gate-no-tensors",
            },
        )
        self.assertEqual(plan["immutable"]["data_release"]["mw_deviation"]["tensor_count"], 0)
        self.assertEqual([row["stage_id"] for row in plan["stages"]], list(productize.STAGES))
        self.assertFalse(plan["process_complete"])
        self.assertFalse(plan["immutable"]["execution_policy"]["allow_live_cpt"])
        self.assertFalse(plan["immutable"]["execution_policy"]["qat_only"])
        self.assertEqual(
            plan["immutable"]["execution_policy"][
                "mlx_global_memory_limit_bytes"
            ],
            8 * 1024**3,
        )
        self.assertEqual(
            plan["immutable"]["execution_policy"][
                "mlx_global_cache_limit_bytes"
            ],
            256 * 1024**2,
        )

    def test_run_dir_collision_never_overwrites_different_plan(self) -> None:
        plan = {"run_fingerprint_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temp:
            requested = Path(temp) / "candidate"
            requested.mkdir()
            (requested / "plan.json").write_text(
                json.dumps({"run_fingerprint_sha256": "b" * 64}), encoding="utf-8"
            )
            resolved = productize.choose_run_dir(requested, plan, resume=True)
            self.assertEqual(resolved.name, "candidate-aaaaaaaaaaaa")
            self.assertNotEqual(resolved, requested)

            resolved.mkdir()
            second = productize.choose_run_dir(requested, plan, resume=False)
            self.assertEqual(second.name, "candidate-aaaaaaaaaaaa-001")

    def test_stage_reuse_requires_fingerprint_and_output_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            expected = {"stage_fingerprint_sha256": "1" * 64}
            receipt = {
                "stage_fingerprint_sha256": "1" * 64,
                "terminal_status": "passed",
                "output_hashes": {str(artifact.resolve()): productize.sha_file(artifact)},
            }
            self.assertTrue(productize._receipt_reusable(receipt, expected))
            artifact.write_text('{"drift":true}', encoding="utf-8")
            self.assertFalse(productize._receipt_reusable(receipt, expected))

    def test_catalog_projects_frozen_metadata_to_public_tool_schema(self) -> None:
        rows = productize._catalog(productize.DEFAULT_EVAL_LOCK)
        self.assertTrue(rows)
        self.assertTrue(all("family" not in row and "tool_id" not in row for row in rows))
        self.assertTrue(all(set(row) <= {
            "name", "description", "parameters", "required_permissions",
            "required_state", "x-mei-permissions", "x-mei-state"
        } for row in rows))
        policy = next(row for row in rows if row["name"] == "update_room_policy")
        self.assertEqual(
            policy["parameters"]["properties"]["work_start"]["pattern"],
            r"^[0-9]{2}:[0-9]{2}$",
        )

    def test_oracle_eval_resolves_through_portable_projection(self) -> None:
        catalog = productize._catalog(productize.DEFAULT_EVAL_LOCK)
        frozen = json.loads(
            (productize.DEFAULT_EVAL_LOCK / "tool-universe-v1.json").read_text(
                encoding="utf-8"
            )
        )["tools"]
        source = next(row for row in frozen if row["name"] == "update_room_policy")
        self.assertEqual(
            source["parameters"]["properties"]["work_start"]["pattern"],
            r"^\d{2}:\d{2}$",
        )
        projected = productize._project_oracle_tools(
            {"oracle_top5": [source]}, catalog
        )
        self.assertEqual(
            projected[0]["parameters"]["properties"]["work_start"]["pattern"],
            r"^[0-9]{2}:[0-9]{2}$",
        )

    def test_oracle_eval_fails_closed_for_unknown_tool(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside deployment projection"):
            productize._project_oracle_tools(
                {"oracle_top5": [{"name": "not_in_frozen_catalog"}]}, []
            )

    def test_locked_eval_mlx_policy_bounds_and_releases_cache(self) -> None:
        fake = mock.Mock()
        fake.set_memory_limit.return_value = 99
        fake.set_cache_limit.return_value = 88
        fake.get_active_memory.return_value = 77
        fake.get_cache_memory.return_value = 0
        fake.get_peak_memory.return_value = 66

        policy = productize._configure_locked_eval_mlx(fake)
        productize._release_locked_eval_mlx(fake, collect_python=False)
        snapshot = productize._locked_eval_memory_snapshot(fake)

        fake.set_memory_limit.assert_called_once_with(8 * 1024**3)
        fake.set_cache_limit.assert_called_once_with(256 * 1024**2)
        self.assertGreaterEqual(fake.clear_cache.call_count, 2)
        fake.reset_peak_memory.assert_called_once_with()
        self.assertEqual(policy["previous_memory_limit_bytes"], 99)
        self.assertEqual(policy["previous_cache_limit_bytes"], 88)
        self.assertEqual(snapshot, {"active_bytes": 77, "cache_bytes": 0, "peak_bytes": 66})

    def test_live_cpt_defers_execution_without_writing_current(self) -> None:
        args = argparse.Namespace()
        before = productize.sha_file(productize.CURRENT_PATH)
        with mock.patch.object(
            productize,
            "live_cpt_workers",
            return_value=[{"pid": 7, "tokens_seen": 400_000_000}],
        ):
            with self.assertRaisesRegex(RuntimeError, "live CPT owns MLX/Metal"):
                productize.execute(args, {})
        self.assertEqual(productize.sha_file(productize.CURRENT_PATH), before)

    def test_explicit_live_cpt_authorization_executes_independent_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(
                allow_live_cpt=True,
                run_dir=Path(temp) / "productize-300m",
                resume=False,
            )
            plan = {"run_fingerprint_sha256": "c" * 64}
            final = {
                "final_audit": {
                    "metrics": {"process_complete": True, "release_eligible": False}
                }
            }
            with (
                mock.patch.object(
                    productize,
                    "live_cpt_workers",
                    return_value=[{"pid": 7, "tokens_seen": 400_000_000}],
                ),
                mock.patch.object(productize, "write_plan"),
                mock.patch.object(
                    productize, "_training_and_package_stages", return_value=final
                ),
            ):
                result = productize.execute(args, plan)
            self.assertTrue(result["process_complete"])
            self.assertFalse(result["release_eligible"])

    def test_qat_only_is_fingerprint_bound_and_stops_before_sft(self) -> None:
        default_args = productize.parse_args(["--dry-run"])
        qat_args = productize.parse_args(["--dry-run", "--qat-only"])
        with mock.patch.object(productize, "live_cpt_workers", return_value=[]):
            default_plan = productize.build_plan(default_args)
            qat_plan = productize.build_plan(qat_args)
        self.assertNotEqual(
            default_plan["run_fingerprint_sha256"],
            qat_plan["run_fingerprint_sha256"],
        )
        self.assertTrue(qat_plan["immutable"]["execution_policy"]["qat_only"])

    def test_float_anchor_subprocess_receives_absolute_base_paths(self) -> None:
        source = PRODUCTIZE_SOURCE.read_text(encoding="utf-8")
        self.assertIn('str(args.base_weights.resolve())', source)
        self.assertIn('str(args.base_release.resolve())', source)

    def test_qat_worker_receives_absolute_artifact_paths(self) -> None:
        source = PRODUCTIZE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("base_release=args.base_release.resolve()", source)
        self.assertIn("base_weights=args.base_weights.resolve()", source)
        self.assertIn("float_anchor=anchor_path.resolve()", source)
        self.assertIn("replay_corpus=args.replay_corpus.resolve()", source)
        self.assertIn('run_dir=(directory / "worker").resolve()', source)

    def test_qat_only_defers_inputs_owned_by_later_stages(self) -> None:
        source = PRODUCTIZE_SOURCE.read_text(encoding="utf-8")
        early_return = source.index('if bool(getattr(args, "qat_only", False)):', source.index("def _training_and_package_stages"))
        self.assertGreater(source.index('mw_rows = load_jsonl(data / "mw-disposition.train.jsonl")'), early_return)
        self.assertGreater(source.index('confidence_rows = load_jsonl(data / "confidence-calibration.train.jsonl")'), early_return)
        self.assertGreater(source.index('narration_train_rows = load_jsonl(args.narration_release / "narration.train.jsonl")'), early_return)

    def test_execute_normalizes_selected_run_dir_before_receipt_registration(self) -> None:
        source = PRODUCTIZE_SOURCE.read_text(encoding="utf-8")
        execute = source[source.index("def execute(") : source.index("def parse_args(")]
        self.assertIn(
            "choose_run_dir(args.run_dir, plan, resume=args.resume).resolve()",
            execute,
        )


if __name__ == "__main__":
    unittest.main()
