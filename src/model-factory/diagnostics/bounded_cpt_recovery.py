from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from common import source_capture
from common.paths import ROOT, ensure_formal_on_path, frozen_tokenizer_path
from common.run_lock import acquire_run_lock, write_heartbeat
from diagnostics.cpt_accumulation_resource import write_new
from evaluation.base.recovery_gate import ROLES, validate_policy


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_binding(config: dict, metadata: dict, gate: dict) -> None:
    if config.get("schema") != "mei-cpt-bounded-recovery-v1":
        raise ValueError("unsupported recovery binding")
    if config.get("scope") != "independent_diagnostic_no_promotion":
        raise ValueError("diagnostic scope must be explicit")
    if config.get("batch_size") != 1 or config.get("grad_accum") != 8 or config.get("seq_len") != 2048:
        raise ValueError("recovery requires batch=1, accumulation=8, seq=2048")
    if config.get("lr") != 3e-5 or config.get("compile_train") is not False:
        raise ValueError("recovery requires fixed 3e-5 LR and measured uncompiled path")
    if type(config.get("steps")) is not int or not 1 <= config["steps"] <= 611:
        raise ValueError("first recovery is bounded to at most 611 updates")
    if metadata.get("tokens_seen") != config["parent_tokens_seen"]:
        raise ValueError("parent exposure mismatch")
    change = config.get("configuration_change", {})
    if change != {"previous_effective_batch_tokens": 2048, "new_effective_batch_tokens": 16384,
                  "reason": "explicit_recovery_of_unpaired_batch_regression"}:
        raise ValueError("explicit batch-change binding required")
    if metadata.get("batch_size", 0) * metadata.get("grad_accum", 0) * metadata.get("seq_len", 0) != 2048:
        raise ValueError("unexpected original effective batch")
    detail = gate.get("detail", {})
    if gate.get("status") != "blocked" or detail.get("benefit_vs_parent_ok") is not False:
        raise ValueError("this diagnostic is restricted to a quality-blocked parent")
    if detail.get("identity_errors") != [] or any(detail.get(key) is not True for key in (
        "readiness_passed", "quality_evidence_present", "exposure_ok", "quota_ok", "checkpoint_ok", "schedule_ok"
    )):
        raise ValueError("parent has an integrity or evidence failure; diagnostic reuse forbidden")
    reuse = config.get("corpus_reuse", {})
    if (reuse.get("decision") != "reuse" or reuse.get("scope") != "bounded_diagnostic_only"
            or reuse.get("formal_reuse_eligible") is not False
            or reuse.get("known_quality_status") != "not_audited_diversity_degraded"
            or reuse.get("max_delta_tokens") != config["steps"] * 16384):
        raise ValueError("explicit bounded corpus reuse decision required")
    if any(config.get(key) is not False for key in (
        "automatic_parent_promotion", "release_eligible", "extend_automatically", "new_corpus_adopted"
    )):
        raise ValueError("recovery may not silently enter formal lineage or extend")


def window_quotas(weights: dict, updates: int) -> dict:
    total = updates * 8
    counts = {role: int(total * weight) for role, weight in weights.items()}
    ranking = sorted(weights, key=lambda role: (-(total * weights[role] - counts[role]), role))
    for role in ranking[:total - sum(counts.values())]:
        counts[role] += 1
    return {role: count * 2048 for role, count in counts.items()}


def verify_inputs(config: dict) -> list[str]:
    return [name for name, expected in config["input_sha256"].items()
            if not (ROOT / name).is_file() or source_capture.digest(ROOT / name) != expected]


def run(config_path: Path, *, replay: bool = False) -> int:
    config = read(config_path)
    metadata = read(ROOT / config["parent_metadata"])
    parent_evidence_key = "parent_release" if replay else "parent_gate"
    original_gate = read(ROOT / config[parent_evidence_key])
    if replay:
        from diagnostics.cpt_batch_replay import validate_replay
        validate_replay(config, metadata, original_gate, read(ROOT / config["original_schedule"]),
                        read(ROOT / config["original_terminal_metadata"]))
    else:
        validate_binding(config, metadata, original_gate)
    policy = read(ROOT / config["gate_policy"])
    validate_policy(policy)
    mix = read(ROOT / config["layout"] / "mix.json")
    if set(mix["sources"]) != ROLES or set(mix["valid_sets"]) != ROLES:
        raise ValueError("recovery requires the frozen six-role layout")
    required = {config[key] for key in ("parent_state", "parent_metadata", "parent_weights", parent_evidence_key,
                                       "gate_policy", "resource_receipt", "probes")}
    if replay:
        required.update(config[key] for key in ("original_schedule", "original_terminal_metadata", "original_gate"))
        required.update(row["weights"] for row in config["comparison_checkpoints"])
        if {row["id"] for row in config["comparison_checkpoints"]} != {"parent-1200m", "original-1500m", "recovery-1510m"}:
            raise ValueError("three frozen comparison checkpoints required")
        if any(row["sha256"] != config["input_sha256"].get(row["weights"]) for row in config["comparison_checkpoints"]):
            raise ValueError("comparison checkpoint hashes disagree")
    required.add(str(Path(config["layout"]) / "mix.json"))
    required.add(str(frozen_tokenizer_path().relative_to(ROOT)))
    required.add("CURRENT.json")
    for source in mix["sources"].values():
        required.update(source["train_shards"])
        required.update(source["valid_shards"])
    for paths in mix["valid_sets"].values():
        required.update(paths)
    if not required.issubset(config["input_sha256"]):
        raise ValueError("recovery input hash closure incomplete")
    drift = verify_inputs(config)
    if drift:
        raise ValueError(f"recovery input drift: {drift}")
    sys.path.insert(0, str(ROOT / "src/corpus-factory"))
    from quality.revocations import revoked_hashes
    revocations = revoked_hashes()
    if any(value in revocations for value in config["input_sha256"].values()):
        raise ValueError("revoked corpus artifact in recovery binding")
    resource = read(ROOT / config["resource_receipt"])
    if resource.get("status") != "passed" or resource.get("memory_budget_passed") is not True:
        raise ValueError("real 51M resource probe has not passed")
    if not replay and resource["parent_state_contract"]["checkpoint_sha256"] != config["input_sha256"][config["parent_state"]]:
        raise ValueError("resource probe used a different parent state")
    output = ROOT / config["output"]
    diagnostic_root = ROOT / "cycles/mei-1.2-51m/exp-01500m/diagnostics"
    if output.resolve().parent != diagnostic_root.resolve() or not output.name.startswith("replay-" if replay else "recovery-"):
        raise ValueError("recovery output must be a new independent 1500M diagnostic directory")
    output.mkdir(parents=True, exist_ok=False)
    def log_event(message: str) -> None:
        with (output / "train.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}\n")

    log_event("START validating inputs and freezing source; preparing parent model")
    lock = acquire_run_lock(output, {"scope": config["scope"], "steps": config["steps"]})
    started = time.monotonic()
    reporter = None
    try:
        write_new(output / "config.json", config)
        sources = source_capture.manifest(ROOT)
        write_new(output / "source-manifest.json", sources)
        capture = source_capture.capture(ROOT, output / "source-capture", sources,
                                         ["input_integrity", "bounded_recovery", "paired_evaluation", "decision"])
        write_new(output / "freeze.json", {"config_sha256": source_capture.digest(config_path),
                                           "source_capture": capture, "inputs": config["input_sha256"]})
        write_new(output / "environment.json", {
            "python": sys.version, "executable": sys.executable,
            "mlx": importlib.metadata.version("mlx"), "numpy": importlib.metadata.version("numpy"),
            "offline": True,
        })
        write_new(output / "PIPELINE.lock.json", {
            "schema": "mei-cycle-pipeline-lock-v1", "model_id": "mei-1.2-51m", "cycle_id": "exp-01500m",
            "pipeline_id": "mei-51m-cpt-batch-replay-v1" if replay else "mei-51m-cpt-bounded-recovery-v1",
            "pipeline_registry": "src/model-factory/contracts/PIPELINES.json",
            "source_capture_mode": "pre_run_frozen", "exact_reproducible": False,
            "reconstruction_note": "Source bytes are recoverable; installed Python/MLX distributions are version-recorded, not archived.",
            "current_entrypoint": "diagnostics.cpt_batch_replay" if replay else "diagnostics.bounded_cpt_recovery",
            "stages": ["input_integrity", "bounded_recovery", "paired_evaluation", "decision"],
            "run_components": [{"role": "independent_diagnostic", "run_id": output.name,
                                "run_fingerprint_sha256": source_capture.json_digest({"config": config, "source": capture}),
                                "evidence_scope": "bounded_recovery_no_automatic_formal_adoption",
                                "source_manifest_sha256": sources["manifest_sha256"]}],
            "source_recovery": {"audits": [str((output / "source-capture/capture.json").relative_to(ROOT))],
                                "manifest_entries": len(sources["files"]), "exact_recoverable": len(sources["files"]),
                                "exact_unavailable": 0, "coverage_basis": "full active source archive"},
        })
        ensure_formal_on_path()
        import mlx.core as mx
        import mlx.optimizers as optim
        from architecture import NeedleZh
        from config import NeedleZhConfig
        from common.checkpoint import flatten_params, load_train_state, save_params, save_train_state
        from common.data import PackedTokenSource, QuotaPackedSources
        from common.train_common import flatten_tree, train_lm_steps
        from orchestration.lifecycle import train_state_contract_report
        from tokenizer import ZhTokenizerV2

        state_path = ROOT / config["parent_state"]
        if state_path.with_suffix(".meta.json") != ROOT / config["parent_metadata"]:
            raise ValueError("parent state/metadata path mismatch")
        state_contract = train_state_contract_report(state_path)
        for key in ("parameter_names_and_order_exact", "parameter_shapes_exact", "optimizer_slots_complete",
                    "sampler_state_present") + (() if replay else ("source_receipt_artifacts_match",)):
            if state_contract.get(key) is not True:
                raise ValueError(f"parent contract failed: {key}")
        model = NeedleZh(NeedleZhConfig.from_spec())
        optimizer = optim.Adam(learning_rate=config["lr"])
        loaded = load_train_state(state_path, model, optimizer, mode="strict")
        if loaded != metadata:
            raise ValueError("parent metadata changed during load")
        if replay:
            original_integrity = read(ROOT / config["original_gate"])["detail"]
            if original_integrity.get("identity_errors") != [] or any(original_integrity.get(key) is not True for key in (
                "exposure_ok", "quota_ok", "checkpoint_ok", "schedule_ok"
            )):
                raise ValueError("original comparison run has integrity failures")
            state_contract["canonical_release_binding"] = {
                "path": config["parent_release"], "sha256": config["input_sha256"][config["parent_release"]],
                "state_and_metadata_hashes_match_release": True,
                "note": "canonical release supplies parent evidence; run-local gate lookup is not applicable to release paths",
            }
        for tree in (model.parameters(), optimizer.state):
            if not all(bool(mx.all(mx.isfinite(value)).item()) for value in flatten_tree(tree).values()):
                raise ValueError("nonfinite parent model or optimizer tensor")
        if sum(value.size for value in flatten_params(model).values()) != 51463797:
            raise ValueError("unexpected model parameter count")
        tokenizer_path = frozen_tokenizer_path()
        tokenizer = ZhTokenizerV2(tokenizer_id=tokenizer_path.stem, vocab_size=None,
                                  manifest_path=tokenizer_path.parent / f"tokenizer-{tokenizer_path.stem}-manifest.json")
        quotas = (read(ROOT / config["original_terminal_metadata"])["stage_tokens_drawn"] if replay
                  else window_quotas(policy["role_weights"], config["steps"]))
        streams = {role: PackedTokenSource([ROOT / name for name in source["train_shards"]], 2048, tokenizer.pad_id)
                   for role, source in sorted(mix["sources"].items())}
        for role, stream in streams.items():
            cursor = metadata["sampler_state"]["token_cursors"][role]
            if cursor < 0 or cursor + quotas[role] > stream.n_predictable_tokens:
                raise ValueError(f"physical corpus capacity exhausted: {role}")
        sampler = QuotaPackedSources(streams, quotas, seq_len=2048, seed=0, stage_id="recovery1")
        sampler.migrate_from_parent(metadata["sampler_state"], reset_sources=())
        if replay:
            from diagnostics.cpt_batch_replay import make_sampler, sampler_preflight
            write_new(output / "sampler-preflight.json", sampler_preflight(config, metadata, tokenizer.pad_id))
            sampler = make_sampler(config, metadata, tokenizer.pad_id)
        write_new(output / "readiness.json", {"status": "passed_for_bounded_diagnostic_only",
                                              "parent_contract": state_contract, "quotas": quotas,
                                              "corpus_reuse": config["corpus_reuse"], "sampler": sampler.state_dict()})
        model.train()
        checkpoints = output / "checkpoints"
        checkpoints.mkdir()
        config_hash = source_capture.digest(config_path)
        lr_schedule = read(ROOT / config["original_schedule"])["lr"] if replay else {}
        target_tokens = config["target_tokens"] if replay else metadata["tokens_seen"] + config["steps"] * 16384
        checkpoint_interval = 512 if replay else 128
        if replay:
            from common.periodic_run_report import PeriodicRunReport
            reporter = PeriodicRunReport(output)
            reporter.start()

        def checkpoint(completed: int, seen: int) -> Path:
            state = checkpoints / f"step-{completed:04d}-state.npz"
            if state.exists():
                saved = read(state.with_suffix(".meta.json"))
                if saved["tokens_seen"] != seen or saved["sampler_state"] != sampler.state_dict():
                    raise ValueError("refusing to overwrite an existing diagnostic checkpoint")
                return state
            identity = {key: metadata[key] for key in (
                "architecture_id", "architecture_sha256", "weight_contract_sha256", "runtime_profile_sha256",
                "training_aux_sha256", "params", "tokenizer_sha256", "manifest_sha256"
            )}
            next_meta = {**identity, "step": metadata["step"] + completed, "tokens_seen": seen,
                         "batch_size": 1, "grad_accum": 8, "seq_len": 2048, "compile_train": False,
                         "rung": "diagnostic-replay" if replay else "diagnostic-recovery",
                         "requested_rung": "1500m" if replay else "diagnostic-recovery",
                         "schedule_kind": "diagnostic", "curriculum_stage": sampler.stage_id,
                         "precision": "fp32", "allow_repeat": False, "init_mode": "continuation",
                         "lr_policy": "cosine_tokens" if replay else "constant", "lr": config["lr"],
                         "lr_horizon_tokens": lr_schedule.get("horizon_tokens", 0),
                         "lr_token_offset": lr_schedule.get("token_offset", 0),
                         "total_steps": metadata["step"] + config["steps"], "corpus_sha256": config_hash,
                         "sampler_state": sampler.state_dict(), "source_token_cursors": dict(sampler.token_cursors),
                         "source_cursors": dict(sampler.token_cursors), "quota_remaining": dict(sampler.quota_remaining),
                         "source_tokens_drawn": dict(sampler.source_tokens_drawn),
                         "stage_tokens_drawn": dict(sampler.stage_tokens_drawn), "window_index": sampler.consumed_windows,
                         "scope": config["scope"], "recovery_config_sha256": config_hash,
                         "parent_tokens_seen": metadata["tokens_seen"], "parent_checkpoint": config["parent_state"],
                         "schedule_sha256": config_hash, "target_tokens": target_tokens,
                         "valid_loss": None, "probe_mean_nll": None, "automatic_parent_promotion": False,
                         "release_eligible": False, "n_exposure_cap_tokens": target_tokens - metadata["tokens_seen"],
                         "tag": "diagnostic_checkpoint", "config_source": "from_spec", "mmap": True}
            save_train_state(state, model, optimizer, next_meta)
            log_event(f"CHECKPOINT step={completed} tokens={seen} file={state.name}")
            return state

        def on_step(step: int, report: dict) -> None:
            completed = step - metadata["step"] + 1
            if not math.isfinite(report["loss"]) or not math.isfinite(report["grad_norm"]):
                raise ValueError("nonfinite recovery update")
            if report["peak_bytes"] > resource["max_peak_allocation_bytes"]:
                raise ValueError("recovery exceeded frozen memory budget")
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"completed_updates": completed, **report}) + "\n")
            if replay:
                elapsed = max(time.monotonic() - training_started, 1e-6)
                delta = report["tokens_seen"] - metadata["tokens_seen"]
                write_heartbeat(output / "heartbeat.json", {
                    "phase": "training", "completed_updates": completed, "planned_updates": config["steps"],
                    "segment_tokens": delta, "target_tokens": target_tokens,
                    "progress_fraction": delta / (target_tokens - metadata["tokens_seen"]),
                    "tokens_per_second": delta / elapsed,
                    "eta_seconds": (target_tokens - report["tokens_seen"]) / max(delta / elapsed, 1e-6),
                    **report,
                })
                log_event(
                    f"TRAIN step={completed}/{config['steps']} "
                    f"progress={100 * delta / (target_tokens - metadata['tokens_seen']):.3f}% "
                    f"tokens={report['tokens_seen']} delta_tokens={delta} "
                    f"loss={report['loss']:.6f} grad_norm={report['grad_norm']:.6f} "
                    f"lr={report['lr']:.8g} tok/s={delta / elapsed:.1f} "
                    f"eta_h={(target_tokens - report['tokens_seen']) / max(delta / elapsed, 1e-6) / 3600:.2f} "
                    f"peak_GiB={report['peak_bytes'] / 2**30:.2f}"
                )
            else:
                write_heartbeat(output / "heartbeat.json", {"completed_updates": completed, **report})
            if completed % 32 == 0 or completed == 1:
                if source_capture.manifest(ROOT)["manifest_sha256"] != sources["manifest_sha256"]:
                    raise ValueError("active source changed during recovery")
                if source_capture.digest(ROOT / "CURRENT.json") != config["input_sha256"]["CURRENT.json"]:
                    raise ValueError("CURRENT changed during recovery")
                print(json.dumps({"completed_updates": completed, **report}), flush=True)
            if (completed % checkpoint_interval == 0 or replay and completed == 32) and completed < config["steps"]:
                checkpoint(completed, report["tokens_seen"])

        training_started = time.monotonic()
        result = train_lm_steps(model, sampler, optimizer=optimizer, steps=config["steps"],
                                lr=config["lr"], lr_final=lr_schedule.get("final", config["lr"]), batch_size=1, grad_accum=8,
                                horizon_tokens=lr_schedule.get("horizon_tokens"),
                                lr_token_offset=lr_schedule.get("token_offset", 0),
                                target_tokens=target_tokens if replay else None,
                                start_step=metadata["step"], start_tokens_seen=metadata["tokens_seen"],
                                allow_repeat=False, compile_train=False, on_step=on_step,
                                stop_path=output / "STOP")
        result.pop("optimizer")
        terminal_state = checkpoint(result["steps"], result["tokens_seen"])
        terminal_weights = checkpoints / f"step-{result['steps']:04d}.npz"
        save_params(model, terminal_weights)
        finite = all(bool(mx.all(mx.isfinite(value)).item())
                     for tree in (model.parameters(), optimizer.state) for value in flatten_tree(tree).values())
        drift = verify_inputs(config)
        if source_capture.manifest(ROOT)["manifest_sha256"] != sources["manifest_sha256"]:
            drift.append("active source")
        if source_capture.digest(config_path) != config_hash:
            drift.append("config")
        errors = source_capture.verify(sources, capture)
        if replay:
            original_terminal = read(ROOT / config["original_terminal_metadata"])
            if sampler.token_cursors != original_terminal["source_token_cursors"]:
                errors.append("replay terminal cursors differ from original")
        complete = (finite and not drift and not errors and not result["paused"]
                    and result["steps"] == config["steps"]
                    and result["tokens_seen"] == target_tokens
                    and sampler.stage_tokens_drawn == quotas)
        write_new(output / "training-receipt.json", {
            "status": "complete" if complete else "stopped_or_failed", "result": result,
            "weights": str(terminal_weights.relative_to(ROOT)), "weights_sha256": source_capture.digest(terminal_weights),
            "state": str(terminal_state.relative_to(ROOT)), "state_sha256": source_capture.digest(terminal_state),
            "all_model_and_optimizer_tensors_finite": finite, "input_drift": drift, "source_errors": errors,
            "sampler": sampler.state_dict(), "release_eligible": False, "automatic_parent_promotion": False,
        })
        if not complete:
            log_event("STOP training incomplete; inspect training-receipt.json")
            return 2
        log_event(f"TRAINING_COMPLETE tokens={result['tokens_seen']} steps={result['steps']}; starting evaluation")
        del model, optimizer
        mx.clear_cache()
        if replay:
            write_heartbeat(output / "heartbeat.json", {"phase": "evaluation", "completed_updates": result["steps"],
                                                       "tokens_seen": result["tokens_seen"], "training_complete": True})
            from diagnostics.cpt_batch_replay import evaluate_replay
            code, decision = evaluate_replay(config, output, terminal_weights)
            write_new(output / "receipt.json", {"status": "complete" if code == 0 else "failed", "decision": decision,
                                               "elapsed_seconds": time.monotonic() - started,
                                               "automatic_parent_promotion": False, "release_eligible": False,
                                               "extended_training": False})
            log_event(f"FINISHED evaluation_exit={code} comparison={decision.get('status')}; inspect comparison.json")
            return code
        evaluation = {"output": str((output / "evaluation").relative_to(ROOT)), "layout": config["layout"],
                      "tokenizer_sha256": config["input_sha256"][str(tokenizer_path.relative_to(ROOT))],
                      "probes": config["probes"], "batch_size": 2, "windows_per_role": 0,
                      "recovery_gate_policy": config["gate_policy"],
                      "checkpoints": [{"id": "parent-1500m", "weights": config["parent_weights"],
                                       "sha256": config["input_sha256"][config["parent_weights"]]},
                                      {"id": "recovery", "weights": str(terminal_weights.relative_to(ROOT)),
                                       "sha256": source_capture.digest(terminal_weights)}]}
        write_new(output / "evaluation.config.json", evaluation)
        from evaluation.base.paired_cpt_diagnostic import run as evaluate_pair
        code = evaluate_pair(output / "evaluation.config.json")
        decision = read(output / "evaluation/recovery-gate.json") if code == 0 else {"status": "evaluation_failed"}
        write_new(output / "receipt.json", {"status": "complete" if code == 0 else "failed", "decision": decision,
                                           "elapsed_seconds": time.monotonic() - started,
                                           "automatic_parent_promotion": False, "release_eligible": False,
                                           "extended_training": False})
        return code
    except Exception as error:
        log_event("FAILED " + traceback.format_exc())
        write_new(output / "failure.json", {"status": "failed", "error": str(error),
                                           "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        if reporter is not None:
            reporter.close()
        os.close(lock)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return run(parser.parse_args().config)


if __name__ == "__main__":
    raise SystemExit(main())
