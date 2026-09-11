from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from common.paths import ROOT
from diagnostics.bounded_cpt_recovery import read
from evaluation.base.recovery_gate import ROLES


def validate_replay(config: dict, metadata: dict, release: dict, schedule: dict, terminal: dict) -> None:
    if config.get("schema") != "mei-cpt-batch-replay-v1" or config.get("scope") != "independent_diagnostic_no_promotion":
        raise ValueError("explicit independent replay binding required")
    if (config.get("batch_size"), config.get("grad_accum"), config.get("seq_len")) != (1, 8, 2048):
        raise ValueError("replay requires effective batch 8 with microbatch 1")
    if metadata.get("batch_size", 0) * metadata.get("grad_accum", 0) * metadata.get("seq_len", 0) != 16384:
        raise ValueError("parent effective batch must be 16384 tokens")
    if metadata.get("tokens_seen") != 1200006656 or config.get("parent_tokens_seen") != metadata["tokens_seen"]:
        raise ValueError("replay must start from original 1200M state")
    if release.get("continuation_checkpoint_eligible") is not True or release.get("numeric_integrity", {}).get("status") != "passed":
        raise ValueError("parent release integrity is not eligible")
    for key, release_key in (("parent_state", "state_sha256"), ("parent_metadata", "state_meta_sha256"),
                             ("parent_weights", "weights_sha256")):
        if config["input_sha256"][config[key]] != release[release_key]:
            raise ValueError(f"canonical parent binding mismatch: {key}")
    if schedule.get("kind") != "cpt" or schedule.get("sampler") != "quota_plan" or schedule.get("allow_repeat") is not False:
        raise ValueError("original quota schedule required")
    if schedule.get("parent_tokens_seen") != metadata["tokens_seen"] or len(schedule.get("curriculum", [])) != 1:
        raise ValueError("original parent/schedule mismatch")
    expected_lr = {"base": 0.0003, "final": 0.00003, "horizon_tokens": 299993344,
                   "kind": "cosine_tokens", "token_offset": 1200006656}
    if schedule.get("lr") != expected_lr or config.get("lr") != expected_lr["base"]:
        raise ValueError("replay must retain original token-based learning rate schedule")
    if config.get("compile_train") is not False:
        raise ValueError("use the resource-tested uncompiled accumulation path")
    if terminal.get("tokens_seen") != 1500001792 or config.get("target_tokens") != terminal["tokens_seen"]:
        raise ValueError("replay must end at the original realized exposure")
    delta = terminal["tokens_seen"] - metadata["tokens_seen"]
    if config.get("steps") != math.ceil(delta / 16384):
        raise ValueError("optimizer update budget mismatch")
    if set(terminal.get("stage_tokens_drawn", {})) != ROLES or sum(terminal["stage_tokens_drawn"].values()) != delta:
        raise ValueError("original per-role exposure is incomplete")
    reuse = config.get("corpus_reuse", {})
    if (reuse.get("decision") != "reuse" or reuse.get("scope") != "explicit_same_slice_replay"
            or reuse.get("formal_reuse_eligible") is not False or reuse.get("max_delta_tokens") != delta):
        raise ValueError("explicit same-slice replay decision required")
    if any(config.get(key) is not False for key in (
        "automatic_parent_promotion", "release_eligible", "extend_automatically", "new_corpus_adopted"
    )):
        raise ValueError("replay may not automatically promote or adopt new corpus")


def make_sampler(config: dict, metadata: dict, pad_id: int):
    from common.data import load_scheduled_train
    schedule = read(ROOT / config["original_schedule"])
    sampler = load_scheduled_train(ROOT / config["layout"], 2048, pad_id,
                                   schedule_path=ROOT / config["original_schedule"],
                                   curriculum_stage=schedule["curriculum"][0]["id"],
                                   tokens_seen=metadata["tokens_seen"])
    sampler.migrate_from_parent(metadata["sampler_state"], reset_sources=())
    return sampler


def sampler_preflight(config: dict, metadata: dict, pad_id: int) -> dict:
    sampler = make_sampler(config, metadata, pad_id)
    terminal = read(ROOT / config["original_terminal_metadata"])
    total = 0
    windows = 0
    order = hashlib.sha256()
    delta = terminal["tokens_seen"] - metadata["tokens_seen"]
    while total < delta:
        window = sampler.take_one()
        if window is None:
            raise ValueError("original slice exhausted before target")
        total += int(sum(window["mask"]))
        windows += 1
        order.update(f"{window['source_id']}:{window['source_token_cursor']}\n".encode())
    if (total != delta or sampler.stage_tokens_drawn != terminal["stage_tokens_drawn"]
            or sampler.token_cursors != terminal["source_token_cursors"]):
        raise ValueError("replayed slice does not match original terminal role exposure/cursors")
    return {"status": "passed", "predicted_tokens": total, "windows": windows,
            "source_cursor_order_sha256": order.hexdigest(), "sampler": sampler.state_dict(),
            "historical_order_assurance": "deterministic original schedule and seed; terminal role/cursor equality",
            "shared_ledger_changed": False}


def main() -> int:
    from diagnostics.bounded_cpt_recovery import run
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return run(parser.parse_args().config, replay=True)


def evaluate_replay(config: dict, output: Path, weights: Path) -> tuple[int, dict]:
    from common.source_capture import digest
    from common.paths import frozen_tokenizer_path
    from diagnostics.cpt_accumulation_resource import write_new
    from evaluation.base.paired_cpt_diagnostic import run as evaluate
    from evaluation.base.recovery_gate import compare
    evaluation = {"schema": "mei-cpt-replay-comparison-config-v1",
                  "output": str((output / "evaluation").relative_to(ROOT)),
                  "layout": config["layout"], "probes": config["probes"],
                  "tokenizer_sha256": config["input_sha256"][str(frozen_tokenizer_path().relative_to(ROOT))],
                  "batch_size": 2, "windows_per_role": 0,
                  "text_progress_log": str((output / "train.log").relative_to(ROOT)),
                  "checkpoints": [*config["comparison_checkpoints"],
                                  {"id": "retrained-1500m", "weights": str(weights.relative_to(ROOT)), "sha256": digest(weights)}]}
    write_new(output / "evaluation.config.json", evaluation)
    code = evaluate(output / "evaluation.config.json")
    if code:
        return code, {"status": "evaluation_failed"}
    policy = read(ROOT / config["gate_policy"])
    candidate = read(output / "evaluation/retrained-1500m.json")
    comparisons = {row["id"]: compare(read(output / f"evaluation/{row['id']}.json"), candidate, policy)
                   for row in config["comparison_checkpoints"]}
    checks = comparisons["original-1500m"]["checks"]
    decision = {"schema": "mei-cpt-batch-replay-comparison-v1",
                "status": "expected_improvement_observed" if all(checks.values()) else "expected_improvement_not_confirmed",
                "comparisons": comparisons, "policy_sha256": digest(ROOT / config["gate_policy"]),
                "primary_comparison": "original-1500m", "promotion_authorized": False,
                "limitations": "single seed; accumulation and compilation path differ; recovery reference has 10M extra exposure; seven probes do not measure product ability"}
    write_new(output / "comparison.json", decision)
    return 0, decision


if __name__ == "__main__":
    raise SystemExit(main())
