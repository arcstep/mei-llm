from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

from common.paths import ARCHITECTURE_DIR, ROOT, ensure_formal_on_path, frozen_tokenizer_path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def selected_indices(count: int, limit: int) -> list[int]:
    if count < 1 or limit < 0:
        raise ValueError("positive window count and nonnegative limit required")
    if not limit or limit >= count:
        return list(range(count))
    return [(index * count) // limit for index in range(limit)]


def weighted_loss(rows: list[dict]) -> float:
    tokens = sum(row["predicted_tokens"] for row in rows)
    if tokens <= 0:
        raise ValueError("no predicted tokens")
    return sum(row["nll_sum"] for row in rows) / tokens


def run(config_path: Path) -> int:
    config = json.loads(config_path.read_text())
    output = resolve(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "config.json", config)
    inputs = {str(config_path.resolve()): digest(config_path)}
    recovery_policy = None
    if config.get("recovery_gate_policy"):
        from evaluation.base.recovery_gate import validate_policy
        policy_path = resolve(config["recovery_gate_policy"])
        recovery_policy = json.loads(policy_path.read_text())
        validate_policy(recovery_policy)
        if config.get("windows_per_role", 0) != 0:
            raise ValueError("recovery decision requires full role validation")
        inputs[str(policy_path)] = digest(policy_path)
    checkpoints = config["checkpoints"]
    multi = config.get("schema") == "mei-cpt-replay-comparison-config-v1"
    allowed_count = 2 <= len(checkpoints) <= 4 if multi else len(checkpoints) == 2
    if not allowed_count or len({row["id"] for row in checkpoints}) != len(checkpoints):
        raise ValueError("exactly two distinctly named checkpoints required")
    if multi and recovery_policy is not None:
        raise ValueError("multi-checkpoint comparison uses an explicit external comparison policy")
    for row in checkpoints:
        path = resolve(row["weights"])
        actual = digest(path)
        if actual != row["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        inputs[str(path)] = actual
    mix_path = resolve(config["layout"]) / "mix.json"
    mix = json.loads(mix_path.read_text())
    inputs[str(mix_path)] = digest(mix_path)
    for paths in mix["valid_sets"].values():
        for value in paths:
            path = resolve(value)
            inputs[str(path)] = digest(path)
    probe_path = resolve(config["probes"])
    inputs[str(probe_path)] = digest(probe_path)
    tokenizer_path = frozen_tokenizer_path()
    if digest(tokenizer_path) != config["tokenizer_sha256"]:
        raise ValueError("tokenizer hash mismatch")
    for path in (tokenizer_path, ROOT / "CURRENT.json"):
        inputs[str(path)] = digest(path)
    roots = [ARCHITECTURE_DIR, ROOT / "src/model-factory", ROOT / "src/mei_llm",
             ROOT / "src/platform/_shared/runtime"]
    sources = sorted({path for folder in roots for path in folder.rglob("*")
                      if path.is_file() and path.suffix in {".py", ".json", ".metal"}})
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    with tarfile.open(output / "source.tar.gz", "x:gz") as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    (output / "dirty.patch").write_bytes(patch)
    write_new(output / "freeze.json", {
        "schema": "mei-cpt-paired-diagnostic-freeze-v1",
        "scope": "diagnostic_only_no_promotion",
        "inputs": inputs,
        "sources": source_hashes,
        "source_archive_sha256": digest(output / "source.tar.gz"),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_patch_sha256": digest(output / "dirty.patch"),
        "pid": os.getpid(),
    })

    ensure_formal_on_path()
    import mlx.core as mx
    from architecture import NeedleZh
    from config import NeedleZhConfig
    from tokenizer import ZhTokenizerV2
    from common.checkpoint import load_params
    from common.data import PackedTokenSource
    from common.train_common import masked_lm_loss, stack_windows
    from evaluation.base.pretrain_probes import eval_probes, load_probes

    tokenizer = ZhTokenizerV2(
        tokenizer_id=tokenizer_path.stem,
        vocab_size=None,
        manifest_path=tokenizer_path.parent / f"tokenizer-{tokenizer_path.stem}-manifest.json",
    )
    probes = load_probes(probe_path)
    started = time.monotonic()
    results = {}
    batch_size = int(config.get("batch_size", 2))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    progress_path = output / "progress.jsonl"
    for checkpoint in checkpoints:
        model = NeedleZh(NeedleZhConfig.from_spec())
        loaded = load_params(model, resolve(checkpoint["weights"]), strict=True, return_report=True)
        if loaded["unexpected"] or loaded["missing"] or loaded["skipped_new"]:
            raise ValueError(f"checkpoint tensor contract mismatch: {loaded}")
        model.eval()
        roles = {}
        with progress_path.open("a", encoding="utf-8") as progress:
            for role, paths in sorted(mix["valid_sets"].items()):
                source = PackedTokenSource([resolve(value) for value in paths], 2048, tokenizer.pad_id)
                indices = selected_indices(source.n_windows, int(config.get("windows_per_role", 0)))
                batches = []
                for offset in range(0, len(indices), batch_size):
                    chosen = indices[offset:offset + batch_size]
                    stacked = stack_windows([source[index] for index in chosen])
                    loss = float(masked_lm_loss(model(stacked["x"])["logits"], stacked["y"], stacked["mask"]))
                    predicted = int(mx.sum(stacked["mask"]).item())
                    if not math.isfinite(loss):
                        raise ValueError("nonfinite validation loss")
                    batches.append({"indices": chosen, "predicted_tokens": predicted,
                                    "nll_sum": loss * predicted})
                    if offset % (64 * batch_size) == 0:
                        status = {"checkpoint": checkpoint["id"], "role": role,
                                  "completed_windows": offset + len(chosen), "total_windows": len(indices),
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}
                        progress.write(json.dumps(status) + "\n")
                        progress.flush()
                        print(json.dumps(status), flush=True)
                        if config.get("text_progress_log"):
                            with resolve(config["text_progress_log"]).open("a", encoding="utf-8") as stream:
                                stream.write(
                                    f"{datetime.now().astimezone().isoformat(timespec='seconds')} "
                                    f"EVAL checkpoint={checkpoint['id']} role={role} "
                                    f"windows={offset + len(chosen)}/{len(indices)}\n"
                                )
                roles[role] = {"valid_loss": weighted_loss(batches),
                               "predicted_tokens": sum(row["predicted_tokens"] for row in batches),
                               "nll_sum": sum(row["nll_sum"] for row in batches),
                               "windows": len(indices), "batches": batches}
                if role == "code" and not config.get("windows_per_role", 0):
                    prefix = [row for row in batches if max(row["indices"]) < 128]
                    roles[role]["first_128_loss"] = weighted_loss(prefix)
                write_new(output / f"{checkpoint['id']}-{role}.json", roles[role])
        report = {"roles": roles, "aggregate_predicted_token_weighted": weighted_loss(list(roles.values())),
                  "probes": eval_probes(model, tokenizer, probes), "load_report": loaded}
        write_new(output / f"{checkpoint['id']}.json", report)
        results[checkpoint["id"]] = report
        del model
        mx.clear_cache()
    drift = [path for path, expected in inputs.items() if digest(Path(path)) != expected]
    drift.extend(path for path, expected in source_hashes.items() if digest(ROOT / path) != expected)
    if recovery_policy is not None and not drift:
        from evaluation.base.recovery_gate import compare
        gate = compare(results[checkpoints[0]["id"]], results[checkpoints[1]["id"]], recovery_policy)
        gate["policy_sha256"] = inputs[str(policy_path)]
        gate["parent_checkpoint"] = checkpoints[0]
        gate["candidate_checkpoint"] = checkpoints[1]
        write_new(output / "recovery-gate.json", gate)
    write_new(output / "receipt.json", {
        "schema": "mei-cpt-paired-diagnostic-v1", "status": "failed" if drift else "complete",
        "changes_weights": False, "promotion_authorized": False,
        "input_or_source_drift": drift,
        "results": {name: {"aggregate_predicted_token_weighted": result["aggregate_predicted_token_weighted"],
                           "probe_mean_nll": result["probes"]["mean_nll"]} for name, result in results.items()},
        "elapsed_seconds": time.monotonic() - started,
    })
    return 2 if drift else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
