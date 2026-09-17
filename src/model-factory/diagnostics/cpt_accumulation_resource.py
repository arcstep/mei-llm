from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from common import source_capture
from common.paths import ROOT, ensure_formal_on_path


def write_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_config(config: dict) -> None:
    if config.get("schema") != "mei-cpt-accumulation-resource-config-v1":
        raise ValueError("unsupported resource probe config")
    if config.get("scope") != "synthetic_resource_only_no_saved_weights":
        raise ValueError("resource probe scope must be explicit")
    if config.get("batch_size") != 1 or config.get("grad_accum") != 8 or config.get("seq_len") != 2048:
        raise ValueError("resource probe requires batch=1, accumulation=8, seq=2048")
    if type(config.get("steps")) is not int or not 1 <= config["steps"] <= 4:
        raise ValueError("resource probe is bounded to 1..4 optimizer updates")
    if not math.isfinite(config["lr"]) or not 0 < config["lr"] <= 3e-5:
        raise ValueError("resource probe learning rate exceeds recovery bound")
    if type(config.get("max_peak_allocation_bytes")) is not int or config["max_peak_allocation_bytes"] <= 0:
        raise ValueError("explicit positive resource budget required")


def run(config_path: Path) -> int:
    config = json.loads(config_path.read_text())
    validate_config(config)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "config.json", config)
    inputs = {str(config_path.resolve()): source_capture.digest(config_path),
              str(ROOT / "CURRENT.json"): source_capture.digest(ROOT / "CURRENT.json")}
    for key in ("parent_state", "parent_metadata"):
        path = ROOT / config[key]
        actual = source_capture.digest(path)
        if actual != config[f"{key}_sha256"]:
            raise ValueError(f"input hash mismatch: {key}")
        inputs[str(path)] = actual
    state_path = ROOT / config["parent_state"]
    if state_path.with_suffix(".meta.json") != ROOT / config["parent_metadata"]:
        raise ValueError("metadata does not belong to selected state")
    sources = source_capture.manifest(ROOT)
    write_new(output / "source-manifest.json", sources)
    binding = source_capture.capture(ROOT, output / "source-capture", sources,
                                     ["strict_state_load", "synthetic_accumulation", "integrity_check"])
    write_new(output / "freeze.json", {"inputs": inputs, "source_capture": binding, "pid": os.getpid()})
    started = time.monotonic()
    try:
        ensure_formal_on_path()
        import mlx.core as mx
        import mlx.optimizers as optim
        from architecture import NeedleZh
        from config import NeedleZhConfig
        from common.checkpoint import load_train_state
        from common.train_common import flatten_tree, peak_bytes, train_lm_steps
        from orchestration.lifecycle import train_state_contract_report

        contract = train_state_contract_report(state_path)
        for key in ("parameter_names_and_order_exact", "parameter_shapes_exact", "optimizer_slots_complete",
                    "sampler_state_present", "source_receipt_artifacts_match"):
            if contract.get(key) is not True:
                raise ValueError(f"parent state contract failed: {key}")
        model = NeedleZh(NeedleZhConfig.from_spec())
        optimizer = optim.Adam(learning_rate=config["lr"])
        metadata = load_train_state(state_path, model, optimizer, mode="strict")
        model.train()
        mx.eval(model.parameters(), optimizer.state)
        params = sum(value.size for value in flatten_tree(model.parameters()).values())
        if params != 51463797:
            raise ValueError(f"model parameter mismatch: {params}")
        sequence_length = config["seq_len"]
        windows = [{"x": [11 + (offset + index) % 1000 for index in range(sequence_length)],
                    "y": [11 + (offset + index + 1) % 1000 for index in range(sequence_length)],
                    "mask": [1] * sequence_length}
                   for offset in range(config["steps"] * config["grad_accum"])]
        updates = []

        def on_step(step: int, report: dict) -> None:
            row = {"step": step, **report}
            updates.append(row)
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)

        result = train_lm_steps(model, windows, optimizer=optimizer, steps=config["steps"],
                                lr=config["lr"], lr_final=config["lr"], batch_size=1, grad_accum=8,
                                allow_repeat=False, compile_train=False, on_step=on_step)
        result.pop("optimizer")
        numeric_finite = all(bool(mx.all(mx.isfinite(value)).item())
                             for value in flatten_tree(model.parameters()).values())
        drift = [path for path, expected in inputs.items() if source_capture.digest(Path(path)) != expected]
        if source_capture.manifest(ROOT)["manifest_sha256"] != sources["manifest_sha256"]:
            drift.append("active source manifest")
        archive_errors = source_capture.verify(sources, binding)
        measured_peak = peak_bytes()
        memory_ok = measured_peak is not None and measured_peak <= config["max_peak_allocation_bytes"]
        passed = (numeric_finite and not drift and not archive_errors
                  and memory_ok
                  and result["steps"] == config["steps"]
                  and result["tokens_seen"] == config["steps"] * 16384)
        write_new(output / "receipt.json", {
            "schema": "mei-cpt-accumulation-resource-v1", "status": "passed" if passed else "failed",
            "params": params, "source_parent_tokens_seen": metadata["tokens_seen"],
            "parent_state_contract": contract, "result": result, "peak_allocation_bytes": measured_peak,
            "memory_budget_passed": memory_ok, "max_peak_allocation_bytes": config["max_peak_allocation_bytes"],
            "model_tensors_finite": numeric_finite, "input_drift": drift,
            "source_archive_errors": archive_errors, "elapsed_seconds": time.monotonic() - started,
            "synthetic_tokens_added_to_cpt_chain": 0, "weights_saved": False,
            "quality_evaluated": False, "automatic_parent_promotion": False,
        })
        return 0 if passed else 2
    except Exception as error:
        write_new(output / "failure.json", {"status": "failed", "error": str(error),
                                           "elapsed_seconds": time.monotonic() - started})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return run(parser.parse_args().config)


if __name__ == "__main__":
    raise SystemExit(main())
