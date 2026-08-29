#!/usr/bin/env python3
"""Promote the completed 51M scratch run without touching any 58M release."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _repo import ROOT, TRAIN_RUNS
from pretrain_gates import REQUIRED_ROLES, SCRATCH_EXPOSURE_TOKENS, SCRATCH_SOURCE_QUOTAS

ARCHITECTURE_ID = "mei-1.0-51m-arch-v1"
MODEL_ID = "mei-1.0-51m-base-scratch300m-v1"
RUN_NAME = "pretrain-mei-1.0-51m-base-scratch300m-v1"
EXPECTED_PARAMS = 51_463_797


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def activate_current() -> None:
    current_path = ROOT / "CURRENT.json"
    current = load(current_path)
    current.update(
        {
            "architecture": f"architecture/{ARCHITECTURE_ID}",
            "architecture_id": ARCHITECTURE_ID,
            "base": f"base/{MODEL_ID}",
            "base_model_id": MODEL_ID,
            "product": "mei-1.0-51m",
            "sft": None,
            "runtime": None,
            "stage": "scratch-300m-promoted",
        }
    )
    pending = current_path.with_suffix(".json.pending-51m")
    pending.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    pending.replace(current_path)


def validate_identity(summary: dict, state_meta: dict) -> str | None:
    for label, payload in (("summary", summary), ("state metadata", state_meta)):
        got = payload.get("architecture_id")
        if got != ARCHITECTURE_ID:
            return f"{label} architecture_id mismatch: got={got!r} expected={ARCHITECTURE_ID!r}"
        params = int(payload.get("params") or 0)
        if params != EXPECTED_PARAMS:
            return f"{label} params mismatch: got={params} expected={EXPECTED_PARAMS}"

    summary_sha = summary.get("architecture_sha256")
    state_sha = state_meta.get("architecture_sha256")
    if not summary_sha or summary_sha != state_sha:
        return (
            "architecture_sha256 missing or inconsistent: "
            f"summary={summary_sha!r} state={state_sha!r}"
        )
    return None


def main() -> int:
    run_dir = TRAIN_RUNS / RUN_NAME
    summary_path = run_dir / "summary.json"
    weights = run_dir / f"{RUN_NAME}.npz"
    state = run_dir / f"{RUN_NAME}-state.npz"
    state_meta_path = state.with_suffix(".meta.json")

    summary = load(summary_path)
    state_meta = load(state_meta_path)
    if not summary or not state_meta:
        return fail(f"missing 51M final summary or state metadata in {run_dir}")
    identity_error = validate_identity(summary, state_meta)
    if identity_error:
        return fail(identity_error)
    if (run_dir / "NOT_FOR_PROMOTE.json").is_file():
        return fail("51M run is marked NOT_FOR_PROMOTE")
    if int(summary.get("tokens_seen") or 0) < SCRATCH_EXPOSURE_TOKENS:
        return fail(f"tokens_seen {summary.get('tokens_seen')} < {SCRATCH_EXPOSURE_TOKENS}")
    if int(summary.get("seq_len") or 0) != 2048:
        return fail("final seq_len must be 2048")
    if summary.get("curriculum_stage") != "s3":
        return fail("final curriculum_stage must be s3")
    if summary.get("allow_repeat"):
        return fail("repeat epochs cannot be promoted")
    if summary.get("parent_checkpoint") not in (None, ""):
        return fail("scratch base cannot have a parent checkpoint")

    drawn = summary.get("source_tokens_drawn") or {}
    for name, quota in SCRATCH_SOURCE_QUOTAS.items():
        got = int(drawn.get(name) or 0)
        if abs(got - quota) > 2048:
            return fail(f"{name} drawn {got} outside quota {quota} ±2048")
    for name in REQUIRED_ROLES:
        if name != "wiki" and summary.get(f"valid_loss_{name}") is None:
            return fail(f"missing valid_loss_{name}")
    if summary.get("valid_loss") is None:
        return fail("missing wiki valid_loss")
    if not weights.is_file() or not state.is_file():
        return fail("missing final 51M weights or train state")

    weights_hash = sha256(weights)
    state_hash = sha256(state)
    dest = ROOT / "base" / MODEL_ID
    release_path = dest / "RELEASE.json"
    if dest.exists():
        existing = load(release_path)
        if (
            existing.get("architecture_id") == ARCHITECTURE_ID
            and existing.get("weights_sha256") == weights_hash
            and existing.get("state_sha256") == state_hash
        ):
            activate_current()
            print(
                json.dumps(
                    {
                        "ok": True,
                        "already_promoted": True,
                        "base": str(dest.relative_to(ROOT)),
                        "current_product": "mei-1.0-51m",
                    },
                    indent=2,
                )
            )
            return 0
        return fail(f"refusing to overwrite non-identical release: {dest}")

    temp = dest.with_name(f".{dest.name}.promoting")
    if temp.exists():
        return fail(f"stale promotion directory exists: {temp}")
    temp.mkdir(parents=True)

    weights_name = f"{MODEL_ID}.npz"
    state_name = f"{MODEL_ID}-state.npz"
    shutil.copy2(weights, temp / weights_name)
    shutil.copy2(state, temp / state_name)
    shutil.copy2(state_meta_path, temp / f"{MODEL_ID}-state.meta.json")
    shutil.copy2(summary_path, temp / "summary.json")

    if sha256(temp / weights_name) != weights_hash or sha256(temp / state_name) != state_hash:
        return fail("copied 51M artifacts failed hash verification")

    release = {
        "model_id": MODEL_ID,
        "architecture_id": ARCHITECTURE_ID,
        "architecture_sha256": summary["architecture_sha256"],
        "kind": "base-scratch",
        "params": EXPECTED_PARAMS,
        "tokens_seen_exposure": int(summary["tokens_seen"]),
        "exposure_tokens": SCRATCH_EXPOSURE_TOKENS,
        "source_tokens_drawn": drawn,
        "seq_curriculum": [512, 1024, 2048],
        "max_trained_seq": 2048,
        "corpus": "corpus/lm-v1",
        "schedule": "corpus/lm-v1/schedule-scratch.json",
        "tokenizer_sha256": summary.get("tokenizer_sha256"),
        "corpus_sha256": summary.get("corpus_sha256"),
        "schedule_sha256": summary.get("schedule_sha256"),
        "valid_loss": summary.get("valid_loss"),
        "valid_loss_hq": summary.get("valid_loss_hq"),
        "valid_loss_structure": summary.get("valid_loss_structure"),
        "valid_loss_colloquial": summary.get("valid_loss_colloquial"),
        "final_probe_mean_nll": summary.get("final_probe_mean_nll"),
        "weights": weights_name,
        "train_state": state_name,
        "weights_sha256": weights_hash,
        "state_sha256": state_hash,
        "parent_checkpoint": None,
        "init_mode": "scratch",
        "source_run": str(run_dir.relative_to(ROOT)),
        "public_distribution_clearance_asserted": False,
        "not_a_claim": "LM loss/probes are not tool-calling ability.",
    }
    (temp / "RELEASE.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    temp.replace(dest)
    activate_current()
    print(
        json.dumps(
            {
                "ok": True,
                "base": str(dest.relative_to(ROOT)),
                "architecture_id": ARCHITECTURE_ID,
                "current_product": "mei-1.0-51m",
                "tokens_seen": summary["tokens_seen"],
                "weights_sha256": weights_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
