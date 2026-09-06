#!/usr/bin/env python3
"""Base-parameterized, receipt-driven 51M productization controller.

The controller is intentionally independent of the 300M model ID.  A frozen
300M, 600M, 900M, or later base is accepted only when its immutable release
resolves to the canonical 51,463,797-weight contract.  Heavy stages are
deferred while any live CPT worker owns MLX/Metal unless the caller records an
explicit concurrent-execution authorization in the immutable run plan.

This module owns orchestration and terminal evidence.  Numerical training
remains in the focused CQ2/SFT modules; package export remains in the portable
v2 exporter.  ``MW deviation`` is represented only by the frozen governance
receipt.  The independent 20-class trainable sidecar is always named
``mw_disposition``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from common._repo import (
    CURRENT_PATH,
    ROOT,
    TOKENIZER_ZH_V1,
    architecture_contracts,
    legacy_weight_contract_sha256,
    resolve_repo_path,
)
from common.mlx_memory_policy_51m import (
    CACHE_LIMIT_BYTES as PRODUCTIZATION_MLX_CACHE_LIMIT_BYTES,
    MEMORY_LIMIT_BYTES as PRODUCTIZATION_MLX_MEMORY_LIMIT_BYTES,
    POLICY_ID as PRODUCTIZATION_MLX_POLICY_ID,
    configure_mlx_memory,
    mlx_memory_snapshot,
    release_mlx_memory,
)


PRODUCT_ID = "mei-1.0-51m"
EXPECTED_PARAMS = 51_463_797
DEFAULT_BASE_DIR = ROOT / "cycles/mei-1.1-51m/exp-00300m/models/base/mei-1.0-51m-base-scratch300m-v1"
DEFAULT_BASE_RELEASE = DEFAULT_BASE_DIR / "RELEASE.json"
DEFAULT_BASE_WEIGHTS = DEFAULT_BASE_DIR / "mei-1.0-51m-base-scratch300m-v1.npz"
DEFAULT_DATA_RELEASE = (
    ROOT / "cycles/mei-1.1-51m/exp-00300m/corpus/sft-suite/historical-notebook-releases/releases/mei-1.0-51m-tool-sft-v3-300m-v2"
)
DEFAULT_EVAL_LOCK = ROOT / "cycles/mei-1.1-51m/_legacy/notebook/evaluation/banks/sft-v2-eval-lock-v3-20class"
DEFAULT_NARRATION_RELEASE = (
    ROOT / "cycles/mei-1.1-51m/exp-00300m/corpus/sft-suite/historical-notebook-releases/releases/mei-1.0-51m-narration-sft-agent300m-v3"
)
DEFAULT_REPLAY_CORPUS = ROOT / "cycles/mei-1.1-51m/exp-00300m/corpus/cpt-delta/lm-v1"
DEFAULT_QUALITY_THRESHOLDS = (
    ROOT / "cycles/mei-1.1-51m/_legacy/notebook/evaluation/jobs/mei-1.0-51m/sft-v2-51m-thresholds-preregister.json"
)
DEFAULT_PACKAGE_ID = "mei-1.0-51m-scratch300m-tool-sft-v3-cq2-v2"
DEFAULT_RUN_DIR = (
    ROOT / "cycles/mei-1.1-51m/exp-00300m/runs/productize-scratch300m-sft-v3-cq2-v2"
)
LOCKED_EVAL_PROGRESS_INTERVAL = 25

TERMINAL = {"passed", "degraded", "blocked"}
STAGES = (
    "float_base_lm_anchor",
    "float_task_control",
    "q4_diagnostic",
    "cq2_qat_v2",
    "retrieval_r0",
    "oracle_top5_fullcall",
    "oracle_top5_agent_continuation",
    "retrieval_r1",
    "tool_index_final",
    "learned_top5_e2e",
    "mw_disposition_20class",
    "confidence_calibration",
    "narration_adapter",
    "package_v2_cq2",
    "mtp_ablation",
    "portable_runtime_gates",
    "resource_measurement",
    "needle2_alignment_report",
    "final_audit",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.replace(path)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def live_cpt_workers(now: float | None = None) -> list[dict[str, Any]]:
    """Reconcile a worker from live PID *and* a recent advancing heartbeat."""

    current = time.time() if now is None else float(now)
    rows: list[dict[str, Any]] = []
    root = ROOT / "cycles/mei-1.1-51m"
    for heartbeat_path in root.glob("exp-*/runs/**/checkpoints/*/heartbeat.json"):
        try:
            heartbeat = load_json(heartbeat_path)
        except RuntimeError:
            continue
        pid = int(heartbeat.get("pid") or 0)
        age = current - float(heartbeat.get("unix") or 0.0)
        if _pid_alive(pid) and 0 <= age < 180:
            rows.append(
                {
                    "pid": pid,
                    "heartbeat": str(heartbeat_path.relative_to(ROOT)),
                    "age_seconds": age,
                    "tokens_seen": int(heartbeat.get("tokens_seen") or 0),
                    "step": int(heartbeat.get("step") or 0),
                    "rung": heartbeat.get("rung"),
                }
            )
    return sorted(rows, key=lambda row: (str(row["heartbeat"]), int(row["pid"])))


def validate_base(release_path: Path, weights_path: Path) -> dict[str, Any]:
    if not release_path.is_file() or not weights_path.is_file():
        raise RuntimeError("selected frozen base release/weights are missing")
    release = load_json(release_path)
    if release.get("architecture_id") != "mei-1.0-51m-arch-v1":
        raise RuntimeError("base architecture_id is not the canonical 51M architecture")
    if (
        int(release.get("params") or release.get("parameter_count") or 0)
        != EXPECTED_PARAMS
    ):
        raise RuntimeError("base parameter count is not exactly 51,463,797")
    actual_sha = sha_file(weights_path)
    if actual_sha != str(release.get("weights_sha256") or ""):
        raise RuntimeError("base weights SHA does not match immutable RELEASE.json")
    contracts = architecture_contracts()
    declared = release.get("weight_contract_sha256") or legacy_weight_contract_sha256(
        str(release.get("architecture_sha256") or "")
    )
    if declared != contracts["weight_contract_sha256"]:
        raise RuntimeError("base does not resolve to the canonical 51M weight contract")
    tokenizer_sha = str(release.get("tokenizer_sha256") or "")
    if tokenizer_sha and tokenizer_sha != sha_file(TOKENIZER_ZH_V1):
        raise RuntimeError("base tokenizer does not match frozen zh-24k-v1")
    productization_eligible = release.get("productization_experiment_eligible")
    if productization_eligible is False or (
        release.get("promotion_prohibited") is True
        and productization_eligible is not True
    ):
        raise RuntimeError(
            "base release is not eligible for an explicit productization experiment"
        )
    numeric_integrity = release.get("numeric_integrity")
    if release.get("kind") == "base-cpt-candidate" and (
        not isinstance(numeric_integrity, dict)
        or numeric_integrity.get("schema") != "mei-cpt-numeric-integrity-v1"
        or numeric_integrity.get("status") != "passed"
        or numeric_integrity.get("all_numeric_tensors_finite") is not True
        or numeric_integrity.get("parameter_names_and_order_exact") is not True
        or numeric_integrity.get("parameter_shapes_exact") is not True
        or int(numeric_integrity.get("parameter_tensor_count") or 0) != 400
        or int(numeric_integrity.get("parameter_count") or 0) != EXPECTED_PARAMS
        or int(numeric_integrity.get("train_state_tokens_seen") or 0)
        != int(release.get("tokens_seen_exposure") or 0)
    ):
        raise RuntimeError("CPT Base has no passed numeric-integrity evidence")
    corpus_quality = release.get("corpus_quality") or {}
    return {
        "base_id": str(release.get("model_id") or release_path.parent.name),
        "release": str(release_path.resolve()),
        "release_sha256": sha_file(release_path),
        "weights": str(weights_path.resolve()),
        "weights_sha256": actual_sha,
        "tokens_seen_exposure": int(
            release.get("tokens_seen_exposure") or release.get("tokens_seen") or 0
        ),
        "tokenizer_sha256": tokenizer_sha or sha_file(TOKENIZER_ZH_V1),
        "base_release_eligible": release.get("release_eligible"),
        "productization_experiment_eligible": (
            True if productization_eligible is None else bool(productization_eligible)
        ),
        "lineage_assurance": release.get("lineage_assurance") or "legacy_unspecified",
        "corpus_diversity_degraded": bool(
            corpus_quality.get("corpus_diversity_degraded")
        ),
        "release_blockers": list(release.get("release_blockers") or []),
        "numeric_integrity": numeric_integrity,
    }


def validate_data_release(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    manifest = load_json(manifest_path)
    schema = str(manifest.get("schema") or "")
    if schema not in {"mei-sft-data-release-v2", "mei-sft-data-release-v3"}:
        raise RuntimeError("SFT input is not a frozen v2/v3 data release")
    if manifest.get("product") != PRODUCT_ID:
        raise RuntimeError("SFT release product identity mismatch")
    if schema == "mei-sft-data-release-v3" and manifest.get("status") != "frozen":
        raise RuntimeError("SFT-v3 input is not frozen")
    outputs = manifest.get("outputs") or {}
    if schema == "mei-sft-data-release-v3":
        required = {
            "retrieval.train.jsonl",
            "retrieval.valid.jsonl",
            "full-call.train.jsonl",
            "full-call.valid.jsonl",
            "agent-continuation.train.jsonl",
            "agent-continuation.valid.jsonl",
            "mw-disposition.train.jsonl",
            "mw-disposition.valid.jsonl",
            "confidence-harvest.train.jsonl",
            "confidence-harvest.valid.jsonl",
            "coverage-receipt.json",
            "isolation-receipt.json",
            "training-schedule.json",
            "host-simulator-v1.json",
            "agent-isolation-receipt.json",
            "mw-disposition-codebook-v1.json",
            "mw-reason-definitions-v2-20class.json",
            "mw-deviation-audit-receipt.json",
        }
        if not required.issubset(outputs):
            raise RuntimeError(
                "frozen SFT-v3 release output inventory is incomplete: "
                + ", ".join(sorted(required - set(outputs)))
            )
    else:
        required = {
            "retrieval.train.jsonl",
            "retrieval.valid.jsonl",
            "full-call.train.jsonl",
            "full-call.valid.jsonl",
            "mw-disposition.train.jsonl",
            "mw-disposition.valid.jsonl",
            "confidence-calibration.train.jsonl",
            "confidence-calibration.valid.jsonl",
        }
        if not required.issubset(outputs):
            raise RuntimeError("frozen SFT-v2 release output inventory is incomplete")
    checked: dict[str, Any] = {}
    for name in sorted(outputs):
        file_path = path / name
        spec = outputs[name]
        if not file_path.is_file():
            raise RuntimeError(f"frozen SFT release artifact missing: {name}")
        digest = sha_file(file_path)
        if digest != spec.get("sha256"):
            raise RuntimeError(f"frozen SFT release hash mismatch: {name}")
        checked[name] = {"sha256": digest}
        if "rows" in spec:
            rows = len(load_jsonl(file_path))
            if rows != int(spec["rows"]):
                raise RuntimeError(f"frozen SFT release row count mismatch: {name}")
            checked[name]["rows"] = rows
    mw = manifest.get("mw_disposition") or {}
    if int(mw.get("n_classes") or 0) != 20 or mw.get("class_ids") != list(range(20)):
        raise RuntimeError("MW disposition release must freeze class ids 0..19")
    deviation = manifest.get("mw_deviation") or {}
    if deviation.get("kind") != "governance_gate_not_head":
        raise RuntimeError("MW deviation semantic boundary is not frozen")
    deviation_path = path / str(deviation.get("receipt") or "")
    if sha_file(deviation_path) != deviation.get("receipt_sha256"):
        raise RuntimeError("MW deviation governance receipt hash mismatch")
    agent_summary = None
    if schema == "mei-sft-data-release-v3":
        for receipt_key in ("coverage_receipt", "isolation_receipt"):
            receipt_spec = manifest.get(receipt_key) or {}
            receipt_path = path / str(receipt_spec.get("file") or "")
            if sha_file(receipt_path) != receipt_spec.get("sha256"):
                raise RuntimeError(f"SFT-v3 {receipt_key} hash mismatch")
            receipt = load_json(receipt_path)
            if receipt.get("status") != "passed":
                raise RuntimeError(f"SFT-v3 {receipt_key} did not pass")
        compatibility = manifest.get("base_compatibility") or {}
        if (
            int(compatibility.get("deployed_parameter_count") or 0) != EXPECTED_PARAMS
            or compatibility.get("tokenizer_id") != "zh-24k-v1"
            or compatibility.get("arbitrary_cumulative_exposure_supported") is not True
            or compatibility.get("exposure_allowlist") is not None
        ):
            raise RuntimeError("SFT-v3 Base compatibility contract drifted")
        confidence = manifest.get("confidence_label_contract") or {}
        if (
            confidence.get("source") != "actual final runtime after fullcall and R1"
            or int(confidence.get("minimum_positive") or 0) < 100
            or int(confidence.get("minimum_negative") or 0) < 100
            or confidence.get("single_class") != "hard_fail"
        ):
            raise RuntimeError("SFT-v3 confidence harvest contract drifted")
    if "agent_continuation" in manifest:
        agent = manifest.get("agent_continuation") or {}
        if (
            agent.get("schema") != "mei-agent-continuation-sft-v1"
            or agent.get("simulator_id") != "mei-agent-host-simulator-v1"
            or set(agent.get("target_kinds") or []) != {"call", "respond"}
        ):
            raise RuntimeError("Agent continuation release contract drifted")
        if schema == "mei-sft-data-release-v3":
            expected_agent_files = {
                "agent-continuation.train.jsonl",
                "agent-continuation.valid.jsonl",
            }
            checked_agent = {name: checked[name] for name in expected_agent_files}
            if checked["host-simulator-v1.json"]["sha256"] != agent.get(
                "simulator_sha256"
            ) or checked["agent-isolation-receipt.json"]["sha256"] != agent.get(
                "isolation_receipt_sha256"
            ):
                raise RuntimeError(
                    "Agent continuation v3 support artifact hash mismatch"
                )
        else:
            agent_files = agent.get("files") or {}
            expected_agent_files = {
                "agent-continuation.train.jsonl",
                "agent-continuation.valid.jsonl",
                "agent-continuation.eval.jsonl",
            }
            if set(agent_files) != expected_agent_files:
                raise RuntimeError("Agent continuation file inventory is incomplete")
            checked_agent = {}
            for name in sorted(expected_agent_files):
                candidate = path / name
                actual_sha = sha_file(candidate)
                actual_rows = len(load_jsonl(candidate))
                expected = agent_files[name]
                if actual_sha != expected.get("sha256") or actual_rows != int(
                    expected.get("rows") or -1
                ):
                    raise RuntimeError(f"Agent continuation artifact drifted: {name}")
                checked_agent[name] = {"sha256": actual_sha, "rows": actual_rows}
            for file_key, hash_key in (
                ("simulator_file", "simulator_sha256"),
                ("isolation_receipt", "isolation_receipt_sha256"),
            ):
                candidate = path / str(agent.get(file_key) or "")
                if sha_file(candidate) != agent.get(hash_key):
                    raise RuntimeError(f"Agent continuation {file_key} hash mismatch")
        if int(agent.get("nonempty_tool_result_rows") or 0) <= 0:
            raise RuntimeError(
                "Agent continuation release has no nonempty ToolResult rows"
            )
        agent_summary = {
            "schema": agent["schema"],
            "simulator_id": agent["simulator_id"],
            "files": checked_agent,
            "target_kinds": ["call", "respond"],
            "nonempty_tool_result_rows": int(agent["nonempty_tool_result_rows"]),
            "group_aware": agent.get("group_aware") is True,
            "family_aware": agent.get("family_aware") is True,
        }
    return {
        "schema": schema,
        "release_id": manifest.get("release_id"),
        "path": str(path.resolve()),
        "manifest_sha256": sha_file(manifest_path),
        "files": checked,
        "mw_disposition": {
            "n_classes": 20,
            "codebook_sha256": mw.get("codebook_sha256"),
            "trainable_sidecar": True,
        },
        "mw_deviation": {
            "receipt_sha256": deviation.get("receipt_sha256"),
            "trainable": False,
            "tensor_count": 0,
            "package_capability": False,
        },
        "isolation": manifest.get("isolation"),
        "agent_continuation": agent_summary,
        "narration_source": manifest.get("narration_source"),
        "confidence_label_contract": manifest.get("confidence_label_contract"),
        "training_schedule": manifest.get("training_schedule"),
    }


def validate_narration_release(
    path: Path, data_release: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema") != "mei-narration-sft-data-release-v2"
        or manifest.get("product") != PRODUCT_ID
        or manifest.get("provider_id") != "mei-zh-narration-adapter-r16-v2"
        or (manifest.get("adapter") or {}).get("kind")
        != "frozen-backbone-logit-residual"
        or (manifest.get("adapter") or {}).get("rank") != 16
        or (manifest.get("adapter") or {}).get("parameter_count") != 392_192
    ):
        raise RuntimeError("narration data release contract drifted")
    narration_source = data_release.get("narration_source") or {}
    directly_bound = ((manifest.get("parents") or {}).get("agent") or {}).get(
        "release_id"
    ) == data_release.get("release_id") and (
        (manifest.get("parents") or {}).get("agent") or {}
    ).get("manifest_sha256") == data_release.get("manifest_sha256")
    explicitly_referenced = (
        narration_source.get("release_id") == manifest.get("release_id")
        and narration_source.get("manifest_sha256") == sha_file(manifest_path)
        and narration_source.get("retrain_on_final_lm") is True
    )
    if not (directly_bound or explicitly_referenced):
        raise RuntimeError(
            "narration release is neither directly bound to nor explicitly "
            "referenced by the Agent SFT release"
        )
    expected_names = {
        "narration.train.jsonl",
        "narration.valid.jsonl",
        "narration.eval.jsonl",
    }
    files = manifest.get("files") or {}
    if set(files) != expected_names:
        raise RuntimeError("narration release file inventory is incomplete")
    checked = {}
    for name in sorted(expected_names):
        candidate = path / name
        rows = load_jsonl(candidate)
        spec = files[name]
        if sha_file(candidate) != spec.get("sha256") or len(rows) != int(
            spec.get("rows") or -1
        ):
            raise RuntimeError(f"narration release artifact drifted: {name}")
        if any(
            row.get("can_execute_tools") is not False
            or row.get("grounding_target") != "bounded-grounded-chinese-v2"
            or not row.get("verified_result_views")
            for row in rows
        ):
            raise RuntimeError(f"narration release contains an unsafe row: {name}")
        checked[name] = {"rows": len(rows), "sha256": spec["sha256"]}
    isolation = manifest.get("isolation_receipt") or {}
    receipt_path = path / str(isolation.get("file") or "")
    if sha_file(receipt_path) != isolation.get("sha256"):
        raise RuntimeError("narration isolation receipt hash mismatch")
    receipt = load_json(receipt_path)
    if (
        receipt.get("schema") != "mei-narration-data-isolation-receipt-v2"
        or receipt.get("status") != "passed"
        or set(receipt.get("status_values") or ()) != {"ok", "error", "cancelled"}
        or receipt.get("group_cross_split") != 0
        or receipt.get("prompt_overlap") != 0
        or receipt.get("terminal_only") is not True
        or receipt.get("executor_access") is not False
    ):
        raise RuntimeError("narration v2 isolation/coverage receipt is incomplete")
    return {
        "release_id": manifest.get("release_id"),
        "path": str(path.resolve()),
        "manifest_sha256": sha_file(manifest_path),
        "provider_id": manifest["provider_id"],
        "prompt_id": manifest.get("prompt_id"),
        "adapter": manifest["adapter"],
        "coverage": manifest.get("coverage"),
        "files": checked,
        "isolation_receipt_sha256": isolation["sha256"],
    }


def validate_eval_lock(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    lock_path = path / "lock.json"
    lock = load_json(lock_path)
    if lock.get("id") != "sft-v2-eval-lock-v3-20class":
        raise RuntimeError("wrong eval lock")
    if int(lock.get("mw_n_classes") or 0) != 20:
        raise RuntimeError("eval lock does not cover all 20 MW disposition classes")
    if lock.get("mw_class_ids") != list(range(20)):
        raise RuntimeError("eval lock MW class ids drifted")
    if lock.get("mw_codebook_sha256") != data["mw_disposition"]["codebook_sha256"]:
        raise RuntimeError("training/eval MW disposition codebooks disagree")
    bank_files = {
        "retrieval_dev": "eval-retrieval-dev.jsonl",
        "retrieval_test": "eval-retrieval-test.jsonl",
        "fullcall_dev": "eval-fullcall-dev.jsonl",
        "fullcall_test": "eval-fullcall-test.jsonl",
        "mw_dev": "eval-mw-dev.jsonl",
        "mw_test": "eval-mw-test.jsonl",
    }
    files: dict[str, str] = {}
    for bank, name in bank_files.items():
        candidate = path / name
        actual_sha = sha_file(candidate)
        with candidate.open(encoding="utf-8") as handle:
            actual_n = sum(1 for line in handle if line.strip())
        expected = (lock.get("banks") or {}).get(bank) or {}
        if actual_sha != expected.get("sha256") or actual_n != expected.get("n"):
            raise RuntimeError(f"eval lock bank drifted: {bank}")
        files[name] = actual_sha
    universe = path / "tool-universe-v1.json"
    files[universe.name] = sha_file(universe)
    if files[universe.name] != lock.get("universe_sha256"):
        raise RuntimeError("eval lock tool universe drifted")
    parent_lock = ROOT / "cycles/mei-1.1-51m/_legacy/notebook/evaluation/banks/sft-v2-eval-lock-v2/lock.json"
    if (
        lock.get("parent_lock") != "sft-v2-eval-lock-v2"
        or lock.get("v2_preserved") is not True
        or sha_file(parent_lock) != lock.get("parent_lock_sha256")
    ):
        raise RuntimeError("20-class eval lock does not preserve its frozen v2 parent")
    return {
        "id": lock["id"],
        "path": str(path.resolve()),
        "lock_sha256": sha_file(lock_path),
        "files": files,
        "universe_sha256": lock.get("universe_sha256"),
        "parent_lock": lock.get("parent_lock"),
        "parent_lock_sha256": lock.get("parent_lock_sha256"),
        "v2_preserved": True,
    }


def validate_quality_thresholds(
    path: Path, eval_lock: dict[str, Any]
) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("registered_before_eval") is not True:
        raise RuntimeError("quality thresholds were not pre-registered")
    if payload.get("lock") != eval_lock.get("parent_lock"):
        raise RuntimeError("quality thresholds are not bound to the preserved v2 lock")
    layers = payload.get("layers") or {}
    required = {
        "recall_at_5_learned_min",
        "recall_at_5_must_exceed_lexical",
        "oracle_top5_fullcall_exact_min",
        "oracle_top5_execute_refuse_ok_min",
        "learned_top5_e2e_exact_min",
        "unsupported_accepted_max",
        "unprovenanced_argument_accepted_max",
    }
    if set(layers) != required:
        raise RuntimeError("quality threshold layer contract drifted")
    if (
        layers["unsupported_accepted_max"] != 0
        or layers["unprovenanced_argument_accepted_max"] != 0
    ):
        raise RuntimeError("safety thresholds must remain absolute zero")
    return {
        "path": str(path.resolve()),
        "sha256": sha_file(path),
        "payload": payload,
        "evaluated_on": eval_lock["id"],
        "threshold_parent_lock_preserved": True,
    }


def source_manifest() -> dict[str, str]:
    relative = (
        "src/model-factory/orchestration/productize_51m.py",
        "src/model-factory/training/qat/qat_cq2_v2_51m.py",
        "src/model-factory/training/qat/cq2_qat_51m.py",
        "src/model-factory/training/qat/cq2_policy_51m.py",
        "src/model-factory/release/quant_pack_51m.py",
        "src/model-factory/training/qat/quant_ops_51m.py",
        "src/model-factory/training/tool_use/train_sft_ondisk_51m.py",
        "src/model-factory/release/freeze_agent_sft_release_51m.py",
        "src/model-factory/release/freeze_narration_sft_release_51m.py",
        "src/model-factory/release/freeze_narration_sft_release_v2_51m.py",
        "src/model-factory/release/pack_cq2_v2_51m.py",
        "src/model-factory/training/heads/mtp_ablation_51m.py",
        "src/model-factory/evaluation/resources/measure_resources_51m.py",
        "src/model-factory/orchestration/run_downstream_resource_gates_51m.py",
        "src/model-factory/tests/test_measure_resources_51m.py",
        "src/model-factory/evaluation/alignment/needle2_alignment_51m.py",
        "src/model-factory/release/final_audit_51m.py",
        "src/model-factory/evaluation/heads/compare_portable_heads_51m.py",
        "src/platform/_shared/rust/mei-sdk-core/examples/resource_51m.rs",
        "src/architecture/mei-1.2-51m/architecture.py",
        "src/architecture/mei-1.2-51m/architecture_contract.py",
        "src/architecture/mei-1.2-51m/cq2_metal.py",
        "src/architecture/mei-1.2-51m/config.py",
        "src/architecture/mei-1.2-51m/fused_ops.py",
        "src/architecture/mei-1.2-51m/heads.py",
        "src/model-factory/common/checkpoint.py",
        "src/model-factory/evaluation/base/freeze_float_baseline_51m.py",
        "src/model-factory/common/mlx_memory_policy_51m.py",
        "src/platform/browser-sdk/smoke_wasm_q4.mjs",
        "src/platform/browser-sdk/smoke_node_v2.mjs",
        "src/platform/browser-sdk/browser.mjs",
        "src/platform/browser-sdk/cq2.mjs",
        "src/platform/browser-sdk/engine.mjs",
        "src/platform/browser-sdk/index.mjs",
        "src/platform/browser-sdk/node-wasm.mjs",
        "src/platform/browser-sdk/package.mjs",
        "src/platform/browser-sdk/protocol.mjs",
        "src/platform/browser-sdk/tool-index.mjs",
        "src/platform/browser-sdk/wasm-abi.mjs",
        "src/platform/_shared/.cargo/config.toml",
        "src/platform/_shared/tools/run_gates.py",
        "src/platform/_shared/runtime/provenance.py",
        "src/platform/_shared/runtime/byte_grammar.py",
        "src/platform/_shared/runtime/schema_subset.py",
        "src/platform/_shared/runtime/tool_index.py",
        "src/platform/_shared/runtime/kv_manager.py",
        "src/platform/_shared/runtime/narration.py",
        "src/platform/python-sdk/mei_sdk/cq2.py",
        "src/platform/python-sdk/mei_sdk/engine.py",
        "src/platform/python-sdk/mei_sdk/package.py",
        "src/platform/python-sdk/mei_sdk/protocol.py",
        "src/platform/python-sdk/mei_sdk/runtime_51m.py",
        "src/platform/_shared/rust/mei-sdk-core/src/byte_grammar.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/cq2.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/engine.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/infer.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/lib.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/model.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/package.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/packed.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/protocol.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/retrieval.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/tool_index.rs",
        "src/platform/_shared/rust/mei-sdk-core/src/vocab.rs",
        "src/platform/_shared/rust/mei-sdk-core/tests/tokenizer_v2.rs",
        "src/platform/_shared/rust/mei-sdk-core/tests/v2_contract.rs",
        "src/platform/_shared/rust/mei-sdk-cli/src/main.rs",
        "src/platform/_shared/rust/mei-sdk-ffi/src/lib.rs",
        "src/platform/_shared/rust/mei-sdk-wasm/src/lib.rs",
        "src/platform/_shared/rust/mei-sdk-core/Cargo.toml",
        "src/platform/_shared/Cargo.toml",
        "src/platform/_shared/Cargo.lock",
        "src/platform/_shared/tools/dump_goldens.py",
        "src/platform/_shared/tools/dump_tokenizer_golden.py",
        "src/platform/_shared/spec/golden/canonical_numbers_v2.json",
        "src/platform/_shared/spec/golden/catalog_fingerprint_v2.json",
        "src/platform/_shared/spec/golden/cq2_v2.json",
        "src/platform/_shared/spec/golden/format_cases_v2.json",
        "src/platform/_shared/spec/golden/package_capabilities.json",
        "src/platform/_shared/spec/golden/parse_cases.json",
        "src/platform/_shared/spec/golden/render_request.json",
        "src/platform/_shared/spec/golden/schema_fingerprint.json",
        "src/platform/_shared/spec/golden/tokenizer_v2.json",
        "src/platform/_shared/spec/golden/tool_index_v2.json",
        "src/platform/_shared/spec/golden/turn_results.json",
        "src/platform/_shared/spec/request-v2.schema.json",
        "src/platform/_shared/spec/tool-result-v2.schema.json",
        "src/platform/_shared/spec/turn-result.schema.json",
        "src/platform/_shared/spec/narration-result-v2.schema.json",
        "src/platform/_shared/spec/model-package-v2.schema.json",
        "src/platform/_shared/spec/resource-measurement-receipt-v1.schema.json",
        "src/platform/_shared/spec/needle2-reference.json",
    )
    return {name: sha_file(ROOT / name) for name in relative}


def environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("mlx", "numpy", "sentencepiece", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    dependency_files = {
        name: sha_file(ROOT / name)
        for name in ("requirements.txt", "src/platform/_shared/Cargo.lock", "src/platform/_shared/Cargo.toml")
    }
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "dependency_files": dependency_files,
        "offline": True,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    allow_live_cpt = bool(getattr(args, "allow_live_cpt", False))
    active_cpt = live_cpt_workers()
    base = validate_base(args.base_release, args.base_weights)
    data = validate_data_release(args.data_release)
    narration_data = validate_narration_release(args.narration_release, data)
    eval_lock = validate_eval_lock(args.eval_lock, data)
    quality_thresholds = validate_quality_thresholds(args.quality_thresholds, eval_lock)
    deployment_catalog = _catalog(args.eval_lock)
    contracts_raw = architecture_contracts()
    contracts = {
        key: str(contracts_raw[key])
        for key in (
            "weight_contract_sha256",
            "runtime_profile_sha256",
            "training_aux_sha256",
        )
    }
    immutable = {
        "product": PRODUCT_ID,
        "package_id": args.package_id,
        "base": base,
        "contracts": contracts,
        "data_release": data,
        "narration_data_release": narration_data,
        "eval_lock": eval_lock,
        "quality_thresholds": quality_thresholds,
        "deployment_tool_catalog": {
            "projection_id": "mei-tool-catalog-portable-projection-v1",
            "source_universe_sha256": eval_lock["files"]["tool-universe-v1.json"],
            "tool_count": len(deployment_catalog),
            "sha256": sha_bytes(canonical_bytes(deployment_catalog)),
            "metadata_removed": [
                "family",
                "fingerprint",
                "similar_to",
                "tool_id",
                "source_toolset",
            ],
            "semantic_rewrites": {r"^\d{2}:\d{2}$": r"^[0-9]{2}:[0-9]{2}$"},
        },
        "current_baseline_sha256": sha_file(CURRENT_PATH),
        "tokenizer_sha256": sha_file(TOKENIZER_ZH_V1),
        "replay_corpus": {},
        "source_manifest": source_manifest(),
        "environment_manifest": environment_manifest(),
        "recipe": {
            "qat_tokens": args.qat_tokens,
            "qat_seq_len": args.qat_seq_len,
            "qat_batch_size": args.qat_batch_size,
            "qat_grad_accum": args.qat_grad_accum,
            "float_control_steps": args.float_control_steps,
            "retrieval_r0_steps": args.retrieval_r0_steps,
            "fullcall_steps": args.fullcall_steps,
            "retrieval_r1_steps": args.retrieval_r1_steps,
            "mw_steps": args.mw_steps,
            "confidence_steps": args.confidence_steps,
            "narration_steps": args.narration_steps,
            "mtp_steps": args.mtp_steps,
            "eval_limit": args.eval_limit,
            "seed": 51,
        },
        "execution_policy": {
            "allow_live_cpt": allow_live_cpt,
            "qat_only": bool(getattr(args, "qat_only", False)),
            "live_cpt_parallel_at_plan_time": bool(active_cpt) and allow_live_cpt,
            "resource_isolation": "separate-run-directories-no-cpt-signals",
            "mlx_memory_policy_id": PRODUCTIZATION_MLX_POLICY_ID,
            "mlx_global_memory_limit_bytes": PRODUCTIZATION_MLX_MEMORY_LIMIT_BYTES,
            "mlx_global_cache_limit_bytes": PRODUCTIZATION_MLX_CACHE_LIMIT_BYTES,
            "locked_eval_clear_cache_per_generation": True,
        },
        "semantic_boundaries": {
            "retrieval": "independent-contrastive-head",
            "mw_disposition": "independent-20class-sidecar",
            "confidence": "independent-calibrated-binary-head",
            "narration_adapter": "independent-frozen-backbone-rank16-generation-sidecar",
            "mw_deviation": "deterministic-governance-gate-no-tensors",
        },
        "quantization": {
            "primary": "mei-cq-v2-g128-wht-codebook",
            "q4": "diagnostic-only",
            "activation_kv": "int8-deployment-contract",
        },
    }
    corpus_files = {
        name: sha_file(args.replay_corpus / name)
        for name in (
            "RELEASE.json",
            "manifest.json",
            "hashes.json",
            "mix.json",
            "schedule-scratch.json",
        )
    }
    immutable["replay_corpus"] = {
        "path": str(args.replay_corpus.resolve()),
        "files": corpus_files,
        "merkle_root_sha256": sha_bytes(canonical_bytes(corpus_files)),
    }
    run_fingerprint = sha_bytes(canonical_bytes(immutable))
    stage_rows = []
    previous = base["weights_sha256"]
    for index, name in enumerate(STAGES, start=1):
        stage_immutable = {
            "run_fingerprint_sha256": run_fingerprint,
            "stage": name,
            "index": index,
            "predecessor_fingerprint_sha256": previous,
        }
        fingerprint = sha_bytes(canonical_bytes(stage_immutable))
        stage_rows.append(
            {
                "index": index,
                "stage_id": name,
                "stage_fingerprint_sha256": fingerprint,
                "predecessor_fingerprint_sha256": previous,
            }
        )
        previous = fingerprint
    return {
        "schema": "mei-51m-productization-plan-v2",
        "run_fingerprint_sha256": run_fingerprint,
        "immutable": immutable,
        "stages": stage_rows,
        "live_cpt_workers": active_cpt,
        "heavy_execution_deferred": bool(active_cpt) and not allow_live_cpt,
        "live_cpt_parallel_authorized": bool(active_cpt) and allow_live_cpt,
        "process_complete": False,
        "release_eligible": False,
    }


def stage_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["stage_id"]): row for row in plan["stages"]}


def _output_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha_file(path) for path in paths}


def _receipt_reusable(
    receipt: dict[str, Any],
    expected: dict[str, Any],
    input_fingerprint: str | None = None,
) -> bool:
    if (
        receipt.get("stage_fingerprint_sha256") != expected["stage_fingerprint_sha256"]
        or receipt.get("terminal_status") not in {"passed", "degraded"}
        or (
            input_fingerprint is not None
            and receipt.get("stage_input_fingerprint_sha256") != input_fingerprint
        )
    ):
        return False
    outputs = receipt.get("output_hashes") or {}
    return bool(outputs) and all(
        resolve_repo_path(path).is_file()
        and sha_file(resolve_repo_path(path)) == digest
        for path, digest in outputs.items()
    )


def _stage_input_fingerprint(
    run_dir: Path, plan: dict[str, Any], name: str
) -> tuple[str, dict[str, Any]]:
    row = stage_map(plan)[name]
    index = int(row["index"])
    if index == 1:
        evidence = {
            "base_release_sha256": plan["immutable"]["base"]["release_sha256"],
            "base_weights_sha256": plan["immutable"]["base"]["weights_sha256"],
        }
    else:
        predecessor = str(plan["stages"][index - 2]["stage_id"])
        receipt_path = run_dir / "stages" / predecessor / "receipt.json"
        receipt = load_json(receipt_path)
        if receipt.get("terminal_status") not in {"passed", "degraded"}:
            raise RuntimeError(f"predecessor stage is not reusable: {predecessor}")
        outputs = receipt.get("output_hashes") or {}
        if not outputs or any(
            not resolve_repo_path(path).is_file()
            or sha_file(resolve_repo_path(path)) != digest
            for path, digest in outputs.items()
        ):
            raise RuntimeError(f"predecessor output hash drift: {predecessor}")
        evidence = {
            "predecessor_stage": predecessor,
            "predecessor_receipt_sha256": sha_file(receipt_path),
            "predecessor_output_hashes": outputs,
        }
    fingerprint = sha_bytes(
        canonical_bytes(
            {
                "stage_plan_fingerprint_sha256": row["stage_fingerprint_sha256"],
                "input_artifacts": evidence,
            }
        )
    )
    return fingerprint, evidence


def run_stage(
    *,
    run_dir: Path,
    plan: dict[str, Any],
    name: str,
    action: Callable[[Path], tuple[dict[str, Any], list[Path]]],
    resume: bool,
) -> dict[str, Any]:
    expected = stage_map(plan)[name]
    immutable = plan["immutable"]
    if sha_file(CURRENT_PATH) != immutable["current_baseline_sha256"]:
        raise RuntimeError("CURRENT.json drifted after productization planning")
    base_weights = resolve_repo_path(immutable["base"]["weights"])
    if sha_file(base_weights) != immutable["base"]["weights_sha256"]:
        raise RuntimeError("selected frozen base changed after productization planning")
    directory = run_dir / "stages" / name
    receipt_path = directory / "receipt.json"
    input_fingerprint, input_evidence = _stage_input_fingerprint(run_dir, plan, name)
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        if _receipt_reusable(receipt, expected, input_fingerprint):
            return {**receipt, "reused": True}
        raise RuntimeError(
            f"stage {name} receipt is not safely reusable; use a new run ID"
        )
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise RuntimeError(
            f"stage {name} is partial; resume explicitly or use a new run ID"
        )
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(
        directory / "running.json",
        {
            "stage_id": name,
            "stage_fingerprint_sha256": expected["stage_fingerprint_sha256"],
            "stage_input_fingerprint_sha256": input_fingerprint,
            "terminal_status": "running",
            "started_unix": started,
        },
    )
    try:
        metrics, outputs = action(directory)
        hashes = _output_hashes(outputs)
        status = (
            "degraded"
            if metrics.get("degraded") or metrics.get("terminal_status") == "degraded"
            else "passed"
        )
        receipt = {
            "schema": "mei-productization-stage-receipt-v2",
            "stage_id": name,
            "stage_fingerprint_sha256": expected["stage_fingerprint_sha256"],
            "stage_input_fingerprint_sha256": input_fingerprint,
            "input_artifacts": input_evidence,
            "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
            "terminal_status": status,
            "started_unix": started,
            "finished_unix": time.time(),
            "metrics": metrics,
            "output_hashes": hashes,
        }
        write_json(receipt_path, receipt)
        (directory / "running.json").unlink(missing_ok=True)
        return receipt
    except Exception as exc:
        receipt = {
            "schema": "mei-productization-stage-receipt-v2",
            "stage_id": name,
            "stage_fingerprint_sha256": expected["stage_fingerprint_sha256"],
            "stage_input_fingerprint_sha256": input_fingerprint,
            "input_artifacts": input_evidence,
            "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
            "terminal_status": "blocked",
            "started_unix": started,
            "finished_unix": time.time(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "output_hashes": {},
        }
        write_json(receipt_path, receipt)
        (directory / "running.json").unlink(missing_ok=True)
        raise


def choose_run_dir(requested: Path, plan: dict[str, Any], *, resume: bool) -> Path:
    if not requested.exists() or not any(requested.iterdir()):
        return requested
    plan_path = requested / "plan.json"
    if resume and plan_path.is_file():
        existing = load_json(plan_path)
        if existing.get("run_fingerprint_sha256") == plan["run_fingerprint_sha256"]:
            return requested
    stem = requested.with_name(
        requested.name + "-" + str(plan["run_fingerprint_sha256"])[:12]
    )
    candidate = stem
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = stem.with_name(f"{stem.name}-{counter:03d}")
    return candidate


def write_plan(run_dir: Path, plan: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "plan.json"
    if path.is_file():
        existing = load_json(path)
        if existing.get("run_fingerprint_sha256") != plan["run_fingerprint_sha256"]:
            raise RuntimeError("run plan fingerprint drift; choose a new run ID")
        return
    write_json(path, plan)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"JSONL row is not an object: {path}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _ensure_python_paths() -> None:
    for path in (
        ROOT / "src/architecture/mei-1.2-51m",
        ROOT / "src/model-factory",
        ROOT / "src/platform/python-sdk",
        ROOT / "src/platform/_shared/runtime",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_runtime(
    master: Path,
    *,
    quantized: bool,
    retrieval_head: Path | None = None,
    mw_head: Path | None = None,
    confidence_head: Path | None = None,
    narration_head: Path | None = None,
    tool_index: Path | None = None,
):
    _ensure_python_paths()
    import mlx.core as mx

    from architecture import NeedleZh, count_params
    from common.checkpoint import load_params
    from config import NeedleZhConfig
    from training.qat.cq2_qat_51m import explicit_group_map, quantize_tree
    from heads import (
        ConfidenceV2Head,
        ContrastiveHead,
        MWDispositionHead,
        NarrationAdapterHead,
    )
    from mei_sdk.runtime_51m import Runtime51M
    from mei_sdk.shared import ToolIndex
    from tokenizer import ZhTokenizerV1

    cfg = NeedleZhConfig.from_spec()
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, master, strict=True)
    mx.eval(model.parameters())
    if count_params(model) != EXPECTED_PARAMS:
        raise RuntimeError("runtime LM changed the 51,463,797-parameter identity")
    if quantized:
        group_map = explicit_group_map(model.parameters())
        model.update(quantize_tree(model.parameters(), group_map, ste=False))
        mx.eval(model.parameters())
    contrastive = None
    if retrieval_head is not None:
        contrastive = ContrastiveHead(cfg.d_model, cfg.n_layers, dim=128, probes=4)
        mx.eval(contrastive.parameters())
        load_params(contrastive, retrieval_head, strict=True)
        mx.eval(contrastive.parameters())
    disposition = None
    if mw_head is not None:
        disposition = MWDispositionHead(cfg.d_model)
        mx.eval(disposition.parameters())
        load_params(disposition, mw_head, strict=True)
        mx.eval(disposition.parameters())
    confidence = None
    if confidence_head is not None:
        confidence = ConfidenceV2Head(cfg.d_model)
        mx.eval(confidence.parameters())
        load_params(confidence, confidence_head, strict=True)
        mx.eval(confidence.parameters())
    narration_adapter = None
    if narration_head is not None:
        narration_adapter = NarrationAdapterHead(cfg.d_model, cfg.vocab_size, rank=16)
        mx.eval(narration_adapter.parameters())
        load_params(narration_adapter, narration_head, strict=True)
        mx.eval(narration_adapter.parameters())
    index = ToolIndex.load(tool_index) if tool_index is not None else None
    mw_receipt_path = mw_head.parent / "receipt.json" if mw_head is not None else None
    mw_receipt_sha256 = (
        sha_file(mw_receipt_path)
        if mw_receipt_path is not None and mw_receipt_path.is_file()
        else ""
    )
    runtime = Runtime51M(
        model,
        ZhTokenizerV1(),
        contrastive=contrastive,
        mw_disposition=disposition,
        conf_v2=confidence,
        index=index,
        model_hash=sha_file(master),
        head_hash=sha_file(retrieval_head) if retrieval_head else "",
        tokenizer_hash=sha_file(TOKENIZER_ZH_V1),
        mw_receipt_sha256=mw_receipt_sha256,
        release_class="experimental",
    )
    if disposition is not None:
        runtime.mw_disposition_head = disposition
    if narration_adapter is not None:
        runtime.narration_adapter = narration_adapter
    return runtime


def _save_model_or_head(value: Any, path: Path) -> None:
    _ensure_python_paths()
    from common.checkpoint import save_params

    path.parent.mkdir(parents=True, exist_ok=True)
    save_params(value, path)


def _catalog(eval_lock: Path) -> list[dict[str, Any]]:
    rows = load_json(eval_lock / "tool-universe-v1.json").get("tools") or []
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("frozen tool universe is empty")
    deployment_keys = (
        "name",
        "description",
        "parameters",
        "required_permissions",
        "required_state",
        "x-mei-permissions",
        "x-mei-state",
    )
    # The frozen universe also carries split/retrieval metadata such as
    # family, tool_id and source_toolset. It is evidence, never part of the
    # public tool schema consumed by the strict runtime registry.
    projected = [
        {key: row[key] for key in deployment_keys if key in row}
        for row in rows
        if isinstance(row, dict)
    ]
    for tool in projected:
        properties = (tool.get("parameters") or {}).get("properties") or {}
        for schema in properties.values():
            if not isinstance(schema, dict) or "pattern" not in schema:
                continue
            pattern = schema["pattern"]
            if pattern == r"^\d{2}:\d{2}$":
                schema["pattern"] = r"^[0-9]{2}:[0-9]{2}$"
            elif isinstance(pattern, str) and r"\d" in pattern:
                raise RuntimeError(f"unregistered portable pattern rewrite: {pattern}")
    _ensure_python_paths()
    from schema_subset import validate_tools

    validate_tools(projected)
    return projected


def _lexical_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())
        if token
    }


def _lexical_top5(query: str, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_tokens = _lexical_tokens(query)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for tool in catalog:
        name = str(tool.get("name") or "")
        body = name + " " + str(tool.get("description") or "")
        tool_tokens = _lexical_tokens(body)
        union = query_tokens | tool_tokens
        score = len(query_tokens & tool_tokens) / len(union) if union else 0.0
        scored.append((score, name, tool))
    scored.sort(key=lambda row: (-row[0], row[1].encode("utf-8")))
    return [row[2] for row in scored[:5]]


def _project_oracle_tools(
    row: dict[str, Any], catalog: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve frozen oracle names through the portable deployment catalogue.

    Evaluation rows retain their original training-time tool objects as frozen
    evidence. Those objects must never bypass the deployment projection used
    by SFT, indexing, and every public runtime. Resolve by stable tool name and
    fail closed if the locked row references a tool outside that projection.
    """

    universe = {str(tool.get("name") or ""): tool for tool in catalog}
    oracle_rows = row.get("oracle_top5") or []
    if not isinstance(oracle_rows, list):
        raise RuntimeError("oracle_top5 must be a list")
    names = [
        str(tool.get("name") or "") for tool in oracle_rows if isinstance(tool, dict)
    ]
    if len(names) != len(oracle_rows) or any(not name for name in names):
        raise RuntimeError("oracle_top5 contains an invalid tool object")
    missing = [name for name in names if name not in universe]
    if missing:
        raise RuntimeError(
            "oracle_top5 references tools outside deployment projection: "
            + ", ".join(sorted(set(missing)))
        )
    return [universe[name] for name in names]


def _configure_locked_eval_mlx(mx: Any) -> dict[str, Any]:
    """Bound MLX/Metal allocations before repeated locked-eval generation."""

    _mx, policy = configure_mlx_memory(mx, reset_peak=True)
    return {
        **policy,
        "clear_cache_per_generation": True,
        "progress_interval_rows": LOCKED_EVAL_PROGRESS_INTERVAL,
    }


def _release_locked_eval_mlx(mx: Any, *, collect_python: bool = False) -> None:
    """Return completed generation buffers to Metal instead of caching them."""

    release_mlx_memory(mx, collect_python=collect_python)


def _locked_eval_memory_snapshot(mx: Any) -> dict[str, int]:
    return mlx_memory_snapshot(mx)


def _locked_quality_eval(
    runtime: Any,
    *,
    eval_lock: Path,
    catalog: list[dict[str, Any]],
    thresholds_path: Path,
    limit: int,
) -> dict[str, Any]:
    """Run the frozen layered quality/safety evaluation on the final LM/R1.

    This is a score receipt, not a mechanism claim.  Missing a floor degrades
    release eligibility but does not stop later independent product stages.
    """

    from training.tool_use.train_sft_ondisk_51m import _calls_equal, deployment_request, encode_fc
    from mei_sdk.protocol import normalize_request
    from mei_sdk.shared import validate_generated_call
    import mlx.core as mx

    memory_policy = _configure_locked_eval_mlx(mx)

    threshold_doc = load_json(thresholds_path)
    if threshold_doc.get("registered_before_eval") is not True:
        raise RuntimeError("quality thresholds were not pre-registered")
    layers = threshold_doc.get("layers") or {}
    expected_thresholds = {
        "recall_at_5_learned_min",
        "recall_at_5_must_exceed_lexical",
        "oracle_top5_fullcall_exact_min",
        "oracle_top5_execute_refuse_ok_min",
        "learned_top5_e2e_exact_min",
        "unsupported_accepted_max",
        "unprovenanced_argument_accepted_max",
    }
    if set(layers) != expected_thresholds:
        raise RuntimeError("quality threshold inventory drifted")

    universe = {str(tool.get("name") or ""): tool for tool in catalog}

    def catalog_of(names: list[Any]) -> list[dict[str, Any]]:
        return [universe[str(name)] for name in names if str(name) in universe]

    def learned_top(query: str, visible: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(visible) <= 5:
            return list(visible)
        query_vector = runtime.embed_text(query)
        # The final R1 index is built over the complete frozen universe.  Use
        # it directly so per-row catalogue filtering does not rebuild or
        # mutate the persisted index.
        return runtime.index.select_tools(visible, query_vector, k=5)

    retrieval_rows = load_jsonl(eval_lock / "eval-retrieval-test.jsonl")
    retrieval_rows = retrieval_rows[: min(max(1, int(limit)), 1000)]
    learned_hits = lexical_hits = 0
    for row in retrieval_rows:
        visible = catalog_of(list(row.get("catalog_tool_names") or []))
        gold = str(row.get("gold_tool") or "")
        learned_names = {
            str(tool.get("name") or "")
            for tool in learned_top(str(row.get("query") or ""), visible)
        }
        lexical_names = {
            str(tool.get("name") or "")
            for tool in _lexical_top5(str(row.get("query") or ""), visible)
        }
        learned_hits += int(gold in learned_names)
        lexical_hits += int(gold in lexical_names)

    fullcall_rows = load_jsonl(eval_lock / "eval-fullcall-test.jsonl")
    fullcall_rows = fullcall_rows[: min(max(1, int(limit)), 1200)]
    oracle_exact = oracle_execute_refuse = learned_exact = 0
    unsupported_accepted = unprovenanced_accepted = 0

    def generate_once(
        row: dict[str, Any], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        encoded = encode_fc(runtime.tokenizer, row, selected_tools=tools)
        if encoded is None:
            return {"function_calls": [], "refuse": True, "execution": "refuse"}
        prompt, _answer = encoded
        decoded = runtime.greedy(
            prompt,
            tools=tools,
            max_new=96,
            decode_mode="constrained",
        )
        request = normalize_request(
            {
                **deployment_request(row),
                "decode_mode": "constrained",
                "max_new": 96,
            }
        )
        return validate_generated_call(
            str(decoded.get("text") or ""),
            tools=tools,
            request=request,
            confidence=None,
            enforce_confidence=False,
        )

    def generate(row: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            return generate_once(row, tools)
        finally:
            # generate_once has returned, so its temporary KV/logit arrays are
            # no longer live and can be returned to Metal immediately.
            _release_locked_eval_mlx(mx)

    for row_index, row in enumerate(fullcall_rows, start=1):
        gold = list(row.get("answers") or [])
        want_execute = str(row.get("kind") or "execute") == "execute"
        oracle_tools = _project_oracle_tools(row, catalog)
        oracle = generate(row, oracle_tools)
        oracle_calls = list(oracle.get("function_calls") or [])
        oracle_hit = (
            _calls_equal(oracle_calls, gold)
            if want_execute
            else (oracle.get("refuse") is True or not oracle_calls)
        )
        oracle_exact += int(oracle_hit)
        got_execute = oracle.get("execution") == "execute"
        oracle_execute_refuse += int(
            (want_execute and got_execute)
            or (not want_execute and (oracle.get("refuse") is True or not oracle_calls))
        )
        unsupported_accepted += int(oracle.get("unsupported_accepted") or 0)
        unprovenanced_accepted += int(
            oracle.get("unprovenanced_argument_accepted") or 0
        )

        visible = catalog_of(list(row.get("catalog_tool_names") or []))
        learned_tools = learned_top(str(row.get("query") or ""), visible)
        learned = generate(row, learned_tools)
        learned_calls = list(learned.get("function_calls") or [])
        learned_hit = (
            _calls_equal(learned_calls, gold)
            if want_execute
            else (learned.get("refuse") is True or not learned_calls)
        )
        learned_exact += int(learned_hit)
        unsupported_accepted += int(learned.get("unsupported_accepted") or 0)
        unprovenanced_accepted += int(
            learned.get("unprovenanced_argument_accepted") or 0
        )
        if row_index % LOCKED_EVAL_PROGRESS_INTERVAL == 0 or row_index == len(
            fullcall_rows
        ):
            _release_locked_eval_mlx(mx, collect_python=True)
            print(
                json.dumps(
                    {
                        "stage": "learned_top5_e2e",
                        "fullcall_rows_complete": row_index,
                        "fullcall_rows_total": len(fullcall_rows),
                        "generations_complete": row_index * 2,
                        "mlx_memory": _locked_eval_memory_snapshot(mx),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    n_retrieval = max(len(retrieval_rows), 1)
    n_fullcall = max(len(fullcall_rows), 1)
    learned_recall = learned_hits / n_retrieval
    lexical_recall = lexical_hits / n_retrieval
    oracle_rate = oracle_exact / n_fullcall
    oracle_er_rate = oracle_execute_refuse / n_fullcall
    learned_rate = learned_exact / n_fullcall
    gates = {
        "recall_at_5_learned": {
            "value": learned_recall,
            "floor": layers["recall_at_5_learned_min"],
            "ok": learned_recall >= float(layers["recall_at_5_learned_min"]),
        },
        "recall_beats_lexical": {
            "learned": learned_recall,
            "lexical": lexical_recall,
            "ok": (not layers["recall_at_5_must_exceed_lexical"])
            or learned_recall > lexical_recall,
        },
        "oracle_top5_fullcall_exact": {
            "value": oracle_rate,
            "floor": layers["oracle_top5_fullcall_exact_min"],
            "ok": oracle_rate >= float(layers["oracle_top5_fullcall_exact_min"]),
        },
        "oracle_execute_refuse": {
            "value": oracle_er_rate,
            "floor": layers["oracle_top5_execute_refuse_ok_min"],
            "ok": oracle_er_rate >= float(layers["oracle_top5_execute_refuse_ok_min"]),
        },
        "learned_top5_e2e_exact": {
            "value": learned_rate,
            "floor": layers["learned_top5_e2e_exact_min"],
            "ok": learned_rate >= float(layers["learned_top5_e2e_exact_min"]),
        },
        "unsupported_accepted": {
            "value": unsupported_accepted,
            "max": layers["unsupported_accepted_max"],
            "ok": unsupported_accepted <= int(layers["unsupported_accepted_max"]),
        },
        "unprovenanced_argument_accepted": {
            "value": unprovenanced_accepted,
            "max": layers["unprovenanced_argument_accepted_max"],
            "ok": unprovenanced_accepted
            <= int(layers["unprovenanced_argument_accepted_max"]),
        },
    }
    _release_locked_eval_mlx(mx, collect_python=True)
    memory_policy["final"] = _locked_eval_memory_snapshot(mx)
    return {
        "n_retrieval": len(retrieval_rows),
        "n_fullcall": len(fullcall_rows),
        "recall_at_5_learned": learned_recall,
        "recall_at_5_lexical": lexical_recall,
        "oracle_top5_fullcall_exact": oracle_rate,
        "oracle_execute_refuse": oracle_er_rate,
        "learned_top5_e2e_exact": learned_rate,
        "unsupported_accepted": unsupported_accepted,
        "unprovenanced_argument_accepted": unprovenanced_accepted,
        "quality_gates": gates,
        "quality_ok": all(bool(row["ok"]) for row in gates.values()),
        "thresholds_sha256": sha_file(thresholds_path),
        "eval_lock_sha256": sha_file(eval_lock / "lock.json"),
        "decode_mode": "constrained",
        "deterministic_gates_applied": True,
        "mlx_memory_policy": memory_policy,
        "not_a_score_claim": True,
    }


def _q4_diagnostic(master: Path, output: Path) -> dict[str, Any]:
    """Build a real all-Q4 WHT/codebook diagnostic; never label it CQ2."""

    _ensure_python_paths()
    import numpy as np

    from training.qat.cq2_policy_51m import lm_storage_dtype, uniform_group_bits
    from mei_sdk.cq2 import TensorToPack, build_tensor_container

    contract = architecture_contracts()["weight_contract"]["tensor_order"]
    arrays = np.load(master, allow_pickle=False)
    if list(arrays.files) != [str(row["name"]) for row in contract]:
        raise RuntimeError("Q4 diagnostic input does not match ordered 51M tensors")
    tensors = []
    q4_groups = safe = 0
    for row in contract:
        name = str(row["name"])
        shape = tuple(int(value) for value in row["shape"])
        storage = lm_storage_dtype(name, shape)
        if storage == "f16":
            tensors.append(TensorToPack(name, shape, arrays[name], "lm", "f16"))
            safe += 1
        else:
            groups = len(uniform_group_bits(name, shape) or ())
            tensors.append(
                TensorToPack(name, shape, arrays[name], "lm", "cq4", (4,) * groups)
            )
            q4_groups += groups
    blob, directory = build_tensor_container(tensors)
    output.write_bytes(blob)
    return {
        "diagnostic": "all-q4-group128-wht-codebook",
        "product_final": False,
        "cq2_claim": False,
        "bytes": len(blob),
        "sha256": sha_file(output),
        "tensor_count": len(directory),
        "q4_groups": q4_groups,
        "safe_f16_tensor_count": safe,
    }


def _subprocess_json(
    command: list[str], *, cwd: Path, log_path: Path
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("NO_PROXY", "*")
    proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    log_path.write_text(
        (proc.stdout or "")
        + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}")
    try:
        value = json.loads(proc.stdout)
    except json.JSONDecodeError:
        value = {"stdout_sha256": sha_file(log_path), "returncode": proc.returncode}
    return value if isinstance(value, dict) else {"value": value}


def _portable_gate_report(package_dir: Path, directory: Path) -> dict[str, Any]:
    """Qualify Python/MLX and Browser-WASM for the current product cycle.

    The Rust core remains a Browser-WASM implementation dependency. Independent
    native Rust, Node and C/FFI product qualification is opt-in via the
    ``extended`` scope and is deferred for the 300M mechanism-validation run.
    """

    directory.mkdir(parents=True, exist_ok=True)
    gate_log = directory / "run-gates.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src/model-factory"), str(ROOT / "src/platform/python-sdk"))
    )
    env["CARGO_TARGET_DIR"] = str(ROOT / ".local/cache/cargo-sdk-target")
    env["MEI_51M_PACKAGE_DIR"] = str(package_dir.resolve())
    runtime_scope = env.get("MEI_RUNTIME_VALIDATION_SCOPE", "python-browser-wasm")
    if runtime_scope not in {"python-browser-wasm", "extended"}:
        raise RuntimeError(f"unsupported runtime validation scope: {runtime_scope}")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/platform/_shared/tools/run_gates.py"),
            "--scope",
            runtime_scope,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    gate_log.write_text(
        (proc.stdout or "")
        + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"portable SDK gates failed; see {gate_log}")

    _ensure_python_paths()
    from mei_sdk.package import load_package

    python_package = load_package(package_dir)
    python_caps = python_package.capabilities()
    parity_keys = (
        "package_id",
        "package_format",
        "hash_verified",
        "tensor_identity_complete",
        "portable_quantization_policy_complete",
        "runtime_quantization_complete",
        "tool_index_payload_complete",
        "retrieval",
        "full_call",
        "mw_disposition",
        "confidence",
        "multi_step",
        "narration",
    )
    python_view = {key: python_caps.get(key) for key in parity_keys}
    required_true = (
        "hash_verified",
        "tensor_identity_complete",
        "portable_quantization_policy_complete",
        "runtime_quantization_complete",
        "tool_index_payload_complete",
        "retrieval",
        "full_call",
        "mw_disposition",
        "confidence",
        "multi_step",
        "narration",
    )
    if any(python_view.get(key) is not True for key in required_true):
        raise RuntimeError(
            "native v2 package is missing a required effective capability"
        )

    index_row = next(
        row
        for row in python_package.manifest["files"]
        if row.get("role") == "tool_index"
    )
    catalog = [
        row["schema"] for row in load_json(package_dir / index_row["path"])["records"]
    ]
    request = {
        "wire_version": "mei-runtime-wire-v2",
        "query": "打开厨房灯",
        "context": {},
        "evidence": [],
        "history": [],
        "tool_results": [],
        "permissions": {},
        "state": {},
        "decode_mode": "constrained",
        "max_new": 1,
    }
    from mei_sdk import Engine

    python_engine = Engine.load(str(package_dir), verify_hashes=True, backend="mlx-cq2")
    try:
        python_engine.register_tools(catalog)
        python_session = python_engine.create_session({"max_steps": 1})
        try:
            python_turn = python_session.complete(request)
        finally:
            python_session.close()
    finally:
        python_engine.close()
    python_cache_raw = python_turn.get("runtime_cache") or {}
    python_cache = {
        key: python_cache_raw.get(key)
        for key in (
            "kv_storage_dtype",
            "activation_quantization",
            "stable_prefix_tokens",
            "rolling_tokens",
            "visible_tokens",
            "output_reserve",
            "incremental_decode_steps",
            "recomputed_decode_steps",
            "cache_growth_bounded",
        )
    }
    if (
        python_turn.get("kind") not in {"call", "respond", "refuse", "error"}
        or python_cache.get("kv_storage_dtype") not in {"int8", "mlx.core.int8"}
        or python_cache.get("cache_growth_bounded") is not True
    ):
        raise RuntimeError("Python/MLX did not execute the native v2 int8 runtime")

    release_wasm_build = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "mei-sdk-wasm",
            "--target",
            "wasm32-unknown-unknown",
        ],
        cwd=ROOT / "src/platform/_shared",
        env=env,
        capture_output=True,
        text=True,
    )
    if release_wasm_build.returncode != 0:
        raise RuntimeError(
            "portable release WASM build failed: "
            + (release_wasm_build.stderr or release_wasm_build.stdout)[-4000:]
        )
    release_wasm = ROOT / ".local/cache/cargo-sdk-target/wasm32-unknown-unknown/release/mei_sdk_wasm.wasm"
    if not release_wasm.is_file():
        raise RuntimeError("portable SDK gates did not build the release WASM runtime")
    node_env = env.copy()
    node_env["MEI_SDK_WASM_PATH"] = str(release_wasm)
    node_proc = subprocess.run(
        ["node", str(ROOT / "src/platform/browser-sdk/smoke_node_v2.mjs")],
        cwd=ROOT,
        env=node_env,
        capture_output=True,
        text=True,
    )
    if node_proc.returncode != 0:
        raise RuntimeError(
            "Browser-WASM Node harness failed: "
            + (node_proc.stderr or node_proc.stdout)[-4000:]
        )
    node_report = json.loads(node_proc.stdout)
    bounded = node_report.get("runtime_contract") or {}
    expected_bounded = {
        "kv_storage_dtype": "int8",
        "activation_quantization": "int8-qdq",
        "activation_quantization_semantics": (
            "int8-symmetric-per-last-axis-vector-qdq"
        ),
        "activation_q_quantized": False,
        "kv_scale_granularity": "per-head-vector",
        "stable_prefix_tokens_max": 1024,
        "rolling_window_tokens": 256,
        "cache_growth_bounded": True,
    }
    if any(bounded.get(key) != value for key, value in expected_bounded.items()):
        raise RuntimeError(f"Browser-WASM bounded int8 contract mismatch: {bounded}")
    if (
        node_report.get("ok") is not True
        or node_report.get("implementation_complete") is not True
        or node_report.get("inference") is not True
        or (node_report.get("runtime_head_execution") or {}).get("mw_disposition")
        != "independent-20class-sidecar"
        or (node_report.get("runtime_head_execution") or {}).get("narration")
        != "independent-frozen-backbone-rank16-generation-sidecar"
        or (node_report.get("narration_numeric_runtime") or {}).get("residual_width")
        != 24_000
        or (node_report.get("narration_numeric_runtime") or {}).get("semantic_boundary")
        != "independent-frozen-backbone-rank16-generation-sidecar"
    ):
        raise RuntimeError(
            f"Browser-WASM mechanism evidence is incomplete: {node_report}"
        )

    if runtime_scope == "python-browser-wasm":
        return {
            "all_gates_passed": True,
            "validation_scope": runtime_scope,
            "run_gates_log": str(gate_log),
            "run_gates_log_sha256": sha_file(gate_log),
            "python_package_validation": True,
            "python_capabilities": python_view,
            "python_mlx_numeric_runtime": {
                "turn_kind": python_turn["kind"],
                "runtime_cache": python_cache,
                "mw_disposition": python_turn.get("mw_disposition"),
            },
            "browser_wasm_package_validation": True,
            "browser_wasm_numeric_runtime": node_report,
            "bounded_int8_runtime": bounded,
            "browser_wasm_build": {
                "profile": "release",
                "artifact_sha256": sha_file(release_wasm),
                "implementation_dependency": "rust-core",
            },
            "deferred_independent_runtimes": ["rust-sdk", "node-sdk", "c-ffi"],
            "semantic_boundaries": {
                "retrieval": "independent-contrastive-head",
                "mw_disposition": "independent-20class-sidecar",
                "confidence": "independent-calibrated-binary-head",
                "mw_deviation": "deterministic-governance-gate-no-tensors",
            },
        }

    release_build = subprocess.run(
        ["cargo", "build", "--release", "-p", "mei-sdk-cli"],
        cwd=ROOT / "src/platform/_shared",
        env=env,
        capture_output=True,
        text=True,
    )
    if release_build.returncode != 0:
        raise RuntimeError(
            "portable release CLI build failed: "
            + (release_build.stderr or release_build.stdout)[-4000:]
        )
    cli = ROOT / ".local/cache/cargo-sdk-target/release/mei-sdk"
    if not cli.is_file():
        raise RuntimeError("portable SDK gates did not build the Rust CLI")
    rust_proc = subprocess.run(
        [str(cli), "validate-package", str(package_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if rust_proc.returncode != 0:
        raise RuntimeError(f"Rust rejected native package: {rust_proc.stderr[-2000:]}")
    rust_caps = json.loads(rust_proc.stdout)
    contract_proc = subprocess.run(
        [str(cli), "runtime-contract"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if contract_proc.returncode != 0:
        raise RuntimeError(
            "Rust runtime contract evidence is unavailable: "
            + (contract_proc.stderr or contract_proc.stdout)[-2000:]
        )
    native_bounded = json.loads(contract_proc.stdout)
    if native_bounded != bounded:
        raise RuntimeError("native Rust and Browser-WASM runtime contracts differ")
    rust_view = {key: rust_caps.get(key) for key in parity_keys}
    if python_view != rust_view:
        raise RuntimeError(
            "Python/Rust package capability parity failed: "
            + json.dumps({"python": python_view, "rust": rust_view}, ensure_ascii=False)
        )

    head_parity_path = directory / "portable-head-parity.json"
    head_parity = _subprocess_json(
        [
            sys.executable,
            str(ROOT / "src/model-factory/evaluation/heads/compare_portable_heads_51m.py"),
            "--package",
            str(package_dir),
            "--rust-cli",
            str(cli),
            "--out",
            str(head_parity_path),
        ],
        cwd=ROOT,
        log_path=directory / "portable-head-parity.log",
    )
    if head_parity.get("all_passed") is not True:
        raise RuntimeError("independent head numerical parity failed")

    from mei_sdk.ffi import NativeEngine

    native = NativeEngine()
    try:
        native.open(str(package_dir))
        native.register_tools(catalog)
        with native.create_session({"max_steps": 1}) as session:
            ffi_turn = session.complete(request)
    finally:
        native.close()
    if (
        ffi_turn.get("kind") not in {"call", "respond", "refuse", "error"}
        or (ffi_turn.get("runtime_cache") or {}).get("kv_storage_dtype") != "int8"
        or (ffi_turn.get("mw_disposition") or {}).get("source") != "mw-head"
    ):
        raise RuntimeError("C ABI did not execute the v2 LM/MW/int8 runtime path")

    return {
        "all_gates_passed": True,
        "validation_scope": runtime_scope,
        "run_gates_log": str(gate_log),
        "run_gates_log_sha256": sha_file(gate_log),
        "python_package_validation": True,
        "rust_package_validation": True,
        "package_capability_parity": True,
        "python_capabilities": python_view,
        "rust_capabilities": rust_view,
        "bounded_int8_runtime": bounded,
        "independent_head_parity": head_parity,
        "independent_head_parity_sha256": sha_file(head_parity_path),
        "c_abi_numeric_runtime": {
            "turn_kind": ffi_turn["kind"],
            "runtime_cache": ffi_turn["runtime_cache"],
            "mw_disposition": ffi_turn["mw_disposition"],
        },
        "browser_wasm_numeric_runtime": node_report,
        "node_wasm_build": {
            "profile": "release",
            "artifact_sha256": sha_file(release_wasm),
        },
        "semantic_boundaries": {
            "retrieval": "independent-contrastive-head",
            "mw_disposition": "independent-20class-sidecar",
            "confidence": "independent-calibrated-binary-head",
            "mw_deviation": "deterministic-governance-gate-no-tensors",
        },
    }


def _training_and_package_stages(
    args: argparse.Namespace, plan: dict[str, Any], run_dir: Path
) -> dict[str, dict[str, Any]]:
    """Execute the fixed order through the single portable CQ2 package."""

    _ensure_python_paths()
    from common.checkpoint import save_params
    from training.qat.cq2_qat_51m import explicit_group_map
    from release.pack_cq2_v2_51m import export as export_cq2
    from training.qat.qat_cq2_v2_51m import run as run_qat
    from training.tool_use.train_sft_ondisk_51m import (
        finalize_tool_index,
        float_task_control,
        save_product_heads,
        train_confidence,
        train_contrastive,
        train_fullcall,
        train_mw_disposition,
        train_narration_adapter,
    )

    receipts: dict[str, dict[str, Any]] = {}
    data = args.data_release
    catalog = _catalog(args.eval_lock)
    tools_by_name = {str(row.get("name") or ""): row for row in catalog}
    retrieval_rows = load_jsonl(data / "retrieval.train.jsonl")
    fullcall_rows = load_jsonl(data / "full-call.train.jsonl")

    anchor_path = run_dir / "stages/float_base_lm_anchor/float-base-lm-anchor.json"
    receipts["float_base_lm_anchor"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="float_base_lm_anchor",
        resume=args.resume,
        action=lambda directory: (
            _subprocess_json(
                [
                    sys.executable,
                    str(ROOT / "src/model-factory/evaluation/base/freeze_float_baseline_51m.py"),
                    "--out-dir",
                    str(directory),
                    "--weights",
                    str(args.base_weights.resolve()),
                    "--release",
                    str(args.base_release.resolve()),
                ],
                cwd=ROOT,
                log_path=directory / "command.log",
            ),
            [directory / "float-base-lm-anchor.json", directory / "command.log"],
        ),
    )

    def float_control_action(directory: Path):
        report = float_task_control(
            fullcall_rows,
            args.float_control_steps,
            parent_weights=args.base_weights,
            tools_by_name=tools_by_name,
            eval_n=min(24, args.eval_limit),
        )
        path = directory / "float-task-control.json"
        write_json(path, report)
        return report, [path]

    receipts["float_task_control"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="float_task_control",
        action=float_control_action,
        resume=args.resume,
    )

    def q4_action(directory: Path):
        output = directory / "q4-diagnostic.bin"
        report = _q4_diagnostic(args.base_weights, output)
        path = directory / "q4-diagnostic.json"
        write_json(path, report)
        return report, [output, path]

    receipts["q4_diagnostic"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="q4_diagnostic",
        action=q4_action,
        resume=args.resume,
    )

    qat_worker = run_dir / "stages/cq2_qat_v2/worker"
    qat_master = qat_worker / "stages/cq2_qat_v2/final-master.npz"

    def qat_action(directory: Path):
        namespace = argparse.Namespace(
            base_release=args.base_release.resolve(),
            base_weights=args.base_weights.resolve(),
            float_anchor=anchor_path.resolve(),
            replay_corpus=args.replay_corpus.resolve(),
            run_dir=(directory / "worker").resolve(),
            target_tokens=args.qat_tokens,
            seq_len=args.qat_seq_len,
            batch_size=args.qat_batch_size,
            grad_accum=args.qat_grad_accum,
            lr=5e-5,
            checkpoint_every_steps=100,
            resume=args.resume,
            dry_run=False,
            allow_live_cpt=bool(getattr(args, "allow_live_cpt", False)),
        )
        report = run_qat(namespace)
        worker_receipt = directory / "worker/stages/cq2_qat_v2/receipt.json"
        return report, [qat_master, worker_receipt]

    receipts["cq2_qat_v2"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="cq2_qat_v2",
        action=qat_action,
        resume=args.resume,
    )

    if bool(getattr(args, "qat_only", False)):
        return receipts

    mw_rows = load_jsonl(data / "mw-disposition.train.jsonl")
    confidence_rows = load_jsonl(data / "confidence-calibration.train.jsonl")
    narration_train_rows = load_jsonl(args.narration_release / "narration.train.jsonl")
    narration_valid_rows = load_jsonl(args.narration_release / "narration.valid.jsonl")

    r0_path = run_dir / "stages/retrieval_r0/retrieval-r0.npz"

    def r0_action(directory: Path):
        runtime = _load_runtime(qat_master, quantized=True)
        report = train_contrastive(
            runtime,
            retrieval_rows,
            steps=args.retrieval_r0_steps,
            lr=1e-3,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            stage_id="r0",
        )
        _save_model_or_head(runtime.contrastive, r0_path)
        path = directory / "retrieval-r0.json"
        write_json(path, report)
        return report, [r0_path, path]

    receipts["retrieval_r0"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="retrieval_r0",
        action=r0_action,
        resume=args.resume,
    )

    final_master = run_dir / "stages/oracle_top5_fullcall/final-master.npz"

    def fullcall_action(directory: Path):
        runtime = _load_runtime(qat_master, quantized=False)
        group_map = explicit_group_map(runtime.model.parameters())
        report = train_fullcall(
            runtime,
            fullcall_rows,
            steps=args.fullcall_steps,
            lr=2e-4,
            group_map=group_map,
            activation_ste=True,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            tools_by_name=tools_by_name,
        )
        save_params(runtime.model, final_master)
        path = directory / "oracle-top5-fullcall.json"
        write_json(path, report)
        return report, [final_master, path]

    receipts["oracle_top5_fullcall"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="oracle_top5_fullcall",
        action=fullcall_action,
        resume=args.resume,
    )

    r1_path = run_dir / "stages/retrieval_r1/retrieval-r1.npz"

    def r1_action(directory: Path):
        runtime = _load_runtime(final_master, quantized=True)
        report = train_contrastive(
            runtime,
            retrieval_rows,
            steps=args.retrieval_r1_steps,
            lr=1e-3,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            stage_id="r1",
        )
        _save_model_or_head(runtime.contrastive, r1_path)
        path = directory / "retrieval-r1.json"
        write_json(path, report)
        return report, [r1_path, path]

    receipts["retrieval_r1"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="retrieval_r1",
        action=r1_action,
        resume=args.resume,
    )

    index_path = run_dir / "stages/tool_index_final/tool-index.json"

    def index_action(directory: Path):
        runtime = _load_runtime(final_master, quantized=True, retrieval_head=r1_path)
        report = finalize_tool_index(
            runtime,
            catalog,
            index_path,
            model_sha256=sha_file(final_master),
            head_sha256=sha_file(r1_path),
            tokenizer_sha256=sha_file(TOKENIZER_ZH_V1),
        )
        path = directory / "tool-index-final.json"
        write_json(path, report)
        return report, [index_path, path]

    receipts["tool_index_final"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="tool_index_final",
        action=index_action,
        resume=args.resume,
    )

    def e2e_action(directory: Path):
        runtime = _load_runtime(
            final_master, quantized=True, retrieval_head=r1_path, tool_index=index_path
        )
        report = _locked_quality_eval(
            runtime,
            eval_lock=args.eval_lock,
            catalog=catalog,
            thresholds_path=args.quality_thresholds,
            limit=args.eval_limit,
        )
        report["degraded"] = not report["quality_ok"]
        path = directory / "learned-top5-e2e.json"
        write_json(path, report)
        return report, [path]

    receipts["learned_top5_e2e"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="learned_top5_e2e",
        action=e2e_action,
        resume=args.resume,
    )

    mw_path = run_dir / "stages/mw_disposition_20class/mw-disposition.npz"

    def mw_action(directory: Path):
        runtime = _load_runtime(
            final_master, quantized=True, retrieval_head=r1_path, tool_index=index_path
        )
        report = train_mw_disposition(
            runtime,
            mw_rows,
            steps=args.mw_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            catalog=catalog,
        )
        _save_model_or_head(runtime.mw_disposition_head, mw_path)
        path = directory / "mw-disposition-20class.json"
        write_json(path, {**report, "mw_deviation_tensor_count": 0})
        return report, [mw_path, path]

    receipts["mw_disposition_20class"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="mw_disposition_20class",
        action=mw_action,
        resume=args.resume,
    )

    confidence_path = run_dir / "stages/confidence_calibration/confidence.npz"

    def confidence_action(directory: Path):
        runtime = _load_runtime(
            final_master, quantized=True, retrieval_head=r1_path, tool_index=index_path
        )
        report = train_confidence(
            runtime,
            confidence_rows,
            steps=args.confidence_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            catalog=catalog,
            outcome_limit=min(args.eval_limit, 128),
        )
        _save_model_or_head(runtime.conf_v2, confidence_path)
        path = directory / "confidence-calibration.json"
        write_json(path, report)
        return report, [confidence_path, path]

    receipts["confidence_calibration"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="confidence_calibration",
        action=confidence_action,
        resume=args.resume,
    )

    narration_path = run_dir / "stages/narration_adapter/narration-adapter.npz"

    def narration_action(directory: Path):
        runtime = _load_runtime(final_master, quantized=True)
        report = train_narration_adapter(
            runtime,
            narration_train_rows,
            narration_valid_rows,
            steps=args.narration_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
        )
        _save_model_or_head(runtime.narration_adapter, narration_path)
        path = directory / "narration-adapter.json"
        write_json(path, report)
        return report, [narration_path, path]

    receipts["narration_adapter"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="narration_adapter",
        action=narration_action,
        resume=args.resume,
    )

    package_dir = run_dir / "candidate" / args.package_id

    def package_action(directory: Path):
        runtime = _load_runtime(
            final_master,
            quantized=False,
            retrieval_head=r1_path,
            mw_head=mw_path,
            confidence_head=confidence_path,
            narration_head=narration_path,
        )
        heads_path = directory / "product-heads.npz"
        heads_report = save_product_heads(runtime, heads_path)
        evidence = {}
        for component, stage_name in {
            "lm": "oracle_top5_fullcall",
            "contrastive": "retrieval_r1",
            "mw_disposition": "mw_disposition_20class",
            "confidence": "confidence_calibration",
            "narration_adapter": "narration_adapter",
        }.items():
            row = receipts[stage_name]
            evidence[component] = {
                "stage_id": stage_name,
                "stage_fingerprint_sha256": row["stage_fingerprint_sha256"],
                "status": "passed",
            }
        evidence_path = directory / "stage-evidence.json"
        write_json(evidence_path, evidence)
        report = export_cq2(
            argparse.Namespace(
                master=final_master,
                heads=heads_path,
                tool_index=index_path,
                stage_evidence=evidence_path,
                out_dir=package_dir,
                package_id=args.package_id,
                parent_package_id=plan["immutable"]["base"]["base_id"],
            )
        )
        report["heads"] = heads_report
        path = directory / "package-v2-cq2.json"
        write_json(path, report)
        outputs = [path, heads_path]
        outputs.extend(sorted(item for item in package_dir.iterdir() if item.is_file()))
        return report, outputs

    receipts["package_v2_cq2"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="package_v2_cq2",
        action=package_action,
        resume=args.resume,
    )

    def mtp_action(directory: Path):
        from training.heads.mtp_ablation_51m import run as run_mtp
        from mei_sdk.package import load_package

        output = directory / "mtp-ablation.json"
        report = run_mtp(
            argparse.Namespace(
                master=final_master,
                replay_corpus=args.replay_corpus,
                out=output,
                steps=args.mtp_steps,
                seq_len=min(args.qat_seq_len, 256),
                batch_size=1,
                lr=1e-3,
                dry_run=False,
            )
        )
        package = load_package(package_dir)
        mtp_names = [
            row.get("name")
            for row in (package.manifest.get("tensor_container") or {}).get("directory")
            or []
            if "mtp" in str(row.get("name") or "").lower()
        ]
        if mtp_names:
            raise RuntimeError(f"deployment package retained MTP tensors: {mtp_names}")
        report["deployment_mtp_tensor_count"] = 0
        write_json(output, report)
        return report, [output]

    receipts["mtp_ablation"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="mtp_ablation",
        action=mtp_action,
        resume=args.resume,
    )

    portable_report_path = (
        run_dir / "stages/portable_runtime_gates/portable-runtime-gates.json"
    )

    def portable_action(directory: Path):
        report = _portable_gate_report(package_dir, directory)
        write_json(portable_report_path, report)
        return report, [portable_report_path, directory / "run-gates.log"]

    receipts["portable_runtime_gates"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="portable_runtime_gates",
        action=portable_action,
        resume=args.resume,
    )

    resource_report_path = (
        run_dir / "stages/resource_measurement/resource-measurement.json"
    )

    def resource_action(directory: Path):
        from evaluation.resources.measure_resources_51m import measure
        from mei_sdk.package import load_package

        # A process may have completed measurement just before the outer stage
        # receipt was written.  Reuse only a fully verified measured copy;
        # otherwise choose a new suffix and preserve the interrupted artifact.
        if resource_report_path.is_file():
            previous = load_json(resource_report_path)
            previous_package = Path(str(previous.get("measured_package") or ""))
            if previous_package.is_dir():
                candidate = load_package(previous_package)
                if candidate.resource_measurement_verified:
                    previous["degraded"] = not bool(previous.get("resource_eligible"))
                    outputs = [resource_report_path]
                    outputs.extend(
                        sorted(
                            path
                            for path in previous_package.iterdir()
                            if path.is_file()
                        )
                    )
                    return previous, outputs

        root = run_dir / "final-candidate"
        root.mkdir(parents=True, exist_ok=True)
        measured_package = root / args.package_id
        counter = 0
        while measured_package.exists():
            counter += 1
            measured_package = root / f"{args.package_id}-measure-{counter:03d}"
        report = measure(
            argparse.Namespace(
                source_package=package_dir,
                out_package=measured_package,
                work_dir=directory / "work",
                out=resource_report_path,
            )
        )
        report["degraded"] = not bool(report.get("resource_eligible"))
        write_json(resource_report_path, report)
        outputs = [resource_report_path]
        outputs.extend(
            sorted(path for path in measured_package.iterdir() if path.is_file())
        )
        return report, outputs

    receipts["resource_measurement"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="resource_measurement",
        action=resource_action,
        resume=args.resume,
    )
    measured_package = Path(
        str(receipts["resource_measurement"]["metrics"]["measured_package"])
    )

    alignment_json = run_dir / "stages/needle2_alignment_report/needle2-alignment.json"
    alignment_md = (
        run_dir / "stages/needle2_alignment_report/needle2-alignment.zh-CN.md"
    )

    def alignment_action(directory: Path):
        from evaluation.alignment.needle2_alignment_51m import run as run_alignment

        report = run_alignment(
            argparse.Namespace(
                run_dir=run_dir,
                package=measured_package,
                portable_gates=portable_report_path,
                resource_report=resource_report_path,
                out_json=alignment_json,
                out_md=alignment_md,
            )
        )
        report["degraded"] = not bool(report.get("mechanism_alignment_validated"))
        write_json(alignment_json, report)
        return report, [alignment_json, alignment_md]

    receipts["needle2_alignment_report"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="needle2_alignment_report",
        action=alignment_action,
        resume=args.resume,
    )

    final_audit_path = run_dir / "stages/final_audit/final-audit.json"
    freeze_proposal_path = run_dir / "stages/final_audit/freeze-proposal.json"

    def final_action(directory: Path):
        from release.final_audit_51m import run as run_final_audit

        report = run_final_audit(
            argparse.Namespace(
                run_dir=run_dir,
                package=measured_package,
                portable_gates=portable_report_path,
                resource_report=resource_report_path,
                alignment_report=alignment_json,
                out=final_audit_path,
                freeze_proposal=freeze_proposal_path,
            )
        )
        report["degraded"] = not bool(report.get("release_eligible"))
        write_json(final_audit_path, report)
        return report, [final_audit_path, freeze_proposal_path]

    receipts["final_audit"] = run_stage(
        run_dir=run_dir,
        plan=plan,
        name="final_audit",
        action=final_action,
        resume=args.resume,
    )
    return receipts


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    active = live_cpt_workers()
    if active and not bool(getattr(args, "allow_live_cpt", False)):
        raise RuntimeError(
            "live CPT owns MLX/Metal; productization heavy stages are deferred: "
            + json.dumps(active, ensure_ascii=False)
        )
    # All artifact paths recorded below are repository-relative.  Normalize the
    # selected run directory once so a caller-provided relative --run-dir does
    # not survive until the final receipt-registration boundary.
    run_dir = choose_run_dir(args.run_dir, plan, resume=args.resume).resolve()
    write_plan(run_dir, plan)
    mx, memory_policy = configure_mlx_memory(reset_peak=True)
    memory_policy_path = run_dir / "mlx-memory-policy.json"
    write_json(
        memory_policy_path,
        {
            **memory_policy,
            "scope": "entire-productizer-process",
            "started_unix": time.time(),
            "initial": mlx_memory_snapshot(mx),
        },
    )
    receipts = _training_and_package_stages(args, plan, run_dir)
    release_mlx_memory(mx, collect_python=True)
    write_json(
        memory_policy_path,
        {
            **memory_policy,
            "scope": "entire-productizer-process",
            "finished_unix": time.time(),
            "final": mlx_memory_snapshot(mx),
        },
    )
    if bool(getattr(args, "qat_only", False)):
        worker_receipt_path = (
            run_dir / "stages/cq2_qat_v2/worker/stages/cq2_qat_v2/receipt.json"
        )
        worker = load_json(worker_receipt_path)
        master_path = Path(str((worker.get("outputs") or {}).get("master") or ""))
        if not master_path.is_absolute():
            master_path = ROOT / master_path
        if (
            worker.get("terminal_status") != "passed"
            or worker.get("quant_math_id") != "mei-cq-v2-g128-wht-codebook"
            or worker.get("parent_id") != plan["immutable"]["base"]["base_id"]
            or not master_path.is_file()
            or sha_file(master_path) != (worker.get("outputs") or {}).get("master_sha256")
        ):
            raise RuntimeError("CQ2-QAT-only output is not a verified Base-bound candidate")
        candidate_path = run_dir / "qat-import-candidate-receipt.json"
        base = plan["immutable"]["base"]
        candidate = {
            "schema": "mei-cq2-qat-import-candidate-receipt-v1",
            "status": "passed",
            "base": {
                "model_id": base["base_id"],
                "tokens_seen_exposure": base["tokens_seen_exposure"],
                "weights_sha256": base["weights_sha256"],
                "numeric_integrity": base.get("numeric_integrity"),
            },
            "quant_math_id": worker["quant_math_id"],
            "master": {
                "path": str(master_path.relative_to(ROOT)),
                "sha256": sha_file(master_path),
                "bytes": master_path.stat().st_size,
            },
            "worker_receipt": {
                "path": str(worker_receipt_path.relative_to(ROOT)),
                "sha256": sha_file(worker_receipt_path),
            },
            "tokens_seen_qat": int((worker.get("metrics") or {}).get("tokens_seen_qat") or 0),
            "valid_loss": (worker.get("metrics") or {}).get("valid_loss"),
            "reuse_policy": (
                "candidate only; productizer must revalidate Base, corpus, quant math, "
                "contracts, stage fingerprint and output SHA before import"
            ),
            "later_base_policy": "each immutable Base requires its own CQ2-QAT stage",
        }
        write_json(candidate_path, candidate)
        return {
            "run_dir": str(run_dir),
            "completed_through": "cq2_qat_v2",
            "stage_receipts": receipts,
            "qat_import_candidate": str(candidate_path),
            "process_complete": False,
            "release_eligible": False,
            "pending_stages": list(STAGES[4:]),
            "mlx_memory_policy": str(memory_policy_path),
        }
    final_metrics = receipts["final_audit"]["metrics"]
    return {
        "run_dir": str(run_dir),
        "completed_through": "final_audit",
        "stage_receipts": receipts,
        "process_complete": bool(final_metrics.get("process_complete")),
        "release_eligible": bool(final_metrics.get("release_eligible")),
        "pending_stages": [],
        "mlx_memory_policy": str(memory_policy_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", type=Path, default=DEFAULT_BASE_RELEASE)
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument("--data-release", type=Path, default=DEFAULT_DATA_RELEASE)
    parser.add_argument(
        "--narration-release", type=Path, default=DEFAULT_NARRATION_RELEASE
    )
    parser.add_argument("--eval-lock", type=Path, default=DEFAULT_EVAL_LOCK)
    parser.add_argument(
        "--quality-thresholds", type=Path, default=DEFAULT_QUALITY_THRESHOLDS
    )
    parser.add_argument("--replay-corpus", type=Path, default=DEFAULT_REPLAY_CORPUS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--package-id", default=DEFAULT_PACKAGE_ID)
    parser.add_argument("--qat-tokens", type=int, default=5_000_000)
    parser.add_argument("--qat-seq-len", type=int, default=512)
    parser.add_argument("--qat-batch-size", type=int, default=4)
    parser.add_argument("--qat-grad-accum", type=int, default=1)
    parser.add_argument("--float-control-steps", type=int, default=200)
    parser.add_argument("--retrieval-r0-steps", type=int, default=400)
    parser.add_argument("--fullcall-steps", type=int, default=4_000)
    parser.add_argument("--retrieval-r1-steps", type=int, default=400)
    parser.add_argument("--mw-steps", type=int, default=400)
    parser.add_argument("--confidence-steps", type=int, default=80)
    parser.add_argument("--narration-steps", type=int, default=1_200)
    parser.add_argument("--mtp-steps", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=1_200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--qat-only",
        action="store_true",
        help=(
            "execute Float anchor/control, Q4 diagnostic and Base-bound CQ2 QAT, "
            "then emit a verified import candidate for the SFT-v4 productizer"
        ),
    )
    parser.add_argument(
        "--allow-live-cpt",
        action="store_true",
        help=(
            "explicitly authorize this independent productization run to share "
            "MLX/Metal with a live CPT worker; recorded in the run fingerprint"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    positive = (
        "qat_tokens",
        "qat_seq_len",
        "qat_batch_size",
        "qat_grad_accum",
        "float_control_steps",
        "retrieval_r0_steps",
        "fullcall_steps",
        "retrieval_r1_steps",
        "mw_steps",
        "confidence_steps",
        "narration_steps",
        "mtp_steps",
        "eval_limit",
    )
    if any(int(getattr(args, field)) <= 0 for field in positive):
        parser.error("all training/evaluation budgets must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    run_dir = choose_run_dir(args.run_dir, plan, resume=args.resume)
    result = {
        **plan,
        "requested_run_dir": str(args.run_dir.resolve()),
        "resolved_run_dir": str(run_dir.resolve()),
        "executable_through": (
            "cq2_qat_v2" if bool(getattr(args, "qat_only", False)) else "final_audit"
        ),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(execute(args, plan), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
