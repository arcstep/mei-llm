#!/usr/bin/env python3
"""Productize one frozen mei-51m Base under the adaptive runtime-v2 contract.

This runner is an immutable continuation of a verified SFT-v4 training prefix.
It never relabels the old prefix as having used the new prompt contract: the
adoption receipt records the runtime-profile transition, then all LM-changing
work is replayed from the adopted Agent master under the exact shared Runtime
prompt.  CPT and CQ2-QAT are deliberately outside this graph.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Keep the runner directly executable from the repository root.  The shared
# adaptive-view module imports the public Python SDK at module import time, so
# its package root must be visible before importing that module.
_BOOTSTRAP_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "CURRENT.json").is_file()
)
_SDK_PYTHON = str(_BOOTSTRAP_ROOT / "platform/python-sdk")
if _SDK_PYTHON not in sys.path:
    sys.path.insert(0, _SDK_PYTHON)

import training.tool_use.adaptive_tool_context_51m as adaptive  # noqa: E402
import orchestration.productize_51m as lifecycle  # noqa: E402
import orchestration.productize_sft_v3_300m as v4  # noqa: E402
import evaluation.tool_use.sft_v3_eval_51m as evaluation  # noqa: E402
import training.tool_use.sft_v3_training_51m as training  # noqa: E402
import contracts.sft_v4_contract_51m as contract  # noqa: E402
from release import freeze_sft_v3_release_51m as freeze_sft_v3_release  # noqa: E402
from common._repo import (  # noqa: E402
    CURRENT_PATH,
    ROOT,
    TOKENIZER_ZH_V1,
    architecture_contracts,
    phase_binding_identity,
    resolve_repo_path,
)


RUNNER_ID = "mei-51m-adaptive-productizer-v5"
PLAN_SCHEMA = "mei-51m-adaptive-productization-plan-v1"
PROGRESS_SCHEMA = "mei-51m-adaptive-productization-progress-v1"
CURRENT_BASELINE_SHA256 = (
    "5b0b68eeb8322bb9cdbef112777b1b346b69a91f3bce7234b1c6370389a42607"
)
PREVIOUS_RUNTIME_PROFILE_SHA256 = (
    "74839b08155e624f14318ca8646166ddc68ee6496720dedac26aa91fdc8bdf43"
)
CURRENT_RUNTIME_PROFILE_SHA256 = (
    "f3a4ab1151e82299fee0214c5f78a18c20d83367d24a9dc073bbcd56312fa512"
)

DEFAULT_DATA_RELEASE = contract.DEFAULT_RELEASE_ROOT / contract.RELEASE_ID
DEFAULT_LINGUISTIC_AUGMENTATION = (
    contract.DEFAULT_RELEASE_ROOT / contract.LINGUISTIC_AUGMENTATION_ID
)
DEFAULT_EVAL_LOCK = contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID
DEFAULT_NARRATION_RELEASE = contract.NARRATION_RELEASE_DIR
DEFAULT_BASE_DIR = ROOT / ".local/artifacts/mei-1.0-51m/exp-000300m/models/base/mei-1.0-51m-base-scratch300m-v1"
DEFAULT_BASE_RELEASE = DEFAULT_BASE_DIR / "RELEASE.json"
DEFAULT_BASE_WEIGHTS = DEFAULT_BASE_DIR / "mei-1.0-51m-base-scratch300m-v1.npz"
DEFAULT_QAT_IMPORT = DEFAULT_DATA_RELEASE / "qat-import-candidate-receipt.json"
DEFAULT_SEED_RUN = (
    ROOT
    / ".local/artifacts/mei-1.0-51m/exp-000300m/runs/"
    "productize-scratch300m-sft-v4-quality-schema-cq2-v2-cbd960d8f4df"
)
DEFAULT_RUN_DIR = (
    ROOT
    / ".local/artifacts/mei-1.0-51m/exp-000300m/runs/"
    "productize-scratch300m-adaptive-v5-cq2-v2"
)
DEFAULT_PACKAGE_ID = "mei-1.0-51m-scratch300m-tool-sft-cq2-v2-adaptive-v5"

EVALUATION_MW_OVERRIDE = {
    "decision": "continue",
    "source": "protocol-test",
}

STAGES = (
    "adopt_training_prefix_v4",
    "seed_adaptive_views_v5",
    "fullcall_alignment_replay_v5",
    "agent_alignment_replay_v5",
    "retrieval_r2_v5",
    "final_adaptive_artifacts_v5",
    "adaptive_generation_eval_v5",
    "mw_disposition_v5",
    "confidence_harvest_v5",
    "confidence_head_v5",
    "narration_adapter_v5",
    "sidecar_runtime_eval_v5",
    "package_v2_cq2_v5",
    "python_runtime_gate_v5",
    "browser_wasm_gate_v5",
    "final_audit_v5",
)

PHASE_SCOPE_STAGES = {
    "sft-alignment": {
        "adopt_training_prefix_v4",
        "seed_adaptive_views_v5",
        "fullcall_alignment_replay_v5",
        "agent_alignment_replay_v5",
        "retrieval_r2_v5",
        "final_adaptive_artifacts_v5",
        "mw_disposition_v5",
        "confidence_harvest_v5",
        "confidence_head_v5",
        "narration_adapter_v5",
    },
    "model-evaluation": {
        "adaptive_generation_eval_v5",
        "sidecar_runtime_eval_v5",
    },
    "runtime-release": {
        "package_v2_cq2_v5",
        "python_runtime_gate_v5",
        "browser_wasm_gate_v5",
        "final_audit_v5",
    },
}


def phase_stage_mode(phase_scope: str | None, stage_id: str) -> str:
    if phase_scope is None or stage_id in PHASE_SCOPE_STAGES[phase_scope]:
        return "execute"
    if phase_scope == "sft-alignment" and stage_id == "adaptive_generation_eval_v5":
        return "skip"
    return "reuse_only"


class PhaseBoundaryReached(RuntimeError):
    def __init__(self, stage_id: str, receipts: dict[str, dict[str, Any]]):
        super().__init__(stage_id)
        self.stage_id = stage_id
        self.receipts = dict(receipts)

ADAPTIVE_PREFIX_ADOPTION_STAGE = "adopt_adaptive_prefix_v5"
ADAPTIVE_PREFIX_STAGES = STAGES[:6]
ADAPTIVE_DOWNSTREAM_STAGES = STAGES[6:]
PACKAGED_RUN_ADOPTION_STAGE = "adopt_packaged_run_v5"
PACKAGED_RUN_STAGES = (
    ADAPTIVE_PREFIX_ADOPTION_STAGE,
    "adaptive_generation_eval_v5",
    "mw_disposition_v5",
    "confidence_harvest_v5",
    "confidence_head_v5",
    "narration_adapter_v5",
    "sidecar_runtime_eval_v5",
    "package_v2_cq2_v5",
)
PACKAGED_RUN_DOWNSTREAM_STAGES = (
    "python_runtime_gate_v5",
    "browser_wasm_gate_v5",
    "final_audit_v5",
)


def _emit(event: str, **values: Any) -> None:
    print(
        contract.canonical_bytes(
            {"event": event, "unix": time.time(), **values}
        ).decode("utf-8"),
        flush=True,
    )


def _browser_gate_metrics(
    measured: Mapping[str, Any],
    *,
    functional: bool,
    build_returncode: int,
    smoke_returncode: int | None,
) -> dict[str, Any]:
    heap_peak = int(measured.get("wasm_heap_peak_bytes") or 0)
    heap_ok = functional and 0 < heap_peak <= 96 * 1024 * 1024
    decode_tok_s = measured.get("steady_decode_tok_s")
    warm_decode_ok = bool(
        functional
        and measured.get("steady_decode_ran") is True
        and isinstance(decode_tok_s, (int, float))
        and math.isfinite(float(decode_tok_s))
        and float(decode_tok_s) >= 100.0
    )
    return {
        "status": "passed",
        "schema": "mei-browser-wasm-adaptive-runtime-gate-v1",
        "build_returncode": build_returncode,
        "smoke_returncode": smoke_returncode,
        "functional_complete": functional,
        "heap_peak_bytes": heap_peak or None,
        "heap_within_96_mib": heap_ok,
        "warm_steady_decode_tok_s": decode_tok_s,
        "warm_steady_decode_100_tok_s_validated": warm_decode_ok,
        "performance_gate_pending_structural_optimization": not warm_decode_ok,
        "measurement_profile": measured.get("measurement_profile"),
        "degraded": not (functional and heap_ok and warm_decode_ok),
    }


def _joint_budget_stress_request(
    request: Mapping[str, Any], *, output_reserve: int
) -> dict[str, Any]:
    """Build the joint-budget probe with the product output reserve intact.

    The ordinary one-token inference probe deliberately sets ``max_new=1`` to
    keep the gate cheap. Reusing that value for the budget stress case changes
    the prompt cap from the product contract's 1920 tokens to 2047 and makes a
    correct runtime look non-compliant.
    """

    if output_reserve <= 0 or output_reserve >= 2048:
        raise ValueError("output_reserve must be between 1 and 2047")
    budget_request = dict(request)
    budget_request["max_new"] = int(output_reserve)
    budget_request["query"] = (
        "请结合当前任务、最新状态和已经验证的工具结果处理这个请求；" * 160
    )
    budget_request["history"] = [
        {"role": "user", "content": "较早历史上下文" * 160},
        {"role": "assistant", "content": "较早答复" * 160},
    ]
    return budget_request


def _write_json(path: Path, value: Any) -> None:
    lifecycle.write_json(path, value)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    count = 0
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(contract.canonical_bytes(row) + b"\n")
            count += 1
    temporary.replace(path)
    return count


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return contract.load_jsonl(path)


def _resolved_input_path(value: str | Path) -> str:
    return str(resolve_repo_path(value).resolve())


def _input_identity_mismatches(
    observed: Mapping[str, str], expected: Mapping[str, str]
) -> dict[str, dict[str, str]]:
    """Compare immutable run inputs without confusing location with identity.

    Canonical model releases are independently copied, hash-locked instances of
    historical run files.  Moving an immutable file into the visible model
    release hierarchy must not invalidate an otherwise reusable run.  Directory
    inputs and scalar identifiers remain path/value exact.
    """

    mismatches: dict[str, dict[str, str]] = {}
    for key, expected_value in expected.items():
        observed_value = observed.get(key, "")
        if observed_value == expected_value:
            continue
        observed_path = Path(observed_value)
        expected_path = Path(expected_value)
        if (
            observed_path.is_file()
            and expected_path.is_file()
            and contract.sha_file(observed_path) == contract.sha_file(expected_path)
        ):
            continue
        mismatches[key] = {
            "observed": observed_value,
            "expected": expected_value,
        }
    return mismatches


def _source_manifest() -> dict[str, str]:
    # Hash complete executable source surfaces instead of a hand-picked list.
    # The earlier list omitted packed/model/CQ2 files even though Browser-WASM
    # results depended on them, which would have made two numerically different
    # runtimes appear to share one run fingerprint.
    roots_and_suffixes = (
        (Path(__file__).parent, {".py"}),
        (ROOT / "models/mei-1.0-51m/architecture", {".py", ".json"}),
        (ROOT / "platform/_shared/runtime", {".py", ".json"}),
        (ROOT / "platform/python-sdk/mei_sdk", {".py"}),
        (ROOT / "platform/_shared/rust/mei-sdk-core/src", {".rs"}),
        (ROOT / "platform/_shared/rust/mei-sdk-wasm/src", {".rs"}),
        (ROOT / "platform/browser-sdk", {".mjs"}),
        (ROOT / "platform/_shared/spec", {".json"}),
    )
    paths: set[Path] = {
        ROOT / "platform/_shared/Cargo.toml",
        ROOT / "platform/_shared/Cargo.lock",
        ROOT / "platform/_shared/.cargo/config.toml",
    }
    for source_root, suffixes in roots_and_suffixes:
        paths.update(
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and path.suffix in suffixes
            and "__pycache__" not in path.parts
        )
    return {
        str(path.relative_to(ROOT)): contract.sha_file(path)
        for path in sorted(paths)
    }


def _validate_qat_import(
    args: argparse.Namespace,
    base_identity: Mapping[str, Any],
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = contract.load_json(args.qat_import_receipt)
    master = resolve_repo_path(str((receipt.get("master") or {}).get("path") or ""))
    worker_path = resolve_repo_path(
        str((receipt.get("worker_receipt") or {}).get("path") or "")
    )
    base = receipt.get("base") or {}
    if (
        receipt.get("schema") != "mei-cq2-qat-import-candidate-receipt-v1"
        or receipt.get("status") != "passed"
        or receipt.get("quant_math_id") != "mei-cq-v2-g128-wht-codebook"
        or base.get("model_id") != base_identity["base_id"]
        or int(base.get("tokens_seen_exposure") or 0)
        != int(base_identity["tokens_seen_exposure"])
        or base.get("weights_sha256") != base_identity["weights_sha256"]
        or not master.is_file()
        or contract.sha_file(master) != (receipt.get("master") or {}).get("sha256")
        or not worker_path.is_file()
        or contract.sha_file(worker_path)
        != (receipt.get("worker_receipt") or {}).get("sha256")
    ):
        raise RuntimeError("CQ2-QAT seed receipt drifted or belongs to another Base")
    worker = contract.load_json(worker_path)
    worker_contracts = worker.get("contracts") or {}
    if (
        worker.get("terminal_status") != "passed"
        or worker.get("parent_id") != base_identity["base_id"]
        or worker.get("quant_math_id") != "mei-cq-v2-g128-wht-codebook"
        or not v4._qat_worker_contracts_compatible(
            worker_contracts,
            {
                name: str(contracts[name])
                for name in (
                    "weight_contract_sha256",
                    "runtime_profile_sha256",
                    "training_aux_sha256",
                )
            },
        )
    ):
        raise RuntimeError("CQ2-QAT seed violates the weight/training contract")
    return {
        "candidate_receipt": str(args.qat_import_receipt.resolve()),
        "candidate_receipt_sha256": contract.sha_file(args.qat_import_receipt),
        "master": str(master.resolve()),
        "master_sha256": contract.sha_file(master),
        "worker_receipt": str(worker_path.resolve()),
        "worker_receipt_sha256": contract.sha_file(worker_path),
        "source_runtime_profile_sha256": worker_contracts["runtime_profile_sha256"],
        "quant_math_id": receipt["quant_math_id"],
    }


def _verify_seed_prefix(seed_run: Path, args: argparse.Namespace) -> dict[str, Any]:
    seed_run = seed_run.resolve()
    parent_plan_path = seed_run / "plan.json"
    if not parent_plan_path.is_file():
        raise RuntimeError("adaptive continuation requires an immutable SFT-v4 seed plan")
    parent_plan = contract.load_json(parent_plan_path)
    parent_immutable = parent_plan.get("immutable") or {}
    evidence = v4._verify_training_prefix_run(seed_run, parent_immutable)
    base = parent_immutable.get("base") or {}
    qat = parent_immutable.get("qat_import") or {}
    expected = {
        "base_release": str(args.base_release.resolve()),
        "base_weights": str(args.base_weights.resolve()),
        "qat_receipt": str(args.qat_import_receipt.resolve()),
        "data_release": str(args.data_release.resolve()),
        "linguistic": str(args.linguistic_augmentation.resolve()),
        "eval_lock": str(args.eval_lock.resolve()),
        "narration": str(args.narration_release.resolve()),
    }
    observed = {
        "base_release": str(resolve_repo_path(str(base.get("release") or "")).resolve()),
        "base_weights": str(resolve_repo_path(str(base.get("weights") or "")).resolve()),
        "qat_receipt": str(
            resolve_repo_path(str(qat.get("candidate_receipt") or "")).resolve()
        ),
        "data_release": str(
            resolve_repo_path(
                str((parent_immutable.get("data_release") or {}).get("path") or "")
            ).resolve()
        ),
        "linguistic": str(
            resolve_repo_path(
                str(
                    (parent_immutable.get("linguistic_augmentation") or {}).get(
                        "path"
                    )
                    or ""
                )
            ).resolve()
        ),
        "eval_lock": str(
            resolve_repo_path(
                str((parent_immutable.get("eval_lock") or {}).get("path") or "")
            ).resolve()
        ),
        "narration": str(
            resolve_repo_path(
                str((parent_immutable.get("narration_release") or {}).get("path") or "")
            ).resolve()
        ),
    }
    mismatches = _input_identity_mismatches(observed, expected)
    if mismatches:
        raise RuntimeError(
            f"SFT-v4 seed inputs do not match selected Base/data: {mismatches}"
        )
    return {
        **evidence,
        "source_contracts": dict(parent_immutable.get("contracts") or {}),
        "runtime_contract_transition": {
            "from": (parent_immutable.get("contracts") or {}).get("runtime_profile_sha256"),
            "to": CURRENT_RUNTIME_PROFILE_SHA256,
            "weight_retraining_required": False,
            "downstream_alignment_replay_required": True,
        },
    }


def _verify_adaptive_prefix_run(
    source_run: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Verify an immutable adaptive-v5 prefix blocked only at downstream eval.

    The adoption boundary is deliberately after every LM-changing stage and
    final R2/index/view materialization.  It cannot adopt MW, confidence,
    narration, package, or runtime-gate outputs.
    """

    source_run = source_run.resolve()
    source_plan_path = source_run / "plan.json"
    if not source_plan_path.is_file():
        raise RuntimeError("adaptive-prefix source run has no immutable plan")
    source_plan = contract.load_json(source_plan_path)
    source_immutable = source_plan.get("immutable") or {}
    source_fingerprint = str(source_plan.get("run_fingerprint_sha256") or "")
    if (
        source_plan.get("schema") != PLAN_SCHEMA
        or len(source_fingerprint) != 64
        or source_immutable.get("runner_id") != RUNNER_ID
    ):
        raise RuntimeError("adaptive-prefix source plan identity is invalid")

    expected_inputs = {
        "base_release": _resolved_input_path(args.base_release),
        "base_weights": _resolved_input_path(args.base_weights),
        "qat_receipt": _resolved_input_path(args.qat_import_receipt),
        "seed_run": _resolved_input_path(args.seed_run),
        "data_release": _resolved_input_path(args.data_release),
        "linguistic": _resolved_input_path(args.linguistic_augmentation),
        "eval_lock": _resolved_input_path(args.eval_lock),
        "narration": _resolved_input_path(args.narration_release),
    }
    source_seed = source_immutable.get("training_prefix_adoption") or {}
    observed_inputs = {
        "base_release": str(
            resolve_repo_path(str((source_immutable.get("base") or {}).get("release") or "")).resolve()
        ),
        "base_weights": str(
            resolve_repo_path(str((source_immutable.get("base") or {}).get("weights") or "")).resolve()
        ),
        "qat_receipt": str(
            resolve_repo_path(str((source_immutable.get("qat_seed") or {}).get("candidate_receipt") or "")).resolve()
        ),
        "seed_run": str(
            resolve_repo_path(str(source_seed.get("parent_run_dir") or args.seed_run)).resolve()
        ),
        "data_release": str(
            resolve_repo_path(str((source_immutable.get("data_release") or {}).get("path") or "")).resolve()
        ),
        "linguistic": str(
            resolve_repo_path(
                str(
                    (source_immutable.get("linguistic_augmentation") or {}).get("path")
                    or ""
                )
            ).resolve()
        ),
        "eval_lock": str(
            resolve_repo_path(str((source_immutable.get("eval_lock") or {}).get("path") or "")).resolve()
        ),
        "narration": str(
            resolve_repo_path(str((source_immutable.get("narration_release") or {}).get("path") or "")).resolve()
        ),
    }
    mismatches = _input_identity_mismatches(observed_inputs, expected_inputs)
    if mismatches:
        raise RuntimeError(
            "adaptive-prefix source inputs differ from this productization: "
            f"{mismatches}"
        )

    expected_recipe = {
        "fullcall_alignment_steps": int(args.fullcall_alignment_steps),
        "agent_alignment_steps": int(args.agent_alignment_steps),
        "retrieval_r2_steps": int(args.retrieval_r2_steps),
    }
    source_recipe = source_immutable.get("recipe") or {}
    if any(source_recipe.get(name) != value for name, value in expected_recipe.items()):
        raise RuntimeError("adaptive-prefix LM/R2 recipe differs from this run")
    current_contracts = architecture_contracts()
    source_contracts = source_immutable.get("contracts") or {}
    for name in (
        "weight_contract_sha256",
        "runtime_profile_sha256",
        "training_aux_sha256",
    ):
        if source_contracts.get(name) != current_contracts.get(name):
            raise RuntimeError(f"adaptive-prefix architecture contract drifted: {name}")

    component_stages: dict[str, Any] = {}
    all_verified_outputs: dict[str, str] = {}
    for stage_id in ADAPTIVE_PREFIX_STAGES:
        receipt_path = source_run / "stages" / stage_id / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"adaptive-prefix receipt is missing: {stage_id}")
        receipt = contract.load_json(receipt_path)
        terminal = str(receipt.get("terminal_status") or "")
        if terminal not in {"passed", "degraded"}:
            raise RuntimeError(f"adaptive-prefix stage is not adoptable: {stage_id}={terminal}")
        if (
            receipt.get("stage_id") != stage_id
            or receipt.get("run_fingerprint_sha256") != source_fingerprint
        ):
            raise RuntimeError(f"adaptive-prefix receipt identity drifted: {stage_id}")
        verified_outputs: dict[str, str] = {}
        for raw_path, expected_sha in sorted((receipt.get("output_hashes") or {}).items()):
            artifact = resolve_repo_path(str(raw_path)).resolve()
            if not artifact.is_file() or contract.sha_file(artifact) != expected_sha:
                raise RuntimeError(f"adaptive-prefix output hash mismatch: {artifact}")
            verified_outputs[str(artifact)] = str(expected_sha)
            all_verified_outputs[str(artifact)] = str(expected_sha)
        if not verified_outputs:
            raise RuntimeError(f"adaptive-prefix stage has no verified output: {stage_id}")
        component_stages[stage_id] = {
            "stage_id": stage_id,
            "terminal_status": terminal,
            "stage_fingerprint_sha256": receipt.get("stage_fingerprint_sha256"),
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": contract.sha_file(receipt_path),
            "verified_outputs": verified_outputs,
        }

    blocked_path = source_run / "stages/adaptive_generation_eval_v5/receipt.json"
    blocked = contract.load_json(blocked_path)
    blocked_error = blocked.get("error") or {}
    if (
        blocked.get("terminal_status") != "blocked"
        or blocked.get("stage_id") != "adaptive_generation_eval_v5"
        or blocked.get("run_fingerprint_sha256") != source_fingerprint
        or "$.mw.source: value does not equal const"
        not in str(blocked_error.get("message") or "")
    ):
        raise RuntimeError("adaptive-prefix source is not blocked at the approved protocol boundary")

    parent_paths = _paths(source_run)
    required_files = {
        "agent": parent_paths["agent"],
        "r2": parent_paths["r2"],
        "calibration": parent_paths["calibration"],
        "index": parent_paths["index"],
        "mw_training_manifest": parent_paths["mw_training_views"] / "manifest.json",
        "mw_eval_manifest": parent_paths["mw_eval_views"] / "manifest.json",
    }
    artifacts: dict[str, Any] = {}
    for name, path in required_files.items():
        resolved = path.resolve()
        expected_sha = all_verified_outputs.get(str(resolved))
        if expected_sha is None or contract.sha_file(resolved) != expected_sha:
            raise RuntimeError(f"adaptive-prefix required artifact is not receipt-bound: {name}")
        artifacts[name] = {"path": str(resolved), "sha256": expected_sha}
    artifacts["mw_training_views"] = {
        "path": str(parent_paths["mw_training_views"].resolve()),
        "manifest_sha256": artifacts["mw_training_manifest"]["sha256"],
    }
    artifacts["mw_eval_views"] = {
        "path": str(parent_paths["mw_eval_views"].resolve()),
        "manifest_sha256": artifacts["mw_eval_manifest"]["sha256"],
    }

    value: dict[str, Any] = {
        "schema": "mei-adaptive-v5-prefix-adoption-evidence-v1",
        "parent_run_dir": str(source_run),
        "parent_run_fingerprint_sha256": source_fingerprint,
        "parent_plan_sha256": contract.sha_file(source_plan_path),
        "adopted_stage_ids": list(ADAPTIVE_PREFIX_STAGES),
        "component_stages": component_stages,
        "verified_outputs": all_verified_outputs,
        "artifacts": artifacts,
        "blocked_boundary": {
            "stage_id": "adaptive_generation_eval_v5",
            "receipt": str(blocked_path.resolve()),
            "receipt_sha256": contract.sha_file(blocked_path),
            "terminal_status": "blocked",
            "error": blocked_error,
        },
        "source_transition": {
            "parent_source_manifest_sha256": contract.sha_bytes(
                contract.canonical_bytes(source_immutable.get("source_manifest") or {})
            ),
            "current_source_manifest_sha256": contract.sha_bytes(
                contract.canonical_bytes(_source_manifest())
            ),
            "scope": "adaptive-eval-protocol-fix-and-recovery-control-plane",
            "lm_or_head_weights_changed": False,
        },
        "current_sha256": contract.sha_file(CURRENT_PATH),
        "current_unchanged": contract.sha_file(CURRENT_PATH)
        == source_immutable.get("current_baseline_sha256"),
    }
    if not value["current_unchanged"]:
        raise RuntimeError("CURRENT.json drifted from adaptive-prefix source baseline")
    value["adoption_input_fingerprint_sha256"] = contract.sha_bytes(
        contract.canonical_bytes(value)
    )
    return value


def _verify_packaged_run(source_run: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Adopt a receipt-complete adaptive-v5 package for current-source gates.

    This boundary is intentionally downstream of all model/head training and
    package export.  Every adopted output remains bound by its original stage
    receipt; the new run owns only Python/Browser execution and final audit. A
    historical source may be blocked at the approved loader bug, while a newer
    source may already have a passed/degraded runtime receipt that is being
    reevaluated after a gate-only source correction.
    """

    source_run = source_run.resolve()
    plan_path = source_run / "plan.json"
    if not plan_path.is_file():
        raise RuntimeError("packaged source run has no immutable plan")
    source_plan = contract.load_json(plan_path)
    immutable = source_plan.get("immutable") or {}
    fingerprint = str(source_plan.get("run_fingerprint_sha256") or "")
    if (
        source_plan.get("schema") != PLAN_SCHEMA
        or immutable.get("runner_id") != RUNNER_ID
        or len(fingerprint) != 64
    ):
        raise RuntimeError("packaged source plan identity is invalid")

    expected = {
        "base_release": _resolved_input_path(args.base_release),
        "base_weights": _resolved_input_path(args.base_weights),
        "qat_receipt": _resolved_input_path(args.qat_import_receipt),
        "seed_run": _resolved_input_path(args.seed_run),
        "data_release": _resolved_input_path(args.data_release),
        "linguistic": _resolved_input_path(args.linguistic_augmentation),
        "eval_lock": _resolved_input_path(args.eval_lock),
        "narration": _resolved_input_path(args.narration_release),
        "package_id": str(args.package_id),
    }
    seed = immutable.get("training_prefix_adoption") or {}
    observed = {
        "base_release": _resolved_input_path(str((immutable.get("base") or {}).get("release") or "")),
        "base_weights": _resolved_input_path(str((immutable.get("base") or {}).get("weights") or "")),
        "qat_receipt": _resolved_input_path(str((immutable.get("qat_seed") or {}).get("candidate_receipt") or "")),
        "seed_run": _resolved_input_path(str(seed.get("parent_run_dir") or "")),
        "data_release": _resolved_input_path(str((immutable.get("data_release") or {}).get("path") or "")),
        "linguistic": _resolved_input_path(str((immutable.get("linguistic_augmentation") or {}).get("path") or "")),
        "eval_lock": _resolved_input_path(str((immutable.get("eval_lock") or {}).get("path") or "")),
        "narration": _resolved_input_path(str((immutable.get("narration_release") or {}).get("path") or "")),
        "package_id": str(immutable.get("package_id") or ""),
    }
    mismatches = _input_identity_mismatches(observed, expected)
    if mismatches:
        raise RuntimeError(
            f"packaged source inputs differ from this run: {mismatches}"
        )
    source_contracts = immutable.get("contracts") or {}
    current_contracts = architecture_contracts()
    for name in (
        "weight_contract_sha256",
        "runtime_profile_sha256",
        "training_aux_sha256",
    ):
        if source_contracts.get(name) != current_contracts.get(name):
            raise RuntimeError(f"packaged source contract drifted: {name}")

    planned = [str(row.get("stage_id") or "") for row in source_plan.get("stages") or []]
    if planned[: len(PACKAGED_RUN_STAGES)] != list(PACKAGED_RUN_STAGES):
        raise RuntimeError("packaged source stage graph is not the approved recovery graph")
    components: dict[str, Any] = {}
    verified_outputs: dict[str, str] = {}
    for stage_id in PACKAGED_RUN_STAGES:
        receipt_path = source_run / "stages" / stage_id / "receipt.json"
        receipt = contract.load_json(receipt_path)
        terminal = str(receipt.get("terminal_status") or "")
        if (
            terminal not in {"passed", "degraded"}
            or receipt.get("stage_id") != stage_id
            or receipt.get("run_fingerprint_sha256") != fingerprint
        ):
            raise RuntimeError(f"packaged source stage is not adoptable: {stage_id}")
        outputs: dict[str, str] = {}
        for raw_path, expected_sha in sorted((receipt.get("output_hashes") or {}).items()):
            path = resolve_repo_path(str(raw_path)).resolve()
            if not path.is_file() or contract.sha_file(path) != expected_sha:
                raise RuntimeError(f"packaged source output hash mismatch: {path}")
            outputs[str(path)] = str(expected_sha)
            verified_outputs[str(path)] = str(expected_sha)
        if not outputs:
            raise RuntimeError(f"packaged source stage has no verified output: {stage_id}")
        components[stage_id] = {
            "stage_id": stage_id,
            "terminal_status": terminal,
            "stage_fingerprint_sha256": receipt.get("stage_fingerprint_sha256"),
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": contract.sha_file(receipt_path),
            "verified_outputs": outputs,
        }

    runtime_boundary_path = source_run / "stages/python_runtime_gate_v5/receipt.json"
    runtime_boundary_receipt = contract.load_json(runtime_boundary_path)
    runtime_terminal = str(runtime_boundary_receipt.get("terminal_status") or "")
    error = runtime_boundary_receipt.get("error") or {}
    if runtime_boundary_receipt.get("run_fingerprint_sha256") != fingerprint:
        raise RuntimeError("packaged source runtime boundary is not fingerprint-bound")
    if runtime_terminal == "blocked":
        if "'tuple' object has no attribute 'retrieval_calibration'" not in str(
            error.get("message") or ""
        ):
            raise RuntimeError(
                "packaged source is blocked outside the approved Python loader boundary"
            )
    elif runtime_terminal not in {"passed", "degraded"}:
        raise RuntimeError("packaged source runtime boundary is not adoptable")

    package_receipt = contract.load_json(
        source_run / "stages/package_v2_cq2_v5/receipt.json"
    )
    package_dir = resolve_repo_path(
        str((package_receipt.get("metrics") or {}).get("package_dir") or "")
    ).resolve()
    manifest = package_dir / "mei-model.json"
    manifest_sha = verified_outputs.get(str(manifest))
    if not package_dir.is_dir() or manifest_sha is None:
        raise RuntimeError("packaged source candidate is not receipt-bound")
    for path in package_dir.iterdir():
        if path.is_file() and str(path.resolve()) not in verified_outputs:
            raise RuntimeError(f"package file is not bound by export receipt: {path}")

    value: dict[str, Any] = {
        "schema": "mei-adaptive-v5-packaged-run-adoption-evidence-v1",
        "parent_run_dir": str(source_run),
        "parent_run_fingerprint_sha256": fingerprint,
        "parent_plan_sha256": contract.sha_file(plan_path),
        "adopted_stage_ids": list(PACKAGED_RUN_STAGES),
        "component_stages": components,
        "verified_outputs": verified_outputs,
        "package": {
            "path": str(package_dir),
            "manifest": str(manifest),
            "manifest_sha256": manifest_sha,
            "package_id": str(args.package_id),
        },
        "runtime_boundary": {
            "stage_id": "python_runtime_gate_v5",
            "receipt": str(runtime_boundary_path.resolve()),
            "receipt_sha256": contract.sha_file(runtime_boundary_path),
            "terminal_status": runtime_terminal,
            "error": error,
        },
        "source_transition": {
            "parent_source_manifest_sha256": contract.sha_bytes(
                contract.canonical_bytes(immutable.get("source_manifest") or {})
            ),
            "current_source_manifest_sha256": contract.sha_bytes(
                contract.canonical_bytes(_source_manifest())
            ),
            "scope": "current-source-runtime-gate-reevaluation",
            "lm_or_head_weights_changed": False,
            "package_bytes_changed": False,
        },
        "current_sha256": contract.sha_file(CURRENT_PATH),
        "current_unchanged": contract.sha_file(CURRENT_PATH)
        == immutable.get("current_baseline_sha256"),
    }
    if not value["current_unchanged"]:
        raise RuntimeError("CURRENT.json drifted from packaged source baseline")
    if runtime_terminal == "blocked":
        value["blocked_boundary"] = dict(value["runtime_boundary"])
    value["adoption_input_fingerprint_sha256"] = contract.sha_bytes(
        contract.canonical_bytes(value)
    )
    return value


def _validated_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if (
        getattr(args, "adopt_adaptive_prefix_run", None) is not None
        and getattr(args, "adopt_packaged_run", None) is not None
    ):
        raise RuntimeError(
            "adaptive-prefix and packaged-run adoption are mutually exclusive"
        )
    common = v4._validate_inputs(args, require_qat=False)
    contracts_raw = architecture_contracts()
    contracts = {
        name: str(contracts_raw[name])
        for name in (
            "weight_contract_sha256",
            "runtime_profile_sha256",
            "training_aux_sha256",
        )
    }
    if contracts["runtime_profile_sha256"] != CURRENT_RUNTIME_PROFILE_SHA256:
        raise RuntimeError("architecture runtime contract is not adaptive prompt-framing v5")
    qat = _validate_qat_import(args, common["base_identity"], contracts)
    seed = _verify_seed_prefix(args.seed_run, args)
    seed_contracts = seed.get("source_contracts") or {}
    if (
        seed_contracts.get("weight_contract_sha256")
        != contracts["weight_contract_sha256"]
        or seed_contracts.get("training_aux_sha256")
        != contracts["training_aux_sha256"]
    ):
        raise RuntimeError("seed run changed the 51M weight or training auxiliary contract")
    if qat["master_sha256"] != (
        contract.load_json(args.seed_run / "plan.json").get("immutable", {}).get("qat_import", {}).get("master_sha256")
    ):
        raise RuntimeError("seed run and selected QAT import disagree")
    adaptive_prefix = None
    if getattr(args, "adopt_adaptive_prefix_run", None) is not None:
        adaptive_prefix = _verify_adaptive_prefix_run(
            args.adopt_adaptive_prefix_run, args
        )
    packaged_run = None
    if getattr(args, "adopt_packaged_run", None) is not None:
        if adaptive_prefix is not None:
            raise RuntimeError(
                "adaptive-prefix and packaged-run adoption are mutually exclusive"
            )
        packaged_run = _verify_packaged_run(args.adopt_packaged_run, args)
    return {
        **common,
        "contracts": contracts,
        "qat_v5": qat,
        "seed_v5": seed,
        "adaptive_v5_prefix": adaptive_prefix,
        "packaged_v5_run": packaged_run,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _validated_inputs(args)
    base = inputs["base_identity"]
    execution_binding = phase_binding_identity()
    immutable = {
        "runner_id": RUNNER_ID,
        "product": contract.PRODUCT_ID,
        "package_id": args.package_id,
        "base": {
            "base_id": base["base_id"],
            "release": base["release"],
            "release_sha256": base["release_sha256"],
            "weights": base["weights"],
            "weights_sha256": base["weights_sha256"],
            "tokens_seen_exposure": base["tokens_seen_exposure"],
            "base_release_eligible": base["base_release_eligible"],
            "productization_experiment_eligible": base[
                "productization_experiment_eligible"
            ],
            "lineage_assurance": base["lineage_assurance"],
            "corpus_diversity_degraded": base["corpus_diversity_degraded"],
            "release_blockers": base["release_blockers"],
        },
        "contracts": inputs["contracts"],
        "data_release": {
            "path": str(args.data_release.resolve()),
            "release_id": inputs["data"]["release_id"],
            "release_fingerprint": inputs["data"]["release_fingerprint"],
            "manifest_sha256": contract.sha_file(args.data_release / "manifest.json"),
        },
        "linguistic_augmentation": {
            "path": str(args.linguistic_augmentation.resolve()),
            "release_id": inputs["linguistic"]["release_id"],
            "release_fingerprint": inputs["linguistic"]["release_fingerprint"],
            "manifest_sha256": contract.sha_file(
                args.linguistic_augmentation / "manifest.json"
            ),
        },
        "eval_lock": {
            "path": str(args.eval_lock.resolve()),
            "id": inputs["eval"]["id"],
            "evaluation_fingerprint": inputs["eval"]["evaluation_fingerprint"],
            "lock_sha256": contract.sha_file(args.eval_lock / "lock.json"),
        },
        "narration_release": {
            "path": str(args.narration_release.resolve()),
            "release_id": inputs["narration"]["release_id"],
            "manifest_sha256": contract.sha_file(
                args.narration_release / "manifest.json"
            ),
        },
        "qat_seed": inputs["qat_v5"],
        "training_prefix_adoption": inputs["seed_v5"],
        "tokenizer_sha256": contract.sha_file(TOKENIZER_ZH_V1),
        "current_baseline_sha256": contract.sha_file(CURRENT_PATH),
        "source_manifest": _source_manifest(),
        "recipe": {
            "fullcall_alignment_steps": int(args.fullcall_alignment_steps),
            "agent_alignment_steps": int(args.agent_alignment_steps),
            "retrieval_r2_steps": int(args.retrieval_r2_steps),
            "mw_steps": int(args.mw_steps),
            "confidence_train_limit": int(args.confidence_train_limit),
            "confidence_valid_limit": int(args.confidence_valid_limit),
            "confidence_steps": int(args.confidence_steps),
            "narration_steps": int(args.narration_steps),
            "alignment_mix": {
                "anchor_top5_standard": 0.25,
                "adaptive_standard": 0.25,
                "adaptive_compact": 0.50,
            },
            "seed": 51,
        },
        "runtime_policy": {
            "max_context_tokens": 2048,
            "default_output_reserve": 128,
            "profiles": {"compact": 1024, "standard": 1536},
            "candidate_batch_size": 5,
            "discard_threshold_configurable": True,
            "expand_threshold_configurable": True,
            "candidate_scan_not_external_tool_step": True,
            "prompt_framing_id": contract.PROMPT_FRAMING_ID,
            "assistant_suffix": contract.ASSISTANT_SUFFIX,
        },
        "validation_scope": {
            "required": ["python", "browser-wasm"],
            "deferred": ["c", "node-public", "rust-public"],
        },
        "failure_policy": "complete-independent-stages-and-mark-release-ineligible",
    }
    if execution_binding is not None:
        immutable["cycle_id"] = execution_binding["cycle_id"]
    if inputs["packaged_v5_run"] is not None:
        immutable["packaged_run_adoption"] = inputs["packaged_v5_run"]
        immutable["stage_graph_mode"] = "adopt-verified-packaged-run"
    elif inputs["adaptive_v5_prefix"] is not None:
        immutable["adaptive_prefix_adoption"] = inputs["adaptive_v5_prefix"]
        immutable["stage_graph_mode"] = "adopt-verified-adaptive-prefix"
    else:
        immutable["stage_graph_mode"] = "full-adaptive-productization"
    run_fingerprint = contract.sha_bytes(contract.canonical_bytes(immutable))
    stages: list[dict[str, Any]] = []
    adaptive_prefix = inputs["adaptive_v5_prefix"]
    packaged_run = inputs["packaged_v5_run"]
    if packaged_run is not None:
        active_stages = (
            PACKAGED_RUN_ADOPTION_STAGE,
            *PACKAGED_RUN_DOWNSTREAM_STAGES,
        )
        predecessor = packaged_run["adoption_input_fingerprint_sha256"]
    elif adaptive_prefix is not None:
        active_stages = (ADAPTIVE_PREFIX_ADOPTION_STAGE, *ADAPTIVE_DOWNSTREAM_STAGES)
        predecessor = adaptive_prefix["adoption_input_fingerprint_sha256"]
    else:
        active_stages = STAGES
        predecessor = base["weights_sha256"]
    for index, stage in enumerate(active_stages, 1):
        fingerprint = contract.sha_bytes(
            contract.canonical_bytes(
                {
                    "run_fingerprint": run_fingerprint,
                    "stage": stage,
                    "index": index,
                    "predecessor": predecessor,
                }
            )
        )
        stages.append(
            {
                "index": index,
                "stage_id": stage,
                "stage_fingerprint_sha256": fingerprint,
                "predecessor_fingerprint_sha256": predecessor,
            }
        )
        predecessor = fingerprint
    live = lifecycle.live_cpt_workers()
    plan = {
        "schema": PLAN_SCHEMA,
        "run_fingerprint_sha256": run_fingerprint,
        "immutable": immutable,
        "stages": stages,
        "live_cpt_workers": live,
        "heavy_execution_deferred": bool(live),
        "process_complete": False,
        "release_eligible": False,
    }
    if execution_binding is not None:
        plan["execution_binding"] = execution_binding
    return plan


def _with_training_bank(
    rows: Sequence[dict[str, Any]], bank: str | None = None
) -> list[dict[str, Any]]:
    return v4._with_training_bank(list(rows), bank)


def _catalogs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    deploy = v4._catalog(args.eval_lock)
    training_doc = contract.load_json(args.data_release / "training-tool-universe.json")
    training_catalog = [
        contract.compact_tool(tool) for tool in training_doc.get("tools") or []
    ]
    holdout_doc = contract.load_json(
        args.eval_lock / "schema-holdout-tool-universe.json"
    )
    holdout = [contract.compact_tool(tool) for tool in holdout_doc.get("tools") or []]
    if len(deploy) != 147 or len(training_catalog) != 211 or len(holdout) != 32:
        raise RuntimeError("adaptive v5 requires 147 deploy, 211 training and 32 holdout tools")
    return deploy, training_catalog, [*deploy, *holdout]


def _data_rows(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    data = args.data_release
    linguistic = args.linguistic_augmentation
    return {
        "fullcall_train": [
            *_with_training_bank(_load_rows(data / "full-call.train.jsonl"), "structural"),
            *_with_training_bank(_load_rows(data / "schema-full-call.train.jsonl"), "schema"),
            *_with_training_bank(
                _load_rows(linguistic / "linguistic-full-call.train.jsonl")
            ),
        ],
        "fullcall_valid": [
            *_load_rows(data / "full-call.valid.jsonl"),
            *_load_rows(data / "schema-full-call.valid.jsonl"),
            *_load_rows(linguistic / "linguistic-full-call.valid.jsonl"),
        ],
        "agent_train": _load_rows(data / "agent-continuation.train.jsonl"),
        "agent_valid": _load_rows(data / "agent-continuation.valid.jsonl"),
        "retrieval_train": [
            *_with_training_bank(_load_rows(data / "retrieval.train.jsonl"), "structural"),
            *_with_training_bank(_load_rows(data / "schema-retrieval.train.jsonl"), "schema"),
            *_with_training_bank(
                _load_rows(linguistic / "linguistic-retrieval.train.jsonl")
            ),
        ],
        "retrieval_valid": [
            *_load_rows(data / "retrieval.valid.jsonl"),
            *_load_rows(data / "schema-retrieval.valid.jsonl"),
            *_load_rows(linguistic / "linguistic-retrieval.valid.jsonl"),
        ],
        "mw_train": _load_rows(data / "mw-disposition.train.jsonl"),
        "mw_valid": _load_rows(data / "mw-disposition.valid.jsonl"),
        "mw_dev": _load_rows(args.eval_lock / "mw.dev.jsonl"),
        "mw_test": _load_rows(args.eval_lock / "mw.test.jsonl"),
        "confidence_train": _load_rows(data / "confidence-harvest.train.jsonl"),
        "confidence_valid": _load_rows(data / "confidence-harvest.valid.jsonl"),
        "confidence_dev": [
            *_load_rows(args.eval_lock / "confidence.dev.jsonl"),
            *_load_rows(args.eval_lock / "schema-confidence.dev.jsonl"),
        ],
        "confidence_test": [
            *_load_rows(args.eval_lock / "confidence.test.jsonl"),
            *_load_rows(args.eval_lock / "schema-confidence.test.jsonl"),
        ],
    }


def _release_runtime(runtime: Any) -> None:
    v4._release_runtime(runtime)
    gc.collect()


def _load_runtime(master: Path, **kwargs: Any) -> Any:
    return v4._load_runtime(master, **kwargs)


def _set_retrieval_calibration(runtime: Any, calibration: Mapping[str, Any]) -> None:
    runtime.retrieval_calibration = {
        "calibration_id": str(calibration["calibration_id"]),
        "scale": float(calibration["scale"]),
        "bias": float(calibration["bias"]),
        "discard_threshold": float(calibration["discard_threshold"]),
        "expand_threshold": float(calibration["expand_threshold"]),
        "validated": bool(calibration["validated"]),
    }


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "seed_calibration": run_dir / "stages/seed_adaptive_views_v5/retrieval-calibration.json",
        "seed_alignment": run_dir / "stages/seed_adaptive_views_v5/alignment-views",
        "fullcall": run_dir / "stages/fullcall_alignment_replay_v5/fullcall-aligned-master.npz",
        "agent": run_dir / "stages/agent_alignment_replay_v5/agent-aligned-master.npz",
        "r2": run_dir / "stages/retrieval_r2_v5/retrieval-r2.npz",
        "calibration": run_dir / "stages/final_adaptive_artifacts_v5/retrieval-calibration.json",
        "index": run_dir / "stages/final_adaptive_artifacts_v5/tool-index-r2.json",
        "mw_training_views": run_dir / "stages/final_adaptive_artifacts_v5/mw-training-visible-batches",
        "mw_eval_views": run_dir / "stages/final_adaptive_artifacts_v5/mw-eval-visible-batches",
        "mw": run_dir / "stages/mw_disposition_v5/mw-disposition.npz",
        "confidence_train": run_dir / "stages/confidence_harvest_v5/train-outcomes.jsonl",
        "confidence_valid": run_dir / "stages/confidence_harvest_v5/valid-outcomes.jsonl",
        "confidence_dev": run_dir / "stages/confidence_harvest_v5/dev-outcomes.jsonl",
        "confidence": run_dir / "stages/confidence_head_v5/confidence.npz",
        "confidence_calibration": run_dir / "stages/confidence_head_v5/confidence-calibration.json",
        "narration": run_dir / "stages/narration_adapter_v5/narration-adapter.npz",
    }


def _artifact_inputs(plan: Mapping[str, Any]) -> dict[str, Any]:
    immutable = plan["immutable"]
    adoption = immutable["training_prefix_adoption"]
    return {
        "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
        "seed_parent_run_fingerprint_sha256": adoption[
            "parent_run_fingerprint_sha256"
        ],
        "seed_agent_sha256": adoption["artifacts"]["final_lm_master"]["sha256"],
        "seed_retrieval_sha256": adoption["artifacts"]["retrieval_r1_head"]["sha256"],
        "seed_index_sha256": adoption["artifacts"]["tool_index"]["sha256"],
        "data_release_fingerprint": immutable["data_release"]["release_fingerprint"],
        "runtime_profile_sha256": immutable["contracts"]["runtime_profile_sha256"],
    }


def _load_frozen_rows(paths: Iterable[Path], *, slim: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if slim:
                    row.pop("full_ranking", None)
                    row.pop("sink", None)
                    row.pop("ordinary", None)
                rows.append(row)
    return rows


def preflight(args: argparse.Namespace, plan: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    if contract.sha_file(CURRENT_PATH) != CURRENT_BASELINE_SHA256:
        raise RuntimeError("CURRENT.json drifted from the protected baseline")
    if _source_manifest() != plan["immutable"]["source_manifest"]:
        raise RuntimeError("adaptive productization source drifted after planning")
    if lifecycle.live_cpt_workers():
        raise RuntimeError("a live CPT worker owns Metal; adaptive productization is deferred")
    adaptive_adoption = plan["immutable"].get("adaptive_prefix_adoption")
    if adaptive_adoption is not None:
        observed = _verify_adaptive_prefix_run(args.adopt_adaptive_prefix_run, args)
        if contract.canonical_bytes(observed) != contract.canonical_bytes(
            adaptive_adoption
        ):
            raise RuntimeError("adaptive-prefix adoption evidence drifted after planning")
    packaged_adoption = plan["immutable"].get("packaged_run_adoption")
    if packaged_adoption is not None:
        observed = _verify_packaged_run(args.adopt_packaged_run, args)
        if contract.canonical_bytes(observed) != contract.canonical_bytes(
            packaged_adoption
        ):
            raise RuntimeError("packaged-run adoption evidence drifted after planning")
    deploy, training_catalog, schema_catalog = _catalogs(args)
    rows = _data_rows(args)
    expected_counts = {
        "fullcall_train": 5_255,
        "agent_train": 3_736,
        "retrieval_train": 3_787,
        "retrieval_valid": 1_138,
        "mw_train": 13_763,
        "mw_valid": 2_705,
        "mw_dev": 1_000,
        "mw_test": 1_000,
        "confidence_train": 5_255,
        "confidence_valid": 2_020,
    }
    actual = {name: len(rows[name]) for name in expected_counts}
    if actual != expected_counts:
        raise RuntimeError(f"frozen SFT-v4 row inventory drifted: {actual}")
    if args.fullcall_alignment_steps != 2_000 or args.agent_alignment_steps != 1_000:
        raise RuntimeError("adaptive v5 replay budgets are frozen at 2000/1000")
    if args.retrieval_r2_steps != 1_600 or args.mw_steps != 2_000:
        raise RuntimeError("adaptive v5 R2/MW budgets are frozen at 1600/2000")
    if args.confidence_steps != 800 or args.narration_steps != 1_200:
        raise RuntimeError("adaptive v5 confidence/narration budgets are frozen at 800/1200")
    tokenizer = freeze_sft_v3_release._load_tokenizer()
    from mei_sdk.shared import minimum_tool_projection_tokens

    eligibility: dict[str, Any] = {}
    for profile, cap in (("compact", 1024), ("standard", 1536)):
        minimums = [
            minimum_tool_projection_tokens(
                tokenizer, tool, stable_overhead=contract.TASK_CONTRACT + "\n"
            )
            for tool in training_catalog
        ]
        eligibility[profile] = {
            "cap": cap,
            "eligible": sum(value <= cap for value in minimums),
            "ineligible": sum(value > cap for value in minimums),
            "max_minimum_tokens": max(minimums),
        }
    if any(value["ineligible"] for value in eligibility.values()):
        raise RuntimeError(f"registered training tool is profile-ineligible: {eligibility}")
    receipt = {
        "schema": "mei-51m-adaptive-productization-preflight-v1",
        "status": "passed",
        "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
        "counts": actual,
        "catalog_counts": {
            "deploy": len(deploy),
            "training": len(training_catalog),
            "schema_eval": len(schema_catalog),
        },
        "profile_eligibility": eligibility,
        "joint_context_contract": {
            "max_context_tokens": 2048,
            "prompt_cap_at_default_output": 1920,
            "assistant_suffix_included": True,
        },
        "seed_verified": True,
        "cpt_retrained": False,
        "qat_retrained": False,
        "adaptive_prefix_adopted": adaptive_adoption is not None,
        "adaptive_prefix_parent_run": (
            adaptive_adoption["parent_run_dir"]
            if adaptive_adoption is not None
            else None
        ),
        "packaged_run_adopted": packaged_adoption is not None,
        "packaged_parent_run": (
            packaged_adoption["parent_run_dir"]
            if packaged_adoption is not None
            else None
        ),
        "current_sha256": contract.sha_file(CURRENT_PATH),
    }
    _write_json(run_dir / "preflight.json", receipt)
    return receipt


def _balanced_frozen_views(paths: Sequence[Path], limit: int | None) -> list[dict[str, Any]]:
    groups = [_load_frozen_rows([path], slim=True) for path in paths]
    if limit is None:
        return [row for group in groups for row in group]
    output: list[dict[str, Any]] = []
    cursor = 0
    while len(output) < int(limit):
        progressed = False
        for group in groups:
            if cursor < len(group):
                output.append(group[cursor])
                progressed = True
                if len(output) >= int(limit):
                    break
        if not progressed:
            break
        cursor += 1
    return output


def _evaluate_adaptive_generation(
    runtime: Any,
    source_rows: Sequence[dict[str, Any]],
    catalog: Sequence[dict[str, Any]],
    calibration: Mapping[str, Any],
    *,
    limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate fixed-five paging without conflating it with the MW head.

    This pre-MW diagnostic uses the gold only to label whether an empty first
    batch is a capability miss and should expose a later batch.  It measures
    LM/retrieval behavior; the actual MW-governed runtime loop is evaluated
    after the independent 20-class head is trained.
    """

    from mei_sdk.protocol import render_budgeted_request
    from mei_sdk.shared import plan_candidate_batches, validate_generated_call

    selected_sources = list(source_rows)[: min(int(limit), len(source_rows))]
    predictions: list[dict[str, Any]] = []
    exact = fixed_first_exact = retrieval_hits = 0
    total_batches = budget_errors = 0
    for index, source in enumerate(selected_sources):
        query = str(source.get("query") or "")
        expected = list(source.get("answers") or [])
        if source.get("answers") is None and source.get("gold_name"):
            expected = [
                {
                    "name": source["gold_name"],
                    "arguments": source.get("gold_args") or {},
                }
            ]
        gold_name = str((expected[0] or {}).get("name") or "") if expected else ""
        ranked = runtime.search_ranked(query, list(catalog))
        expand = float(calibration["expand_threshold"])
        if not calibration.get("validated"):
            expand = float(calibration["discard_threshold"])
        plan = plan_candidate_batches(
            ranked,
            discard_threshold=float(calibration["discard_threshold"]),
            expand_threshold=expand,
        )
        queue = [candidate for batch in plan.batches for candidate in batch]
        retrieval_hits += int(
            not gold_name or gold_name in {candidate.tool_id for candidate in plan.candidates}
        )
        final_calls: list[dict[str, Any]] = []
        terminal = "retrieval_no_match" if not queue else "candidate_exhausted"
        batch_records: list[dict[str, Any]] = []
        profile = "compact" if index % 2 == 0 else "standard"
        while queue:
            requested = list(queue[:5])
            del queue[: len(requested)]
            request = training.deployment_request_v3(source)
            request["mw"] = dict(EVALUATION_MW_OVERRIDE)
            rendered = render_budgeted_request(
                request,
                [candidate.schema for candidate in requested],
                runtime.tokenizer,
                relevances=[candidate.relevance for candidate in requested],
                runtime_profile=profile,
                output_reserve=128,
            )
            names = list(rendered.get("selected_tools") or [])
            if len(names) < len(requested):
                queue = requested[len(names) :] + queue
            if rendered.get("error") or not names:
                budget_errors += 1
                terminal = "context_unrepresentable"
                break
            tools = list(rendered["_validation_tools"])
            sink_ids = runtime.tokenizer.encode(
                rendered["sink"], add_bos=True, add_eos=False
            )
            ordinary_ids = runtime.tokenizer.encode(
                rendered["ordinary"], add_bos=False, add_eos=False
            )
            decoded = runtime.greedy(
                sink_ids + ordinary_ids,
                tools=tools,
                max_new=96,
                decode_mode="constrained",
                sink_ids=sink_ids,
                ordinary_ids=ordinary_ids,
                runtime_profile=profile,
            )
            validated = validate_generated_call(
                str(decoded.get("text") or ""),
                tools=tools,
                request={**rendered["request"], "mw": dict(EVALUATION_MW_OVERRIDE)},
                confidence=None,
                enforce_confidence=False,
            )
            calls = list(validated.get("function_calls") or [])
            total_batches += 1
            batch_records.append(
                {
                    "batch": len(batch_records) + 1,
                    "tools": names,
                    "prompt_tokens": rendered["prompt_tokens"],
                    "schema_budget": rendered["schema_budget"],
                    "input_budget": rendered["input_budget"],
                    "schema_projection_sha256": rendered[
                        "schema_projection_sha256"
                    ],
                    "text": decoded.get("text"),
                    "validation_error": validated.get("error"),
                }
            )
            if calls:
                final_calls = calls
                terminal = "call"
                break
            if str(decoded.get("text") or "").strip() == "[]" and gold_name and gold_name not in names:
                terminal = "capability_insufficient"
                continue
            terminal = "model_no_call" if not calls else "call"
            break
        row_exact = training._calls_equal(final_calls, expected)
        exact += int(row_exact)
        fixed_first_exact += int(row_exact and len(batch_records) <= 1)
        predictions.append(
            {
                "sample_id": source.get("sample_id"),
                "expected_calls": expected,
                "predicted_calls": final_calls,
                "exact": row_exact,
                "runtime_profile": profile,
                "terminal": terminal,
                "batches": batch_records,
            }
        )
        if index == 0 or (index + 1) % 25 == 0:
            _emit(
                "adaptive_eval_progress",
                step=index + 1,
                total=len(selected_sources),
                exact=exact,
                scanned_batches=total_batches,
            )
    count = len(selected_sources)
    return (
        {
            "rows": count,
            "exact": exact,
            "exact_rate": exact / count if count else 0.0,
            "fixed_first_exact": fixed_first_exact,
            "fixed_first_exact_rate": fixed_first_exact / count if count else 0.0,
            "adaptive_minus_fixed_first": (
                (exact - fixed_first_exact) / count if count else 0.0
            ),
            "retrieval_eligible_hit_rate": retrieval_hits / count if count else 0.0,
            "scanned_batches": total_batches,
            "mean_batches": total_batches / count if count else 0.0,
            "context_unrepresentable": budget_errors,
            "evaluation_boundary": "pre-mw-generation-and-retrieval-only",
        },
        predictions,
    )


def execute(args: argparse.Namespace, plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if plan["heavy_execution_deferred"] or lifecycle.live_cpt_workers():
        raise RuntimeError("live CPT owns Metal; adaptive v5 execution is deferred")
    from common.checkpoint import save_params
    from training.qat.cq2_qat_51m import explicit_group_map

    deploy_catalog, training_catalog, schema_eval_catalog = _catalogs(args)
    training_tools = {str(tool["name"]): tool for tool in training_catalog}
    rows = _data_rows(args)
    paths = _paths(run_dir)
    seed_adoption = plan["immutable"]["training_prefix_adoption"]
    adaptive_adoption = plan["immutable"].get("adaptive_prefix_adoption")
    packaged_adoption = plan["immutable"].get("packaged_run_adoption")
    full_replay = adaptive_adoption is None and packaged_adoption is None
    seed_agent = Path(seed_adoption["artifacts"]["final_lm_master"]["path"])
    seed_r1 = Path(seed_adoption["artifacts"]["retrieval_r1_head"]["path"])
    seed_index = Path(seed_adoption["artifacts"]["tool_index"]["path"])
    if adaptive_adoption is not None:
        adopted_artifacts = adaptive_adoption["artifacts"]
        paths.update(
            {
                "agent": Path(adopted_artifacts["agent"]["path"]),
                "r2": Path(adopted_artifacts["r2"]["path"]),
                "calibration": Path(adopted_artifacts["calibration"]["path"]),
                "index": Path(adopted_artifacts["index"]["path"]),
                "mw_training_views": Path(
                    adopted_artifacts["mw_training_views"]["path"]
                ),
                "mw_eval_views": Path(adopted_artifacts["mw_eval_views"]["path"]),
            }
        )
    receipts: dict[str, dict[str, Any]] = {}

    def run(name: str, action: Any) -> dict[str, Any]:
        phase_scope = getattr(args, "phase_scope", None)
        mode = phase_stage_mode(phase_scope, name)
        execution_binding = plan.get("execution_binding")
        stage_binding_path = run_dir / "stages" / name / "phase-binding.json"
        expected_stage_binding = (
            {"schema": "mei-51m-stage-phase-binding-v1", **execution_binding}
            if phase_scope and execution_binding is not None and mode == "execute"
            else None
        )
        if mode == "skip":
            _emit("stage_skipped", stage=name, phase_scope=phase_scope)
            return {
                "stage_id": name,
                "terminal_status": "skipped_by_phase_scope",
            }
        if mode == "reuse_only":
            receipt_path = run_dir / "stages" / name / "receipt.json"
            if not receipt_path.is_file():
                raise RuntimeError(
                    f"{phase_scope} cannot execute non-phase stage {name}; "
                    f"reusable receipt missing: {receipt_path}"
                )

            def forbidden_action(_: Path):
                raise RuntimeError(
                    f"{phase_scope} cannot execute non-phase stage {name}; "
                    "a reusable upstream receipt is required"
                )

            action = forbidden_action
        if (
            expected_stage_binding is not None
            and (run_dir / "stages" / name / "receipt.json").is_file()
        ):
            if not stage_binding_path.is_file():
                raise RuntimeError(
                    f"phase-bound stage lacks binding evidence: {stage_binding_path}"
                )
            if contract.load_json(stage_binding_path) != expected_stage_binding:
                raise RuntimeError(
                    f"phase-bound stage binding changed: {stage_binding_path}"
                )
        if _source_manifest() != plan["immutable"]["source_manifest"]:
            raise RuntimeError("adaptive v5 source drifted during execution")
        if lifecycle.live_cpt_workers():
            raise RuntimeError(f"live CPT reclaimed Metal before {name}")
        _emit("stage_start", stage=name, run_dir=str(run_dir))
        try:
            receipt = lifecycle.run_stage(
                run_dir=run_dir,
                plan=plan,
                name=name,
                action=action,
                resume=args.resume,
            )
        except Exception as exc:
            _emit(
                "stage_blocked",
                stage=name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        receipts[name] = receipt
        if expected_stage_binding is not None and not stage_binding_path.is_file():
            _write_json(stage_binding_path, expected_stage_binding)
        _emit(
            "stage_complete",
            stage=name,
            terminal_status=receipt["terminal_status"],
            reused=bool(receipt.get("reused")),
        )
        if getattr(args, "stop_after_stage", None) == name:
            raise PhaseBoundaryReached(name, receipts)
        return receipt

    def adoption_action(directory: Path):
        observed = _verify_seed_prefix(args.seed_run, args)
        if contract.canonical_bytes(observed) != contract.canonical_bytes(
            seed_adoption
        ):
            raise RuntimeError("SFT-v4 training-prefix evidence drifted after planning")
        report = {
            "schema": "mei-adaptive-v5-training-prefix-adoption-receipt-v1",
            "status": "passed",
            "parent_run": str(args.seed_run.resolve()),
            "parent_run_fingerprint_sha256": seed_adoption[
                "parent_run_fingerprint_sha256"
            ],
            "adopted_stage_ids": seed_adoption["stage_ids"],
            "artifacts": seed_adoption["artifacts"],
            "runtime_contract_transition": seed_adoption[
                "runtime_contract_transition"
            ],
            "cpt_retrained": False,
            "qat_retrained": False,
            "parent_run_mutated": False,
        }
        path = directory / "adoption.json"
        _write_json(path, report)
        return report, [
            path,
            seed_agent,
            seed_r1,
            seed_index,
        ]

    if full_replay:
        run("adopt_training_prefix_v4", adoption_action)
    elif adaptive_adoption is not None:
        def adaptive_prefix_adoption_action(directory: Path):
            observed = _verify_adaptive_prefix_run(
                args.adopt_adaptive_prefix_run, args
            )
            if contract.canonical_bytes(observed) != contract.canonical_bytes(
                adaptive_adoption
            ):
                raise RuntimeError(
                    "adaptive-v5 prefix evidence drifted after planning"
                )
            degraded_stages = sorted(
                stage_id
                for stage_id, evidence in adaptive_adoption[
                    "component_stages"
                ].items()
                if evidence["terminal_status"] == "degraded"
            )
            report = {
                "schema": "mei-adaptive-v5-prefix-adoption-receipt-v1",
                "status": "passed",
                "parent_run": adaptive_adoption["parent_run_dir"],
                "parent_run_fingerprint_sha256": adaptive_adoption[
                    "parent_run_fingerprint_sha256"
                ],
                "adopted_stage_ids": adaptive_adoption["adopted_stage_ids"],
                "component_stages": adaptive_adoption["component_stages"],
                "blocked_boundary": adaptive_adoption["blocked_boundary"],
                "source_transition": adaptive_adoption["source_transition"],
                "degraded_adopted_stages": degraded_stages,
                "degraded": bool(degraded_stages),
                "cpt_retrained": False,
                "qat_retrained": False,
                "lm_replayed": False,
                "retrieval_r2_retrained": False,
                "parent_run_mutated": False,
            }
            path = directory / "adoption.json"
            _write_json(path, report)
            verified = [
                Path(value)
                for value in adaptive_adoption["verified_outputs"]
            ]
            return report, [path, *verified]

        run(ADAPTIVE_PREFIX_ADOPTION_STAGE, adaptive_prefix_adoption_action)
    else:
        def packaged_run_adoption_action(directory: Path):
            observed = _verify_packaged_run(args.adopt_packaged_run, args)
            if contract.canonical_bytes(observed) != contract.canonical_bytes(
                packaged_adoption
            ):
                raise RuntimeError("packaged-run evidence drifted after planning")
            degraded_stages = sorted(
                stage_id
                for stage_id, evidence in packaged_adoption[
                    "component_stages"
                ].items()
                if evidence["terminal_status"] == "degraded"
            )
            report = {
                "schema": "mei-adaptive-v5-packaged-run-adoption-receipt-v1",
                "status": "passed",
                "parent_run": packaged_adoption["parent_run_dir"],
                "parent_run_fingerprint_sha256": packaged_adoption[
                    "parent_run_fingerprint_sha256"
                ],
                "adopted_stage_ids": packaged_adoption["adopted_stage_ids"],
                "component_stages": packaged_adoption["component_stages"],
                "package": packaged_adoption["package"],
                "runtime_boundary": packaged_adoption["runtime_boundary"],
                "blocked_boundary": packaged_adoption.get("blocked_boundary"),
                "source_transition": packaged_adoption["source_transition"],
                "degraded_adopted_stages": degraded_stages,
                "degraded": bool(degraded_stages),
                "cpt_retrained": False,
                "qat_retrained": False,
                "lm_or_head_retrained": False,
                "package_rebuilt": False,
                "parent_run_mutated": False,
            }
            path = directory / "adoption.json"
            _write_json(path, report)
            verified = [
                Path(value)
                for value in packaged_adoption["verified_outputs"]
            ]
            return report, [path, *verified]

        run(PACKAGED_RUN_ADOPTION_STAGE, packaged_run_adoption_action)

    def seed_views_action(directory: Path):
        runtime = _load_runtime(
            seed_agent,
            quantized=True,
            retrieval_head=seed_r1,
            tool_index=seed_index,
        )
        calibration, evidence = adaptive.calibrate_retrieval_v1(
            runtime,
            rows["retrieval_valid"],
            training_catalog,
            model_sha256=contract.sha_file(seed_agent),
            retrieval_head_sha256=contract.sha_file(seed_r1),
            tokenizer_sha256=contract.sha_file(TOKENIZER_ZH_V1),
        )
        _set_retrieval_calibration(runtime, calibration)
        _write_json(paths["seed_calibration"], calibration)
        calibration_evidence = directory / "retrieval-calibration-validation.jsonl"
        _write_jsonl(calibration_evidence, evidence)
        alignment_manifest = adaptive.freeze_alignment_views(
            runtime,
            {
                "full_call": {"train": rows["fullcall_train"]},
                "agent_continuation": {"train": rows["agent_train"]},
            },
            training_catalog,
            calibration,
            paths["seed_alignment"],
            input_artifacts=_artifact_inputs(plan),
            row_limits={
                "full_call.train": args.fullcall_alignment_steps,
                "agent_continuation.train": args.agent_alignment_steps,
            },
        )
        report = {
            "status": "passed",
            "schema": "mei-seed-adaptive-views-stage-v1",
            "calibration": calibration,
            "alignment_release_fingerprint_sha256": alignment_manifest[
                "release_fingerprint_sha256"
            ],
            "alignment_coverage": alignment_manifest["coverage"],
            "degraded": not bool(calibration["validated"]),
            "fallback_scan_all_eligible": bool(
                calibration["fallback_scan_all_eligible"]
            ),
        }
        report_path = directory / "seed-adaptive-views.json"
        _write_json(report_path, report)
        outputs = [
            paths["seed_calibration"],
            calibration_evidence,
            report_path,
            *sorted(path for path in paths["seed_alignment"].rglob("*") if path.is_file()),
        ]
        _release_runtime(runtime)
        return report, outputs

    if full_replay:
        run("seed_adaptive_views_v5", seed_views_action)

    def fullcall_alignment_action(directory: Path):
        replay_rows = _load_rows(paths["seed_alignment"] / "full_call.train.jsonl")
        if len(replay_rows) != args.fullcall_alignment_steps:
            raise RuntimeError("frozen Full-call alignment view count drifted")
        runtime = _load_runtime(seed_agent, quantized=False)
        group_map = explicit_group_map(runtime.model.parameters())
        report = training.train_lm_sft_v3(
            runtime,
            replay_rows,
            training_tools,
            steps=args.fullcall_alignment_steps,
            lr=2e-4,
            group_map=group_map,
            activation_ste=True,
            sampler=training.FULLCALL_SAMPLER_ID,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            stage_id="fullcall-alignment-v5",
        )
        save_params(runtime.model, paths["fullcall"])
        report.update(
            {
                "status": "passed",
                "runtime_shared_prompt_framing": True,
                "assistant_suffix": contract.ASSISTANT_SUFFIX,
                "cpt_retrained": False,
                "qat_retrained": False,
            }
        )
        report_path = directory / "fullcall-alignment.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["fullcall"], report_path]

    if full_replay:
        run("fullcall_alignment_replay_v5", fullcall_alignment_action)

    def agent_alignment_action(directory: Path):
        replay_rows = _load_rows(
            paths["seed_alignment"] / "agent_continuation.train.jsonl"
        )
        if len(replay_rows) != args.agent_alignment_steps:
            raise RuntimeError("frozen Agent alignment view count drifted")
        runtime = _load_runtime(paths["fullcall"], quantized=False)
        group_map = explicit_group_map(runtime.model.parameters())
        report = training.train_lm_sft_v3(
            runtime,
            replay_rows,
            training_tools,
            steps=args.agent_alignment_steps,
            lr=1e-4,
            group_map=group_map,
            activation_ste=True,
            sampler=training.AGENT_SAMPLER_ID,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            stage_id="agent-alignment-v5",
        )
        save_params(runtime.model, paths["agent"])
        report.update(
            {
                "status": "passed",
                "runtime_shared_prompt_framing": True,
                "candidate_batches_are_not_external_steps": True,
            }
        )
        report_path = directory / "agent-alignment.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["agent"], report_path]

    if full_replay:
        run("agent_alignment_replay_v5", agent_alignment_action)

    def retrieval_r2_action(directory: Path):
        runtime = _load_runtime(paths["agent"], quantized=True)
        report = training.train_retrieval_v3(
            runtime,
            rows["retrieval_train"],
            training_tools,
            steps=args.retrieval_r2_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
            stage_id="r2-v5",
        )
        lifecycle._save_model_or_head(runtime.contrastive, paths["r2"])
        report.update(
            {
                "status": "passed",
                "head_role": "retrieval_relevance",
                "independent_from_mw_and_execution_confidence": True,
            }
        )
        report_path = directory / "retrieval-r2.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["r2"], report_path]

    if full_replay:
        run("retrieval_r2_v5", retrieval_r2_action)

    def final_artifacts_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r2"],
            tool_index=seed_index,
        )
        calibration, evidence = adaptive.calibrate_retrieval_v1(
            runtime,
            rows["retrieval_valid"],
            training_catalog,
            model_sha256=contract.sha_file(paths["agent"]),
            retrieval_head_sha256=contract.sha_file(paths["r2"]),
            tokenizer_sha256=contract.sha_file(TOKENIZER_ZH_V1),
        )
        _set_retrieval_calibration(runtime, calibration)
        _write_json(paths["calibration"], calibration)
        evidence_path = directory / "retrieval-calibration-validation.jsonl"
        _write_jsonl(evidence_path, evidence)
        index_report = training.finalize_tool_index_v3(
            runtime,
            deploy_catalog,
            paths["index"],
            model_sha256=contract.sha_file(paths["agent"]),
            head_sha256=contract.sha_file(paths["r2"]),
            tokenizer_sha256=contract.sha_file(TOKENIZER_ZH_V1),
        )
        training_manifest = adaptive.freeze_mw_visible_batches(
            runtime,
            {"train": rows["mw_train"], "valid": rows["mw_valid"]},
            training_catalog,
            calibration,
            paths["mw_training_views"],
            input_artifacts=_artifact_inputs(plan),
        )
        # The deploy/eval catalog is separate, so evaluation views can never
        # silently inherit training-only schemas.
        eval_manifest = adaptive.freeze_mw_visible_batches(
            runtime,
            {"dev": rows["mw_dev"], "test": rows["mw_test"]},
            deploy_catalog,
            calibration,
            paths["mw_eval_views"],
            input_artifacts={
                **_artifact_inputs(plan),
                "deploy_catalog_sha256": contract.sha_bytes(
                    contract.canonical_bytes(deploy_catalog)
                ),
            },
        )
        report = {
            "status": "passed",
            "schema": "mei-final-adaptive-artifacts-stage-v1",
            "calibration": calibration,
            "tool_index": index_report,
            "mw_training_release_fingerprint_sha256": training_manifest[
                "release_fingerprint_sha256"
            ],
            "mw_eval_release_fingerprint_sha256": eval_manifest[
                "release_fingerprint_sha256"
            ],
            "mw_trainer_live_retrieval_allowed": False,
            "degraded": not bool(calibration["validated"]),
        }
        report_path = directory / "final-adaptive-artifacts.json"
        _write_json(report_path, report)
        outputs = [
            paths["calibration"],
            evidence_path,
            paths["index"],
            report_path,
            *sorted(
                path
                for root in (paths["mw_training_views"], paths["mw_eval_views"])
                for path in root.rglob("*")
                if path.is_file()
            ),
        ]
        _release_runtime(runtime)
        return report, outputs

    if full_replay:
        run("final_adaptive_artifacts_v5", final_artifacts_action)

    def adaptive_eval_action(directory: Path):
        calibration = contract.load_json(paths["calibration"])
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r2"],
            tool_index=paths["index"],
        )
        _set_retrieval_calibration(runtime, calibration)
        per_bank = max(1, args.eval_limit // 3)
        reports: dict[str, dict[str, Any]] = {}
        prediction_paths: list[Path] = []
        for split in ("dev", "test"):
            reports[split] = {}
            for name, filename, bank_catalog in (
                ("structural", f"fullcall.{split}.jsonl", deploy_catalog),
                ("natural", f"natural-fullcall.{split}.jsonl", deploy_catalog),
                (
                    "schema_holdout",
                    f"schema-fullcall.{split}.jsonl",
                    schema_eval_catalog,
                ),
            ):
                bank_rows = _load_rows(args.eval_lock / filename)
                report, predictions = _evaluate_adaptive_generation(
                    runtime,
                    bank_rows,
                    bank_catalog,
                    calibration,
                    limit=min(len(bank_rows), per_bank),
                )
                reports[split][name] = report
                prediction_path = directory / f"{name}.{split}.predictions.jsonl"
                _write_jsonl(prediction_path, predictions)
                prediction_paths.append(prediction_path)
        context_errors = sum(
            int(report["context_unrepresentable"])
            for split_reports in reports.values()
            for report in split_reports.values()
        )
        adaptive_worse = any(
            float(report["adaptive_minus_fixed_first"]) < -0.01
            for split_reports in reports.values()
            for report in split_reports.values()
        )
        report = {
            "status": "passed",
            "schema": "mei-adaptive-generation-eval-v1",
            "splits": reports,
            "threshold_tuning_splits": ["retrieval.valid"],
            "locked_test_used_for_threshold_tuning": False,
            "joint_budget_errors": context_errors,
            "adaptive_vs_fixed_first_drop_within_one_point": not adaptive_worse,
            "quality_scope": "retrieval-and-generation-before-mw-confidence",
            "degraded": context_errors != 0 or adaptive_worse,
        }
        report_path = directory / "adaptive-generation-eval.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [report_path, *prediction_paths]

    if packaged_adoption is None:
        run("adaptive_generation_eval_v5", adaptive_eval_action)

    def mw_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r2"],
            tool_index=paths["index"],
        )
        _set_retrieval_calibration(runtime, contract.load_json(paths["calibration"]))
        train_paths = sorted(paths["mw_training_views"].glob("mw-visible.train.*.jsonl"))
        train_views = _load_frozen_rows(train_paths, slim=True)
        train_report = training.train_mw_disposition_v3(
            runtime,
            train_views,
            {},
            [],
            steps=args.mw_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
        )
        lifecycle._save_model_or_head(runtime.mw_disposition_head, paths["mw"])
        eval_reports: dict[str, Any] = {}
        prediction_outputs: list[Path] = []
        for split in ("dev", "test"):
            split_paths = sorted(
                paths["mw_eval_views"].glob(f"mw-visible.{split}.*.jsonl")
            )
            split_views = _balanced_frozen_views(split_paths, args.mw_eval_limit)
            split_report, predictions = evaluation.evaluate_frozen_mw_batches(
                runtime,
                split_views,
                limit=None,
            )
            eval_reports[split] = split_report
            prediction_path = directory / f"mw-frozen.{split}.predictions.jsonl"
            _write_jsonl(prediction_path, predictions)
            prediction_outputs.append(prediction_path)
        cells = {
            (
                int(row["effective_reason_class_id"]),
                str(row["retrieval_mode"]),
                str(row["runtime_profile"]),
            )
            for row in train_views
            if row.get("mw_eligible") is True
        }
        expected_cells = {
            (label, mode, profile)
            for label in range(20)
            for mode in ("oracle", "learned")
            for profile in ("compact", "standard")
        }
        missing_cells = sorted(expected_cells - cells)
        floor = v4._metric_floors(args.eval_lock)["mw_disposition"]["release_floor"]
        gates = {
            split: evaluation.threshold_status(split_report, floor)
            for split, split_report in eval_reports.items()
        }
        report = {
            "status": "passed",
            "schema": "mei-mw-disposition-batched-stage-v1",
            "train": train_report,
            "eval": eval_reports,
            "gates": gates,
            "missing_class_mode_profile_cells": [list(value) for value in missing_cells],
            "retrieval_relevance_head": "independent",
            "execution_confidence_head": "independent",
            "mw_disposition_classes": 20,
            "trainer_live_retrieval_calls": 0,
            "old_600_step_checkpoint_reused": False,
            "locked_test_used_for_training_or_thresholds": False,
            "degraded": not all(gate["passed"] for gate in gates.values())
            or bool(missing_cells),
        }
        report_path = directory / "mw-disposition-v5.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["mw"], report_path, *prediction_outputs]

    if packaged_adoption is None:
        run("mw_disposition_v5", mw_action)

    def confidence_harvest_action(directory: Path):
        calibration = contract.load_json(paths["calibration"])
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r2"],
            tool_index=paths["index"],
            mw_head=paths["mw"],
        )
        _set_retrieval_calibration(runtime, calibration)
        train_outcomes, train_receipt = training.harvest_confidence_outcomes_v5(
            runtime,
            rows["confidence_train"],
            training_tools,
            training_catalog,
            limit=args.confidence_train_limit,
            strict_class_floor=False,
        )
        valid_outcomes, valid_receipt = training.harvest_confidence_outcomes_v5(
            runtime,
            rows["confidence_valid"],
            training_tools,
            training_catalog,
            limit=args.confidence_valid_limit,
            strict_class_floor=False,
        )
        schema_eval_tools = {str(tool["name"]): tool for tool in schema_eval_catalog}
        dev_outcomes, dev_receipt = training.harvest_confidence_outcomes_v5(
            runtime,
            rows["confidence_dev"],
            schema_eval_tools,
            schema_eval_catalog,
            limit=len(rows["confidence_dev"]),
            minimum_class_rows=1,
            strict_class_floor=False,
        )
        _write_jsonl(paths["confidence_train"], train_outcomes)
        _write_jsonl(paths["confidence_valid"], valid_outcomes)
        _write_jsonl(paths["confidence_dev"], dev_outcomes)
        floor_passed = bool(
            train_receipt["head_eligible_class_floor_passed"]
            and valid_receipt["head_eligible_class_floor_passed"]
        )
        report = {
            "status": "passed",
            "schema": "mei-confidence-adaptive-harvest-stage-v1",
            "train": train_receipt,
            "valid": valid_receipt,
            "dev": dev_receipt,
            "actual_runtime_pipeline": True,
            "confidence_enforcement_during_harvest": False,
            "mw_is_independent_input_gate": True,
            "degraded": not floor_passed,
        }
        report_path = directory / "confidence-harvest-v5.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [
            paths["confidence_train"],
            paths["confidence_valid"],
            paths["confidence_dev"],
            report_path,
        ]

    if packaged_adoption is None:
        run("confidence_harvest_v5", confidence_harvest_action)

    def confidence_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r2"],
            tool_index=paths["index"],
        )
        train_outcomes = _load_rows(paths["confidence_train"])
        valid_outcomes = _load_rows(paths["confidence_valid"])
        train_counts = {
            label: sum(
                int(row.get("head_eligible") is not False and int(row["label"]) == label)
                for row in train_outcomes
            )
            for label in (0, 1)
        }
        valid_counts = {
            label: sum(
                int(row.get("head_eligible") is not False and int(row["label"]) == label)
                for row in valid_outcomes
            )
            for label in (0, 1)
        }
        minimum = 100 if min(*train_counts.values(), *valid_counts.values()) >= 100 else 1
        if min(*train_counts.values(), *valid_counts.values()) < 1:
            raise RuntimeError(
                f"actual runtime produced no binary confidence coverage: {train_counts}/{valid_counts}"
            )
        train_report, confidence_calibration = training.train_confidence_v3(
            runtime,
            train_outcomes,
            valid_outcomes,
            steps=args.confidence_steps,
            minimum_class_rows=minimum,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
        )
        lifecycle._save_model_or_head(runtime.conf_v2, paths["confidence"])
        _write_json(
            paths["confidence_calibration"],
            {
                "score_contract": training.CONFIDENCE_SCORE_ID,
                **confidence_calibration,
            },
        )
        floor = v4._metric_floors(args.eval_lock)["confidence"]["release_floor"]
        gate = evaluation.threshold_status(train_report["valid_metrics"], floor)
        report = {
            "status": "passed",
            "schema": "mei-confidence-adaptive-stage-v1",
            "train": train_report,
            "gate": gate,
            "requested_minimum_class_rows": 100,
            "effective_minimum_class_rows": minimum,
            "train_class_counts": train_counts,
            "valid_class_counts": valid_counts,
            "degraded": minimum < 100 or not gate["passed"],
        }
        report_path = directory / "confidence-v5.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [
            paths["confidence"],
            paths["confidence_calibration"],
            report_path,
        ]

    if packaged_adoption is None:
        run("confidence_head_v5", confidence_action)

    def narration_action(directory: Path):
        from training.tool_use.train_sft_ondisk_51m import train_narration_adapter

        runtime = _load_runtime(paths["agent"], quantized=True)
        report = train_narration_adapter(
            runtime,
            _load_rows(args.narration_release / "narration.train.jsonl"),
            _load_rows(args.narration_release / "narration.valid.jsonl"),
            steps=args.narration_steps,
            checkpoint_dir=directory / "checkpoints",
            resume=args.resume,
        )
        lifecycle._save_model_or_head(runtime.narration_adapter, paths["narration"])
        report.update(
            {
                "status": "passed",
                "rank": 16,
                "terminal_only": True,
                "backbone_frozen": True,
            }
        )
        report_path = directory / "narration-adapter-v5.json"
        _write_json(report_path, report)
        _release_runtime(runtime)
        return report, [paths["narration"], report_path]

    if packaged_adoption is None:
        run("narration_adapter_v5", narration_action)

    def sidecar_eval_action(directory: Path):
        runtime = _load_runtime(
            paths["agent"],
            quantized=True,
            retrieval_head=paths["r2"],
            tool_index=paths["index"],
            mw_head=paths["mw"],
            confidence_head=paths["confidence"],
            narration_head=paths["narration"],
        )
        _set_retrieval_calibration(runtime, contract.load_json(paths["calibration"]))
        confidence_document = contract.load_json(paths["confidence_calibration"])
        confidence_dev, confidence_dev_predictions = evaluation.evaluate_confidence(
            runtime,
            _load_rows(paths["confidence_dev"]),
            {
                "scale": float(confidence_document["scale"]),
                "bias": float(confidence_document["bias"]),
            },
            candidates=rows["confidence_dev"],
            minimum_class_rows=1,
        )
        narration_dev, narration_dev_predictions = evaluation.evaluate_narration(
            runtime,
            _load_rows(args.eval_lock / "narration.dev.jsonl"),
            limit=args.narration_eval_limit,
        )
        schema_eval_tools = {str(tool["name"]): tool for tool in schema_eval_catalog}
        confidence_test_outcomes, confidence_test_harvest = (
            training.harvest_confidence_outcomes_v5(
                runtime,
                rows["confidence_test"],
                schema_eval_tools,
                schema_eval_catalog,
                limit=len(rows["confidence_test"]),
                minimum_class_rows=1,
                strict_class_floor=False,
            )
        )
        confidence_test, confidence_test_predictions = evaluation.evaluate_confidence(
            runtime,
            confidence_test_outcomes,
            {
                "scale": float(confidence_document["scale"]),
                "bias": float(confidence_document["bias"]),
            },
            candidates=rows["confidence_test"],
            minimum_class_rows=1,
        )
        narration_test, narration_test_predictions = evaluation.evaluate_narration(
            runtime,
            _load_rows(args.eval_lock / "narration.test.jsonl"),
            limit=args.narration_eval_limit,
        )
        floors = v4._metric_floors(args.eval_lock)
        confidence_gates = {
            split: evaluation.threshold_status(
                report, floors["confidence"]["release_floor"]
            )
            for split, report in {
                "dev": confidence_dev,
                "test": confidence_test,
            }.items()
        }
        narration_gates = {
            split: evaluation.threshold_status(
                report, floors["narration"]["release_floor"]
            )
            for split, report in {
                "dev": narration_dev,
                "test": narration_test,
            }.items()
        }
        report = {
            "status": "passed",
            "schema": "mei-adaptive-sidecar-runtime-eval-v1",
            "confidence": {"dev": confidence_dev, "test": confidence_test},
            "confidence_test_harvest": confidence_test_harvest,
            "narration": {"dev": narration_dev, "test": narration_test},
            "gates": {
                "confidence": confidence_gates,
                "narration": narration_gates,
            },
            "narration_terminal_only": True,
            "head_boundaries": {
                "retrieval_relevance": "candidate-set-only",
                "mw_disposition": "independent-20class-policy",
                "execution_confidence": "validated-call-execution-only",
                "narration": "terminal-result-generation-only",
            },
            "locked_test_used_for_training_or_thresholds": False,
            "degraded": not all(
                gate["passed"]
                for family in (confidence_gates, narration_gates)
                for gate in family.values()
            ),
        }
        report_path = directory / "sidecar-runtime-eval.json"
        confidence_dev_path = directory / "confidence.dev.predictions.jsonl"
        confidence_test_outcomes_path = directory / "confidence.test.outcomes.jsonl"
        confidence_test_path = directory / "confidence.test.predictions.jsonl"
        narration_dev_path = directory / "narration.dev.predictions.jsonl"
        narration_test_path = directory / "narration.test.predictions.jsonl"
        _write_json(report_path, report)
        _write_jsonl(confidence_dev_path, confidence_dev_predictions)
        _write_jsonl(confidence_test_outcomes_path, confidence_test_outcomes)
        _write_jsonl(confidence_test_path, confidence_test_predictions)
        _write_jsonl(narration_dev_path, narration_dev_predictions)
        _write_jsonl(narration_test_path, narration_test_predictions)
        _release_runtime(runtime)
        return report, [
            report_path,
            confidence_dev_path,
            confidence_test_outcomes_path,
            confidence_test_path,
            narration_dev_path,
            narration_test_path,
        ]

    if packaged_adoption is None:
        run("sidecar_runtime_eval_v5", sidecar_eval_action)

    package_dir = (
        Path(packaged_adoption["package"]["path"])
        if packaged_adoption is not None
        else run_dir / "candidate" / args.package_id
    )

    def component_evidence(stage: str) -> dict[str, Any]:
        if stage not in receipts and adaptive_adoption is not None:
            adopted = adaptive_adoption["component_stages"].get(stage)
            if adopted is None:
                raise RuntimeError(f"adopted component stage is missing: {stage}")
            return {
                "stage_id": stage,
                "stage_fingerprint_sha256": adopted[
                    "stage_fingerprint_sha256"
                ],
                "status": "passed",
                "terminal_status": adopted["terminal_status"],
                "source_run_fingerprint_sha256": adaptive_adoption[
                    "parent_run_fingerprint_sha256"
                ],
                "adopted_by_stage": ADAPTIVE_PREFIX_ADOPTION_STAGE,
                "receipt_sha256": adopted["receipt_sha256"],
            }
        receipt = receipts[stage]
        return {
            "stage_id": stage,
            "stage_fingerprint_sha256": receipt["stage_fingerprint_sha256"],
            "status": "passed",
            "terminal_status": receipt["terminal_status"],
            "source_run_fingerprint_sha256": plan["run_fingerprint_sha256"],
        }

    def package_action(directory: Path):
        from release.pack_cq2_v2_51m import export as export_cq2
        from training.tool_use.train_sft_ondisk_51m import save_product_heads

        runtime = _load_runtime(
            paths["agent"],
            quantized=False,
            retrieval_head=paths["r2"],
            mw_head=paths["mw"],
            confidence_head=paths["confidence"],
            narration_head=paths["narration"],
        )
        heads_path = directory / "product-heads.npz"
        heads_report = save_product_heads(runtime, heads_path)
        evidence = {
            component: component_evidence(stage)
            for component, stage in {
                "lm": "agent_alignment_replay_v5",
                "contrastive": "retrieval_r2_v5",
                "mw_disposition": "mw_disposition_v5",
                "confidence": "confidence_head_v5",
                "narration_adapter": "narration_adapter_v5",
            }.items()
        }
        evidence_path = directory / "stage-evidence.json"
        _write_json(evidence_path, evidence)
        export_report = export_cq2(
            argparse.Namespace(
                master=paths["agent"],
                heads=heads_path,
                tool_index=paths["index"],
                retrieval_calibration=paths["calibration"],
                stage_evidence=evidence_path,
                out_dir=package_dir,
                package_id=args.package_id,
                parent_package_id=plan["immutable"]["base"]["base_id"],
            )
        )
        report = {
            "status": "passed",
            "schema": "mei-adaptive-package-v5-stage-v1",
            **export_report,
            "heads": heads_report,
            "runtime_profile_sha256": CURRENT_RUNTIME_PROFILE_SHA256,
            "weight_parameter_count": 51_463_797,
            "degraded": not bool(export_report["package_within_limit"]),
        }
        report_path = directory / "package-v2-cq2-v5.json"
        _write_json(report_path, report)
        outputs = [
            heads_path,
            evidence_path,
            report_path,
            *sorted(path for path in package_dir.iterdir() if path.is_file()),
        ]
        _release_runtime(runtime)
        return report, outputs

    if packaged_adoption is None:
        run("package_v2_cq2_v5", package_action)

    def python_gate_action(directory: Path):
        from mei_sdk.engine import Engine
        from mei_sdk.package import load_package
        from mei_sdk.runtime_51m import complete_51m, load_51m_runtime

        package = load_package(package_dir)
        capabilities = package.capabilities()
        runtime, load_report = load_51m_runtime(package, backend="mlx-fused")
        engine = Engine(
            package,
            runtime=runtime,
            load_report=load_report,
            backend="mlx-fused",
        )
        registration = engine.register_tools(deploy_catalog)
        engine_capabilities = engine.capabilities()
        probe = rows["confidence_dev"][0]
        request = training.deployment_request_v3(probe)
        request["catalog"] = deploy_catalog
        request["max_new"] = 1
        result = complete_51m(runtime, request, enforce_confidence=False)
        no_budget_exception = result.get("error") != "tool_schema_budget_exceeded"
        output_reserve = int(
            plan["immutable"]["runtime_policy"]["default_output_reserve"]
        )
        prompt_cap = int(
            plan["immutable"]["runtime_policy"]["max_context_tokens"]
        ) - output_reserve
        budget_request = _joint_budget_stress_request(
            request, output_reserve=output_reserve
        )
        budget_result = complete_51m(
            runtime, budget_request, enforce_confidence=False
        )
        speed_request = dict(request)
        speed_request["max_new"] = 32
        speed_result = complete_51m(
            runtime, speed_request, enforce_confidence=False
        )
        speed_tokens = int(speed_result.get("output_tokens") or 0)
        speed_decode_ms = float(
            (speed_result.get("timings") or {}).get("decode_ms") or 0.0
        )
        warm_decode_tok_s = (
            speed_tokens / (speed_decode_ms / 1000.0)
            if speed_tokens > 0 and speed_decode_ms > 0
            else None
        )
        warm_decode_ok = bool(
            speed_tokens >= 8
            and warm_decode_tok_s is not None
            and warm_decode_tok_s >= 280.0
        )
        rss_probe = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        rss_kib = (
            int(rss_probe.stdout.strip())
            if rss_probe.returncode == 0 and rss_probe.stdout.strip().isdigit()
            else None
        )
        prompt_within_contract = bool(
            isinstance(budget_result.get("prompt_tokens"), int)
            and int(budget_result["prompt_tokens"]) <= prompt_cap
        )
        budget_planning_ok = bool(
            budget_result.get("error") != "tool_schema_budget_exceeded"
            and prompt_within_contract
            and len(budget_result.get("selected_tools") or []) <= 5
        )
        registration_ok = bool(
            registration.get("registered") == len(deploy_catalog)
            and not any(registration.get("profile_ineligible", {}).values())
        )
        head_boundaries_ok = bool(
            capabilities.get("heads", {}).get("contrastive")
            and capabilities.get("heads", {}).get("mw_disposition")
            and capabilities.get("heads", {}).get("confidence")
            and capabilities.get("heads", {}).get("narration_adapter")
        )
        report = {
            "status": "passed",
            "schema": "mei-python-adaptive-runtime-gate-v1",
            "package_hash_verified": capabilities["hash_verified"],
            "tensor_identity_complete": capabilities["tensor_identity_complete"],
            "runtime_profile": capabilities["runtime_profile"],
            "retrieval_calibration": capabilities["retrieval_calibration"],
            "load_report": load_report,
            "engine_capabilities": engine_capabilities,
            "tool_registration": registration,
            "probe": {
                "error": result.get("error"),
                "selected_tools": result.get("selected_tools"),
                "scanned_batches": len(
                    (result.get("retrieval") or {}).get("scanned_batches") or []
                ),
                "prompt_tokens": result.get("prompt_tokens"),
                "schema_budget": result.get("schema_budget"),
                "input_budget": result.get("input_budget"),
            },
            "joint_budget_stress": {
                "error": budget_result.get("error"),
                "prompt_tokens": budget_result.get("prompt_tokens"),
                "selected_tool_count": len(
                    budget_result.get("selected_tools") or []
                ),
                "schema_budget": budget_result.get("schema_budget"),
                "input_budget": budget_result.get("input_budget"),
                "schema_projection_sha256": budget_result.get(
                    "schema_projection_sha256"
                ),
                "output_reserve_tokens": output_reserve,
                "prompt_cap_tokens": prompt_cap,
                "prompt_within_1920": prompt_within_contract,
                "budget_planning_ok": budget_planning_ok,
            },
            "warm_decode": {
                "output_tokens": speed_tokens,
                "decode_ms": speed_decode_ms,
                "tokens_per_second": warm_decode_tok_s,
                "target_tokens_per_second": 280.0,
                "target_validated": warm_decode_ok,
                "backend": "mlx-fused",
            },
            "process_rss": {
                "rss_kib": rss_kib,
                "rss_bytes": rss_kib * 1024 if rss_kib is not None else None,
                "measurement": "ps-current-process-after-package-load-and-warm-probes",
            },
            "public_engine_contract": {
                "register_tools": registration_ok,
                "create_session": engine_capabilities.get("stateful_tool_loop")
                is True,
                "fixed_five_candidate_batch": engine_capabilities.get(
                    "candidate_batch_size"
                )
                == 5,
                "candidate_batch_scan": engine_capabilities.get(
                    "candidate_batch_scan"
                )
                is True,
                "head_roles_separate": head_boundaries_ok,
            },
            "tool_schema_budget_exception": not no_budget_exception,
            "degraded": not (
                capabilities["hash_verified"]
                and capabilities["tensor_identity_complete"]
                and no_budget_exception
                and budget_planning_ok
                and registration_ok
                and head_boundaries_ok
                and warm_decode_ok
            ),
        }
        report_path = directory / "python-runtime-gate.json"
        _write_json(report_path, report)
        engine.close()
        _release_runtime(runtime)
        return report, [report_path]

    run("python_runtime_gate_v5", python_gate_action)

    def browser_wasm_action(directory: Path):
        browser_driver = ROOT / "model-factory/evaluation/resources/smoke_wasm_q4_51m.py"
        smoke = subprocess.run(
            [
                sys.executable,
                str(browser_driver),
                "--package-dir",
                str(package_dir),
                "--jobs-dir",
                str(directory),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        smoke_log = directory / "wasm-smoke.log"
        smoke_log.write_text(smoke.stdout, encoding="utf-8")
        node_report = directory / "wasm-q4-smoke.json"
        browser_report = directory / "wasm-q4-browser-smoke.json"
        node_measured = contract.load_json(node_report) if node_report.is_file() else {}
        measured = (
            contract.load_json(browser_report) if browser_report.is_file() else {}
        )
        functional = bool(
            smoke.returncode == 0
            and measured.get("ok")
            and measured.get("browser_worker") is True
            and measured.get("numeric_forward_ran") is True
            and measured.get("bounded_int8_cache_ran") is True
        )
        report = _browser_gate_metrics(
            measured,
            functional=functional,
            build_returncode=0 if node_measured.get("compiled") else smoke.returncode,
            smoke_returncode=smoke.returncode,
        )
        report.update(
            {
                "actual_chromium_worker": measured.get("browser_worker") is True,
                "numeric_forward_ran": measured.get("numeric_forward_ran") is True,
                "bounded_int8_cache_ran": measured.get("bounded_int8_cache_ran") is True,
                "node_wasm_diagnostic_ok": node_measured.get("ok") is True,
                "browser_measurement": measured,
                "node_measurement": node_measured,
            }
        )
        report_path = directory / "browser-wasm-gate.json"
        _write_json(report_path, report)
        outputs = [smoke_log, report_path]
        if node_report.is_file():
            outputs.append(node_report)
        if browser_report.is_file():
            outputs.append(browser_report)
        return report, outputs

    run("browser_wasm_gate_v5", browser_wasm_action)

    def final_audit_action(directory: Path):
        active_adoption = packaged_adoption or adaptive_adoption
        stage_statuses = {
            name: receipt["terminal_status"] for name, receipt in receipts.items()
        }
        adopted_stage_statuses = (
            {
                name: evidence["terminal_status"]
                for name, evidence in active_adoption[
                    "component_stages"
                ].items()
            }
            if active_adoption is not None
            else {}
        )
        package_manifest = contract.load_json(package_dir / "mei-model.json")
        python_gate = receipts["python_runtime_gate_v5"]["metrics"]
        browser_gate = receipts["browser_wasm_gate_v5"]["metrics"]
        release_reasons: list[str] = []
        if any(
            status == "degraded"
            for status in (*stage_statuses.values(), *adopted_stage_statuses.values())
        ):
            release_reasons.append("one_or_more_quality_or_runtime_gates_degraded")
        if int((package_manifest.get("resources") or {}).get("package_bytes") or 0) > 18 * 1024 * 1024:
            release_reasons.append("package_size")
        if not browser_gate.get("functional_complete"):
            release_reasons.append("browser_wasm_functional_gate_failed")
        if not browser_gate.get("warm_steady_decode_100_tok_s_validated"):
            release_reasons.append(
                "browser_wasm_warm_decode_100_tok_s_not_validated"
            )
        if not browser_gate.get("heap_within_96_mib"):
            release_reasons.append("browser_wasm_heap_96_mib_not_validated")
        if not (python_gate.get("warm_decode") or {}).get("target_validated"):
            release_reasons.append("python_mlx_warm_decode_280_tok_s_not_validated")
        report = {
            "status": "passed",
            "schema": "mei-51m-adaptive-productization-final-audit-v1",
            "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
            "base_id": plan["immutable"]["base"]["base_id"],
            "base_exposure_tokens": plan["immutable"]["base"][
                "tokens_seen_exposure"
            ],
            "stage_statuses": stage_statuses,
            "adopted_stage_statuses": adopted_stage_statuses,
            "adoption_mode": plan["immutable"]["stage_graph_mode"],
            "adoption_parent_run": (
                active_adoption["parent_run_dir"]
                if active_adoption is not None
                else None
            ),
            "superseded_runtime_boundary": (
                active_adoption.get("runtime_boundary")
                or active_adoption.get("blocked_boundary")
                if active_adoption is not None
                else None
            ),
            "superseded_blocked_boundary": (
                active_adoption.get("blocked_boundary")
                if active_adoption is not None
                else None
            ),
            "process_complete": True,
            "release_eligible": not release_reasons,
            "release_ineligible_reasons": sorted(set(release_reasons)),
            "weight_contract_sha256": plan["immutable"]["contracts"][
                "weight_contract_sha256"
            ],
            "runtime_profile_sha256": plan["immutable"]["contracts"][
                "runtime_profile_sha256"
            ],
            "weight_parameter_count": 51_463_797,
            "mtp_tensor_count": sum(
                int("mtp" in str(row.get("name") or "").lower())
                for row in (package_manifest.get("tensor_container") or {}).get(
                    "directory", []
                )
            ),
            "current_sha256": contract.sha_file(CURRENT_PATH),
            "current_unchanged": contract.sha_file(CURRENT_PATH)
            == plan["immutable"]["current_baseline_sha256"],
            "current_modified": False,
        }
        report_path = directory / "final-audit.json"
        _write_json(report_path, report)
        return report, [report_path]

    run("final_audit_v5", final_audit_action)
    final = receipts["final_audit_v5"]["metrics"]
    active_adoption = packaged_adoption or adaptive_adoption
    return {
        "schema": PROGRESS_SCHEMA,
        "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
        "run_dir": str(run_dir),
        "stages": {
            name: receipt["terminal_status"] for name, receipt in receipts.items()
        },
        "adopted_stages": (
            {
                name: evidence["terminal_status"]
                for name, evidence in active_adoption[
                    "component_stages"
                ].items()
            }
            if active_adoption is not None
            else {}
        ),
        "candidate": str(package_dir),
        "process_complete": bool(final["process_complete"]),
        "release_eligible": bool(final["release_eligible"]),
        "release_ineligible_reasons": final["release_ineligible_reasons"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", type=Path, default=DEFAULT_BASE_RELEASE)
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument("--qat-import-receipt", type=Path, default=DEFAULT_QAT_IMPORT)
    parser.add_argument("--seed-run", type=Path, default=DEFAULT_SEED_RUN)
    parser.add_argument("--adopt-adaptive-prefix-run", type=Path)
    parser.add_argument("--adopt-packaged-run", type=Path)
    parser.add_argument("--data-release", type=Path, default=DEFAULT_DATA_RELEASE)
    parser.add_argument(
        "--linguistic-augmentation",
        type=Path,
        default=DEFAULT_LINGUISTIC_AUGMENTATION,
    )
    parser.add_argument("--eval-lock", type=Path, default=DEFAULT_EVAL_LOCK)
    parser.add_argument(
        "--narration-release", type=Path, default=DEFAULT_NARRATION_RELEASE
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--package-id", default=DEFAULT_PACKAGE_ID)
    parser.add_argument(
        "--expected-data-release-id", default=contract.RELEASE_ID
    )
    parser.add_argument("--expected-eval-lock-id", default=contract.EVAL_ID)
    parser.add_argument(
        "--expected-linguistic-release-id",
        default=contract.LINGUISTIC_AUGMENTATION_ID,
    )
    parser.add_argument("--fullcall-alignment-steps", type=int, default=2_000)
    parser.add_argument("--agent-alignment-steps", type=int, default=1_000)
    parser.add_argument("--retrieval-r2-steps", type=int, default=1_600)
    parser.add_argument("--mw-steps", type=int, default=2_000)
    parser.add_argument("--confidence-train-limit", type=int, default=5_255)
    parser.add_argument("--confidence-valid-limit", type=int, default=2_020)
    parser.add_argument("--confidence-steps", type=int, default=800)
    parser.add_argument("--narration-steps", type=int, default=1_200)
    parser.add_argument("--eval-limit", type=int, default=588)
    parser.add_argument("--mw-eval-limit", type=int, default=1_000)
    parser.add_argument("--narration-eval-limit", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-stage", choices=STAGES)
    parser.add_argument("--phase-scope", choices=sorted(PHASE_SCOPE_STAGES))
    args = parser.parse_args(argv)
    if (
        args.phase_scope
        and args.stop_after_stage
        and args.stop_after_stage not in PHASE_SCOPE_STAGES[args.phase_scope]
    ):
        parser.error(
            f"--stop-after-stage {args.stop_after_stage} is outside "
            f"--phase-scope {args.phase_scope}"
        )
    for name, value in vars(args).items():
        if (name.endswith("steps") or name.endswith("limit")) and int(value) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    requested = args.run_dir.resolve()
    if args.dry_run:
        print(
            json.dumps(
                {"plan": plan, "requested_run_dir": str(requested), "dry_run": True},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if plan["heavy_execution_deferred"]:
        print(
            json.dumps(
                {"status": "deferred_live_cpt", "workers": plan["live_cpt_workers"]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 75
    run_dir = lifecycle.choose_run_dir(requested, plan, resume=args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    lifecycle.write_plan(run_dir, plan)
    preflight(args, plan, run_dir)
    _emit(
        "run_start",
        run_dir=str(run_dir),
        run_fingerprint_sha256=plan["run_fingerprint_sha256"],
    )
    try:
        progress = execute(args, plan, run_dir)
    except PhaseBoundaryReached as boundary:
        progress = {
            "schema": "mei-51m-adaptive-v5-phase-progress-v1",
            "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
            "run_dir": str(run_dir),
            "phase_complete": True,
            "stopped_after_stage": boundary.stage_id,
            "stages": {
                name: receipt["terminal_status"]
                for name, receipt in boundary.receipts.items()
            },
            "process_complete": False,
            "release_eligible": False,
        }
    _write_json(run_dir / "progress.json", progress)
    _emit(
        "run_complete",
        run_dir=str(run_dir),
        process_complete=progress["process_complete"],
        release_eligible=progress["release_eligible"],
    )
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
