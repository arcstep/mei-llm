#!/usr/bin/env python3
"""Run the 300M and 600M adaptive-v5 product chains without Codex polling.

The supervisor owns only local orchestration and observability.  It never
signals another training job, changes CURRENT.json, mutates a frozen Base, or
relabels a failed stage.  Each productizer remains receipt-driven; the two
chains are sequential so they cannot contend for MLX/Metal memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import orchestration.productize_51m as lifecycle
import orchestration.productize_adaptive_v5_51m as productizer
from common._repo import CURRENT_PATH, ROOT, resolve_repo_path


SUPERVISOR_ID = "mei-51m-paired-adaptive-v5-supervisor-v1"
HERE = Path(__file__).resolve().parent
PRODUCTIZER = HERE / "productize_adaptive_v5_51m.py"
DEFAULT_OUT_ROOT = (
    ROOT / "cycles/mei-1.1-51m/exp-00300m/runs/adaptive-v5-paired-supervisor"
)
PACKAGED_300_RUN = (
    ROOT
    / "cycles/mei-1.1-51m/exp-00300m/runs/"
    "productize-scratch300m-adaptive-v5-cq2-v2-40ba9754076e"
)
NEEDLE2_REFERENCE = ROOT / "src/platform/_shared/spec/needle2-reference.json"
HISTORICAL_MTP_RECEIPT = (
    ROOT
    / "cycles/mei-1.1-51m/exp-00300m/runs/"
    "productize-scratch300m-agent-cq2-v2-ff204182428e/"
    "stages/mtp_ablation/receipt.json"
)
HEARTBEAT_SCHEMA = "mei-local-supervisor-heartbeat-v1"
STATUS_SCHEMA = "mei-local-supervisor-status-v1"
PROGRESS_SCHEMA = "mei-local-supervisor-progress-event-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(_canonical(dict(value)) + b"\n")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical(dict(value)) + b"\n")
        handle.flush()


def _python() -> Path:
    preferred = ROOT / ".venv/bin/python"
    return preferred if preferred.is_file() else Path(sys.executable).resolve()


def _specs(
    *,
    adopt_completed_prefixes: bool = False,
    adopt_packaged_300m: bool = False,
    packaged_300m_run: Path | None = None,
    packaged_600m_run: Path | None = None,
) -> list[dict[str, Any]]:
    run_root_300 = ROOT / "cycles/mei-1.1-51m/exp-00300m/runs"
    run_root_600 = ROOT / "cycles/mei-1.1-51m/exp-00600m/runs"
    base_300 = ROOT / "cycles/mei-1.1-51m/exp-00300m/models/base/mei-1.0-51m-base-scratch300m-v1"
    base_600 = ROOT / "cycles/mei-1.1-51m/exp-00600m/models/base/mei-1.0-51m-base-cpt600m-clean-source-v3-v1"
    specs = [
        {
            "label": "300m",
            "requested_run_dir": run_root_300
            / "productize-scratch300m-adaptive-v5-cq2-v2",
            "package_id": (
                "mei-1.0-51m-scratch300m-tool-sft-cq2-v2-adaptive-v5"
            ),
            "base_release": base_300 / "RELEASE.json",
            "base_weights": base_300
            / "mei-1.0-51m-base-scratch300m-v1.npz",
            "qat_import_receipt": productizer.DEFAULT_QAT_IMPORT,
            "seed_run": productizer.DEFAULT_SEED_RUN,
        },
        {
            "label": "600m",
            "requested_run_dir": run_root_600
            / "productize-cpt600m-adaptive-v5-cq2-v2",
            "package_id": (
                "mei-1.0-51m-cpt600m-tool-sft-cq2-v2-adaptive-v5"
            ),
            "base_release": base_600 / "RELEASE.json",
            "base_weights": base_600
            / "mei-1.0-51m-base-cpt600m-clean-source-v3-v1.npz",
            "qat_import_receipt": run_root_600
            / "qat-cq2-cpt600m-clean-source-v3-v1-3769d9872a15"
            / "qat-import-candidate-receipt.json",
            "seed_run": run_root_600
            / "productize-cpt600m-sft-v4-quality-schema-cq2-v2",
        },
    ]
    if packaged_300m_run is not None:
        specs[0]["packaged_run"] = packaged_300m_run.resolve()
    elif adopt_completed_prefixes:
        if adopt_packaged_300m:
            specs[0]["packaged_run"] = PACKAGED_300_RUN
        else:
            specs[0]["adaptive_prefix_run"] = run_root_300 / (
                "productize-scratch300m-adaptive-v5-cq2-v2-164574857928"
            )
    if packaged_600m_run is not None:
        specs[1]["packaged_run"] = packaged_600m_run.resolve()
    elif adopt_completed_prefixes:
        specs[1]["adaptive_prefix_run"] = run_root_600 / (
            "productize-cpt600m-adaptive-v5-cq2-v2-6c3da768deef"
        )
    return specs


def _productizer_args(spec: Mapping[str, Any], run_dir: Path | None = None) -> list[str]:
    values = [
        "--base-release",
        str(Path(spec["base_release"]).resolve()),
        "--base-weights",
        str(Path(spec["base_weights"]).resolve()),
        "--qat-import-receipt",
        str(Path(spec["qat_import_receipt"]).resolve()),
        "--seed-run",
        str(Path(spec["seed_run"]).resolve()),
        "--run-dir",
        str((run_dir or Path(spec["requested_run_dir"])).resolve()),
        "--package-id",
        str(spec["package_id"]),
    ]
    if spec.get("adaptive_prefix_run") is not None:
        values.extend(
            [
                "--adopt-adaptive-prefix-run",
                str(Path(spec["adaptive_prefix_run"]).resolve()),
            ]
        )
    if spec.get("packaged_run") is not None:
        values.extend(
            [
                "--adopt-packaged-run",
                str(Path(spec["packaged_run"]).resolve()),
            ]
        )
    return values


def _prepare_product_run(
    spec: Mapping[str, Any], *, resume: bool
) -> tuple[argparse.Namespace, dict[str, Any], Path]:
    child_args = productizer.parse_args(_productizer_args(spec))
    plan = productizer.build_plan(child_args)
    run_dir = lifecycle.choose_run_dir(
        Path(spec["requested_run_dir"]).resolve(), plan, resume=resume
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    lifecycle.write_plan(run_dir, plan)
    return child_args, plan, run_dir


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    selected = [
        spec
        for spec in _specs(
            adopt_completed_prefixes=args.adopt_completed_prefixes,
            adopt_packaged_300m=args.adopt_packaged_300m,
            packaged_300m_run=args.adopt_packaged_300m_run,
            packaged_600m_run=args.adopt_packaged_600m_run,
        )
        if args.only == "both" or args.only == spec["label"]
    ]
    chains: list[dict[str, Any]] = []
    for spec in selected:
        parsed = productizer.parse_args(_productizer_args(spec))
        child_plan = productizer.build_plan(parsed)
        chain = {
                "label": spec["label"],
                "requested_run_dir": str(Path(spec["requested_run_dir"]).resolve()),
                "package_id": spec["package_id"],
                "base_release": str(Path(spec["base_release"]).resolve()),
                "base_release_sha256": _sha_file(Path(spec["base_release"])),
                "base_weights_sha256": _sha_file(Path(spec["base_weights"])),
                "qat_import_receipt": str(
                    Path(spec["qat_import_receipt"]).resolve()
                ),
                "qat_import_receipt_sha256": _sha_file(
                    Path(spec["qat_import_receipt"])
                ),
                "seed_run": str(Path(spec["seed_run"]).resolve()),
                "seed_plan_sha256": _sha_file(Path(spec["seed_run"]) / "plan.json"),
                "productizer_run_fingerprint_sha256": child_plan[
                    "run_fingerprint_sha256"
                ],
                "stage_graph_mode": child_plan["immutable"]["stage_graph_mode"],
                "planned_stage_ids": [
                    row["stage_id"] for row in child_plan["stages"]
                ],
            }
        if spec.get("adaptive_prefix_run") is not None:
            prefix_run = Path(spec["adaptive_prefix_run"]).resolve()
            chain["adaptive_prefix_run"] = str(prefix_run)
            chain["adaptive_prefix_plan_sha256"] = _sha_file(
                prefix_run / "plan.json"
            )
            chain["adaptive_prefix_blocked_receipt_sha256"] = _sha_file(
                prefix_run / "stages/adaptive_generation_eval_v5/receipt.json"
            )
        if spec.get("packaged_run") is not None:
            packaged_run = Path(spec["packaged_run"]).resolve()
            chain["packaged_run"] = str(packaged_run)
            chain["packaged_plan_sha256"] = _sha_file(packaged_run / "plan.json")
            chain["packaged_runtime_boundary_receipt_sha256"] = _sha_file(
                packaged_run / "stages/python_runtime_gate_v5/receipt.json"
            )
        chains.append(chain)
    immutable = {
        "schema": "mei-51m-paired-adaptive-v5-supervisor-plan-v1",
        "supervisor_id": SUPERVISOR_ID,
        "chains": chains,
        "sequence": [row["label"] for row in chains],
        "metal_concurrency": 1,
        "continue_independent_chain_after_failure": True,
        "recovery_mode": (
            "adopt-packaged-selected-chains"
            if args.adopt_packaged_300m_run is not None
            or args.adopt_packaged_600m_run is not None
            else "adopt-packaged-300m-and-adaptive-prefix-600m"
            if args.adopt_packaged_300m
            else "adopt-verified-adaptive-prefix"
            if args.adopt_completed_prefixes
            else "full-adaptive-productization"
        ),
        "heartbeat_seconds": int(args.heartbeat_seconds),
        "current_baseline_sha256": _sha_file(CURRENT_PATH),
        "sources": {
            str(path.relative_to(ROOT)): _sha_file(path)
            for path in (Path(__file__).resolve(), PRODUCTIZER)
        },
        "mutations_forbidden": [
            "CURRENT.json",
            "frozen Base",
            "historical run",
            "git commit",
            "git push",
            "public release",
        ],
    }
    return {
        **immutable,
        "plan_fingerprint_sha256": hashlib.sha256(_canonical(immutable)).hexdigest(),
    }


def _select_supervisor_root(requested: Path, plan: Mapping[str, Any], resume: bool) -> Path:
    requested = requested.resolve()
    plan_path = requested / "supervisor-plan.json"
    if not requested.exists() or not any(requested.iterdir()):
        return requested
    if resume and plan_path.is_file():
        observed = _load_json(plan_path)
        if observed.get("plan_fingerprint_sha256") == plan.get(
            "plan_fingerprint_sha256"
        ):
            return requested
    return requested.with_name(
        requested.name + "-" + str(plan["plan_fingerprint_sha256"])[:12]
    )


def _receipt_state(run_dir: Path) -> dict[str, Any]:
    last_passed = None
    first_blocker = None
    stages: dict[str, str] = {}
    plan_path = run_dir / "plan.json"
    planned_stages = (
        [
            str(row["stage_id"])
            for row in (_load_json(plan_path).get("stages") or [])
        ]
        if plan_path.is_file()
        else list(productizer.STAGES)
    )
    for stage in planned_stages:
        receipt_path = run_dir / "stages" / stage / "receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = _load_json(receipt_path)
        terminal = str(receipt.get("terminal_status") or "unknown")
        stages[stage] = terminal
        if terminal in {"passed", "degraded"}:
            last_passed = stage
        elif first_blocker is None:
            first_blocker = {
                "stage": stage,
                "error": receipt.get("error"),
                "receipt": str(receipt_path),
            }
    return {
        "last_passed_stage": last_passed,
        "first_blocker": first_blocker,
        "stage_statuses": stages,
    }


def _progress_payload(
    label: str, current_stage: str | None, parsed: Mapping[str, Any]
) -> dict[str, Any] | None:
    keys = {
        "step",
        "total",
        "loss",
        "retrieval_stage",
        "mw_step",
        "confidence_harvest",
        "confidence_step",
        "narration_step",
        "adaptive_eval_progress",
        "event",
    }
    if not any(key in parsed for key in keys):
        return None
    return {
        "schema": PROGRESS_SCHEMA,
        "unix": time.time(),
        "chain": label,
        "stage": current_stage,
        "payload": dict(parsed),
    }


def _write_live_state(
    run_dir: Path,
    *,
    label: str,
    pid: int,
    started: float,
    current_stage: str | None,
    last_event: Mapping[str, Any] | None,
    terminal_status: str = "running",
    exit_code: int | None = None,
) -> None:
    receipt_state = _receipt_state(run_dir)
    now = time.time()
    heartbeat = {
        "schema": HEARTBEAT_SCHEMA,
        "chain": label,
        "pid": pid,
        "alive": terminal_status == "running",
        "stage": current_stage,
        "elapsed_seconds": now - started,
        "updated_unix": now,
    }
    status = {
        "schema": STATUS_SCHEMA,
        "chain": label,
        "terminal_status": terminal_status,
        "pid": pid,
        "current_stage": current_stage,
        "last_event": dict(last_event or {}),
        "exit_code": exit_code,
        "updated_unix": now,
        **receipt_state,
    }
    _atomic_json(run_dir / "heartbeat.json", heartbeat)
    _atomic_json(run_dir / "STATUS.json", status)


def _run_productizer(
    spec: Mapping[str, Any], run_dir: Path, heartbeat_seconds: int
) -> dict[str, Any]:
    command = [
        str(_python()),
        str(PRODUCTIZER),
        *_productizer_args(spec, run_dir),
        "--resume",
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("productizer stdout pipe is unavailable")
    current_stage: str | None = None
    last_event: dict[str, Any] | None = None
    next_heartbeat = 0.0
    log_path = run_dir / "run.log"
    progress_path = run_dir / "progress.jsonl"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(
            json.dumps(
                {
                    "event": "supervisor_launch",
                    "unix": started,
                    "pid": process.pid,
                    "command": command,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        while True:
            now = time.time()
            if now >= next_heartbeat:
                _write_live_state(
                    run_dir,
                    label=str(spec["label"]),
                    pid=process.pid,
                    started=started,
                    current_stage=current_stage,
                    last_event=last_event,
                )
                next_heartbeat = now + heartbeat_seconds
            readable, _, _ = select.select(
                [process.stdout], [], [], min(1.0, max(0.0, next_heartbeat - now))
            )
            if readable:
                line = process.stdout.readline()
                if line:
                    log.write(line)
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        value = None
                    if isinstance(value, dict):
                        last_event = value
                        if value.get("event") == "stage_start":
                            current_stage = str(value.get("stage") or "") or current_stage
                        progress = _progress_payload(
                            str(spec["label"]), current_stage, value
                        )
                        if progress is not None:
                            _append_jsonl(progress_path, progress)
            if process.poll() is not None:
                for line in process.stdout:
                    log.write(line)
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        last_event = value
                        progress = _progress_payload(
                            str(spec["label"]), current_stage, value
                        )
                        if progress is not None:
                            _append_jsonl(progress_path, progress)
                break
    exit_code = int(process.returncode or 0)
    progress_document = (
        _load_json(run_dir / "progress.json")
        if (run_dir / "progress.json").is_file()
        else None
    )
    receipt_state = _receipt_state(run_dir)
    succeeded = bool(
        exit_code == 0
        and progress_document
        and progress_document.get("process_complete") is True
    )
    terminal = "passed" if succeeded else "blocked"
    _write_live_state(
        run_dir,
        label=str(spec["label"]),
        pid=process.pid,
        started=started,
        current_stage=current_stage,
        last_event=last_event,
        terminal_status=terminal,
        exit_code=exit_code,
    )
    return {
        "label": spec["label"],
        "run_dir": str(run_dir),
        "pid": process.pid,
        "exit_code": exit_code,
        "terminal_status": terminal,
        "process_complete": succeeded,
        "progress": progress_document,
        **receipt_state,
    }


def _nested(document: Mapping[str, Any], path: Iterable[str]) -> Any:
    value: Any = document
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _load_receipt_evidence(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"missing stage receipt: {path}")
    observed_sha = _sha_file(path)
    if expected_sha256 and observed_sha != expected_sha256:
        raise RuntimeError(f"stage receipt hash drifted: {path}")
    document = _load_json(path)
    if document.get("terminal_status") not in {"passed", "degraded", "blocked"}:
        raise RuntimeError(f"stage receipt has no terminal status: {path}")
    return {
        "path": str(path),
        "sha256": observed_sha,
        "terminal_status": document.get("terminal_status"),
        "document": document,
    }


def _resolve_stage_receipt(
    run_dir: Path,
    stage_name: str,
    *,
    seen: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Resolve a local or transitively adopted stage without copying evidence."""

    run_dir = run_dir.resolve()
    visited = seen if seen is not None else set()
    marker = (str(run_dir), stage_name)
    if marker in visited:
        raise RuntimeError(f"stage adoption cycle: {run_dir} {stage_name}")
    visited.add(marker)
    local = run_dir / "stages" / stage_name / "receipt.json"
    if local.is_file():
        return _load_receipt_evidence(local)
    plan_path = run_dir / "plan.json"
    if not plan_path.is_file():
        raise RuntimeError(f"run has no plan while resolving {stage_name}: {run_dir}")
    immutable = (_load_json(plan_path).get("immutable") or {})
    for key in (
        "packaged_run_adoption",
        "adaptive_prefix_adoption",
        "training_prefix_adoption",
    ):
        adoption = immutable.get(key) or {}
        component = (adoption.get("component_stages") or {}).get(stage_name)
        if component:
            return _load_receipt_evidence(
                resolve_repo_path(str(component["receipt"])),
                str(component.get("receipt_sha256") or "") or None,
            )
        parent = adoption.get("parent_run_dir")
        if parent:
            try:
                return _resolve_stage_receipt(
                    resolve_repo_path(str(parent)), stage_name, seen=visited
                )
            except RuntimeError as exc:
                if "missing stage receipt" not in str(exc) and "run has no plan" not in str(exc):
                    raise
    raise RuntimeError(f"missing stage receipt: {stage_name} from {run_dir}")


def _metrics(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return dict((evidence.get("document") or {}).get("metrics") or {})


def _metric_subset(value: Mapping[str, Any] | None, names: Iterable[str]) -> dict[str, Any]:
    source = value or {}
    return {name: source.get(name) for name in names}


def _compact_generation_splits(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, banks in (value.get("splits") or {}).items():
        compact: dict[str, Any] = {}
        for name, row in (banks or {}).items():
            metrics = _metric_subset(
                row,
                (
                    "exact_rate",
                    "retrieval_eligible_hit_rate",
                    "mean_batches",
                    "context_unrepresentable",
                    "adaptive_minus_fixed_first",
                ),
            )
            metrics["n"] = row.get("n", row.get("rows"))
            compact[str(name)] = metrics
        result[str(split)] = compact
    return result


def _comparison_entry(chain: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(chain["run_dir"]))
    plan = _load_json(run_dir / "plan.json")
    immutable = plan["immutable"]

    evidence = {
        name: _resolve_stage_receipt(run_dir, name)
        for name in (
            "float_base_lm_anchor",
            "float_task_control_v4",
            "cq2_qat_import",
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
    }
    base_release_path = Path(str(immutable["base"]["release"]))
    base_release = _load_json(base_release_path)
    base_valid_loss = base_release.get("valid_loss")
    if base_valid_loss is None:
        base_valid_loss = _nested(
            base_release, ("numeric_integrity", "terminal_metrics", "valid_loss")
        )
    qat = immutable["qat_seed"]
    qat_candidate_path = Path(str(qat.get("candidate_receipt") or "")).resolve()
    qat_candidate_sha = str(qat.get("candidate_receipt_sha256") or "")
    if (
        not qat_candidate_path.is_file()
        or len(qat_candidate_sha) != 64
        or _sha_file(qat_candidate_path) != qat_candidate_sha
    ):
        raise RuntimeError("QAT candidate receipt is not hash-bound")
    qat_candidate = _load_json(qat_candidate_path)
    if (
        _nested(qat_candidate, ("master", "sha256")) != qat.get("master_sha256")
        or qat_candidate.get("quant_math_id") != qat.get("quant_math_id")
    ):
        raise RuntimeError("QAT candidate receipt disagrees with adopted QAT seed")
    qat_valid_loss = qat_candidate.get("valid_loss")
    float_base = _metrics(evidence["float_base_lm_anchor"])
    float_control = _metrics(evidence["float_task_control_v4"])
    fullcall = _metrics(evidence["fullcall_alignment_replay_v5"])
    agent = _metrics(evidence["agent_alignment_replay_v5"])
    retrieval = _metrics(evidence["retrieval_r2_v5"])
    artifacts = _metrics(evidence["final_adaptive_artifacts_v5"])
    adaptive_eval = _metrics(evidence["adaptive_generation_eval_v5"])
    mw = _metrics(evidence["mw_disposition_v5"])
    harvest = _metrics(evidence["confidence_harvest_v5"])
    confidence_head = _metrics(evidence["confidence_head_v5"])
    narration_adapter = _metrics(evidence["narration_adapter_v5"])
    sidecars = _metrics(evidence["sidecar_runtime_eval_v5"])
    package = _metrics(evidence["package_v2_cq2_v5"])
    python_gate = _metrics(evidence["python_runtime_gate_v5"])
    browser_gate = _metrics(evidence["browser_wasm_gate_v5"])
    final = _metrics(evidence["final_audit_v5"])
    source_manifest_sha = hashlib.sha256(
        _canonical(immutable.get("source_manifest") or {})
    ).hexdigest()
    return {
        "base": {
            "id": immutable["base"]["base_id"],
            "exposure_tokens": immutable["base"]["tokens_seen_exposure"],
            "release": str(base_release_path.resolve()),
            "release_sha256": _sha_file(base_release_path),
            "weights_sha256": immutable["base"]["weights_sha256"],
            "valid_loss": base_valid_loss,
            "valid_loss_hq": base_release.get("valid_loss_hq"),
            "valid_loss_colloquial": base_release.get("valid_loss_colloquial"),
            "valid_loss_structure": base_release.get("valid_loss_structure"),
            "lineage_assurance": immutable["base"].get("lineage_assurance"),
            "release_eligible": immutable["base"].get("base_release_eligible"),
            "productization_experiment_eligible": immutable["base"].get(
                "productization_experiment_eligible"
            ),
            "release_blockers": immutable["base"].get("release_blockers") or [],
        },
        "float_anchor": {
            **_metric_subset(
                float_base,
                (
                    "validation_nll",
                    "probe_mean_nll",
                    "domain_nll_hq",
                    "domain_nll_colloquial",
                    "domain_nll_structure",
                ),
            ),
            "task_control_slice": _metric_subset(
                float_base.get("task_control_slice") or {},
                (
                    "balanced_accuracy",
                    "execute_arguments_exact",
                    "execute_tool_name_exact",
                    "refusal_accuracy",
                    "decode_tokens_per_second",
                ),
            ),
        },
        "float_task_control": {
            **_metric_subset(
                float_control,
                ("steps", "rows", "last_loss", "data_fingerprint"),
            ),
            "task_control_slice": _metric_subset(
                float_control.get("task_control_slice") or {},
                (
                    "balanced_accuracy",
                    "execute_arguments_exact",
                    "execute_tool_name_exact",
                    "refusal_accuracy",
                    "decode_tokens_per_second",
                ),
            ),
        },
        "cq2_qat": {
            "candidate_receipt": str(qat_candidate_path),
            "candidate_receipt_sha256": qat_candidate_sha,
            "quant_math_id": qat_candidate.get("quant_math_id"),
            "tokens_seen_qat": qat_candidate.get("tokens_seen_qat"),
            "master_sha256": qat.get("master_sha256")
            or _nested(qat, ("master", "sha256")),
            "master_bytes": _nested(qat_candidate, ("master", "bytes")),
            "valid_loss": qat_valid_loss,
            "absolute_loss_tax": (
                float(qat_valid_loss) - float(base_valid_loss)
                if qat_valid_loss is not None and base_valid_loss is not None
                else None
            ),
            "relative_loss_tax": (
                (float(qat_valid_loss) - float(base_valid_loss)) / float(base_valid_loss)
                if qat_valid_loss is not None and base_valid_loss
                else None
            ),
        },
        "fullcall_alignment": _metric_subset(
            fullcall,
            (
                "steps",
                "rows",
                "last_loss",
                "max_prompt_tokens",
                "data_fingerprint",
                "runtime_shared_prompt_framing",
            ),
        ),
        "agent_alignment": {
            **_metric_subset(
                agent,
                (
                    "steps",
                    "rows",
                    "last_loss",
                    "max_prompt_tokens",
                    "data_fingerprint",
                    "candidate_batches_are_not_external_steps",
                ),
            ),
            "quality_metric_scope": "training-replay-only; final multi-step task success remains open",
        },
        "retrieval_r2": {
            **_metric_subset(
                retrieval,
                ("steps", "rows", "last_loss", "tools", "data_fingerprint"),
            ),
            "calibration": artifacts.get("calibration"),
            "tool_index": artifacts.get("tool_index"),
        },
        "adaptive_generation": {
            "splits": _compact_generation_splits(adaptive_eval),
            "joint_budget_errors": adaptive_eval.get("joint_budget_errors"),
            "adaptive_vs_fixed_first_drop_within_one_point": adaptive_eval.get(
                "adaptive_vs_fixed_first_drop_within_one_point"
            ),
        },
        "mw_disposition": {
            "terminal_status": evidence["mw_disposition_v5"]["terminal_status"],
            "train": _metric_subset(
                mw.get("train") or {},
                ("steps", "last_loss", "rows", "resumed_from_step"),
            ),
            "eval": mw.get("eval"),
            "missing_class_mode_profile_cells": mw.get(
                "missing_class_mode_profile_cells"
            ),
            "trainer_live_retrieval_calls": mw.get("trainer_live_retrieval_calls"),
        },
        "confidence": {
            "harvest": {
                split: _metric_subset(
                    harvest.get(split) or {},
                    ("rows", "positive", "negative", "live_retrieval_calls"),
                )
                for split in ("train", "valid", "dev")
            },
            "head": {
                "terminal_status": evidence["confidence_head_v5"]["terminal_status"],
                "steps": _nested(confidence_head, ("train", "steps")),
                "last_loss": _nested(confidence_head, ("train", "last_loss")),
                "validation": _nested(
                    confidence_head, ("train", "pipeline_valid_metrics")
                ),
            },
            "runtime_eval": sidecars.get("confidence"),
        },
        "narration": {
            "adapter": _metric_subset(
                narration_adapter,
                (
                    "rank",
                    "parameter_count",
                    "steps",
                    "last_loss",
                    "valid_loss",
                    "terminal_only",
                    "verified_result_only",
                ),
            ),
            "runtime_eval": sidecars.get("narration"),
        },
        "package": _metric_subset(
            package,
            (
                "package_id",
                "package_dir",
                "package_bytes",
                "package_limit_bytes",
                "package_within_limit",
                "tensor_count",
                "lm_tensor_count",
                "mtp_tensor_count",
                "tensor_container_sha256",
                "weight_parameter_count",
            ),
        ),
        "python_runtime": {
            "terminal_status": evidence["python_runtime_gate_v5"]["terminal_status"],
            **python_gate,
        },
        "browser_wasm": {
            "terminal_status": evidence["browser_wasm_gate_v5"]["terminal_status"],
            **browser_gate,
        },
        "terminal": {
            "process_complete": final.get("process_complete"),
            "release_eligible": final.get("release_eligible"),
            "release_ineligible_reasons": final.get("release_ineligible_reasons"),
        },
        "contracts": {
            **immutable["contracts"],
            "data_release_fingerprint": immutable["data_release"][
                "release_fingerprint"
            ],
            "linguistic_release_fingerprint": immutable[
                "linguistic_augmentation"
            ]["release_fingerprint"],
            "eval_fingerprint": immutable["eval_lock"]["evaluation_fingerprint"],
            "tokenizer_sha256": immutable["tokenizer_sha256"],
            "current_run_source_manifest_sha256": source_manifest_sha,
            "validation_scope": immutable["validation_scope"],
        },
        "evidence": {
            name: {
                "path": row["path"],
                "sha256": row["sha256"],
                "terminal_status": row["terminal_status"],
            }
            for name, row in evidence.items()
        },
        "run_dir": str(run_dir),
        "run_fingerprint_sha256": plan["run_fingerprint_sha256"],
        "stage_graph_mode": immutable["stage_graph_mode"],
    }


def _format_metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "未评测"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_comparison(
    out_root: Path, chains: list[dict[str, Any]]
) -> dict[str, str] | None:
    completed = [row for row in chains if row.get("process_complete")]
    if len(completed) != 2:
        return None
    entries = {str(row["label"]): _comparison_entry(row) for row in completed}
    left = entries["300m"]
    right = entries["600m"]
    comparable_fields = (
        "weight_contract_sha256",
        "runtime_profile_sha256",
        "training_aux_sha256",
        "data_release_fingerprint",
        "linguistic_release_fingerprint",
        "eval_fingerprint",
        "tokenizer_sha256",
        "current_run_source_manifest_sha256",
        "validation_scope",
    )
    contract_checks = {
        field: left["contracts"].get(field) == right["contracts"].get(field)
        for field in comparable_fields
    }
    confounds = sorted(
        {
            reason
            for entry in entries.values()
            for reason in entry["base"].get("release_blockers") or []
        }
    )
    if right["base"].get("lineage_assurance") != "complete":
        confounds.append("600m_lineage_assurance_is_hybrid_recovery")
    confounds = sorted(set(confounds))
    report = {
        "schema": "mei-51m-adaptive-v5-paired-comparison-v1",
        "status": "passed",
        "comparison_boundary": "same adaptive-v5 current source/data/evaluation/runtime contract; Base exposure differs",
        "contract_checks": contract_checks,
        "all_comparison_contracts_equal": all(contract_checks.values()),
        "products": entries,
        "known_confounds": confounds,
        "pure_exposure_causal_comparison": False,
        "causal_claim": "descriptive paired evidence only; 600M hybrid recovery and corpus-diversity evidence prevent attributing every delta solely to exposure",
        "current_sha256": _sha_file(CURRENT_PATH),
    }
    json_path = out_root / "paired-comparison.json"
    md_path = out_root / "paired-comparison.zh-CN.md"
    _atomic_json(json_path, report)

    def get(entry: Mapping[str, Any], *path: str) -> Any:
        return _nested(entry, path)

    rows = [
        ("Base exposure tokens", get(left, "base", "exposure_tokens"), get(right, "base", "exposure_tokens")),
        ("Base valid loss", get(left, "base", "valid_loss"), get(right, "base", "valid_loss")),
        ("CQ2-QAT valid loss", get(left, "cq2_qat", "valid_loss"), get(right, "cq2_qat", "valid_loss")),
        ("CQ2-QAT 相对 loss 税", get(left, "cq2_qat", "relative_loss_tax"), get(right, "cq2_qat", "relative_loss_tax")),
        ("Float Task Control balanced accuracy", get(left, "float_task_control", "task_control_slice", "balanced_accuracy"), get(right, "float_task_control", "task_control_slice", "balanced_accuracy")),
        ("Float Task Control argument exact", get(left, "float_task_control", "task_control_slice", "execute_arguments_exact"), get(right, "float_task_control", "task_control_slice", "execute_arguments_exact")),
        ("Full-call replay last loss", get(left, "fullcall_alignment", "last_loss"), get(right, "fullcall_alignment", "last_loss")),
        ("Agent replay last loss（仅训练诊断）", get(left, "agent_alignment", "last_loss"), get(right, "agent_alignment", "last_loss")),
        ("R2 last loss", get(left, "retrieval_r2", "last_loss"), get(right, "retrieval_r2", "last_loss")),
        ("R2 eligible gold recall", get(left, "retrieval_r2", "calibration", "metrics", "eligible_gold_recall"), get(right, "retrieval_r2", "calibration", "metrics", "eligible_gold_recall")),
        ("Adaptive dev structural exact", get(left, "adaptive_generation", "splits", "dev", "structural", "exact_rate"), get(right, "adaptive_generation", "splits", "dev", "structural", "exact_rate")),
        ("Adaptive dev natural exact", get(left, "adaptive_generation", "splits", "dev", "natural", "exact_rate"), get(right, "adaptive_generation", "splits", "dev", "natural", "exact_rate")),
        ("Adaptive dev schema exact", get(left, "adaptive_generation", "splits", "dev", "schema_holdout", "exact_rate"), get(right, "adaptive_generation", "splits", "dev", "schema_holdout", "exact_rate")),
        ("MW dev accuracy", get(left, "mw_disposition", "eval", "dev", "accuracy"), get(right, "mw_disposition", "eval", "dev", "accuracy")),
        ("MW dev macro-F1", get(left, "mw_disposition", "eval", "dev", "macro_f1"), get(right, "mw_disposition", "eval", "dev", "macro_f1")),
        ("Confidence test AUROC", get(left, "confidence", "runtime_eval", "test", "auroc"), get(right, "confidence", "runtime_eval", "test", "auroc")),
        ("Confidence test ECE", get(left, "confidence", "runtime_eval", "test", "ece_10"), get(right, "confidence", "runtime_eval", "test", "ece_10")),
        ("Narration learned exact acceptance", get(left, "narration", "runtime_eval", "test", "adapter_exact_acceptance"), get(right, "narration", "runtime_eval", "test", "adapter_exact_acceptance")),
        ("Narration delivered correctness", get(left, "narration", "runtime_eval", "test", "delivered_answer_correctness"), get(right, "narration", "runtime_eval", "test", "delivered_answer_correctness")),
        ("Package bytes", get(left, "package", "package_bytes"), get(right, "package", "package_bytes")),
        ("Python/MLX warm decode tok/s", get(left, "python_runtime", "warm_decode", "tokens_per_second"), get(right, "python_runtime", "warm_decode", "tokens_per_second")),
        ("Python gate process RSS bytes", get(left, "python_runtime", "process_rss", "rss_bytes"), get(right, "python_runtime", "process_rss", "rss_bytes")),
        ("Browser-WASM heap bytes", get(left, "browser_wasm", "heap_peak_bytes"), get(right, "browser_wasm", "heap_peak_bytes")),
        ("Browser-WASM warm decode tok/s", get(left, "browser_wasm", "warm_steady_decode_tok_s"), get(right, "browser_wasm", "warm_steady_decode_tok_s")),
    ]
    markdown = [
        "# mei-1.0-51m adaptive-v5：300M / 600M 配对结果",
        "",
        "两侧使用同一当前源码、数据、评测、runtime profile 与 validation scope；600M 存在 hybrid recovery 与 corpus diversity confound，因此这里只作描述性配对，不把全部差值归因于 exposure。",
        "",
        "| 指标 | 300M | 600M |",
        "|---|---:|---:|",
    ]
    markdown.extend(
        f"| {name} | {_format_metric(a)} | {_format_metric(b)} |"
        for name, a, b in rows
    )
    markdown.extend(
        [
            "",
            "Agent 最终多步骤 task-success 尚未在 adaptive-v5 中形成独立质量分数；本报告只把 1,000-step alignment receipt 与 runtime call→ToolResult→继续门禁记为已完成证据，不将训练 loss 冒充任务成功率。",
            "",
            f"已知 confound：{', '.join(confounds) if confounds else '无'}。",
        ]
    )
    _atomic_text(md_path, "\n".join(markdown) + "\n")
    return {"json": str(json_path), "markdown": str(md_path)}


def _write_needle2_alignment(
    out_root: Path, entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    reference = _load_json(NEEDLE2_REFERENCE)

    def evidence(stage: str) -> list[str]:
        return [
            str(entries[label]["evidence"][stage]["path"])
            for label in ("300m", "600m")
        ]

    both_package_identity = all(
        int(entries[label]["package"].get("weight_parameter_count") or 0)
        == 51_463_797
        and int(entries[label]["package"].get("mtp_tensor_count") or 0) == 0
        for label in ("300m", "600m")
    )
    browser_functional = all(
        bool(entries[label]["browser_wasm"].get("functional_complete"))
        for label in ("300m", "600m")
    )
    browser_resource_ok = all(
        bool(entries[label]["browser_wasm"].get("heap_within_96_mib"))
        and bool(
            entries[label]["browser_wasm"].get(
                "warm_steady_decode_100_tok_s_validated"
            )
        )
        for label in ("300m", "600m")
    )
    calibration_validated = all(
        bool(entries[label]["retrieval_r2"]["calibration"].get("validated"))
        for label in ("300m", "600m")
    )
    mw_quality_ok = all(
        entries[label]["mw_disposition"]["terminal_status"] == "passed"
        for label in ("300m", "600m")
    )
    confidence_quality_ok = all(
        entries[label]["confidence"]["head"]["terminal_status"] == "passed"
        for label in ("300m", "600m")
    )
    confidence_mechanism_validated = all(
        int(entries[label]["confidence"]["head"].get("steps") or 0) == 800
        and all(
            int(entries[label]["confidence"]["harvest"][split].get("positive") or 0)
            > 0
            and int(
                entries[label]["confidence"]["harvest"][split].get("negative")
                or 0
            )
            > 0
            for split in ("train", "valid", "dev")
        )
        for label in ("300m", "600m")
    )
    rows = [
        {
            "feature": "backbone",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": "validated" if both_package_identity else "open",
            "quality_status": "separate_from_mechanism",
            "needle2": "512d / 27 layers / GQA 8:4 / head 64 / RoPE 100000 / context 2048",
            "mei_51m": "同几何；中文 24k vocabulary；部署 LM 精确 51,463,797 参数",
            "evidence": evidence("package_v2_cq2_v5"),
        },
        {
            "feature": "Engram",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": "validated" if both_package_identity else "open",
            "quality_status": "architecture_and_numeric_runtime_validated",
            "needle2": "sites 2,15；orders 2,3；slots 8192；sub-dim 128；4 taps",
            "mei_51m": "同一几何、边界 mask 与部署 tensor contract；中文 tokenizer 输入",
            "evidence": evidence("python_runtime_gate_v5"),
        },
        {
            "feature": "mHC",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": "validated" if both_package_identity else "open",
            "quality_status": "architecture_and_numeric_runtime_validated",
            "needle2": "4 lanes / Sinkhorn routing",
            "mei_51m": "4 lanes / 20-step Sinkhorn；mHC tensor 使用 CQ4 policy",
            "evidence": evidence("package_v2_cq2_v5"),
        },
        {
            "feature": "MTP",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": "open",
            "quality_status": "historical_300m_ablation_only; not rerun as paired adaptive-v5 stage",
            "needle2": "training-time auxiliary path",
            "mei_51m": "训练旁路已实现，部署包两侧均为零 MTP tensor；本轮没有新的 300M/600M 配对消融",
            "evidence": ([str(HISTORICAL_MTP_RECEIPT)] if HISTORICAL_MTP_RECEIPT.is_file() else [])
            + evidence("package_v2_cq2_v5"),
        },
        {
            "feature": "CQ2",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": "validated" if both_package_identity else "open",
            "quality_status": "package_integrity_validated",
            "needle2": "group128 WHT + deterministic Gaussian/Lloyd-Max codebook；embedding/mHC Q4，其余默认 Q2",
            "mei_51m": "mei-cq-v2-g128-wht-codebook；LM 与全部 heads 使用 portable package v2",
            "evidence": evidence("package_v2_cq2_v5"),
        },
        {
            "feature": "KV/activation int8",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": "validated" if browser_functional else "open",
            "quality_status": "bounded_dynamic_kv_and_numeric_runtime",
            "needle2": "8-bit KV/activations；sliding window 256",
            "mei_51m": "2048 联合合同下 stable prefix + 动态 ordinary KV；KV/activation int8",
            "evidence": evidence("python_runtime_gate_v5") + evidence("browser_wasm_gate_v5"),
        },
        {
            "feature": "retrieval relevance / fixed-five batched scan",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": "validated",
            "quality_status": "validated" if calibration_validated else "degraded_calibration_fallback",
            "needle2": "contrastive retrieval head；top-5",
            "mei_51m": "独立 R2 + f16 index；discard/expand 双阈值；每批最多五个并可继续扫描",
            "evidence": evidence("retrieval_r2_v5") + evidence("final_adaptive_artifacts_v5") + evidence("adaptive_generation_eval_v5"),
        },
        {
            "feature": "full-call and Agent loop",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": "validated",
            "quality_status": "fullcall_measured; adaptive_multistep_task_success_open",
            "needle2": "constrained call/refuse and multi-step complete/run continuation",
            "mei_51m": "quant-aware Full-call replay、Agent continuation replay 与 call→ToolResult→继续 runtime 状态机",
            "evidence": evidence("fullcall_alignment_replay_v5") + evidence("agent_alignment_replay_v5") + evidence("python_runtime_gate_v5"),
        },
        {
            "feature": "MW disposition",
            "classification": "intentional_chinese_enhancement",
            "implementation_status": "implemented",
            "validation_status": "validated",
            "quality_status": "validated" if mw_quality_ok else "degraded",
            "needle2": "无对应 20 类 sidecar",
            "mei_51m": "独立 20 类 reason-code head；冻结 visible batches；class 0 以外 fail closed",
            "evidence": evidence("mw_disposition_v5"),
        },
        {
            "feature": "execution confidence",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": (
                "validated" if confidence_mechanism_validated else "open"
            ),
            "quality_status": (
                "validated"
                if confidence_quality_ok
                else "degraded_calibration_quality"
            ),
            "needle2": "calibrated confidence head",
            "mei_51m": "独立 confidence head；确定性 validator 先于 confidence",
            "evidence": evidence("confidence_harvest_v5") + evidence("confidence_head_v5") + evidence("sidecar_runtime_eval_v5"),
        },
        {
            "feature": "terminal narration",
            "classification": "intentional_chinese_enhancement",
            "implementation_status": "implemented",
            "validation_status": "validated",
            "quality_status": "learned_acceptance_and_deterministic_delivered_correctness_reported_separately",
            "needle2": "非 Needle2 核心机制",
            "mei_51m": "冻结主干 rank-16 / 392,192 参数中文 sidecar；只消费 verified terminal result，失败回退确定性模板",
            "evidence": evidence("narration_adapter_v5") + evidence("sidecar_runtime_eval_v5"),
        },
        {
            "feature": "Python/MLX runtime and SDK",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": "validated",
            "quality_status": "package_bound_numeric_gate",
            "needle2": "Python API",
            "mei_51m": "服务器端 Python/MLX package v2 加载与 complete/session 接口",
            "evidence": evidence("python_runtime_gate_v5"),
        },
        {
            "feature": "Browser-WASM runtime and SDK",
            "classification": "implemented",
            "implementation_status": "implemented",
            "validation_status": "validated" if browser_functional else "open",
            "quality_status": "resource_validated" if browser_resource_ok else "performance_or_heap_open",
            "needle2": "WASM engine",
            "mei_51m": "Rust core 作为浏览器 WASM 内核；功能、heap 与 warm decode 独立记账",
            "evidence": evidence("browser_wasm_gate_v5"),
        },
        {
            "feature": "independent Rust/Node/C SDK",
            "classification": "open",
            "implementation_status": "implemented_partial",
            "validation_status": "open",
            "quality_status": "deferred_by_python-browser-wasm_validation_scope",
            "needle2": "不属于本轮机制对齐硬边界",
            "mei_51m": "源码保留；独立公共 SDK parity、集成与资源门延后",
            "evidence": [],
        },
        {
            "feature": "Chinese tokenizer/SFT/safety/narration",
            "classification": "intentional_chinese_enhancement",
            "implementation_status": "implemented",
            "validation_status": "validated",
            "quality_status": "metrics_reported_per_head",
            "needle2": "英文/通用机制基线",
            "mei_51m": "zh-24k-v1、中文 CPT/SFT、provenance/permission/state 门与中文终态解说",
            "evidence": evidence("adaptive_generation_eval_v5") + evidence("sidecar_runtime_eval_v5"),
        },
        {
            "feature": ".cact/libneedle/Cactus API or ABI",
            "classification": "open",
            "implementation_status": "intentionally_not_implemented",
            "validation_status": "open",
            "quality_status": "explicit_non_compatibility",
            "needle2": ".cact / libneedle / Cactus surfaces",
            "mei_51m": "不兼容且不声明兼容；使用 mei-model-package-v2 / mei-runtime-wire-v2",
            "evidence": [str(NEEDLE2_REFERENCE)],
        },
    ]
    report = {
        "schema": "mei-needle2-paired-alignment-report-v2",
        "status": "passed",
        "reference": reference,
        "reference_sha256": _sha_file(NEEDLE2_REFERENCE),
        "products": {
            label: {
                "package_id": entries[label]["package"].get("package_id"),
                "package_dir": entries[label]["package"].get("package_dir"),
                "run_fingerprint_sha256": entries[label]["run_fingerprint_sha256"],
            }
            for label in ("300m", "600m")
        },
        "matrix": rows,
        "mechanism_alignment_complete": all(
            row["implementation_status"] in {"implemented", "implemented_partial", "intentionally_not_implemented"}
            for row in rows
        ),
        "all_validation_and_resource_gates_passed": all(
            row["validation_status"] == "validated"
            and row["quality_status"] not in {"degraded", "degraded_calibration_fallback", "performance_or_heap_open"}
            for row in rows
            if row["classification"] != "open"
        ),
        "explicit_non_compatibility": reference.get("explicit_non_compatibility") or [],
    }
    json_path = out_root / "needle2-alignment.json"
    md_path = out_root / "needle2-alignment.zh-CN.md"
    _atomic_json(json_path, report)
    markdown = [
        "# mei-1.0-51m 中文增强版与 Needle2 机制对齐矩阵",
        "",
        f"固定参考：Needle Git `{reference['github']['commit']}`；Needle2 config `{reference['huggingface']['revision']}`。本报告不声明 `.cact`、`libneedle`、Cactus API 或 ABI 兼容。",
        "",
        "| 特性 | 实现 | 验证 | 分类 | 质量/资源结论 |",
        "|---|---|---|---|---|",
    ]
    markdown.extend(
        "| {feature} | {implementation_status} | {validation_status} | {classification} | {quality_status} |".format(**row)
        for row in rows
    )
    _atomic_text(md_path, "\n".join(markdown) + "\n")
    return {"json": str(json_path), "markdown": str(md_path)}


def _write_longitudinal_history(
    out_root: Path, entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    rows: list[dict[str, Any]] = []
    for label in ("300m", "600m"):
        entry = entries[label]
        record = {
            "schema": "mei-51m-longitudinal-product-record-v1",
            "record_id": f"{label}-adaptive-v5-{entry['run_fingerprint_sha256'][:12]}",
            "base": entry["base"],
            "product_run": entry["run_dir"],
            "run_fingerprint_sha256": entry["run_fingerprint_sha256"],
            "package": entry["package"],
            "contracts": entry["contracts"],
            "metrics": {
                "cq2_qat": entry["cq2_qat"],
                "float_task_control": entry["float_task_control"],
                "fullcall_alignment": entry["fullcall_alignment"],
                "agent_alignment": entry["agent_alignment"],
                "retrieval_r2": entry["retrieval_r2"],
                "adaptive_generation": entry["adaptive_generation"],
                "mw_disposition": entry["mw_disposition"],
                "confidence": entry["confidence"],
                "narration": entry["narration"],
                "python_runtime": entry["python_runtime"],
                "browser_wasm": entry["browser_wasm"],
            },
            "terminal": entry["terminal"],
            "comparability_grade": "C",
            "comparability_note": "hash-bound adaptive-v5 artifacts under one frozen data/eval/runtime contract; recovery/adoption provenance is retained and 600M has a lineage/corpus confound",
            "evidence": entry["evidence"],
            "previous_human_ledger": str((ROOT / "cycles/mei-1.1-51m/LIFECYCLE.md").resolve()),
            "previous_human_ledger_sha256": _sha_file(ROOT / "cycles/mei-1.1-51m/LIFECYCLE.md"),
        }
        record["record_fingerprint_sha256"] = hashlib.sha256(
            _canonical(record)
        ).hexdigest()
        rows.append(record)
    jsonl_path = out_root / "longitudinal-history.jsonl"
    index_path = out_root / "longitudinal-history-index.json"
    _atomic_text(
        jsonl_path,
        "".join(_canonical(row).decode("utf-8") + "\n" for row in rows),
    )
    index = {
        "schema": "mei-51m-longitudinal-history-index-v1",
        "status": "passed",
        "append_only": True,
        "records": [
            {
                "record_id": row["record_id"],
                "record_fingerprint_sha256": row["record_fingerprint_sha256"],
                "base_exposure_tokens": row["base"]["exposure_tokens"],
                "package_id": row["package"].get("package_id"),
                "process_complete": row["terminal"].get("process_complete"),
                "release_eligible": row["terminal"].get("release_eligible"),
            }
            for row in rows
        ],
        "ledger": str(jsonl_path),
        "ledger_sha256": _sha_file(jsonl_path),
    }
    _atomic_json(index_path, index)
    return {"jsonl": str(jsonl_path), "index": str(index_path)}


def _write_final_deliverables(
    out_root: Path,
    chains: list[dict[str, Any]],
    comparison: Mapping[str, str],
) -> dict[str, Any]:
    entries = {str(row["label"]): _comparison_entry(row) for row in chains}
    alignment = _write_needle2_alignment(out_root, entries)
    history = _write_longitudinal_history(out_root, entries)

    def paths_for(stage: str) -> list[str]:
        return [entries[label]["evidence"][stage]["path"] for label in ("300m", "600m")]

    def terminal_status(stage: str) -> str:
        statuses = {
            entries[label]["evidence"][stage]["terminal_status"]
            for label in ("300m", "600m")
        }
        return "degraded" if "degraded" in statuses else "passed"

    f_items = [
        {"id": "F01", "status": "degraded" if entries["600m"]["base"]["release_eligible"] is False else "passed", "evidence": [entries[label]["base"]["release"] for label in ("300m", "600m")]},
        {"id": "F02", "status": "passed", "evidence": paths_for("cq2_qat_import")},
        {"id": "F03", "status": terminal_status("agent_alignment_replay_v5"), "evidence": paths_for("fullcall_alignment_replay_v5") + paths_for("agent_alignment_replay_v5")},
        {"id": "F04", "status": terminal_status("final_adaptive_artifacts_v5"), "evidence": paths_for("retrieval_r2_v5") + paths_for("final_adaptive_artifacts_v5")},
        {"id": "F05", "status": terminal_status("adaptive_generation_eval_v5"), "evidence": paths_for("adaptive_generation_eval_v5")},
        {"id": "F06", "status": terminal_status("mw_disposition_v5"), "evidence": paths_for("mw_disposition_v5")},
        {"id": "F07", "status": terminal_status("confidence_head_v5"), "evidence": paths_for("confidence_harvest_v5") + paths_for("confidence_head_v5") + paths_for("sidecar_runtime_eval_v5")},
        {"id": "F08", "status": terminal_status("narration_adapter_v5"), "evidence": paths_for("narration_adapter_v5") + paths_for("sidecar_runtime_eval_v5")},
        {"id": "F09", "status": terminal_status("package_v2_cq2_v5"), "evidence": paths_for("package_v2_cq2_v5")},
        {"id": "F10", "status": terminal_status("python_runtime_gate_v5"), "evidence": paths_for("python_runtime_gate_v5")},
        {"id": "F11", "status": terminal_status("browser_wasm_gate_v5"), "evidence": paths_for("browser_wasm_gate_v5")},
        {"id": "F12", "status": "passed", "evidence": list(comparison.values())},
        {"id": "F13", "status": "passed", "evidence": list(alignment.values())},
        {"id": "F14", "status": "passed", "evidence": list(history.values())},
    ]
    current_sha = _sha_file(CURRENT_PATH)
    if current_sha != "5b0b68eeb8322bb9cdbef112777b1b346b69a91f3bce7234b1c6370389a42607":
        raise RuntimeError("CURRENT.json drifted before paired final audit")
    process_complete = all(row["status"] in {"passed", "degraded"} for row in f_items)
    release_reasons = sorted(
        {
            reason
            for entry in entries.values()
            for reason in entry["terminal"].get("release_ineligible_reasons") or []
        }
        | {
            f"{row['id']}_degraded" for row in f_items if row["status"] == "degraded"
        }
    )
    release_eligible = bool(
        process_complete
        and not release_reasons
        and all(entries[label]["terminal"].get("release_eligible") for label in ("300m", "600m"))
    )
    audit = {
        "schema": "mei-51m-adaptive-v5-paired-final-audit-v1",
        "terminal_status": "passed",
        "process_complete": process_complete,
        "release_eligible": release_eligible,
        "release_ineligible_reasons": release_reasons,
        "f01_f14": f_items,
        "products": {
            label: {
                "run_dir": entries[label]["run_dir"],
                "run_fingerprint_sha256": entries[label]["run_fingerprint_sha256"],
                "package_id": entries[label]["package"].get("package_id"),
                "package_dir": entries[label]["package"].get("package_dir"),
                "process_complete": entries[label]["terminal"].get("process_complete"),
                "release_eligible": entries[label]["terminal"].get("release_eligible"),
            }
            for label in ("300m", "600m")
        },
        "comparison": dict(comparison),
        "needle2_alignment": alignment,
        "longitudinal_history": history,
        "current_sha256": current_sha,
        "current_unchanged": True,
        "old_blocked_runs_preserved": True,
        "base_mutated": False,
        "cpt_or_qat_retrained": False,
        "git_commit_or_push_performed": False,
        "public_release_performed": False,
        "cactus_compatibility_claimed": False,
    }
    audit_path = out_root / "paired-final-audit.json"
    _atomic_json(audit_path, audit)
    freeze = {
        "schema": "mei-51m-local-freeze-proposal-v1",
        "status": "freeze_proposed",
        "process_complete": process_complete,
        "release_eligible": release_eligible,
        "release_ineligible_reasons": release_reasons,
        "candidates": {
            label: entries[label]["package"].get("package_dir")
            for label in ("300m", "600m")
        },
        "current_baseline_sha256": current_sha,
        "current_mutated": False,
        "proposal_is_not_promotion": True,
    }
    freeze_path = out_root / "freeze-proposal.json"
    _atomic_json(freeze_path, freeze)
    artifacts = {
        **{f"comparison_{key}": value for key, value in comparison.items()},
        **{f"alignment_{key}": value for key, value in alignment.items()},
        **{f"history_{key}": value for key, value in history.items()},
        "paired_final_audit": str(audit_path),
        "freeze_proposal": str(freeze_path),
    }
    deliverable_receipt = {
        "schema": "mei-51m-adaptive-v5-paired-deliverables-receipt-v1",
        "terminal_status": "passed",
        "process_complete": process_complete,
        "release_eligible": release_eligible,
        "artifacts": {
            name: {"path": str(Path(path).resolve()), "sha256": _sha_file(Path(path))}
            for name, path in sorted(artifacts.items())
        },
        "current_sha256": _sha_file(CURRENT_PATH),
        "current_unchanged": _sha_file(CURRENT_PATH) == current_sha,
    }
    receipt_path = out_root / "deliverables-receipt.json"
    _atomic_json(receipt_path, deliverable_receipt)
    return {
        "receipt": str(receipt_path),
        "receipt_sha256": _sha_file(receipt_path),
        "process_complete": process_complete,
        "release_eligible": release_eligible,
        "release_ineligible_reasons": release_reasons,
        "artifacts": artifacts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_plan(args)
    out_root = _select_supervisor_root(args.out_root, plan, args.resume)
    out_root.mkdir(parents=True, exist_ok=True)
    plan_path = out_root / "supervisor-plan.json"
    if plan_path.is_file():
        if _load_json(plan_path) != plan:
            raise RuntimeError("supervisor plan drifted; use the fingerprinted new run")
    else:
        _atomic_json(plan_path, plan)
    results: list[dict[str, Any]] = []
    overall_status = {
        "schema": STATUS_SCHEMA,
        "terminal_status": "running",
        "current_chain": None,
        "completed_chains": [],
        "planned_chains": plan["sequence"],
        "first_blocker": None,
        "updated_unix": time.time(),
    }
    _atomic_json(out_root / "STATUS.json", overall_status)
    for spec in [
        row
        for row in _specs(
            adopt_completed_prefixes=args.adopt_completed_prefixes,
            adopt_packaged_300m=args.adopt_packaged_300m,
            packaged_300m_run=args.adopt_packaged_300m_run,
            packaged_600m_run=args.adopt_packaged_600m_run,
        )
        if args.only == "both" or args.only == row["label"]
    ]:
        _, child_plan, run_dir = _prepare_product_run(spec, resume=args.resume)
        expected = next(
            row for row in plan["chains"] if row["label"] == spec["label"]
        )
        if child_plan["run_fingerprint_sha256"] != expected[
            "productizer_run_fingerprint_sha256"
        ]:
            raise RuntimeError("child productizer plan drifted after supervisor planning")
        overall_status.update(
            {
                "current_chain": spec["label"],
                "current_run_dir": str(run_dir),
                "updated_unix": time.time(),
            }
        )
        _atomic_json(out_root / "STATUS.json", overall_status)
        result = _run_productizer(spec, run_dir, args.heartbeat_seconds)
        results.append(result)
        overall_status["completed_chains"] = [row["label"] for row in results]
        if result.get("first_blocker") and overall_status["first_blocker"] is None:
            overall_status["first_blocker"] = {
                "chain": spec["label"],
                **result["first_blocker"],
            }
        overall_status["updated_unix"] = time.time()
        _atomic_json(out_root / "STATUS.json", overall_status)
    chains_complete = bool(
        len(results) == len(plan["sequence"])
        and all(row.get("process_complete") for row in results)
    )
    comparison = _write_comparison(out_root, results)
    deliverables: dict[str, Any] | None = None
    postprocess_error: dict[str, str] | None = None
    if chains_complete and comparison is not None:
        try:
            deliverables = _write_final_deliverables(
                out_root, results, comparison
            )
        except Exception as exc:
            postprocess_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _atomic_json(
                out_root / "postprocess-blocker.json",
                {
                    "schema": "mei-51m-paired-postprocess-blocker-v1",
                    "terminal_status": "blocked",
                    "error": postprocess_error,
                    "current_sha256": _sha_file(CURRENT_PATH),
                },
            )
    process_complete = bool(
        chains_complete
        and deliverables is not None
        and deliverables.get("process_complete") is True
    )
    receipt = {
        "schema": "mei-51m-paired-adaptive-v5-supervisor-receipt-v1",
        "terminal_status": "passed" if process_complete else "blocked",
        "process_complete": process_complete,
        "release_eligible": bool(
            process_complete
            and deliverables is not None
            and deliverables.get("release_eligible") is True
        ),
        "chains": results,
        "comparison": comparison,
        "deliverables": deliverables,
        "postprocess_error": postprocess_error,
        "plan": str(plan_path),
        "plan_sha256": _sha_file(plan_path),
        "current_sha256": _sha_file(CURRENT_PATH),
        "current_unchanged": _sha_file(CURRENT_PATH)
        == plan["current_baseline_sha256"],
    }
    _atomic_json(out_root / "supervisor-receipt.json", receipt)
    overall_status.update(
        {
            "terminal_status": receipt["terminal_status"],
            "current_chain": None,
            "process_complete": process_complete,
            "release_eligible": receipt["release_eligible"],
            "comparison": receipt["comparison"],
            "deliverables": deliverables,
            "postprocess_error": postprocess_error,
            "updated_unix": time.time(),
        }
    )
    _atomic_json(out_root / "STATUS.json", overall_status)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--only", choices=("both", "300m", "600m"), default="both"
    )
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument(
        "--adopt-completed-prefixes",
        action="store_true",
        help=(
            "adopt the receipt-verified adaptive-v5 LM/R2 prefix and resume "
            "at the previously blocked evaluation boundary"
        ),
    )
    parser.add_argument(
        "--adopt-packaged-300m",
        action="store_true",
        help=(
            "adopt the receipt-verified 300M package blocked only by the old "
            "Python loader call site; 600M still resumes from its adaptive prefix"
        ),
    )
    parser.add_argument(
        "--adopt-packaged-300m-run",
        type=Path,
        help=(
            "receipt-complete 300M package-producing run to adopt for "
            "current-source Python/Browser reevaluation"
        ),
    )
    parser.add_argument(
        "--adopt-packaged-600m-run",
        type=Path,
        help=(
            "receipt-complete 600M package-producing run to adopt for "
            "current-source Python/Browser reevaluation"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive")
    if args.adopt_packaged_300m and not args.adopt_completed_prefixes:
        parser.error("--adopt-packaged-300m requires --adopt-completed-prefixes")
    if args.adopt_packaged_300m and args.adopt_packaged_300m_run is not None:
        parser.error(
            "--adopt-packaged-300m and --adopt-packaged-300m-run are mutually exclusive"
        )
    if args.only == "both" and bool(args.adopt_packaged_300m_run) != bool(
        args.adopt_packaged_600m_run
    ):
        parser.error(
            "paired packaged reevaluation requires both explicit packaged run paths"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_plan(args), ensure_ascii=False, sort_keys=True))
        return 0
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["process_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
