#!/usr/bin/env python3
"""Verify the terminal 300M/600M adaptive-v5 goal evidence.

This is an audit-only entrypoint.  It never trains, repacks, promotes, or
rewrites an existing run.  A successful invocation writes a new receipt that
binds the paired deliverables, every referenced stage receipt and output,
frozen Base files, the F01--F14 SSOT, AGENTS.md, and the canonical/Cursor skill
mirror into one immutable completion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from common._repo import CURRENT_PATH, ROOT


SCHEMA = "mei-51m-adaptive-v5-goal-completion-audit-v1"
EXPECTED_CURRENT_SHA256 = (
    "5b0b68eeb8322bb9cdbef112777b1b346b69a91f3bce7234b1c6370389a42607"
)
EXPECTED_GIT_HEAD = "17513c4da11bec8ccf07f43eba92756a08450848"
EXPECTED_F_IDS = [f"F{index:02d}" for index in range(1, 15)]
EXPECTED_STAGE_STEPS = {
    "fullcall_alignment": 2_000,
    "agent_alignment": 1_000,
    "retrieval_r2": 1_600,
}
DEFAULT_PAIRED_ROOT = (
    ROOT
    / "cycles/mei-1.1-51m/exp-00300m/runs/"
    "adaptive-v5-paired-final-gates-v4"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "cycles/mei-1.1-51m/exp-00300m/runs/"
    "adaptive-v5-goal-completion-audit-v1"
)
DEFAULT_GUIDE = ROOT / "docs/mei-1.0-51m-work-guide.md"
DEFAULT_AGENTS = ROOT / "AGENTS.md"
DEFAULT_CANONICAL_SKILL = ROOT.parent / ".agents/skills/mei-51m-cpt-lifecycle"
DEFAULT_CURSOR_SKILL = ROOT.parent / ".cursor/skills/mei-51m-cpt-lifecycle"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_file(path: Path, cache: dict[Path, str] | None = None) -> str:
    path = path.resolve()
    if cache is not None and path in cache:
        return cache[path]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    if cache is not None:
        cache[path] = value
    return value


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


def _is_workspace_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.parent.resolve())
        return True
    except ValueError:
        return False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _verify_hash_map(
    values: Mapping[str, str],
    *,
    cache: dict[Path, str],
    label: str,
) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    for raw_path, expected in sorted(values.items()):
        path = Path(raw_path).resolve()
        _require(_is_workspace_path(path), f"{label} escapes workspace: {path}")
        _require(path.is_file(), f"{label} missing file: {path}")
        observed = _sha_file(path, cache)
        _require(observed == expected, f"{label} hash mismatch: {path}")
        verified.append({"path": str(path), "sha256": observed})
    return verified


def _skill_files(root: Path) -> dict[str, Path]:
    _require(root.is_dir(), f"skill root missing: {root}")
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def _verify_skill_mirror(
    canonical_root: Path,
    cursor_root: Path,
    *,
    cache: dict[Path, str],
) -> dict[str, Any]:
    canonical = _skill_files(canonical_root.resolve())
    cursor = _skill_files(cursor_root.resolve())
    _require(set(canonical) == set(cursor), "canonical/Cursor skill file sets differ")
    files: dict[str, dict[str, str]] = {}
    for relative in sorted(canonical):
        canonical_sha = _sha_file(canonical[relative], cache)
        cursor_sha = _sha_file(cursor[relative], cache)
        _require(canonical_sha == cursor_sha, f"skill mirror drift: {relative}")
        files[relative] = {
            "canonical_path": str(canonical[relative].resolve()),
            "cursor_path": str(cursor[relative].resolve()),
            "sha256": canonical_sha,
        }
    return {
        "byte_identical": True,
        "file_count": len(files),
        "files": files,
    }


def _verify_receipt(
    path: Path,
    *,
    cache: dict[Path, str],
    expected_sha256: str | None = None,
    expected_terminal: str | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    _require(_is_workspace_path(path), f"receipt escapes workspace: {path}")
    _require(path.is_file(), f"missing receipt: {path}")
    observed_sha = _sha_file(path, cache)
    if expected_sha256 is not None:
        _require(observed_sha == expected_sha256, f"receipt hash mismatch: {path}")
    receipt = _load_json(path)
    terminal = str(receipt.get("terminal_status") or "")
    if expected_terminal is not None:
        _require(terminal == expected_terminal, f"receipt terminal drift: {path}")
    else:
        _require(terminal in {"passed", "degraded"}, f"non-terminal receipt: {path}")
    outputs = _verify_hash_map(
        receipt.get("output_hashes") or {}, cache=cache, label=f"{path} output"
    )
    predecessor = receipt.get("input_artifacts") or {}
    predecessor_sha = predecessor.get("predecessor_receipt_sha256")
    predecessor_stage = predecessor.get("predecessor_stage")
    if predecessor_sha and predecessor_stage:
        predecessor_path = path.parent.parent / str(predecessor_stage) / "receipt.json"
        if predecessor_path.is_file():
            _require(
                _sha_file(predecessor_path, cache) == predecessor_sha,
                f"predecessor receipt hash mismatch: {predecessor_path}",
            )
    return {
        "path": str(path),
        "sha256": observed_sha,
        "stage_id": receipt.get("stage_id"),
        "terminal_status": terminal,
        "verified_output_count": len(outputs),
    }


def _verify_base(
    release_path: Path,
    *,
    cache: dict[Path, str],
    minimum_exposure: int,
) -> dict[str, Any]:
    release_path = release_path.resolve()
    release = _load_json(release_path)
    _require(release.get("params") == 51_463_797, f"wrong Base params: {release_path}")
    exposure = int(release.get("tokens_seen_exposure") or 0)
    _require(exposure >= minimum_exposure, f"Base exposure too small: {release_path}")
    model_id = str(release.get("model_id"))
    weights_name = str(release.get("weights") or f"{model_id}.npz")
    weights_path = release_path.parent / weights_name
    state_path = release_path.parent / f"{model_id}-state.npz"
    _require(weights_path.is_file(), f"Base weights missing: {weights_path}")
    _require(state_path.is_file(), f"Base state missing: {state_path}")
    weights_sha = _sha_file(weights_path, cache)
    state_sha = _sha_file(state_path, cache)
    _require(weights_sha == release.get("weights_sha256"), f"Base weights drift: {weights_path}")
    _require(state_sha == release.get("state_sha256"), f"Base state drift: {state_path}")
    return {
        "release": str(release_path),
        "release_sha256": _sha_file(release_path, cache),
        "model_id": model_id,
        "exposure_tokens": exposure,
        "params": 51_463_797,
        "weights": str(weights_path.resolve()),
        "weights_sha256": weights_sha,
        "state": str(state_path.resolve()),
        "state_sha256": state_sha,
    }


def _verify_product_contract(label: str, product: Mapping[str, Any]) -> dict[str, Any]:
    for key, steps in EXPECTED_STAGE_STEPS.items():
        _require(
            (product.get(key) or {}).get("steps") == steps,
            f"{label} {key} steps are not {steps}",
        )
    qat = product.get("cq2_qat") or {}
    _require(qat.get("tokens_seen_qat") == 5_001_216, f"{label} QAT exposure drift")
    _require(
        qat.get("quant_math_id") == "mei-cq-v2-g128-wht-codebook",
        f"{label} CQ2 math drift",
    )
    mw = product.get("mw_disposition") or {}
    _require((mw.get("train") or {}).get("steps") == 2_000, f"{label} MW steps drift")
    confidence = product.get("confidence") or {}
    _require((confidence.get("head") or {}).get("steps") == 800, f"{label} confidence steps drift")
    for split in ("train", "valid", "dev"):
        row = (confidence.get("harvest") or {}).get(split) or {}
        _require(row.get("positive", 0) > 0 and row.get("negative", 0) > 0, f"{label} confidence {split} lacks both classes")
    narration = (product.get("narration") or {}).get("adapter") or {}
    _require(narration.get("rank") == 16, f"{label} narration rank drift")
    _require(narration.get("steps") == 1_200, f"{label} narration steps drift")
    _require(narration.get("parameter_count") == 392_192, f"{label} narration params drift")
    _require(narration.get("terminal_only") is True, f"{label} narration is not terminal-only")
    _require(narration.get("verified_result_only") is True, f"{label} narration is not verified-result-only")
    package = product.get("package") or {}
    _require(package.get("weight_parameter_count") == 51_463_797, f"{label} package LM identity drift")
    _require(package.get("mtp_tensor_count") == 0, f"{label} package contains MTP tensors")
    _require(package.get("package_within_limit") is True, f"{label} package exceeds 18 MiB")
    python_runtime = product.get("python_runtime") or {}
    profile = (python_runtime.get("engine_capabilities") or {}).get("runtime_profile") or {}
    _require(profile.get("candidate_batch_size") == 5, f"{label} candidate batch size drift")
    _require(profile.get("max_context_tokens") == 2_048, f"{label} context limit drift")
    _require(profile.get("ordinary_window_policy") == "dynamic_remainder", f"{label} dynamic KV missing")
    _require(profile.get("default_output_tokens") == 128, f"{label} output reserve drift")
    _require((python_runtime.get("engine_capabilities") or {}).get("stateful_tool_loop") is True, f"{label} Agent loop missing")
    budget = python_runtime.get("joint_budget_stress") or {}
    _require(budget.get("budget_planning_ok") is True, f"{label} joint budget gate failed")
    _require(budget.get("output_reserve_tokens") == 128, f"{label} budget output reserve drift")
    _require(budget.get("prompt_cap_tokens") == 1_920, f"{label} budget prompt cap drift")
    _require(python_runtime.get("tool_schema_budget_exception") is False, f"{label} schema budget exception")
    browser = product.get("browser_wasm") or {}
    _require(browser.get("functional_complete") is True, f"{label} Browser-WASM incomplete")
    _require(browser.get("actual_chromium_worker") is True, f"{label} Browser gate was not Chromium Worker")
    _require(browser.get("numeric_forward_ran") is True, f"{label} Browser numeric forward missing")
    _require(browser.get("bounded_int8_cache_ran") is True, f"{label} Browser bounded int8 KV missing")
    _require(browser.get("heap_within_96_mib") is True, f"{label} Browser heap gate failed")
    terminal = product.get("terminal") or {}
    _require(terminal.get("process_complete") is True, f"{label} product not process-complete")
    return {
        "fullcall_steps": 2_000,
        "agent_steps": 1_000,
        "retrieval_r2_steps": 1_600,
        "mw_steps": 2_000,
        "confidence_steps": 800,
        "narration_steps": 1_200,
        "package_bytes": package.get("package_bytes"),
        "python_warm_decode_tok_s": (python_runtime.get("warm_decode") or {}).get("tokens_per_second"),
        "browser_warm_decode_tok_s": browser.get("warm_steady_decode_tok_s"),
        "browser_heap_bytes": browser.get("heap_peak_bytes"),
        "process_complete": True,
        "release_eligible": terminal.get("release_eligible"),
    }


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    cache: dict[Path, str] = {}
    verifier_source = Path(__file__).resolve()
    paired_root = args.paired_root.resolve()
    final_path = paired_root / "paired-final-audit.json"
    deliverables_path = paired_root / "deliverables-receipt.json"
    comparison_path = paired_root / "paired-comparison.json"
    alignment_path = paired_root / "needle2-alignment.json"
    history_path = paired_root / "longitudinal-history-index.json"
    supervisor_path = paired_root / "supervisor-receipt.json"
    for path in (
        final_path,
        deliverables_path,
        comparison_path,
        alignment_path,
        history_path,
        supervisor_path,
    ):
        _require(path.is_file(), f"paired deliverable missing: {path}")

    final = _load_json(final_path)
    deliverables = _load_json(deliverables_path)
    comparison = _load_json(comparison_path)
    alignment = _load_json(alignment_path)
    history = _load_json(history_path)
    supervisor = _load_json(supervisor_path)

    _require(final.get("process_complete") is True, "paired final audit is not process-complete")
    _require(supervisor.get("process_complete") is True, "supervisor is not process-complete")
    _require(deliverables.get("process_complete") is True, "deliverables are not process-complete")
    f_items = final.get("f01_f14") or []
    _require([row.get("id") for row in f_items] == EXPECTED_F_IDS, "F01--F14 boundary drift")
    _require(
        all(row.get("status") in {"passed", "degraded"} for row in f_items),
        "one or more F01--F14 items lack a terminal status",
    )
    _require(final.get("current_unchanged") is True, "paired final audit reports CURRENT drift")
    _require(final.get("base_mutated") is False, "paired final audit reports Base mutation")
    _require(final.get("cpt_or_qat_retrained") is False, "paired flow retrained CPT/QAT")
    _require(final.get("old_blocked_runs_preserved") is True, "old blocked runs not preserved")
    _require(final.get("cactus_compatibility_claimed") is False, "Cactus compatibility was claimed")

    current_sha = _sha_file(CURRENT_PATH, cache)
    _require(current_sha == args.expected_current_sha256, "CURRENT.json changed")
    _require(comparison.get("all_comparison_contracts_equal") is True, "paired contracts differ")
    _require(
        all((comparison.get("contract_checks") or {}).values()),
        "one or more paired comparison contract checks failed",
    )
    _require(alignment.get("mechanism_alignment_complete") is True, "Needle2 mechanism matrix incomplete")
    _require(
        set(alignment.get("explicit_non_compatibility") or [])
        == {".cact package format", "libneedle binary", "Cactus API", "Cactus ABI"},
        "explicit Needle2 non-compatibility boundary drift",
    )
    _require(history.get("append_only") is True, "longitudinal history is not append-only")
    _require(
        _sha_file(Path(history["ledger"]), cache) == history.get("ledger_sha256"),
        "longitudinal ledger hash mismatch",
    )

    deliverable_files = _verify_hash_map(
        {
            str(value["path"]): str(value["sha256"])
            for value in (deliverables.get("artifacts") or {}).values()
        },
        cache=cache,
        label="paired deliverable",
    )

    receipt_records: dict[str, dict[str, Any]] = {}
    for label, product in sorted((comparison.get("products") or {}).items()):
        for stage_id, evidence in sorted((product.get("evidence") or {}).items()):
            path = Path(evidence["path"]).resolve()
            receipt_records[str(path)] = _verify_receipt(
                path,
                cache=cache,
                expected_sha256=str(evidence["sha256"]),
                expected_terminal=str(evidence["terminal_status"]),
            )
        product_run = Path(product["run_dir"]).resolve()
        adoption_path = product_run / "stages/adopt_packaged_run_v5/receipt.json"
        adoption = _verify_receipt(adoption_path, cache=cache)
        adoption_metrics = _load_json(adoption_path).get("metrics") or {}
        _require(adoption_metrics.get("lm_or_head_retrained") is False, f"{label} adoption retrained heads/LM")
        _require(adoption_metrics.get("package_rebuilt") is False, f"{label} adoption rebuilt package")
        _require(adoption_metrics.get("cpt_retrained") is False, f"{label} adoption retrained CPT")
        _require(adoption_metrics.get("qat_retrained") is False, f"{label} adoption retrained QAT")
        blocked = adoption_metrics.get("blocked_boundary")
        if blocked:
            blocked_path = Path(blocked["receipt"]).resolve()
            _verify_receipt(
                blocked_path,
                cache=cache,
                expected_sha256=str(blocked["receipt_sha256"]),
                expected_terminal="blocked",
            )
        receipt_records[str(adoption_path)] = adoption

    products = comparison.get("products") or {}
    _require(set(products) == {"300m", "600m"}, "paired products are not exactly 300m/600m")
    product_contracts = {
        label: _verify_product_contract(label, product)
        for label, product in sorted(products.items())
    }
    bases = {
        "300m": _verify_base(
            Path(products["300m"]["base"]["release"]),
            cache=cache,
            minimum_exposure=300_000_000,
        ),
        "600m": _verify_base(
            Path(products["600m"]["base"]["release"]),
            cache=cache,
            minimum_exposure=600_000_000,
        ),
    }

    guide_text = args.guide.read_text(encoding="utf-8")
    _require(all(f"`{item}`" in guide_text for item in EXPECTED_F_IDS), "guide lacks F01--F14")
    _require("目标锁定句" in guide_text, "guide lacks the locked objective")
    _require("process_complete=true / release_eligible=false" in guide_text, "guide lacks terminal semantics")
    skill_mirror = _verify_skill_mirror(
        args.canonical_skill, args.cursor_skill, cache=cache
    )
    governance = {
        "guide": {"path": str(args.guide.resolve()), "sha256": _sha_file(args.guide, cache)},
        "agents": {"path": str(args.agents.resolve()), "sha256": _sha_file(args.agents, cache)},
        "skill_mirror": skill_mirror,
        "single_ssot": str(args.guide.resolve()),
    }

    git_head = _git_head()
    _require(git_head == args.expected_git_head, "Git HEAD changed from the protected baseline")
    requirements = [
        {"id": "R01", "requirement": "F01--F14 terminal boundary", "status": "passed", "evidence": str(final_path)},
        {"id": "R02", "requirement": "paired deliverable hashes", "status": "passed", "evidence_count": len(deliverable_files)},
        {"id": "R03", "requirement": "all referenced stage receipts and outputs", "status": "passed", "receipt_count": len(receipt_records)},
        {"id": "R04", "requirement": "immutable 300M/600M Base identity", "status": "passed", "bases": bases},
        {"id": "R05", "requirement": "CPT/QAT not retrained; packaged adoption hash-bound", "status": "passed"},
        {"id": "R06", "requirement": "Full-call/Agent/R2/MW/confidence/narration stage budgets", "status": "passed", "products": product_contracts},
        {"id": "R07", "requirement": "fixed-five, 2048 joint budget, dynamic KV and Agent loop", "status": "passed"},
        {"id": "R08", "requirement": "unified CQ2 v2 packages and Browser/Python gates", "status": "passed"},
        {"id": "R09", "requirement": "same-contract paired comparison and history", "status": "passed", "evidence": str(comparison_path)},
        {"id": "R10", "requirement": "Chinese Needle2 mechanism matrix and explicit non-compatibility", "status": "passed", "evidence": str(alignment_path)},
        {"id": "R11", "requirement": "AGENTS/SSOT/canonical-Cursor skill governance", "status": "passed", "governance": governance},
        {"id": "R12", "requirement": "CURRENT/Base/old runs protected and Git HEAD unchanged", "status": "passed", "current_sha256": current_sha, "git_head": git_head},
    ]
    plan_inputs = {
        "paired_root": str(paired_root),
        "paired_final_audit_sha256": _sha_file(final_path, cache),
        "deliverables_receipt_sha256": _sha_file(deliverables_path, cache),
        "comparison_sha256": _sha_file(comparison_path, cache),
        "alignment_sha256": _sha_file(alignment_path, cache),
        "history_index_sha256": _sha_file(history_path, cache),
        "supervisor_receipt_sha256": _sha_file(supervisor_path, cache),
        "current_sha256": current_sha,
        "guide_sha256": governance["guide"]["sha256"],
        "agents_sha256": governance["agents"]["sha256"],
        "skill_files": {
            relative: row["sha256"]
            for relative, row in skill_mirror["files"].items()
        },
        "verifier_source_sha256": _sha_file(verifier_source, cache),
        "git_head": git_head,
    }
    audit_fingerprint = hashlib.sha256(_canonical(plan_inputs)).hexdigest()
    return {
        "schema": SCHEMA,
        "terminal_status": "passed",
        "process_complete": True,
        "release_eligible": bool(final.get("release_eligible")),
        "release_ineligible_reasons": final.get("release_ineligible_reasons") or [],
        "audit_fingerprint_sha256": audit_fingerprint,
        "requirements": requirements,
        "f01_f14": f_items,
        "products": product_contracts,
        "verified_stage_receipts": sorted(receipt_records.values(), key=lambda row: row["path"]),
        "verified_unique_file_count": len(cache),
        "governance": governance,
        "protected_state": {
            "current_sha256": current_sha,
            "current_unchanged": True,
            "base_mutated": False,
            "old_blocked_runs_preserved": True,
            "cpt_or_qat_retrained": False,
            "git_head": git_head,
            "git_commit_performed": False,
            "git_push_performed": False,
            "public_release_performed": False,
            "cactus_compatibility_claimed": False,
        },
        "plan_inputs": plan_inputs,
        "source": {
            "path": str(verifier_source),
            "sha256": _sha_file(verifier_source, cache),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-root", type=Path, default=DEFAULT_PAIRED_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--agents", type=Path, default=DEFAULT_AGENTS)
    parser.add_argument("--canonical-skill", type=Path, default=DEFAULT_CANONICAL_SKILL)
    parser.add_argument("--cursor-skill", type=Path, default=DEFAULT_CURSOR_SKILL)
    parser.add_argument("--expected-current-sha256", default=EXPECTED_CURRENT_SHA256)
    parser.add_argument("--expected-git-head", default=EXPECTED_GIT_HEAD)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    _require(
        not out_dir.exists() or not any(out_dir.iterdir()),
        f"completion audit output already exists; choose a new directory: {out_dir}",
    )
    audit = build_audit(args)
    if args.dry_run:
        return audit
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "completion-audit.json"
    _atomic_json(audit_path, audit)
    receipt = {
        "schema": "mei-51m-adaptive-v5-goal-completion-receipt-v1",
        "terminal_status": "passed",
        "process_complete": True,
        "release_eligible": audit["release_eligible"],
        "audit": str(audit_path),
        "audit_sha256": _sha_file(audit_path),
        "audit_fingerprint_sha256": audit["audit_fingerprint_sha256"],
        "current_sha256": _sha_file(CURRENT_PATH),
        "current_unchanged": True,
    }
    receipt_path = out_dir / "receipt.json"
    _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": _sha_file(receipt_path)}


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
