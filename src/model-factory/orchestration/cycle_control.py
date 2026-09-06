from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mei_llm.registry import Registry


PHASES = [
    {
        "id": "sourcing",
        "requires": ["parent exposure", "natural pool inventory"],
    },
    {
        "id": "corpus_factory",
        "requires": ["candidate mix", "registered family targets"],
    },
    {
        "id": "corpus_quality",
        "requires": ["source/synthetic/SFT receipts", "reuse decision"],
    },
    {
        "id": "cpt",
        "pipeline": "mei-51m-cpt-lifecycle-v1",
        "requires": ["frozen CPT delta", "quality passed", "parent Base"],
    },
    {
        "id": "productization",
        "pipeline": "mei-51m-adaptive-productization-v5",
        "requires": ["frozen Base candidate", "SFT release", "eval lock"],
    },
    {
        "id": "finalize",
        "requires": ["final audit", "independent eligibility axes"],
    },
]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _entry(registry: Registry, cycle_id: str) -> dict[str, Any]:
    rows = [
        row
        for row in registry.cycles().get("cycles", [])
        if row.get("cycle_id") == cycle_id
    ]
    if len(rows) != 1:
        raise KeyError(f"unknown or duplicate cycle: {cycle_id}")
    return rows[0]


def _previous(
    registry: Registry, target: int, lineage: str | None = None
) -> dict[str, Any] | None:
    rows = [
        row
        for row in registry.cycles().get("cycles", [])
        if int(row.get("target_exposure_tokens") or 0) < target
        and (lineage is None or row.get("lineage") == lineage)
    ]
    return max(rows, key=lambda row: int(row["target_exposure_tokens"])) if rows else None


def plan(registry: Registry, cycle_id: str) -> dict[str, Any]:
    row = _entry(registry, cycle_id)
    target = int(row["target_exposure_tokens"])
    parent = _previous(registry, target, row.get("lineage"))
    parent_actual = int(
        (parent or {}).get("actual_exposure_tokens")
        or (parent or {}).get("target_exposure_tokens")
        or 0
    )
    return {
        "schema": "mei-51m-cycle-plan-v1",
        "cycle_id": cycle_id,
        "registry_status": row["status"],
        "target_exposure_tokens": target,
        "parent_cycle": (parent or {}).get("cycle_id"),
        "parent_actual_exposure_tokens": parent_actual,
        "required_increment_tokens": target - parent_actual,
        "phases": PHASES,
        "materialize_cycle_directory": False,
        "current_sha256": registry.current_sha256(),
    }


def init_proposal(
    registry: Registry, cycle_id: str, *, out: Path | None = None
) -> dict[str, Any]:
    result = plan(registry, cycle_id)
    if not str(result["registry_status"]).startswith("planned"):
        raise RuntimeError("cycle init proposal is only valid for a planned cycle")
    result = {
        **result,
        "schema": "mei-51m-cycle-init-proposal-v1",
        "status": "ready_for_sourcing",
        "write_scope": "proposal_only",
    }
    if out is not None:
        encoded = (
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite cycle proposal: {out}")
        out.write_bytes(encoded)
    return result


def status(registry: Registry, cycle_id: str) -> dict[str, Any]:
    row = _entry(registry, cycle_id)
    result: dict[str, Any] = {
        "schema": "mei-51m-cycle-status-v1",
        "cycle_id": cycle_id,
        "registry": row,
        "current_sha256": registry.current_sha256(),
    }
    path_value = row.get("path")
    if not path_value:
        result.update({"materialized": False, "next_phase": "sourcing"})
        return result
    path = registry.root / str(path_value)
    contract = _load(path)
    result.update(
        {
            "materialized": True,
            "contract_path": str(path),
            "contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "contract": contract,
            "next_phase": (
                "terminal"
                if contract.get("process_complete") is True
                else "inspect_cycle_contract"
            ),
        }
    )
    return result


def verify(registry: Registry) -> dict[str, Any]:
    errors: list[str] = []
    cycle_ids: set[str] = set()
    targets_by_lineage: dict[str, set[int]] = {}
    for row in registry.cycles().get("cycles", []):
        cycle_id = str(row.get("cycle_id") or "")
        target = int(row.get("target_exposure_tokens") or 0)
        if cycle_id in cycle_ids:
            errors.append(f"duplicate cycle id: {cycle_id}")
        lineage = str(row.get("lineage") or "legacy")
        targets = targets_by_lineage.setdefault(lineage, set())
        if target in targets:
            errors.append(f"duplicate target exposure in lineage {lineage}: {target}")
        targets.add(target)
        cycle_ids.add(cycle_id)
        if target <= 0:
            errors.append(f"invalid target exposure: {cycle_id}")
        if "path" not in row:
            if not str(row.get("status") or "").startswith("planned"):
                errors.append(f"unmaterialized non-planned cycle: {cycle_id}")
            continue
        path = registry.root / str(row["path"])
        if not path.is_file():
            errors.append(f"missing cycle contract: {path}")
            continue
        contract = _load(path)
        if contract.get("cycle_id") != cycle_id:
            errors.append(f"cycle id mismatch: {path}")
        if int(contract.get("target_exposure_tokens") or 0) != target:
            errors.append(f"target mismatch: {path}")
        pipeline_lock = contract.get("pipeline_lock")
        if not pipeline_lock or not (registry.root / pipeline_lock).is_file():
            errors.append(f"missing pipeline lock: {cycle_id}")
        eligibility = contract.get("eligibility") or {}
        for key in (
            "continuation_checkpoint_eligible",
            "automatic_parent_promotion_eligible",
            "corpus_reuse_eligible",
        ):
            if key not in eligibility:
                errors.append(f"missing eligibility axis {key}: {cycle_id}")
    return {
        "schema": "mei-51m-cycle-registry-verification-v1",
        "ok": not errors,
        "cycle_count": len(cycle_ids),
        "errors": errors,
        "current_sha256": registry.current_sha256(),
    }
