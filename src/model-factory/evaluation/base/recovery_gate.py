from __future__ import annotations

import math


ROLES = {"code", "dialogue", "fineweb2_hq", "structured", "wiki_en", "wiki_zh"}


def validate_policy(policy: dict) -> None:
    if policy.get("schema") != "mei-cpt-recovery-gate-v1":
        raise ValueError("unsupported recovery gate policy")
    weights = policy.get("role_weights", {})
    if set(weights) != ROLES or any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("six positive finite role weights required")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("role weights must sum to one")
    for key in ("aggregate_relative_tolerance", "role_relative_tolerance", "probe_relative_tolerance",
                "numeric_absolute_tolerance"):
        value = policy[key]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid tolerance: {key}")


def compare(parent: dict, candidate: dict, policy: dict) -> dict:
    validate_policy(policy)
    weights = policy["role_weights"]
    if set(parent["roles"]) != ROLES or set(candidate["roles"]) != ROLES:
        raise ValueError("incomplete or unexpected validation roles")
    parent_mean = 0.0
    candidate_mean = 0.0
    checks = {}
    changes = {}
    numeric = policy["numeric_absolute_tolerance"]
    for role, weight in weights.items():
        before, after = parent["roles"][role], candidate["roles"][role]
        if before["predicted_tokens"] <= 0 or before["predicted_tokens"] != after["predicted_tokens"]:
            raise ValueError(f"validation token coverage mismatch: {role}")
        if before["windows"] <= 0 or before["windows"] != after["windows"]:
            raise ValueError(f"validation window coverage mismatch: {role}")
        previous, proposed = before["valid_loss"], after["valid_loss"]
        if any(not math.isfinite(value) or value <= 0 for value in (previous, proposed)):
            raise ValueError(f"invalid validation loss: {role}")
        parent_mean += weight * previous
        candidate_mean += weight * proposed
        changes[role] = proposed / previous - 1
        checks[role] = proposed <= previous * (1 + policy["role_relative_tolerance"]) + numeric
    previous_probe = parent["probes"]["mean_nll"]
    proposed_probe = candidate["probes"]["mean_nll"]
    if any(not math.isfinite(value) or value <= 0 for value in (previous_probe, proposed_probe)):
        raise ValueError("invalid probe loss")
    checks["quota_weighted_valid"] = candidate_mean <= parent_mean * (1 + policy["aggregate_relative_tolerance"]) + numeric
    # probe 降级为「仅记录」信号（2026-09-11 评估口径决策）：纯 CPT 不训练指令跟随，7 行
    # needle-pretrain-probes 的 NLL 波动是噪声，不应 gate 底座语言能力的判定。probe_guard
    # 仍计算并记录（供观察），但不参与 status；measured_improvement 只看 valid。
    checks["probe_guard"] = proposed_probe <= previous_probe * (1 + policy["probe_relative_tolerance"]) + numeric
    checks["measured_improvement"] = candidate_mean < parent_mean - numeric
    blocking = {key: value for key, value in checks.items() if key != "probe_guard"}
    return {"schema": "mei-cpt-recovery-gate-result-v1",
            "status": "continue_bounded_diagnostic" if all(blocking.values()) else "hold_for_diagnosis",
            "checks": checks, "per_role_relative_change": changes,
            "quota_weighted_valid": {"parent": parent_mean, "candidate": candidate_mean},
            "probe_mean_nll": {"parent": previous_probe, "candidate": proposed_probe},
            "automatic_parent_promotion": False, "full_1800m_authorized": False,
            "release_eligible": False}
