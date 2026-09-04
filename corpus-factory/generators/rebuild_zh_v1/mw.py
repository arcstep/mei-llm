#!/usr/bin/env python3
"""20-class MW disposition generator.

Consumes frozen per-batch visible views (never live retrieval) per
AGENTS.md: for `capability_insufficient` the effective label training target
is produced by constructing a visible 5-tool batch that deliberately excludes
the target tool; for `ready_to_execute` the batch includes it. All 20 classes
get direct coverage plus boundary minimal pairs sourced from
mw-reason-definitions-v2-20class.json's `neighbors` graph.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from . import common as C
from . import mw_scenarios as MS
from . import retrieval as R
from . import text_variants as tv


def _diversify(query: str, seed: str) -> str:
    """mw_scenarios builders (other than class 0) emit one fixed template per
    class -- several of them (missing_external_fact, ambiguous_scope,
    deixis_unresolved, correction_incomplete, safety_judgment,
    unsupported_scope, injection_rejected, offtopic, negation_cancels,
    partial_sequence_blocked) don't even vary by seed, only by tool. At
    per-class counts above the ~147-tool deploy universe, cycling through
    tools alone plus a single variant pick (7 kinds) is not enough entropy,
    so stack two independently-seeded transforms for ~7x7 combinations."""

    kind_a = tv.VARIANT_KINDS[int(C.sha256_text(seed + ":a")[:6], 16) % len(tv.VARIANT_KINDS)]
    kind_b = tv.VARIANT_KINDS[int(C.sha256_text(seed + ":b")[:6], 16) % len(tv.VARIANT_KINDS)]
    out = tv.build_variant(query, kind_a, seed=seed + ":a")
    if kind_b != kind_a:
        out = tv.build_variant(out, kind_b, seed=seed + ":b")
    return out

FAMILY = "mw_disposition"


def _visible_batch(deploy: C.ToolRegistry, target: dict[str, Any] | None, seed: str, *, include_target: bool) -> list[str]:
    tools = list(deploy.tools)
    pool = [t for t in tools if not target or t.get("name") != target.get("name")]
    ranked = R._candidates_sorted_by_similarity(target, pool) if target else pool
    rnd = random.Random(seed)
    top = ranked[: min(len(ranked), 12)] or ranked
    rnd.shuffle(top)
    picks = top[:4] if target and include_target else top[:5]
    names = [t.get("name") for t in picks]
    if target and include_target:
        pos = int(C.sha256_text(seed)[:4], 16) % 5
        names.insert(pos, target.get("name"))
        names = names[:5]
    return names


def _budget(deploy: C.ToolRegistry, batch_names: Sequence[str], query: str, context, evidence, history, profile: str) -> C.BudgetResult:
    tools_batch = [deploy.by_name[n] for n in batch_names if n in deploy.by_name]
    _, receipt = C.fit_batch_to_budget(
        tools_batch=tools_batch, query=query, context=context, evidence=evidence,
        history=history, tool_results=[], profile=profile,
    )
    return receipt


def _make_row(
    *, deploy: C.ToolRegistry, reason_code: str, class_id: int, tool: dict[str, Any], scenario: dict[str, Any],
    retrieved_tools: list[str], effective_class_id: int, effective_reason_code: str,
    profile: str, minimal_pair_id: str | None, seed: str,
) -> dict[str, Any]:
    receipt = _budget(deploy, retrieved_tools, scenario["query"], scenario["context"], scenario["evidence"], scenario["history"], profile)
    row = {
        "case_id": C.case_id(FAMILY, reason_code, seed),
        "cf_group": C.cf_group(FAMILY, reason_code, tool.get("name", "none"), seed[-8:]),
        "family": FAMILY,
        "task": "mw_disposition",
        "generator_version": C.GENERATOR_VERSION,
        "candidate_tool": tool.get("name"),
        "query": scenario["query"],
        "context": scenario["context"],
        "evidence": scenario["evidence"],
        "history": scenario["history"],
        "permissions": scenario["permissions"],
        "state": scenario["state"],
        "retrieved_tools": retrieved_tools,
        "raw_reason_code": reason_code,
        "raw_reason_class_id": class_id,
        "reason_code": effective_reason_code,
        "reason_class_id": effective_class_id,
        "oracle_or_learned": "oracle",
        "profile": profile,
        "minimal_pair_id": minimal_pair_id,
        "budget": {
            "profile": receipt.profile, "prompt_tokens": receipt.prompt_tokens, "cap": receipt.cap,
            "fits": receipt.fits, "context_unrepresentable": receipt.context_unrepresentable,
        },
    }
    return row


def gen_class_rows(deploy: C.ToolRegistry, reason_code: str, *, count: int) -> list[dict[str, Any]]:
    tools = list(deploy.tools)
    cls = next(c for c in C.mw_classes() if c["reason_code"] == reason_code)
    builder = MS.BUILDERS[reason_code]
    rows = []
    for i in range(count):
        tool = tools[(hash(reason_code) + i) % len(tools)]
        seed = f"mw:{reason_code}:{i}"
        if reason_code in MS.NEEDS_SIBLING:
            other = tools[(hash(reason_code) + i + 5) % len(tools)]
            scenario = builder(tool, seed, other)
        else:
            scenario = builder(tool, seed)
        if reason_code != "ready_to_execute":  # class 0's builder already diversifies internally
            scenario = dict(scenario, query=_diversify(scenario["query"], seed))
        profile = "compact" if i % 2 == 0 else "standard"

        if reason_code == "capability_insufficient":
            retrieved = _visible_batch(deploy, tool, seed, include_target=False)
            eff_id, eff_code = cls["class_id"], reason_code
        elif reason_code == "ready_to_execute":
            retrieved = _visible_batch(deploy, tool, seed, include_target=True)
            eff_id, eff_code = cls["class_id"], reason_code
        else:
            # Non-scan-sensitive classes: batch visibility does not change
            # the effective label (only class 0 <-> class 10 does).
            include = i % 2 == 0
            retrieved = _visible_batch(deploy, tool, seed, include_target=include)
            eff_id, eff_code = cls["class_id"], reason_code

        rows.append(_make_row(
            deploy=deploy, reason_code=reason_code, class_id=cls["class_id"], tool=tool, scenario=scenario,
            retrieved_tools=retrieved, effective_class_id=eff_id, effective_reason_code=eff_code,
            profile=profile, minimal_pair_id=None, seed=seed,
        ))
    return rows


def gen_class0_batch_visibility_pairs(deploy: C.ToolRegistry, *, count: int) -> list[dict[str, Any]]:
    """The canonical class-0/class-10 minimal pair required by
    adaptive-tool-context.md: identical underlying ready-to-execute query,
    one row with the target visible (effective class 0), one with it
    excluded from the visible batch (effective class 10)."""

    tools = list(deploy.tools)
    rows = []
    cls0 = next(c for c in C.mw_classes() if c["reason_code"] == "ready_to_execute")
    cls10 = next(c for c in C.mw_classes() if c["reason_code"] == "capability_insufficient")
    for i in range(count):
        tool = tools[i % len(tools)]
        seed = f"mw:visibility_pair:{i}"
        scenario = MS.class_0_ready_to_execute(tool, seed)
        pair_id = f"pair:{tool.get('name')}:{i}"

        visible_in = _visible_batch(deploy, tool, seed + ":in", include_target=True)
        rows.append(_make_row(
            deploy=deploy, reason_code="ready_to_execute", class_id=cls0["class_id"], tool=tool, scenario=scenario,
            retrieved_tools=visible_in, effective_class_id=cls0["class_id"], effective_reason_code="ready_to_execute",
            profile="standard", minimal_pair_id=pair_id, seed=seed + ":in",
        ))
        visible_out = _visible_batch(deploy, tool, seed + ":out", include_target=False)
        rows.append(_make_row(
            deploy=deploy, reason_code="ready_to_execute", class_id=cls0["class_id"], tool=tool, scenario=scenario,
            retrieved_tools=visible_out, effective_class_id=cls10["class_id"], effective_reason_code="capability_insufficient",
            profile="standard", minimal_pair_id=pair_id, seed=seed + ":out",
        ))
    return rows


def gen_neighbor_minimal_pairs(deploy: C.ToolRegistry, *, count: int) -> list[dict[str, Any]]:
    tools = list(deploy.tools)
    neighbors = C.mw_neighbors()
    codebook = {c["reason_code"]: c for c in C.mw_classes()}
    codes = [c for c in neighbors if neighbors[c]]
    rows = []
    for i in range(count):
        code_a = codes[i % len(codes)]
        neigh_list = neighbors[code_a]
        code_b = neigh_list[i % len(neigh_list)]
        if code_b not in MS.BUILDERS:
            continue
        tool = tools[i % len(tools)]
        pair_id = f"neighbor_pair:{code_a}:{code_b}:{i}"
        for code in (code_a, code_b):
            builder = MS.BUILDERS[code]
            seed = f"mw:neighbor:{code}:{i}"
            if code in MS.NEEDS_SIBLING:
                other = tools[(i + 9) % len(tools)]
                scenario = builder(tool, seed, other)
            else:
                scenario = builder(tool, seed)
            if code != "ready_to_execute":
                scenario = dict(scenario, query=_diversify(scenario["query"], seed))
            include = code == "ready_to_execute"
            retrieved = _visible_batch(deploy, tool, seed, include_target=include)
            cls = codebook[code]
            rows.append(_make_row(
                deploy=deploy, reason_code=code, class_id=cls["class_id"], tool=tool, scenario=scenario,
                retrieved_tools=retrieved, effective_class_id=cls["class_id"], effective_reason_code=code,
                profile="standard", minimal_pair_id=pair_id, seed=seed,
            ))
    return rows


def generate(deploy: C.ToolRegistry, *, per_class_count: int, visibility_pair_count: int, neighbor_pair_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in C.mw_reason_codes():
        rows += gen_class_rows(deploy, code, count=per_class_count)
    rows += gen_class0_batch_visibility_pairs(deploy, count=visibility_pair_count)
    rows += gen_neighbor_minimal_pairs(deploy, count=neighbor_pair_count)

    for row in rows:
        row["split"] = C.assign_split(row["cf_group"])

    coverage = {
        "total_rows": len(rows),
        "by_effective_class": _count_by(rows, "reason_code"),
        "by_raw_class": _count_by(rows, "raw_reason_code"),
        "by_split": _count_by(rows, "split"),
        "minimal_pairs": sum(1 for r in rows if r.get("minimal_pair_id")),
    }
    return rows, coverage


def _count_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out
