"""Prepare an immutable Simplified Chinese/English successor without conversion."""
from collections import Counter
from pathlib import Path
import copy
import json
import shutil

from profiling import resolve_path, ROOT, digest
from tokenizer_candidate import LosslessProcessor, rows, write_new


def run(config, out):
    out = resolve_path(ROOT / out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "implementation.py.snapshot")
    write_new(out / "config.json", config)
    reserved = set(config["reserved_source_ids"])
    old_recipe = json.loads((resolve_path(ROOT / config["candidate_recipe"])).read_text())
    ledger = {"policy": "reserve dedicated sources; preserve all original characters", "splits": {}}
    groups = {}
    for split in ("train", "dev"):
        source = resolve_path(ROOT / old_recipe[split + "_jsonl"])
        counts, removed, split_groups = Counter(), Counter(), set()
        with (out / (split + ".jsonl")).open("x") as handle:
            for row in rows(source):
                sid = row["source_id"]
                if sid in reserved:
                    removed[sid] += len(row["text"])
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[sid] += len(row["text"])
                split_groups.add(row["group"])
        groups[split] = split_groups
        ledger["splits"][split] = {"parent": str(source.relative_to(ROOT)),
            "parent_sha256": digest(source), "characters": sum(counts.values()),
            "characters_by_source": dict(counts), "reserved_characters_by_source": dict(removed)}
    if groups["train"] & groups["dev"]:
        raise ValueError("sample association groups overlap")
    ledger["training_target_characters"] = 20_000_000
    ledger["training_shortfall_characters"] = max(0, 20_000_000-ledger["splits"]["train"]["characters"])
    ledger["sampling_decision"] = "Reuse existing isolated fragments once; do not duplicate to fill sample budget. Domain shares are actual, not claimed target shares."
    write_new(out / "sample-manifest.json", ledger)
    candidate = copy.deepcopy(old_recipe)
    candidate["tokenizer_id"] = config["tokenizer_id"]
    for key, filename in [("train_jsonl", "train.jsonl"), ("dev_jsonl", "dev.jsonl"), ("sample_manifest", "sample-manifest.json")]:
        candidate[key] = str((out / filename).relative_to(ROOT))
    candidate["input_sha256"] = {key: digest(out / filename) for key, filename in
                                 [("train", "train.jsonl"), ("dev", "dev.jsonl"), ("manifest", "sample-manifest.json")]}
    write_new(out / "candidate-config.json", candidate)
    parent = resolve_path(ROOT / config["corpus_plan"])
    plan = json.loads(parent.read_text())
    for key in ["frozen_release", "final_tokenizer", "frozen_count_basis", "selected_net_tokens", "quality_admitted_tokens", "candidate_total_tokens", "tokenizer_candidates"]:
        plan.pop(key, None)
    plan.update(plan_id=config["plan_id"], status="pending_successor_tokenizer_and_encoding",
                parent_plan={"path": config["corpus_plan"], "sha256": digest(parent)},
                reserved_source_ids=sorted(reserved), language_weights={"foundation": {"zh_hans": 8, "en": 1}, "oral": {"zh_hans": 1}},
                count_basis="Successor counts pending; parent inventory counts only describe old tokenizer.")
    for domain in plan["domains"]:
        domain["source_ids"] = [s for s in domain["source_ids"] if s not in reserved]
        domain.pop("selected_net_tokens", None)
        domain.pop("language_target_tokens", None)
        domain["language_policy"] = "Dedicated Traditional sources reserved; incidental characters retained without conversion."
        if domain["domain"] in ("foundation", "oral"):
            domain["planning_target_tokens"] = domain["planning_target_tokens"] * 9 // 10
    cumulative = 0
    for phase in plan["phases"]:
        for domain in ("foundation", "oral"):
            phase["domain_target_tokens"][domain] = phase["domain_target_tokens"][domain] * 9 // 10
        for key in ("base_name", "actual_incremental_tokens", "actual_cumulative_tokens", "selected_manifest"):
            phase.pop(key, None)
        phase["incremental_target_tokens"] = sum(phase["domain_target_tokens"].values())
        cumulative += phase["incremental_target_tokens"]
        phase["cumulative_target_tokens"] = cumulative
    plan["planning_target_tokens"] = sum(d["planning_target_tokens"] for d in plan["domains"])
    plan["allocation_status"] = "Removed Traditional quota; no automatic cross-domain backfill; new tokenizer measured counts pending."
    plan["user_decisions"].update(natural_language_ratio=[8, 1],
        natural_language_ratio_status="Simplified Chinese:English 8:1 within foundation; Traditional dedicated sources reserve only; oral prioritizes Simplified.",
        cumulative_base_targets=[p["cumulative_target_tokens"] for p in plan["phases"]])
    plan["tokenizer_selection_status"] = "pending_successor_audit"
    plan["blockers"] = ["Successor candidate, browser audit and new encoding must pass before inputs_ready."]
    write_new(out / "corpus-plan.json", plan)
    freeze = json.loads((resolve_path(ROOT / config["freeze_recipe"])).read_text())
    freeze.update(release_id=config["release_id"],
        tokenizer_manifest=f'models/mei-1.2-51m/tokenizer/candidates/{config["tokenizer_id"]}/RELEASE.json',
        tokenizer_train_sample=candidate["train_jsonl"], tokenizer_dev_sample=candidate["dev_jsonl"],
        corpus_plan=str((out / "corpus-plan.json").relative_to(ROOT)),
        release_dir=f'corpus/pools/{config["release_id"]}')
    write_new(out / "freeze-config.json", freeze)
    processor = LosslessProcessor(resolve_path(ROOT / config["baseline_model"]))
    diagnostics = {"measurement": "Compact JSON pilot view, not deployed serializer. First five tools are positional size diagnostics, not retrieval or model success.", "sources": {}}
    def size(value):
        return len(processor.encode(json.dumps(value, ensure_ascii=False, separators=(",", ":"))))
    for name, rel in config["pilots"].items():
        cases = []
        for row in rows_pilot(resolve_path(ROOT / rel)):
            messages, tools = row["messages"], row["visible_tools"]
            view = {"messages": messages, "tools": tools, "target": row["gold_lm_target"]}
            first = {**view, "tools": tools[:5]}
            cases.append({"case_id": row["case_id"], "tool_count": len(tools), "message_count": len(messages),
                "full": size(view), "first_five": size(first), "tools_only": size(tools),
                "messages_only": size(messages), "system_only": size([m for m in messages if m["role"] == "system"]),
                "non_system_only": size([m for m in messages if m["role"] != "system"])})
        def stats(key):
            lengths = sorted(c[key] for c in cases)
            return {"median": lengths[len(lengths)//2], "p95": lengths[min(len(lengths)-1,int(len(lengths)*.95))], "max": max(lengths), "over_1920": sum(n>1920 for n in lengths)}
        diagnostics["sources"][name] = {"records": len(cases), "tool_counts": dict(Counter(c["tool_count"] for c in cases)),
            "metrics": {key: stats(key) for key in ("full", "first_five", "tools_only", "messages_only", "system_only", "non_system_only")},
            "cases": cases, "parent_sha256": digest(resolve_path(ROOT / rel))}
    write_new(out / "context-budget-breakdown.json", diagnostics)
    return {"status": "scope_prepared", "sample": ledger, "planning_target_tokens": plan["planning_target_tokens"],
            "budget": {k: {a:b for a,b in v.items() if a != "cases"} for k,v in diagnostics["sources"].items()}}


def rows_pilot(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)
