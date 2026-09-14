"""Prepare v1.2 node-task fixtures, audits, and token-reference reserves offline."""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import re
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]

BEHAVIORS = (
    "single_check",
    "independent_checks",
    "numeric_boundary",
    "negation_revision",
    "missing_or_ambiguous",
    "permission_confirmation",
    "later_batch_scan",
    "sequential_reference",
    "parallel_join",
    "failure_retry",
    "cancel_or_offline",
    "completion_narration",
)

FIELD_PACKS = (
    ("订单号", "数量", "状态"), ("客户编号", "金额", "类别"),
    ("资产编号", "单价", "状态"), ("发票号码", "税额", "类型"),
    ("设备编号", "温度", "模式"), ("工单号", "时长", "级别"),
    ("批次号", "重量", "状态"), ("合同编号", "金额", "阶段"),
    ("员工编号", "工时", "班次"), ("项目编码", "进度", "状态"),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical(row)); count += 1
    return count, digest(path)


def _task(task_id: str, tool: str, args: dict[str, Any], after: list[str] | None = None,
          gate: str = "all_passed") -> dict[str, Any]:
    return {"id": task_id, "tool": tool, "args": args, "after": after or [],
            "gate": gate, "required": True}


def _plan(tasks: list[dict[str, Any]], revision: int = 1) -> dict[str, Any]:
    return {"schema": "mei-node-plan-v1", "revision": revision, "status": "draft", "tasks": tasks}


def _edit_call(task: dict[str, Any], name: str = "plan_add_task") -> dict[str, Any]:
    return {"name": name, "arguments": {"task_id": task["id"], "tool": task["tool"],
        "arguments": task["args"], "after": task["after"], "gate": task["gate"]}}


def engineering_cases() -> list[dict[str, Any]]:
    """Generate 120 related engineering fixtures, never independent benchmark claims."""
    cases: list[dict[str, Any]] = []
    for behavior in BEHAVIORS:
        for index, (key, number, status) in enumerate(FIELD_PACKS):
            suffix = f"{index + 1:02d}"
            required = _task("required", "check_required", {"column": key})
            unique = _task("unique", "check_unique", {"column": key})
            numeric = _task("number", "check_number", {"column": number, "min": 0, "max": 100})
            enum = _task("enum", "check_enum", {"column": status, "allowed": "正常,暂停,关闭"})
            sheet = {"name": "数据", "rows": [[key, number, status],
                [f"K{suffix}", 50, "正常"], [f"K{suffix}", 101, "未知"], ["", 0, "关闭"]]}
            plan = _plan([required])
            query = f"检查「{key}」是否为空。"
            target: dict[str, Any] | list[Any] = _edit_call(required)
            state: dict[str, str] = {}
            options: dict[str, Any] = {}
            expected = {"action": "execute", "reason": "ready_tasks", "task_ids": ["required"]}
            checks = [required]
            narration_input = None

            if behavior == "independent_checks":
                plan = _plan([required, unique]); query = f"「{key}」不能为空且不能重复，两项分别检查。"
                target = _edit_call(required); expected["task_ids"] = ["required", "unique"]; checks = [required, unique]
            elif behavior == "numeric_boundary":
                plan = _plan([numeric]); query = f"确认「{number}」是0到100之间的数字，包含边界。"
                target = _edit_call(numeric); expected["task_ids"] = ["number"]; checks = [numeric]
            elif behavior == "negation_revision":
                plan = _plan([unique]); query = f"不要检查「{key}」重复，改为检查空值。"
                target = _edit_call(required, "plan_update_task")
                target["arguments"]["task_id"] = "unique"; required["id"] = "unique"
                expected = {"action": "execute", "reason": "ready_tasks", "task_ids": ["unique"]}; checks = [unique]
            elif behavior == "missing_or_ambiguous":
                plan = _plan([]); query = "检查这列有没有问题。"; target = []
                expected = {"action": "clarify", "reason": "missing_column", "task_ids": []}; checks = []
                options = {"manualDisposition": True}
            elif behavior == "permission_confirmation":
                plan = _plan([required]); query = f"检查「{key}」，但执行前先让我确认。"; target = []
                options = {"requiresApproval": True, "approvalGranted": False}
                expected = {"action": "review", "reason": "approval_required", "task_ids": ["required"]}; checks = []
            elif behavior == "later_batch_scan":
                plan = _plan([enum]); query = f"「{status}」只允许正常、暂停、关闭。"; target = []
                expected = {"action": "review", "reason": "capability_insufficient_current_batch", "task_ids": ["enum"]}
                checks = []; options = {"manualDisposition": True, "laterBatchTool": "check_enum"}
            elif behavior == "sequential_reference":
                unique_after = _task("unique", "check_unique", {"column": key}, ["required"])
                plan = _plan([required, unique_after]); query = f"先检查「{key}」空值，通过后再查重复。"
                target = _edit_call(required); expected["task_ids"] = ["required"]; checks = [required, unique_after]
            elif behavior == "parallel_join":
                enum_after = _task("enum", "check_enum", {"column": status, "allowed": "正常,暂停,关闭"},
                                   ["required", "number"])
                plan = _plan([required, numeric, enum_after]); query = f"分别检查「{key}」和「{number}」，两项通过后检查「{status}」。"
                target = _edit_call(required); expected["task_ids"] = ["number", "required"]; checks = [required, numeric, enum_after]
            elif behavior == "failure_retry":
                plan = _plan([required]); query = f"检查「{key}」；如果工具失败，带证据请求复核。"; target = []
                state = {"required": "failed"}; options = {"evidenceRefs": [f"error:{suffix}"]}
                expected = {"action": "escalate", "reason": "required_task_failed", "task_ids": ["required"]}; checks = []
            elif behavior == "cancel_or_offline":
                plan = _plan([required]); query = f"取消这次「{key}」检查。"; target = []
                state = {"required": "cancelled"}; expected = {"action": "stop", "reason": "required_task_cancelled", "task_ids": ["required"]}; checks = []
            elif behavior == "completion_narration":
                plan = _plan([required]); query = f"「{key}」检查结束后说明结果。"; target = []
                state = {"required": "passed"}; options = {"evidenceRefs": [f"result:{suffix}"]}
                expected = {"action": "complete", "reason": "all_required_passed", "task_ids": ["required"]}; checks = []
                narration_input = {"verified": True, "results": [{"status": "ok", "issue_count": 0}]}

            cases.append({
                "schema": "mei-node-task-case-v1", "case_id": f"ENG-{behavior}-{suffix}",
                "association_group": f"engineering:{behavior}", "split": "engineering",
                "independent": False, "behavior": behavior, "language": "zh_hans",
                "source": "programmatic_engineering_fixture", "query": query, "sheet": sheet,
                "visible_tools": [task["tool"] for task in plan["tasks"]], "plan": plan,
                "states": state, "state_options": options, "gold_lm_target": target,
                "gold_disposition": expected, "check_tasks": checks,
                "narration_input": narration_input,
                "evidence": [{"source": "user_query", "locator": [0, len(query)],
                              "verified": True}],
                "training_eligible": False,
            })
    return cases


def validate_engineering(cases: list[dict[str, Any]]) -> dict[str, Any]:
    script = r"""
import fs from 'node:fs';
import {runCheck} from './src/demos/data-check/checks.mjs';
import {validatePlan, newDraft, applyPlanCall, evaluateState, disposition, narrate} from './src/demos/data-check/node-runtime.mjs';
const cases=JSON.parse(fs.readFileSync(0,'utf8')); const rows=[];
for (const c of cases) {
  validatePlan(c.plan);
  let derived=null;
  if (c.gold_lm_target && !Array.isArray(c.gold_lm_target)) {
    const base=c.gold_lm_target.name==='plan_add_task' ? newDraft() : c.plan;
    derived=applyPlanCall(base,c.gold_lm_target);
  }
  let actual;
  if (c.state_options.manualDisposition) actual=disposition({...c.gold_disposition,evidence_refs:[]});
  else actual=evaluateState(c.plan,c.states,c.state_options);
  for (const key of ['action','reason']) if (actual[key]!==c.gold_disposition[key]) throw Error(`${c.case_id}:${key}`);
  if (JSON.stringify([...actual.task_ids].sort())!==JSON.stringify([...c.gold_disposition.task_ids].sort())) throw Error(`${c.case_id}:task_ids`);
  const checks=c.check_tasks.map(t=>runCheck(c.sheet,t.tool,t.args));
  const explanation=c.narration_input ? narrate(c.narration_input) : null;
  rows.push({case_id:c.case_id,plan_edit_valid:derived!==null,disposition:actual,checks,explanation});
}
process.stdout.write(JSON.stringify({cases:rows.length,rows}));
"""
    process = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT,
                             input=json.dumps(cases, ensure_ascii=False), text=True,
                             capture_output=True, check=True)
    return json.loads(process.stdout)


def audit_legacy(legacy_manifest: Path) -> dict[str, Any]:
    release = json.loads(legacy_manifest.read_text())
    root = legacy_manifest.parent
    decisions = {
        "retrieval": "conditional_reuse_semantic_rows_recompile_views",
        "full_call": "conditional_reuse_after_provenance_budget_and_gold_replay",
        "agent": "conditional_reuse_source_trajectories_recompile_current_step_queries",
        "trajectory": "conditional_reuse_source_trajectories_recompile_current_step_queries",
        "mw_disposition": "replace_labels_with_v1.2_mapping_keep_historical_regression",
        "confidence": "replace_labels_from_new_model_outcomes",
        "narration": "conditional_reuse_verified_result_rows_after_fact_review",
    }
    result: dict[str, Any] = {"schema": "mei-v12-legacy-binding-audit-v1",
        "manifest": str(legacy_manifest.relative_to(ROOT)), "manifest_sha256": digest(legacy_manifest),
        "bindings": {}}
    group_splits: dict[str, set[str]] = defaultdict(set)
    for binding, info in release["families"].items():
        path = root / info["semantic_path"]
        counts = Counter(); reason = Counter(); sources = set()
        with path.open() as handle:
            for line in handle:
                row = json.loads(line); split = str(row.get("split") or "unknown")
                counts["rows"] += 1; counts[f"split:{split}"] += 1
                group = str(row.get("cf_group") or row.get("case_id") or "")
                if group: group_splits[group].add(split)
                if row.get("budget", {}).get("fits") is False: counts["budget_not_fit"] += 1
                if row.get("kind") == "execute":
                    counts["execute"] += 1
                    if not row.get("answers"): counts["execute_empty_target"] += 1
                    if not row.get("evidence"): counts["execute_empty_evidence"] += 1
                if row.get("kind") == "refuse": counts["refuse"] += 1
                if binding == "confidence" and row.get("label") is None: counts["pending_confidence_label"] += 1
                if row.get("reason_code"): reason[str(row["reason_code"])] += 1
                sources.add(str(row.get("source_family") or row.get("family") or binding))
        result["bindings"][binding] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path),
            "counts": dict(counts), "reason_counts": dict(reason), "source_families": sorted(sources),
            "decision": decisions[binding], "training_ready": False}
    leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    result["association_groups_crossing_splits"] = len(leaked)
    result["association_group_examples"] = leaked[:20]
    result["conclusion"] = "no binding is adopted without current tokenizer/runtime recompilation"
    return result


def _token_files(release: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["source"]: row for row in release["token_files"]}


def _quota_selection(db: sqlite3.Connection, target: int, release: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    totals = {domain: tokens for domain, tokens in db.execute(
        "SELECT r.domain,sum(o.token_length) FROM records r JOIN selected_offsets o ON r.id=o.record_id WHERE r.split='train' GROUP BY r.domain")}
    total = sum(totals.values()); files = _token_files(release)
    targets = {domain: target * tokens // total for domain, tokens in totals.items()}
    if targets: targets[max(targets, key=targets.get)] += target - sum(targets.values())
    rows: list[dict[str, Any]] = []; actual: dict[str, int] = {}
    for domain in sorted(targets):
        used = 0
        query = """SELECT r.id,r.source,r.domain,r.language,r.group_key,r.text_sha,
                   o.phase,o.output_offset,o.token_length,r.priority
                   FROM records r JOIN selected_offsets o ON r.id=o.record_id
                   WHERE r.split='train' AND r.domain=? ORDER BY r.priority,r.id"""
        for row in db.execute(query, (domain,)):
            if used >= targets[domain]: break
            item = dict(row); token_file = files[item["source"]]
            item.update(token_file=token_file["path"], token_file_sha256=token_file["sha256"])
            rows.append(item); used += int(item["token_length"])
        actual[domain] = used
    rows.sort(key=lambda row: hashlib.sha256(f"20260914:{row['id']}".encode()).hexdigest())
    return rows, actual


def qat_reserves(cpt_release: Path, out: Path) -> dict[str, Any]:
    release = json.loads(cpt_release.read_text()); db_path = cpt_release.parent / "record-index.sqlite"
    db = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True); db.row_factory = sqlite3.Row
    manifests = {}
    for name, target in (("reserve-100m", 100_000_000), ("first-20m", 20_000_000), ("stats-1m", 1_000_000)):
        rows, actual = _quota_selection(db, target, release)
        path = out / "qat" / f"{name}.jsonl"; count, file_sha = write_jsonl(path, rows)
        manifests[name] = {"path": str(path.relative_to(out)), "sha256": file_sha,
            "target_tokens": target, "actual_tokens": sum(actual.values()), "domain_tokens": actual,
            "records": count, "split": "train"}
    dev_rows = []
    used = 0
    query = """SELECT r.id,r.source,r.domain,r.language,r.group_key,r.text_sha,r.file_index,
               r.token_offset,r.token_length,r.priority,i.bin_path,i.bin_sha
               FROM records r JOIN input_files i ON r.file_index=i.file_index
               WHERE r.split='dev' ORDER BY r.priority,r.id"""
    for row in db.execute(query):
        if used >= 500_000: break
        dev_rows.append(dict(row)); used += int(row["token_length"])
    db.close()
    path = out / "qat" / "language-dev-500k.jsonl"; count, file_sha = write_jsonl(path, dev_rows)
    manifests["language-dev-500k"] = {"path": str(path.relative_to(out)), "sha256": file_sha,
        "target_tokens": 500_000, "actual_tokens": used, "records": count, "split": "dev"}
    result = {"schema": "mei-v12-qat-reference-reserve-v1", "status": "references_prepared_not_training_authorized",
        "cpt_release": str(cpt_release.relative_to(ROOT)), "cpt_release_sha256": digest(cpt_release),
        "record_index_sha256": digest(db_path), "manifests": manifests,
        "float_controls": "same task SFT release, seed, serializer and scorer required before QAT",
        "model_training_started": False}
    write_json(out / "qat" / "manifest.json", result)
    return result


def verify_qat_references(cpt_release: Path, qat: dict[str, Any], out: Path) -> dict[str, Any]:
    """Verify every emitted reference against its immutable token file and byte bounds."""
    release_root = cpt_release.parent
    checked_files: dict[str, dict[str, Any]] = {}
    totals = Counter()
    failures: list[dict[str, Any]] = []
    for name, manifest in qat["manifests"].items():
        path = out / manifest["path"]
        actual_tokens = 0
        with path.open() as handle:
            for index, line in enumerate(handle):
                row = json.loads(line); totals["references"] += 1
                actual_tokens += int(row["token_length"])
                if row.get("split") and row["split"] != manifest["split"]:
                    failures.append({"manifest": name, "row": index, "reason": "split_mismatch"})
                if "token_file" in row:
                    token_path = release_root / row["token_file"]
                    expected_sha = row["token_file_sha256"]
                    offset = int(row["output_offset"])
                else:
                    token_path = ROOT / row["bin_path"]
                    expected_sha = row["bin_sha"]
                    offset = int(row["token_offset"])
                key = str(token_path)
                if key not in checked_files:
                    checked_files[key] = {"sha256": digest(token_path), "bytes": token_path.stat().st_size}
                file_info = checked_files[key]
                if file_info["sha256"] != expected_sha:
                    failures.append({"manifest": name, "row": index, "reason": "file_hash_mismatch"})
                if (offset + int(row["token_length"])) * 2 > file_info["bytes"]:
                    failures.append({"manifest": name, "row": index, "reason": "out_of_bounds"})
        if actual_tokens != manifest["actual_tokens"]:
            failures.append({"manifest": name, "reason": "token_total_mismatch",
                             "found": actual_tokens, "expected": manifest["actual_tokens"]})
        totals[f"{name}:tokens"] = actual_tokens
    report = {"schema": "mei-v12-qat-reference-verification-v1",
        "status": "passed" if not failures else "failed", "counts": dict(totals),
        "unique_token_files": len(checked_files), "failures": failures[:100],
        "all_references_checked": True}
    write_json(out / "qat" / "verification.json", report)
    if failures:
        raise ValueError(f"QAT reference verification failed: {failures[:3]}")
    return report


def source_capacity_audit(config: dict[str, Any], cpt_release: Path, out: Path) -> dict[str, Any]:
    """Count local task candidates without promoting them to verified SFT or independent Eval."""
    db_path = cpt_release.parent / "record-index.sqlite"
    db = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    cpt = {row[0]: {"records": row[1], "tokens": row[2], "groups": row[3]}
           for row in db.execute("""SELECT r.source,count(*),sum(o.token_length),count(distinct r.group_key)
             FROM records r JOIN selected_offsets o ON r.id=o.record_id
             GROUP BY r.source""")}
    db.close()
    rows = []
    for source in config.get("local_task_sources", []):
        manifest_path = ROOT / source["manifest"]
        data_path = ROOT / source["path"]
        if digest(manifest_path) != source["manifest_sha256"] or digest(data_path) != source["sha256"]:
            raise ValueError(f"local task source binding changed: {source['id']}")
        manifest = json.loads(manifest_path.read_text())
        sample = Counter(); groups = set()
        with data_path.open() as handle:
            for index, line in enumerate(handle):
                if index >= 200: break
                row = json.loads(line); sample["records"] += 1
                group = row.get("group_id") or row.get("metadata", {}).get("source_uuid")
                if group is not None: groups.add(str(group))
                metadata = row.get("metadata", {})
                sample["calls"] += int(metadata.get("checked_single_calls") or metadata.get("calls") or 0)
                sample["structurally_checked"] += int(
                    bool(metadata.get("checked_single_calls")) or not metadata.get("issues", ["unknown"]))
                sample["semantic_review_pending"] += int(metadata.get("semantic_review") == "pending")
                sample["executable_gold_false"] += int(metadata.get("executable_gold") is False)
        if source["id"] == "toolace":
            inventory = {"candidate_records": manifest["records"], "candidate_calls": manifest["checked_calls"],
                         "candidate_groups": manifest["unique_schema_groups"]}
        elif source["id"] == "nemotron":
            inventory = {"candidate_records": manifest["counts"]["candidate_records"],
                         "candidate_calls": manifest["counts"]["candidate_calls"],
                         "candidate_groups": None}
        else:
            entry = next(item for item in manifest["sources"] if item["source_id"] == source["id"])
            inventory = {"candidate_records": entry["counts"]["written"], "candidate_calls": 0,
                         "candidate_groups": entry["counts"]["written"]}
        overlap_sources = source.get("cpt_source_ids", [source["id"]])
        overlap = {key: cpt.get(key, {"records": 0, "tokens": 0, "groups": 0}) for key in overlap_sources}
        rows.append({"source": source["id"], "kind": source["kind"], "inventory": inventory,
            "pilot": {**dict(sample), "groups": len(groups), "limit": 200},
            "current_cpt_selection": overlap,
            "sft_use": source["sft_use"], "independent_eval_use": "not_available_from_current_selection",
            "static_training_ready": 0, "reason": source["reason"]})
    report = {"schema": "mei-v12-local-task-source-capacity-v1",
        "status": "capacity_known_gold_not_admitted", "sources": rows,
        "independent_targets": {"train": 2400, "dev": 360, "calibration": 180, "locked_test": 360},
        "admitted": {"train": 0, "dev": 0, "calibration": 0, "locked_test": 0},
        "conclusion": "local volume is ample for SFT adaptation, but no public row is promoted before semantic review and current-catalog or source-contract validation",
        "eval_isolation": "ToolACE, Nemotron, CrossWOZ and RiSAWOZ groups currently selected for CPT cannot be called independent locked Eval unless reserved groups are removed from a successor CPT release",
        "next_batches": ["review and compile at most 200 ToolACE rows", "review and compile at most 200 Nemotron rows",
                         "reserve new locked Eval from source families absent from CPT or regenerate after CPT family exclusion"]}
    write_json(out / "sources" / "capacity-audit.json", report)
    return report


def _toolace_views(row: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    first, remainder = row["text"].split("\n", 1)
    tools = json.loads(first[len("Tools: "):])
    context: list[dict[str, Any]] = []
    views = []
    parts = re.split(r"(?m)^(user|assistant|tool): ", remainder)
    for role, content in zip(parts[1::2], parts[2::2]):
        content = content.rstrip("\n")
        if role == "assistant" and content.lstrip().startswith("["):
            target = json.loads(content)
            if len(target) == 1:
                views.append({"context": list(context), "target": target, "visible_tools": tools})
                if len(views) >= limit: break
        context.append({"role": role, "content": content})
    return views


def _nemotron_views(row: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    lines = row["text"].splitlines()
    tools = json.loads(lines[0][len("Tools: "):])
    context: list[dict[str, Any]] = []
    views = []
    for line in lines[1:]:
        turn = json.loads(line)
        calls = turn.get("tool_calls") or []
        if turn.get("role") == "assistant" and len(calls) == 1:
            function = calls[0]["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str): arguments = json.loads(arguments)
            views.append({"context": list(context),
                "target": [{"name": function["name"], "arguments": arguments}],
                "visible_tools": tools})
            if len(views) >= limit: break
        context.append(turn)
    return views


def task_candidate_pilots(config: dict[str, Any], out: Path) -> dict[str, Any]:
    """Compile bounded source pilots while keeping semantic admission explicitly pending."""
    reports = {}
    for source in config.get("local_task_sources", []):
        if source["id"] not in {"toolace", "nemotron"}: continue
        parser = _toolace_views if source["id"] == "toolace" else _nemotron_views
        candidates = []; source_rows = 0; source_with_views = 0; calls = 0; rejected = Counter()
        with (ROOT / source["path"]).open() as handle:
            for line in handle:
                if source_rows >= 200: break
                source_rows += 1; row = json.loads(line)
                try:
                    views = parser(row, 8)
                except (ValueError, KeyError, TypeError) as error:
                    rejected[type(error).__name__] += 1; continue
                if not views:
                    rejected["no_single_call_view"] += 1; continue
                source_with_views += 1
                for view_index, view in enumerate(views):
                    calls += 1
                    identifier = hashlib.sha256(
                        f"{source['id']}:{row.get('group_id')}:{source_rows - 1}:{view_index}".encode()).hexdigest()[:20]
                    candidates.append({"schema": "mei-task-sft-candidate-v1", "case_id": identifier,
                        "association_group": f"{source['id']}:{row.get('group_id')}", "source": source["id"],
                        "origin": row.get("origin"), "language": "en", "split": "candidate_train_cpt_shared",
                        "cpt_shared": True, "messages": view["context"], "visible_tools": view["visible_tools"],
                        "gold_lm_target": view["target"], "target_shape": "single_call",
                        "source_validation": "wire_schema_checked_upstream",
                        "semantic_review": "pending", "current_catalog_validation": "not_applicable_external_catalog",
                        "training_eligible": False, "locked_eval_eligible": False})
        path = out / "sources" / f"{source['id']}-pilot-candidates.jsonl"
        count, file_sha = write_jsonl(path, candidates)
        reports[source["id"]] = {"source_rows_target": 200, "source_rows_actual": source_rows,
            "source_rows_with_views": source_with_views, "candidate_views": count,
            "single_calls": calls, "rejected": dict(rejected), "path": str(path.relative_to(out)),
            "sha256": file_sha, "admitted_training_cases": 0,
            "status": "compiled_pending_semantic_review"}
    report = {"schema": "mei-v12-task-adapter-pilots-v1", "sources": reports,
        "independent_cases": 0, "limitations": [
            "source validation is structural and does not certify semantic intent",
            "all pilot families are shared with the current CPT selection",
            "pilot views are not encoded SFT and are not training authorized"]}
    write_json(out / "sources" / "pilot-report.json", report)
    return report


def _read_tokens(path: Path, offset: int, length: int) -> list[int]:
    values = array("H")
    with path.open("rb") as handle:
        handle.seek(offset * 2); values.frombytes(handle.read(length * 2))
    if sys.byteorder != "little": values.byteswap()
    return values.tolist()


CAPABILITIES = {
    "zh_colloquial_negation": {"where": "r.domain='oral' AND r.language='zh_hans'", "markers": ("不", "别", "先", "再", "如果")},
    "condition_and_order": {"where": "r.language='zh_hans'", "markers": ("如果", "然后", "通过后", "先", "再")},
    "numeric_and_units": {"where": "r.domain IN ('code_structure','task_tools')", "markers": ("minimum", "maximum", "number", "数字", "范围")},
    "field_and_schema": {"where": "r.domain IN ('code_structure','task_tools')", "markers": ("schema", "properties", "required", "字段", "参数")},
    "result_and_state": {"where": "r.domain='task_tools'", "markers": ("result", "status", "state", "结果", "状态")},
    "bilingual_tool_context": {"where": "r.domain='task_tools'", "markers": ("tool", "function", "调用", "工具")},
}


def cpt_alignment_audit(cpt_release: Path, tokenizer_model: Path, out: Path) -> dict[str, Any]:
    from tokenizer_candidate import LosslessProcessor
    processor = LosslessProcessor(tokenizer_model)
    db_path = cpt_release.parent / "record-index.sqlite"
    db = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True); db.row_factory = sqlite3.Row
    samples = []; cells = {}
    for capability, spec in CAPABILITIES.items():
        selected = []; scanned = 0; support = Counter()
        query = f"""SELECT r.id,r.source,r.domain,r.language,r.group_key,r.text_sha,
                    r.file_index,r.token_offset,r.token_length,r.priority,i.bin_path,i.bin_sha
                    FROM records r JOIN input_files i ON r.file_index=i.file_index
                    WHERE r.split='dev' AND {spec['where']} ORDER BY r.priority,r.id LIMIT 2000"""
        for row in db.execute(query):
            if len(selected) >= 50: break
            scanned += 1; item = dict(row); path = ROOT / item["bin_path"]
            ids = _read_tokens(path, int(item["token_offset"]), min(int(item["token_length"]), 2048))
            if ids and ids[0] == 2: ids = ids[1:]
            if ids and ids[-1] == 1: ids = ids[:-1]
            text = processor.decode(ids); markers = [marker for marker in spec["markers"] if marker.casefold() in text.casefold()]
            if capability in {"zh_colloquial_negation", "condition_and_order"} and not markers: continue
            evidence = "lexical_marker" if markers else "source_family_proxy"
            support[evidence] += 1
            selected.append({"capability": capability, "record_id": item["id"], "source": item["source"],
                "domain": item["domain"], "language": item["language"], "group_key": item["group_key"],
                "text_sha": item["text_sha"], "markers": markers, "evidence_level": evidence,
                "excerpt": text[:800], "manual_semantic_review": "pending"})
        samples.extend(selected)
        cells[capability] = {"target": 50, "selected": len(selected), "records_scanned": scanned,
            "evidence_levels": dict(support), "model_mastery": "pending_model_probe"}
    db.close()
    path = out / "alignment" / "cpt-evidence.jsonl"; count, file_sha = write_jsonl(path, samples)
    result = {"schema": "mei-cpt-sft-eval-alignment-audit-v1", "status": "candidates_prepared_for_semantic_review",
        "target_max": 300, "actual": count, "cells": cells, "evidence_path": str(path.relative_to(out)),
        "evidence_sha256": file_sha, "limitations": [
            "lexical markers and source families locate evidence but do not prove semantic coverage",
            "CPT presence does not prove model mastery",
            "English tool material plus Chinese requests is a cross-source capability and needs model probes"],
        "cpt_revision_decision": "pending_semantic_review",
        "cpt_revision_required": None}
    write_json(out / "alignment" / "report.json", result)
    return result


def tokenizer_task_audit(tokenizer_model: Path, legacy_manifest: Path, cases: list[dict[str, Any]],
                         cpt_release: Path, pilots: dict[str, Any], out: Path) -> dict[str, Any]:
    from tokenizer_candidate import LosslessProcessor
    processor = LosslessProcessor(tokenizer_model); cells = defaultdict(lambda: Counter(records=0, characters=0, tokens=0, roundtrip_failures=0))
    cell_lengths: dict[str, list[int]] = defaultdict(list)
    hard = []
    texts = [("engineering", case["query"] + "\n" + json.dumps(case["gold_lm_target"], ensure_ascii=False, separators=(",", ":"))) for case in cases]
    release = json.loads(legacy_manifest.read_text()); root = legacy_manifest.parent
    for binding, info in release["families"].items():
        with (root / info["semantic_path"]).open() as handle:
            for line in handle:
                row = json.loads(line)
                pieces = [str(row.get("query") or "")]
                if row.get("answers") is not None: pieces.append(json.dumps(row["answers"], ensure_ascii=False, separators=(",", ":")))
                if row.get("narration_target"): pieces.append(str(row["narration_target"]))
                texts.append((binding, "\n".join(piece for piece in pieces if piece)))
    for source, info in pilots["sources"].items():
        with (out / info["path"]).open() as handle:
            for line in handle:
                row = json.loads(line)
                texts.append((f"pilot_{source}", json.dumps({"messages": row["messages"],
                    "tools": row["visible_tools"], "target": row["gold_lm_target"]},
                    ensure_ascii=False, separators=(",", ":"))))
    lengths = []
    for domain, text in texts:
        ids = processor.encode(text); decoded = processor.decode(ids); cell = cells[domain]
        cell.update(records=1, characters=len(text), tokens=len(ids), roundtrip_failures=int(decoded != text))
        cell_lengths[domain].append(len(ids))
        lengths.append(len(ids))
        if decoded != text and len(hard) < 20: hard.append({"domain": domain, "before": text, "after": decoded})
    boundary = ["  前导 空格\t字段\r\n", "ＡＢＣ ① é é", "000123 -24.50 C:\\tmp\\a", "literal ▁ marker", "😀𠀀"]
    boundary_rows = []
    for text in boundary:
        ids = processor.encode(text); decoded = processor.decode(ids)
        boundary_rows.append({"text": text, "tokens": len(ids), "exact": decoded == text})
    lengths.sort(); p95 = lengths[min(len(lengths) - 1, int(len(lengths) * .95))] if lengths else 0
    cpt = json.loads(cpt_release.read_text()); runtime = cpt["tokenizer"]["runtime_audit"]
    runtime_path = ROOT / runtime["path"]
    runtime_ok = (digest(runtime_path) == runtime["sha256"] and runtime["real_browser_passed"]
                  and runtime["rust_passed"] and cpt["tokenizer"]["model_sha256"] == digest(tokenizer_model))
    cell_report = {}
    for key, value in sorted(cells.items()):
        domain_lengths = sorted(cell_lengths[key]); item = dict(value)
        item["p95_tokens"] = domain_lengths[min(len(domain_lengths) - 1, int(len(domain_lengths) * .95))]
        item["max_tokens"] = domain_lengths[-1]
        item["over_1920"] = sum(length > 1920 for length in domain_lengths)
        cell_report[key] = item
    report = {"schema": "mei-v12-task-tokenizer-audit-v1", "status": "passed" if not hard and all(x["exact"] for x in boundary_rows) and runtime_ok else "failed",
        "model": str(tokenizer_model.relative_to(ROOT)), "model_sha256": digest(tokenizer_model),
        "texts": len(texts), "task_token_p95": p95, "cells": cell_report,
        "boundary": boundary_rows, "roundtrip_failures": hard,
        "decision": "retain_for_current_stage_pending_independent_task_sample" if not hard else "new_candidate_or_runtime_fix_required",
        "decision_basis": "no hard task-text preservation failure; oversized public schemas require retrieval/compaction rather than a larger vocabulary; final adoption remains pending independent task train material" if not hard else "hard preservation failure",
        "runtime_audit": {**runtime, "binding_verified": runtime_ok},
        "limitations": ["real Browser-WASM evidence is the frozen 264-case runtime audit; newly compiled task candidates have Python preservation coverage and require a refreshed browser sample at final SFT encoding",
                        "per-text lengths do not replace the complete 2048 prompt/schema/state/output-reserve budget audit"],
        "locked_test_used": False, "capacity_comparison_performed": False}
    write_json(out / "tokenizer" / "task-audit.json", report)
    return report


def run(config: dict[str, Any], out: Path) -> dict[str, Any]:
    if out.exists(): raise ValueError("use a new output ID")
    if shutil.disk_usage(ROOT).free < int(config.get("reserve_free_bytes", 100 * 1024**3)):
        raise ValueError("disk reserve reached")
    cpt = ROOT / config["cpt_release"]; tokenizer = ROOT / config["tokenizer_model"]
    legacy = ROOT / config["legacy_sft_manifest"]
    expected = config["input_sha256"]
    actual = {"cpt_release": digest(cpt), "tokenizer_model": digest(tokenizer), "legacy_sft_manifest": digest(legacy)}
    if actual != expected: raise ValueError(f"input binding changed: {actual}")
    out.mkdir(parents=True)
    write_json(out / "config.json", config)
    cases = engineering_cases()
    if len(cases) != 120 or Counter(case["behavior"] for case in cases) != Counter({name: 10 for name in BEHAVIORS}):
        raise ValueError("engineering case quota mismatch")
    verification = validate_engineering(cases)
    write_jsonl(out / "engineering" / "cases.jsonl", cases)
    write_json(out / "engineering" / "verification.json", verification)
    legacy_audit = audit_legacy(legacy); write_json(out / "legacy" / "binding-audit.json", legacy_audit)
    alignment = cpt_alignment_audit(cpt, tokenizer, out)
    capacity = source_capacity_audit(config, cpt, out)
    pilots = task_candidate_pilots(config, out)
    tokenizer_audit = tokenizer_task_audit(tokenizer, legacy, cases, cpt, pilots, out)
    qat = qat_reserves(cpt, out)
    qat_verification = verify_qat_references(cpt, qat, out)
    coverage = {"schema": "mei-v12-task-preparation-coverage-v1", "engineering": {"actual": 120, "target": 120},
        "independent": {"train": {"actual": 0, "target": 2400}, "dev": {"actual": 0, "target": 360},
                        "calibration": {"actual": 0, "target": 180}, "locked_test": {"actual": 0, "target": 360}},
        "formal_reserve": {"sft_first": {"actual": 0, "target": 50000}, "sft_extended": {"actual": 0, "target": 100000},
                           "node_closed_loop": {"actual": 0, "target": 10000},
                           "eval_dev": {"actual": 0, "target": 1200}, "eval_locked": {"actual": 0, "target": 1200}},
        "behaviors": {name: {"engineering": 10, "independent": 0} for name in BEHAVIORS},
        "bindings": {name: {"engineering_source_cases": 120 if name in {"planning", "mw_disposition"} else
                             (100 if name == "full_call" else 10 if name == "narration" else 0),
                             "independent_training_cases": 0, "status": "fixture_only_or_pending"}
                     for name in ("planning", "retrieval", "full_call", "agent", "mw_disposition", "confidence", "narration")},
        "confidence": "pending_model", "model_training_started": False}
    write_json(out / "coverage.json", coverage)
    report = {"schema": "mei-v12-task-preparation-report-v1", "status": "engineering_and_audits_complete_static_training_data_incomplete",
        "engineering": {"actual": 120, "target": 120, "verified": verification["cases"]},
        "independent_train": {"actual": 0, "target": 2400},
        "cpt_alignment": {"actual": alignment["actual"], "target_max": 300, "status": alignment["status"]},
        "tokenizer": {"status": tokenizer_audit["status"], "decision": tokenizer_audit["decision"]},
        "qat": {name: {"actual_tokens": row["actual_tokens"], "target_tokens": row["target_tokens"]}
                for name, row in qat["manifests"].items()},
        "qat_verification": qat_verification["status"],
        "source_capacity": capacity["status"],
        "adapter_pilots": {name: row["candidate_views"] for name, row in pilots["sources"].items()},
        "legacy_bindings": len(legacy_audit["bindings"]), "downloads": 0, "teacher_calls": 0,
        "training_started": False, "current_mutated": False,
        "next_gate": "manual semantic review of CPT evidence and source-capacity/adaptor work for independent task cases"}
    write_json(out / "REPORT.json", report)
    sources = [Path(__file__), ROOT / "src/demos/data-check/node-runtime.mjs", cpt, tokenizer, legacy]
    write_json(out / "SOURCE-BINDING.json", [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sources])
    files = {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*")) if path.is_file()}
    write_json(out / "FILES.json", files)
    return report
