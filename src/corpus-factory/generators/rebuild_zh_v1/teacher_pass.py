#!/usr/bin/env python3
"""DeepSeek teacher pass over a skeleton SFT release (query-only paraphrase).

Governance contract (visible-chinese-query-only-v1, provider enabled):
- mutable field: **only** ``query``. Gold (gold_name/gold_args/answers/mw
  routes/confidence labels/evidence/state) is untouched — it was compiled
  locally by the skeleton build and stays that way.
- protected values (``值NNN`` placeholders + every standalone number) must
  survive **verbatim** in any accepted variant; a variant that drops,
  alters, or fabricates one fails closed (original query is kept).
- one API call covers BATCH_ROWS rows and asks for K=2 variants per row;
  usage (prompt/completion tokens) is recorded per call in a cost ledger.
- the pass never mutates the source release: it copies it to a new,
  write-once release id that supersedes the skeleton, and recomputes the
  artifact Merkle root over the rewritten semantic/compiled jsonl files.
- only train-split rows are polished; dev/test/valid stay pristine as
  held-out evidence for the eval lock.

Run:
  python teacher_pass.py --release-root <.../sft-suite> \
    --release-id <skeleton-id> --out-id <new-id> \
    [--env-file ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebuild_zh_v1 import common as C

PROVIDER = "deepseek"
MODEL = "deepseek-chat"
TEACHER_ID = f"{PROVIDER}-{MODEL}-20260905"
K_VARIANTS = 2
BATCH_ROWS = 4
PRICING_CNY_PER_MTOK = {
    "prompt": 2.0,
    "completion": 8.0,
    "note": "assumption: deepseek-chat ¥2/M input, ¥8/M output; ledger records real token usage per call",
}

DEFAULT_SAMPLE = {
    "retrieval": 300,
    "full_call": 300,
    "agent": 200,
    "mw_disposition": 600,
    "narration": 300,
    "confidence": 300,
}
# v3 rebalanced sampling: full_call covers 12 scenarios (6 arg_norm_*), so it
# gets a larger share; MW shrank (500/class -> 200/class).
V3_SAMPLE = {
    "retrieval": 300,
    "full_call": 400,
    "agent": 200,
    "mw_disposition": 500,
    "narration": 300,
    "confidence": 300,
    "trajectory": 300,
}
FAMILIES = tuple(DEFAULT_SAMPLE) + ("trajectory",)

VALUE_RE = re.compile(r"值\d+")
# 注意：Python re 的 \w 匹配中文，不能用 \w 做数字边界——否则中文相邻的
# 数字（"份数是3"）会漏保护。边界用 ASCII 集合；句尾 "10." 的句点允许。
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?!\.\d)(?![A-Za-z0-9_])")
CJK_RE = re.compile(r"[一-鿿]")

SYSTEM_PROMPT = (
    "你是一名中文口语改写器。输入是一个 JSON 对象 items，键是行号，值是该行的查询语句。"
    "请为每个查询给出两个自然、口语化、措辞风格彼此不同的中文改写版本。"
    "硬性要求：\n"
    "1) 原文中的数字、编号、占位值（例如 值3072、47.4、ISO-8601、2026-09-05 这类片段）"
    "必须一字不差地原样保留，不得改写、换算、增减任何数字；\n"
    "2) 不得改变查询的意图、目标工具、参数与业务含义，只改表达方式；\n"
    "3) 两个版本彼此措辞明显不同，且都与原文不同；\n"
    "4) 保持口语化的中文，不要输出解释性文字。\n"
    '输出必须是 JSON 对象，格式：{"items": {"行号": {"v1": "版本一", "v2": "版本二"}, ...}}。'
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        env[key.strip()] = value.strip().strip('"').strip("'")
    missing = [k for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL") if not env.get(k)]
    if missing:
        raise RuntimeError(f"env file {path} missing: {missing}")
    return env


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(C.canonical_json(r) + "\n" for r in rows), encoding="utf-8")


def protected_values(query: str) -> list[str]:
    seen: list[str] = []
    for match in VALUE_RE.finditer(query):
        if match.group() not in seen:
            seen.append(match.group())
    for match in NUMBER_RE.finditer(query):
        if match.group() not in seen:
            seen.append(match.group())
    return seen


def verify_variant(original: str, variant: Any, family: str) -> str | None:
    """Return None if acceptable, else a failure reason (fail closed)."""
    if not isinstance(variant, str) or not variant.strip():
        return "empty"
    if variant == original:
        return "identical-to-original"
    if CJK_RE.search(variant) is None:
        return "no-cjk"
    if not (len(original) * 0.4 <= len(variant) <= len(original) * 2.5):
        return "length-out-of-range"
    for value in protected_values(original):
        if value not in variant:
            return f"protected-lost:{value}"
    for match in VALUE_RE.finditer(variant):  # no fabricated placeholders
        if match.group() not in original:
            return f"fabricated-value:{match.group()}"
    for match in NUMBER_RE.finditer(variant):  # no fabricated numbers
        if match.group() not in original:
            return f"fabricated-number:{match.group()}"
    forbidden = C.mw_reason_codes() if family == "mw_disposition" else ()
    hits = C.scan_leakage(variant, forbidden)
    if hits:
        return f"leakage:{','.join(hits)}"
    return None


def call_deepseek(env: dict[str, str], items: dict[str, str]) -> tuple[dict[str, Any], dict[str, int]]:
    """One chat call for BATCH_ROWS rows; returns (parsed_json, usage)."""
    url = env["DEEPSEEK_BASE_URL"].rstrip("/") + "/chat/completions"
    payload = {
        "model": MODEL,
        "temperature": 1.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {env['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    usage = {k: int(body.get("usage", {}).get(k, 0)) for k in ("prompt_tokens", "completion_tokens")}
    return parsed, usage


def build_manifest(base_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in sorted(base_dir.rglob("*.jsonl")):
        rel = str(path.relative_to(base_dir))
        data = path.read_bytes()
        artifacts[rel] = {
            "bytes": len(data),
            "rows": data.count(b"\n"),
            "sha256": C.sha256_bytes(data),
        }
    return {"artifacts": artifacts, "artifact_merkle_root": C.merkle_root(artifacts)}


def sample_train_rows(release_dir: Path, sample: dict[str, int], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    picked: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        rows = read_jsonl(release_dir / "compiled" / family / "train.jsonl")
        # 留出改写余量：prompt_tokens 已近 cap 的行跳过（教师改写通常会变长，
        # 近上限的行改写后必然触发预算 fail-closed，白花 API 钱）。
        margin = 120
        pool = [
            r for r in rows
            if int((r.get("budget") or {}).get("prompt_tokens") or 0) <= int((r.get("budget") or {}).get("cap") or 0) - margin
        ]
        n = min(sample.get(family, 0), len(pool))
        picked[family] = rng.sample(pool, n)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--release-id", required=True)
    ap.add_argument("--out-id", required=True)
    ap.add_argument(
        "--env-file",
        type=Path,
        default=Path(os.environ.get("CLAUDE_JOB_DIR", str(Path.home() / ".claude/jobs/90c534da"))) / "tmp/deepseek.env",
    )
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help="JSONL scratch of already-accepted batches; rows present here are skipped on rerun",
    )
    ap.add_argument(
        "--sample",
        choices=("v2", "v3"),
        default="v2",
        help="sampling profile: v2 (original) or v3 (rebalanced)",
    )
    args = ap.parse_args()
    sample_profile = DEFAULT_SAMPLE if args.sample == "v2" else V3_SAMPLE

    source_dir = args.release_root / args.release_id
    out_dir = args.release_root / args.out_id
    if not source_dir.is_dir():
        raise SystemExit(f"source release not found: {source_dir}")
    if out_dir.exists():
        raise SystemExit(f"write-once refusal: out release already exists: {out_dir}")

    t0 = time.time()
    log("verifying source release integrity (merkle over jsonl artifacts)")
    source_manifest = build_manifest(source_dir)
    source_release = json.loads((source_dir / "release-manifest.json").read_text(encoding="utf-8"))
    if source_manifest["artifact_merkle_root"] != source_release["artifact_merkle_root"]:
        raise SystemExit("source release merkle mismatch — refusing to build on tampered release")

    # dedup index: normalized query -> case_ids, across all six families
    query_index: dict[str, set[str]] = {}
    for family in FAMILIES:
        for row in read_jsonl(source_dir / "semantic" / f"{family}.jsonl"):
            query_index.setdefault(C.normalize_text(str(row["query"])), set()).add(str(row["case_id"]))

    log("sampling train rows (stratified, seeded)")
    picked = sample_train_rows(source_dir, sample_profile, args.seed)
    total = sum(len(rows) for rows in picked.values())
    log(f"sampled {total} rows: " + ", ".join(f"{f}:{len(picked[f])}" for f in FAMILIES))

    env: dict[str, str] | None = None
    if not args.dry_run:
        env = load_env(args.env_file)

    # batched calls, 4 rows each, keys r0..r3
    ledger: list[dict[str, Any]] = []
    per_row: dict[str, dict[str, Any]] = {}
    stats = {"calls": 0, "ok_calls": 0, "variants_requested": 0, "variants_accepted": 0}
    failures: dict[str, int] = {}

    resume_path: Path | None = None
    if args.resume_file is not None:
        resume_path = args.resume_file if args.resume_file.is_absolute() else Path.cwd() / args.resume_file
        if resume_path.is_file():
            for line in resume_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                per_row[entry["case_id"]] = {
                    key: entry[key] for key in ("before_query", "kept_variant", "after_query", "verification")
                    if key in entry
                }
            log(f"resume file loaded: {len(per_row)} rows already processed")

    batch_items: list[tuple[str, str]] = []  # (case_id, query)
    for family in FAMILIES:
        for row in picked[family]:
            case_id = str(row["case_id"])
            if case_id not in per_row:  # skip rows already processed in a prior run
                batch_items.append((case_id, str(row["query"])))

    def flush_batch(items: list[tuple[str, str]], call_index: int) -> None:
        stats["calls"] += 1
        if args.dry_run:
            ledger.append({
                "call_index": call_index, "provider": PROVIDER, "model": MODEL,
                "case_ids": [cid for cid, _ in items], "rows_covered": len(items),
                "ok": True, "dry_run": True,
            })
            return
        mapping = {f"r{i}": query for i, (_, query) in enumerate(items)}
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        parsed: dict[str, Any] | None = None
        last_error = ""
        for attempt in range(2):
            try:
                parsed, usage = call_deepseek(env or {}, mapping)
                break
            except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log(f"call {call_index} attempt {attempt + 1} failed: {last_error}")
                time.sleep(2.0)
        if parsed is None:
            ledger.append({
                "call_index": call_index, "provider": PROVIDER, "model": MODEL,
                "case_ids": [cid for cid, _ in items], "rows_covered": len(items),
                "ok": False, "error": last_error, **usage,
            })
            for case_id, original in items:
                per_row[case_id].update({
                    "kept_variant": None,
                    "after_query": original,
                    "verification": {},
                })
            return  # rows stay untouched
        ledger.append({
            "call_index": call_index, "provider": PROVIDER, "model": MODEL,
            "case_ids": [cid for cid, _ in items], "rows_covered": len(items),
            "ok": True, **usage,
        })
        stats["ok_calls"] += 1
        wrapped = parsed.get("items") if isinstance(parsed.get("items"), dict) else None
        for i, (case_id, original) in enumerate(items):
            key = f"r{i}"
            variants = (wrapped or {}).get(key) if wrapped is not None else None
            stats["variants_requested"] += K_VARIANTS
            family = case_id.split(":", 1)[0]
            record = per_row[case_id]
            chosen: str | None = None
            verified: dict[str, str] = {}
            candidates: list[str] = []
            if isinstance(variants, dict):
                for vkey in (f"v{j}" for j in range(1, K_VARIANTS + 1)):
                    candidate = variants.get(vkey)
                    if not isinstance(candidate, str):
                        verified[vkey] = "missing"
                        continue
                    reason = verify_variant(original, candidate, family)
                    verified[vkey] = reason or "ok"
                    if reason is None:
                        # dedup: normalized new query must not collide with any other row
                        norm = C.normalize_text(candidate)
                        others = query_index.get(norm, set()) - {case_id}
                        if others:
                            verified[vkey] = f"duplicate-of:{sorted(others)[0][:20]}"
                        else:
                            candidates.append((candidate, vkey))
            if candidates:
                def distinctness(pair: tuple[str, str]) -> float:
                    text, _ = pair
                    return C.jaccard(C.char_trigrams(original), C.char_trigrams(text))
                chosen, chosen_key = min(candidates, key=distinctness)
                stats["variants_accepted"] += 1
                record["verification"] = verified
                record["kept_variant"] = chosen_key
                record["after_query"] = chosen
            else:
                record["verification"] = verified
                record["kept_variant"] = None
                record["after_query"] = original
                reasons = [r for r in verified.values() if r != "ok"]
                for reason in reasons:
                    failures[reason] = failures.get(reason, 0) + 1

    for i in range(0, len(batch_items), BATCH_ROWS):
        batch = batch_items[i : i + BATCH_ROWS]
        per_row.update({cid: {"before_query": q} for cid, q in batch})
        flush_batch(batch, i // BATCH_ROWS + 1)
        if resume_path is not None:
            with resume_path.open("a", encoding="utf-8") as handle:
                for case_id, _ in batch:
                    record = per_row[case_id]
                    handle.write(json.dumps({
                        "case_id": case_id,
                        "before_query": record["before_query"],
                        "kept_variant": record.get("kept_variant"),
                        "after_query": record.get("after_query"),
                        "verification": record.get("verification") or {},
                    }, ensure_ascii=False) + "\n")
        time.sleep(0.15)
        if i and i % 100 == 0:
            log(f"processed {i}/{len(batch_items)} rows, {stats['variants_accepted']} variants accepted")

    if args.dry_run:
        log(f"dry run: would call {stats['calls']} times for {total} rows "
            f"(est ~¥{(total * 600 * (PRICING_CNY_PER_MTOK['prompt'] + 2 * PRICING_CNY_PER_MTOK['completion'])) / 1e6:.2f}); "
            "no API calls made, no release written")
        return 0

    log(f"API calls done: {stats['calls']} calls, {stats['ok_calls']} ok, "
        f"{stats['variants_accepted']}/{total * K_VARIANTS} variants accepted")
    if failures:
        log("verification failure reasons: " + ", ".join(f"{k}×{v}" for k, v in sorted(failures.items())))

    # apply accepted variants (copy-on-write into the new release)
    # Phase A: pure compute — resolve final query/budget per polished row
    # before touching the new release dir (fail closed on budget overflow).
    replacements: dict[str, dict[str, str]] = {family: {} for family in FAMILIES}
    applied = 0
    for family in FAMILIES:
        for row in read_jsonl(source_dir / "semantic" / f"{family}.jsonl"):
            case_id = str(row["case_id"])
            rec = per_row.get(case_id)
            if not rec or not rec.get("kept_variant"):
                continue
            new_query = rec["after_query"]
            if new_query == row["query"]:
                continue
            budget = dict(row.get("budget") or {})
            if "prompt_tokens" in budget:
                delta = C.count_tokens(new_query) - C.count_tokens(str(row["query"]))
                new_total = int(budget["prompt_tokens"]) + delta
                if new_total > int(budget["cap"]):
                    rec["after_query"] = str(row["query"])
                    rec["kept_variant"] = None
                    rec["budget_note"] = "variant over budget cap — original kept"
                    continue
                budget["prompt_tokens"] = new_total
                budget["fits"] = new_total <= int(budget["cap"])
            replacements[family][case_id] = new_query
            rec["after_sha256"] = C.sha256_text(new_query)
            rec["new_budget"] = budget
            applied += 1

    # Phase B: copy-on-write the release, apply replacements, add envelopes.
    log(f"applying {applied} polished queries into {args.out_id}")
    shutil.copytree(source_dir, out_dir)
    for family in FAMILIES:
        if not replacements[family]:
            continue
        semantic_path = out_dir / "semantic" / f"{family}.jsonl"
        rows = read_jsonl(semantic_path)
        for row in rows:
            case_id = str(row["case_id"])
            if case_id in replacements[family]:
                rec = per_row[case_id]
                row["query"] = replacements[family][case_id]
                if "new_budget" in rec:
                    row["budget"] = rec["new_budget"]
                row["teacher"] = {
                    "teacher_id": TEACHER_ID,
                    "provider": PROVIDER,
                    "model": MODEL,
                    "mutable_fields": ["query"],
                    "protected_values": protected_values(str(rec["before_query"])),
                    "kept_variant": rec["kept_variant"],
                }
        write_jsonl(semantic_path, rows)
        # mirror the semantic rows' final state (query + budget + teacher)
        semantic_by_id = {str(r["case_id"]): r for r in rows}
        train_path = out_dir / "compiled" / family / "train.jsonl"
        train_rows = read_jsonl(train_path)
        for row in train_rows:
            source = semantic_by_id.get(str(row["case_id"]))
            if source is None or str(row["case_id"]) not in replacements[family]:
                continue
            row["query"] = source["query"]
            row["budget"] = source.get("budget", row.get("budget"))
            row["teacher"] = source["teacher"]
        write_jsonl(train_path, train_rows)

    log(f"applied {applied} polished queries")

    # recompute artifact manifest + merkle for the new release
    new_manifest = build_manifest(out_dir)
    (out_dir / "manifests" / "artifact-manifest.json").write_text(
        json.dumps(new_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    total_prompt = sum(c.get("prompt_tokens", 0) for c in ledger)
    total_completion = sum(c.get("completion_tokens", 0) for c in ledger)
    total_cny = (total_prompt * PRICING_CNY_PER_MTOK["prompt"] + total_completion * PRICING_CNY_PER_MTOK["completion"]) / 1e6

    report = {
        "schema": "mei-51m-sft-teacher-pass-report-v1",
        "teacher": {"teacher_id": TEACHER_ID, "provider": PROVIDER, "model": MODEL,
                    "base_url": (env or {}).get("DEEPSEEK_BASE_URL", "dry-run"),
                    "variants_per_row": K_VARIANTS, "rows_per_call": BATCH_ROWS},
        "source_release": {"release_id": args.release_id,
                           "artifact_merkle_root": source_manifest["artifact_merkle_root"],
                           "integrity_verified": True},
        "supersedes": {"release_id": args.release_id,
                       "reason": "query-only teacher paraphrase (visible-chinese-query-only-v1); gold untouched; train split only"},
        "sampling": {"seed": args.seed, "split": "train", "per_family": {f: len(picked[f]) for f in FAMILIES},
                     "total_sampled": total},
        "results": {
            "rows_polished": applied,
            "rows_kept_original": total - applied,
            "variants_accepted": stats["variants_accepted"],
            "verification_failure_reasons": failures,
        },
        "cost_ledger": {
            "pricing_cny_per_mtok": PRICING_CNY_PER_MTOK,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cny_est": round(total_cny, 4),
            "calls": ledger,
        },
        "per_row": {
            case_id: {
                "family": case_id.split(":", 1)[0],
                "before_sha256": C.sha256_text(str(rec["before_query"])),
                "after_sha256": rec.get("after_sha256"),
                "kept_variant": rec.get("kept_variant"),
                "verification": rec.get("verification") or {},
                "protected_values": protected_values(str(rec["before_query"])),
            }
            for case_id, rec in per_row.items()
        },
    }
    governance_dir = out_dir / "governance"
    (governance_dir / "teacher-pass-v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    release_manifest = json.loads((out_dir / "release-manifest.json").read_text(encoding="utf-8"))
    release_manifest.update({
        "release_id": args.out_id,
        "supersedes": {"release_id": args.release_id,
                       "artifact_merkle_root": source_manifest["artifact_merkle_root"],
                       "reason": "query-only teacher paraphrase (visible-chinese-query-only-v1); gold untouched; train split only"},
        "teacher": {"teacher_id": TEACHER_ID, "provider": PROVIDER, "model": MODEL,
                    "calls": stats["calls"], "rows_polished": applied,
                    "pricing_cny_per_mtok": PRICING_CNY_PER_MTOK,
                    "note": "only train-split rows sampled; dev/test/valid pristine"},
        "provider_calls": stats["calls"],
        "paid_cny": round(total_cny, 4),
        "artifact_merkle_root": new_manifest["artifact_merkle_root"],
        "human_review": {
            **source_release.get("human_review", {}),
            "note": (source_release.get("human_review", {}).get("note", "")
                     + " Teacher pass (deepseek-chat) polished query phrasing on a stratified train sample; "
                     "protected values verified verbatim per row; ledger in governance/teacher-pass-v1.json."),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_seconds": round(time.time() - t0, 2),
    })
    (out_dir / "release-manifest.json").write_text(
        json.dumps(release_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    log(f"done in {time.time() - t0:.1f}s; polished {applied}/{total} rows; est cost ¥{total_cny:.2f}")
    print(json.dumps({
        "release_id": args.out_id,
        "merkle_root": new_manifest["artifact_merkle_root"],
        "rows_polished": applied,
        "rows_sampled": total,
        "calls": stats["calls"],
        "cny_est": round(total_cny, 4),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
