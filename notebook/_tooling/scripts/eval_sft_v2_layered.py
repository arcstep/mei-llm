"""Layered Full-call / MW scoring. Does not change eval_mei_tool_schema_v1.score_item."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from eval_needle_toolcall_v0 import calls_equal, extract_arguments, norm_calls, values_equal
from repo_paths import SFT_V2_BASELINE_CONTRACT_V2

REASON_CODES_16 = json.loads(SFT_V2_BASELINE_CONTRACT_V2.read_text(encoding="utf-8"))["tracks"]["mw"][
    "reason_codes"
]

FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
TRAIL_EXPLAIN_RE = re.compile(r"(因为|所以|解释|我来|如下)")


def _strip_fence(text: str) -> tuple[str, bool]:
    raw = (text or "").strip()
    m = FENCE_RE.search(raw)
    if m:
        return m.group(1).strip(), True
    return raw, False


def _load_json_blob(text: str) -> Any | None:
    s = (text or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for start, end in (("[", "]"), ("{", "}")):
        i = s.find(start)
        j = s.rfind(end)
        if i >= 0 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None


def extract_native_tool_calls(tool_calls: Any) -> list[dict]:
    if not tool_calls:
        return []
    out = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = str((fn or {}).get("name") or item.get("name") or "")
        args = (fn or {}).get("arguments") or item.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            out.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return out


def extract_fullcall_calls(text: str, *, tool_calls: Any = None) -> tuple[list[dict], dict[str, Any]]:
    """Loose content extraction. Format flags are separate."""
    flags = {
        "empty_text": not (text or "").strip() and not tool_calls,
        "fenced": False,
        "used_parameters": False,
        "wrapper": False,
        "trailing_text": False,
        "native_tool_calls": bool(tool_calls),
        "parse_ok": False,
        "utf8_ok": True,
    }
    if tool_calls:
        calls = extract_native_tool_calls(tool_calls)
        flags["parse_ok"] = True
        return norm_calls(calls, accept_parameters=True), flags
    stripped, fenced = _strip_fence(text or "")
    flags["fenced"] = fenced
    blob = _load_json_blob(stripped)
    if blob is None:
        return [], flags
    flags["parse_ok"] = True
    if isinstance(blob, dict) and not any(k in blob for k in ("name", "function", "arguments", "parameters", "tool")):
        # wrapper like {"calls":[...]} or {"tool_call":{...}}
        for key in ("calls", "function_calls", "tools", "tool_call", "result"):
            if key in blob:
                blob = blob[key]
                flags["wrapper"] = True
                break
    if isinstance(blob, dict):
        flags["wrapper"] = flags["wrapper"] or any(k in blob for k in ("tool", "function"))
        name = blob.get("name") or (blob.get("tool") if isinstance(blob.get("tool"), str) else None)
        if isinstance(blob.get("function"), dict):
            name = name or blob["function"].get("name")
            args = blob["function"].get("arguments") or blob["function"].get("parameters")
            flags["used_parameters"] = "parameters" in blob["function"] and "arguments" not in blob["function"]
            blob = {"name": name, "arguments": args or {}}
        elif "parameters" in blob and "arguments" not in blob:
            flags["used_parameters"] = True
            blob = {"name": name or blob.get("name"), "arguments": blob.get("parameters") or {}}
        blob = [blob]
    if not isinstance(blob, list):
        return [], flags
    for item in blob:
        if isinstance(item, dict) and "parameters" in item and "arguments" not in item:
            flags["used_parameters"] = True
    remainder = stripped
    flags["trailing_text"] = bool(TRAIL_EXPLAIN_RE.search(remainder)) and remainder.strip() not in {"[]", ""}
    if fenced:
        flags["trailing_text"] = True
    return norm_calls(blob, accept_parameters=True), flags


def format_ok_fullcall(text: str, *, mode: str, flags: dict, extracted: list[dict], tool_calls: Any = None) -> tuple[bool, str]:
    raw = (text or "").strip()
    if mode == "native_tools":
        if not tool_calls:
            return False, "missing_native_tool_calls"
        if raw and raw not in {"", "[]"}:
            # content in message is allowed empty; extra chatter is format fail
            if TRAIL_EXPLAIN_RE.search(raw):
                return False, "native_with_explanation"
        return True, "ok"
    if mode == "format_constrained":
        blob = _load_json_blob(raw)
        if blob is None:
            return False, "not_json"
        if not isinstance(blob, list):
            return False, "not_array"
        if len(blob) > 1:
            return False, "multi_call"
        if blob and (not isinstance(blob[0], dict) or "name" not in blob[0] or "arguments" not in blob[0]):
            return False, "envelope"
        return True, "ok"
    # prompt_adapted_raw: exact envelope, no fence, no parameters alias, no trailing explain
    if flags.get("fenced"):
        return False, "markdown_fence"
    if flags.get("used_parameters"):
        return False, "parameters_alias"
    if flags.get("wrapper"):
        return False, "wrapper"
    if flags.get("trailing_text"):
        return False, "trailing_text"
    if raw in {"[]", ""}:
        return True, "ok"
    blob = _load_json_blob(raw)
    if not isinstance(blob, list):
        return False, "not_array"
    if len(blob) > 1:
        return False, "multi_call"
    if blob:
        item = blob[0]
        if not isinstance(item, dict) or "name" not in item or "arguments" not in item:
            return False, "envelope"
        extra = set(item) - {"name", "arguments"}
        if extra:
            return False, "extra_keys"
    # raw text should be exactly the JSON (allow trailing newline)
    try:
        dumped = json.dumps(blob, ensure_ascii=False, separators=(",", ":"))
        compact = "".join(raw.split())
        if compact not in {dumped, json.dumps(blob, ensure_ascii=False)}:
            # still accept equivalent JSON whitespace
            json.loads(raw)
    except json.JSONDecodeError:
        return False, "not_json"
    return True, "ok"


def legal_calls(pred: list[dict], allowed_names: set[str]) -> tuple[bool, str]:
    if not pred:
        return True, "ok"
    if len(pred) > 1:
        return False, "multi_call"
    name = str(pred[0].get("name") or "")
    if name not in allowed_names:
        return False, "unknown_tool"
    return True, "ok"


def content_match(pred: list[dict], gold: list[dict]) -> dict[str, bool]:
    gold_n = [c.get("name") for c in gold]
    pred_n = [c.get("name") for c in pred]
    execute_gold = bool(gold)
    execute_pred = bool(pred)
    name_ok = pred_n == gold_n
    args_ok = False
    if name_ok:
        if not gold:
            args_ok = True
        else:
            ga = gold[0].get("arguments") or {}
            pa = pred[0].get("arguments") or {}
            args_ok = set(ga) == set(pa) and all(values_equal(ga[k], pa[k]) for k in ga)
    return {
        "execute_refuse": execute_pred == execute_gold,
        "tool_name": name_ok,
        "arguments": args_ok,
        "content_exact": calls_equal(pred, gold),
    }


def score_fullcall(
    row: dict,
    *,
    raw_text: str,
    mode: str,
    tool_calls: Any = None,
    selected_tools: list[dict] | None = None,
) -> dict[str, Any]:
    gold = list(row.get("answers") or [])
    tools = selected_tools or row.get("retrieved_tools_oracle") or row.get("catalog_tools") or []
    if tools and isinstance(tools[0], str):
        names = set(tools)
    else:
        names = {str(t.get("name")) for t in tools if t.get("name")}
    extracted, flags = extract_fullcall_calls(raw_text, tool_calls=tool_calls)
    content = content_match(extracted, gold)
    fmt_ok, fmt_reason = format_ok_fullcall(raw_text, mode=mode, flags=flags, extracted=extracted, tool_calls=tool_calls)
    legal, legal_reason = legal_calls(extracted, names)
    strict = bool(content["content_exact"] and fmt_ok and legal)
    return {
        "content": content,
        "format_ok": fmt_ok,
        "format_reason": fmt_reason,
        "legal": legal,
        "legal_reason": legal_reason,
        "strict_e2e": strict,
        "extracted": extracted,
        "flags": flags,
        "mode": mode,
    }


def extract_mw(text: str) -> tuple[str, dict[str, Any]]:
    raw = (text or "").strip()
    flags = {"empty": not raw, "multi": False, "substring": False, "unknown": False}
    if not raw:
        return "", flags
    hits = [c for c in REASON_CODES_16 if c in raw]
    unique = list(dict.fromkeys(hits))
    if len(unique) > 1:
        flags["multi"] = True
        return unique[0], flags
    if len(unique) == 1:
        flags["substring"] = raw != unique[0]
        return unique[0], flags
    token = raw.split()[0].strip(".,;:。，[]()\"'`")
    flags["unknown"] = token not in REASON_CODES_16
    return token, flags


def format_ok_mw(raw: str, extracted: str, flags: dict) -> tuple[bool, str]:
    text = (raw or "").strip()
    if not text:
        return False, "empty"
    if flags.get("multi"):
        return False, "multi_code"
    if text != extracted or extracted not in REASON_CODES_16:
        return False, "not_bare_code"
    return True, "ok"


def score_mw(row: dict, *, raw_text: str) -> dict[str, Any]:
    gold = str(row.get("reason_code") or "")
    extracted, flags = extract_mw(raw_text)
    content_ok = extracted == gold and gold in REASON_CODES_16
    fmt_ok, fmt_reason = format_ok_mw(raw_text, extracted, flags)
    strict = content_ok and fmt_ok
    unsafe = gold != "ready_to_execute" and extracted == "ready_to_execute"
    return {
        "gold": gold,
        "pred": extracted,
        "content_ok": content_ok,
        "format_ok": fmt_ok,
        "format_reason": fmt_reason,
        "strict_e2e": strict,
        "unsafe_execute": unsafe,
        "flags": flags,
    }


def confusion(rows: list[dict]) -> dict[str, Any]:
    codes = list(REASON_CODES_16) + ["<illegal_or_empty>"]
    matrix = {g: {p: 0 for p in codes} for g in REASON_CODES_16}
    per = {c: {"tp": 0, "fp": 0, "fn": 0} for c in REASON_CODES_16}
    for row in rows:
        gold = row["gold"]
        pred = row["pred"] if row["pred"] in REASON_CODES_16 else "<illegal_or_empty>"
        if gold in matrix:
            matrix[gold][pred] = matrix[gold].get(pred, 0) + 1
        if gold in per:
            if pred == gold:
                per[gold]["tp"] += 1
            else:
                per[gold]["fn"] += 1
                if pred in per:
                    per[pred]["fp"] += 1
    f1s = []
    per_class = {}
    for c, st in per.items():
        prec = st["tp"] / max(1, st["tp"] + st["fp"])
        rec = st["tp"] / max(1, st["tp"] + st["fn"])
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        per_class[c] = {"precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6), **st}
        f1s.append(f1)
    return {
        "per_class": per_class,
        "macro_f1": round(sum(f1s) / max(1, len(f1s)), 6),
        "matrix": matrix,
    }


def cascade_score(*, retrieval_hit: bool, generator_strict: bool, generator_content: bool, generator_format: bool) -> dict[str, Any]:
    pipeline_fail_reason = None
    if not retrieval_hit:
        pipeline_fail_reason = "retrieval_miss"
        pipeline_strict = False
    elif not generator_strict:
        pipeline_fail_reason = "generator"
        pipeline_strict = False
    else:
        pipeline_strict = True
    return {
        "retrieval_hit@5": retrieval_hit,
        "generator_strict_given_hit": bool(retrieval_hit and generator_strict),
        "generator_content_fail": bool(retrieval_hit and not generator_content),
        "generator_format_fail": bool(retrieval_hit and generator_content and not generator_format),
        "pipeline_strict": pipeline_strict,
        "pipeline_fail_reason": pipeline_fail_reason,
    }
