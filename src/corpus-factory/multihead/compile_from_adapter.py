#!/usr/bin/env python3
"""桥接：新框架 adapter（TaskEvidence）→ 老脚本 make_view 格式 → compile_views 能吃的 views.jsonl。

定位：填补框架缺口。新框架 multihead/ 有 6 个 adapter（moss/toolace/api_bank/
crosswoz/risawoz/nemotron），把 raw 解析成统一 TaskEvidence；但 TaskEvidence 的
字段与 compile_views.py 要吃的 view 字段（calls/schema_errors/behavior/
fits_five_tool_catalog/encoding/id/group_id）不匹配。老脚本 public_sft_scale_trial.
make_view 正好产出这些字段，但只硬编码了 MOSS/ToolACE 两个 source。

本脚本把「adapter 的 TaskEvidence」喂给「老脚本 make_view」，产出 compile_views
能吃的 views.jsonl。接入新数据集（CrossWOZ/RiSAWOZ 等）零新 adapter 代码，只需
确保 adapter 产出标准 TaskEvidence。

复用的现成代码：
- public_sft_scale_trial.make_view（老脚本，已验证与 compile_views 零 drift）
- multihead adapter（新框架，已把 raw 解析成 TaskEvidence）

归一化派生规则（有依据，非猜值）：
1. 剥离 format 注解：format 是 JSON Schema 纯 annotation，不改变类型校验，
   byte_grammar 约束解码也不处理 format。API-Bank 源 schema 大量含 format。
2. 补全 OpenAI 函数格式省略的 type：Nemotron 的 parameters 顶层缺 type（隐含
   object）、enum 参数缺 type（隐含 string）。schema_check 要求显式 type。
   规则：有 properties/required→object；有 items→array；有 enum→string。
3. 字符串数字 → 数字：源数据把 integer/number 参数值写成字符串（"2000" vs 2000），
   schema_check 类型校验失败。schema 声明 integer/number 且值可解析为数字时转数字。

用法（需 .venv，因 make_view 加载 tokenizer 做 roundtrip 校验）：
    .venv/bin/python compile_from_adapter.py --source-id api_bank --raw <目录> --out <views.jsonl> --max-records 200
    .venv/bin/python compile_from_adapter.py --source-id nemotron --raw <jsonl> --out <views.jsonl> --max-records 200
    .venv/bin/python compile_from_adapter.py --source-id api_bank --raw <目录> --levels 1 --max-records 5000 --out <views.jsonl>

产出 views.jsonl 后，再用 compile_views.py 出 fullcall/retrieval。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 平铺 import：老脚本在 sources/，adapter 在本目录（multihead/）。
_MULTIHEAD = Path(__file__).resolve().parent
_SOURCES = _MULTIHEAD.parent / "sources"
for _p in (str(_SOURCES), str(_MULTIHEAD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from public_sft_scale_trial import make_view  # noqa: E402
from source_manager import load_tokenizer  # noqa: E402
from profiling import resolve_path  # noqa: E402
import adapter_api_bank  # noqa: E402,F401  触发注册
import adapter_nemotron  # noqa: E402,F401  触发注册
import adapter_crosswoz  # noqa: E402,F401  触发注册
import adapter_risawoz  # noqa: E402,F401  触发注册
import adapter_msagent_bench  # noqa: E402,F401  触发注册
from adapter_base import adapter_for, AdapterLimits  # noqa: E402

_DEFAULT_MANIFEST = "models/mei-1.2-51m/tokenizer/candidates/mei-24k-lossless-hans-en-20260914-v1/RELEASE.json"


def _normalize_schema(node: Any) -> Any:
    """剥离 format 注解 + 补全 OpenAI 函数格式省略的 type（见模块 docstring 规则 1/2）。"""
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k == "format":
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _normalize_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            out[k] = _normalize_schema(v)
        else:
            out[k] = v
    if "type" not in out:
        if "properties" in out or "required" in out:
            out["type"] = "object"
        elif "items" in out:
            out["type"] = "array"
        elif "enum" in out:
            out["type"] = "string"
    return out


def _normalize_args(args: dict[str, Any], props: dict[str, Any]) -> dict[str, Any]:
    """字符串数字 → 数字（schema 声明 integer/number，见模块 docstring 规则 3）。"""
    out: dict[str, Any] = {}
    for k, v in args.items():
        schema = props.get(k, {})
        st = schema.get("type")
        if st in ("integer", "number") and isinstance(v, str):
            s = v.strip()
            try:
                out[k] = int(s) if st == "integer" else float(s)
                continue
            except (ValueError, TypeError):
                pass
        out[k] = v
    return out


def evidence_to_views(evidence: Any, tokenizer: Any) -> list[dict[str, Any]]:
    """TaskEvidence → 老脚本 make_view 的 view dict 列表（只取含 tool_calls 的 assistant 事件）。"""
    gid = evidence.identity.case_id
    tools = [dict(t) for t in evidence.tool_env.catalog]
    for t in tools:
        t["parameters"] = _normalize_schema(t.get("parameters", {}))
    by_name = {t["name"]: t for t in tools}
    timeline = evidence.timeline
    views: list[dict[str, Any]] = []
    for i, event in enumerate(timeline):
        if event.role != "assistant" or not event.tool_calls:
            continue
        calls = []
        for c in event.tool_calls:
            props = (by_name.get(c.name) or {}).get("parameters", {}).get("properties", {})
            calls.append({"name": c.name, "arguments": _normalize_args(c.arguments, props)})
        history = [{"event_id": e.event_id, "role": e.role, "content": (e.content or "")}
                   for e in timeline[:i]]
        after = [{"event_id": e.event_id, "role": e.role, "content": (e.content or "")}
                 for e in timeline[i + 1:]]
        raw_call = json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
        try:
            view = make_view(gid, i, calls, tools, history, after, raw_call, tokenizer)
        except (ValueError, TypeError, KeyError):
            continue
        view["source"] = evidence.identity.source_id
        views.append(view)
    return views


def main() -> int:
    parser = argparse.ArgumentParser(description="adapter TaskEvidence → make_view 格式 views.jsonl")
    parser.add_argument("--source-id", required=True, help="已注册的 source_id（api_bank/nemotron/...）")
    parser.add_argument("--raw", required=True, help="原始数据路径（api_bank 为目录，nemotron 为 jsonl）")
    parser.add_argument("--out", required=True, help="产出 views.jsonl 路径")
    parser.add_argument("--max-records", type=int, default=200, help="试批上限")
    parser.add_argument("--levels", help="只跑指定 level（逗号分隔，如 1 或 1,2,3；仅 api_bank 生效）")
    parser.add_argument("--tokenizer-manifest", default=_DEFAULT_MANIFEST)
    args = parser.parse_args()

    tokenizer = load_tokenizer(resolve_path(args.tokenizer_manifest))
    adapter = adapter_for(args.source_id)()
    if args.levels and args.source_id == "api_bank":
        raw_dir = Path(args.raw)
        evidences = []
        for lv in [int(x) for x in args.levels.split(",")]:
            api_path = raw_dir / f"training-data__lv{lv}-api-train.json"
            resp_path = raw_dir / f"training-data__lv{lv}-response-train.json"
            evs = adapter._load_level(api_path, resp_path, lv, AdapterLimits(max_records=args.max_records))
            evidences.extend(evs)
            if len(evidences) >= args.max_records:
                break
        evidences = evidences[: args.max_records]
    else:
        evidences = adapter.load(Path(args.raw), limits=AdapterLimits(max_records=args.max_records))

    views: list[dict[str, Any]] = []
    for ev in evidences:
        views.extend(evidence_to_views(ev, tokenizer))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in views) + "\n", encoding="utf-8")

    schema_clean = sum(1 for v in views if not v.get("schema_errors") and v.get("calls"))
    single_call = sum(1 for v in views if len(v.get("calls") or []) == 1)
    fits5 = sum(1 for v in views if v.get("fits_five_tool_catalog"))
    fits2048 = sum(1 for v in views if (v.get("encoding") or {}).get("fits_2048"))
    tools = {t["name"] for ev in evidences for t in ev.tool_env.catalog}
    print(json.dumps({
        "evidences": len(evidences),
        "views": len(views),
        "schema_clean_nonempty": schema_clean,
        "single_call": single_call,
        "fits_five_tool_catalog": fits5,
        "fits_2048": fits2048,
        "distinct_tools": len(tools),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
