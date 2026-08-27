#!/usr/bin/env python3
"""Runtime/evaluator RAG parity tests. Uses tiny random 58M, no training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from repo_paths import MODEL_MEI_58M, SCRIPTS_ROOT

sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(MODEL_MEI_58M))

import mlx.core as mx

from architecture import NeedleZh
from config import NeedleZhConfig
from retrieval_rag_protocol import (
    CatalogIndex,
    encode_text_ids,
    rank_by_dot,
    render_tool_text,
    select_topk_tools,
)
from runtime_v2 import RuntimeV2
from tokenizer import ZhTokenizerV1
from tool_index import ToolIndex

CATALOG = [
    {"name": "get_weather", "description": "查询城市天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "set_lights", "description": "设置房间灯光", "parameters": {"type": "object", "properties": {"room": {"type": "string"}}, "required": ["room"]}},
    {"name": "set_switch", "description": "开关切换", "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "pay_invoice", "description": "支付发票", "parameters": {"type": "object", "properties": {"invoice_id": {"type": "string"}}, "required": ["invoice_id"]}},
    {"name": "book_flight", "description": "预订航班", "parameters": {"type": "object", "properties": {"from_city": {"type": "string"}}, "required": ["from_city"]}},
    {"name": "track_shipment", "description": "追踪运单", "parameters": {"type": "object", "properties": {"waybill": {"type": "string"}}, "required": ["waybill"]}},
    {"name": "start_incubator", "description": "启动培养箱", "parameters": {"type": "object", "properties": {"chamber": {"type": "string"}}, "required": ["chamber"]}},
]


def _tiny():
    cfg = NeedleZhConfig().tiny(contrastive_head_v2=True, confidence_v2=True)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    return model, ZhTokenizerV1()


def test_short_circuit() -> dict:
    model, tok = _tiny()
    rt = RuntimeV2(model, tok, catalog=CATALOG[:3])
    got = rt._select("随便", CATALOG[:3])
    return {"ok": [t["name"] for t in got] == [t["name"] for t in CATALOG[:3]], "n": len(got)}


def test_builds_index_and_raw_query() -> dict:
    model, tok = _tiny()
    rt = RuntimeV2(model, tok, catalog=CATALOG)
    selected = rt._select("查询天气", CATALOG)
    names = [t["name"] for t in selected]
    return {
        "ok": len(selected) == 5 and len(rt.index.records) == len(CATALOG) and "get_weather" in {t["name"] for t in CATALOG},
        "n_index": len(rt.index.records),
        "selected": names,
        "schema_fp": bool(rt._index_schema_fp),
    }


def test_fingerprint_invalidation() -> dict:
    model, tok = _tiny()
    rt = RuntimeV2(model, tok, catalog=CATALOG)
    rt._select("查询天气", CATALOG)
    fp1 = rt._index_schema_fp
    changed = list(CATALOG) + [
        {"name": "new_tool", "description": "全新工具", "parameters": {"type": "object", "properties": {}, "required": []}}
    ]
    rt._select("查询天气", changed)
    return {"ok": rt._index_schema_fp != fp1 and len(rt.index.records) == len(changed), "old": fp1, "new": rt._index_schema_fp}


def test_brute_force_matches_index() -> dict:
    model, tok = _tiny()
    rt = RuntimeV2(model, tok, catalog=CATALOG)

    def encode(text: str):
        return [float(x) for x in rt._embed_text(text).tolist()]

    idx = CatalogIndex(encode, k=5)
    idx.build(CATALOG)
    query = "支付那张发票"
    protocol_names = [t["name"] for t in idx.search(query)]
    runtime_names = [t["name"] for t in rt._select(query, CATALOG)]
    items = list(zip(idx.names, idx.vectors))
    brute = [n for n, _ in rank_by_dot(encode(query), items, k=5)]
    return {
        "ok": protocol_names == brute and set(runtime_names) == set(protocol_names) and len(runtime_names) == 5,
        "protocol": protocol_names,
        "runtime": runtime_names,
        "brute": brute,
    }


def test_no_match_still_returns_five() -> dict:
    model, tok = _tiny()
    rt = RuntimeV2(model, tok, catalog=CATALOG)
    got = rt._select("碱金属遇水为什么会放氢", CATALOG)
    return {"ok": len(got) == 5, "names": [t["name"] for t in got]}


def test_render_is_compact_schema() -> dict:
    text = render_tool_text(CATALOG[0])
    return {"ok": text.startswith("<tools>") and "get_weather" in text and "parameters" in text, "n": len(text)}


def main() -> int:
    report = {
        "short_circuit": test_short_circuit(),
        "index_build": test_builds_index_and_raw_query(),
        "invalidate": test_fingerprint_invalidation(),
        "parity": test_brute_force_matches_index(),
        "no_match": test_no_match_still_returns_five(),
        "render": test_render_is_compact_schema(),
    }
    ok = all(v.get("ok") for v in report.values())
    print(json.dumps({"ok": ok, **report}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
