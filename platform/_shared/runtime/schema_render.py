"""Canonical schema-conditioned prompt renderer (train = student = Qwen)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SERIALIZER_ID = "mei-schema-serializer-v1"
ROUTE_SERIALIZER_ID = "mei-route-serializer-v1"
PRODUCT_SFT_SEQ_LEN = 2048
TOOLS_OPEN = "<tools>"
TOOLS_CLOSE = "</tools>"
TASK_CONTRACT = (
    "任务：只从当前请求 <routes> 里选择一个 route_id，或拒绝。"
    "只输出 [] 或 {\"route_id\":整数}。"
    "禁止输出工具名、参数名、参数值或解释。"
)


def load_toolset_json(toolset_id: str, *, root: Path | None = None) -> dict[str, Any]:
    if not toolset_id:
        raise ValueError("toolset_id is required; no default VRM")
    if root is None:
        from _repo import EVAL_SHARED_ROOT

        root = EVAL_SHARED_ROOT
    path = Path(root) / "toolsets" / f"{toolset_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing toolset {toolset_id}: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw.get("tools"):
        raise ValueError(f"toolset {toolset_id} has no tools")
    return raw


def compact_tools(toolset: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in toolset.get("tools") or []:
        params = tool.get("parameters") or {"type": "object", "properties": {}}
        out.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description") or "",
                "parameters": params,
            }
        )
    return out


def dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def schema_hash(toolset: dict[str, Any]) -> str:
    payload = dumps_canonical(compact_tools(toolset))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_tools_block(toolset: dict[str, Any]) -> str:
    return TOOLS_OPEN + dumps_canonical(compact_tools(toolset)) + TOOLS_CLOSE


def format_user_text(row: dict[str, Any], *, lang: str | None = None) -> str:
    query = str(row.get("query") or "").strip()
    scene = row.get("scene")
    use_en = str(lang or row.get("lang") or "zh").lower().startswith("en")
    if use_en:
        if isinstance(scene, str) and scene.strip():
            return f"Scene: {scene.strip()}. User: {query}"
        return f"User: {query}"
    if isinstance(scene, str) and scene.strip():
        return f"场景：{scene.strip()}。用户：{query}"
    return f"用户：{query}"


def render_entities_block(entities: list[dict[str, Any]] | None) -> str:
    if not entities:
        return ""
    lines = []
    for ent in entities:
        canon = ent.get("canonical")
        et = ent.get("type")
        eid = ent.get("entity_id")
        lines.append(f"{eid}:{et}={canon}")
    return "<entities>" + dumps_canonical(lines) + "</entities>"


def render_routes_block(manifest_dict: dict[str, Any], *, manifest_hash: str = "") -> str:
    rows = []
    for r in manifest_dict.get("routes") or []:
        rows.append(
            {
                "route_id": r.get("route_id"),
                "name": r.get("name"),
                "arguments": r.get("arguments") or {},
            }
        )
    inner = dumps_canonical({"hash": manifest_hash, "routes": rows})
    return "<routes>" + inner + "</routes>"


def render_route_request(
    row: dict[str, Any],
    toolset: dict[str, Any],
    *,
    manifest_dict: dict[str, Any],
    manifest_hash: str = "",
    entities: list[dict[str, Any]] | None = None,
    lang: str | None = None,
) -> str:
    parts = [TASK_CONTRACT, render_tools_block(toolset)]
    scene = row.get("scene")
    if isinstance(scene, str) and scene.strip():
        parts.append("<scene>" + scene.strip() + "</scene>")
    ent = entities if entities is not None else row.get("entities")
    if isinstance(ent, list) and ent:
        parts.append(render_entities_block(ent))
    parts.append(render_routes_block(manifest_dict, manifest_hash=manifest_hash))
    query = str(row.get("query") or "").strip()
    use_en = str(lang or row.get("lang") or "zh").lower().startswith("en")
    parts.append(("User: " if use_en else "用户：") + query)
    return "\n".join(parts)


def render_request(row: dict[str, Any], toolset: dict[str, Any], *, lang: str | None = None) -> str:
    serializer = str(row.get("serializer") or SERIALIZER_ID)
    if serializer == ROUTE_SERIALIZER_ID or str(row.get("protocol") or "") == "mei-route-protocol-v1":
        from route_compiler import compile_routes
        from route_protocol import manifest_hash as _mh
        from candidates import ToolContext

        ctx = ToolContext(
            query=str(row.get("query") or ""),
            scene=row.get("scene") if isinstance(row.get("scene"), str) else None,
            toolset=toolset,
            entities=list(row.get("entities") or []),
            lexicon=dict(row.get("lexicon") or {}),
            param_types=dict(row.get("param_types") or {}),
        )
        manifest = compile_routes(ctx)
        return render_route_request(
            row,
            toolset,
            manifest_dict=manifest.as_dict(),
            manifest_hash=_mh(manifest),
            entities=ctx.entities,
            lang=lang,
        )
    return render_tools_block(toolset) + format_user_text(row, lang=lang)


def resolve_toolset(
    row: dict[str, Any] | None = None,
    toolset: dict[str, Any] | None = None,
    toolset_id: str | None = None,
) -> dict[str, Any]:
    if toolset is not None:
        if "tools" not in toolset:
            raise ValueError("toolset must be a raw JSON object with tools")
        return toolset
    tid = toolset_id or (str((row or {}).get("toolset_id") or "") if row else "")
    if not tid:
        raise ValueError("toolset or toolset_id is required; no default VRM")
    return load_toolset_json(tid)
