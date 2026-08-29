"""51M packed-Q4 runtime: load, retrieve, constrained decode, validate, confidence gate."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from .errors import SdkError
from .package import ModelPackage
from .protocol import MAX_SELECTED_TOOLS, parse_v2_text, render_request
from .version import SDK_ROOT

_MEI_LLM = SDK_ROOT.parent
_ARCH_51 = _MEI_LLM / "architecture" / "mei-1.0-51m-arch-v1"
_TRAIN = _MEI_LLM / "training" / "mei-1.0-58m-train-v1"
_RUNTIME_58 = _MEI_LLM / "runtime" / "mei-1.0-58m-needle2-v2"
_RUNTIME_SHARED = _MEI_LLM / "runtime" / "_shared"

EXECUTE_HIGH = 0.70
ESCALATE_LOW = 0.35


def _ensure(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _token_bytes(tokenizer, tok_id: int) -> bytes:
    if tok_id in {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}:
        return b""
    piece = tokenizer.sp.id_to_piece(int(tok_id))
    if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
        try:
            return bytes([int(piece[3:5], 16)])
        except ValueError:
            return b""
    if piece.startswith("▁"):
        piece = " " + piece[1:]
    try:
        return piece.encode("utf-8")
    except UnicodeEncodeError:
        return b""


def call_templates(tools: list[dict[str, Any]]) -> list[str]:
    """Canonical JSON strings the constrained decoder may emit."""
    templates = ["[]"]
    for tool in tools:
        name = str(tool.get("name") or "")
        if not name:
            continue
        params = tool.get("parameters") or {}
        props = params.get("properties") or {}
        bool_keys = [k for k, spec in props.items() if (spec or {}).get("type") == "boolean"][:3]
        if not bool_keys:
            templates.append(
                json.dumps([{"name": name, "arguments": {}}], ensure_ascii=False, separators=(",", ":"))
            )
            continue
        for mask in range(1 << len(bool_keys)):
            args = {bool_keys[i]: bool((mask >> i) & 1) for i in range(len(bool_keys))}
            templates.append(
                json.dumps([{"name": name, "arguments": args}], ensure_ascii=False, separators=(",", ":"))
            )
    return templates


def _legal_json_prefix(prefix: str, templates: list[str]) -> bool:
    if not prefix:
        return True
    return any(tmpl.startswith(prefix) for tmpl in templates)


def _is_accept(prefix: str) -> bool:
    try:
        obj = json.loads(prefix)
    except json.JSONDecodeError:
        return False
    if obj == []:
        return True
    return isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict) and "name" in obj[0] and "arguments" in obj[0]


def mask_logits_for_grammar(logits, tokenizer, prefix_bytes: bytes, templates: list[str], *, k: int = 1024):
    import mlx.core as mx

    row = logits[0] if logits.ndim == 2 else logits
    prefix = prefix_bytes.decode("utf-8", errors="ignore")
    vocab = int(row.shape[-1])
    if _is_accept(prefix) or any(tmpl == prefix for tmpl in templates):
        mask = mx.arange(vocab) == int(tokenizer.eos_id)
        return mx.where(mask, row, mx.array(-1e9, dtype=row.dtype))
    k = min(int(k), vocab - 1)
    ranked = mx.argpartition(-row.astype(mx.float32), kth=k)[:k]
    mx.eval(ranked)
    allowed: list[int] = []

    def consider(tok: int) -> None:
        if tok in {tokenizer.pad_id, tokenizer.bos_id}:
            return
        extra = _token_bytes(tokenizer, tok)
        if not extra:
            return
        trial = prefix_bytes + extra
        try:
            text = trial.decode("utf-8")
        except UnicodeDecodeError:
            return
        if _legal_json_prefix(text, templates):
            allowed.append(tok)

    for tok in ranked.tolist():
        consider(int(tok))
        if len(allowed) >= 16:
            break
    if not allowed:
        for tok in range(vocab):
            consider(tok)
            if len(allowed) >= 16:
                break
    if not allowed:
        return row
    mask = mx.zeros((vocab,), dtype=mx.bool_)
    for tok in allowed:
        mask = mx.logical_or(mask, mx.arange(vocab) == tok)
    return mx.where(mask, row, mx.array(-1e9, dtype=row.dtype))


def validate_call(text: str, *, tools: list[dict[str, Any]], query: str, system_facts: str = "") -> dict[str, Any]:
    parsed = parse_v2_text(text)
    names = {str(t.get("name") or "") for t in tools}
    if not parsed.get("ok"):
        return {
            "ok": False,
            "function_calls": [],
            "error": parsed.get("error") or "parse",
            "refuse": True,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
        }
    if parsed.get("refuse"):
        return {
            "ok": True,
            "function_calls": [],
            "error": None,
            "refuse": True,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
        }
    call = parsed["function_calls"][0]
    name = str(call.get("name") or "")
    if name not in names:
        return {
            "ok": False,
            "function_calls": [],
            "error": "unsupported_tool",
            "refuse": True,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
        }
    tool = next(t for t in tools if str(t.get("name") or "") == name)
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    required = [str(x) for x in (params.get("required") or [])]
    args = call.get("arguments") or {}
    evidence_blob = " ".join([query or "", system_facts or ""])
    unsupported = 0
    unprov = 0
    for key, value in args.items():
        if key not in props:
            unsupported += 1
        else:
            needle = str(value)
            if needle and needle not in evidence_blob and needle not in {"true", "false", "[]", "{}"}:
                # booleans/empty containers may be schema defaults; strings/numbers need evidence
                if not isinstance(value, bool):
                    unprov += 1
    missing_required = [k for k in required if k not in args]
    if unsupported or unprov or missing_required:
        return {
            "ok": False,
            "function_calls": [],
            "error": "validator",
            "refuse": True,
            "unsupported_accepted": 0,
            "unprovenanced_argument_accepted": 0,
            "would_have_accepted_unsupported": unsupported,
            "would_have_accepted_unprovenanced": unprov,
            "missing_required": missing_required,
        }
    return {
        "ok": True,
        "function_calls": [call],
        "error": None,
        "refuse": False,
        "unsupported_accepted": 0,
        "unprovenanced_argument_accepted": 0,
    }


def apply_confidence_gate(validated: dict[str, Any], confidence: float | None) -> dict[str, Any]:
    """confidence actually changes execute / escalate / refuse."""
    out = dict(validated)
    value = None if confidence is None else float(confidence)
    out["confidence_value"] = value
    if not validated.get("ok") or validated.get("refuse"):
        out["execution"] = "refuse"
        return out
    if value is None:
        out["ok"] = False
        out["refuse"] = True
        out["function_calls"] = []
        out["error"] = "confidence_unavailable"
        out["execution"] = "refuse"
        return out
    if value >= EXECUTE_HIGH:
        out["execution"] = "execute"
        return out
    if value >= ESCALATE_LOW:
        out["execution"] = "escalate"
        out["ok"] = True
        out["refuse"] = True
        out["function_calls"] = []
        out["error"] = None
        out["escalate"] = True
        return out
    out["execution"] = "refuse"
    out["ok"] = True
    out["refuse"] = True
    out["function_calls"] = []
    return out


class Runtime51M:
    def __init__(self, model, tokenizer, *, contrastive=None, conf_v2=None, index=None):
        self.model = model
        self.tokenizer = tokenizer
        self.contrastive = contrastive
        self.conf_v2 = conf_v2
        self.index = index or {}
        self._tool_vecs: dict[str, Any] = {}
        self._catalog_fp = ""

    def embed_text(self, text: str):
        import mlx.core as mx

        ids = self.tokenizer.encode(text, add_bos=True, add_eos=False)[:256]
        arr = mx.array([ids], dtype=mx.int32)
        out = self.model(arr, return_cells=True)
        cells = out.get("cells")
        if self.contrastive is None or cells is None:
            hidden = out["hidden"][:, -1, :].astype(mx.float32)
            norm = mx.sqrt(mx.sum(hidden * hidden, axis=-1, keepdims=True) + 1e-8)
            return hidden / norm
        vec = self.contrastive(cells)
        mx.eval(vec)
        return vec[0]

    def build_index(self, catalog: list[dict[str, Any]]) -> None:
        from .canonical import schema_fingerprint

        fp = schema_fingerprint(catalog)
        if fp == self._catalog_fp and len(self._tool_vecs) == len(catalog):
            return
        self._tool_vecs = {}
        for tool in catalog:
            name = str(tool.get("name") or "")
            blob = json.dumps(
                {"name": name, "description": tool.get("description") or "", "parameters": tool.get("parameters") or {}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            self._tool_vecs[name] = (tool, self.embed_text(blob))
        self._catalog_fp = fp
        self.index = {"schema_fp": fp, "n": len(catalog)}

    def search_top_k(self, query: str, catalog: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
        import mlx.core as mx

        if len(catalog) <= k:
            return list(catalog)
        self.build_index(catalog)
        q = self.embed_text(query)
        scored = []
        for name, (tool, vec) in self._tool_vecs.items():
            sim = float(mx.sum(q * vec).item())
            scored.append((sim, tool))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [tool for _, tool in scored[:k]]

    def _forward(self, ids: list[int], *, cache=None, return_confidence: bool = False, return_cells: bool = False):
        import mlx.core as mx

        arr = mx.array([ids], dtype=mx.int32)
        return self.model(
            arr,
            cache=cache,
            return_confidence=return_confidence,
            return_cells=return_cells,
        )

    def greedy(
        self,
        prompt_ids: list[int],
        *,
        tools: list[dict[str, Any]],
        max_new: int = 96,
        decode_mode: str = "constrained",
    ) -> dict[str, Any]:
        import mlx.core as mx

        templates = call_templates(tools)
        pieces: list[int] = []
        prefix = b""
        logprob_sum = 0.0
        n_out = 0
        out = self._forward(prompt_ids, return_confidence=True, return_cells=True)
        logits = out["logits"][:, -1, :]
        cache = out.get("cache")
        cells = out.get("cells")
        conf_logit = out.get("confidence_logit")
        t0 = time.perf_counter()
        for _ in range(max_new):
            row = logits[0]
            if decode_mode != "raw":
                row = mask_logits_for_grammar(row, self.tokenizer, prefix, templates)
            tok = int(mx.argmax(row).item())
            logp = float((row[tok] - mx.logsumexp(row)).item())
            if tok == self.tokenizer.eos_id:
                break
            extra = _token_bytes(self.tokenizer, tok)
            prefix = prefix + extra
            pieces.append(tok)
            logprob_sum += logp
            n_out += 1
            text = prefix.decode("utf-8", errors="ignore")
            if decode_mode != "raw" and _is_accept(text):
                break
            step = self.model(
                mx.array([[tok]], dtype=mx.int32),
                cache=cache,
            )
            logits = step["logits"][:, -1, :]
            cache = step.get("cache")
        decode_ms = (time.perf_counter() - t0) * 1000
        text = prefix.decode("utf-8", errors="ignore")
        conf = None
        if self.conf_v2 is not None and cells is not None:
            logit = self.conf_v2(cells)
            mx.eval(logit)
            conf = combine_conf(float(logit.item()), logprob_sum)
        elif conf_logit is not None:
            mx.eval(conf_logit)
            conf = combine_conf(float(conf_logit[0].item()), logprob_sum)
        return {
            "text": text,
            "ids": pieces,
            "n_out": n_out,
            "decode_ms": decode_ms,
            "confidence": conf,
            "logprob_sum": logprob_sum,
        }


def combine_conf(head_logit: float, decode_logprob: float) -> float:
    p_head = 1.0 / (1.0 + math.exp(-float(head_logit)))
    p_dec = float(math.exp(min(0.0, decode_logprob)))
    return min(p_head, p_dec)


def load_51m_runtime(package: ModelPackage, *, backend: str = "mlx-reference"):
    import mlx.core as mx
    import mlx.utils as xu
    import numpy as np

    _ensure(_ARCH_51)
    _ensure(_TRAIN)
    from architecture import NeedleZh, count_params
    from config import NeedleZhConfig
    from tokenizer import ZhTokenizerV1

    from quant_pack_51m import load_pack_file

    weights = (package.path / str(package.manifest["weights"]["file"])).resolve()
    tokenizer_path = (package.path / str(package.manifest["tokenizer"]["file"])).resolve()
    if not weights.is_file():
        raise SdkError("file_not_found", f"missing weights: {weights}")
    tok = ZhTokenizerV1(tokenizer_path)
    arch = package.manifest.get("architecture") or {}
    cfg = NeedleZhConfig.from_spec(_ARCH_51 / "spec" / "model.json")
    if str(arch.get("id") or cfg.architecture_id) != cfg.architecture_id:
        raise SdkError("package_invalid", f"architecture id mismatch: {arch.get('id')}")
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    fmt = str(package.manifest["weights"].get("format") or "")
    if fmt == "mei-q4-packed-v1":
        _header, arrays = load_pack_file(weights)
        from checkpoint import flatten_params

        current = flatten_params(model)
        n = 0
        for name, val in current.items():
            if name not in arrays:
                continue
            arr = arrays[name]
            if tuple(arr.shape) != tuple(val.shape):
                raise SdkError("package_invalid", f"tensor shape mismatch {name}")
            current[name] = mx.array(np.asarray(arr, dtype=np.float32))
            n += 1
        model.update(xu.tree_unflatten(list(current.items())))
        mx.eval(model.parameters())
        n_loaded = n
    else:
        from checkpoint import load_params

        report = load_params(model, weights, strict=False, return_report=True)
        n_loaded = (report or {}).get("n_loaded") if isinstance(report, dict) else report
    if count_params(model) != 51_463_797:
        raise SdkError("package_invalid", "51M param count mismatch after load")
    contrastive = None
    conf_v2 = None
    head_path = package.path / "heads.npz"
    if head_path.is_file():
        from heads import ConfidenceV2Head, ContrastiveHead

        try:
            blob = mx.load(str(head_path))
        except Exception:
            blob = {}
        if any(k.startswith("contrastive") for k in blob):
            contrastive_meta = (package.manifest.get("heads") or {}).get("contrastive") or {}
            contrastive = ContrastiveHead(
                cfg.d_model,
                cfg.n_layers,
                dim=int(contrastive_meta.get("dim") or 128),
                probes=int(contrastive_meta.get("probes") or 4),
            )
            mx.eval(contrastive.parameters())
            contrastive.update(_unflatten_prefix(blob, "contrastive"))
            mx.eval(contrastive.parameters())
        if any(k.startswith("conf_v2") for k in blob):
            conf_v2 = ConfidenceV2Head(cfg.d_model)
            mx.eval(conf_v2.parameters())
            conf_v2.update(_unflatten_prefix(blob, "conf_v2"))
            mx.eval(conf_v2.parameters())
    runtime = Runtime51M(model, tok, contrastive=contrastive, conf_v2=conf_v2)
    return runtime, {
        "n_loaded": n_loaded,
        "weights": str(weights),
        "tokenizer": str(tokenizer_path),
        "package_id": package.package_id,
        "backend": backend,
        "product": "mei-1.0-51m",
        "inference": True,
    }


def _unflatten_prefix(blob: dict, prefix: str):
    import mlx.utils as xu

    items = []
    for key, val in blob.items():
        if key == prefix or key.startswith(prefix + "."):
            rel = key[len(prefix) :].lstrip(".")
            items.append((rel or key, val))
    return xu.tree_unflatten(items) if items else {}


def complete_51m(runtime: Runtime51M, request: dict[str, Any]) -> dict[str, Any]:
    query = str(request.get("query") or "")
    oracle = request.get("oracle_tools")
    catalog = list(request.get("catalog") or [])
    if oracle is not None:
        selected = list(oracle)
        if len(selected) > MAX_SELECTED_TOOLS:
            raise SdkError("too_many_tools")
    else:
        selected = runtime.search_top_k(query, catalog, k=MAX_SELECTED_TOOLS) if catalog else []
        if not selected:
            selected = catalog[:MAX_SELECTED_TOOLS]
    rendered = render_request(request, selected)
    ids = runtime.tokenizer.encode(rendered["prompt"], add_bos=True, add_eos=False)
    t_prefill = time.perf_counter()
    decode = runtime.greedy(
        ids,
        tools=selected,
        max_new=int(request.get("max_new") or 96),
        decode_mode=str(request.get("decode_mode") or "constrained"),
    )
    prefill_ms = (time.perf_counter() - t_prefill) * 1000 - float(decode.get("decode_ms") or 0)
    validated = validate_call(
        decode.get("text") or "",
        tools=selected,
        query=query,
        system_facts=str(request.get("system_facts") or ""),
    )
    gated = apply_confidence_gate(validated, decode.get("confidence"))
    return {
        "selected_tools": [str(t.get("name") or "") for t in selected],
        "text": decode.get("text"),
        "validated": gated,
        "confidence": gated.get("confidence_value"),
        "execution": gated.get("execution"),
        "prompt_tokens": len(ids),
        "output_tokens": decode.get("n_out"),
        "timings": {"prefill_ms": prefill_ms, "decode_ms": decode.get("decode_ms")},
        "decode": {"mode": str(request.get("decode_mode") or "constrained")},
        "error": gated.get("error"),
    }
