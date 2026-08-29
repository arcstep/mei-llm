"""Needle2-aligned complete()/run() runtime. Not used by Route-ID v1 runners."""

from __future__ import annotations

import time
from typing import Any, Callable

import mlx.core as mx

from byte_grammar import compile_byte_grammar, is_accept_bytes
from confidence_v2 import combine_confidence
from decode import greedy_byte_grammar, greedy_unconstrained
from kv_manager import KVManager
from prompt_v2 import encode_v2_segments, render_v2_request
from provenance_validator_v2 import validate_generated_call
from retrieval_rag_protocol import (
    K_DEFAULT,
    catalog_schema_fingerprint,
    encode_text_ids,
    render_tool_text,
)
from tool_index import ToolIndex

AUTO_EXECUTE = {"error_rate_95ucb_max": 0.01, "coverage_min": 0.5}
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


class RuntimeV2:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        catalog: list[dict[str, Any]] | None = None,
        index: ToolIndex | None = None,
        ordinary_cap: int = 256,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        tool_executor: Callable[[dict[str, Any]], str] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.catalog = list(catalog or [])
        self.index = index or ToolIndex()
        self.ordinary_cap = ordinary_cap
        self.confidence_threshold = confidence_threshold
        self.tool_executor = tool_executor
        self._index_schema_fp = ""

    def _embed_text(self, text: str):
        ids = encode_text_ids(self.tokenizer, text)
        arr = mx.array([ids], dtype=mx.int32)
        out = self.model(arr, return_contrastive=True, return_cells=True)
        vec = out.get("contrastive")
        if vec is None:
            return None
        row = vec[0].astype(mx.float32)
        norm = mx.sqrt(mx.sum(row * row)) + mx.array(1e-8, dtype=mx.float32)
        return row / norm

    def _ensure_index(self, tools: list[dict[str, Any]]) -> None:
        fp = catalog_schema_fingerprint(tools)
        if self._index_schema_fp == fp and self.index.records and len(self.index.records) == len(tools):
            return
        self.index.records.clear()
        for tool in tools:
            emb = self._embed_text(render_tool_text(tool))
            if emb is None:
                continue
            self.index.upsert(tool, emb)
        self._index_schema_fp = fp

    def _select(self, query: str, catalog: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        tools = list(catalog or self.catalog)
        if len(tools) <= K_DEFAULT:
            return tools
        self._ensure_index(tools)
        q_emb = self._embed_text(query)
        if q_emb is None:
            return tools[:K_DEFAULT]
        return self.index.select_tools(tools, q_emb, k=K_DEFAULT)

    def complete(
        self,
        query: str,
        *,
        catalog: list[dict[str, Any]] | None = None,
        system_facts: str | None = None,
        history: list[dict[str, str]] | None = None,
        prior_tool_results: list[str] | None = None,
        entities: list[dict[str, Any]] | None = None,
        permissions: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        oracle_tools: list[dict[str, Any]] | None = None,
        decode_mode: str = "constrained",
        max_new: int = 96,
    ) -> dict[str, Any]:
        selected = oracle_tools if oracle_tools is not None else self._select(query, catalog)
        rendered = render_v2_request(
            tools=selected,
            query=query,
            system_facts=system_facts,
            history=history,
            prior_tool_results=prior_tool_results,
        )
        if rendered["leaks"]:
            return {
                "ok": False,
                "error": "gold_leak:" + ",".join(rendered["leaks"]),
                "function_calls": [],
                "blocked": True,
            }
        enc = encode_v2_segments(self.tokenizer, rendered)
        t_prefill = time.perf_counter()
        kv = KVManager(ordinary_cap=self.ordinary_cap)
        prefill_out = kv.prefill_forward(
            self.model,
            enc["sink_ids"],
            enc["ordinary_ids"],
            reserve_tokens=max_new,
        )
        start_logits = prefill_out["logits"][:, -1, :]
        prefill_ms = (time.perf_counter() - t_prefill) * 1000
        t_decode = time.perf_counter()
        if decode_mode == "raw":
            decode_out = greedy_unconstrained(
                self.model,
                self.tokenizer,
                enc["prompt_ids"],
                max_new=max_new,
                kv=kv,
                start_logits=start_logits,
            )
        else:
            decode_out = greedy_byte_grammar(
                self.model,
                self.tokenizer,
                enc["prompt_ids"],
                selected,
                max_new=max_new,
                kv=kv,
                start_logits=start_logits,
            )
        decode_ms = (time.perf_counter() - t_decode) * 1000
        text = decode_out.get("text") or ""
        t_val = time.perf_counter()
        validated = validate_generated_call(
            text,
            tools=selected,
            query=query,
            system_facts=system_facts or "",
            prior_tool_results=prior_tool_results,
            entities=entities,
            permissions=permissions,
            state=state,
        )
        conf_head = 0.0
        if self.confidence_threshold > 0 and bool(getattr(self.model.cfg, "confidence_v2", False)):
            last_ids = kv.visible_ids[-min(16, len(kv.visible_ids)) :]
            last_pos = kv.visible_positions[-len(last_ids) :]
            cout = self.model(
                mx.array([last_ids], dtype=mx.int32),
                position_ids=mx.array(last_pos, dtype=mx.int32),
                return_confidence=True,
                return_cells=True,
            )
            logit = cout.get("confidence_v2_logit")
            if logit is not None:
                conf_head = float(logit[0].item())
        conf = combine_confidence(
            conf_head,
            decode_out.get("mean_token_logprob") or 0.0,
        )
        blocked = (not validated.get("refuse")) and conf < self.confidence_threshold
        if blocked:
            validated = {
                **validated,
                "ok": True,
                "function_calls": [],
                "refuse": True,
                "blocked_by_confidence": True,
                "error": None,
            }
        n_prompt = len(enc.get("prompt_ids") or [])
        n_out = len(decode_out.get("ids") or [])
        grammar_ms = float((decode_out.get("timings") or {}).get("grammar_ms") or 0.0)
        validate_ms = (time.perf_counter() - t_val) * 1000
        output_tok_s = (n_out / (decode_ms / 1000.0)) if decode_ms > 0 and n_out else 0.0
        return {
            "text": text,
            "selected_tools": [t.get("name") for t in selected],
            "prompt_leaks": rendered["leaks"],
            "decode": decode_out,
            "validated": validated,
            "confidence": conf,
            "execute": bool(validated.get("ok") and not validated.get("refuse") and not blocked),
            "kv_visible": len(kv.visible_ids),
            "kv_ordinary": len(kv.ordinary_ids),
            "sink_len": len(kv.sink_ids),
            "prompt_tokens": n_prompt,
            "output_tokens": n_out,
            "output_tok_s": output_tok_s,
            "timings": {
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "grammar_ms": grammar_ms,
                "validate_ms": validate_ms,
            },
        }

    def run(self, query: str, *, max_steps: int = 4, **kwargs) -> dict[str, Any]:
        history = list(kwargs.pop("history", None) or [])
        prior = list(kwargs.pop("prior_tool_results", None) or [])
        steps = []
        for i in range(max_steps):
            turn = self.complete(query, history=history, prior_tool_results=prior, **kwargs)
            steps.append(turn)
            if not turn.get("execute"):
                return {"ok": True, "steps": steps, "stopped": "empty_or_blocked", "function_calls": []}
            call = turn["validated"]["function_calls"][0]
            if self.tool_executor is None:
                return {"ok": True, "steps": steps, "stopped": "no_executor", "function_calls": [call]}
            result = self.tool_executor(call)
            prior.append(result)
            history.append({"role": "assistant", "content": turn.get("text") or ""})
        return {"ok": False, "steps": steps, "stopped": "max_steps", "function_calls": []}
