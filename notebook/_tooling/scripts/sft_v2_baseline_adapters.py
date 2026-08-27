#!/usr/bin/env python3
"""Unified sft-v2 baseline adapters.

Qwen 0.6B is not invented. MiniMind same-data training is not authorized here.
Retrieval chat output is labeled end-to-end tool selection, never embedding.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from repo_paths import EXTERNAL_BASELINES, ROOT, SFT_V2_BASELINE_MODELS, TASKS_ROOT, TASK_NEEDLE_ZH
from sft_v2_baseline_lib import REASON_CODES_16, STUDENT_SYSTEM, env_path, rel

MODELS = json.loads(SFT_V2_BASELINE_MODELS.read_text(encoding="utf-8"))["models"]
MODEL_BY_ID = {m["id"]: m for m in MODELS}


def model_spec(model_id: str) -> dict:
    if model_id not in MODEL_BY_ID:
        raise KeyError(model_id)
    return MODEL_BY_ID[model_id]


def ollama_available(host: str, tag: str, timeout: int = 8) -> tuple[bool, str]:
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"ollama_unreachable:{exc.__class__.__name__}"
    names = {str(m.get("name") or m.get("model") or "") for m in data.get("models") or []}
    if tag in names or any(tag.split(":")[0] in n for n in names if n.startswith(tag.split(":")[0])):
        if tag in names:
            return True, "ok"
        # allow prefix match like qwen3.5:0.8b-mlx
        for n in names:
            if n == tag or n.startswith(tag):
                return True, "ok"
        return False, f"tag_missing:{tag}"
    return False, f"tag_missing:{tag}"


def minimind_dir(size: str) -> Path | None:
    env_name = "MEI_MINIMIND_25M" if size.startswith("25") else "MEI_MINIMIND_45M"
    p = env_path(env_name)
    if p:
        return p
    cand = EXTERNAL_BASELINES / f"minimind-{size}"
    if cand.exists() and any(cand.iterdir()):
        return cand
    return None


def minimind_status(size: str) -> dict[str, Any]:
    d = minimind_dir(size)
    if d is None:
        return {
            "available": False,
            "status": "missing_checkpoint",
            "comparable": False,
            "note": "Place weights under notebook/archive/external-baselines/minimind-{size} or set MEI_MINIMIND_* . Same-data comparison requires identical retrieval/MW heads.",
        }
    has_head_marker = (d / "contrastive_head.json").is_file() or (d / "mw_head.json").is_file()
    return {
        "available": True,
        "status": "architecture_diagnostic_only" if not has_head_marker else "heads_present",
        "comparable": bool(has_head_marker),
        "path": rel(d),
        "note": "Without identical retrieval/MW heads this row is report-only and cannot promote.",
    }


class Adapter:
    def __init__(self, spec: dict):
        self.spec = spec
        self.id = spec["id"]
        self.role = spec["role"]
        self.params = spec.get("params")
        self.backend = spec.get("backend")
        self.status = "init"
        self.note = ""
        self._chat: Callable[..., tuple[str, dict]] | None = None

    def attach_chat(self, fn: Callable[..., tuple[str, dict]]) -> None:
        self._chat = fn
        self.status = "ready"

    def generate(
        self,
        prompt_text: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        system: str | None = None,
        tools: list | None = None,
        response_format: dict | str | None = None,
    ) -> tuple[str, dict]:
        if self._chat is None:
            return "", {"status": self.status, "error": "not_ready"}
        return self._chat(
            prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            tools=tools,
            response_format=response_format,
        )


def always_refuse(_prompt: str, **_k: Any) -> tuple[str, dict]:
    t0 = time.perf_counter()
    return "[]", {"wall_ms": (time.perf_counter() - t0) * 1000, "backend": "deterministic"}


def student_user_prompt(prompt_text: str) -> str:
    text = prompt_text or ""
    if text.startswith(STUDENT_SYSTEM):
        return text[len(STUDENT_SYSTEM) :].lstrip("\n")
    return text


def parse_e2e_tool_name(text: str, catalog: list[dict] | None = None) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    names = [str(t.get("name") or "") for t in (catalog or []) if t.get("name")]
    compact = raw.replace("`", "").replace("*", "")
    first = compact.split()[0].strip(".,;:。，[]()\"'")
    upper = first.upper()
    if upper in {"NONE", "NULL", "N/A", "无"} or compact.upper().startswith("NONE"):
        return "NONE"
    hits = [name for name in names if name and name == first]
    if len(hits) == 1:
        return hits[0]
    # strict: whole-token match only; substring collisions are not content-exact
    tokens = compact.replace(",", " ").replace("[", " ").replace("]", " ").split()
    token_hits = [name for name in names if name in tokens]
    if len(token_hits) == 1:
        return token_hits[0]
    return first if first in names else ""


def parse_reason_code(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    for code in REASON_CODES_16:
        if code in raw:
            return code
    return raw.split()[0].strip(".,;:。，[]()\"'")


def majority_mw_factory(majority: str) -> Callable[..., tuple[str, dict]]:
    def _fn(_prompt: str, **_k: Any) -> tuple[str, dict]:
        t0 = time.perf_counter()
        return majority, {"wall_ms": (time.perf_counter() - t0) * 1000, "backend": "deterministic"}

    return _fn


def qwen_ollama_factory(host: str, tag: str, timeout: int) -> Callable[..., tuple[str, dict]]:
    from run_eval_needle_qwen_v0 import ollama_chat

    def _fn(
        prompt_text: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        system: str | None = None,
        tools: list | None = None,
        response_format: dict | str | None = None,
    ) -> tuple[str, dict]:
        text, lat = ollama_chat(
            host,
            tag,
            system or STUDENT_SYSTEM,
            prompt_text,
            timeout,
            temperature,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
        )
        lat = dict(lat or {})
        lat["max_tokens"] = max_tokens
        lat["backend"] = "ollama"
        return text, lat

    return _fn


def mei58m_status() -> dict[str, Any]:
    spec = model_spec("mei-58m-base-300m-no-sft")
    ckpt = ROOT / spec["checkpoint"]
    if not ckpt.is_file():
        return {"available": False, "status": "missing_checkpoint", "path": spec["checkpoint"]}
    return {"available": True, "status": "checkpoint_present", "path": spec["checkpoint"], "bytes": ckpt.stat().st_size}


def load_mei58m():
    import sys

    sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
    from architecture import NeedleZh
    from checkpoint import load_params
    from config import NeedleZhConfig
    from tokenizer import ZhTokenizerV1

    spec = model_spec("mei-58m-base-300m-no-sft")
    ckpt = ROOT / spec["checkpoint"]
    tok = ZhTokenizerV1()
    cfg = NeedleZhConfig.from_target_v2()
    model = NeedleZh(cfg)
    report = load_params(
        model,
        ckpt,
        strict=False,
        allow_missing_prefixes=("contrastive", "mw", "confidence", "conf"),
        return_report=True,
    )
    import mlx.core as mx

    mx.eval(model.parameters())
    return model, tok, report


def adapter_matrix(*, host: str = "http://127.0.0.1:11434", timeout: int = 8) -> list[dict[str, Any]]:
    rows = []
    for ad in resolve_adapters(host=host, timeout=timeout):
        rows.append(
            {
                "id": ad.id,
                "role": ad.role,
                "params": ad.params,
                "backend": ad.backend,
                "status": ad.status,
                "note": ad.note,
                "retrieval_scoring": ad.spec.get("retrieval_scoring"),
                "same_data_sft_this_round": ad.spec.get("same_data_sft_this_round"),
            }
        )
    return rows


def resolve_adapters(*, host: str, timeout: int, include: set[str] | None = None) -> list[Adapter]:
    out: list[Adapter] = []
    for spec in MODELS:
        if include and spec["id"] not in include and spec["role"] not in (include or ()):
            continue
        ad = Adapter(spec)
        if spec["id"] == "always-refuse":
            ad.attach_chat(always_refuse)
        elif spec["backend"] == "ollama":
            ok, why = ollama_available(host, spec["ollama_tag"])
            if ok:
                ad.attach_chat(qwen_ollama_factory(host, spec["ollama_tag"], timeout))
            else:
                ad.status = why
                ad.note = "zero-shot skipped; same-data SFT not this round"
        elif spec["id"].startswith("minimind"):
            size = "25m" if "25" in spec["id"] else "45m"
            st = minimind_status(size)
            ad.status = st["status"]
            ad.note = st.get("note") or ""
        elif spec["id"] == "mei-58m-base-300m-no-sft":
            st = mei58m_status()
            ad.status = st["status"]
            ad.note = "no-SFT self baseline; generation uses product runtime if load succeeds"
        elif spec["id"] in {"lexical-retrieval", "majority-mw", "random-init-58m"}:
            ad.status = "ready"
        out.append(ad)
    return out


if __name__ == "__main__":
    print(json.dumps({"ok": True, "models": adapter_matrix()}, ensure_ascii=False, indent=2))
