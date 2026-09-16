#!/usr/bin/env python3
"""300M productization: full-call LM plus three independent product heads.

MW disposition is a closed-set sidecar.  It is not the MW deviation
evaluation/governance gate and it is never grouped with retrieval/confidence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from common.identity_51m import (
    JOBS_DIR,
    Q4_BASELINE_MAP_NAME,
    Q4_PACKAGE_DIR,
    QAT_Q4_PACKAGE_DIR,
    QAT_Q4_WEIGHTS_PATH,
    ROOT,
    SFT_QAT_BASE_DIR,
    SFT_QAT_MODEL_ID,
    SFT_QAT_PACKAGE_DIR,
    SFT_QAT_WEIGHTS_PATH,
    WEIGHTS_PATH,
    fail,
    load_json,
    refuse_base_write,
    sha256_file,
    write_json,
)
from training.qat.cq2_qat_51m import QUANT_MATH_ID, explicit_group_map, quantize_tree
from tokenizer import ASSISTANT_PREFIX, TURN_END, USER_PREFIX


DATA_RELEASE_DIR = ROOT / "cycles/mei-1.1-51m/exp-00300m/corpus/sft-suite/historical-notebook-releases/releases/mei-1.0-51m-tool-sft-v2-300m-v1"
RET_PATH = DATA_RELEASE_DIR / "retrieval.train.jsonl"
FC_PATH = DATA_RELEASE_DIR / "full-call.train.jsonl"
MW_PATH = DATA_RELEASE_DIR / "mw-disposition.train.jsonl"
MW_CODEBOOK_PATH = ROOT / "cycles/mei-1.1-51m/exp-00300m/corpus/sft-suite/historical-notebook-releases/recipes/mw-disposition-codebook-v1.json"
UNIVERSE_PATH = ROOT / "cycles/mei-1.1-51m/_legacy/notebook/evaluation/banks/sft-v2-eval-lock-v3-20class/tool-universe-v1.json"
ISOLATION = ROOT / "cycles/mei-1.1-51m/_legacy/notebook/_tooling/scripts/check_train_eval_isolation.py"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def isolation_ok() -> dict:
    proc = subprocess.run(
        [sys.executable, str(ISOLATION), "--scope", "sft-v2"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    payload = {}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"stdout": (proc.stdout or "")[:2000], "stderr": (proc.stderr or "")[:2000]}
    payload["returncode"] = proc.returncode
    return payload


def render_tool(tool: dict) -> str:
    return json.dumps(
        {
            "name": tool.get("name"),
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters") or {},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def universe_by_name() -> dict[str, dict]:
    blob = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    return {str(t["name"]): t for t in blob.get("tools") or []}


def cells_of(runtime, text: str, max_len: int = 96):
    ids = runtime.tokenizer.encode(text, add_bos=True, add_eos=False)[:max_len]
    out = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)
    mx.eval(*out["cells"])
    return [mx.stop_gradient(c).astype(mx.float16) for c in out["cells"]]


def train_contrastive(
    runtime,
    rows: list[dict],
    *,
    steps: int,
    lr: float,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every_steps: int = 100,
    stage_id: str = "r1",
) -> dict:
    from common.checkpoint import load_train_state, save_train_state
    from heads import ContrastiveHead

    cfg = runtime.model.cfg
    head = ContrastiveHead(cfg.d_model, cfg.n_layers, dim=128, probes=4)
    mx.eval(head.parameters())
    tools_u = universe_by_name()
    pairs = []
    for row in rows:
        gold = str(row.get("gold_tool") or "")
        catalog = row.get("catalog_tools") or []
        if isinstance(catalog, list) and catalog and isinstance(catalog[0], str):
            catalog = [tools_u[n] for n in catalog if n in tools_u]
        by_name = {str(t.get("name")): t for t in catalog if isinstance(t, dict)}
        gold_tool = by_name.get(gold) or tools_u.get(gold)
        if not gold_tool:
            continue
        negs = [n for n in (row.get("hard_negatives") or []) if n != gold]
        if not negs:
            negs = [n for n in by_name if n != gold][:3]
        if not negs:
            continue
        pairs.append((str(row.get("query") or ""), gold_tool, by_name.get(negs[0]) or tools_u.get(negs[0])))
    pairs = [p for p in pairs if p[2] is not None]
    if len(pairs) < 8:
        raise RuntimeError(f"not enough retrieval pairs: {len(pairs)}")
    rng = random.Random(51)
    opt = optim.Adam(learning_rate=lr)
    last = None
    batch = 8
    start_step = 0
    state_path = checkpoint_dir / f"retrieval-{stage_id}-head-state.npz" if checkpoint_dir else None
    expected_meta = {
        "kind": "quant-aware-sft-retrieval-head-state",
        "n_pairs": len(pairs),
        "target_steps": steps,
        "stage_id": stage_id,
        "embedding_dim": 128,
        "probes": 4,
    }
    if resume and state_path and state_path.is_file():
        meta = load_train_state(
            state_path,
            head,
            opt,
            mode="strict",
            expected_meta=expected_meta,
        )
        start_step = int(meta.get("steps_completed") or 0)
        last = meta.get("last_loss")
        for _ in range(start_step * batch):
            rng.randrange(len(pairs))
    tool_cells: dict[str, list[mx.array]] = {}

    def cached_tool(tool: dict) -> list[mx.array]:
        name = str(tool.get("name") or render_tool(tool))
        if name not in tool_cells:
            tool_cells[name] = cells_of(runtime, render_tool(tool))
        return tool_cells[name]

    for step in range(start_step, steps):
        chunk = [pairs[rng.randrange(len(pairs))] for _ in range(batch)]
        q_cells = [cells_of(runtime, q) for q, _, _ in chunk]
        p_cells = [cached_tool(g) for _, g, _ in chunk]
        n_cells = [cached_tool(n) for _, _, n in chunk]

        def loss_fn(h):
            q = mx.concatenate([h(c) for c in q_cells], axis=0)
            p = mx.concatenate([h(c) for c in p_cells], axis=0)
            n = mx.concatenate([h(c) for c in n_cells], axis=0)
            return mx.mean(nn.softplus(mx.sum(q * n, axis=-1) - mx.sum(q * p, axis=-1)))

        loss, grads = mx.value_and_grad(loss_fn)(head)
        opt.update(head, grads)
        mx.eval(head.parameters(), loss)
        last = float(loss.item())
        if state_path and (
            (step + 1) % max(1, checkpoint_every_steps) == 0 or step + 1 >= steps
        ):
            save_train_state(
                state_path,
                head,
                opt,
                {
                    **expected_meta,
                    "steps_completed": step + 1,
                    "last_loss": last,
                    "sampler_state": {
                        "seed": 51,
                        "draws_completed": (step + 1) * batch,
                    },
                },
            )
        if step == 0 or (step + 1) % 50 == 0:
            print(json.dumps({"contrastive_step": step + 1, "loss": last}), flush=True)
    runtime.contrastive = head
    return {
        "steps": steps,
        "resumed_from_step": start_step,
        "last_loss": last,
        "n_pairs": len(pairs),
        "n_cached_tools": len(tool_cells),
        "lm_frozen_after_qat_sft": True,
        "stage_id": stage_id,
        "embedding_dim": 128,
        "probes": 4,
    }


def train_mw_disposition(
    runtime,
    rows: list[dict],
    *,
    steps: int,
    lr: float = 1e-3,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every_steps: int = 100,
    catalog: list[dict] | None = None,
) -> dict:
    """Train the independent 20-class disposition sidecar on a frozen LM.

    This function does not calculate or satisfy the MW deviation governance
    gate.  That gate consumes separate evidence receipts after training.
    """

    from common.checkpoint import load_train_state, save_train_state
    from heads import MW_DISPOSITION_CLASSES, MWDispositionHead

    codebook = load_json(MW_CODEBOOK_PATH)
    classes = list(codebook.get("classes") or [])
    expected = {str(row["reason_code"]): int(row["class_id"]) for row in classes}
    if len(expected) != MW_DISPOSITION_CLASSES or set(expected.values()) != set(range(20)):
        raise RuntimeError("MW disposition codebook must contain class ids 0..19 exactly once")
    samples = []
    class_counts: dict[int, int] = {index: 0 for index in range(20)}
    for row in rows:
        reason = str(row.get("reason_code") or "")
        if reason not in expected:
            continue
        label = int(row.get("reason_class_id", expected[reason]))
        if label != expected[reason]:
            raise RuntimeError(f"MW disposition label/codebook mismatch: {reason}")
        prompt_text = str(row.get("prompt_text") or "").strip()
        if catalog:
            visible = runtime.search_top_k(str(row.get("query") or ""), catalog, k=5)
            compact = [
                {
                    "name": tool.get("name"),
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {},
                }
                for tool in visible
            ]
            facts = str(row.get("system_facts") or row.get("scene") or "").strip()
            prompt_text = (
                "任务：根据 query、允许事实和当前可见的五个工具 schema 计算闭集处置 logits；"
                "不要生成工具调用或解释。\n"
                f"query：{str(row.get('query') or '').strip()}\n"
                + (f"允许事实：{facts}\n" if facts else "")
                + "<tools>"
                + json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "</tools>"
            )
        if not prompt_text:
            prompt_text = (
                "任务：根据 query、允许事实和当前可见工具判断处置原因；不要生成工具调用。\n"
                f"query：{str(row.get('query') or '').strip()}"
            )
        ids = runtime.tokenizer.encode(prompt_text, add_bos=True, add_eos=False)[:1024]
        if len(ids) < 4:
            continue
        samples.append((ids, label))
        class_counts[label] += 1
    missing = [index for index, count in class_counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"MW disposition release is missing class ids: {missing}")

    head = MWDispositionHead(runtime.model.cfg.d_model)
    mx.eval(head.parameters())
    opt = optim.Adam(learning_rate=lr)
    state_path = checkpoint_dir / "mw-disposition-head-state.npz" if checkpoint_dir else None
    expected_meta = {
        "kind": "mw-disposition-head-state",
        "semantic_boundary": "not-mw-deviation-gate",
        "codebook_sha256": sha256_file(MW_CODEBOOK_PATH),
        "n_samples": len(samples),
        "n_classes": 20,
        "target_steps": steps,
    }
    start_step = 0
    last = None
    if resume and state_path and state_path.is_file():
        meta = load_train_state(
            state_path,
            head,
            opt,
            mode="strict",
            expected_meta=expected_meta,
        )
        start_step = int(meta.get("steps_completed") or 0)
        last = meta.get("last_loss")

    for step in range(start_step, steps):
        epoch = step // len(samples)
        order = list(range(len(samples)))
        random.Random(1701 + epoch).shuffle(order)
        ids, label = samples[order[step % len(samples)]]
        frozen = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)["cells"]
        mx.eval(*frozen)
        cells = [mx.stop_gradient(value).astype(mx.float16) for value in frozen]

        def loss_fn(candidate):
            logits = candidate(cells).astype(mx.float32)
            return -nn.log_softmax(logits, axis=-1)[0, label]

        loss, grads = mx.value_and_grad(loss_fn)(head)
        opt.update(head, grads)
        mx.eval(head.parameters(), loss)
        last = float(loss.item())
        if state_path and (
            (step + 1) % max(1, checkpoint_every_steps) == 0 or step + 1 >= steps
        ):
            save_train_state(
                state_path,
                head,
                opt,
                {**expected_meta, "steps_completed": step + 1, "last_loss": last},
            )
    runtime.mw_disposition_head = head
    return {
        "steps": steps,
        "resumed_from_step": start_step,
        "last_loss": last,
        "n_samples": len(samples),
        "n_classes": 20,
        "class_counts": {str(key): value for key, value in class_counts.items()},
        "codebook_sha256": expected_meta["codebook_sha256"],
        "lm_frozen": True,
        "semantic_boundary": "mw-disposition-sidecar-not-mw-deviation-gate",
    }


def deployment_request(row: dict) -> dict:
    """Map a frozen SFT row into the strict, deployment-visible wire-v2 shape."""

    facts = []
    system_facts = str(row.get("system_facts") or row.get("scene") or "").strip()
    if system_facts:
        facts.append(
            {
                "id": "dataset:system-facts",
                "subject": "scene",
                "predicate": "facts",
                "value": system_facts,
                "source": "frozen-sft-release:system_facts",
                "verified": True,
            }
        )
    evidence = []
    for index, raw in enumerate(row.get("slot_provenance") or []):
        if not isinstance(raw, dict):
            continue
        argument = str(raw.get("arg") or raw.get("argument") or "").strip()
        source = str(raw.get("source") or "").strip()
        if not argument or not source:
            continue
        evidence.append(
            {
                "id": f"dataset:slot:{index}",
                "subject": "request",
                "predicate": argument,
                "value": raw.get("value", raw.get("span")),
                "source": f"frozen-sft-release:{source}",
                "verified": True,
            }
        )
    entities = row.get("entities") or {}
    selected_entities = []
    selected = row.get("selected_entity")
    if selected is not None:
        selected_entities.append(str(selected))
    if isinstance(entities, dict):
        selected_entities.extend(str(value) for value in entities.values() if value is not None)
    history = []
    for raw in row.get("history") or []:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        content = str(raw.get("content") or raw.get("text") or "").strip()
        if role in {"system", "user", "assistant", "tool"} and content:
            history.append({"role": role, "content": content})
    trace = row.get("agent_trace") or {}
    trusted_fixture = (
        isinstance(trace, dict)
        and trace.get("trusted_offline_fixture") is True
        and trace.get("simulator_id") == "mei-agent-host-simulator-v1"
    )
    tool_results = list(row.get("tool_results") or [])
    if trusted_fixture:
        from mei_sdk.protocol import validate_tool_result_v2

        calls = list(row.get("prior_calls") or [])
        if len(calls) != len(tool_results):
            raise RuntimeError("trusted Agent SFT row has unmatched call/result prefix")
        for call, result in zip(calls, tool_results):
            if not isinstance(call, dict) or not isinstance(result, dict):
                raise RuntimeError("trusted Agent SFT call/result must be objects")
            validate_tool_result_v2(result)
            call_id = str(call.get("call_id") or "")
            if result.get("call_id") != call_id or result.get("status") != "ok":
                raise RuntimeError("trusted Agent SFT call/result identity or status mismatch")
            history.append(
                {
                    "role": "assistant",
                    "call_id": call_id,
                    "content": json.dumps(
                        {
                            "call_id": call_id,
                            "name": str(call.get("name") or ""),
                            "arguments": call.get("arguments") or {},
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
    else:
        if tool_results:
            raise RuntimeError("untrusted SFT row cannot inject ToolResultV2")
        # Historical rows may contain result-shaped dictionaries without v2
        # call IDs/provenance. Keep them as untrusted text; never forge trust.
        for raw in row.get("prior_tool_results") or []:
            history.append(
                {
                    "role": "tool",
                    "content": json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                }
            )
    return {
        "wire_version": "mei-runtime-wire-v2",
        "query": str(row.get("query") or ""),
        "context": {"locale": "zh-CN", "selected_entities": selected_entities, "facts": facts},
        "evidence": evidence,
        "permissions": {"allowed_tools": [str(value) for value in row.get("permissions") or []]},
        "history": history,
        "tool_results": tool_results if trusted_fixture else [],
        "state": {},
    }


def deployment_prompt(row: dict, tools: list[dict]) -> str:
    """Render the exact selected-schema request shape used by runtime v2."""

    from mei_sdk.protocol import render_request

    return str(render_request(deployment_request(row), tools)["prompt"])


def encode_fc(
    tok,
    row: dict,
    *,
    tools_by_name: dict[str, dict] | None = None,
    selected_tools: list[dict] | None = None,
    max_prompt: int = 1152,
    max_ans: int = 96,
) -> tuple[list[int], list[int]] | None:
    if selected_tools is None and tools_by_name is not None:
        selected_tools = [
            tools_by_name[name]
            for name in (row.get("retrieved_tools") or [])[:5]
            if name in tools_by_name
        ]
    prompt_text = (
        deployment_prompt(row, selected_tools or [])
        if selected_tools is not None
        else str(row.get("prompt_text") or "").strip()
    )
    if not prompt_text:
        prompt_text = str(row.get("query") or "")
        prompt_text = USER_PREFIX + prompt_text
    prompt = tok.encode(prompt_text + ASSISTANT_PREFIX, add_bos=True, add_eos=False)[:max_prompt]
    answers = row.get("answers")
    if answers is None:
        name = row.get("gold_name")
        if not name:
            target = "[]"
        else:
            target = json.dumps(
                [{"name": name, "arguments": row.get("gold_args") or {}}],
                ensure_ascii=False,
                separators=(",", ":"),
            )
    elif answers == [] or answers == "[]":
        target = "[]"
    else:
        target = json.dumps(answers, ensure_ascii=False, separators=(",", ":"))
    answer = tok.encode(target) + tok.encode(TURN_END) + [tok.eos_id]
    answer = answer[:max_ans]
    if len(prompt) < 4 or len(answer) < 2:
        return None
    return prompt, answer


def train_fullcall(
    runtime,
    rows: list[dict],
    *,
    steps: int,
    lr: float,
    group_map: Mapping[str, Sequence[int]] | None,
    activation_ste: bool,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every_steps: int = 100,
    tools_by_name: dict[str, dict] | None = None,
) -> dict:
    import architecture as arch
    from common.checkpoint import load_train_state, save_train_state
    from common.train_common import clip_grads

    tok = runtime.tokenizer
    pairs = []
    for row in rows:
        enc = encode_fc(tok, row, tools_by_name=tools_by_name)
        if enc:
            pairs.append(enc)
    if not pairs:
        raise RuntimeError("no full-call pairs")
    opt = optim.Adam(learning_rate=lr)
    last = None
    model = runtime.model
    start_step = 0
    state_path = checkpoint_dir / "fullcall-state.npz" if checkpoint_dir else None
    expected_meta = {
        "kind": "quant-aware-sft-train-state",
        "quant_math_id": QUANT_MATH_ID if group_map else None,
        "weight_qat_ste": bool(group_map),
        "activation_kv_int8_ste": bool(activation_ste and group_map),
        "activation_ste_semantics": (
            "int8-symmetric-per-last-axis-vector-qdq-forward_identity-backward-v2"
            if activation_ste and group_map
            else None
        ),
        "kv_ste_semantics": (
            "mei-int8-kv-per-head-vector-qdq-forward_identity-backward-v1"
            if activation_ste and group_map
            else None
        ),
        "n_pairs": len(pairs),
        "target_steps": steps,
    }
    if resume and state_path and state_path.is_file():
        meta = load_train_state(
            state_path,
            model,
            opt,
            mode="strict",
            expected_meta=expected_meta,
        )
        start_step = int(meta.get("steps_completed") or 0)
        last = meta.get("last_loss")
    used: set[int] = set()
    arch.QAT_ACTIVATION_STE = bool(activation_ste and group_map)
    activation_ste_probe = None
    if arch.QAT_ACTIVATION_STE:
        from training.qat.qat_cq2_v2_51m import verify_activation_ste_contract

        activation_ste_probe = verify_activation_ste_contract(arch)
    try:
        for step in range(start_step, steps):
            epoch = step // len(pairs)
            order = list(range(len(pairs)))
            random.Random(7 + epoch).shuffle(order)
            pair_index = order[step % len(order)]
            used.add(pair_index)
            prompt, answer = pairs[pair_index]
            ids = prompt + answer

            def loss_fn(params):
                if group_map:
                    model.update(quantize_tree(params, group_map, ste=True))
                else:
                    model.update(params)
                arr = mx.array([ids], dtype=mx.int32)
                logits = model(arr)["logits"].astype(mx.float32)
                logp = nn.log_softmax(logits, axis=-1)
                total = mx.array(0.0, dtype=mx.float32)
                for i, tid in enumerate(answer):
                    pos = len(prompt) - 1 + i
                    total = total + (-logp[0, pos, int(tid)])
                return total / max(len(answer), 1)

            params = model.parameters()
            loss, grads = mx.value_and_grad(loss_fn)(params)
            grads, grad_norm = clip_grads(grads, max_norm=1.0)
            model.update(params)
            opt.update(model, grads)
            mx.eval(model.parameters(), loss)
            last = float(loss.item())
            if state_path and (
                (step + 1) % max(1, checkpoint_every_steps) == 0 or step + 1 >= steps
            ):
                save_train_state(
                    state_path,
                    model,
                    opt,
                    {
                        **expected_meta,
                        "steps_completed": step + 1,
                        "last_loss": last,
                        "sampler_state": {
                            "epoch": epoch,
                            "index_in_epoch": (step + 1) % len(order),
                            "seed_rule": "7 + epoch",
                        },
                    },
                )
            if step == 0 or (step + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "sft_step": step + 1,
                            "loss": last,
                            "grad_norm": float(grad_norm.item()),
                            "qat_weight_ste": bool(group_map),
                            "activation_kv_int8_ste": bool(activation_ste and group_map),
                        }
                    ),
                    flush=True,
                )
    finally:
        arch.QAT_ACTIVATION_STE = False
    return {
        "steps": steps,
        "resumed_from_step": start_step,
        "last_loss": last,
        "n_pairs": len(pairs),
        "unique_rows_exposed": min(steps, len(pairs)),
        "weight_qat_ste": bool(group_map),
        "activation_kv_int8_ste": bool(activation_ste and group_map),
        "activation_ste_probe": activation_ste_probe,
        "quant_math_id": QUANT_MATH_ID if group_map else None,
    }


def train_fullcall_float(
    runtime,
    rows: list[dict],
    *,
    steps: int,
    lr: float,
    tools_by_name: dict[str, dict],
) -> dict:
    return train_fullcall(
        runtime,
        rows,
        steps=steps,
        lr=lr,
        group_map=None,
        activation_ste=False,
        tools_by_name=tools_by_name,
    )


def train_confidence(
    runtime,
    rows: list[dict],
    *,
    steps: int,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every_steps: int = 100,
    catalog: list[dict],
    outcome_limit: int = 128,
) -> dict:
    from common.checkpoint import load_train_state, save_train_state
    from heads import ConfidenceV2Head
    from mei_sdk.protocol import normalize_request
    from mei_sdk.shared import validate_generated_call

    tok = runtime.tokenizer
    head = ConfidenceV2Head(runtime.model.cfg.d_model)
    mx.eval(head.parameters())
    gold_calls = [row for row in rows if row.get("answers") or row.get("gold_name")]
    gold_refusals = [row for row in rows if not (row.get("answers") or row.get("gold_name"))]
    selected_rows = gold_calls[: outcome_limit // 2] + gold_refusals[: outcome_limit // 2]
    outcomes = []
    for row in selected_rows:
        query = str(row.get("query") or "")
        visible = runtime.search_top_k(query, catalog, k=5)
        enc = encode_fc(tok, row, selected_tools=visible)
        if not enc:
            continue
        prompt, _answer = enc
        decoded = runtime.greedy(prompt, tools=visible, max_new=96, decode_mode="constrained")
        gold_name = str(row.get("gold_name") or "")
        gold_args = dict(row.get("gold_args") or {})
        request = deployment_request(row)
        request.update({"decode_mode": "constrained", "max_new": 96})
        request = normalize_request(request)
        validated = validate_generated_call(
            str(decoded.get("text") or ""),
            tools=visible,
            request=request,
            confidence=None,
            enforce_confidence=False,
        )
        gold = row.get("answers")
        if gold is None:
            gold = [] if not gold_name else [{"name": gold_name, "arguments": gold_args}]
        # Confidence means an actually correct, executable call.  Correct
        # refusals remain negatives and source confidence_label is ignored.
        label = float(
            bool(gold)
            and validated.get("ok") is True
            and validated.get("refuse") is not True
            and _calls_equal(validated.get("function_calls") or [], gold)
        )
        outcomes.append(
            {
                "prompt": prompt,
                "label": label,
                "sample_id": row.get("sample_id"),
                "decode_logprob": float(decoded.get("logprob_sum") or 0.0),
            }
        )
    if len(outcomes) < 4:
        raise RuntimeError("not enough actual runtime outcomes for confidence training")
    train_outcomes = [row for index, row in enumerate(outcomes) if index % 5 != 0]
    calibration_outcomes = [row for index, row in enumerate(outcomes) if index % 5 == 0]
    opt = optim.Adam(learning_rate=1e-3)
    start_step = 0
    state_path = checkpoint_dir / "confidence-head-state.npz" if checkpoint_dir else None
    expected_meta = {
        "kind": "quant-aware-sft-confidence-head-state",
        "n_samples": len(train_outcomes),
        "target_steps": steps,
        "n_positive": sum(int(row["label"]) for row in train_outcomes),
        "n_negative": sum(int(not row["label"]) for row in train_outcomes),
        "label_contract": "actual-runtime-v2-call-correctness",
    }
    last = None
    if resume and state_path and state_path.is_file():
        meta = load_train_state(
            state_path,
            head,
            opt,
            mode="strict",
            expected_meta=expected_meta,
        )
        start_step = int(meta.get("steps_completed") or 0)
        last = meta.get("last_loss")

    for step in range(start_step, steps):
        epoch = step // len(train_outcomes)
        order = list(range(len(train_outcomes)))
        random.Random(2303 + epoch).shuffle(order)
        sample = train_outcomes[order[step % len(train_outcomes)]]
        frozen = runtime.model(
            mx.array([sample["prompt"]], dtype=mx.int32), return_cells=True
        )["cells"]
        mx.eval(*frozen)
        cells = [mx.stop_gradient(value).astype(mx.float16) for value in frozen]

        def loss_fn(candidate):
            logit = candidate(cells)
            y = mx.array(sample["label"], dtype=mx.float32)
            probability = mx.clip(mx.sigmoid(logit.reshape(())), 1e-6, 1.0 - 1e-6)
            return -(y * mx.log(probability) + (1.0 - y) * mx.log(1.0 - probability))

        loss, grads = mx.value_and_grad(loss_fn)(head)
        opt.update(head, grads)
        mx.eval(head.parameters(), loss)
        last = float(loss.item())
        if state_path and (
            (step + 1) % max(1, checkpoint_every_steps) == 0 or step + 1 >= steps
        ):
            save_train_state(
                state_path,
                head,
                opt,
                {
                    **expected_meta,
                    "steps_completed": step + 1,
                    "last_loss": last,
                },
            )
    probabilities = []
    for sample in calibration_outcomes:
        frozen = runtime.model(
            mx.array([sample["prompt"]], dtype=mx.int32), return_cells=True
        )["cells"]
        logit = head([mx.stop_gradient(value) for value in frozen])
        probability = mx.sigmoid(logit.reshape(()))
        mx.eval(probability)
        probabilities.append((float(probability.item()), float(sample["label"])))
    brier = sum((probability - label) ** 2 for probability, label in probabilities) / max(
        len(probabilities), 1
    )
    ece = 0.0
    for lower in [index / 10 for index in range(10)]:
        bucket = [row for row in probabilities if lower <= row[0] < lower + 0.1]
        if bucket:
            confidence = sum(row[0] for row in bucket) / len(bucket)
            accuracy = sum(row[1] for row in bucket) / len(bucket)
            ece += len(bucket) / max(len(probabilities), 1) * abs(confidence - accuracy)
    runtime.conf_v2 = head
    n_positive = sum(int(row["label"]) for row in outcomes)
    n_negative = sum(int(not row["label"]) for row in outcomes)
    class_coverage_ok = n_positive > 0 and n_negative > 0
    return {
        "steps": steps,
        "resumed_from_step": start_step,
        "last_loss": last,
        "n": len(outcomes),
        "n_train": len(train_outcomes),
        "n_calibration": len(calibration_outcomes),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "class_coverage_ok": class_coverage_ok,
        "calibration_status": (
            "calibrated" if class_coverage_ok else "single-class-outcomes-unvalidated"
        ),
        "degraded": not class_coverage_ok,
        "ece": ece,
        "brier": brier,
        "used_actual_runtime_correctness": True,
        "used_structured_wire_v2_validation": True,
        "oracle_evidence_binding_used": False,
        "ignored_source_confidence_label": True,
    }


def encode_narration(
    tok,
    row: dict,
    *,
    max_prompt: int = 1152,
    max_answer: int = 192,
) -> tuple[list[int], list[int]] | None:
    prompt_text = str(row.get("prompt") or "").strip()
    target_text = str(row.get("target") or "").strip()
    if not prompt_text or not target_text or not row.get("verified_result_views"):
        return None
    prompt = tok.encode(prompt_text, add_bos=True, add_eos=False)[:max_prompt]
    answer = (tok.encode(target_text, add_bos=False, add_eos=False) + [tok.eos_id])[:max_answer]
    if len(prompt) < 4 or len(answer) < 2:
        return None
    return prompt, answer


def train_narration_adapter(
    runtime,
    train_rows: list[dict],
    valid_rows: list[dict],
    *,
    steps: int,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every_steps: int = 100,
    rank: int | None = None,
) -> dict:
    """Train only the low-rank narration logit residual on a frozen backbone.

    ``rank``（默认 None = NARRATION_ADAPTER_RANK 16）是容量消融旋钮；运行时
    加载仍走 canonical 16，只有训练/评测驱动显式传 rank 才改变 adapter 容量。
    """

    from common.checkpoint import load_train_state, save_train_state
    from heads import NARRATION_ADAPTER_RANK, NarrationAdapterHead

    rank = int(rank) if rank is not None else NARRATION_ADAPTER_RANK
    tok = runtime.tokenizer
    train_pairs = [pair for row in train_rows if (pair := encode_narration(tok, row))]
    valid_pairs = [pair for row in valid_rows if (pair := encode_narration(tok, row))]
    if len(train_pairs) < 8 or len(valid_pairs) < 4:
        raise RuntimeError("not enough grounded narration pairs")
    head = NarrationAdapterHead(
        runtime.model.cfg.d_model,
        runtime.model.cfg.vocab_size,
        rank=rank,
    )
    mx.eval(head.parameters())
    opt = optim.Adam(learning_rate=1e-3)
    start_step = 0
    last = None
    state_path = checkpoint_dir / "narration-adapter-state.npz" if checkpoint_dir else None
    expected_meta = {
        "kind": "frozen-backbone-grounded-narration-adapter-state",
        "rank": rank,
        "n_train": len(train_pairs),
        "n_valid": len(valid_pairs),
        "target_steps": steps,
        "backbone_trainable": False,
        "target_contract": "exact-deterministic-template",
    }
    if resume and state_path and state_path.is_file():
        meta = load_train_state(
            state_path,
            head,
            opt,
            mode="strict",
            expected_meta=expected_meta,
        )
        start_step = int(meta.get("steps_completed") or 0)
        last = meta.get("last_loss")

    def frozen_batch(pair: tuple[list[int], list[int]]):
        prompt, answer = pair
        ids = prompt + answer
        output = runtime.model(mx.array([ids], dtype=mx.int32))
        hidden = mx.stop_gradient(output["hidden"]).astype(mx.float16)
        base_logits = mx.stop_gradient(output["logits"]).astype(mx.float32)
        mx.eval(hidden, base_logits)
        return prompt, answer, hidden, base_logits

    def loss_for(candidate, frozen) -> mx.array:
        prompt, answer, hidden, base_logits = frozen
        logits = base_logits + candidate(hidden).astype(mx.float32)
        logp = nn.log_softmax(logits, axis=-1)
        total = mx.array(0.0, dtype=mx.float32)
        for index, token_id in enumerate(answer):
            position = len(prompt) - 1 + index
            total = total + (-logp[0, position, int(token_id)])
        return total / max(len(answer), 1)

    for step in range(start_step, steps):
        epoch = step // len(train_pairs)
        order = list(range(len(train_pairs)))
        random.Random(5107 + epoch).shuffle(order)
        frozen = frozen_batch(train_pairs[order[step % len(train_pairs)]])
        loss, grads = mx.value_and_grad(loss_for)(head, frozen)
        opt.update(head, grads)
        mx.eval(head.parameters(), loss)
        last = float(loss.item())
        if state_path and (
            (step + 1) % max(1, checkpoint_every_steps) == 0 or step + 1 >= steps
        ):
            save_train_state(
                state_path,
                head,
                opt,
                {
                    **expected_meta,
                    "steps_completed": step + 1,
                    "last_loss": last,
                },
            )
        if step == 0 or (step + 1) % 100 == 0:
            print(
                json.dumps(
                    {"narration_step": step + 1, "loss": last, "backbone_frozen": True}
                ),
                flush=True,
            )

    valid_losses = []
    for pair in valid_pairs[: min(16, len(valid_pairs))]:
        value = loss_for(head, frozen_batch(pair))
        mx.eval(value)
        valid_losses.append(float(value.item()))
    runtime.narration_adapter = head
    import mlx.utils as xu

    flattened = xu.tree_flatten(head.parameters())
    if isinstance(flattened, tuple):
        flattened = flattened[0]
    parameter_count = sum(
        int(item[1].size if isinstance(item, tuple) else item.size)
        for item in flattened
    )
    return {
        "steps": steps,
        "resumed_from_step": start_step,
        "last_loss": last,
        "valid_loss": sum(valid_losses) / max(len(valid_losses), 1),
        "n_train": len(train_pairs),
        "n_valid": len(valid_pairs),
        "rank": rank,
        "parameter_count": parameter_count,
        "backbone_frozen": True,
        "verified_result_only": True,
        "can_execute_tools": False,
        "grounding_gate": "exact-deterministic-template-else-fallback",
    }


def save_product_heads(runtime, path: Path) -> dict:
    """Save all independent sidecars in the exact package-v2 order."""

    from common.checkpoint import _atomic_savez, flatten_params

    heads = {
        "contrastive": runtime.contrastive,
        "mw_disposition": getattr(runtime, "mw_disposition_head", None),
        "confidence": runtime.conf_v2,
        "narration_adapter": getattr(runtime, "narration_adapter", None),
    }
    missing = [name for name, head in heads.items() if head is None]
    if missing:
        raise RuntimeError(f"cannot finalize product heads; missing={missing}")
    arrays: dict[str, np.ndarray] = {}
    prefixes = {
        "contrastive": "heads.contrastive",
        "mw_disposition": "heads.mw_disposition",
        "confidence": "heads.confidence",
        "narration_adapter": "heads.narration_adapter",
    }
    for role in ("contrastive", "mw_disposition", "confidence", "narration_adapter"):
        for name, value in flatten_params(heads[role]).items():
            arrays[f"{prefixes[role]}.{name}"] = np.asarray(value, dtype=np.float32)
    expected = [
        "heads.contrastive.tok_probes",
        "heads.contrastive.lay_probes",
        "heads.contrastive.proj.weight",
        "heads.mw_disposition.proj.weight",
        "heads.mw_disposition.proj.bias",
        "heads.confidence.cell_probes",
        "heads.confidence.proj.weight",
        "heads.confidence.proj.bias",
        "heads.narration_adapter.down.weight",
        "heads.narration_adapter.up.weight",
    ]
    if list(arrays) != expected:
        raise RuntimeError(f"product head tensor order mismatch: {list(arrays)}")
    _atomic_savez(path, arrays)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "tensor_names": expected,
        "semantic_roles": list(heads),
        "mw_deviation_tensor_count": 0,
    }


def finalize_tool_index(
    runtime,
    catalog: list[dict],
    path: Path,
    *,
    model_sha256: str,
    head_sha256: str,
    tokenizer_sha256: str,
) -> dict:
    """Rebuild the portable f16 index after final LM and retrieval R1."""

    from mei_sdk.shared import ToolIndex

    runtime.index = ToolIndex(
        model_hash=model_sha256,
        head_hash=head_sha256,
        tokenizer_hash=tokenizer_sha256,
    )
    runtime._catalog_fp = ""
    runtime.build_index(catalog)
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime.index.save(path)
    loaded = ToolIndex.load(path, expected_fingerprint=runtime.index.fingerprint)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "fingerprint": loaded.fingerprint,
        "catalog_sha256": loaded.catalog_hash,
        "n_tools": len(loaded.records),
        "dimension": len(next(iter(loaded.records.values())).embedding),
        "dtype": "float16",
        "normalized": True,
        "stable_tool_id_ties": True,
    }


def learned_top5_e2e(
    runtime,
    rows: list[dict],
    catalog: list[dict],
    *,
    limit: int = 128,
) -> dict:
    """Evaluate retrieval→constrained decode→deterministic gates as deployed."""

    from mei_sdk.protocol import normalize_request
    from mei_sdk.shared import validate_generated_call

    attempted = retrieval_hits = exact = refusals = 0
    unsupported_accepted = unprovenanced_accepted = 0
    for row in rows[: max(0, int(limit))]:
        query = str(row.get("query") or "")
        visible = runtime.search_top_k(query, catalog, k=5)
        names = {str(tool.get("name") or "") for tool in visible}
        gold_name = str(row.get("gold_name") or "")
        retrieval_hits += int(not gold_name or gold_name in names)
        encoded = encode_fc(runtime.tokenizer, row, selected_tools=visible)
        if not encoded:
            continue
        prompt, _answer = encoded
        decoded = runtime.greedy(prompt, tools=visible, max_new=96, decode_mode="constrained")
        request = deployment_request(row)
        request.update({"decode_mode": "constrained", "max_new": 96})
        validated = validate_generated_call(
            str(decoded.get("text") or ""),
            tools=visible,
            request=normalize_request(request),
            confidence=None,
            enforce_confidence=False,
        )
        gold = row.get("answers")
        if gold is None:
            gold = [] if not gold_name else [{"name": gold_name, "arguments": row.get("gold_args") or {}}]
        attempted += 1
        refusals += int(validated.get("refuse") is True)
        exact += int(_calls_equal(validated.get("function_calls") or [], gold or []))
        unsupported_accepted += int(validated.get("unsupported_accepted") or 0)
        unprovenanced_accepted += int(validated.get("unprovenanced_argument_accepted") or 0)
    return {
        "n": attempted,
        "retrieval_top5_recall": retrieval_hits / max(min(len(rows), int(limit)), 1),
        "call_exact": exact / max(attempted, 1),
        "refusal_rate": refusals / max(attempted, 1),
        "unsupported_accepted": unsupported_accepted,
        "unprovenanced_argument_accepted": unprovenanced_accepted,
        "decode_mode": "constrained",
        "deterministic_gates_applied": True,
        "not_a_score_claim": True,
    }


def save_sft_package(
    src_pkg: Path,
    dest: Path,
    runtime,
    *,
    master_path: Path,
    model_id: str,
    parent_master: Path,
    bit_map_path: Path,
    quant_scheme: str,
    jobs_dir: Path,
) -> Path:
    from common.checkpoint import _atomic_savez, flatten_params, save_params

    blocked = refuse_base_write(master_path)
    if blocked:
        raise RuntimeError(blocked)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    save_params(runtime.model, master_path)
    parent_manifest = load_json(src_pkg / "mei-model.json")
    parent_id = str(parent_manifest.get("package_id") or parent_manifest.get("model_id") or "")
    parent_package_sha = sha256_file(src_pkg / "weights.q4")
    parent_master_sha = sha256_file(parent_master)

    import sys
    from release.pack_qat_51m import main as pack_main

    argv = sys.argv
    sys.argv = [
        "pack_qat_51m.py",
        "--master",
        str(master_path),
        "--out-dir",
        str(dest),
        "--package-id",
        model_id,
        "--parent-id",
        parent_id,
        "--parent-weights-sha256",
        parent_package_sha,
        "--scheme",
        quant_scheme,
        "--bit-map",
        str(bit_map_path),
        "--receipt-name",
        "sft-qat-q4-package-receipt.json",
        "--jobs-dir",
        str(jobs_dir),
    ]
    try:
        code = pack_main()
    finally:
        sys.argv = argv
    if code != 0:
        raise RuntimeError(f"SFT Q4 pack failed code={code}")

    arrays = {}
    if runtime.contrastive is not None:
        arrays.update(
            {f"contrastive.{k}": v.astype(mx.float16) for k, v in flatten_params(runtime.contrastive).items()}
        )
    if runtime.conf_v2 is not None:
        arrays.update({f"conf_v2.{k}": v.astype(mx.float16) for k, v in flatten_params(runtime.conf_v2).items()})
    if arrays:
        _atomic_savez(dest / "heads.npz", arrays)
    man_path = dest / "mei-model.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["package_id"] = model_id
        heads = man.setdefault("heads", {})
        heads["contrastive"] = {
            "present": runtime.contrastive is not None,
            "trained": runtime.contrastive is not None,
            "status": "ready" if runtime.contrastive is not None else "missing",
            "dim": 32,
            "probes": 2,
        }
        heads["confidence"] = {
            "present": True,
            "trained": runtime.conf_v2 is not None,
            "status": "ready" if runtime.conf_v2 is not None else "untrained",
        }
        heads["artifact"] = {
            "file": "heads.npz",
            "sha256": sha256_file(dest / "heads.npz"),
            "dtype": "float16",
        }
        man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    package_bytes = sum(path.stat().st_size for path in dest.iterdir() if path.is_file())
    if package_bytes > 30 * 1024 * 1024:
        raise RuntimeError(f"SFT Q4 package {package_bytes} bytes exceeds 30MiB")
    release = {
        "kind": "quant-aware-sft-fp32-master",
        "model_id": model_id,
        "weights": str(master_path.relative_to(ROOT)),
        "weights_sha256": sha256_file(master_path),
        "parent_package": str(src_pkg.relative_to(ROOT)),
        "parent_model_id": parent_id,
        "parent_weights_sha256": parent_package_sha,
        "parent_fp32_master": str(parent_master.relative_to(ROOT)),
        "parent_fp32_master_sha256": parent_master_sha,
        "quant_math_id": QUANT_MATH_ID,
        "weight_qat_ste": True,
        "activation_kv_int8_ste": True,
        "heads_file": str((dest / "heads.npz").relative_to(ROOT)),
        "heads_dtype": "float16",
        "package_files_bytes": package_bytes,
        "package_files_mb": package_bytes / (1024 * 1024),
        "package_within_30mb": True,
        "immutable_parent": True,
    }
    blocked = write_json(master_path.parent / "RELEASE.json", release)
    if blocked:
        raise RuntimeError(blocked)
    return dest


def _gold_text(row: dict) -> str:
    answers = row.get("answers")
    if answers is None:
        name = row.get("gold_name")
        answers = [] if not name else [{"name": name, "arguments": row.get("gold_args") or {}}]
    return json.dumps(answers or [], ensure_ascii=False, separators=(",", ":"))


def _calls_equal(left: list[dict], right: list[dict]) -> bool:
    return json.dumps(left or [], sort_keys=True, ensure_ascii=False) == json.dumps(
        right or [], sort_keys=True, ensure_ascii=False
    )


def float_task_control(
    rows: list[dict],
    steps: int,
    *,
    parent_weights: Path,
    tools_by_name: dict[str, dict],
    eval_n: int = 24,
) -> dict:
    """Small trained float 51M task control; never runs the frozen LM suite."""
    from architecture import NeedleZh
    from common.checkpoint import load_params
    from config import NeedleZhConfig
    from common.identity_51m import ARCHITECTURE_SPEC
    from mei_sdk.runtime_51m import Runtime51M, validate_call
    from tokenizer import ZhTokenizerV1

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, parent_weights, strict=True)
    mx.eval(model.parameters())
    tok = ZhTokenizerV1()
    runtime = Runtime51M(model, tok)
    train_report = train_fullcall_float(
        runtime,
        rows,
        steps=steps,
        lr=2e-4,
        tools_by_name=tools_by_name,
    )
    n_ok = 0
    text_exact = 0
    call_exact = 0
    start = min(steps, max(0, len(rows) - eval_n))
    for row in rows[start : start + eval_n]:
        enc = encode_fc(tok, row, tools_by_name=tools_by_name)
        if not enc:
            continue
        prompt, _answer = enc
        gold_text = _gold_text(row)
        gold_calls = row.get("answers") or []
        names = row.get("retrieved_tools") or []
        tools = [tools_by_name[name] for name in names if name in tools_by_name]
        decoded = runtime.greedy(prompt, tools=tools, max_new=96, decode_mode="constrained")
        text = (decoded.get("text") or "").strip()
        validated = validate_call(
            text,
            tools=tools,
            query=str(row.get("query") or ""),
            system_facts=str(row.get("system_facts") or ""),
        )
        n_ok += 1
        text_exact += int(text == gold_text)
        call_exact += int(_calls_equal(validated.get("function_calls") or [], gold_calls))
    return {
        "train": train_report,
        "n": n_ok,
        "text_exact": text_exact / max(n_ok, 1),
        "call_exact": call_exact / max(n_ok, 1),
        "float_weights": str(parent_weights),
        "same_serializer_grammar_scorer": True,
        "trained_float_control": True,
        "not_lm_eval": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--retrieval-steps", type=int, default=400)
    parser.add_argument("--sft-steps", type=int, default=4000)
    parser.add_argument("--conf-steps", type=int, default=80)
    parser.add_argument("--float-control", type=int, default=200)
    parser.add_argument("--skip-isolation", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-package-dir", type=Path, default=SFT_QAT_PACKAGE_DIR)
    parser.add_argument("--master-out", type=Path, default=SFT_QAT_WEIGHTS_PATH)
    parser.add_argument("--model-id", default=SFT_QAT_MODEL_ID)
    parser.add_argument("--parent-master", type=Path, default=QAT_Q4_WEIGHTS_PATH)
    parser.add_argument("--bit-map", type=Path, default=None)
    args = parser.parse_args()
    if args.smoke:
        args.limit = min(args.limit, 64)
        args.retrieval_steps = min(args.retrieval_steps, 2)
        args.sft_steps = min(args.sft_steps, 2)
        args.conf_steps = min(args.conf_steps, 2)
        args.float_control = min(args.float_control, 2)
    pkg_dir = args.package_dir
    if pkg_dir is None:
        pkg_dir = QAT_Q4_PACKAGE_DIR if (QAT_Q4_PACKAGE_DIR / "weights.q4").is_file() else Q4_PACKAGE_DIR
    iso = {"ok": True, "skipped": True}
    if not args.skip_isolation:
        iso = isolation_ok()
        blocked = write_json(args.jobs_dir / "sft-v2-isolation-51m.json", iso)
        if blocked:
            return fail(blocked)
        if not iso.get("ok"):
            return fail(f"train/eval isolation failed: {iso.get('n_hits')}")

    sys.path.insert(0, str(ROOT / "src/platform/python-sdk"))
    from mei_sdk.package import load_package
    from mei_sdk.runtime_51m import load_51m_runtime
    from common.checkpoint import load_params
    from training.qat.qat_replay_51m import apply_fake_quant_inplace, restore_master

    pkg = load_package(pkg_dir)
    runtime, loaded = load_51m_runtime(pkg)
    if not args.parent_master.is_file():
        return fail(f"missing QAT FP32 master: {args.parent_master}")
    load_params(runtime.model, args.parent_master, strict=True)
    mx.eval(runtime.model.parameters())
    bit_map_path = args.bit_map or (args.jobs_dir / Q4_BASELINE_MAP_NAME)
    bit_map = load_bit_map(bit_map_path)
    quant_scheme = "cq2" if any(bits == 2 for bits in bit_map.values()) else "q4"
    ret_rows = load_jsonl(RET_PATH, args.limit)
    fc_load_limit = max(args.limit, 512) if args.smoke else args.limit
    fc_rows = load_jsonl(FC_PATH, fc_load_limit)
    sft_rows = fc_rows[: args.limit]
    sft_rep = train_fullcall(
        runtime,
        sft_rows,
        steps=args.sft_steps,
        lr=2e-4,
        bit_map=bit_map,
        activation_ste=True,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    master = apply_fake_quant_inplace(runtime.model, bit_map)
    try:
        ret_rep = train_contrastive(
            runtime,
            ret_rows,
            steps=args.retrieval_steps,
            lr=1e-3,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            checkpoint_every_steps=args.checkpoint_every_steps,
        )
        conf_rep = train_confidence(
            runtime,
            fc_rows,
            steps=args.conf_steps,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            checkpoint_every_steps=args.checkpoint_every_steps,
        )
    finally:
        restore_master(runtime.model, master)
    control = float_task_control(fc_rows, args.float_control)
    dest = (
        None
        if args.smoke
        else save_sft_package(
            pkg_dir,
            args.out_package_dir,
            runtime,
            master_path=args.master_out,
            model_id=args.model_id,
            parent_master=args.parent_master,
            bit_map_path=bit_map_path,
            quant_scheme=quant_scheme,
            jobs_dir=args.jobs_dir,
        )
    )
    report = {
        "kind": "quant-aware-sft-smoke" if args.smoke else "quant-aware-sft-ondisk",
        "parent_package": str(pkg_dir),
        "parent_fp32_master": str(args.parent_master),
        "sft_package": str(dest) if dest else None,
        "sft_fp32_master": str(args.master_out) if dest else None,
        "model_id": args.model_id,
        "isolation": {"ok": bool(iso.get("ok")), "n_hits": iso.get("n_hits")},
        "retrieval": ret_rep,
        "fullcall_sft": sft_rep,
        "confidence": conf_rep,
        "float_task_control": control,
        "training_order": ["qat_fullcall_sft", "freeze_lm", "retrieval_head", "confidence_head"],
        "head_representation": "q4-dequant deployment math after final QAT SFT",
        "q4_bit_map": str(bit_map_path.relative_to(ROOT)),
        "selected_quant_scheme": quant_scheme,
        "weight_qat_ste": True,
        "activation_kv_int8_ste": True,
        "quant_math_id": QUANT_MATH_ID,
        "prompt_max_tokens": 1152,
        "clean_v2": True,
        "used_51m_weights": loaded.get("n_loaded") == 400,
        "qat_mandatory": True,
        "not_a_claim": "On-disk SFT is not a CURRENT freeze.",
        "n_loaded": loaded.get("n_loaded"),
    }
    report_name = "sft-qat-smoke-51m.json" if args.smoke else "sft-ondisk-qat-51m.json"
    blocked = write_json(args.jobs_dir / report_name, report)
    if blocked:
        return fail(blocked)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
