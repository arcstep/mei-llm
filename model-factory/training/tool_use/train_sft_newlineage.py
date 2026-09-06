#!/usr/bin/env python3
"""SFT trainer for the zh-v2 rebuild lineage (7-family rebuild-v4 corpus).

Trains the scratch-300M base with next-token CE over family-specific prompts.
The six heads collaborate through DATA, not separate output heads: trajectory
rows carry the full chain (mw route -> ask/fill -> call -> terminal with
confidence+narration), and the retrieval family carries the MW stop reasons
(stop_* terminals). Confidence supervision comes from trajectory terminals
(the confidence family's labels are null pre-harvest by design).

Prompt contract (plain text, no special role tokens -- zh-24k-v3 has none):
  【任务】system 【候选工具】catalog 【用户】query 【助手】gold
Labels mask everything except the gold span.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from common.checkpoint import load_params, save_params  # noqa: E402

TRAINABLE_FAMILIES = ("retrieval", "full_call", "agent", "mw_disposition", "narration", "trajectory")
SEQ_CAP = 2048
IGNORE_ID = -100

SYSTEMS = {
    "retrieval": "你是工具检索助手。根据用户需求，从候选工具中按相关性从高到低排序，每行输出一个工具名，最多输出5个；都不相关时只输出「无匹配」；遇到必须停止的情况输出「停止：原因码」。",
    "full_call": "你是工具调用助手。请输出一个 JSON 工具调用：{\"tool\":\"工具名\",\"arguments\":{...}}；信息不足无法执行时输出：{\"refuse\":true,\"reason\":\"说明\"}。",
    "agent": "你是多步工具助手。按步骤完成任务：每一步输出一行 JSON 工具调用，最后输出完成说明。",
    "mw_disposition": "你是任务分析器。分析用户请求，输出 JSON：{\"reason_code\":\"原因码\",\"candidate_tool\":\"候选工具名\"}。",
    "narration": "你是结果叙述助手。根据工具调用结果，用一句简洁中文向用户叙述结果。",
    "trajectory": "你是工具智能体。分析用户请求：信息齐全就执行；缺信息就先询问用户或调用查询工具补齐；完成后输出结果叙述和置信度（high/mid/low）。",
}

MW_STOP_REASONS = {
    "authority_required", "illegal_pair", "injection_rejected", "missing_permission_token",
    "negation_cancels", "offtopic", "partial_sequence_blocked", "safety_judgment",
    "scene_conflict", "unknown_slot_value", "unknown_tool", "unsupported_scope",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Prompt rendering: (prompt_text, gold_text) per family row.
# ---------------------------------------------------------------------------
def _tools_block(names: list[str], deploy: Any) -> str:
    lines = ["【候选工具】"]
    for name in names:
        tool = deploy.get(name)
        desc = ""
        if tool is not None:
            desc = str(tool.get("description") or "").rstrip("。")[:48]
        lines.append(f"- {name}" + (f"：{desc}" if desc else ""))
    return "\n".join(lines)


def render_retrieval(row: dict[str, Any], deploy: Any) -> tuple[str, str]:
    reason = str(row.get("retrieval_terminal_reason") or "")
    found = row.get("found_at_batch_index")
    if reason == "found" and found is not None:
        names = row["batches"][found]["tool_names"]
        gold = "\n".join(names)
    elif reason.startswith("stop_"):
        code = reason.removeprefix("stop_")
        gold = f"停止：{code}"
    else:
        gold = "无匹配"
    prompt = f"【任务】{SYSTEMS['retrieval']}\n" + _tools_block(row["catalog_tool_names"], deploy) + f"\n【用户】{row['query']}\n【助手】"
    return prompt, gold


def render_full_call(row: dict[str, Any], deploy: Any) -> tuple[str, str]:
    catalog = list(dict.fromkeys([row["gold_name"]] + row.get("oracle_top5")[1:] if row.get("gold_name") else row.get("oracle_top5") or []))
    names = [n for n in catalog if n]
    if row.get("gold_name"):
        gold = json.dumps({"tool": row["gold_name"], "arguments": row.get("gold_args") or {}}, ensure_ascii=False)
    else:
        missing = row.get("missing_or_conflicting_fields") or []
        gold = json.dumps({"refuse": True, "reason": f"缺少或冲突：{'、'.join(missing) if missing else '不支持'}"}, ensure_ascii=False)
    prompt = f"【任务】{SYSTEMS['full_call']}\n" + _tools_block(names[:6], deploy) + f"\n【用户】{row['query']}\n【助手】"
    return prompt, gold


def render_agent(row: dict[str, Any], deploy: Any) -> tuple[str, str]:
    lines = []
    for step in row.get("steps") or []:
        call = step.get("call") or {}
        lines.append(json.dumps({"tool": call.get("name"), "arguments": call.get("arguments") or {}}, ensure_ascii=False))
    lines.append("好的，已完成。")
    gold = "\n".join(lines)
    prompt = f"【任务】{SYSTEMS['agent']}\n" + _tools_block(row["catalog_tool_names"], deploy) + f"\n【用户】{row['query']}\n【助手】"
    return prompt, gold


def render_mw(row: dict[str, Any], deploy: Any) -> tuple[str, str]:
    gold = json.dumps({"reason_code": row.get("reason_code"), "candidate_tool": row.get("candidate_tool")}, ensure_ascii=False)
    names = (row.get("retrieved_tools") or [])[:8]
    prompt = f"【任务】{SYSTEMS['mw_disposition']}\n" + _tools_block(names, deploy) + f"\n【用户】{row['query']}\n【助手】"
    return prompt, gold


def render_narration(row: dict[str, Any], deploy: Any) -> tuple[str, str]:
    result = row.get("verified_terminal_result") or {}
    payload = result.get("payload") or {}
    context_lines = []
    for key, value in list(payload.items())[:6]:
        context_lines.append(f"{key}: {value}")
    context = "【工具结果】\n" + "\n".join(context_lines) if context_lines else "【工具结果】（无）"
    gold = str(row.get("narration_target") or "")
    prompt = f"【任务】{SYSTEMS['narration']}\n{context}\n【用户】{row['query']}\n【助手】"
    return prompt, gold


def render_trajectory(row: dict[str, Any], deploy: Any) -> tuple[str, str]:
    lines: list[str] = []
    for step in row.get("steps") or []:
        role = step.get("role")
        if role == "mw":
            lines.append("分析：" + json.dumps(
                {"route": step.get("route"), "reason_code": step.get("reason_code"),
                 "candidate_tool": step.get("candidate_tool"), "missing_fields": step.get("missing_fields") or []},
                ensure_ascii=False))
        elif role == "ask_user":
            lines.append(f"询问：{step.get('question')}")
            lines.append(f"用户：{step.get('user_reply')}")
        elif role == "call":
            call = step.get("call") or {}
            lines.append("调用：" + json.dumps({"tool": call.get("name"), "arguments": call.get("arguments") or {}}, ensure_ascii=False))
        elif role == "terminal":
            lines.append(f"结果：{step.get('narration')}")
            lines.append("置信：" + json.dumps({"confidence": step.get("confidence")}, ensure_ascii=False))
    gold = "\n".join(lines)
    prompt = f"【任务】{SYSTEMS['trajectory']}\n" + _tools_block(row["catalog_tool_names"], deploy) + f"\n【用户】{row['query']}\n【助手】"
    return prompt, gold


RENDERERS = {
    "retrieval": render_retrieval,
    "full_call": render_full_call,
    "agent": render_agent,
    "mw_disposition": render_mw,
    "narration": render_narration,
    "trajectory": render_trajectory,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DeployIndex:
    def __init__(self) -> None:
        sys_path = Path(__file__).resolve()
        root = sys_path.parents[3]
        tools_path = (
            root / "artifacts/mei-1.2-51m/legacy/mei-1.0-51m/exp-00300m/corpus/sft-suite"
            / "historical-notebook-releases/releases/mei-1.0-51m-tool-sft-v4-300m-v4/tool-universe.json"
        )
        raw = json.loads(tools_path.read_text(encoding="utf-8"))
        self.by_name = {str(t.get("name") or t.get("tool_id")): t for t in raw["tools"]}

    def get(self, name: str) -> dict[str, Any] | None:
        return self.by_name.get(name)


def load_dataset(release_dir: Path, deploy: DeployIndex, smoke: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in TRAINABLE_FAMILIES:
        path = release_dir / "compiled" / family / "train.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, gold = RENDERERS[family](row, deploy)
            rows.append({"family": family, "case_id": row["case_id"], "prompt": prompt, "gold": gold})
        log(f"loaded {family}: {sum(1 for r in rows if r['family'] == family)} rows")
    if smoke:
        rows = rows[:64]
    return rows


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def encode_pair(tok: Any, prompt: str, gold: str) -> tuple[mx.array, mx.array]:
    prompt_ids = tok.encode(prompt)
    gold_ids = tok.encode(gold)
    if len(prompt_ids) + len(gold_ids) > SEQ_CAP:
        overflow = len(prompt_ids) + len(gold_ids) - SEQ_CAP
        prompt_ids = prompt_ids[overflow:]
    ids = prompt_ids + gold_ids
    labels = [IGNORE_ID] * len(prompt_ids) + gold_ids
    return mx.array(ids, dtype=mx.int32), mx.array(labels, dtype=mx.int32)


def ce_loss(model: Any, ids: mx.array, labels: mx.array) -> tuple[mx.array, mx.array]:
    out = model(ids)["logits"]
    logits = out[:, :-1, :]
    targets = labels[:, 1:]
    safe_targets = mx.where(targets == IGNORE_ID, 0, targets)
    loss = nn.losses.cross_entropy(logits, safe_targets, reduction="none")
    mask = (targets != IGNORE_ID).astype(loss.dtype)
    n = mask.sum()
    if float(n) == 0.0:
        return mx.array(0.0), n
    return (loss * mask).sum() / n, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-dir", type=Path, required=True)
    ap.add_argument("--base-state", type=Path, default=None, help="CPT checkpoint state npz (new lineage); omit for random init (smoke)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=0, help="0 = one epoch over the dataset")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-final", type=float, default=5e-6)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--eval-every-steps", type=int, default=250)
    ap.add_argument("--save-every-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quant-aware", action="store_true",
                    help="fake-quant STE in every forward (SFT on the quantized model; requires a QAT-tuned base)")
    ap.add_argument("--quant-bits", type=int, default=4, help="quant-aware 位宽：4（Q4 基准）或 2（Q2 目标）")
    args = ap.parse_args()

    t0 = time.time()
    if args.out_dir.exists() and not args.resume:
        raise SystemExit(f"write-once refusal: out dir exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import sys as _sys
    from common._repo import ARCHITECTURE_DIR  # noqa: E402
    _sys.path.insert(0, str(ARCHITECTURE_DIR))
    from architecture import NeedleZh  # noqa: E402
    from config import NeedleZhConfig  # noqa: E402
    from training.cpt.train_pretrain import _frozen_tokenizer  # noqa: E402

    tok = _frozen_tokenizer()
    model = NeedleZh(NeedleZhConfig().tiny() if args.smoke else NeedleZhConfig.from_spec())
    mx.eval(model.parameters())
    if args.base_state:
        log(f"loading base state {args.base_state}")
        report = load_params(model, args.base_state, strict=False, allow_missing_prefixes=(), return_report=True)
        n_loaded = report.get("n") if isinstance(report, dict) else "?"
        log(f"loaded {n_loaded} params from base")
    else:
        log("no base state -- random init (smoke only)")

    deploy = DeployIndex()
    rows = load_dataset(args.release_dir, deploy, args.smoke)
    log(f"dataset: {len(rows)} rows")

    steps = args.steps or ((len(rows) + args.batch_size - 1) // args.batch_size)
    total_steps = steps
    if args.smoke:
        steps = min(steps, 30)

    lr_schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, args.lr, args.warmup_steps),
         optim.cosine_decay(args.lr, max(total_steps - args.warmup_steps, 1), args.lr_final)],
        [args.warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule)
    state = [model.state, optimizer.state]
    quant_names: list[str] = []
    bits_by_name: dict[str, int] = {}
    if args.quant_aware:
        from training.qat.qat_newlineage import (  # noqa: E402
            _leaf_of, collect_quant_pairs, pack_model, quantize_inplace, restore_inplace,
        )
        quant_names, bits_by_name = collect_quant_pairs(model)
        log(f"quant-aware SFT: {len(quant_names)} weight tensors fake-quantized per step (legacy f16/cq4/cq2 policy)")

    def step_fn(batch_ids, batch_labels):
        if args.quant_aware:
            originals = {}
            for name in quant_names:
                param, leaf = _leaf_of(model, name)
                originals[name] = getattr(param, leaf)

            def loss_fn(model):
                quantize_inplace(model, quant_names, bits_by_name)
                return ce_loss(model, batch_ids, batch_labels)[0]
            loss_and_grads = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grads(model)
            restore_inplace(model, originals)
            optimizer.update(model, grads)
            return loss

        def loss_fn(model):
            return ce_loss(model, batch_ids, batch_labels)[0]
        loss_and_grads = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grads(model)
        optimizer.update(model, grads)
        return loss

    if not args.no_compile:
        try:
            step_fn = mx.compile(step_fn)
        except Exception as exc:  # Metal 驱动问题：编译路径在本机不可靠
            log(f"mx.compile unavailable ({exc}); falling back to eager")
            step_fn = step_fn

    rng = __import__("random").Random(args.seed)
    batch_idx = list(range(len(rows)))
    rng.shuffle(batch_idx)

    # eval split: reuse the release's valid rows
    valid_pairs = []
    for family in TRAINABLE_FAMILIES:
        vpath = args.release_dir / "compiled" / family / "valid.jsonl"
        for line in vpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, gold = RENDERERS[family](row, deploy)
            valid_pairs.append((family, encode_pair(tok, prompt, gold)))
    valid_by_family: dict[str, list[tuple[mx.array, mx.array]]] = {}
    for family, pair in valid_pairs:
        valid_by_family.setdefault(family, []).append(pair)
    log(f"valid: {len(valid_pairs)} pairs across {len(valid_by_family)} families")

    best_loss: float | None = None
    epoch = 0
    for step in range(steps):
        start = (step * args.batch_size) % len(rows)
        picks = batch_idx[start : start + args.batch_size]
        if len(picks) < args.batch_size:  # wrap
            picks += batch_idx[: args.batch_size - len(picks)]
        encoded = [encode_pair(tok, rows[i]["prompt"], rows[i]["gold"]) for i in picks]
        max_len = min(max(len(ids) for ids, _ in encoded), SEQ_CAP)
        ids_b = mx.array([list(ids.tolist()[:max_len]) + [0] * (max_len - len(ids)) for ids, _ in encoded], dtype=mx.int32)
        lab_b = mx.array([list(labs.tolist()[:max_len]) + [IGNORE_ID] * (max_len - len(labs)) for _, labs in encoded], dtype=mx.int32)
        loss = step_fn(ids_b, lab_b)
        mx.eval(loss, model.parameters())
        if step % 50 == 0 or step == steps - 1:
            elapsed = time.time() - t0
            log(f"step={step}/{steps} loss={float(loss):.4f} elapsed={elapsed:.0f}s")

        if (step % args.eval_every_steps == 0 and step > 0) or step == steps - 1:
            family_losses: dict[str, list[float]] = {}
            for family, pairs in valid_by_family.items():
                for ids, labs in pairs[:60]:
                    loss_v, _ = ce_loss(model, mx.array([ids.tolist()]), mx.array([labs.tolist()]))
                    family_losses.setdefault(family, []).append(float(loss_v))
            summary = {f: (sum(v) / len(v)) for f, v in family_losses.items()}
            mean = sum(summary.values()) / len(summary) if summary else float("inf")
            log("valid: " + ", ".join(f"{f}={v:.4f}" for f, v in sorted(summary.items())))
            (args.out_dir / "summary.json").write_text(json.dumps({
                "step": step, "loss": float(loss), "valid_loss": mean,
                "valid_loss_by_family": summary, "total_steps": total_steps,
                "params": None, "base_state": str(args.base_state),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if best_loss is None or mean < best_loss:
                best_loss = mean
                save_params(model, args.out_dir / "sft-best.npz")
                log(f"saved best (valid={mean:.4f})")

        if step % args.save_every_steps == 0 and step > 0:
            save_params(model, args.out_dir / "sft-last.npz")

    save_params(model, args.out_dir / "sft-final.npz")
    if args.quant_aware:
        from training.qat.qat_newlineage import pack_model  # noqa: E402
        pack_model(model, quant_names, bits_by_name, args.out_dir / "sft-qat-cq.pack")
    log(f"done in {time.time() - t0:.0f}s; best valid {best_loss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
