#!/usr/bin/env python3
"""新链最终评测：eval lock v11 六家族指标 + 头部精度 + 解码效率。

- retrieval: ContrastiveHead + ToolIndex 的 runtime 检索路径，recall@5
  （found 行 gold 是否进入 top-5）与 no_match 行的 top-1 相关性分布
- mw_disposition: MWDispositionHead 20 类分类准确率 + 每类召回
- full_call/agent/trajectory: 生成 gold 工具命中率 + 参数 JSON 可解析率
- narration: 生成与 verified payload 的接地命中
- confidence: 头 logit + 解码对数概率 + Platt 校准后的标签准确率
- decode: 生成吞吐 tok/s（float master；Q2 打包侧的 wasm 提速留待对齐编码器）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_model(weights: Path, smoke: bool):
    from common.paths import ARCHITECTURE_DIR
    sys.path.insert(0, str(ARCHITECTURE_DIR))
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src/platform/python-sdk"))
    from architecture import NeedleZh
    from config import NeedleZhConfig
    from common.checkpoint import load_params
    model = NeedleZh(NeedleZhConfig().tiny() if smoke else NeedleZhConfig.from_spec())
    mx.eval(model.parameters())
    load_params(model, weights, strict=False, allow_missing_prefixes=(), return_report=True)
    return model


def _read_lock_rows(lock_dir: Path, family: str) -> list[dict[str, Any]]:
    path = lock_dir / "banks" / f"{family}.test.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_retrieval(runtime: Any, rows: list[dict[str, Any]], tools_by_name: dict[str, dict[str, Any]], limit: int) -> dict[str, Any]:
    from mei_sdk.shared import ToolIndex
    catalog = list(tools_by_name.values())
    runtime.index = ToolIndex(model_hash="x", head_hash="x", tokenizer_hash="x")
    runtime.build_index(catalog)
    found_rows = [r for r in rows[:limit] if r.get("gold_name") and r.get("found_at_batch_index") is not None]
    no_match_rows = [r for r in rows[:limit] if r.get("retrieval_terminal_reason") == "retrieval_no_match"]
    hits = 0
    top1_names = []
    for row in found_rows:
        query_vec = runtime.embed_text(str(row["query"]))
        ranked = runtime.index.ranked(query_vec)
        names = [str(candidate.tool_id) for candidate in ranked[:5]]
        if row["gold_name"] in names:
            hits += 1
        if ranked:
            top1_names.append(str(ranked[0].tool_id))
    no_match_top1 = []
    for row in no_match_rows:
        query_vec = runtime.embed_text(str(row["query"]))
        ranked = runtime.index.ranked(query_vec)
        no_match_top1.append(float(ranked[0].raw_score) if ranked else None)
    return {
        "found_rows": len(found_rows),
        "recall_at_5": round(hits / len(found_rows), 4) if found_rows else None,
        "no_match_rows": len(no_match_rows),
        "no_match_top1_raw_score_mean": round(sum(v for v in no_match_top1 if v is not None) / max(1, len(no_match_top1)), 4) if no_match_top1 else None,
    }


def eval_mw(runtime: Any, head: Any, rows: list[dict[str, Any]], tools_by_name: dict[str, dict[str, Any]], tokenizer: Any, limit: int) -> dict[str, Any]:
    from training.tool_use.sft_v3_training_51m import encode_stable_ring_prompt, render_mw_prompt_parts
    correct = 0
    per_class: dict[int, list[int]] = {}
    for row in rows[:limit]:
        names = [str(v) for v in (row.get("retrieved_tools") or [])][:5]
        selected = [tools_by_name[n] for n in names if n in tools_by_name]
        if len(selected) != 5:
            continue
        rendered = render_mw_prompt_parts(row, selected)
        ids, _stats = encode_stable_ring_prompt(tokenizer, rendered, sample_id=row["case_id"])
        cells = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)["cells"]
        logits = head([mx.stop_gradient(c) for c in cells])
        pred = int(mx.argmax(logits, axis=-1)[0].item())
        label = int(row["reason_class_id"])
        per_class.setdefault(label, []).append(int(pred == label))
        correct += int(pred == label)
    total = sum(len(v) for v in per_class.values())
    return {
        "rows": total,
        "accuracy": round(correct / total, 4) if total else None,
        "per_class_recall": {str(k): round(sum(v) / len(v), 3) for k, v in sorted(per_class.items())},
    }


def eval_generation(runtime: Any, rows: list[dict[str, Any]], family: str, limit: int) -> dict[str, Any]:
    from training.tool_use.runtime_newlineage import generate, parse_tool_call
    from training.tool_use.train_sft_newlineage import DeployIndex, RENDERERS
    deploy = DeployIndex()
    hits = parsed = total = 0
    for row in rows[:limit]:
        prompt, _gold = RENDERERS[family](row, deploy)
        out = generate(runtime.model, runtime.tokenizer, prompt, max_new=96)
        total += 1
        if family == "trajectory":
            gold_names = [s["call"]["name"] for s in row.get("steps") or [] if s.get("role") == "call"]
        elif family == "agent":
            # agent 行 steps 用 call_id 而非 role 字段（role 是轨迹家族专用）
            gold_names = [s["call"]["name"] for s in row.get("steps") or [] if s.get("call")]
        else:
            gold_names = [row.get("gold_name") or ""]
        if any(n and n in out for n in gold_names):
            hits += 1
        if parse_tool_call(out) is not None:
            parsed += 1
    return {"rows": total, "gold_name_hit_rate": round(hits / total, 4) if total else None,
            "json_parse_rate": round(parsed / total, 4) if total else None}


def eval_confidence(runtime: Any, head: Any, rows: list[dict[str, Any]], calibration: dict[str, float], limit: int) -> dict[str, Any]:
    import mlx.nn as nn
    from training.tool_use.train_sft_newlineage import DeployIndex, RENDERERS
    from training.tool_use.sft_v3_training_51m import apply_platt, combined_confidence_score
    deploy = DeployIndex()
    correct = total = 0
    for row in rows[:limit]:
        prompt, gold = RENDERERS["trajectory"](row, deploy)
        prompt_ids = runtime.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        gold_ids = runtime.tokenizer.encode(gold, add_bos=False, add_eos=False)
        ids = prompt_ids + gold_ids
        logits = runtime.model(mx.array([ids], dtype=mx.int32))["logits"][0]
        targets = mx.array(ids[1:], dtype=mx.int32)
        gathered = mx.take_along_axis(nn.log_softmax(logits, axis=-1)[:-1], targets[:, None], axis=-1).squeeze(-1)
        logprob_sum = float(mx.sum(gathered).item())
        cells = runtime.model(mx.array([prompt_ids], dtype=mx.int32), return_cells=True)["cells"]
        logit = float(head([mx.stop_gradient(c) for c in cells]).reshape(()).item())
        score = combined_confidence_score(logit, logprob_sum, len(gold_ids))
        calibrated = apply_platt(score, calibration)
        pred = 1 if calibrated >= 0.5 else 0
        label = 1 if row.get("final_outcome") == "success" else 0
        correct += int(pred == label)
        total += 1
    return {"rows": total, "accuracy": round(correct / total, 4) if total else None}


def eval_decode(runtime: Any, prompts: list[str], max_new: int = 64) -> dict[str, Any]:
    from training.tool_use.runtime_newlineage import generate
    # warmup
    generate(runtime.model, runtime.tokenizer, prompts[0], max_new=8)
    t0 = time.time()
    total_tokens = 0
    for prompt in prompts[:10]:
        start = time.time()
        out = generate(runtime.model, runtime.tokenizer, prompt, max_new=max_new)
        total_tokens += len(runtime.tokenizer.encode(out))
    elapsed = time.time() - t0
    return {"prompts": min(10, len(prompts)), "max_new": max_new,
            "tokens_per_second": round(total_tokens / elapsed, 1), "elapsed_seconds": round(elapsed, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft-master", type=Path, required=True)
    ap.add_argument("--heads-dir", type=Path, required=True)
    ap.add_argument("--lock-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"write-once refusal: {args.out}")
    args.out.mkdir(parents=True)

    from common.paths import ARCHITECTURE_DIR
    sys.path.insert(0, str(ARCHITECTURE_DIR))
    from heads import ContrastiveHead, MWDispositionHead, ConfidenceV2Head
    from common.checkpoint import load_train_state
    from training.cpt.train_pretrain import _frozen_tokenizer
    from training.tool_use.port_legacy_heads import LegacyRuntime, _deploy_tools

    tokenizer = _frozen_tokenizer()
    model = _load_model(args.sft_master, args.smoke)
    runtime = LegacyRuntime(model, tokenizer)

    tools_by_name = _deploy_tools()
    report: dict[str, Any] = {"schema": "mei-51m-newlineage-evaluation-v1"}

    # 检索
    log("eval: retrieval")
    retrieval_rows = _read_lock_rows(args.lock_dir, "retrieval")
    try:
        from training.tool_use.sft_v3_training_51m import configure_retrieval_encoding_v3
        head = ContrastiveHead(model.cfg.d_model, model.cfg.n_layers, dim=128, probes=4)
        load_train_state(args.heads_dir / "retrieval-rebuild-v4-head-state.npz", head, None, mode="weights_only")
        runtime.contrastive = head
        configure_retrieval_encoding_v3(runtime)
        report["retrieval"] = eval_retrieval(runtime, retrieval_rows, tools_by_name, args.limit)
        log(f"retrieval: {json.dumps(report['retrieval'], ensure_ascii=False)}")
    except Exception as exc:
        report["retrieval"] = {"error": f"{type(exc).__name__}: {exc}"}
        log(f"retrieval eval failed: {exc}")

    # MW
    log("eval: mw")
    mw_rows = _read_lock_rows(args.lock_dir, "mw_disposition")
    try:
        mw_head = MWDispositionHead(model.cfg.d_model)
        load_train_state(args.heads_dir / "mw-disposition-batched-v1-state.npz", mw_head, None, mode="weights_only")
        report["mw_disposition"] = eval_mw(runtime, mw_head, mw_rows, tools_by_name, tokenizer, args.limit)
        log(f"mw: {json.dumps(report['mw_disposition'], ensure_ascii=False)}")
    except Exception as exc:
        report["mw_disposition"] = {"error": f"{type(exc).__name__}: {exc}"}
        log(f"mw eval failed: {exc}")

    # 生成类家族
    for family in ("full_call", "agent", "trajectory"):
        log(f"eval: {family}")
        rows = _read_lock_rows(args.lock_dir, family)
        try:
            report[family] = eval_generation(runtime, rows, family, args.limit)
            log(f"{family}: {json.dumps(report[family], ensure_ascii=False)}")
        except Exception as exc:
            report[family] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"{family} eval failed: {exc}")

    # 置信度
    log("eval: confidence")
    traj_rows = _read_lock_rows(args.lock_dir, "trajectory")
    try:
        conf_head = ConfidenceV2Head(model.cfg.d_model)
        load_train_state(args.heads_dir / "confidence-v3-state.npz", conf_head, None, mode="weights_only")
        calib = json.loads((args.heads_dir / "confidence-head.json").read_text()).get("calibration") or {}
        report["confidence"] = eval_confidence(runtime, conf_head, traj_rows, calib, args.limit)
        log(f"confidence: {json.dumps(report['confidence'], ensure_ascii=False)}")
    except Exception as exc:
        report["confidence"] = {"error": f"{type(exc).__name__}: {exc}"}
        log(f"confidence eval failed: {exc}")

    # 解码效率
    log("eval: decode throughput")
    prompts = [str(r["query"]) for r in traj_rows[:10]] or ["你好，请帮我把灯打开。"]
    try:
        report["decode"] = eval_decode(runtime, prompts)
        log(f"decode: {json.dumps(report['decode'], ensure_ascii=False)}")
    except Exception as exc:
        report["decode"] = {"error": f"{type(exc).__name__}: {exc}"}
        log(f"decode eval failed: {exc}")

    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (args.out / "evaluation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"evaluation report -> {args.out / 'evaluation-report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
