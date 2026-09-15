#!/usr/bin/env python3
"""最小闭环：公开 SFT 语义样板 → 工具LM + 检索头训练验证。

诊断性实验，不是正式 cycle release。回答一个问题：44 个静态语义样板
（``corpus/pools/task-trials/2026-09-14-public-sft-1000-review-r02/``）能否被
现役训练原语消化——encode 不报错、loss 能下降、检索头能训。

范围：只做**单轮 + 单调用**样板（现役 serializer ``serialize_tool_target`` 硬编码
``len(answers) != 1`` 抛错）。多调用、多轮、Mei 147 设备工具对齐均不在此列。

本脚本放在 worktree（隔离 tracked 代码改动），但运行在**主 checkout 资产环境**里：
通过 ``--root`` 指向主 checkout，import 主 checkout 的 src/ 代码，读取主 checkout 的
权重与语料，产出也写回主 checkout 的 cycles/（四分离成果落位）。

训练：
- 工具LM：复用 ``encode_fullcall_row`` 做编译 + 最简顺序 loss 循环（现役三个
  sampler 都要求 execute/refuse 两类行，本批全是 execute，故不套 sampler）。
- 检索：直接调 ``train_retrieval_v3``（gold 工具种类 >= batch_size 即可）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 主 checkout 绝对路径（本脚本在 worktree 里，但必须操作主 checkout 的资产与代码）。
PRIMARY_ROOT = Path("/Users/xuehongwei/codeup/mei-projects/mei-llm")

_NUM_DISTRACTORS = 4


def _ensure_sys_path(root: Path) -> None:
    for p in (
        root / "src/architecture/mei-1.2-51m",
        root / "src/model-factory",
        root / "src/platform/python-sdk",
        root / "src/platform/_shared/runtime",
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _specimens_path(root: Path) -> Path:
    return (
        root
        / "corpus/pools/task-trials/2026-09-14-public-sft-1000-review-r02"
        / "semantic-specimens.jsonl"
    )


def _base_npz(root: Path) -> Path:
    return (
        root
        / "models/mei-1.2-51m/releases/exp-01800m/base"
        / "mei-1.2-51m-base-cpt1800m-v1"
        / "mei-1.2-51m-base-cpt1800m-v1.npz"
    )


def _load_specimens(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_tools_by_name(rows: list[dict]) -> dict[str, dict]:
    tools: dict[str, dict] = {}
    for row in rows:
        for tool in row.get("source_catalog") or []:
            name = tool.get("name")
            if not name or name in tools:
                continue
            tools[name] = {
                "name": name,
                "description": tool.get("description") or "",
                "parameters": tool.get("parameters")
                or {"type": "object", "properties": {}, "required": []},
            }
    return tools


def _distractors(
    gold: str, catalog_names: list[str], all_names: list[str], k: int = _NUM_DISTRACTORS
) -> list[str]:
    """优先用同组可见但未选用的工具，不足再从全局目录按序补足。"""
    chosen: list[str] = [n for n in catalog_names if n != gold]
    for n in all_names:
        if len(chosen) >= k:
            break
        if n != gold and n not in chosen:
            chosen.append(n)
    return chosen[:k]


def convert(
    rows: list[dict], tools_by_name: dict[str, dict], serialize
) -> tuple[list[dict], list[dict], list[dict]]:
    """返回 (fullcall_rows, retrieval_rows, skipped)。只做单轮+单调用。"""
    all_names = sorted(tools_by_name)
    fullcall: list[dict] = []
    retrieval: list[dict] = []
    skipped: list[dict] = []
    for row in rows:
        calls = row.get("calls") or []
        msgs = row.get("input", {}).get("messages") or []
        if len(calls) != 1:
            skipped.append({"id": row.get("id"), "reason": "multi_call"})
            continue
        # 只做单轮单 user 消息，避免未来结果泄漏 / 定位歧义。
        if len(msgs) != 1 or msgs[0].get("role") != "user":
            skipped.append({"id": row.get("id"), "reason": "not_single_turn"})
            continue
        query = str(msgs[0].get("content") or "").strip()
        if not query:
            skipped.append({"id": row.get("id"), "reason": "empty_query"})
            continue
        call = calls[0]
        gold_name = str(call["name"])
        answers = [{"name": gold_name, "arguments": call.get("arguments") or {}}]
        target_text = serialize(answers)
        catalog_names = [t.get("name") for t in row.get("source_catalog") or []]
        negatives = _distractors(gold_name, catalog_names, all_names)
        sample_id = str(row["id"])
        fullcall.append(
            {
                "sample_id": sample_id,
                "query": query,
                "history": [],
                "retrieved_tools": [gold_name] + negatives,
                "answers": answers,
                "target_text": target_text,
                "kind": "execute",
            }
        )
        retrieval.append(
            {
                "sample_id": sample_id,
                "case_id": sample_id,
                "query": query,
                "gold_tool": gold_name,
                "hard_negatives": negatives,
                "catalog_tools": catalog_names,
                "kind": "hard_positive",
            }
        )
    return fullcall, retrieval, skipped


def _write_rows(
    out: Path, tools: dict, fullcall: list, retrieval: list, skipped: list
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tools.json").write_text(
        json.dumps(tools, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    for name, rows in (("fullcall.jsonl", fullcall), ("retrieval.jsonl", retrieval)):
        with (out / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "skipped.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _train_lm_sequential(
    runtime, compiled: list[tuple[list[int], list[int], dict]], *, steps: int, lr: float
) -> list[float]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from common.train_common import clip_grads

    model = runtime.model
    optimizer = optim.Adam(learning_rate=lr)
    losses: list[float] = []
    n = len(compiled)
    for step in range(steps):
        prompt, answer, _stats = compiled[step % n]
        ids = prompt + answer

        def loss_fn(parameters):
            model.update(parameters)
            logits = model(mx.array([ids], dtype=mx.int32))["logits"].astype(mx.float32)
            log_probabilities = nn.log_softmax(logits, axis=-1)
            terms = [
                -log_probabilities[0, len(prompt) - 1 + index, int(token_id)]
                for index, token_id in enumerate(answer)
            ]
            return mx.mean(mx.stack(terms))

        parameters = model.parameters()
        loss, gradients = mx.value_and_grad(loss_fn)(parameters)
        gradients, _grad_norm = clip_grads(gradients, max_norm=1.0)
        model.update(parameters)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), loss)
        step_loss = float(loss.item())
        losses.append(step_loss)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"  [lm] step {step + 1}/{steps} loss={step_loss:.4f}", flush=True)
    return losses


def _greedy_continue(model, ids: list[int], *, max_new: int, eos_id: int) -> list[int]:
    import mlx.core as mx

    out = list(ids)
    for _ in range(max_new):
        logits = model(mx.array([out], dtype=mx.int32))["logits"][0, -1]
        tok = int(mx.argmax(logits).item())
        if tok == eos_id:
            break
        out.append(tok)
    return out


def _generate_check(runtime, rows: list[dict], pairs: list) -> list[dict]:
    """留出样本训后 greedy 生成，判断 gold 工具名是否出现在生成文本里。"""
    tokenizer = runtime.tokenizer
    results: list[dict] = []
    for row, (prompt, _answer, _stats) in zip(rows, pairs):
        gold = str(row["answers"][0]["name"])
        gen_ids = _greedy_continue(runtime.model, prompt, max_new=128, eos_id=tokenizer.eos_id)
        text = tokenizer.decode(gen_ids[len(prompt):])
        results.append(
            {
                "sample_id": row["sample_id"],
                "gold_tool": gold,
                "hit": gold in text,
                "norm_hit": gold.replace(" ", "") in text.replace(" ", ""),
                "generated": text,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(PRIMARY_ROOT))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--retrieval-steps", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--holdout", type=int, default=5)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    _ensure_sys_path(root)

    import contracts.sft_v3_contract_51m as contract
    import training.tool_use.sft_v3_training_51m as training

    specs_path = _specimens_path(root)
    base_npz = _base_npz(root)
    out = Path(args.out_dir) if args.out_dir else (
        root / "cycles/mei-1.2-51m/exp-corpus-survey/runs/2026-09-15-minimal-public-sft-loop"
    )

    rows = _load_specimens(specs_path)
    tools = _build_tools_by_name(rows)
    fullcall, retrieval, skipped = convert(rows, tools, contract.serialize_tool_target)
    print(
        f"样板 {len(rows)} → full-call {len(fullcall)} / retrieval {len(retrieval)} "
        f"/ 跳过 {len(skipped)} / 工具目录 {len(tools)}"
    )
    if not fullcall:
        print("没有可转换的单轮单调用样板，退出")
        return 1

    _write_rows(out, tools, fullcall, retrieval, skipped)
    print(f"转换产物已写：{out}")

    # 加载 1800M float runtime（不量化，最小闭环先 float）。
    from orchestration.productize_51m import _load_runtime

    runtime = _load_runtime(base_npz, quantized=False)
    tokenizer = runtime.tokenizer

    # 1. 编译验证：encode_fullcall_row 逐行，serializer drift / token 预算都在这。
    compiled: list[tuple[list[int], list[int], dict]] = []
    failures: list[dict] = []
    for row in fullcall:
        try:
            compiled.append(training.encode_fullcall_row(tokenizer, row, tools))
        except Exception as error:  # noqa: BLE001 - 诊断脚本逐行定位
            failures.append({"sample_id": row.get("sample_id"), "error": str(error)})
    print(f"编译验证：通过 {len(compiled)}/{len(fullcall)}，失败 {len(failures)}")
    for failure in failures:
        print(f"  [compile-fail] {failure['sample_id']}: {failure['error']}")

    if args.skip_training or not compiled:
        report = {
            "specimens": len(rows),
            "fullcall_rows": len(fullcall),
            "retrieval_rows": len(retrieval),
            "skipped": skipped,
            "tools": len(tools),
            "compiled": len(compiled),
            "compile_failures": failures,
        }
        (out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("跳过训练（--skip-training 或无编译通过行）")
        return 0

    # 2. 留出 holdout：最后 N 个不训练，训后生成判正误。
    holdout_n = min(max(args.holdout, 0), len(compiled) - 1)
    train_compiled = compiled[:-holdout_n] if holdout_n else compiled
    holdout_pairs = compiled[-holdout_n:] if holdout_n else []
    holdout_rows = fullcall[-holdout_n:] if holdout_n else []
    print(f"留出 {holdout_n} 个样本（训练 {len(train_compiled)} 个）")

    # 3. 工具LM 最简顺序训练。
    print(f"工具LM 顺序训练 {args.steps} 步 (lr={args.lr})")
    lm_losses = _train_lm_sequential(runtime, train_compiled, steps=args.steps, lr=args.lr)

    # 4. 训后生成验证（留出样本）。
    gen_results = _generate_check(runtime, holdout_rows, holdout_pairs) if holdout_pairs else []
    if gen_results:
        hits = sum(1 for r in gen_results if r["hit"])
        norm_hits = sum(1 for r in gen_results if r["norm_hit"])
        print(f"留出生成：精确命中 {hits}/{len(gen_results)}，去空格命中 {norm_hits}/{len(gen_results)}")
        for r in gen_results:
            print(f"  [holdout] {r['sample_id']} gold={r['gold_tool']} hit={r['hit']} norm_hit={r['norm_hit']} gen={r['generated'][:80]!r}")

    # 5. 检索头训练（全量检索正例，不参与留出）。
    n_tools = len({row["gold_tool"] for row in retrieval})
    batch_size = max(1, min(8, n_tools))
    print(f"检索训练 {args.retrieval_steps} 步 (gold 工具 {n_tools} 种, batch={batch_size})")
    retrieval_report = training.train_retrieval_v3(
        runtime,
        retrieval,
        tools,
        steps=args.retrieval_steps,
        lr=1e-3,
        batch_size=batch_size,
        stage_id="minimal-retrieval-v1",
    )

    report = {
        "base": str(base_npz),
        "specimens": len(rows),
        "fullcall_rows": len(fullcall),
        "retrieval_rows": len(retrieval),
        "skipped": skipped,
        "tools": len(tools),
        "compiled": len(compiled),
        "compile_failures": failures,
        "holdout": {
            "n": holdout_n,
            "hits": sum(1 for r in gen_results if r["hit"]),
            "norm_hits": sum(1 for r in gen_results if r["norm_hit"]),
            "total": len(gen_results),
            "results": gen_results,
        },
        "lm": {
            "steps": args.steps,
            "lr": args.lr,
            "first_loss": lm_losses[0],
            "last_loss": lm_losses[-1],
            "min_loss": min(lm_losses),
            "losses": lm_losses,
        },
        "retrieval": {
            "last_loss": retrieval_report.get("last_loss"),
            "tools": retrieval_report.get("tools"),
            "steps": retrieval_report.get("steps"),
        },
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"完成。LM loss {lm_losses[0]:.4f} → {lm_losses[-1]:.4f} "
        f"(min {min(lm_losses):.4f})；retrieval loss {retrieval_report.get('last_loss')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
