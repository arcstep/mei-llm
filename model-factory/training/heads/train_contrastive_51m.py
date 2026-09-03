#!/usr/bin/env python3
"""Train a 51M ContrastiveHead on a frozen Q4 backbone. Synthetic catalog.

Saves heads.npz next to the Q4 package. Does not unfreeze the 51,463,797 base.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from common.identity_51m import (
    JOBS_DIR,
    Q4_PACKAGE_DIR,
    fail,
    write_json,
)


def _toy_catalog() -> list[dict]:
    names = [
        ("light.set", "打开或关闭灯"),
        ("light.dim", "调节灯光亮度"),
        ("ac.set_temp", "设置空调温度"),
        ("ac.mode", "设置空调模式"),
        ("curtain.open", "打开窗帘"),
        ("curtain.close", "关闭窗帘"),
        ("tv.power", "电视开关"),
        ("tv.input", "切换电视输入源"),
        ("lock.lock", "锁门"),
        ("lock.unlock", "开门解锁"),
        ("fan.speed", "设置风扇转速"),
        ("scene.sleep", "睡眠场景"),
    ]
    out = []
    for name, desc in names:
        out.append(
            {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}},
            }
        )
    # Pad to >5 so retrieval must rank.
    for i in range(147 - len(out)):
        out.append(
            {
                "name": f"extra.tool_{i}",
                "description": f"占位工具{i}",
                "parameters": {"type": "object", "properties": {}},
            }
        )
    return out


def render_tool(tool: dict) -> str:
    return json.dumps(
        {"name": tool["name"], "description": tool.get("description") or "", "parameters": tool.get("parameters") or {}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=Q4_PACKAGE_DIR)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    args = parser.parse_args()
    import sys

    sdk_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "CURRENT.json").is_file()) / "platform/python-sdk"
    sys.path.insert(0, str(sdk_root))
    from common.checkpoint import flatten_params
    from heads import ContrastiveHead
    from mei_sdk.package import load_package
    from mei_sdk.runtime_51m import load_51m_runtime

    pkg = load_package(args.package_dir)
    runtime, _report = load_51m_runtime(pkg)
    cfg = runtime.model.cfg
    head = ContrastiveHead(cfg.d_model, cfg.n_layers)
    mx.eval(head.parameters())
    catalog = _toy_catalog()
    queries = [
        ("把灯打开", "light.set"),
        ("空调调到26度", "ac.set_temp"),
        ("关掉窗帘", "curtain.close"),
        ("电视换到hdmi", "tv.input"),
        ("锁上门", "lock.lock"),
        ("风扇开快一点", "fan.speed"),
        ("我要睡觉了", "scene.sleep"),
        ("把灯调暗", "light.dim"),
    ]
    tools_by_name = {t["name"]: t for t in catalog}

    def cells_of(text: str):
        ids = runtime.tokenizer.encode(text, add_bos=True, add_eos=False)[:64]
        out = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)
        mx.eval(*out["cells"])
        return [mx.stop_gradient(c) for c in out["cells"]]

    q_cells = [cells_of(q) for q, _ in queries]
    pos_cells = [cells_of(render_tool(tools_by_name[g])) for _, g in queries]
    neg_cells = [cells_of(render_tool(catalog[-1])) for _ in queries]
    opt = optim.Adam(learning_rate=1e-3)

    def batch_loss(h):
        q = mx.concatenate([h(c) for c in q_cells], axis=0)
        p = mx.concatenate([h(c) for c in pos_cells], axis=0)
        n = mx.concatenate([h(c) for c in neg_cells], axis=0)
        pos = mx.sum(q * p, axis=-1)
        neg = mx.sum(q * n, axis=-1)
        return mx.mean(nn.softplus(neg - pos))

    last = None
    for _ in range(args.steps):
        loss, grads = mx.value_and_grad(batch_loss)(head)
        opt.update(head, grads)
        mx.eval(head.parameters(), loss)
        last = float(loss.item())

    runtime.contrastive = head
    hits = 0
    for query, gold in queries:
        top = runtime.search_top_k(query, catalog, k=5)
        names = [str(t.get("name")) for t in top]
        hits += int(gold in names)
    recall5 = hits / len(queries)

    from common.checkpoint import _atomic_savez, flatten_params

    arrays = {f"contrastive.{k}": v for k, v in flatten_params(head).items()}
    out_heads = args.package_dir / "heads.npz"
    _atomic_savez(out_heads, arrays)
    report = {
        "stage": "P5",
        "kind": "contrastive-head-q4",
        "steps": args.steps,
        "last_loss": last,
        "catalog_n": len(catalog),
        "recall_at_5": recall5,
        "heads": str(out_heads),
        "backbone_frozen": True,
        "quantized_backbone": True,
        "product_final": False,
        "qat_mandatory": True,
        "not_a_claim": "Toy catalog recall is not a product retrieval freeze.",
    }
    blocked = write_json(args.jobs_dir / "retrieval-head-51m.json", report)
    if blocked:
        return fail(blocked)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if recall5 >= 0.5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
