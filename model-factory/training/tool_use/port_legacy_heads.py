#!/usr/bin/env python3
"""旧链 per-head 训练管线移植（用户拍板方案 A）。

在新链 LM SFT 完成的基座上，用旧链的目标函数与头部架构训练产品头：
  - retrieval: ContrastiveHead InfoNCE（旧链 RETRIEVAL_OBJECTIVE_ID，
    查询/工具文本经 frozen backbone cells 编码，384 token 上限）
  - mw_disposition: MWDispositionHead 20 类（stable-ring 提示词，
    mei-mw-visible-batch-v1 语义）
  - confidence: ConfidenceV2Head（标签来自轨迹终点确定性编译，
    替代旧链 actual-r1-model 标注——差距已记录）
  - 检索索引导出：finalize_tool_index_v3（ToolIndex 负载，runtime 可用）
LM SFT 本身由 train_sft_newlineage.py 完成；叙述 adapter 与多阶段课程
（旧链 retrieval 400 / sft 4000 / conf 80 / float-control 200 的精确步数
分布）留待下一步移植。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LegacyRuntime:
    """旧链 train_*_v3 期望的 runtime 接口：model / tokenizer / contrastive /
    index / embed_text / build_index。"""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.contrastive = None
        self.index = None
        self._catalog_fp = ""

    def build_index(self, catalog: list[dict[str, Any]]) -> None:
        from mei_sdk.shared import catalog_fingerprint

        # portable 子集校验：剔除 family 等非 portable 键（旧链 runtime 用投影后的工具注册表）
        from schema_subset import TOOL_KEYS

        projected = [{key: tool[key] for key in tool if key in TOOL_KEYS} for tool in catalog]
        fp = catalog_fingerprint(projected)
        if fp == self._catalog_fp and self.index is not None and len(self.index.records) == len(projected):
            return
        self.index.build(projected, self.embed_text)
        self._catalog_fp = fp


def _deploy_tools() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    tools_path = (
        root / ".local/artifacts/mei-1.0-51m/exp-000300m/corpus/sft-suite"
        / "historical-notebook-releases/releases/mei-1.0-51m-tool-sft-v4-300m-v4/tool-universe.json"
    )
    raw = json.loads(tools_path.read_text(encoding="utf-8"))
    return {str(t.get("name") or t.get("tool_id")): t for t in raw["tools"]}


def build_retrieval_rows(release_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = release_dir / "compiled/retrieval/train.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        negatives = list(row.get("hard_negative_names") or [])
        if len(negatives) != 4 or not row.get("gold_name"):
            continue  # 旧链契约要求恰好 4 个难负例
        rows.append({
            "sample_id": row["case_id"],
            "gold_tool": row["gold_name"],
            "hard_negatives": negatives,
            "query": row["query"],
            "_training_bank": "",  # 旧链 bank 加权分区（SFT_V4_BANK_WEIGHTS）后续对齐；空串走 tool-uniform 调度
        })
    return rows


def build_mw_rows(release_dir: Path, tools_by_name: dict[str, dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    from training.tool_use.sft_v3_training_51m import (
        encode_stable_ring_prompt,
        render_mw_prompt_parts,
    )

    rows: list[dict[str, Any]] = []
    path = release_dir / "compiled/mw_disposition/train.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        names = [str(value) for value in (row.get("retrieved_tools") or [])][:5]
        if len(names) != 5:
            continue
        selected = [tools_by_name[name] for name in names if name in tools_by_name]
        if len(selected) != 5:
            continue
        rendered = render_mw_prompt_parts(row, selected)
        prompt_ids, stats = encode_stable_ring_prompt(tokenizer, rendered, sample_id=row["case_id"])
        rows.append({
            "schema": "mei-mw-visible-batch-v1",
            "view_id": row["case_id"],
            "sample_id": row["case_id"],
            "mw_eligible": True,
            "effective_reason_class_id": int(row["reason_class_id"]),
            "original_reason_class_id": int(row["reason_class_id"]),
            "retrieval_mode": "oracle",
            "runtime_profile": "standard",
            "prompt": rendered["prompt"],
            "prompt_ids": prompt_ids,
            "prompt_tokens": stats["prompt_tokens"],
        })
    return rows


def build_confidence_outcomes(
    release_dir: Path, tokenizer: Any, model: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """轨迹终点 → 置信度头训练行。logprob_sum = 模型对 gold 续写的解码
    对数概率（旧链由 r1 实际执行时采集，本移植用冻结模型直接计算）。"""
    import mlx.core as mx
    import mlx.nn as nn
    from training.tool_use.train_sft_newlineage import DeployIndex, RENDERERS

    deploy = DeployIndex()

    def outcome_of(row: dict[str, Any]) -> dict[str, Any]:
        prompt, gold = RENDERERS["trajectory"](row, deploy)
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        gold_ids = tokenizer.encode(gold, add_bos=False, add_eos=False)
        ids = prompt_ids + gold_ids
        logits = model(mx.array([ids], dtype=mx.int32))["logits"][0]
        targets = mx.array(ids[1:], dtype=mx.int32)
        log_probs = nn.log_softmax(logits, axis=-1)[:-1]
        gathered = mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
        logprob_sum = float(mx.sum(gathered).item())
        return {
            "sample_id": row["case_id"],
            "label": 1 if row.get("final_outcome") == "success" else 0,
            "head_eligible": True,
            "prompt_ids": prompt_ids,
            "logprob_sum": logprob_sum,
            "output_tokens": len(gold_ids),
        }

    train_rows: list[dict[str, Any]] = []
    path = release_dir / "compiled/trajectory/train.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            train_rows.append(outcome_of(json.loads(line)))
    valid_rows: list[dict[str, Any]] = []
    vpath = release_dir / "compiled/trajectory/valid.jsonl"
    for line in vpath.read_text(encoding="utf-8").splitlines():
        if line.strip():
            valid_rows.append(outcome_of(json.loads(line)))
    return train_rows, valid_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-dir", type=Path, required=True)
    ap.add_argument("--base-master", type=Path, required=True, help="LM SFT 完成后的 float master")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--retrieval-steps", type=int, default=300)
    ap.add_argument("--mw-steps", type=int, default=200)
    ap.add_argument("--conf-steps", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"write-once refusal: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    import mlx.core as mx
    from common._repo import ARCHITECTURE_DIR
    sys.path.insert(0, str(ARCHITECTURE_DIR))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "platform/python-sdk"))
    from architecture import NeedleZh
    from config import NeedleZhConfig
    from common.checkpoint import load_params
    from training.cpt.train_pretrain import _frozen_tokenizer
    from training.tool_use.sft_v3_training_51m import (
        configure_retrieval_encoding_v3,
        finalize_tool_index_v3,
        train_confidence_v3,
        train_mw_disposition_v3,
        train_retrieval_v3,
    )

    tokenizer = _frozen_tokenizer()
    model = NeedleZh(NeedleZhConfig().tiny() if args.smoke else NeedleZhConfig.from_spec())
    mx.eval(model.parameters())
    log(f"loading SFT master {args.base_master}")
    load_params(model, args.base_master, strict=False, allow_missing_prefixes=(), return_report=True)
    runtime = LegacyRuntime(model, tokenizer)

    tools_by_name = _deploy_tools()
    catalog = list(tools_by_name.values())

    retrieval_rows = build_retrieval_rows(args.release_dir)
    mw_rows = build_mw_rows(args.release_dir, tools_by_name, tokenizer)
    conf_train, conf_valid = build_confidence_outcomes(args.release_dir, tokenizer, model)
    log(f"adapter rows: retrieval={len(retrieval_rows)}, mw={len(mw_rows)}, "
        f"confidence train={len(conf_train)}/valid={len(conf_valid)}")
    if args.smoke:
        retrieval_rows = retrieval_rows[:200]
        # MW 冒烟需保留 20 类全覆盖：每类取前 6 行
        import random as _random
        _rng = _random.Random(20260905)
        by_class: dict[int, list[dict[str, Any]]] = {}
        for row in mw_rows:
            by_class.setdefault(int(row["effective_reason_class_id"]), []).append(row)
        mw_rows = [row for rows in by_class.values() for row in rows[:6]]

    # 1) 检索 InfoNCE（旧链目标函数）
    log("phase 1/3: retrieval InfoNCE head")
    result_retrieval = train_retrieval_v3(
        runtime, retrieval_rows, tools_by_name,
        steps=min(args.retrieval_steps, 30) if args.smoke else args.retrieval_steps,
        lr=1e-3, batch_size=8, temperature=0.07,
        checkpoint_dir=args.out_dir, stage_id="rebuild-v4",
    )
    (args.out_dir / "retrieval-head.json").write_text(
        json.dumps(result_retrieval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 2) MW 处置头
    log("phase 2/3: MW disposition head")
    result_mw = train_mw_disposition_v3(
        runtime, mw_rows, tools_by_name, catalog,
        steps=min(args.mw_steps, 20) if args.smoke else args.mw_steps,
        lr=1e-3, checkpoint_dir=args.out_dir,
    )
    (args.out_dir / "mw-head.json").write_text(
        json.dumps(result_mw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 3) 置信度头
    log("phase 3/3: confidence head")
    result_conf, calibration = train_confidence_v3(
        runtime, conf_train, conf_valid,
        steps=min(args.conf_steps, 20) if args.smoke else args.conf_steps,
        lr=1e-3, minimum_class_rows=50,  # 旧链 100 是 r1 标注质量门槛；轨迹确定性标签可放宽
        checkpoint_dir=args.out_dir,
    )
    (args.out_dir / "confidence-head.json").write_text(
        json.dumps({"result": result_conf, "calibration": calibration}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    # 4) 检索索引导出（runtime ToolIndex 负载）
    log("phase 4/4: tool index export")
    import hashlib
    from common._repo import frozen_tokenizer_path

    def _sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    index_report = finalize_tool_index_v3(
        runtime, catalog, args.out_dir / "tool-index-v1",
        model_sha256=_sha(args.base_master),
        head_sha256=_sha(args.out_dir / "retrieval-rebuild-v4-head-state.npz"),
        tokenizer_sha256=_sha(frozen_tokenizer_path()),
    )
    (args.out_dir / "tool-index.json").write_text(
        json.dumps(index_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("port_legacy_heads done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
