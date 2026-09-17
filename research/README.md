# 研究探索与情报（research/）

这里是「不训练模型自身」的研究工作——探索实验、浏览器 demo、诊断脚本、情报与新思路。
与产品线（`src/` 源码、`corpus/` 语料、`cycles/` 训练过程、`models/` 成果）分离，独立成入口。

## 目录

- `explore/`：探索实验（大体积本地资产，**不进 git**，索引见下方）
  - `exp-vocab-study/`：词表研究（A10 PyTorch 移植交接，HANDOFF / PLAN / evidence-a10-*）
  - 注：`exp-corpus-survey/` 不在探索区——它是语料盘点**过程账本**（被 adoption/tokenizer
    manifest/receipt 大量引用），留在 `cycles/mei-1.2-51m/exp-corpus-survey/`。
- `demos/`：浏览器数值实验 demo
  - `runtime-bench/`：CPU-WASM / GPU-WebGPU 数值性能对照实验
  - `data-check` 不在此处——它是 `corpus-factory` node_task 生成的生产依赖，留在 `src/demos/data-check/`
- `diagnostics/`：诊断脚本（未登记进 PIPELINES.json）
  - `train_public_sft_minimal.py` / `eval_public_sft_minimal.py`：公开 SFT 最小闭环诊断

## 情报与思路（在独立文档仓 `docs/`）

产品仓不跟踪 `docs/`（由 `docs/.git` 独立管理）。研究相关的文字情报与新思路去那里找：

- 情报：`docs/intel/`（`_index.md`、竞争扫描、论文、日报）
- 新思路 / 交接：`docs/draft/`（按日期命名的探索笔记与交接文档）

## 旧链内嵌探索（不迁移，属历史证据）

`cycles/mei-1.1-51m/` 旧链归档内还嵌有两处探索产物，保留原位作为历史证据，不物理迁出：

- `cycles/mei-1.1-51m/comparisons`
- `cycles/mei-1.1-51m/synthetic-cpt-audit-v1`
