# mei-1.0-51m 周期总览

这里是模型长期迭代的唯一主入口。模型参数始终为 51,463,797；300M、600M 等是累计
token exposure。每轮同时绑定语料、Base、QAT/SFT、heads、Runtime 门禁与决策。

## 已执行周期

| Cycle | 状态 | Base loss | Natural exact | MW macro-F1 | WASM tok/s | 决策 |
|---|---|---:|---:|---:|---:|---|
| [exp-000300m](exp-000300m/README.md) | process complete / release ineligible | 3.170981 | 0.578947 dev | 0.040449 dev | 84.560829 | 当前正式 Base；产品候选不发布 |
| [exp-000600m](exp-000600m/README.md) | process complete / release ineligible | 3.082247 | 0.719298 dev | 0.030093 dev | 83.378160 | 可续训 checkpoint；不自动晋升父版 |

## 300M / 600M 同合同指标

| 维度 | 300M | 600M | 解释 |
|---|---:|---:|---|
| 实际累计 exposure | 300,000,485 | 600,001,765 | 600M 的实际增量为 300,001,280 |
| Base valid loss | 3.170981 | 3.082247 | 600M 改善 |
| CQ2-QAT valid loss | 3.191214 | 3.133236 | 两侧量化后均有损失税 |
| CQ2-QAT 相对 loss 税 | 0.006381 | 0.016543 | 600M 税更高 |
| Float Task balanced accuracy | 0.580224 | 0.979478 | 600M 显著改善 |
| Float Task argument exact | 0.167910 | 0.899254 | 这是 control 指标，不冒充最终 full-call 分项 |
| Retrieval eligible gold recall | 0.995606 | 0.995606 | 过滤后的总体召回接近目标 |
| No-match 误选率 | 0.833333（60 条） | 0.850000（60 条） | 双阈值校准失败，均不可用 |
| rank>5 gold 保留率 | 0.571429（7 条） | 0.166667（6 条） | 后续批候选漏召回严重；样本也过少 |
| Adaptive dev structural exact | 0.540816 | 0.576531 | 小幅改善 |
| Adaptive dev natural exact | 0.578947 | 0.719298 | 600M 改善 |
| Adaptive dev schema exact | 0.500000 | 0.500000 | 无改善 |
| 最终 full-call 工具名/参数分项 | 未独立测量 | 未独立测量 | 现有 adaptive exact 是联合指标，不能拆写 |
| Agent 多步任务成功率 | 未测量 | 未测量 | 仅有训练 replay loss；质量门禁仍 open |
| MW accuracy / macro-F1 | 0.288000 / 0.040449 | 0.226000 / 0.030093 | 20 类 disposition 均不可用 |
| Confidence AUROC / ECE | 0.994726 / 0.119225 | 0.958787 / 0.193082 | 600M 校准更差 |
| Narration learned / delivered | 0.043333 / 0.958333 | 0.018333 / 0.958333 | delivered 依赖确定性 fallback |
| CQ2 package bytes | 18,858,065 | 18,858,025 | 均低于 18 MiB 上限 |
| Python/MLX warm decode tok/s | 233.091390 | 233.642799 | 均低于 280 目标 |
| Python gate RSS bytes | 988,364,800 | 988,069,888 | 加载并 warm probe 后进程 RSS |
| Browser-WASM heap bytes | 82,837,504 | 81,920,000 | 均低于 96 MiB 上限 |
| Browser-WASM warm decode tok/s | 84.560829 | 83.378160 | 均低于 100 目标 |

## 本轮 CPT 增量语料

| Slice | 300M scratch | 600M delta | 约占本轮 |
|---|---:|---:|---:|
| wiki | 166,657,644 | 166,657,317 | 55.55% |
| hq | 98,372,582 | 98,372,261 | 32.79% |
| structure | 4,861,158 | 4,861,154 | 1.62% |
| colloquial | 30,108,616 | 30,108,783 | 10.04% |

这两个 mix 的数量近乎相同，但 600M 的增量语料审计为
`corpus_diversity_degraded`，不能原样复用。配对还存在 `hybrid_recovery` 混杂，因此这里只
提供描述性证据，不能把全部差值归因于 exposure。

## 资格与缺口

| 资格轴 | 300M | 600M |
|---|---|---|
| process complete | true | true |
| product release eligible | false | false |
| continuation checkpoint eligible | true | true |
| automatic parent promotion eligible | true（当前正式 Base） | false |
| corpus reuse eligible | 待同范围多样性复核 | false |
| lineage assurance | legacy unspecified | hybrid recovery |

下一步不是立即扩大到 900M，而是先用语料工厂补足 no-match、rank>5、最终 full-call 分项、
1–4 步 Agent、20 类 MW、真实 outcome/confidence 与短中文 narration，并冻结可跨周期复用的
评测。质量不足不会抹掉已完成机制，只会保持 release ineligible。

真实模型文件不埋在 cycle 或 run 中。统一从
[`models/mei-1.0-51m/releases/`](../../models/mei-1.0-51m/releases/) 进入；cycle 负责解释
语料、训练、指标和决策，model releases 负责保存不可变二进制资产。

## 训练与评估流水线

- [300M pipeline](exp-000300m/pipeline/PIPELINE.md)：历史 scratch Base + adaptive-v5 产品化。
- [600M pipeline](exp-000600m/pipeline/PIPELINE.md)：300M parent 的 CPT + 同合同 adaptive-v5 产品化。

两轮产品化都由多个不可变 run 恢复闭环，现已逐项记录 run fingerprint、plan hash、stage
顺序和 source manifest。历史 source capture 的最弱覆盖均为精确恢复 `213/242`，另有
29 项仅剩哈希，所以是 `source_capture_mode=reconstructed / exact_reproducible=false`。
这不会把现有权重或分数变成无效，但明确阻止我们声称旧流程可逐字节重放。未来 cycle
必须在启动前冻结完整 source bundle 和每阶段源码闭包。

## 规划周期

`exp-000900m`、`exp-001200m`、`exp-001500m`、`exp-001800m`、`exp-002100m`、
`exp-002400m`、`exp-002700m`、`exp-003000m` 和 `exp-012000m` 只登记在 registry；开始
实际工作前不创建空目录。

```bash
PYTHONPATH=src .venv/bin/python -m mei_llm cycle list
PYTHONPATH=src .venv/bin/python -m mei_llm cycle compare exp-000300m exp-000600m
```
