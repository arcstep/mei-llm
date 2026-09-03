# 300M / 600M 同合同对比

| 指标 | 300M | 600M | 观察 |
|---|---:|---:|---|
| Base valid loss | 3.170981 | 3.082247 | 600M 改善 |
| Float Task balanced accuracy | 0.580224 | 0.979478 | 600M 大幅改善 |
| Float Task argument exact | 0.167910 | 0.899254 | 600M 大幅改善 |
| Adaptive dev natural exact | 0.578947 | 0.719298 | 600M 改善 |
| Adaptive dev structural exact | 0.540816 | 0.576531 | 小幅改善 |
| Schema holdout exact | 0.500000 | 0.500000 | 无改善 |
| MW dev macro-F1 | 0.040449 | 0.030093 | 600M 更差，均不可用 |
| Confidence test ECE | 0.119225 | 0.193082 | 600M 校准更差 |
| Learned narration acceptance | 0.043333 | 0.018333 | 600M 更差，依赖模板 fallback |
| Python/MLX warm decode | 233.091 | 233.643 | 基本相同 |
| Browser-WASM warm decode | 84.561 | 83.378 | 基本相同，均低于 100 目标 |

结论：600M 显著增强了 Base 与主要工具调用学习能力，但没有自动解决检索拒绝边界、MW
20 类区分、confidence 校准和 learned narration。下一步应先修复语料与冻结评测，再决定
是否进入 900M；不能只扩大 exposure。

配对只作描述性证据，因为 600M 有 `hybrid_recovery` 和 `corpus_diversity_degraded`。
