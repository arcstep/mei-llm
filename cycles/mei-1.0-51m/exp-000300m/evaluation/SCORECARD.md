# 300M Scorecard

| 指标 | 结果 |
|---|---:|
| Base valid loss | 3.170981 |
| CQ2-QAT valid loss | 3.191214 |
| Float Task Control balanced accuracy | 0.580224 |
| Float Task Control argument exact | 0.167910 |
| R2 eligible gold recall | 0.995606 |
| Adaptive dev structural exact | 0.540816 |
| Adaptive dev natural exact | 0.578947 |
| Adaptive dev schema exact | 0.500000 |
| MW dev accuracy | 0.288000 |
| MW dev macro-F1 | 0.040449 |
| Confidence test AUROC | 0.994726 |
| Confidence test ECE | 0.119225 |
| Narration learned exact acceptance | 0.043333 |
| Narration delivered correctness | 0.958333 |
| Python/MLX warm decode | 233.091 tok/s |
| Browser-WASM warm decode | 84.561 tok/s |
| Browser-WASM heap | 82,837,504 bytes |

Agent 只有 alignment replay 与 call→ToolResult→继续的机制证据；最终多步骤 task-success
仍未形成独立冻结质量分数。
