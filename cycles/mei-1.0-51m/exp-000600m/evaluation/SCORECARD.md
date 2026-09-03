# 600M Scorecard

| 指标 | 结果 |
|---|---:|
| Base valid loss | 3.082247 |
| CQ2-QAT valid loss | 3.133236 |
| Float Task Control balanced accuracy | 0.979478 |
| Float Task Control argument exact | 0.899254 |
| R2 eligible gold recall | 0.995606 |
| Adaptive dev structural exact | 0.576531 |
| Adaptive dev natural exact | 0.719298 |
| Adaptive dev schema exact | 0.500000 |
| MW dev accuracy | 0.226000 |
| MW dev macro-F1 | 0.030093 |
| Confidence test AUROC | 0.958787 |
| Confidence test ECE | 0.193082 |
| Narration learned exact acceptance | 0.018333 |
| Narration delivered correctness | 0.958333 |
| Python/MLX warm decode | 233.643 tok/s |
| Browser-WASM warm decode | 83.378 tok/s |
| Browser-WASM heap | 81,920,000 bytes |

这些结果不能完全归因于 exposure：600M 具有 hybrid recovery 与 corpus diversity confound。
