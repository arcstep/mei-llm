# STATUS —— mei-51m-v1.3-800m-tool-sft-cq2-v2-adaptive-v5

v1.3-800m 的 quant-aware 五头 SFT 产物（与 v1.2-sft-1.8b 同流程 `adaptive-v5`）。

## 血统

- base：`mei-51m-v1.3-800m-base`（51.5M 参数、800,000,000 token、hans-en-24k-v1 词表）
- 流程：phase-binding 产品化主线 `adaptive-v5`（16-stage，`package_v2_cq2_v5 passed`，5 个 stage degraded）
- 量化：`mei-cq-v2-g128-wht-codebook`（QAT master `selected-master.npz` 已迁回并作为 QAT 凭证记录）
- 状态：`process_complete=true`，`release_eligible=false`（2 个 runtime gate degraded，见下）
- 产物：11 文件 + ASSETS.json（大二进制 tensors.bin 18MB / tokenizer.model 353KB gitignore，仓外备份）

## 五头跑分（口径 = float base 独立训单头 + holdout 评测，同 v1.2 STATUS.md）

| head | 任务 | 指标 | v1.2-1.8b | v1.3-800m | 结论 |
|---|---|---|---|---|---|
| retrieval | 判别（选工具）| recall@1 | 95.7% | **96.03%** | 持平/略优 |
| tool_lm | 生成（工具调用）| execute exact | 77.1% | **77.07%** | 持平（容量天花板）|
| disposition | 判别（18 类 MW 偏离）| macro_f1 | 0.944 | **0.9585** | 略优 |
| confidence | 判别（二分类）| AUROC | 0.9807 | **0.9918** | 略优 |
| narration | 生成（设备解说）| adapter exact | 43%¹ | **65.6%** | 均 degraded，走模板 |

¹ v1.2 的 43% 为 sidecar rank 消融「exact 天花板」口径（出处已归档）；v1.3 为同脚本复现的
`adapter_exact_rate`（贪婪逐字，500 样本 328 命中）。两者词表不同（hans-en-24k-v1 vs zh-24k-v3），
且 `learned_adapter_quality=degraded`，**narration 头两代均走确定性模板兜底，不追逐字 exact**。

## productize 内置口径（16-stage learned-mode 指标，另一维度）

| 指标 | v1.2-1.8b | v1.3-800m |
|---|---|---|
| tool_lm dev/natural exact | 0.526 | **0.763** |
| tool_lm test/natural exact | 0.540 | **0.660** |
| tool_lm dev/structural exact | 0.500 | **0.602** |
| retrieval last_loss | 0.0064 | 0.0068（持平）|
| mw_disposition dev/accuracy | 0.022 | 0.149（均 degraded）|
| confidence valid AUROC | 0.804 | **0.994** |
| confidence sidecar test AUROC | 0.937 | **0.999** |
| narration adapter exact | 0.113 | **0.278** |

## release_eligible=false 原因（runtime 部署门，非能力门）

- `python_runtime_gate_v5` degraded：warm_decode 235.7 tok/s（阈值 280）+ retrieval calibration discard/expand 未验证。
- `browser_wasm_gate_v5` degraded：本机缺 node/wasm 工具链，`node_wasm_diagnostic_ok=False`、heap/吞吐全 None。
- 这两个 gate 是量化 runtime 部署门，**不影响五头能力跑分**。

## 结论

- **判别三头（retrieval / disposition / confidence）可产品化**：v1.3 持平或略优于 v1.2，均达可用水平。
- **tool_lm**：execute exact 0.7707 与 v1.2 持平，是 51.5M 架构容量天花板（与 CPT 规模无关）。
- **narration**：adapter exact 65.6%（高于 v1.2 的 43%），但 `learned_adapter_quality=degraded`，主路径仍走确定性模板引擎，模型头仅 paraphrase 兜底。
- **核心发现**：v1.3-800m（800M CPT）在五头任务上**不逊于甚至略优于** v1.2-1.8b（1800M CPT）——尤其是
  productize 内置口径下 tool_lm 生成 exact 0.526→0.763、confidence AUROC 0.937→0.999 大幅提升。
  归因方向：词表切换（hans-en-24k-v1）对工具调用/判别任务可能有利，且 800M CPT 已足够支撑 51.5M 架构的五头容量天花板。

## 评测出处

- 五头单头评测脚本 + 结果：`cycles/mei-1.3-51m/exp-00800m/head-eval/runs/`（`run_five_heads.sh`，float base 独立训单头 + holdout）
- productize 内置指标：`cycles/mei-1.3-51m/exp-00800m/runs/productize-v13-800m-adaptive-v5/`（16 stage receipt）
- v1.2 基线：`models/mei-1.2-51m/releases/exp-01800m/products/mei-1.2-51m-cpt1800m-tool-sft-cq2-v2-adaptive-v5/STATUS.md`
