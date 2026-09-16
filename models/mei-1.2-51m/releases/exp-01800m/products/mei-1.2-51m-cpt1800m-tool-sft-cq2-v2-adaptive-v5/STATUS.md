# STATUS —— mei-1.2-51m-cpt1800m-tool-sft-cq2-v2-adaptive-v5

当前 **SFT 对比基线**（唯一五头完整、可作 SFT 对照的产物）。

## 血统

- base：`mei-1.2-51m-base-cpt1800m-v1`（51.5M 参数、1,800,001,024 token、zh-24k-v3 词表）
- 流程：phase-binding 产品化主线 `adaptive-v5`（16-stage，package passed，7 个 stage degraded）
- 状态：`release_class=candidate`（非 final release）
- 产物：11 文件 + ASSETS.json（大二进制 tensors.bin 18MB / tokenizer.model 591KB gitignore，仓外备份）

## 五头跑分（口径 = 各自语料 holdout 评测，非产品端到端）

| head | 任务 | 指标 | 值 | 口径 |
|---|---|---|---|---|
| retrieval | 判别（选工具）| recall@1 | **95.7%** | Mei 147 工具目录 |
| retrieval | 判别（选工具）| recall@5 | 63–64.5% | 公开 4 批大目录 |
| tool_lm | 生成（工具调用）| execute exact | **77.1%** | Mei 147 工具目录 |
| tool_lm | 生成（工具调用）| 工具名 / 参数 exact | 87–89% / 32% | 公开大目录 |
| disposition | 判别（18 类 MW 偏离）| macro_f1 | **0.944** | 移出检索终态后 18 类 |
| confidence | 判别（二分类）| AUROC | **0.9807** | skeleton-v2 harvest |
| narration | 生成（设备解说）| exact 天花板 | 43% | sidecar rank 消融，饱和于 rank-128 |

## 能不能用 / 怎么用

- **判别三头（retrieval / disposition / confidence）可产品化**：指标达可用水平，sidecar 架构适配。
- **tool_lm**：工具选择 + execute 可用（Mei 147 口径），参数精确填充（exact 32%）是 51.5M 容量瓶颈，非语料能解。
- **narration**：**主路径走确定性模板引擎（NarrationProvider），模型头只作 paraphrase 兜底**，不追求逐字 exact（sidecar 天花板 43%）。runtime 使用时应调模板而非依赖模型生成。

## 报告出处

- 三头（disposition/confidence/narration）：`cycles/mei-1.2-51m/exp-corpus-survey/runs/2026-09-16-three-head-loop/`
- 公开语料（retrieval/tool_lm）：`cycles/mei-1.2-51m/exp-corpus-survey/runs/2026-09-15-public-sft-scale-experiment/`
- 语料状态总表：`cycles/mei-1.2-51m/exp-corpus-survey/runs/SFT-CORPUS-STATUS-SUMMARY.md`
