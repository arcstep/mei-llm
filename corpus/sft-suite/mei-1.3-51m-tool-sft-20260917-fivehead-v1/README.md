# mei-1.3-51m-tool-sft-20260917-fivehead-v1

v1.3-800m 五头 SFT 语料的统一锁定版本（2026-09-17 日期命名）。

## 定位

这是 v1.3 SFT 的单一真相源语料。把 9/15 公开线提取合成的 tool_lm/retrieval 4k，与
9/16 三头评测闭环产出的 confidence/disposition/narration 成果，物理克隆为一份**自包含、
独立管理、可讨论**的新 release。彻底切断与 skeleton-v2（Mei 147 工具、zh-24k-v3 词表时代）
的命名和绑定关系——以后只说这个版本名，不再翻 cycles 探索目录，也不再引用 skeleton-v2。

## 五头语料清单

| 头 | 任务形态 | 语料源 | 规模 | 形态 |
|---|---|---|---|---|
| tool_lm | 生成（工具调用）| 公开线 fullcall（9663 → 采纳 4000）| train 4000 + holdout 300 | 公开工具调用原始 jsonl |
| retrieval | 判别（选工具）| 公开线 retrieval（12755 → 采纳 4000）| train 4000 + holdout 200 | 公开检索原始 jsonl |
| confidence | 判别（二分类）| skeleton-v2 confidence 样本 + harvest 实跑 label | train 824 + valid 126（harvest 729/110）| 输入样本 + label 靠实跑 |
| disposition | 判别（18 类 MW 偏离）| skeleton-v2 mw_disposition 样本 | train 8367 / valid 1416 / dev 989 / test 1028 | 20 类原始样本，训练用 18 类口径 |
| narration | 生成（设备解说）| freeze v2 合成（20 family 设备控制）| 代码合成 | freeze 脚本，训练时合成 |

## 已知跑分（按各头语料口径，勿混）

| 头 | 跑分 | 口径说明 |
|---|---|---|
| tool_lm | 工具名选对 88.7%、参数 exact **32%** | 公开 4k 大目录口径；参数 exact 32% 是 51.5M 容量瓶颈，非语料多样性问题 |
| retrieval | recall@5 63-64.5% | 公开大目录口径 |
| confidence | AUROC **0.9807** / AUPRC 0.984 | harvest 729/110，三头最佳 |
| disposition | 18 类 macro_f1 **0.944** | 移出 class 0/10（检索终态归 retrieval 头）|
| narration | sidecar exact 天花板 43% | 拍板确定性模板兜底，不追逐字 |

> ⚠️ **口径区分**：skeleton-v2 的 Mei 147 小目录（full_call 1643 / retrieval 2096）曾跑出
> tool_lm execute exact 77.1%、retrieval recall@1 95.7%——那是**旧语料**（Mei 147 工具、
> 精造 counterfactual）的跑分，**不适用于本 release**。本 release 的 tool_lm/retrieval 是
> 公开线 4k（通用工具、更大目录），跑分见上表。

## 词表与编译契约

- v1.3 训练词表 = `hans-en-24k-v1`（24000 unigram，`frozen_in_training_use`）。
- **tool_lm / retrieval**：公开线原始语义 jsonl，直接用 v1.3 词表 encode。
- **confidence / disposition**：克隆自 skeleton-v2（zh-24k-v3 时代 compiled），训练前需用
  hans-en-24k-v1 重新 encode。
- **confidence label**：label 语义 = `exact_call_or_correct_refusal_after_r1`，只能 harvest
  实跑产生（检索头 + tool_lm 能力实跑 → 精确比对），不能离线标注。本目录 `train.jsonl`/
  `valid.jsonl` 是输入样本（含 expected_kind/expected_call），label 由 harvest 补全；已跑通的
  harvest 成果见 `harvest-train-receipt.json`（729 条）/ `harvest-valid-receipt.json`（110 条）。
- **disposition 口径**：样本是 20 类原始，训练时用 `--exclude-retrieval-terminal` 移出
  class 0/10（检索终态），只训/评 18 类 MW 偏离。
- **narration**：`freeze_narration_sft_release_v2_51m.py` 的 `semantic_cases(index)` 生成
  20 个设备控制语义 family（温度/湿度/启停/报错/取消/亮度/门锁/风扇/计时器/场景/音乐/
  no-op/partial/多步），target 由 `NarrationProvider.narrate` 确定性生成。主路径走模板引擎
  兜底，模型头只作 paraphrase 兜底。

## 治理

- `governance_status: adopted`，`locked: true`，**锁定后禁止就地修改**。
- 每头语料的来源证据（探索报告）见 `STATUS.json` 的 `evidence` 字段。
- 文件哈希清单见 `manifest.json`。

## 来源目录（只读，勿从那里改）

- 公开线：`corpus/pools/task-trials/2026-09-15-public-sft-scale-experiment/data/`
- 三头样本/脚本：`corpus/sft-suite/mei-1.0-51m-tool-sft-v5-rebuild-300mv2-skeleton-v2/`
  （confidence、mw_disposition 样本）、`src/model-factory/release/freeze_narration_sft_release_v2_51m.py`
- 探索报告：`cycles/mei-1.2-51m/exp-corpus-survey/runs/SFT-CORPUS-STATUS-SUMMARY.md`、
  `cycles/mei-1.2-51m/exp-corpus-survey/runs/2026-09-16-three-head-loop/`
