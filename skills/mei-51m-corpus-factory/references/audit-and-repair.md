# 审计与原地修复 vs 替换

## 用途

任何"旧语料要不要修/要不要换"的决策，先跑审计拿证据；任何扩量，先跑 audit
门。审计报告落新 release 的 `governance/` 子目录（先例：
`sft-zh-rebuild-v2/governance/old-mw-corpus-audit.json`）。

## 审计门（audit funcs 已内置于 build.py，逐项核对输出）

1. **grounding 存在率**（最重要的质量门）：按 reason_code/场景档位统计
   `candidate_tool` 非 null 率、`retrieved_tools` 覆盖、evidence/permissions/
   state 是否空壳。任何类 100% 无工具级 ground truth = 不可修复，只能替换。
2. **dedup**：exact 行级 + raw query 层分开报；近似用 char-trigram Jaccard
   ≥0.92。**undesigned exact-dup**（不是刻意保留的重复）是 bug 信号，
   不是可接受噪声；扩量后必须重测。
3. **leakage**：`scan_leakage` 检查新行与同 release 其他 family、与已发布
   release、与 eval-lock 的文本/工具泄漏。泄漏必须为零。
4. **budget**：`fit_batch_to_budget` 后无超 2048 行；prompt 各 profile
   （compact/standard）不越软上限。
5. **schema 合法性**：full_call gold_args 过完整 schema `validate_instance`；
   工具对象过 portable projection 后能通过 schema_subset。
6. **label 完整性**：MW 行 `raw_reason_code`/`reason_code` 都在 0-19；
   class-0 且 target 不在 visible batch 的行已按 batch-visibility 规则重标
   为 10，且 raw 保留；confidence 行 label=null 且
   label_state="pending_actual_final_runtime_outcome"。
7. **模板多样性**（hash 之外）：归一化后模板/句式计数、固定句覆盖率、语义槽
   连贯性、train/valid 是否共享同一模板骨架。hash 干净 ≠ 自然多样。

## 原地修复可行性判据

| 缺陷类型 | 前提 | 结论 |
|---|---|---|
| label 偏差 / 漏 relabel | grounding 列齐全（candidate_tool 非空率 >0 且可见批信息可重建） | 可原地修复：重放 batch-visibility 规则，产物进 supersede 新 ID |
| 缺 grounding / 列全 null | 无工具级 ground truth 可重建 | 不可修复 → 审计报告 + replace |
| 模板退化（句式单一） | 模板计数够低但 grounding 好 | 扩变体层（text_variants/旋转）后按新 ID 重新铸币，不静默扩旧行 |
| 双句号/标点/文本瑕疵 | 行级 | 生成器修后重跑（幂等 rerun 同 ID 仅限未持久化场景；已持久化走新 ID） |

## 先例（旧 MW 语料审计 → 替换决策）

- 对象：`mei-1.0-51m-tool-sft-v4-300m-v4` 的 mw-disposition.train.jsonl
  （13,763 行）。
- 发现：20 类中 **12 类 `candidate_tool` 100% null**，包括
  `capability_insufficient` 自己（0/375）；缺工具级 ground truth →
  batch-visibility 重标注类"原地修复"没有可执行输入。
- 结论：replace 方向成立；顺带暴露 v1 替换料 mw 140/类偏薄（旧料 294-3263/类），
  v2 扩到 500/类。
- 反例提醒：旧语料行数多（13,753）不等于质量好——行数与可用性分开报告，
  禁止用行数反驳审计结论。

## 报告格式（governance JSON）

schema 至少含：被审计 release id + 文件、按类统计表（行数/candidate_tool
非空率/retrievable 率）、undesigned dup 率（raw 与行级）、budget/leakage
结论、每类"可修复 / 不可修复"判定与理由、决策（repair_in_place /
replace / retire）、审计日期与审计人（脚本+版本）。报告写完后才动产物。
