# 任务准备与转换试批

批次命名采用 YYYY-MM-DD-用途-rNN；r 为产物修订，与模型代际无关。旧 release ID 保留，旧路径由迁移表解析。

当前SFT语义样板入口：[2026-09-14-qualified-sft-specimens-r03/REVIEW.md](2026-09-14-qualified-sft-specimens-r03/REVIEW.md)。30个来源任务组、36个视图；LM36、检索正配对35、处置36、解说10，confidence待模型。属于开发语义样板，未晋升生产训练包。r01／r02为同轮开发记录，不作为当前入口。

- [2026-09-14-mei-51m-v1.3-task-preparation-r01](2026-09-14-mei-51m-v1.3-task-preparation-r01/)
- [2026-09-14-mei-51m-v1.3-task-preparation-r02](2026-09-14-mei-51m-v1.3-task-preparation-r02/)
- [2026-09-14-mei-51m-v1.3-task-preparation-r03](2026-09-14-mei-51m-v1.3-task-preparation-r03/)
- [2026-09-14-mei-51m-v1.3-task-preparation-r04](2026-09-14-mei-51m-v1.3-task-preparation-r04/)
- [2026-09-14-mei-51m-v1.3-task-preparation-r05](2026-09-14-mei-51m-v1.3-task-preparation-r05/)
- [2026-09-14-mei-51m-v1.3-task-preparation-r06](2026-09-14-mei-51m-v1.3-task-preparation-r06/)
- [2026-09-14-mei-node-task-preparation-r02](2026-09-14-mei-node-task-preparation-r02/)
- [2026-09-14-mei-public-five-head-trial-r02](2026-09-14-mei-public-five-head-trial-r02/)
- [2026-09-14-public-sft-source-collection-r01](2026-09-14-public-sft-source-collection-r01/)
- [2026-09-14-public-sft-multihead-conversion-r05](2026-09-14-public-sft-multihead-conversion-r05/)：MOSS、API-Bank、CrossWOZ、RiSAWOZ 各100组／视图的五头候选转换与评估。
- [2026-09-14-sft-candidate-triage-r01](2026-09-14-sft-candidate-triage-r01/)：按五头准备规则重整六来源2,487个视图、22个细类，保留逐条证据索引与核查项。

返回[语料入口](../../README.md)。

## 2026-09-14 公开SFT千组试验（当前）

- [原文转换报告](2026-09-14-public-sft-1000-r03/REPORT.json)：MOSS600＋ToolACE400，共1,000原始对话组，2,045视图。
- [逐例语义审核](2026-09-14-public-sft-1000-review-r02/REVIEW.md)：审核63视图，44个语义与结构、预览预算均通过的样板。生产准入0，confidence待模型。
- [语义样板JSONL](2026-09-14-public-sft-1000-review-r02/semantic-specimens.jsonl)。所有新目录均为v1.3储备；转换r01/r02及审核r01是同批过程证据，不额外计容量。新44组与旧30组不直接相加为独立量。
