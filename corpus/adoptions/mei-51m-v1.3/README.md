# v1.3：本次简英训练准备

2026-09-14 20:04（北京时间）。新一代CPT输入已冻结并完成A10校验，正式训练已开始。用户授权连续完成约25亿token，阶段评估不阻塞训练。准备文件名中的 `mei-v12-*` 是纠正前 ID，保留用于追溯；不能据此归为历史 v1.2 已训练语料。

## 当前选择与锁定状态

- **词表已投入本次训练**：[简英 24K release](../../../models/mei-1.2-51m/tokenizer/candidates/mei-24k-lossless-hans-en-20260914-v1/RELEASE.json)。仅本次新模型采用，历史模型和CURRENT不改。
- **CPT 已锁定**：[25亿输入 release](../../pools/frozen/2026-09-14-mei-51m-v1.3-cpt-25b-r01/RELEASE.json)，选中原文编码2,550,204,074 token，计划有效曝光2,500,001,792 token。三个阶段新增800,000,000／800,000,000／900,001,792 token；21个细来源全部进入三个阶段。旧打包失败证据保留，不能代替这个明确的新release。
- **QAT 待迁移**：旧约 1 亿 token 引用储备仍绑定旧 CPT 和词表，不能直接给新链使用。
- **SFT 五头已锁定（2026-09-17）**：锁定到日期命名新 release [`mei-1.3-51m-tool-sft-20260917-fivehead-v1`](../../sft-suite/mei-1.3-51m-tool-sft-20260917-fivehead-v1/)（tool_lm/retrieval 公开线 4k；confidence/disposition 克隆自 skeleton-v2 样本；narration freeze 合成）。锁定相对位置见 adoption.json 的 `current_locked_inputs.sft`，文件哈希见 release 内 manifest.json。

因此 [adoption.json](adoption.json) 的 `current_locked_inputs` 已显式绑定：CPT（确切 release+哈希）与 SFT（`mei-1.3-51m-tool-sft-20260917-fivehead-v1`）已锁定，QAT／Eval 仍未新增锁定。不能通过“取最大 vN”或“找到 RELEASE.json”自动继承。

## 当前 CPT 五类初选量

- 基础语言：1,505,552,382／1,530,000,000 token。
- 口语：328,580,893／360,000,000 token。
- 文学与剧本：51,625,118／50,000,000 token。
- 代码／结构：314,429,499／300,000,000 token。
- 任务／工具：350,016,182／350,000,000 token。

上述是选中原文的领域量，不等于全部参与loss的曝光。新的训练曝光及各阶段领域、语言、细来源区间已单独核对；选中未曝光部分保留为储备，不重复填额。繁体专项保留为储备，混合来源原文仍保真。

## 哪些旧成果不再是当前输入

[旧含繁体 CPT v4](../../pools/frozen-history/2026-09-14-mei-51m-v1.3-cpt-inputs-r04/RELEASE.json) 的约 27.645 亿冻结记录保留，但已被简英方案取代。v1/v2/v3 是其前序产物。它们都属于这次新训准备过程的历史包，不是旧 v1.2 的实际训练语料。

## 下一步

CUDA及阶段边界恢复、MLX快照加载已通过小批机制验证；全量上传、远端哈希验收及正式开训均已完成。按800M／1600M／约2500M累计观察点持续训练并独立评估。QAT后续显式重绑，SFT／Eval未准入材料保持候选。训练实际进度以A10运行证据为准。

返回[语料入口](../../README.md)。
