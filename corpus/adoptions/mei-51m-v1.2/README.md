# v1.2：1800M 基线采用入口

这里优先对应 `mei-51m-v1.2-1800m-base` 及其下游声明绑定，不假定所有同目录实验使用同一套语料。

## CPT 实际绑定

- [1800M Base release](../../../models/mei-1.2-51m/releases/exp-01800m/base/mei-1.2-51m-base-cpt1800m-v1/RELEASE.json)。
- [源 run](../../../cycles/mei-1.2-51m/exp-01800m/runs/mei-1.2-51m/cpt-1800m-v1-c03/RUN.json) 明确引用 [1800M layout](../../pools/zh-v2-pool/layouts/zh-v2-layout-exp-001800m-v1-v1/RELEASE.json)。
- 实际累计曝光为 1,800,001,024 token，最后一段新增 299,998,208。最后一段 layout 不能被当作整条 18 亿血缘；前段须继续沿 parent 查询。
- Base release 的语料质量状态是 `not_audited`。历史训练使用与当前质量准入分开。

## QAT／SFT／Eval 的声明输入

- QAT：[binding](../../../cycles/mei-1.2-51m/exp-01800m/runs/mei-1.2-51m/qat-cq2-v1/binding.json) 的 replay 引用旧 [300M layout](../../pools/zh-v2-pool/layouts/zh-v2-layout-300m-v1/manifest.json)。
- SFT bootstrap：[binding](../../../cycles/mei-1.2-51m/exp-01800m/runs/mei-1.2-51m/sft-bootstrap-v4/binding.json) 分别绑定旧链的工具 SFT 主包、语言增补、解说包和 longitudinal Eval v7。
- Adaptive：[binding](../../../cycles/mei-1.2-51m/exp-01800m/runs/mei-1.2-51m/sft-adaptive-v5/binding.json) 声明上述独立材料，并引用 bootstrap run；实际 adaptive 阶段还会派生可见视图。

2026-09-14 核对的 10 个上述下游输入引用均存在且元数据 SHA256 与 binding 一致。此检查不包含所有载荷、消费行数、所有重试和后续整改 run，不能把这页当作每次 SFT 的完整训练清单；具体 refs 与检查结果见 [adoption.json](adoption.json)。

`corpus/sft-suite/...teacher-v4`、`eval-lock/...v11` 的存在，不能推翻某个 run 明确记录的主包和 v7 绑定。不同 run 的采用关系必须分开列。

这些是历史 v1.2 的绑定，不自动成为 v1.3 输入。返回[语料入口](../../README.md)。
