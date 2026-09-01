# mei-llm 布局

仓根是正式模型产品主路径。`notebook/` 只承担外围准备与治理。

```text
CURRENT.json
tokenizer/zh-24k-v1/                    # 正式词表
corpus/lm-v1/                            # 已接收语料
architecture/mei-1.0-51m-arch-v1/      # 现行 51M 主干 + 几何合同
training/mei-1.0-51m-train-v1/          # 正式 trainer / data / recipes
training/runs/                           # 正式训练状态与中间 checkpoint
runtime/mei-1.0-51m-route-v1/           # frozen Route-ID runtime
runtime/mei-1.0-51m-needle2-v2/         # 历史 Python+MLX 参考实现（非 SDK 产品名）
sdk/                                     # 实验性 MEI Runtime 嵌入式 SDK（先协议/ABI，后拆仓）
base/                                    # immutable pretrain/CPT 权重
sft/                                     # 正式 SFT 权重
notebook/                                # 语料生产、研究、测试、验证、评测、归档
```

约定：

1. **Pretrain** = 随机初始化；**CPT** = 从已有 base 续训。二者发布目标都是 `base/`，不是 `cpt/`。
2. 正式训练从 `training/` 执行，只消费根 `tokenizer/` `corpus/` `architecture/`。不得把 trainer 实现留在 notebook。
3. `CURRENT.json` 是人读入口。禁止再维护根目录 `corpora/index.json` 或 `tasks/` 双入口。
4. 已发布 `corpus/lm-v1/` 是冻结消费面：只保留发布合同与 `tokens/*.bin`。加工态（raw、idx、审计、游标、账本）在 `notebook/corpus/lm-v1/`。口语 30.1M mixed-fleet 包已准入内部从零预训练（`roles_complete=true`，`public_distribution_clearance_asserted=false`）。从零四角色混训只使用 `schedule-scratch.json`（`sampler=quota_plan`，300M 精确配额，512→1024→2048 课程）。旧 300M 后续 CPT 计划在 `notebook/archive/plans/20260827-cpt-after-300m/`。
5. Needle 主干与 runtime 是正式成果：`architecture/` 与 `runtime/`。`notebook/_tooling/model` 仅为兼容 symlink。
6. `CURRENT.runtime` 未与正式权重组成可部署 release 前可为 `null`；这不否定历史参考实现，也不等于 `sdk/` 已是产品 Runtime。
7. 隔离门禁：`notebook/_tooling/scripts/check_train_eval_isolation.py --scope cpt-v2|sft-v2`。
8. 现行产品只登记 `mei-1.0-51m`；immutable base 是 `mei-1.0-51m-base-scratch300m-v1`。51M runtime/heads 与 0.8B 专家线只作迁移参考或外部实验。
9. 嵌入式 SDK 入口是 `sdk/`。公共名是 MEI Runtime；`CURRENT.runtime=null` 期间只标 experimental。
10. 场景目标是工具调用后基于已验证结果生成用户可读解说。NarrationProvider 后端无关、无执行权限；任何 51M tool/narration 后训使用新 model ID，禁止覆盖 base。当前 tied LM head 不等于已具备 head-only/adapter 能力。
11. PTQ 扫描是诊断，不是产品 bit-map。QAT 强制。短 QAT pilot 入口在 `training/mei-1.0-51m-train-v1/check_qat_pilot_readiness.py`，未实现 STE 前禁止开训。Float/PTQ 账本在 `notebook/evaluation/jobs/mei-1.0-51m/`。
