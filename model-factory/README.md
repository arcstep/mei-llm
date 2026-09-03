# 模型工厂

这里保存 **当前可用的训练、评估、编排与发布方法**。它与 `models/`（模型定义）、
`corpus-factory/`（数据生产）、`cycles/`（每轮事实）和 `platform/`（当前推理能力）并列，
是 mei-51m 生命周期中不可替代的一等成果。

模型工厂不是模型权重目录，也不是历史 run 目录：

- 正式模型二进制在 [`models/mei-1.0-51m/releases/`](../models/mei-1.0-51m/releases/)；
- 300M、600M 每轮实际用了什么代码，由各 cycle 的 `pipeline/PIPELINE.lock.json` 固化；
- 大型 checkpoint、日志和逐步 receipt 在 Gitignored `.local/artifacts/`；
- 本目录只维护一套当前实现，旧实现通过 Git 和 cycle source capture 恢复。

## 十个入口各自回答什么

| 目录 | 作用 | 是否可作为新 cycle 正式入口 |
|---|---|---|
| `contracts/` | 数据、SFT、pipeline lock 的机器合同 | 是 |
| `training/` | CPT、QAT、工具调用与各功能头的权重训练 | 是 |
| `evaluation/` | Base、工具调用、各 head、资源和 Needle2 对齐评估 | 是 |
| `orchestration/` | 生命周期状态机、阶段 DAG、恢复和无人值守衔接 | 是 |
| `release/` | Base 注册、不可变冻结、CQ2/Q4 打包与最终审计 | 是 |
| `recipes/` | 不改算法时可版本化调整的训练/门禁参数 | 是 |
| `common/` | 上述模块共享的 checkpoint、路径、锁和身份实现 | 仅被调用 |
| `tests/` | 模型工厂和仓库合同的回归门禁 | 仅验证 |
| `diagnostics/` | benchmark、scan、A/B 等一次性诊断 | 否 |
| `compatibility/` | 为历史 receipt 保留的旧入口和说明 | 否 |

因此，日常不会从一百多个脚本中猜入口：先看
[`FACTORY.json`](FACTORY.json) 和 [`contracts/PIPELINES.json`](contracts/PIPELINES.json)，
再通过统一 CLI 或已登记的模块启动。

## 当前正式流水线

- CPT / 任意累计 exposure：`python -m orchestration.lifecycle_51m`
- adaptive-v5 产品化：`python -m orchestration.productize_adaptive_v5_51m`
- 300M/600M 同合同配对监督：`python -m orchestration.supervise_paired_adaptive_v5_51m`

推荐从仓根统一进入：

```bash
PYTHONPATH=src .venv/bin/python -m mei_llm cycle list
PYTHONPATH=src .venv/bin/python -m mei_llm cycle show exp-000600m
```

流水线文件名中的 `v3`、`v4`、`v5` 是既有合同/阶段协议名，不等于“哪个模型更新”。
现有名称暂不做无证据改名，以免切断历史 receipt；它们是否仍可用于新运行，以
[`contracts/CODE_CATALOG.json`](contracts/CODE_CATALOG.json) 为准。

## 一个 cycle 如何绑定代码

每个已执行 cycle 必须有：

```text
cycles/<model>/<cycle>/pipeline/
├── PIPELINE.md                 人能读懂的本轮方法说明
├── PIPELINE.lock.json          阶段、入口、源码证据与 run 指纹
├── SOURCE_RECOVERY.json        后半链历史源码字节审计
└── SOURCE_RECOVERY.prefix.json 前半链历史源码字节审计
```

`PIPELINE.lock.json` 只描述已经发生的事实。旧运行若当时没有提交源码或冻结完整 source
bundle，就必须标记 `source_capture_mode=reconstructed` 和 `exact_reproducible=false`；
不能拿今天的源码冒充当时源码。未来正式运行必须在启动前同时固化 Git revision、脏树
patch/source bundle、全量 source manifest，以及每个 stage 实际引用的源码闭包。

## 源码保全规则

1. `.internal/` 不得保存唯一训练、评估或编排实现。
2. 新 cycle 只能使用 `current` 流水线；`diagnostic_only` 和 `retired` 代码不能进入正式 lineage。
3. 当前实现由 Git 演进；每轮事实由 cycle lock 和不可变 source capture 固化，二者缺一不可。
4. 旧脚本只有在确认没有唯一逻辑且对应历史字节可恢复后才可删除。
5. 权重、语料、源码和评测是四类独立资产，任何一类都不能由另外三类“推测重建”。

迁移前隐藏源码另有只读恢复包，详情见
[`compatibility/PRE_REFACTOR_SOURCE_SNAPSHOT.json`](compatibility/PRE_REFACTOR_SOURCE_SNAPSHOT.json)。

