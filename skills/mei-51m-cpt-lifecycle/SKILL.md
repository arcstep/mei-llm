---
name: mei-51m-cpt-lifecycle
description: >-
  Deprecated compatibility router for older requests that mention the single
  mei-1.0-51m lifecycle Skill. Use only to classify the request and hand it to
  cycle-orchestrator, corpus-sourcing, corpus-factory, corpus-quality,
  cpt-training, QAT, SFT alignment, model evaluation, runtime/release, or
  productization orchestration. Contains no training or corpus algorithm.
---

# 兼容路由：mei-1.0-51m lifecycle

本 Skill 已拆分，不再拥有执行逻辑：

- cycle 初始化、汇总、终态：`mei-51m-cycle-orchestrator`
- 天然池、下载、入池：`mei-51m-corpus-sourcing`
- 合成、编译、release：`mei-51m-corpus-factory`
- 语料审计与复用：`mei-51m-corpus-quality`
- CPT/start/resume/Base candidate：`mei-51m-cpt-training`
- QAT：`mei-51m-qat-training`
- SFT/head：`mei-51m-sft-alignment`
- 模型评测：`mei-51m-model-evaluation`
- package/SDK/资源/final audit：`mei-51m-runtime-release`
- 跨阶段只读编排：`mei-51m-productization`

若旧自动化必须经此入口运行：

```bash
python scripts/route.py --task cpt --action status -- --run-id <id>
```

新代码禁止依赖此路由。迁移说明见 `references/migration.md`。
