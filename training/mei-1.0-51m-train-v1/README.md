# mei-1.0-51m-train-v1

正式 pretrain / SFT trainer。从仓根执行，不依赖 notebook 实现。

- **Pretrain / scratch**：随机初始化；`NeedleZhConfig.from_spec()`；消费 `corpus/lm-v1` 四角色 `schedule-scratch.json`（quota_plan，300M 曝光）。
- **课程**：150M@512 → 100M@1024 → 50M@2048；token-based cosine LR，horizon=300M。
- **CPT exposure**：唯一模型始终是 51M；300M、600M、900M、1B、2B+ 或其他正整数都是累计 exposure。每个新语料包必须带 immutable manifest/hash、许可证登记、去重/污染证据和与累计目标唯一匹配的 `schedule-cpt-<rung>.json`。
- **v2 heads**：`from_target_v2()` 不进默认 LM pretrain。
- **长度**：scratch 课程从 512 起；CPT 固定 2048；runtime ordinary KV ≈256 是推理合同。

```bash
.venv/bin/python training/mei-1.0-51m-train-v1/train_pretrain.py --count-params
.venv/bin/python training/mei-1.0-51m-train-v1/train_pretrain.py --smoke
.venv/bin/python training/mei-1.0-51m-train-v1/check_pretrain_readiness.py --require-formal
.venv/bin/python training/mei-1.0-51m-train-v1/lifecycle_51m.py baseline
.venv/bin/python training/mei-1.0-51m-train-v1/lifecycle_51m.py init \
  --run-id <run-id> --corpus-dir <immutable-corpus-dir> --target-exposure 600000000
.venv/bin/python training/mei-1.0-51m-train-v1/lifecycle_51m.py plan --track cpt --run-id <run-id>
.venv/bin/python training/mei-1.0-51m-train-v1/lifecycle_51m.py resume --track cpt --run-id <run-id>
.venv/bin/python training/mei-1.0-51m-train-v1/lifecycle_51m.py status --run-id <run-id>
.venv/bin/python training/mei-1.0-51m-train-v1/lifecycle_51m.py verify

# CPT gate 通过后：注册候选与冻结建议均不修改 CURRENT.json
.venv/bin/python training/mei-1.0-51m-train-v1/base_candidate_51m.py register-base-candidate --run-id <run-id>
.venv/bin/python training/mei-1.0-51m-train-v1/base_candidate_51m.py propose-freeze --candidate base/<candidate-id>
```

完整阶段顺序和失败分支以 `recipes/cpt-lifecycle-v1.json` 为机器真源。Run 只写
`training/runs/mei-1.0-51m/<run-id>/`；receipt 的 fingerprint 一致才复用，输入变化会使该阶段及下游失效。
`--track cpt` 只运行至 CPT gate；产品化 track 与 base lineage 分离。`architecture_sha256`
仅保留源码 provenance，checkpoint 兼容性由 weight contract 判断；runtime profile 与 training aux 各自独立哈希。
CPT/QAT/SFT 读取各自 optimizer/sampler checkpoint，lock-v2 读取完整 shard receipt。全流程强制离线并在每个阶段校验
`CURRENT.json` 未变化。lock 分数未过门仍继续资源基线，由 hard-gate 写出 `eligible=false` 的冻结建议；除非用户明确下令冻结，`CURRENT.sft/runtime` 必须保持 `null`。
