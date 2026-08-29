# mei-1.0-58m-train-v1

正式 pretrain / SFT trainer。从仓根执行，不依赖 notebook 实现。

- **Pretrain / scratch**：随机初始化；`NeedleZhConfig.from_spec()`；消费 `corpus/lm-v1` 四角色 `schedule-scratch.json`（quota_plan，300M 曝光）。
- **课程**：150M@512 → 100M@1024 → 50M@2048；token-based cosine LR，horizon=300M。
- **CPT 1B**：父节点 `base/mei-1.0-58m-base-scratch300m-v1`；消费 `corpus/lm-v2` + `schedule-cpt-1b.json`；Adam 连续；wiki/HQ 从真实 cursor 续抽，fresh structure/口语从 0；段 LR `3e-5→1e-5`。独立服务 `cpt_service.py`。归档旧 CPT `schedule.json` 仍拒绝。
- **v2 heads**：`from_target_v2()` 不进默认 LM pretrain。
- **长度**：scratch 课程从 512 起；CPT 固定 2048；runtime ordinary KV ≈256 是推理合同。

```bash
.venv/bin/python training/mei-1.0-58m-train-v1/train_pretrain.py --count-params
.venv/bin/python training/mei-1.0-58m-train-v1/train_pretrain.py --smoke
.venv/bin/python training/mei-1.0-58m-train-v1/check_pretrain_readiness.py --require-formal
.venv/bin/python training/mei-1.0-58m-train-v1/check_cpt_readiness.py --require-formal
.venv/bin/python training/mei-1.0-58m-train-v1/train_pretrain.py --rung cpt-smoke
.venv/bin/python training/mei-1.0-58m-train-v1/cpt_service.py start --rung cpt-5m --compile
.venv/bin/python training/mei-1.0-58m-train-v1/cpt_service.py start --rung 1b --compile
.venv/bin/python training/mei-1.0-58m-train-v1/cpt_service.py status
.venv/bin/python training/mei-1.0-58m-train-v1/cpt_service.py pause
.venv/bin/python training/mei-1.0-58m-train-v1/cpt_service.py resume
.venv/bin/python training/mei-1.0-58m-train-v1/cpt_service.py tail
.venv/bin/python training/mei-1.0-58m-train-v1/promote_cpt1b_base.py
.venv/bin/python training/mei-1.0-58m-train-v1/train_sft.py --pack <pack.jsonl> --smoke
```

`--smoke` 从冻结 `tokens/*.bin` 截取少量 mmap 窗口。Run 输出到 `training/runs/<run-id>/`。smoke / pilot / cpt-smoke / cpt-5m 写 `NOT_FOR_PROMOTE.json`。正式 1B 写入 `training/runs/pretrain-1b-cpt-from-scratch300m/`，不得覆盖 300M scratch run。
