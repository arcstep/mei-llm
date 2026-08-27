# mei-1.0-58m-train-v1

正式 pretrain / SFT trainer。从仓根执行，不依赖 notebook 实现。

- **Pretrain / scratch**：随机初始化；`NeedleZhConfig.from_spec()`；消费 `corpus/lm-v1` 四角色 `schedule-scratch.json`（quota_plan，300M 曝光，口语与 structure 完整消费）。
- **课程**：150M@512 → 100M@1024 → 50M@2048；token-based cosine LR，horizon=300M。
- **CPT**：已归档到 `notebook/archive/plans/20260827-cpt-after-300m/`，正式入口拒绝 `schedule.json` 与任何 parent checkpoint。
- **v2 heads**：`from_target_v2()` 不进默认 LM pretrain。
- **长度**：默认课程从 512 起；位置上限 2048；runtime ordinary KV ≈256 是推理合同。

```bash
.venv/bin/python training/mei-1.0-58m-train-v1/train_pretrain.py --count-params
.venv/bin/python training/mei-1.0-58m-train-v1/train_pretrain.py --smoke
.venv/bin/python training/mei-1.0-58m-train-v1/check_pretrain_readiness.py --require-formal
.venv/bin/python training/mei-1.0-58m-train-v1/train_pretrain.py --rung pilot-1m
.venv/bin/python training/mei-1.0-58m-train-v1/run_scratch_curriculum.py --pilot-5m
.venv/bin/python training/mei-1.0-58m-train-v1/run_scratch_curriculum.py
.venv/bin/python training/mei-1.0-58m-train-v1/train_sft.py --pack <pack.jsonl> --smoke
```

`--smoke` 从冻结 `tokens/*.bin` 截取少量 mmap 窗口。Run 输出到 `training/runs/<run-id>/`。smoke / pilot 写 `NOT_FOR_PROMOTE.json`。
