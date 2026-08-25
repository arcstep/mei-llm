# eval（共享评测面）

评测题卡按 **bank** 存放，不按模型复制。训练任务在 `tasks/index.json` 里订阅 bank。

```text
eval/
  shared/toolsets/     # 多 bank / 多 task 共用的工具 schema
  banks/
    mei-expert-v0/     # 现有 MEI 专家行为题
    needle-toolcall-v0/# 端侧工具调用 exact-match（中英协议烟测）
    needle-vrm-agent-v0/# 家居数字人闭集 exact-match（中文产品 EVAL）
    needle-vrm-agent-en-v0/# 英文对照（官方 Needle 2；非 KPI）
    needle-pretrain-probes-v0/
```

约定：

- 评测 `item_id` 使用 `EVAL-…` 前缀；train 不得出现这些 id 或题面。
- 新模型默认先订阅已有 bank，再开自己的 bank。
- runner 产物写 `experiments/runs/`，不写进 bank 目录。Qwen 闭集路由基线：`scripts/run_eval_needle_qwen_v0.py`（行为基线，不是 needle-zh 代理；`summary.json` 含 exact-match 与时延）。
- 家居数字人可见回放：`eval/playground/vrm-agent-v0/`（金标回放，不是 judge）。
