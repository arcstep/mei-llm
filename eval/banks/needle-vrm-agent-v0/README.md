# needle-vrm-agent-v0

家居数字人闭集工具评测。订阅方不限于 `needle-zh`。画面 playground 只回放金标，**不是** judge。

- 机器面：`eval-bank-v0.jsonl`（公开 48 题 `split=dev` + holdout `split=eval`；hash 见 `holdout-v1.lock.json`）
- 工具 schema：`eval/shared/toolsets/needle-vrm-agent-v0.json`（16 技能，槽为 enum）
- 判定：`pass=exact_match`。槽值必须整词/枚举相等。
- 可选 `scene`：中文短状态。评分与演示须拼进前缀（`场景：…。用户：…`）。金标依赖场景时禁止靠画面推断。
- 离题 / 缺槽 / 场景冲突 / 非法搭配：`function_calls` 必须为 `[]`。
- 本库 **禁止** 进入任何 task 的 train seed。隔离扫 `EVAL-NVA-*` 与 `query` 正文。
- 英文对照（**不是** 产品 KPI）：`eval/banks/needle-vrm-agent-en-v0/`，给官方 Needle 2 / 英文 Qwen 用。

与 `needle-toolcall-v0`（天气/灯/发票协议烟测）分开，KPI 不混。

可见回放：`eval/playground/vrm-agent-v0/`（从 `eval/` 起 HTTP 根）。

```bash
python3 scripts/eval_needle_toolcall_v0.py --bank eval/banks/needle-vrm-agent-v0/eval-bank-v0.jsonl
python3 scripts/check_train_eval_isolation.py --all
# Qwen 行为基线（不是 needle-zh 代理）；产物在 experiments/runs/，含 exact-match 与时延
python3 scripts/run_eval_needle_qwen_v0.py --backend ollama --model qwen3.5:0.8b-mlx
python3 scripts/eval_needle_toolcall_v0.py \
  --bank eval/banks/needle-vrm-agent-v0/eval-bank-v0.jsonl \
  --predictions experiments/runs/<run>/predictions.jsonl
```
