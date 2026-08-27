# needle-vrm-agent-v0

家居数字人闭集工具评测。订阅方不限于 `needle-zh`。画面 playground 只回放金标，**不是** judge。

## 版本

| 版本 | 文件 | 角色 | 条数 |
|---|---|---|---|
| **v1** | `eval-bank-v0.jsonl` + `holdout-v1.lock.json` | **历史冻结回归集**。字节级不可改。 | 212（48 dev + 164 eval） |
| **v2** | `eval-bank-v2.jsonl` + `holdout-v2.lock.json` | **当前 2K KPI**。配额见 `holdout-v2.recipe.json`。 | 2000（200 dev + 1800 eval） |

Runner **必须**显式传 `--bank`，禁止按目录顺序猜题库。默认 Qwen runner 仍指向 v1，跑当前 KPI 时改传 v2。

- 工具 schema：`notebook/evaluation/shared/toolsets/needle-vrm-agent-v0.json`（16 技能，槽为 enum；v2 不扩工具）
- 判定：`pass=exact_match`。槽值必须整词/枚举相等。总分之外必须分别报告 execute / refuse；恒 `[]` 不能当高分。
- 可选 `scene`：中文短状态。评分与演示须拼进前缀（`场景：…。用户：…`）。
- 离题 / 缺槽 / 场景冲突 / 非法搭配：整条 `function_calls` 必须为 `[]`。
- `family=sequence`：金答顺序等于语句执行顺序。
- 本库 **禁止** 进入任何 task 的 train seed。隔离扫 `EVAL-NVA-*` / `EVAL-NVA2-*` 与 `query` 正文。
- 英文对照（**不是** 产品 KPI）：`notebook/evaluation/banks/needle-vrm-agent-en-v0/`。

与 `needle-toolcall-v0`（天气/灯/发票协议烟测）分开，KPI 不混。

可见回放：`notebook/evaluation/playground/vrm-agent-v0/`（从 `eval/` 起 HTTP 根）。

```bash
# v1 历史回归
python3 notebook/_tooling/scripts/eval_needle_toolcall_v0.py --bank notebook/evaluation/banks/needle-vrm-agent-v0/eval-bank-v0.jsonl
python3 notebook/_tooling/scripts/run_eval_needle_qwen_v0.py --backend ollama --model qwen3.5:0.8b-mlx

# v2 当前 2K KPI（必须显式 --bank）
python3 notebook/_tooling/scripts/validate_needle_vrm_holdout_v2.py
python3 notebook/_tooling/scripts/eval_needle_toolcall_v0.py --bank notebook/evaluation/banks/needle-vrm-agent-v0/eval-bank-v2.jsonl
python3 notebook/_tooling/scripts/eval_needle_zh_families.py --bank notebook/evaluation/banks/needle-vrm-agent-v0/eval-bank-v2.jsonl --split eval --smoke
python3 notebook/_tooling/scripts/run_eval_needle_qwen_v0.py \
  --bank notebook/evaluation/banks/needle-vrm-agent-v0/eval-bank-v2.jsonl \
  --backend ollama --model qwen3.5:0.8b-mlx
python3 notebook/_tooling/scripts/check_train_eval_isolation.py --all
```
