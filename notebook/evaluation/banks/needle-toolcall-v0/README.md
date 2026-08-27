# needle-toolcall-v0

共享工具调用评测库。订阅方不限于 `needle-zh`：任何端侧路由器都可以跑同一份 exact-match 题。

- 机器面：`eval-bank-v0.jsonl`
- 工具 schema：`notebook/evaluation/shared/toolsets/needle-home-v0.json`、`needle-invoice-v0.json`
- 判定：`pass=exact_match`。槽值必须整词相等；禁止用 query 子串当城市/房间。
- 离题：`function_calls` 必须为 `[]`。不要求模型输出自然语言解释。
- 本库 **禁止** 进入任何 task 的 train seed。隔离脚本会扫 `EVAL-NTC-*`。

评分：

```bash
python3 notebook/_tooling/scripts/eval_needle_toolcall_v0.py
python3 notebook/_tooling/scripts/eval_needle_toolcall_v0.py --predictions path/to/preds.jsonl
```
