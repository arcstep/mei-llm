# needle-vrm-mw-v0

独立 **MW 治理** 验证集。**不是**一期 KPI，不替代 `needle-vrm-agent-v0` v1/v2。

| 文件 | 角色 |
|---|---|
| `eval-bank-v0.jsonl` | 1000 条（100 dev + 900 eval） |
| `holdout-mw-v0.recipe.json` | 配额与生成合同 |
| `holdout-mw-v0.lock.json` | 冻结 hash |
| `eval-bank-v0.review.jsonl` | 协议审查样本（隔离扫描会跳过 `.review.`） |

Gold 是五类动作：`execute / expand / shape / escalate / stop`。可观察 cell 仅 `MW.OK | SF.PAR | SF.AMB | SF.PRE | Unknown`。

评分：`scripts/eval_needle_mw_v0.py`。一期 `execute/[]` 预测必须走 `--projection phase1`：`[]` 只记 `non-execute`，**禁止**把所有空调用算成 `stop`。

```bash
cd mei-llm
PYTHONPATH=scripts python3 notebook/_tooling/scripts/validate_needle_mw_holdout_v0.py
PYTHONPATH=scripts python3 notebook/_tooling/scripts/eval_needle_mw_v0.py --strategy always_stop --split eval
PYTHONPATH=scripts python3 notebook/_tooling/scripts/eval_needle_mw_v0.py --strategy always_refuse --split eval --projection phase1
```
