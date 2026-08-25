# mei-llm 布局

共享资产在仓根；每条可训练模型是 `tasks/<id>/` 下的一条 task。

```text
corpora/                 # 共享语料槽（大 shard 不入库）
eval/
  shared/toolsets/       # 多 bank / 多 task 共用的工具 schema
  banks/<bank-id>/       # 评测题；禁止拷进 train
tasks/
  index.json             # 登记：seed、banks、corpora
  <task-id>/             # 配方、seed 覆盖、mlx adapter
scripts/                 # runner；默认路径走 repo_paths.py
experiments/runs/        # 跑分产物（gitignore）
```

约定：

1. task 订阅 corpora 与 eval banks，不要按模型复制题库。
2. `EVAL-*` 与评测题面不得出现在任何 train seed。门禁：`scripts/check_train_eval_isolation.py --all`。
3. 新模型线 = 新 `tasks/<id>/` + 登记一行。能力没变就复用已有 bank。

现行 task：`mei-expert-qwen35-0p8b`、`needle-zh`。
