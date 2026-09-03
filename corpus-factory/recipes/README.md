# Recipes

Recipe 描述可重复的数据生产参数，不保存一次性结果。每次执行必须由 cycle 的
`corpus_plan_id` 引用，并把工厂 Git revision、输入来源、输出 Merkle root、质量证据和
人工审查写入该 cycle 的 artifact/receipt。
