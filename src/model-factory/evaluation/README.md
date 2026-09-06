# Evaluation

只读评估按问题拆分：`base/`、`tool_use/`、`heads/`、`resources/` 和 `alignment/`。
评估不得改写模型权重；正式分数必须同时绑定 eval lock、模型 artifact 和源码 manifest。

