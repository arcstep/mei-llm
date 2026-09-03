# Head evaluation

这里分别评估功能头，禁止把不同目标混成一个“多头准确率”。

- `compare_portable_heads_51m.py`：retrieval、MW、confidence 等 portable head 对照；
- `evaluate_narration_adapter_51m.py`：learned narration 与 deterministic delivered
  correctness 分开报告。

每个 head 的 tensor、数据、指标和 release eligibility 必须独立出现在 cycle scorecard。
