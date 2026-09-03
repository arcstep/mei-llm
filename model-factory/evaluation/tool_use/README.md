# Tool-use evaluation

这里评估 retrieval、full-call、参数填写、Agent 与联合预算行为。

- `freeze_longitudinal_eval_v7_51m.py`：当前跨 rung 锁定评测集；
- `longitudinal_eval_metrics_51m.py`：可比指标定义；
- `sft_v3_eval_51m.py`：仍被当前流水线调用的历史协议评估器；
- `validate_product_chain_51m.py`：产品链完整性；
- `eval_lock_v2_51m.py`、`freeze_longitudinal_eval_51m.py`：历史/兼容锁实现。

文件后缀是评测协议身份，不是按数字自动选最新版；正式选择见 pipeline lock。
