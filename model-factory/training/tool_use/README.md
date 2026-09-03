# Tool-use training

这里实现主干上的工具检索、full-call、Agent continuation 和联合上下文预算对齐。

- `adaptive_tool_context_51m.py`：固定五工具分批扫描、阈值与联合预算视图；
- `sft_v3_training_51m.py`：仍被 adaptive-v5 复用的训练原语；`v3` 是历史协议名，
  不是当前产品版本；
- `preflight_sft_v3_51m.py`：该协议的输入门禁；
- `train_sft*.py`：通用/历史 worker，不能单独宣称得到完整产品；
- `train_sft_ondisk_51m.py`：大数据集按盘读取 worker。

新 cycle 必须从 `orchestration.productize_adaptive_v5_51m` 发起；不得在本目录按最大版本号
猜测入口。
