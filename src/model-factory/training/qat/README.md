# QAT training

这里保存量化感知训练的数值实现。当前产品主线是 CQ2；Q4 用于诊断、回退和 parity，不能
因文件名更短或版本号不同而自动成为主线。

- `qat_cq2_v2_51m.py`：当前 CQ2 QAT 协调器；
- `cq2_qat_51m.py`、`cq2_policy_51m.py`、`quant_ops_51m.py`：CQ2 数值 worker 与合同；
- `qat_cq2_replay_51m.py`：既有 replay 合同；
- `qat_replay_51m.py`、`qat_51m.py`：Q4/历史兼容训练部件；
- `check_qat_pilot_readiness.py`：pilot 门禁。

正式采用哪个实现只由 `contracts/PIPELINES.json`、recipe 与 cycle pipeline lock 共同决定。
