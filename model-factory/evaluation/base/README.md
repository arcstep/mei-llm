# Base evaluation

评估未产品化 Base 的语言损失和固定 probes，不训练权重。

- `freeze_float_baseline_51m.py`：冻结 Float Base-LM anchor/control；
- `pretrain_probes.py`：预训练能力 probes。

输出只有在绑定 Base hash、eval lock 与源码 manifest 后，才能进入 cycle scorecard。
