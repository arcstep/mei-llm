# CPT training

这里实现 Base 的 scratch/continued pretraining、续训状态和语料门禁。新 cycle 的唯一正式
入口是 `orchestration.lifecycle`；本目录文件均为被该状态机调用的 worker 或 gate，不能
绕过 lifecycle 直接形成正式 lineage。

- `train_pretrain.py`：数值训练 worker；
- `cpt_service.py`：运行/恢复服务；
- `check_*readiness.py`、`*_gates.py`：启动前与完成后门禁；
- `audit_cpt_synthetic_diversity_51m.py`：增量语料多样性审计；
- `run_scratch_curriculum*.py`：保留的 scratch curriculum 实现，不代表当前 cycle 选择。

累计 exposure、parent、delta corpus 与源码锁最终以 cycle 的 `CYCLE.json` 和
`pipeline/PIPELINE.lock.json` 为准。
