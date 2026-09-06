# Contracts

- `PIPELINES.json`：允许被选择的正式流水线及状态。
- `CODE_CATALOG.json`：训练/评估二级域、正式入口及兼容入口的用途和可用状态。
- `LEGACY_PATH_MAP.json`：旧扁平 training 的 131 项映射，并兼容中间阶段曾出现的
  `.internal/src/mei_llm/training` 路径。
- `pipeline-lock.schema.json`：每个 cycle 绑定源码与 run 证据的合同。
- `sft_*_contract_51m.py`：仍被当前实现消费的可执行 SFT 合同；名称保留用于历史 receipt 对照。
