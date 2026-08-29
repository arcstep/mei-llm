# 模型包

- 微型协议包（门禁用）：`fixtures/packages/tiny-protocol-v1/`
- 51M Q4 量化包（端侧推理；WASM 只加载此格式，拒绝 float npz）：`mei-1.0-51m-base-scratch300m-q4-v1/`
- 升格 base 清单（指向仓内 tokenizer + `pretrain-300m-scratch.npz`）：`mei-1.0-58m-base-scratch300m-v1/`

四个 head 必须写全。当前 51M Q4 盘：`lm=ready`（量化词预测），`contrastive` / `mw_disposition` / `confidence` 仍可能是 `missing`，直到检索/置信头写入 `heads.npz`。
