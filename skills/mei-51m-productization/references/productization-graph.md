# Productization graph

## Current phase graph

`QAT → SFT bootstrap → SFT adaptive → model evaluation → runtime/release`

每个箭头传递 artifact URI/path、manifest/receipt SHA 与上游 run fingerprint，
不传“默认 300M 路径”。pipeline entrypoint 与参数映射来自 hash-bound recipe，
所以代码或管线优化会产生新 binding，而不会静默污染旧 run。

模型评测与 runtime/release 在同一个 adaptive run 上 resume：前者停在
`sidecar_runtime_eval_v5`，后者只复用 receipts 后执行 package、Python、
Browser-WASM 和 final audit。SFT scope 跳过 adaptive generation/sidecar；
evaluation/runtime scope 对 scope 外 action 使用“只能复用、禁止执行”门禁。
LM/head/package 不得伪装重训。
