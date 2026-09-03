# Runtime

这里仅保留当前 Runtime 实现。历史 Route-ID 与早期 Needle2 目录不再复制到活跃树，
需要时从 Git 历史或迁移后的只读 artifact 取证。

- [`runtime/`](runtime/) 是语言无关的当前语义真源：schema projection、联合上下文预算、
  constrained decode、KV、检索、provenance 与终态 narration。
- Python/MLX 和 Browser-WASM 产品入口分别位于 [`../python-sdk/`](../python-sdk/) 与
  [`../browser-sdk/`](../browser-sdk/)；Rust core 仅作为 WASM
  内核依赖，不表示独立 Rust SDK 已进入当前交付范围。
- `CURRENT.runtime` 只有在 Runtime 与正式权重组成可部署 release 后才允许赋值；本次迁移未修改它。
