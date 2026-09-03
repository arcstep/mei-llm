# Local state

这里存放不进入公开 Git 的本机大文件与可重建状态：

- `artifacts/`：按 model/cycle 归档的语料、权重、package 与 run；
- `cache/`：Rust/WASM 等可重建的构建缓存；
- `dist/`：本地发布导出物。

不要从本目录人工判断周期状态；使用 `cycles/mei-1.0-51m/` 的页面或统一 `mei` CLI。
移动或清理这里的内容前必须先通过 artifact registry 核对 URI、哈希与唯一性。
