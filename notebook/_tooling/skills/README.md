# mei-llm-owned Cursor Skills

本目录是 training / corpus skills 的发布真源：

- `meilang-corpus-distill`

Skill 自包含工作流与辅助脚本，但不捆绑外部语料树、文档树或 sibling workspace。`corpus_root`、`tools_root`、`source_roots` 和 release/catalog 版本必须由调用方显式传入。

仓群运行面由 `mei-env` 的 Cursor skill sync 工具投影。
