# needle-zh / mei-1.0-58m

中文端侧工具调用模型。`needle-zh` 是历史目录名，正式族名为 `mei-1.0-58m`。

## 重要状态

- **当前实现**：Route-ID v1，可复现 legacy。
- **批准目标**：Needle2-aligned retrieval top-5 + byte grammar + 完整工具 JSON + MEI validator。
- **目标尚未实现**：当前 runner、pack、bank 和 promote 仍只消费 v1。
- **训练未授权**：不得因 target specs 已落盘而启动 CPT/SFT/QAT。

先读：

- [`DESIGN.md`](DESIGN.md)：当前/目标边界；
- [`spec/README.md`](spec/README.md)：machine specs 状态索引；
- `spec/*v2.target*`：只供后续实现，不是当前配置。

现有资产：

- `model/`：58m 主干与 Route-ID legacy；
- `train/` / `recipes/`：历史 Route-ID 数据/配方；
- `checkpoints/registry/`：immutable checkpoint lineage；
- `spec/`：v1 frozen 与 v2 target 分账。

禁止修改 v1 protocol/hash/promote 语义来伪装 v2，也禁止让现有脚本自动拾取 `*.target.*`。

产品与训练 SSOT 在私有 monorepo 另行维护；本目录保持公开仓自包含。
