# needle-zh / mei-1.0-51m

中文端侧工具调用模型。`needle-zh` 是历史目录名，正式族名为 `mei-1.0-51m`。

## 重要状态

- **Legacy**：Route-ID v1 可复现；`promote` / pack / bank 仍只消费 v1。
- **v2 runtime**：Needle2-aligned retrieval top-5 + byte grammar + 完整工具 JSON + MEI validator 已接线并通过分层测试。
- **不是发布模型**：正式 retrieval / SFT / confidence 训练未开始；独立 CPT 在正式 qwen-plus 口语源（≥30M unique + 质量/隔离/盲评）齐备前 fail-closed。离线合成口语（`zh-pretrain-colloquial-synth-v1`）只是工程对照：其 38.1M 是含重复 `frame_id` 的暴露量，不能打开 CPT。正式口语源是 `notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen-plus/`。
- **Parent**：仅 `mei-1.0-51m-base-cpt300m-v1`。禁止用 Route-ID SFT checkpoint 当 v2 parent。

先读：

- [`DESIGN.md`](DESIGN.md)：legacy / v2 边界；
- [`spec/README.md`](spec/README.md)：machine specs 状态索引。

验证：

```bash
python3 notebook/_tooling/model/mei-1.0-51m/check_student.py
python3 notebook/_tooling/scripts/test_mei_route_runtime.py
python3 notebook/_tooling/scripts/test_mei_v2_runtime.py
python3 notebook/_tooling/scripts/check_train_eval_isolation.py --scope cpt-v2
python3 notebook/_tooling/scripts/test_colloquial_synth_pipeline.py
python3 notebook/_tooling/scripts/build_zh_pretrain_colloquial_synth.py --smoke
python3 notebook/_tooling/scripts/produce_colloquial_qwen.py --smoke
python3 notebook/_tooling/scripts/run_cpt_v2_staircase.py
```

现有资产：

- `model/`：51m 主干、Route-ID legacy、v2 runtime；
- `train/` / `recipes/`：历史 Route-ID 数据/配方；
- `checkpoints/registry/`：immutable checkpoint lineage；
- `spec/`：v1 frozen 与 v2 contract。

产品与训练方法 SSOT 在私有 monorepo 另行维护；本目录保持公开仓自包含。
