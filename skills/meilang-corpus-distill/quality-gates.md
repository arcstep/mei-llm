# 质量门

语料流水线的发现与起草可半自动，结构与验证必须严格；自动化弱处诚实标注 manual review。

## 人工 review 门

1. **真实任务**：不是答案型假 prompt。
2. **最小 context**：够完成任务但未泄漏完整解法。
3. **真源锚定**：output 可回溯源码或 runtime truth。
4. **Verifier 可信**：checks 可执行，或明确 manual review。

任一不满足，停止发布。

## 机械门

使用调用方 `tools_root` 中的 corpus validate、verifier、split-index 和 family coverage 工具，不在 Skill 内另造产品语料校验逻辑。

管理可见性：

```bash
python3 skills/meilang-corpus-distill/scripts/corpus_stats.py --corpus-root /path/to/corpus
python3 skills/meilang-corpus-distill/scripts/family_snapshot.py \
  --corpus-root /path/to/corpus --family-id <family_id>
```

## Build 专项门

优先 compiler 或 patch 真值。片段型 output 无法独立 compile 时，改用 `source_trace` / patch 断言或 manual review，禁止虚假声称 `compile_pass`。

## Access 专项门

必须 grounded 于 runtime truth，并说明 query state 与 metric/dataset 真源；不能猜源码。

## Docs 专项门

断言须有明确 source files，output 紧凑、任务导向，避免大段复制。

## 何时标注 manual review

- verifier 只能是 heuristic。
- 任务跨过多文件或 mode。
- runtime replay 不可用。
- sample 使用尚未锁定的模式。

允许 manual review，不允许假确定性。

## Batch 专项门

1. 每行有稳定 family 锚点。
2. 每行 lane 明确。
3. 每行有真实 `source_paths`。
4. 不以 transcript-only 任务为主。
5. 不 overfit 局部 workaround。
6. promote 前 spot-check 至少 3 条，并覆盖每个主导 `task_type` 的 verifier 计划。

## Publish-ready 定义

仅当 family/task type 明确、source paths 存在、context 最小、output grounded、verifier 可信、validate 通过，并且 verifiers 通过或剩余步骤明确 manual 时，才是 publish-ready。dry-run 只证明结构与路由合法。
