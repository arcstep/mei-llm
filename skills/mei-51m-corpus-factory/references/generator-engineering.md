# 生成器工程约定

## 模块布局（rebuild_zh_v1，新增主题先进这里）

- `common.py`：共享基建——ToolRegistry（`load_deploy_tools` 147/14、
  `load_training_tools` 211/22，运行时加载不硬编码）、`sha256_*`、
  `canonical_json`、`count_tokens`（真实 ZhTokenizerV1，字符回退）、
  `fit_batch_to_budget`（JOINT_BUDGET=2048、FIXED_PROMPT_RESERVE=200、
  BATCH_SIZE=5、COMPACT_CAP=1024、STANDARD_CAP=1536）、`assign_split` +
  `cf_group`（sha256 分桶）、`case_id`、`DedupIndex`（exact +
  char-trigram Jaccard ≥0.92）、`scan_leakage`、`simulate_tool_result`/
  `seeded_value`（确定性 host 模拟器）、`stratified_sample`。
- `text_variants.py`：中文 NL 变化，`build_variant` + `VARIANT_KINDS` 7 类
  （synonym/typo/colloquial/ellipsis/coreference/mixed_zh_en/…）。纯 seeded、
  无外部调用。
- `mw_scenarios.py`：20 类 per-class 场景 builder → {query, context, evidence,
  history, permissions, state}，只 grounded 在真实 tool description/schema +
  邻居线索。`BUILDERS`、`SCAN_STOP_CLASSES`、`NEEDS_SIBLING`、
  `_base_query`（rstrip 尾句号）。
- 每个 family 一个文件（retrieval.py/fullcall.py/agent.py/mw.py/confidence.py/
  narration.py），`build.py` 编排 targets + audit + gate + write + manifest。

## 确定性规则（可复跑是审计的前提）

- 所有随机走 seed（常为 case_id / 主题名），同一个 seed 必须产出逐字节相同行。
- 工具选择要旋转：`_rotate(tools, seed, n)` 按 seed 轮转，禁止写死
  `tools[0], tools[1]`（先例：固定槽位造成 93% undesigned exact-dup）。
- seed 转整数用 `int(hashlib.sha256(seed.encode()).hexdigest()[:6], 16)`；
  禁止 `int(seed[-2:], 16)`（先例：seed 尾字符是 `ic` 时 ValueError）。
- 文本变体可堆叠两个独立 seed（先例：`_diversify()` 把 raw dedup 从 ~39%
  压到 ~16.3%），但堆叠后必须重跑正式 audit，不许只看头几十行。

## Schema 与 grounding 规则

- 工具对象过平台校验（`platform/_shared/runtime/schema_subset.py`）前必须
  投影成 portable wire 形状：只留 `name/description/parameters`（+permission/
  state keys）；registry 的 provenance 元数据（family/fingerprint/similar_to）
  会触发 UnsupportedSchemaError。例：
  ```python
  catalog = [{"name": t["name"], "description": t["description"],
              "parameters": t["parameters"]} for t in deploy.tools]
  ```
- 场景内容禁止脱离工具真实 description/schema 编造；full_call 的 gold_args
  必须过 `C.validate_instance` 对完整 schema 验证。
- 引用工具 description 作 query 素材时，先 `.rstrip("。").rstrip()`，防止
  拼接出 `。。。。`（先例：MW 批量双句号）。
- 确定性转换（unit ℃↔℉、中文数字、边界值）要从候选池选带目标字段的工具
  （如 `_tools_with_temperature_field`），命中率不够先查池子，不悄悄放宽
  "可跳过"分支（先例：命中率 166/250 → 197/250 是靠池子修好的）。

## 语言与预算

- 中文 NL 默认 简体，可混简短英术语（mixed_zh_en 类变体允许 code/API 名）。
- 每条语料的 prompt 组装走 `fit_batch_to_budget`，输出留 128 reserve；
  断言硬上限 2048，不许用字符数手估 token。

## 常见的"叶子腐烂"模式（写生成器时对照）

1. 模板不随 seed 变（同一条字符串 × N seed → 扩量即精确重复）；
2. 固定工具槽位/固定目标工具名；
3. 尾句号拼接、空 candidate_tool、simulate 结果非空但未标记 verified；
4. `int('ic', 16)` 类脆 seed 解析；
5. 中文提示以逗号/空格结尾导致 next-token 训练目标退化；
6. 跨 family 复制行结构但 label/语义字段张冠李戴（泄漏测试抓不到时靠
   schema 审计抓）。
