# runtime/_shared

跨运行时语义的唯一真源，禁止在模型目录或各语言 SDK 复制一套不同规则：

- `schema_subset.py`：可移植 JSON Schema 子集及 fail-closed 注册校验；
- `byte_grammar.py` / `grammar.py` / `decode.py`：UTF-8 constrained decoding；
- `provenance.py`：schema → provenance → permission → state → MW → confidence 门禁；
- `tool_index.py`：完整 provenance fingerprint、normalized-f16 index、稳定 top-5；
- `kv_manager.py`：1024-token stable sink + 256-token ordinary ring；
- `narration.py`：只消费已验证结果的确定性中文解说；
- `canonical_json.py`：wire-v2 递归 UTF-8 键序 canonical JSON，供 fingerprint、call ID 和 trust receipt 共用；
- `schema_render.py` / `normalizers.py`：历史 v1 训练序列化和确定性归一化。

Python+MLX 是数值 oracle；Python、Rust 和浏览器必须以相同 golden 验证这些语义。

MW 的 learned 输出是 20 类 `reason_code`，不是直接生成三态 disposition：仅校准后的 class 0 可投影为 `continue`，其他类别均 fail closed 为 `stop`；只有确定性策略能在同时给出非空 `allowed_tools` 时产生 `constrain`。Narration 只读取 Session 已验收的原始 `ToolResultV2`，相同 `call_id` 不能替换 payload。

wire-v2 的 canonical JSON 对对象键递归排序、数组保序，并拒绝 NaN/Inf、非字符串对象键和 lone surrogate；历史 `mei-schema-serializer-v1` 不被原地改写。

冻结的 `zh-24k-v1` 没有 SentencePiece byte fallback。grammar 使用有状态 dummy-prefix transducer 并禁用 `unk_id`，因此 Unicode 候选验证是完整的，但模型生成只覆盖该冻结词表可表达的字符；不能把它表述为任意 Unicode 生成能力。
