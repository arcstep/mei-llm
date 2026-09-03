# Python SDK

在仓库根目录设置 `PYTHONPATH=platform/python-sdk` 后：

```python
from mei_sdk import Engine
engine = Engine.load("platform/_shared/fixtures/packages/tiny-protocol-v1")
engine.register_tools([
    {"name": "clock.read", "parameters": {"type": "object", "properties": {}}}
])
session = engine.create_session()
turn = session.complete({
    "wire_version": "mei-runtime-wire-v2",
    "query": "现在几点？",
    "candidate_text": '[{"name":"clock.read","arguments":{}}]',
})
session.submit_tool_result({
    "wire_version": "mei-runtime-wire-v2",
    "call_id": turn["call"]["call_id"],
    "status": "ok", "payload": {"hour": 12},
    "provenance": {"source": "host_executor", "verified": True},
})
```

绑定优先级：

1. 协议实现（本包，始终可用）
2. `mei_sdk.ffi.NativeEngine` — 包装同一 `libmei_sdk`
3. `mei_sdk.reference` — 显式调用 51M Python+MLX 数值 oracle

`platform/_shared/runtime` 是 retrieval / grammar / provenance / KV / narration 语义真源；数值对照以 MLX oracle 为准，CQ2 与 wire 对照以 `platform/_shared/spec/golden/` 为准。

`mei-model-package-v1` 和 `mei-runtime-wire-v1` 仅通过只读 degraded adapter 接受，不能声明完整 v2 capability。

完整 v2 包必须用 `files[]` 枚举除 `mei-model.json` 外的每个 payload，并闭环文件大小、SHA、训练 receipt 与 MW reason-code codebook；任何未声明文件、symlink、越界/重叠 tensor 或非 canonical ready head 都会 fail closed。MW head 固定为 20 类 reason code，只有 class 0 同时满足 `p>=0.70`、top1/top2 margin `>=0.15` 才映射为 `continue`，其余映射为 `stop`；`constrain` 只能来自带显式 `allowed_tools` 的确定性策略。

51M 身份不是参数量加总：loader 同时固定三类 contract SHA、完整 architecture/runtime/tokenizer 常量，以及有序 400 个 LM tensor 的 name/shape；27 个 `attn_gate` 标量使用合法的 `shape=[]`。任意额外/缺失/乱序 tensor 或 MTP 残留都不能取得完整身份。

`candidate` / `release` 包拒绝 `decode_mode=raw`。`raw` 和 `candidate_text` 仅用于 experimental 协议/数值对照，正式 MLX 推理必须加载全哈希验证、完整 head/capability 的 CQ2 v2 包。

Python+MLX 只报告数值/协议诊断推理，不冒充端侧资格。`resources` 中的自报峰值不产生 `resource_eligible`；在锁定并验证 measurement receipt 之前，`product_ready` 与 `release_eligible` 保持 fail closed。
