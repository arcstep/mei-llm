# vrm-agent-v0 playground

金标回放。**HUD JSON 是真源**；画面只是把 `function_calls` 演出来。眼睛「看起来对」不算过评测。

不入库 VRM。可选加载公开金样（需联网），失败则用胶囊人。

## 启动

在 `mei-llm/eval/` 下起静态服务（这样题库 JSONL 与 playground 同域）：

```bash
cd mei-llm/eval
python3 -m http.server 8765
```

打开：<http://127.0.0.1:8765/playground/vrm-agent-v0/>

可选 query：`?vrm=<https://...vrm>` 覆盖模型 URL。默认尝试 pixiv three-vrm 示例模型；失败自动降级。

three / `@pixiv/three-vrm` 走 CDN，不从 `mei-lang/stock` 拷贝。

## 不会做的事

- 不跑 needle-zh / Qwen
- 不把 playground 当 judge
- 不提交 `.vrm` 二进制
