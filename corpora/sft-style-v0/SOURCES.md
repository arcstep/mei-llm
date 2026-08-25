# SOURCES · sft-style-v0

| 源 | 许可 | 使用范围 |
|----|------|----------|
| [thu-coai/CrossWOZ](https://github.com/thu-coai/CrossWOZ) `data/train.json` | Apache-2.0 | 仅 train；dev/test 永不读取 |
| [thu-coai/KdConv](https://github.com/thu-coai/KdConv) train 对话 | 仓 Apache-2.0 | 仅 train；剔除 travel/电话/KG 事实；不作 gold |

禁止：把公开对话直接写成 `home-sft-*.jsonl` 训练行；沿用原始标注当 needle-zh answers。
