# Quality gates

## 300M/600M lessons

- 600M Base loss 和 Float Task 指标改善，但合成 structure/colloquial 模板坍缩，
  因而 `corpus_reuse_eligible=false`。
- exact-unique 可由变化 ID 伪造；必须看规范化模板分布和 source-specific loss。
- no-match、rank>5、MW macro-F1、confidence ECE、learned narration 均说明
  语料统计不能冒充模型成功。
- 旧 MW 数据 20 类中 12 类缺 `candidate_tool` grounding，无法可靠原地修复。

## 最低证据

天然源需要 license + manifest hash；合成源需要 template audit、eval leakage=0
和具名语义评审；SFT 需要独立 family/split/eval lock、schema/grounding 与训练后
locked eval。任一 required receipt blocked 时默认 `retire`，不得自动滚入下一 cycle。
