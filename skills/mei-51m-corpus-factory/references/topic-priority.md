# 主题树账本与选树规则

## 选树规则

下一棵树 = **度量最坏的能力** × **验证路径最便宜的**。扩叶投资之前先确认
瓶颈在语料（叶密度/grounding），而不是训练后校准或行为层——后者扩语料无效。

## 现状账本（截至 sft-zh-rebuild-v2 + MW 诊断）

| 树 | 状态 | 证据 | 下一步建议 |
|---|---|---|---|
| mw_disposition | **已培育**：v2 11,800 行（train 8,320 / valid 1,465 / dev 1,007 / test 1,008）；20 类全 grounded | 诊断（raw base 600 步）：dev 0.6975/0.3403，test 0.7075/0.3464；旧管线 reference 0.226/0.030093 | 可收尾转向下一棵树；test 仍 <0.8 属正常（raw-base 上限未知），正式 SFT 后才能下结论 |
| retrieval（no-match / rank>5） | **下一候选**：旧基线 no-match false-selection 0.85、rank>5 gold retention 0.167（旧 bank 仅 6-7 行，统计无意义）；v8-v2 bank 已补齐 dev/test 各 ~190 行 | 待跑便宜诊断（ContrastiveHead 探针 + oracle ranking，见 diagnostic-validation.md） | 扩叶方向：no-match offtopic/near-miss 区分、rank 6-20 深位保留、cross-batch exhausted、stop-before-scan、hard-negative discrimination（现档位 220-550/类） |
| full_call / agent | 基线尚可；v2 未变（full_call 2,340 / agent 1,080） | 老语料已具 schema grounding | 暂不扩叶；等 retrieval/MW 上正式 SFT 后的行为度量 |
| confidence | 低优先扩叶：v2 1,150 行 label=null，等真实 runtime harvest | ECE 0.193 是旧管线数字 | 瓶颈大概率在 post-SFT 校准/适配头训练，不在叶密度 |
| narration | 低优先扩叶：v2 900 行只吃 verified ToolResult | learned-acceptance 0.018（旧） | 同上，post-SFT 行为层优先 |
| schema_generalization / 跨语言-BFCL 层 | deferred：未重建（从旧 v4 release reuse 决策待定） | — | 等上面树出正式 SFT 度量后再定 reuse/replace/retire |

## 扩一棵树的决策门（完成一轮后自问）

1. 诊断/度量到达发布门了吗？到 → 登记、选下一棵；
2. 没到，且证据指向叶密度（模板数、grounding 空率、类内分散度）→ 投资树叶
   （diversity/边界场景），**不**无限加同类模板；
3. 没到，且证据指向训练后环节（calibration、alignment 顺序）→ 停语料投资，
   写 `pending_sft` 等正式管线结果；
4. 度量缺行数支撑（如 bank <100 行/档）→ 先扩 eval bank，不先扩训练料。

## 红线

- 每轮结束把新数字写回本账本与 CORPUS.md（追加式）；
- 未跑正式 SFT 前，所有模型质量指标保持 `pending_sft` 措辞，诊断数字只作
  方向性证据；
- 真实度量数字（如 0.85 / 0.167 / 0.030093）是旧正式管线在 CQ2+aligned
  检查点上的值——引用时带限定，别当 raw-base 可比基线。
