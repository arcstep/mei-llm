# 树干设计：主题 → 全光谱 taxonomy

## 用途

把"我想给模型补某个场景主题"的讨论，转成可执行的树干：先跨 CPT/MW/SFT 环节
全光谱分类展开（讨论层有意分类），再落每环节的 family、场景档位、配额与 gates。
参考实现：`corpus-factory/generators/rebuild_zh_v1/build.py` 的 targets dict、
`mw_scenarios.py` 的 20 类 BUILDERS、`retrieval.py` 的档位拆分。

## 树干（顶层分类发生在设计层，不在产物层）

对每个新主题依次过四个问题，答案就是树的"树枝"：

1. **CPT 树枝**：这个主题是否暴露需要诊断的 CPT 表征缺口？天然池盘点与
   candidate mix 交给 `mei-51m-corpus-sourcing`；只有质量诊断和预注册 trigger
   允许时，本 Skill 才生成有上限的合成 gap-fill。cycle 绑定交回
   `mei-51m-cycle-orchestrator`。
2. **MW 树枝**：主题会不会触发哪些 MW 场景/误判？对应哪些 reason_code 类？
   哪些类需要新增场景 builder（新代码），哪些类已有 builder 只需扩量？
3. **SFT 树枝**：主题在六个 sft_capability family 里各长什么样——
   - retrieval（no-match / rank1-20 placement / cross-batch scanning /
     stop-before-scan / hard-negative discrimination）；
   - full_call（required_only / optional / boundary / unit conversion / refusal）；
   - agent（单步直答 / 先查后做 / 失败换路 / 状态变更多步）；
   - mw_disposition（batch-visibility relabel 规则下 20 类）；confidence
     （label=null，等真实 runtime harvest）；narration（只消费 verified
     ToolResults）。
4. **eval 树枝**：主题由哪个 eval-lock bank 度量？行数够不够统计（对照
   v7 先例：no-match 仅 60 行、rank>5 仅 6-7 行 → v8 补齐各 ~190 行）？
   缺则必须先扩 eval bank 再谈训练门。

## 配额写法（机器可读、数字落地）

- 每个 family 的档位写成 dict 常量，key = 场景档位，value = 目标行数：
  ```python
  RETRIEVAL_TARGETS = {
      "rank_1_5": 450, "rank_6_10": 400, "rank_11_15": 220, "rank_16_20": 220,
      "no_match": 550, "cross_batch_exhausted": 220, "stop_before_scan": 420,
      "hard_negative_discrimination": 280,  # 合计 2,760
  }
  ```
- 每档带 gates：no-match ≤5% false-selection、rank>5 gold retention ≥99%、
  batch size 固定 5、2048-token joint budget、MW 20 类非泄漏。
- 配额依据写进 PLAN.json/讨论纪要：来自哪条真实度量（gap 数字或 registry
  计数），禁止"感觉差不多"。
- 每档目标行数要能支撑评估：dev+test 至少 ~100-200 行/档位才有统计意义，
  从全量里经 `stratified_sample`/`cf_group` 切出，不与训练行共享模板骨架。

## 现成锚点（改前先读）

- 六 family 现状：retrieval 2,760 / full_call 2,340 / agent 1,080 /
  mw_disposition 11,800 / confidence 1,150 / narration 900（v2 release）。
- MW 类清单：20 个 reason_code，`mw_scenarios.py` 的 `BUILDERS` keyed 按类；
  `SCAN_STOP_CLASSES` 是 terminal 语义，`NEEDS_SIBLING = {ambiguous_scope,
  mixed_intent}`。
- 分桶隔离：`common.assign_split`（sha256 分桶）+ `cf_group`，eval-lock 拷贝
  dev/test 时用同一机制，保证泄漏为零且阈值调参可用 dev。

## 树冠检查单（展开完自查）

- [ ] 每个"树枝"都落到具体文件/函数或明确的"暂不做，登记为 deferred"；
- [ ] 各环节产物互不混 schema（retrieval 行不吃 full_call schema，MW 不吃
      retrieval 的 label 语义）；
- [ ] 每个配额数字都能追溯到 gap 度量或 registry 行数；
- [ ] 该主题的 eval 覆盖够统计，且已冻结（`locked_test_used_for_threshold_tuning=false`）；
- [ ] 讨论产物（树干图/配额表）落成机器可读 dict + PLAN 段，而不是只存在于
      对话里。
