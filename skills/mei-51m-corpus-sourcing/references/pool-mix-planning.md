# CPT 天然池盘点与每轮 mix 规划

## 角色与池子（v2 现役，词表 zh-24k-v3）

- **v2 天然角色**（来自注册表 `corpus-factory/sources/source_registry.json`）：
  fineweb2_hq / wiki_zh / wiki_en / dialogue / structured / code。
  池实物：`artifacts/mei-1.0-51m/pools/zh-v2-pool/`（raw → admitted → pools），
  权威锚点 = 各 admitted manifest v2 + 池 release `zh-v2-pool-natural-v4`。
- **旧 v1 角色**（wiki/fineweb2_hq/structure/colloquial，词表 zh-24k-v1）只作
  回归证据，不进入新链消费。structure/colloquial 合成角色已 retire，其语义位置
  由 v2 的 structured/dialogue 天然角色顶替（不复活合成）。
- **余额推导式**：`剩余(role) = 池 release tokens_by_role − Σ(各轮 cpt.json quotas[role])`。

## 账本快照

### v1 历史账本（旧链 300M/600M 记账，仅作回归证据，不再消费）

| 角色 | 池总量 | 300M 消耗 | 600M 增量 | 累计消耗 | 剩余 |
|---|---|---|---|---|---|
| wiki | 649,904,474 | 166,657,644 | 166,657,317 | 333,314,961 | 316,589,513 |
| hq | 383,617,452 | 98,372,582 | 98,372,261 | 196,744,843 | 186,872,609 |
| 天然合计 | 1,033,521,926 | 265,030,226 | 265,029,578 | 530,059,804 | 503,462,122 |

- 旧链记账比例（两轮先例）：wiki 55.6% / hq 32.8% / structure 1.6% / colloquial 10.0%。
  新链比例**不沿用旧链**，以 v2 candidate（下表）为准。
- 合成池不跨轮复用：每轮新铸；structure/colloquial 已 retire，其语义位置由
  v2 的 structured/dialogue 天然角色顶替（不复活合成）。
- v1 逐文档账本已丢失（只剩 token 片），v2 池以新账本起步——跨代去重无从执行。

### v2 现役账本（从零重建，词表 zh-24k-v3）

权威锚点：池 release `zh-v2-pool-natural-v4`（v3 为 32K 代，已被 24K v3 代 supersede）（.local/artifacts/mei-1.0-51m/
zh-v2-pool/pools/）与各 admitted manifest；余额推导式不变。

| role | 池总量（v4，zh-24k-v3 代） | 7 轮配额（每轮×7） | 覆盖 |
|---|---|---|---|
| fineweb2_hq | 1,109,428,850 | 131M×7=917M | ✅ |
| wiki_zh | 805,350,781 | 80M×7=560M | ✅ |
| dialogue | 230,520,439 | 30M×7=210M | ✅（用户拍板封顶） |
| structured | 433,451,965 | 25M×7=175M | ✅ |
| code | 627,361,740 | 23M×7=161M | ✅ |
| wiki_en | 295,783,980 | 11M×7=77M | ✅ |

首轮 candidate `mix-zhv2-300m-c01-v3`（passed，zh-24k-v3 代）：hq 135M /
wiki_zh 81M / dialogue 39M / structured 16.2M / code 21M / wiki_en 7.8M。
**词表代际：24K v3（参数契约 51,463,797 保持；32K v2 代作废为证据）。**
structured 首轮配额按池容量收窄（0.055→0.054，supersede 登记）；DBLP 已入池，
第 2 轮起可恢复 25M/轮。每轮重跑 plan_mix，比例调整 = supersede + reason。

## 每轮 300M mix 规划流程（rung 开工前）

1. 读三样输入：上一轮 `cycle cpt.json` 配额；上一轮 synthetic-diversity receipt（通过/坍缩）；
   本表推导余额。
2. 产出**机器可读 candidate mix**：预算 300M、分角色 `token_quota`、来源池、片级选择语义
   （沿用 `skip_seen`/`allow_repeat=false`/unseen-first）、curriculum 分段（s1 512 / s2 1024 / s3 2048，
   比例随整轮配额），附**理由链**——每个数字可追溯（记账延续 / receipt 证据 / 池余额 / 下载入池）。
3. 规则：
   - 天然配额只从池内分配，candidate 总额 ≤ 池余额——**超池预算禁止**；
   - 合成角色只有"上一轮 receipt 通过 + 本轮铸币前模板审计过"才给配额；坍缩/retire 角色默认回填天然；
   - 比例默认沿用最近一轮已记账值；**调整 = supersede 事件**（新 candidate ID + 书面 reason），不改旧记录；
   - 池余额 < 下一轮 natural 配额 → 预算收窄或触发下载协议，不硬凑。
4. 登记：candidate mix 作为设计产物（带 ID 与理由链）落 release `governance/`；把配额**绑定进 cycle 的
   cpt.json 是 `mei-51m-cycle-orchestrator` 的动作**，本 Skill 交回不代做。

## 调节杠杆表

| 触发输入 | 信号 | 动作 |
|---|---|---|
| rung-n receipt 坍缩（structure/colloquial） | 该角色低损窗口 100% 低于阈值 | 下一轮该角色配额 = 0 并 retire；余额回填天然（v1 教训，v2 天然角色无此门） |
| 池余额 < 下一轮 natural 配额 | 余额推导即时可见 | 授权下载（见协议）或降该角色比例 + supersede 登记 |
| 新片下载入池 | 池总量变化 | 账本快照追加一行；下一轮 candidate 重算 |
| 天然角色 run-time loss 异常坍缩 | natural 无需模板审计 | 先查管线/schedule/去重（bug 信号），不调比例掩盖 |
| 天然角色长期单调重复 | 低损但文档级重复 | 查 document-level dedup 与 skip_seen 是否失效 |

## FineWeb2-HQ / wiki 新片下载协议（默认离线，须显式授权）

1. **触发**：余额 < 下一轮 natural 配额，或用户点名某批片集。
2. **前置**：用户显式授权一次 = 一批片集（离线默认不下载，见 SKILL 原则 10；付费/新下载均属需授权动作）。
3. **步骤**：记录来源与 license（band A/B/C 政策裁决见
   `corpus-factory/quality/policy/source-policy-v1.json`）→ sha256 校验 → 冻结词表
   （当前 zh-24k-v3，`TOKENIZER.json` 指针 + `--expected-tokenizer-id` 双绑）encode →
   document-level dedup vs 已消费（unseen-first，seen-ledger 追加式）→
   铸**新池 release ID**（supersede 链 + reason，不覆盖旧 release）→ 本表追加快照行。
4. 红线：不下则预算收窄，不硬凑超池；下载后未过校验/去重不得进 schedule；旧池不删不改。

## 红线汇总

- 下载默认不做；每次下载 = 用户显式授权事件；
- candidate mix 超池禁止（预算收窄优于违约）；
- 调比例必须带 reason 写成 supersede，不在旧记录上改；
- 语料统计（片数/tokens/池余额）≠ 模型质量证据；天然角色质量看 run-time loss 与下游诊断，
  合成角色另加一道铸币前模板审计（见 audit-and-repair.md）。

## v2 操作要点（从零重建）

- **plan-mix v2**：`--fraction role=0.xx`（Σ=1，floor_last_role 兜底取余）或
  `--quota role=N` 显式配额；`--hq-fraction` 为弃用别名（= fineweb2_hq F /
  wiki_zh 1−F，容量键 wiki 自动映射 wiki_zh）。比例变更 = supersede 事件 +
  书面 `--reason`（缺 reason 时 candidate 标 `policy.needs_reason`，技能层拦截）。
- **新来源下载**：两段式授权（dry-run 出令牌 → 令牌匹配才联网），详见
  download-and-admission.md。
- **池 release v2**：绑定单一 tokenizer 代际（混合代际禁止铸池）；
  大源（>1B tokens）必须分批 admit（单批驻留内存上限约 1B tokens）。
