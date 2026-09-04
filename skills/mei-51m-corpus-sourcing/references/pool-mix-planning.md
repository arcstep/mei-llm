# CPT 天然池盘点与每轮 mix 规划

## 角色与池子（mei-1.0-51m，zh-24k-v1）

- **天然角色**（第一类资源，可靠、可耗尽）：
  - `wiki` = `zh-pretrain-v0`，lm-v1 基线目录 `exp-000300m/corpus/cpt-delta/lm-v1/language/zh-pretrain-v0/tokens/`，100 片在盘；
  - `hq` = FineWeb2-HQ `cmn_Hani`，`language/hq/tokens/`，49 片在盘（license ODC-By-1.0 + Common Crawl ToU；hardlinks 自
    `notebook/archive/corpus/zh-pretrain-v2/tokens/hq-*`，raw 未丢、provenance 完好）。
- **合成角色**（gap-fill 语义，默认不放大）：`structure`（曾 1.6%/轮）、`colloquial`（曾 10%/轮）。
  审计结论（2026-09-03，见 `.local/artifacts/mei-1.0-51m/synthetic-cpt-audit-v1/`）：structure lm-v1 与 lm-v2
  均坍缩、colloquial lm-v2 坍缩 → 600M 增量已 `retire_for_future_reuse`。合成份额默认回填天然；
  重建须先过铸币前模板审计 + 用户政策决定，才谈放回 mix。
- **权威锚点**（余额一律**即时推导**，不另建静态账本防漂移）：
  - 池总量：`exp-000300m/corpus/cpt-delta/lm-v1/mix.json`（`n_wiki_train_tokens=649,904,474`、
    `n_hq_train_tokens=383,617,452`）；
  - 每轮消耗：`cycles/mei-1.0-51m/exp-000300m/corpus/cpt.json`（`cpt.source_quotas`）与
    `exp-000600m/...`（`cpt.incremental_quotas`）；
  - 片级选择记账：`lm-v1/schedule-scratch.json`、`exp-000600m/corpus/cpt-delta/lm-v2-cpt-600m/schedule-cpt-600m.json`
    （`sampler=quota_plan`、`allow_repeat=false`、`skip_seen_*`、`max_epochs` = 配额/池总量）。
- **余额推导式**：`剩余(role) = mix.json n_<role>_train_tokens − Σ(各轮 cpt.json quotas[role])`。

## 账本快照（as-of 2026-09-03，600M 记账后）

| 角色 | 池总量 | 300M 消耗 | 600M 增量 | 累计消耗 | 剩余（≈轮数 @记账速率） |
|---|---|---|---|---|---|
| wiki | 649,904,474 | 166,657,644 | 166,657,317 | 333,314,961 | 316,589,513（≈1.9 轮 @166.66M） |
| hq | 383,617,452 | 98,372,582 | 98,372,261 | 196,744,843 | 186,872,609（≈1.9 轮 @98.37M） |
| 天然合计 | 1,033,521,926 | 265,030,226 | 265,029,578 | 530,059,804 | 503,462,122 |

- 记账固定比例（两轮先例，300M 内）：wiki 55.6% / hq 32.8% / structure 1.6% / colloquial 10.0%
  （natural 每轮 265.03M = 88.4%）。factory-v3 `cpt_policy` 默认（hq 65 / wiki 35）只是政策参照，
  **不一致时以 cycle 记账为准**；采用工厂默认意味着比例调整事件。
- 轮次推演：按记账比例 900M 整轮足额，**1.2B 前必缺**（天然只够 ~1.9 轮）；纯天然 65/35 同样 1.2B 前告急
  （hq 是"短"的一侧：@195M/轮 ≈0.96 轮）。**结论：1.2B 之前必须开下载门（hq 尚有 ~100 片可拉）或调比例，两者都是需登记的显式决策。**
- 合成池不跨轮复用：每轮新铸（300M 轮 lm-v1 合成池 max_epochs=1.0 吃满；600M 增量新铸 lm-v2 版已 retire）。

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
| rung-n receipt 坍缩（structure/colloquial） | 该角色低损窗口 100% 低于阈值 | 下一轮该角色配额 = 0 并 retire；余额回填天然 |
| 池余额 < 下一轮 natural 配额 | 按推演 1.2B 前必然出现 | 授权下载（见协议）或降该角色比例 + supersede 登记 |
| 新片下载入池 | 池总量变化 | 账本快照追加一行；下一轮 candidate 重算 |
| 天然角色 run-time loss 异常坍缩 | natural 无需模板审计 | 先查管线/schedule/去重（bug 信号），不调比例掩盖 |
| 天然角色长期单调重复 | 低损但文档级重复 | 查 document-level dedup 与 skip_seen 是否失效 |

## FineWeb2-HQ / wiki 新片下载协议（默认离线，须显式授权）

1. **触发**：余额 < 下一轮 natural 配额，或用户点名某批片集。
2. **前置**：用户显式授权一次 = 一批片集（离线默认不下载，见 SKILL 原则 10；付费/新下载均属需授权动作）。
3. **步骤**：记录来源与 license（hq：FineWeb2-HQ `cmn_Hani`，ODC-By-1.0 + CC ToU；
   `public_distribution_clearance_asserted` 保持 false）→ sha256 校验 → zh-24k-v1 tokenize →
   document-level dedup vs 已消费（unseen-first，参照 schedule `skip_seen`/`allow_repeat=false`）→
   铸**新池 release ID**（如 `zh-pretrain-hq-v2`，不覆盖旧 `zh-pretrain-hq`，supersede 链 + reason）→
   更新池目录 RELEASE.json / SOURCES.md / hashes → 本表追加快照行。
4. 红线：不下则预算收窄，不硬凑超池；下载后未过校验/去重不得进 schedule；旧池不删不改。

## 红线汇总

- 下载默认不做；每次下载 = 用户显式授权事件；
- candidate mix 超池禁止（预算收窄优于违约）；
- 调比例必须带 reason 写成 supersede，不在旧记录上改；
- 语料统计（片数/tokens/池余额）≠ 模型质量证据；天然角色质量看 run-time loss 与下游诊断，
  合成角色另加一道铸币前模板审计（见 audit-and-repair.md）。
