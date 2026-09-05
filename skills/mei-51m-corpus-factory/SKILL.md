---
name: mei-51m-corpus-factory
description: >-
  按垂直主题培育 mei-1.0-51m 中文工具 Agent 语料时使用：为新场景主题做跨
  CPT/MW/SFT 环节的全光谱展开设计、在 corpus-factory/generators 下新增或
  修改生成器、设计某主题的场景与配额、审计旧语料能否原地修复、扩量前的
  多样性/模板 bug 修复、复用 model-factory/diagnostics 做训练前的便宜能力
  验证、或把新 release 登记回 cycle 的 corpus/sft.json/eval.json 与
  CORPUS.md 时使用。天然池、下载和入池交给 mei-51m-corpus-sourcing，独立
  质量裁决交给 mei-51m-corpus-quality；本 Skill 不启动 CPT/SFT/QAT。
---

# mei-1.0-51m 语料工厂（垂直主题培育）

## 不变量

1. **冻结目标**：每次任务显式绑定 cycle、已冻结 Base 与其 weights hash；
   600M Base 只作历史回归 fixture，不作未来默认值。`CURRENT.json` 只读、
   永不修改；不启动 CPT/SFT/QAT。
2. **Release 不可变**：每次铸币用**新 ID**（`-v<n+1>`）并 supersede 旧 ID，
   旧版保留为证据、绝不覆盖或删除；manifest 带 artifact merkle root。
3. **真实注册表**：工具与配额运行时从 live registries 读取（147 deploy tools /
   14 families；211 training tools / 22 families），绝不硬编码；场景只 grounded
   在真实 tool description/schema + 邻居线索，不用虚构工具或编造 schema。
4. **Token 预算**：用真实 zh-24k-v1 tokenizer（`count_tokens`）计 2048-token
   joint budget：prompt ≤ 1920（output reserve 128）；compact 1024 / standard
   1536 软上限；超限由 `fit_batch_to_budget` 裁剪，不手写近似。
5. **MW codebook**：20 类 reason_code（ready_to_execute=0 …
   partial_sequence_blocked=19）。batch-visibility 规则：class-0 target 工具
   不在 visible batch → effective label 10（capability_insufficient）。
   `raw_*` 与 effective label 分开存；仅 class-10 可推进扫描；
   safety/permission/state/evidence 与 exhaustion/limit 是 terminal。
   MW deviation ≠ MW disposition ≠ retrieval ≠ confidence。
6. **全光谱设计、产物硬分离**：讨论新主题默认跨 CPT/MW/SFT 环节分类展开（树干
   层按环节分杈、有意分类）；写入 frozen release 时各 family 严格分离
   （`sft_capability` 行集不混族、schema 不混用）；诊断与修复不污染正式产物。
7. **隔离与去重门**：split 由 `cf_group`（sha256 分桶）分配，与 eval-lock 共享
   隔离机制；dedup = exact + char-trigram Jaccard ≥ 0.92；`scan_leakage` 为零
   是硬门。hash/去重成功只证完整性与隔离，不证自然度——diversity 需模板级审计。
8. **职责交接**：天然池 inventory/mix/download/admission 由
   `mei-51m-corpus-sourcing` 产生；本 Skill 只消费其 hash-bound pool/mix。
   自动审计完成后必须交给 `mei-51m-corpus-quality` 做独立质量与复用裁决，
   不自行宣布语料成功。

## 执行入口

本 Skill 自带稳定薄入口；入口只负责定位 `mei-llm`、校验参数与调用
`corpus-factory/`，不复制生成器实现。

```bash
# 先检查环境（只读）
python scripts/doctor.py

# 只查看 pilot 计划，不写产物
python scripts/build.py \
  --cycle-id exp-000600m \
  --release-id mei-1.0-51m-<new-release-id> \
  --pilot --plan

# 注册本轮 family/配额 overlay（write-once）
python scripts/register_targets.py \
  --set RETRIEVAL_TARGETS.no_match=20 \
  --out /new/targets.json

# 只规划 eval lock；加 --execute 才创建
python scripts/build_eval_lock.py \
  --cycle-id exp-000600m \
  --sft-release-id mei-1.0-51m-<new-release-id> \
  --eval-lock-id mei-51m-<new-eval-id>

# 执行后校验指定 release
python scripts/validate.py --release-id mei-1.0-51m-<new-release-id>
```

运行 `scripts/build.py` 必须显式提供新 release ID；目标目录已存在时拒绝覆盖。
实际生成前先读 `references/how-to-run.md`。从 `.cursor/skills`、
`.agents/skills` 投影运行时也会自动定位 `mei-llm/mei-llm`；必要时设置
`MEI_LLM_ROOT`。

## 按任务读取

- 设计新主题的 taxonomy / 配额 / gates → `references/taxonomy-design.md`
- 写或改生成器 → `references/generator-engineering.md`
- 审计旧语料、决定原地修复 vs 替换 → `references/audit-and-repair.md`
- 扩量前后做便宜能力验证 → `references/diagnostic-validation.md`
- 铸币新 release / eval-lock / 登记回 cycle → `references/release-and-registration.md`
- 选下一棵树、查缺口账本 → `references/topic-priority.md`
- **脚本参数、新 release 派生、输出路径、常见错误** → `references/how-to-run.md`

## 执行原则

1. **一次一棵树**：单轮只深度培育一个 vertical topic（树干→树枝→树杈→树叶，
   边生成边审计边裁剪）。禁止跨 family 广而浅齐头并进——树枝便宜、树叶昂贵，
   同时铺太多主题必然停在树干/树枝层。
2. **全光谱展开**：新主题默认按 CPT/MW/SFT 各环节展开成"树枝"再落场景；但遵守
   不变量 6，frozen 产物保持各环节既有硬边界与各自 schema。
3. **数字落地**：配额与目标来自真实缺口度量（如 no-match false-selection 0.85、
   rank>5 gold retention 0.167、MW macro-F1 0.030093、confidence ECE 0.193）
   或 registry 计数，不拍脑袋造数；目标写成机器可读 dict（如
   `RETRIEVAL_TARGETS`/`MW_TARGETS`），每档带 gates。
4. **先小规模试生成**：pilot 先验 schema 合法性、token 预算、dedup 与 grounding，
   过门才扩到全量配额。
5. **审计先于替换**：替换旧语料前必须出审计证据——检查 grounding 列存在率
   （旧 MW 先例：20 类中 12 类 `candidate_tool` 100% null，含
   `capability_insufficient` 0/375 → 无法原地修复）。有 grounding 的缺陷（如
   label 偏差）才可原地修复；无 grounding 才换新。审计报告落 release `governance/`。
6. **修复先于扩量**：多样性/模板 bug（seed 不敏感模板、固定 tool 槽、双句号等）
   在 pilot 规模修掉再扩量。扩量不得放大 undesigned exact-dup（MW 先例：不修
   ≈39% → 修复后 16.3%），也不得为过门放水阈值。
7. **便宜诊断验证**：正式 pipeline 前用 raw float base + 单 fresh head +
   oracle-mode 固定视图跑几百步（约 1 分钟），验证语料携带可学习信号。脚本放
   `model-factory/diagnostics/` 并标 `diagnostic_only`（CODE_CATALOG.json 同步）。
   报告如实标注非严格 A/B——raw-float-base 数字不得与 CQ2+aligned 正式管线数字
   直接比较，旧基线只作 reference。
8. **新 ID 铸币**：语料 release 与 eval-lock 同步推进（`-v<n+1>` ↔ `-v<n+1>`），
   supersede 链写清 reason；verifier 校验新旧双版本与 CURRENT.json 不变。
9. **登记回周期 + 跨 cycle 复用**：release/eval-lock/known_quality_gaps/状态
   字面量写回铸币当轮 cycle 的 `corpus/{sft.json,eval.json}`，叙事追加到
   `CORPUS.md`（不重写旧记录）。**SFT release 是一次铸造、跨 cycle 复用**——
   每个 300M 增量**不**重新准备 SFT 语料；后续 cycle 通过 binding 引用同一冻结
   release，由 corpus-quality 每轮重裁 reuse/replace/retire，仅当新 Base 评测
   出现能力缺口时才铸造新 family（新 release ID）。`corpus_release_eligible`/
   `ready_for_<cycle>_sft` 等字面量一致；真实模型指标未测前一律 `pending_sft`，
   不用语料统计冒充。
10. **边界与授权**：默认离线（无付费教师模型、外部 provider、新下载）。不发布、
    不 commit/push、不修改 CURRENT.json、不启动 CPT/SFT/QAT——训练类动作分别交回
    `mei-51m-cpt-training` / `mei-51m-qat-training` /
    `mei-51m-sft-alignment`。普通错误自修自跑；仅数据污染、哈希冲突、唯一文件
    损坏、需要外部授权时才暂停询问。
11. **只消费已批准输入**：生成任务必须绑定 sourcing 的 pool/mix receipt；
    scale 前必须绑定 corpus-quality 的上一阶段通过证据。输入变化即新 fingerprint
    和新 release ID，不原地改写。

## 完成报告

每个主题树完成一轮（pilot/scale/diagnostic/决策）后按固定格式汇报：

- 主题与全光谱分支（树干层按环节分类）、各环节配额与 gates；
- pilot/audit/scale 每道门的实测值（dedup、leakage、budget、grounding、模板数）；
- 新 release ID + merkle root、supersede 链、eval-lock ID；
- 诊断指标（dev/test accuracy/macro-F1 或对应度量）+ reference 基线 + 措辞限定；
- known_quality_gaps、本轮未决项、下一棵树建议。
