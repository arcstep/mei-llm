# Corpus Factory

语料工厂与模型架构同级，是每个 exposure cycle 的常规生产系统。这里保存当前最新版的
生成、验证和 release 合同；工厂实现历史由 Git 追踪。实际语料不在这里按“最新状态”
堆积。语料成果位于 `corpus/`，来源调查与生产过程证据位于 `cycles/mei-1.2-51m/`；
训练周期通过各自的 corpus binding 显式引用，不回写旧receipt或用迁移后路径改写历史。

生产节奏：

```text
需求账本 → 来源快速确认与原文归集并行 → 候选整理与实际容量
         → 配比裁决 → source/provenance → normalize → verifier
         → dedup/diversity → family/group split → leakage
         → human review → immutable cycle-bound release
```

正式入口：

- `sources/profiling.py`：经 `corpus source profile` / `corpus evaluate audit-coverage` 进行调查、覆盖审计与待审证据准备；见[调查使用说明](sources/PROFILING.md)。旧M1字段保留历史含义；当前候选归集不等待全来源统一评级。
- `sources/materialize.py`：`corpus source materialize` 原文候选准备；支持已取调查原件、完整JSONL流、Parquet批量、字幕整份文本与汇总库存。
- `sources/source_manager.py`：天然池盘点、mix、授权下载、入池与 pool release；
- `generators/`：CPT/SFT/Eval 合成和编译；
- `quality/audit.py`：独立质量 receipt 与复用裁决；
- `verifiers/`：冻结设计和 release 的只读校验。

硬边界：

- CPT、SFT、Eval 分别冻结，不能用同一份 release 混淆职责。
- 每个旧 slice 在下一周期必须明确 `reuse`、`replace` 或 `retire`。
- hash/dedup/leakage 通过不等于语言质量通过；合成数据还要审计模板多样性、自然度和语义一致性。
- `corpus_diversity_degraded` 默认令 `corpus_reuse_eligible=false`。
- Retrieval、MW disposition、confidence 是三套独立数据/标签合同。
- Eval/gold marker 不得进入训练数据。

查看某轮实际语料：

```bash
PYTHONPATH=src .venv/bin/python -m mei_llm corpus show exp-00300m
```

## 批量原文准备

统一入口：`PYTHONPATH=src .venv/bin/python -m mei_llm corpus source materialize --config RECIPE --out NEW_ID`；远程正文另加 `--allow-network`。旧候选和receipt不覆盖。

- `mode: parquet_bulk`：固定文件清单和种子，随机行组、保留整条正文；成功原件按片保存CRC/哈希之外的SHA收据及原始位置，失败不生成完成manifest，新ID可哈希绑定采用已完成片。
- 普通来源 `kind: jsonl_stream`：完整读取已经明确选择的训练来源，保留对话和分组；`count_tokens: true`启用正文实测计数。
- `mode: subtitle_bulk`：整包下载与校验，按完整作品/演讲处理；字幕保守同目录版本选择、TED按独立文档，跨语言作品关联仍待审。原始字幕语言/评分/非机器标志不是质量证明。
- `mode: delivery_inventory`：核对候选文件与行哈希，汇总跨池精确正文并生成重复排除清单，不回写原始文件。

计量统一使用现役冻结词表`zh-24k-v3`，逐正文`encode_document`，含BOS/EOS，不计JSON元数据。未来候选词表确定后需重编码复核配额；当前实测量不能宣称是未知新词表下的token数。并列保留记录数、正文字符及UTF-8字节。

当前执行基线是30亿token候选及后续SFT/独立验收、匹配和词表审计，阶段产出不是结束条件。来源许可、完整性和评测隔离照常保留；第一轮不新增LLM扩写正文、不靠重复包装填补容量。本轮不训练、不替换tokenizer/权重/CURRENT。
