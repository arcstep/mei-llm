# Corpus Factory

语料工厂与模型架构同级，是每个 exposure cycle 的常规生产系统。这里保存当前最新版的
生成、验证和 release 合同；工厂实现历史由 Git 追踪。实际语料不在这里按“最新状态”
堆积，而必须绑定到一个 cycle，存入 `.local/artifacts/<model>/<cycle>/corpus/`，并由该周期的
`corpus/{cpt,sft,eval}.json` 引用。

生产节奏：

```text
需求账本 → source/provenance → normalize/synthesize → verifier
         → dedup/diversity → family/group split → leakage
         → human review → immutable cycle-bound release
```

正式入口：

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
PYTHONPATH=src .venv/bin/python -m mei_llm corpus show exp-000600m
```
