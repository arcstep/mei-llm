# mei-llm

MEI 的本地模型工程仓。唯一部署 LM 参数固定为 51,463,797；300M / 600M / … / 1800M 表示
累计 CPT token exposure，不是参数规模。

代际：**v1.1**（历史归档链）、**v1.2**（1800M 所在链）、**v1.3**（当前 A10 新训）。

## 从哪里开始

导航权威是 [AGENTS.md](AGENTS.md)。仓库按「四分离 + 研究线」布局：

```text
src/       源码：architecture / corpus-factory / model-factory / platform / mei_llm（控制面）
corpus/    语料成果：pools / sft-suite / adoptions / catalog / eval-lock
cycles/    训练过程账本：每个 exposure rung 的 receipt / 评测 / 决策
models/    冻结成果：tokenizer + releases 里的 base / SFT 产物
research/  研究探索与情报（不训练模型自身的工作；情报本体见 docs/intel）
docs/      独立文档仓（SSOT + draft + intel + archive，自有 .git）
```

- 当前主线用哪套：`CURRENT.json` + `corpus/adoptions/mei-51m-v1.3/adoption.json`（语料单一真相源）
- 某代际模型 / 语料 / 过程：`models/mei-X-51m/`、`corpus/`、`cycles/mei-X-51m/`
- 训练 / 评测 / 发布入口：`src/model-factory/contracts/PIPELINES.json`（不看文件名版本后缀）
- runtime 实现：`src/platform/`（python-sdk / browser-sdk / _shared/rust）

## 统一只读导航

```bash
PYTHONPATH=src .venv/bin/python -m mei_llm model status
PYTHONPATH=src .venv/bin/python -m mei_llm cycle list
PYTHONPATH=src .venv/bin/python -m mei_llm artifact resolve '<URI-or-legacy-path>'
```

写入型命令受不可变 artifact、fingerprint、离线与 `CURRENT.json` 只读规则约束。权重、大语料、
真实安全审查与备案实例不进入公开 Git。
