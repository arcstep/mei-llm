# Needle-zh 语料探查池

一次性探查，**不是**生产语料。样本与 wiki-hashes 已 gitignore。设计报告已归入私有 monorepo archive，不作为本仓依赖。

全集：`universe.json`（40 项）。实测摘要：`summary.json`（14 张卡）。脚本：`../../scripts/probe_zh_corpus_sources.py`。

本机占用约 3.5GB（2026-08-25）。

## Probe Cards

| source_id | 档 | 账本 | n_docs | tokens | UNK | wiki overlap | 抽检噪声 | 硬门 | 采用理由 |
|-----------|----|------|--------|--------|-----|--------------|----------|------|----------|
| fineweb2-hq-cmn-hani | C | CPT | 8,000 | 10.86M | 0.048% | 0.19% | 7.5% | 过 | HQ 主池 Level A |
| fineweb2-hq-cmn-hani-b | C | CPT | 40,000 | 51.08M | 0.042% | 0.08% | 7.5% | 过 | 同一 parquet Level B，分布稳定 |
| fineweb2-cmn-hani-test | D | 禁止训练 | 8,000 | 11.94M | 0.091% | 0.01% | 15% | 过 | 仅分布探针 |
| hplt2-zho-hans | E | — | 8,000 | 23.40M | 0.065% | 0 | 22.5% | 噪声失败 | 不进主池 |
| hplt3-cmn-hans | C | 默认 0 | 6,629 | 33.87M | 0.039% | 0 | 17.5% | noise_ok false | 长尾对照；语言码 `cmn_Hans` |
| chinese-c4 | D | 工程基线 | 8,000 | 6.57M | 0.097% | 0 | 2.5% | 过 | 许可证据不足 |
| skypile-150b | C | 未授权=0 | 8,000 | 7.63M | 0.027% | 0 | 5.0% | 过 | 社区许可管模型，不升格 |
| crosswoz-train | B | SFT | 5,012 | 1.13M | 0.24% | 0 | 0% | 过 | Apache-2.0；dev/test 隔离 |
| kdconv-train | C | SFT | 3,600 | 1.35M | 0.22% | 0 | 27.5%* | 过 | *电话误标；去 KG/电话 |
| schemastore-json | B | CPT 结构 | 400 | 7.09M | ~0 | 0 | 7.5% | 过 | 仅仓内 schema |
| oai-openapi-examples | B | CPT 结构 | 3 | 2,980 | 0 | 0 | 启发式假阳性 | 人工过 | tag 3.0.3 三份 yaml |
| json-schema-test-suite | D | 评测灵感 | 400 | 0.14M | 0.03% | 0 | 0% | 过 | 禁止训练 |
| toolace | C | 非本产品 CPT | 4,000 | 8.03M | ~0 | 0 | 2.5% | 过 | 英文 FC；须查 BFCL |
| api-bank-init | D | 错切片 | 200 | 0.06M | 0 | 0 | 0% | 过 | train 对话不在 git 仓 |

未采样（许可/访问）：`cci3-hq`、`cci4-base-zh-cc`、`telechat-ptd`、`wanjuan1-zh-nonwiki`、`chinesewebtext2`。排除名单见 `universe.json` 的 `pre_decision=E`。
