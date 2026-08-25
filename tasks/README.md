# tasks（训练任务 / 模型线）

一条 task = 一种要训的模型（或同一产品合同的一个阶段），**不是**一份评测题库。

语料与评测在仓库根的 `corpora/`、`eval/`；本目录只放配方、种子覆盖、adapter、DESIGN。

| id | 现状 |
|----|------|
| `mei-expert-qwen35-0p8b` | 既有 Qwen3.5 0.8B SFT / MLX 烟测 |
| `needle-zh` | 中文端侧工具调用一期（从零；词表+预训+SFT） |

登记：`tasks/index.json`。新增 task 时同步登记 `train_seed`、`eval_banks`、`corpora`。
