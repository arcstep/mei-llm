# mw-sft-v0 交接合同（给并行训练任务）

> **本任务不训练、不改训练入口、不覆盖 `home-sft-2k/10k`。**  
> 并行训练任务只需读路径、schema、hash。不必合并本目录以外的生成器代码。

## 消费什么

| 资产 | 路径 | sha256 |
|---|---|---|
| SFT 包 | `tasks/needle-zh/train/packs/mw-sft-v0-2k.jsonl` | `fd6ff3c76ea86dadc950c2cf4fe49abadb1a52befefa2ae95c343cec32129b2f` |
| recipe | `tasks/needle-zh/recipes/mw-sft-v0.recipe.json` | `0bea0a4245462e85c2139a74a768bc32f50afb41fada96bdf54ba9b08f80a51d` |
| 字段合同 | `eval/shared/mw-governance-v0.json` | `bf0a954adbd5e06118e860b4ac84a7ad32acd23628b79ee4fcfc95050a94160c` |
| 独立验证集 | `eval/banks/needle-vrm-mw-v0/eval-bank-v0.jsonl` | `699d90856f27c178c38b41cfd8869a85e18991c1b056296c283eacc8c38a3f06` |
| 验证 lock | `eval/banks/needle-vrm-mw-v0/holdout-mw-v0.lock.json` | `3974625a7cc86de0b97a82ce60428707271e13db22ae46a9de398f8f188c8cda` |
| 评分器 | `scripts/eval_needle_mw_v0.py` | 一期投影：`[]` → `non-execute`，禁止算成 `stop` |

**不要**把该 SFT 包登记进 `tasks/index.json` 的一期 `train_seeds`，除非训练任务自己显式决定。本数据准备任务不会改 index。

## 每条样本字段

| 字段 | 含义 |
|---|---|
| `query` / 可选 `scene` | 用户句；训练时拼成 `场景：…。用户：…` 或 `用户：…` |
| `act` | `execute\|expand\|shape\|escalate\|stop`（**不是** null；一期 home-sft 的 `act=null` 合同不适用） |
| `cell` | `MW.OK\|SF.PAR\|SF.AMB\|SF.PRE\|Unknown` |
| `reason_code` | 闭集，见 schema |
| `gaps` | 闭集字符串列表 |
| `function_calls` | **仅** `act=execute` 非空；其它动作必须 `[]` |
| `answers` | 与 `function_calls` 相同，仅便于对照；不要按一期 refuse=`[]` 来训五类动作 |

Gold 由程序从 canonical case 重算。教师（若有）只能改 query/scene。

## 一期投影

若 checkpoint 仍只输出工具 JSON / `[]`：

- 有调用 → `execute`
- `[]` → `non-execute`（**不是** `stop`）
- 五类 `act` macro-F1 对该模型报告 `unsupported`

## 禁止

- 改 v1/v2 题库、grammar、词表、runtime、CPT、home-sft 包
- 把 rubble 17 格写进单轮 gold
- 把本包与 MW 验证集的 canonical / 模板 / 反事实组混用
