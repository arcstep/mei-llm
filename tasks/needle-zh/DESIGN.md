# needle-zh

中文端侧工具调用路由器（Needle 类产品合同）。不是通用聊天模型。

对用户可以「又调又说」；结构上必须**先路由、后生成**。本模型只输出 `execute` 或 `[]`。说话器（若有）只读已执行的工具 JSON + 垂直术语，不得再调工具，不得把 `[]` 改写成「请补充城市」。说话器权重可替换，不与本路由器嫁接词表。

首个学生是 **Needle 2 参考配方的中文变体**（`d_model=512`、27 层、8/4 GQA、RoPE、HadamardMLP、engram 2/15、mHC 4 lanes、vocab 24000，MLX 计数约 58.5M）。这是工程基线，不宣称论文纯 SAN。运行时是 Apple Silicon MLX + 自有 checkpoint，不接官方 libneedle。

冻结规格：`spec/model.json`、`spec/special-tokens.json`、`spec/runtime.md`、`spec/gates.json`。

## 一期产品合同

- 输出只有 `execute`（函数调用）或空 `[]`。
- 空 call 表示「覆盖不了 / 不要执行」，不是「请用户补充」。缺槽、场景冲突、非法搭配、离题均为 `[]`。
- 槽值 exact-match。用 query 子串当槽值算失败。
- 离题大约 1/8 的 SFT，必须空 call。
- 一期就要校准中文置信头。官方 Needle 2 LoRA 不更新该头。
- 不要输出「请补充城市」这类自然语言。
- 不要在英文 Needle 2 上全参微调救中文（8k SentencePiece 冻结）。
- **预训是本路线必经步骤**。随机初始化 + 同一套 SFT 只作负对照。Qwen 旁路只验证数据能否被学会。

## 二期（以后）

把 `[]` 拆成 `expand` / `shape` / `escalate` / `stop`。改后训练与语法，不丢一期词表/基座。一期 `execute/[]` 与置信校准通过前，不把二期动作混进训练。

## 共享 vs 本 task

| 共享 | 本 task |
|------|---------|
| `corpora/zh-vocab-v0` | tokenizer v1（`zh-24k-v1`；旧 `zh-24k.model` 为草稿） |
| `corpora/zh-pretrain-v0` | 窄中文预训槽（探针验收；token 阶梯） |
| `eval/banks/needle-toolcall-v0` | exact-match 协议烟测 |
| `eval/banks/needle-vrm-agent-v0` | 家居数字人产品 EVAL（48 公开 dev + holdout） |
| `eval/banks/needle-vrm-agent-en-v0` | 英文对照（**不是** KPI） |
| `eval/banks/needle-pretrain-probes-v0` | 预训探针 |
| `eval/shared/toolsets/` | 工具 schema |

本 task 协议回归种子：`train/seed/sft-phase1-v0.jsonl`（天气/灯/发票）。家居垂直 SFT 在 `train/packs/`。

## 预训验收（不是产品 KPI）

产品 KPI 是 SFT 后在 `needle-vrm-agent-v0` holdout 上的 exact-match，以及同一次跑分的完整作答时延（产物在 `experiments/runs/`）。对照：随机+SFT vs 预训+同一套 SFT。

工作门槛见 `spec/gates.json`。硬件证伪必须改规格，禁止静默放宽。

## 命令

```bash
python3 scripts/validate_needle_sft_seed.py
python3 scripts/check_train_eval_isolation.py --all
python3 scripts/eval_needle_toolcall_v0.py --bank eval/banks/needle-vrm-agent-v0/eval-bank-v0.jsonl
python3 tasks/needle-zh/model/check_student.py
python3 scripts/train_zh_vocab_spm.py --freeze-v1
```
