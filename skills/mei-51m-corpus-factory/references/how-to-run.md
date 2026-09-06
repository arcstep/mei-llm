# Skill 执行入口

## 入口与实现边界

本 Skill 的 `scripts/` 是稳定薄入口，负责：

- 从 Skill 真源或 `.cursor/skills`、`.agents/skills` 投影定位 `mei-llm`；
- 检查环境与参数；
- 强制使用显式的新 release ID；
- 拒绝覆盖已有不可变 release；
- 调用 `corpus-factory/` 中的唯一生成器与 verifier 实现。

生成逻辑仍在 `corpus-factory/`，禁止复制进 Skill。

以下命令可在 Skill 目录执行。若从任意其他目录执行，请使用脚本完整路径。

## 一、环境检查

```bash
python scripts/doctor.py
```

检查：

- `corpus-factory`、`CURRENT.json` 与 verifier 是否存在；
- 当前 Python 是否为仓内 `.venv/bin/python`；
- deploy/training 工具注册表能否加载及其数量；
- zh-24k-v1 tokenizer 是否回退；
- `.local/artifacts` 输出位置是否可写。

严格要求真实 tokenizer：

```bash
python scripts/doctor.py --strict-tokenizer
```

该命令只读，不生成或移动任何语料。

## 二、构建 SFT release

先看执行计划：

```bash
python scripts/build.py \
  --release-id mei-1.0-51m-exp-000600m-sft-<topic>-pilot-v1 \
  --pilot \
  --plan
```

确认后执行：

```bash
python scripts/build.py \
  --release-id mei-1.0-51m-exp-000600m-sft-<topic>-pilot-v1 \
  --pilot
```

正式扩量：

```bash
python scripts/build.py \
  --release-id mei-1.0-51m-exp-000600m-sft-<topic>-v1
```

约束：

- `--release-id` 必填，且必须以 `mei-1.0-51m-` 开头；
- 输出目录已存在时立即拒绝，不覆盖、不删除；
- `--pilot` 将所有正整数目标计数封顶到 `--pilot-limit`（默认 20）；
- `--plan` 只打印根目录、输出目录和实际 targets，不写文件；
- 生成器仍是 `corpus-factory/generators/rebuild_zh_v1/build.py`。

输出：

```text
artifacts/mei-1.0-51m/legacy/exp-000600m/corpus/sft-suite/<RELEASE_ID>/
├── semantic/
├── compiled/
├── governance/
│   ├── coverage.json
│   ├── audits.json
│   └── gates.json
├── manifests/artifact-manifest.json
└── release-manifest.json
```

## 三、验证指定 release

```bash
python scripts/validate.py \
  --release-id mei-1.0-51m-exp-000600m-sft-<topic>-pilot-v1
```

如果已生成对应 eval-lock：

```bash
python scripts/validate.py \
  --release-id mei-1.0-51m-exp-000600m-sft-<topic>-v1 \
  --eval-lock-id mei-51m-<topic>-eval-v1
```

验证 artifact 的 byte/hash/merkle、`CURRENT.json` 未被修改以及冻结 Base
权重未变化。返回码 0 表示通过，1 表示失败。

## 四、修改生成器

1. 在 `corpus-factory/generators/rebuild_zh_v1/` 修改或新增 family。
2. 在 `build.py` 注册 family 和 targets；不要在 Skill 脚本复制生成逻辑。
3. 用新的 pilot release ID 运行 `scripts/build.py --pilot --plan`。
4. 执行 pilot，检查 `governance/{coverage,audits,gates}.json`。
5. 修复生成器后使用另一个新 ID；失败产物也不原地覆盖。
6. pilot 通过后再使用新的正式 release ID 扩量。

## 五、factory-v3 治理子命令

需求账本、worklist、scale campaign 等治理动作仍走：

```bash
python corpus-factory/generators/factory_51m.py <子命令> --help
```

主要子命令：`inventory-baselines`、`build-demand-ledger`、
`verify-demand-ledger`、`freeze-cpt-policy`、`freeze-sft-validation`、
`create-worklist`、`freeze-scale-campaign`、`register-scale-gate`、
`prepare-scale-review`、`record-scale-review`。

## 六、定位规则

脚本按以下顺序寻找代码仓：

1. 环境变量 `MEI_LLM_ROOT`；
2. 脚本所在路径向上查找；
3. 当前工作目录向上查找；
4. 各级目录下的 `mei-llm/mei-llm` 或 `mei-llm`。

若不在 `mei-projects` 工作区：

```bash
MEI_LLM_ROOT=/absolute/path/to/mei-llm \
  python scripts/doctor.py
```
