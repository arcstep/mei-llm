# mei-1.3-51m 货架导航

目标：与 v1.2（1800M CPT）及后续新版本对照时，一眼知道**用哪个跑**。非基线的产物一律
归档/标注，不留在 releases/ 货架上混淆。

## 一、用哪个跑（活跃基线）

### CPT base 缩放曲线（作 CPT 对照）

| base | tokens_seen | status |
|---|---|---|
| `mei-51m-v1.3-800m-base` | 800,000,000 | copied_verified_diagnostic_complete |
| `mei-51m-v1.3-1600m-base` | 1,600,000,000 | copied_verified |

A10 CPT 目标 2,500,001,792 token；阶段里程碑复制核验后按实际累计曝光登记，后续继续递增。

### SFT（当前唯一 adaptive-v5 产物，治理失败，不可作最终产物）

- product：`releases/exp-00800m/products/mei-51m-v1.3-800m-tool-sft-cq2-v2-adaptive-v5/`
- ⚠️ 该产物 productize 主线**误用 v1.2 旧语料**（`mei-1.0-51m-tool-sft-v4-300m-v4` 等），
  `release_eligible=false`；**须用 `mei-1.3-51m-tool-sft-20260917-fivehead-v1` 语料重跑**后另立正式产物。
- 五头跑分数值有效但基于治理失败语料，**不作为 v1.3 最终能力结论**（详见该 STATUS.md）。

## 二、命名坑

- `800M`/`1600M`/`2.5b` 是 **CPT token exposure**，不是参数规模（参数恒 51,463,797）。
- `cq2` 是 2-bit 量化格式，`v2` 是打包格式版本，与训练流程代际无关。
