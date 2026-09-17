# mei-1.2-51m 货架导航

目标：与将来的新版本（2.5b/1.6b/0.8b 等）对照时，一眼知道**用哪个跑**。
不是历史轨迹陈列，非基线的产物一律归档，不留在 releases/ 货架上混淆。

## 一、用哪个跑（活跃基线）

### 当前 SFT 基线 —— 五头完整（唯一作 SFT 对照的产物）

- product：`releases/exp-01800m/products/mei-1.2-51m-cpt1800m-tool-sft-cq2-v2-adaptive-v5/`
- 依赖 base：`releases/exp-01800m/base/mei-1.2-51m-base-cpt1800m-v1/`
- 五头全部训练（含 narration），走 phase-binding 主线 `adaptive-v5` 流程。
- 后续 2.5b/1.6b/0.8b 若要 apples-to-apples 对比，**必须用同一套 adaptive-v5 流程 + 相同语料**。

### CPT token exposure 缩放曲线（base，作 CPT 对照）

| base | tokens_seen | valid_loss | probe_mean_nll |
|---|---|---|---|
| `base-cpt300m-v1` | 300,000,768 | 1.906 | 4.586 |
| `base-cpt600m-v1` | 600,006,144 | 1.613 | 4.446 |
| `base-cpt900m-v1` | 900,007,424 | 1.545 | 4.287 |
| `base-cpt1200m-v1` | 1,200,006,656 | 1.494 | 4.144 |
| `base-cpt1800m-v1` | 1,800,001,024 | 1.394 | **3.600** |

valid_loss / probe 随 exposure 单调下降，是同一条续训链上的干净缩放曲线，可直接作 CPT 对照。

## 二、已归档（历史流程，不作基线）

以下产物 `cq2-v1` 时代的 SFT，**narration 头是零初始化占位、未训练**（stage_id
`zero-init-untrained-narration-adapter`），且其余头走的是老流程（quant-aware-zhv2 /
retrieval-rebuild-v4 / confidence-v3 / mw-disposition-batched-v1）。与 adaptive-v5 不是
同一代流程，**不能作「相同流程 × 不同 base」的对照**，已归档到 `archive/`：

- `archive/exp-00300m/.../mei-1.2-51m-cpt300m-tool-sft-cq2-v1`
- `archive/exp-00600m/.../mei-1.2-51m-cpt600m-tool-sft-cq2-v1`
- `archive/exp-00900m/.../mei-1.2-51m-cpt900m-tool-sft-cq2-v1`

对应的 **CPT base 保留**（base 是续训链的一部分，与 SFT 流程无关）。

### 不进 releases/ 的中间态（在 cycles/ 里，天然非货架）

- 1500M：`continuation-anchor`（RELEASE.json 明确 `release_eligible=false`、
  `promotion_prohibited=true`，valid 1.509 vs 父 1.494 退化，只作 1800M 的形式父节点）。
- 2100M：用户拍板暂停，停在 1,927,310,848 tokens，未完成。

## 三、命名的两个坑（避免再踩）

1. **`cq2` 是 2-bit 量化格式**（Walsh-Hadamard + Q2/Q4 codebook），不是语料版本；
   `v1`/`v2` 是量化打包格式版本，与训练流程代际无关。
2. **`1.8b/2.5b/1.6b/0.8b` 是 token exposure，不是参数规模**（参数恒 51.5M）。
