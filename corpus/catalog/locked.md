# 锁定记录与编译包

2026-09-14 目录快照。仅做导航，不以目录存在或 metadata 状态证明质量通过；原文、窗口和编译副本不重复加总。

**v1.3 SFT 五头语料已锁定（2026-09-17 决定）。** CPT 见[采用清单](../adoptions/mei-51m-v1.3/adoption.json)；SFT 五头统一 Mei 147 见下方「当前 SFT 锁定」。下列历史冻结记录须结合采用清单查看，不按目录名 vN 推断最新。

## 历史CPT

- [zh-v2-layout-300m-v1](../pools/zh-v2-pool/layouts/zh-v2-layout-300m-v1)：旧链 layout；存在不等于已被执行，1800M 采用关系见 v1.2 清单。
- [zh-v2-layout-exp-000600m-v2-v1](../pools/zh-v2-pool/layouts/zh-v2-layout-exp-000600m-v2-v1)：旧链 layout；存在不等于已被执行，1800M 采用关系见 v1.2 清单。
- [zh-v2-layout-exp-000900m-v2-v1](../pools/zh-v2-pool/layouts/zh-v2-layout-exp-000900m-v2-v1)：旧链 layout；存在不等于已被执行，1800M 采用关系见 v1.2 清单。
- [zh-v2-layout-exp-001200m-v2-v1](../pools/zh-v2-pool/layouts/zh-v2-layout-exp-001200m-v2-v1)：旧链 layout；存在不等于已被执行，1800M 采用关系见 v1.2 清单。
- [zh-v2-layout-exp-001500m-v2-v1](../pools/zh-v2-pool/layouts/zh-v2-layout-exp-001500m-v2-v1)：旧链 layout；存在不等于已被执行，1800M 采用关系见 v1.2 清单。
- [zh-v2-layout-exp-001800m-v1-v1](../pools/zh-v2-pool/layouts/zh-v2-layout-exp-001800m-v1-v1)：旧链 layout；存在不等于已被执行，1800M 采用关系见 v1.2 清单。
- [zh-v2-layout-exp-002100m-v1-v1](../pools/zh-v2-pool/layouts/zh-v2-layout-exp-002100m-v1-v1)：旧链 layout；存在不等于已被执行，1800M 采用关系见 v1.2 清单。

## 历史 SFT 输入 release（cycles 主 checkout，git 未跟踪）

位于主 checkout `cycles/mei-1.1-51m/exp-00300m/corpus/sft-suite/historical-notebook-releases/releases/`（git 未跟踪、worktree 无）。这些是 v1.1/v1.2 历史训练输入 release，禁止就地修改/删除（v1.2 receipt 哈希依赖）。采纳关系见 [v1.2 adoption.json](../adoptions/mei-51m-v1.2/adoption.json)。

- **v1.2 采纳（3）**：`mei-1.0-51m-tool-sft-v4-300m-v4`（data_release）、`mei-1.0-51m-tool-sft-linguistic-aug300m-v2`（linguistic_augmentation）、`mei-1.0-51m-narration-sft-agent300m-v3`（narration_release）。
- **v3 系列前序（7，未采纳）**：`mei-1.0-51m-tool-sft-v3-300m-v1` … `-v7`。
- **v4 系列前序（3，未采纳）**：`mei-1.0-51m-tool-sft-v4-300m-v1` … `-v3`。
- **v2 系列（2，未采纳）**：`mei-1.0-51m-tool-sft-v2-300m-v1`、`mei-1.0-51m-tool-sft-v2-agent300m-v1`。
- **narration 前序（2，未采纳）**：`mei-1.0-51m-narration-sft-agent300m-v1`、`mei-1.0-51m-narration-sft-agent300m-v2`。
- **aug 未采纳（1）**：`mei-1.0-51m-tool-sft-natural-aug300m-v1`。
- **gap-pilot40 系列（4，未采纳）**：`mei-1.0-51m-sft-gap-scale-pilot40-sft-{full_call-boundary,multi_step-3_4,mw-disposition,schema-generalization}-v1`。

## 当前 SFT 锁定（v1.3，2026-09-17 决定）

五头 SFT 语料按「用最新的」**统一 Mei 147 skeleton-v2**（依据 SFT-CORPUS-STATUS-SUMMARY.md 第 66/74 行「质量>量，可进产品=Mei 147 小目录口径」）：

| 头 | 语料线 | family | 规模（train/total） |
|---|---|---|---|
| retrieval | Mei 147 skeleton-v2 | retrieval | 2096 / 2760 |
| tool_lm | Mei 147 skeleton-v2 | full_call | 1643 / 2340 |
| disposition | Mei 147 skeleton-v2 | mw_disposition | 8367 / 11800 |
| confidence | Mei 147 skeleton-v2 | confidence | 824 / 1150 |
| narration | Mei 147 skeleton-v2 | narration | 607 / 900 |

skeleton-v2 为 zh-24k-v3 词表时代的纯文本 semantic，重跑需用 hans-en-24k-v1 重新 encode。公开资料线（3388 通用工具）为能力画像探索、不纳入 SFT 输入。

## 混合/待分配

- [2026-09-14-mei-51m-v1.3-cpt-inputs-r01](../pools/frozen-history/2026-09-14-mei-51m-v1.3-cpt-inputs-r01)：本次重训的历史冻结包；简英方案已取代，不是历史 v1.2 训练输入。
- [2026-09-14-mei-51m-v1.3-cpt-inputs-r02](../pools/frozen-history/2026-09-14-mei-51m-v1.3-cpt-inputs-r02)：本次重训的历史冻结包；简英方案已取代，不是历史 v1.2 训练输入。
- [2026-09-14-mei-51m-v1.3-cpt-inputs-r03](../pools/frozen-history/2026-09-14-mei-51m-v1.3-cpt-inputs-r03)：本次重训的历史冻结包；简英方案已取代，不是历史 v1.2 训练输入。
- [2026-09-14-mei-51m-v1.3-cpt-inputs-r04](../pools/frozen-history/2026-09-14-mei-51m-v1.3-cpt-inputs-r04)：本次重训的历史冻结包；简英方案已取代，不是历史 v1.2 训练输入。
- [mei-1.0-51m-tool-sft-v5-rebuild-300mv2-skeleton-v1](../sft-suite/mei-1.0-51m-tool-sft-v5-rebuild-300mv2-skeleton-v1)：superseded；未胜出，保留冻结事实。
- [mei-1.0-51m-tool-sft-v5-rebuild-300mv2-skeleton-v2](../sft-suite/mei-1.0-51m-tool-sft-v5-rebuild-300mv2-skeleton-v2)：**已采用**（v1.3 SFT 五头统一 Mei 147）。
- [mei-1.0-51m-tool-sft-v5-rebuild-300mv2-skeleton-v3](../sft-suite/mei-1.0-51m-tool-sft-v5-rebuild-300mv2-skeleton-v3)：superseded；未胜出，保留冻结事实。
- [mei-1.0-51m-tool-sft-v5-rebuild-300mv2-skeleton-v4](../sft-suite/mei-1.0-51m-tool-sft-v5-rebuild-300mv2-skeleton-v4)：superseded；未胜出，保留冻结事实。
- [mei-1.0-51m-tool-sft-v5-rebuild-300mv2-teacher-v1](../sft-suite/mei-1.0-51m-tool-sft-v5-rebuild-300mv2-teacher-v1)：superseded；teacher 蒸馏分支未胜出。
- [mei-1.0-51m-tool-sft-v5-rebuild-300mv2-teacher-v2](../sft-suite/mei-1.0-51m-tool-sft-v5-rebuild-300mv2-teacher-v2)：superseded；teacher 蒸馏分支未胜出。
- [mei-1.0-51m-tool-sft-v5-rebuild-300mv2-teacher-v3](../sft-suite/mei-1.0-51m-tool-sft-v5-rebuild-300mv2-teacher-v3)：superseded；teacher 蒸馏分支未胜出。
- [mei-1.0-51m-tool-sft-v5-rebuild-300mv2-teacher-v4](../sft-suite/mei-1.0-51m-tool-sft-v5-rebuild-300mv2-teacher-v4)：superseded；teacher 蒸馏分支未胜出。
- [mei-51m-longitudinal-eval-v10-zhv2-rebuild-argnorm](../eval-lock/mei-51m-longitudinal-eval-v10-zhv2-rebuild-argnorm)：历史 Eval lock；逐 run 使用情况待核对。
- [mei-51m-longitudinal-eval-v11-zhv2-rebuild-trajectory](../eval-lock/mei-51m-longitudinal-eval-v11-zhv2-rebuild-trajectory)：历史 Eval lock；逐 run 使用情况待核对。
- [mei-51m-longitudinal-eval-v9-zhv2-rebuild](../eval-lock/mei-51m-longitudinal-eval-v9-zhv2-rebuild)：历史 Eval lock；逐 run 使用情况待核对。

目录证据路径与本次读取的元数据哈希见 [inventory.json](inventory.json)。返回[语料入口](../README.md)。
