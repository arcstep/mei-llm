# 语料治理规则

本文件固化语料（CPT/SFT）治理的规则约定，任何开发 Agent 与训练流程都必须遵守。违反红线会重演
2026-09-17 发现的 v1.3-800m SFT 语料治理失败（productize 主线误用旧语料 `v4-300m-v4`）。

## 单一真相源

- 「当前该用哪个语料」**只读** `corpus/adoptions/mei-51m-v1.3/adoption.json` 的
  `current_locked_inputs`（cpt/sft/eval，每头指向锁定相对位置+sha256）。
- 不得从文件名 vN、目录修改时间、`sft-suite/` 平铺顺序推断「最新」。
- 不新造第二套真相源（例如 current-release.json）；采用关系一律写进 adoption.json。

## 四态与状态标记

- 四态：**备选 / 合格候选 / 训练锁定 / 实际使用**（见 `corpus/README.md`「四种状态必须分开」）。
- SFT 编译包用原地 `STATUS.json` 标记：`adopted`（已采用，`locked:true`）或 `superseded`
  （被取代，保留冻结事实）。任务试批同理。

## 锁定与不可就地修改

- 被采纳语料锁定后**禁止就地修改、重命名、删除**；变更走新 release + 显式 reuse/replace/retire。
- 锁定锚点 = `manifest.json` / `release-manifest.json` 的 sha256，在 adoption.json 内声明。
- 历史训练输入 release（如 cycles 里 historical-notebook-releases/releases）被 receipt 以哈希引用，
  原地不动（迁移会毁坏历史 receipt 可复现性，AGENTS.md fail-closed）。

## 成果落位（四分离）

- 提取/合成的 SFT 语料成果发布到 `corpus/`（`sft-suite/`、`pools/task-trials/`），不留 cycles。
- cycles 转移成果后只留说明（README 指向 corpus registry），不长期保存大文件。
- `models/` 不保存语料；models 产物必须**自包含**（架构版本、CPT 语料、SFT 语料、验证情况），
  不从 cycles 间接翻找。SFT 产物 STATUS.md 必须声明「SFT 语料」来源。

## 脚本默认值红线

- 禁止把语料版本默认值硬编码进脚本（只会默认用最旧版本）。
- `productize_adaptive_v5_51m.py` 的语料/权重输入已改为必填，缺失 fail-fast（报错指向 adoption.json）。
- `freeze_*.py` 与 CPT 脚本的默认值只登记、不改，避免影响在跑 run。

## 2026-09-17 决定记录

- v1.3 SFT 五头语料锁定到日期命名新 release **`mei-1.3-51m-tool-sft-20260917-fivehead-v1`**
  （`sft-suite/` 下，独立自包含、锁定不可改）：tool_lm/retrieval = 公开线 4k（脱离 Mei 147）；
  confidence/disposition 样本克隆自 skeleton-v2（绕不开的样本底子，克隆为独立文件）；
  narration = freeze v2 代码合成。彻底切断与 skeleton-v2 的命名/绑定关系。
- 公开资料线（`pools/task-trials/2026-09-15-public-sft-scale-experiment/`，3388 通用工具）的
  tool_lm/retrieval 4k 已作为新 release 的这两头纳入 SFT 输入。
- v1.3-800m 的 productize 主线（`mei-51m-v1.3-800m-tool-sft-cq2-v2-adaptive-v5`）误用旧语料
  `v4-300m-v4`，已在其 STATUS.md 标记治理失败，需用 `mei-1.3-51m-tool-sft-20260917-fivehead-v1` 语料重跑。
