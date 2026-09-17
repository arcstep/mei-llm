# mei-51m v1.3 at 800M

v1.3 首个 CPT 阶段观察点（800,000,000 token）。目录按产品化顺序逐步沉淀不可变成果。

## 产物清单

| 产物 | 类型 | 自包含语料声明 | 状态 |
|---|---|---|---|
| [`base/mei-51m-v1.3-800m-base/`](base/mei-51m-v1.3-800m-base/) | CPT milestone | RELEASE.json 内 `input_release` 指向 CPT 冻结语料（`corpus/pools/frozen/2026-09-14-mei-51m-v1.3-cpt-25b-r01`，sha256 已声明） | `copied_verified_diagnostic_complete`，release_eligible=false |
| [`products/mei-51m-v1.3-800m-base-cq2-qat-r01/`](products/mei-51m-v1.3-800m-base-cq2-qat-r01/) | QAT 结果 | RELEASE.json 内 `base_release_id` 链到 base | `copied_hash_verified`，quality_passed=true |
| [`products/mei-51m-v1.3-800m-tool-sft-cq2-v2-adaptive-v5/`](products/mei-51m-v1.3-800m-tool-sft-cq2-v2-adaptive-v5/) | 五头 SFT | **治理失败**：productize 主线用了旧语料 `v4-300m-v4`，未切到 v1.3 两条线拼接 | **需重跑** |

## 语料真相源

当前该用哪个语料，只读 [`corpus/adoptions/mei-51m-v1.3/adoption.json`](../../../../corpus/adoptions/mei-51m-v1.3/adoption.json) 的 `current_locked_inputs`（cpt/sft/eval，每头指向锁定相对位置+哈希），不按目录名 vN 或修改时间推断「最新」。

## 说明

- base / QAT 产物已自包含：架构、CPT 语料、验证均在各自 RELEASE.json 内声明，不依赖 cycles 翻找。
- SFT 产物因语料治理失败需重跑（正确语料 = 两条线拼接：retrieval/tool_lm 公开资料线 + 三头 Mei 147 skeleton-v2，见其 STATUS.md「SFT 语料」节）。
