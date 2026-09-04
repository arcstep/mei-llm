# Compatibility migration

旧生命周期把 CPT 与产品化放在同一 recipe 的 `cpt/product/full` tracks。
当前边界为：

- `mei-51m-cpt-lifecycle-v1` pipeline 只消费 `cpt-training-v1.json`；
- `mei-51m-adaptive-productization-v5` 独立消费
  `productization-adaptive-v5.json`；
- cycle 只协调两个 pipeline 和三类 corpus Skills；
- compatibility route 不产生 receipt、权重或 release。

待所有调用方改用六个专职 Skill 后可删除此兼容包。
