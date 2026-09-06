# Alignment and longitudinal comparison

这里回答两个跨域问题：不同 exposure cycle 是否在同合同下改善，以及当前机制与 Needle2
参考实现对齐到哪里。

- `compare_longitudinal_products_51m.py`：300M/600M/后续 rung 的可比性与 confound 审计；
- `needle2_alignment_51m.py`：机制对齐矩阵，不声明 Cactus ABI/API 兼容。

只有 eval lock、runtime source、语料差异和 lineage confound 都显式记录时，才允许把差值
归因于 Base exposure。
