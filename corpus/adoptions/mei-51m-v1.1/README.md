# v1.1：归档旧链采用入口

历史资产名 `mei-1.0-51m` 与当前归档名 v1.1 对应，不另算一代。这里引用现有证据，不复制旧语料。

- [300M cycle](../../../cycles/mei-1.1-51m/exp-00300m/CYCLE.json)：累计 300,000,485 token；[CPT](../../../cycles/mei-1.1-51m/exp-00300m/corpus/cpt.json)、[SFT](../../../cycles/mei-1.1-51m/exp-00300m/corpus/sft.json)、[Eval](../../../cycles/mei-1.1-51m/exp-00300m/corpus/eval.json)。
- [600M cycle](../../../cycles/mei-1.1-51m/exp-00600m/CYCLE.json)：累计 600,001,765 token；[CPT](../../../cycles/mei-1.1-51m/exp-00600m/corpus/cpt.json)、[SFT](../../../cycles/mei-1.1-51m/exp-00600m/corpus/sft.json)、[Eval](../../../cycles/mei-1.1-51m/exp-00600m/corpus/eval.json)。

上述数值来自 cycle 记录；两轮产品 release_eligible 均为 false。它们证明历史记录的状态，不证明可直接复用其语料。QAT 的逐 run 使用范围、累计血缘和原始源码缺失情况仍需沿各 cycle 进一步核对。

证据路径、读取时的 SHA256 与待核对范围在 [adoption.json](adoption.json)。此文件是导航快照，不是训练器采用指令。返回[语料入口](../../README.md)。
