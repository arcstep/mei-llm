# Generators

当前生效的 CPT/SFT/Eval 生成器入口。迁移完成后，历史 `notebook/_tooling/scripts` 中的
生成逻辑按职责收敛到这里；不同规模和不同轮次使用 recipe，不复制脚本。

- `factory_51m.py` + `compiler_v1.py`：factory-v3，human-review-gated 的 worklist/
  campaign 管线，目前只实现 `full_call.boundary`/`schema.generalization`/
  `multi_step.3_4`/`mw.disposition` 四个 pilot cell，从未覆盖 retrieval/no-match/
  confidence/narration。
- `rebuild_zh_v1/`：offline 批量模板生成套件，覆盖 factory-v3 完全没有实现的
  retrieval/no-match/fixed-five 跨批扫描、confidence candidate 冻结、terminal
  narration，以及重新设计过的 full_call/argument-filling、agent 1-4 步、MW
  20-class（含批次可见性 class0↔class10 minimal pair 与 neighbor 边界对）。
  `build.py` 是编排入口；`common.py` 提供工具注册表加载、真实 zh-24k-v1
  tokenizer 预算器（2048 联合预算合同）、hash/split/去重/泄漏扫描；
  `text_variants.py`/`mw_scenarios.py` 是两个 family 共享的中文自然语言与
  20 类场景构造器。当前唯一实际产出：
  `mei-1.0-51m-exp-000600m-sft-zh-rebuild-v1`（绑定 600M base）。复用
  factory-v1 (`compiler_v1.py`) 的 `validate_instance`/`merkle_root`，不复制。
