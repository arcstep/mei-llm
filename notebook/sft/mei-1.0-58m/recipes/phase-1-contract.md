# Phase 1 contract (authoring, Route-ID)

- Schema: `query`, `toolset_id`, optional `scene` / `entities`, reserved `act=null`. Product rows set `schema_conditioned=true`, `serializer=mei-route-serializer-v1`, `protocol=mei-route-protocol-v1`.
- Every request injects tools + compiled `<routes>` at encode time. Callers must pass a toolset; **no default VRM**. Entity registry is per-request input, never baked into weights.
- Internal gold is `[]` or `{"route_id":N}`. External gold is `{name, arguments}` with at most one call. Off-topic, missing required slots, unprovenanced values, ambiguous entities, and scene conflict: `[]`. Never ask the user to fill a slot.
- Scene conflict and illegal closed-set pairs are also `[]` (product gold, not “schema invalid”). Grammar only allows `[]` or a manifest-local route ID.
- Enum values need evidence/const/default. Being listed in schema enum is not enough.
- Train queries must not duplicate any `notebook/evaluation/banks/` prompts (isolation `--all` plus normalized body / entity / alias / counterfactual / manifest checks). Schema SFT splits **whole toolset families** and **holdout entities**, not just paraphrases.
- Chinese numbers in train/eval gold are Arabic numerals (`3500` not `三千五百`).
- Slot values are enums / canonical tokens / normalizer outputs with provenance, never the full query string.
- Product SFT `seq_len=2048`. If tools or routes do not fit, drop the row.
- Teacher/templates may only change natural wording; after rewrite, re-run the same compiler. If provenance breaks, refuse the sample.
- Forbidden surface: `训练登记` / `评测执行` / `holdout` / `闭集检查` / `schema 题` / `#n` cycle ids.
- 说话器另线：不得把用户原句当路由输入；`act` 一期可空，勿删列。
- Home-vertical SFT remains a seen-toolset regression lane. MW five-act packs stay independent until Route-ID mainline passes.
- v1 packs `mei-tool-sft-v1-*` and bank `mei-tool-schema-v1` are frozen `invalid_for_publish`.
