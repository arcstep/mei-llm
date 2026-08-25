# Phase 1 contract (authoring)

- Schema: `query`, `toolset_id`, `answers`, optional `reasoning`, reserved `act=null`.
- `answers` is a list of `{name, arguments}`. Off-topic **and missing required slots**: `[]`. Never ask the user to fill a slot.
- Scene conflict and illegal closed-set pairs are also `[]` (product gold, not “schema invalid”).
- Train queries must not duplicate any `eval/banks/needle-*/` prompts (isolation `--all`).
- Chinese numbers in train/eval gold are Arabic numerals (`3500` not `三千五百`).
- Slot values are enums / canonical tokens, never the full query string.
- 说话器另线：不得把用户原句当路由输入；`act` 一期可空，勿删列。
- Home-vertical SFT gold is assigned by schema program; a teacher may only write query/scene variants.
