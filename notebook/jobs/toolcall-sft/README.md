# toolcall-sft

唯一落点：`notebook/jobs/toolcall-sft/`。三账三 Job：retrieval / full-call / MW。

- fleet：`notebook/sft/mei-1.0-51m/recipes/sft-synth-fleet-v1.json`
- 计量单位：accepted unique row
- 教师只改写 query；gold 由 compiler/validator 确定
- 付费 bake-off / 扩量必须另确认预算和 model snapshot
