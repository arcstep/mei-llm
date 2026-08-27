# 工作流 Playbook

## Workflow 0：任务评估与路由

回答当前能否产出真实 artifact、属于哪个 lane、应产出单条/小簇/worklist/治理结论，以及以哪个 family cluster 为锚。需要覆盖信息时先跑本 Skill 的 stats/snapshot 脚本。

## Workflow 1：具体问题 → build sample

1. 确认 `route_mode=build`，而非 runtime QA。
2. 选择已有 family。
3. 使用调用方 `tools_root` 查询未覆盖 task type 并生成 stub。
4. 填写 `instruction`、`context`、`output`、`verifier.checks`。
5. 人工 review 后 dry-run；仅通过后正式发布。

## Workflow 2：Family 扩展

1. 查 family 未覆盖的 task type。
2. 按调用方 maturity tier 优先级排序。
3. 每次选一个 task，保持一条 sample 聚焦一个问题。
4. 通过门控后再发布。

## Workflow 2B：一任务 → 多候选

适用于成熟 family 有许多真源变体或多个相邻 family 共享稳定模式：

1. 跑 corpus stats 与 anchor family snapshot。
2. 定义 batch 目标（首批通常 12～24）。
3. 先产出 candidate worklist，再起草最终 JSON。
4. 按 `task_type` 与 verifier mode 分组。

```json
[
  {
    "candidate_id": "",
    "family_id": "",
    "route_mode": "",
    "task_type": "",
    "source_paths": [],
    "why_it_is_worth_distilling": "",
    "verification_mode": ""
  }
]
```

## Workflow 3：Transcript → 候选任务

Transcript 只作发现源。提取请求形态、源码线索、诊断/repair 信号与接受结果，映射分类字段，再到调用方给定 canonical source 重新 grounding。

中间产物：

```json
{
  "candidate_task": "",
  "family_hint": "",
  "route_mode": "",
  "source_paths": [],
  "needed_context": [],
  "draft_task_type": "",
  "why_it_is_worth_distilling": ""
}
```

## Workflow 4：真实 repair/migrate → sample

确认变更属于稳定 family，识别原始错误、触发诊断、最终 patch 及推荐理由，再选 `build.repair.diagnostic`、`build.repair.fix` 或 `build.migrate.recommended_style`。保持 sample 窄并添加 compile 向 verifier。

## Workflow 5：Docs 向 author 知识 sample

确认 lane 与 task type 不混淆；使用 source-backed assertions，优先 scene summary、structure explainer、migration note、inventory refresh、benchmark note。output 保持紧凑且可追溯。

## Workflow 6：验证优先发布演练

1. 在临时文件创建 draft JSON。
2. 用调用方 append 工具执行 dry-run。
3. review route、目标文件、`task_type`、verifier mode 与重复风险。
4. dry-run 可信后才 promote；dry-run 通过本身不代表 publish-ready。

## 升级规则

依赖未来语法、family 不清、context 无法最小化、答案无法验证或 surface 未成熟时暂停，不发布。
