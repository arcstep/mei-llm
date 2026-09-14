import {TOOLS as CHECK_TOOLS} from './catalog.mjs';
import {validateCall} from './checks.mjs';

const checkNames = new Set(CHECK_TOOLS.map(tool => tool.name));
const ACTIONS = new Set(['execute', 'wait', 'clarify', 'review', 'escalate', 'complete', 'stop']);
const TERMINAL = new Set(['passed', 'issues', 'failed', 'cancelled']);

export const PLAN_TOOLS = [
  {
    name: 'plan_add_task',
    description: '向当前检查计划增加一项任务；只修改草案，不执行检查。',
    parameters: {type: 'object', additionalProperties: false,
      required: ['task_id', 'tool', 'arguments', 'after', 'gate'], properties: {
        task_id: {type: 'string'}, tool: {type: 'string'}, arguments: {type: 'object'},
        after: {type: 'array', items: {type: 'string'}},
        gate: {type: 'string', enum: ['all_succeeded', 'all_passed']},
      }},
  },
  {
    name: 'plan_update_task',
    description: '修改当前草案中的一项任务；只修改草案，不执行检查。',
    parameters: {type: 'object', additionalProperties: false,
      required: ['task_id', 'tool', 'arguments', 'after', 'gate'], properties: {
        task_id: {type: 'string'}, tool: {type: 'string'}, arguments: {type: 'object'},
        after: {type: 'array', items: {type: 'string'}},
        gate: {type: 'string', enum: ['all_succeeded', 'all_passed']},
      }},
  },
  {
    name: 'plan_remove_task',
    description: '从当前检查计划草案删除一项任务。',
    parameters: {type: 'object', additionalProperties: false,
      required: ['task_id'], properties: {task_id: {type: 'string'}}},
  },
  {
    name: 'plan_submit',
    description: '提交当前检查计划草案供宿主确认；不执行检查。',
    parameters: {type: 'object', additionalProperties: false, required: [], properties: {}},
  },
];

function sameKeys(value, required) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value).sort().join('\0') === [...required].sort().join('\0');
}

function validateTask(task) {
  if (!sameKeys(task, ['id', 'tool', 'args', 'after', 'gate', 'required'])) throw Error('task_shape');
  if (!task.id || typeof task.id !== 'string') throw Error('task_id');
  if (!checkNames.has(task.tool)) throw Error('unknown_check_tool');
  if (!Array.isArray(task.after) || new Set(task.after).size !== task.after.length) throw Error('dependencies');
  if (!['all_succeeded', 'all_passed'].includes(task.gate)) throw Error('gate');
  if (typeof task.required !== 'boolean') throw Error('required');
  validateCall(task.tool, task.args);
}

export function validatePlan(plan) {
  if (!plan || plan.schema !== 'mei-node-plan-v1' || !Number.isInteger(plan.revision) || plan.revision < 1) throw Error('plan_header');
  if (!Array.isArray(plan.tasks) || plan.tasks.length > 6) throw Error('task_count');
  for (const task of plan.tasks) validateTask(task);
  const ids = plan.tasks.map(task => task.id);
  if (new Set(ids).size !== ids.length) throw Error('duplicate_task');
  const byId = new Map(plan.tasks.map(task => [task.id, task]));
  for (const task of plan.tasks) if (task.after.some(id => !byId.has(id) || id === task.id)) throw Error('unknown_dependency');
  const visiting = new Set(), visited = new Set();
  function visit(id) {
    if (visiting.has(id)) throw Error('dependency_cycle');
    if (visited.has(id)) return;
    visiting.add(id);
    for (const dependency of byId.get(id).after) visit(dependency);
    visiting.delete(id); visited.add(id);
  }
  for (const id of ids) visit(id);
  const depth = new Map();
  function taskDepth(id) {
    if (depth.has(id)) return depth.get(id);
    const value = 1 + Math.max(0, ...byId.get(id).after.map(taskDepth));
    depth.set(id, value); return value;
  }
  if (ids.some(id => taskDepth(id) > 3)) throw Error('dependency_depth');
  return plan;
}

export function newDraft(revision = 1) {
  return {schema: 'mei-node-plan-v1', revision, status: 'draft', tasks: []};
}

export function applyPlanCall(plan, call) {
  validatePlan(plan);
  const next = structuredClone(plan);
  const args = call?.arguments;
  if (!PLAN_TOOLS.some(tool => tool.name === call?.name)) throw Error('unknown_plan_tool');
  if (call.name === 'plan_submit') {
    if (!sameKeys(args, [])) throw Error('submit_arguments');
    if (!next.tasks.length) throw Error('empty_plan');
    next.status = 'submitted'; next.revision += 1;
    return validatePlan(next);
  }
  if (call.name === 'plan_remove_task') {
    if (!sameKeys(args, ['task_id'])) throw Error('remove_arguments');
    if (next.tasks.some(task => task.after.includes(args.task_id))) throw Error('task_has_dependants');
    const length = next.tasks.length;
    next.tasks = next.tasks.filter(task => task.id !== args.task_id);
    if (next.tasks.length === length) throw Error('task_not_found');
  } else {
    if (!sameKeys(args, ['task_id', 'tool', 'arguments', 'after', 'gate'])) throw Error('edit_arguments');
    const task = {id: args.task_id, tool: args.tool, args: args.arguments,
      after: args.after, gate: args.gate, required: true};
    validateTask(task);
    const index = next.tasks.findIndex(item => item.id === args.task_id);
    if (call.name === 'plan_add_task') {
      if (index >= 0) throw Error('duplicate_task');
      next.tasks.push(task);
    } else {
      if (index < 0) throw Error('task_not_found');
      next.tasks[index] = task;
    }
  }
  next.revision += 1;
  return validatePlan(next);
}

export function readyTasks(plan, states = {}) {
  validatePlan(plan);
  return plan.tasks.filter(task => (states[task.id] ?? 'pending') === 'pending' && task.after.every(id => {
    const state = states[id];
    return task.gate === 'all_passed' ? state === 'passed' : ['passed', 'issues'].includes(state);
  })).map(task => task.id).sort();
}

export function disposition(view) {
  const action = view?.action;
  if (!ACTIONS.has(action)) throw Error('unknown_disposition');
  if (!view.reason || typeof view.reason !== 'string') throw Error('missing_reason');
  if (!Array.isArray(view.task_ids) || !Array.isArray(view.evidence_refs)) throw Error('disposition_references');
  if (view.task_ids.some(id => typeof id !== 'string') || view.evidence_refs.some(id => typeof id !== 'string')) throw Error('disposition_reference_type');
  return {schema: 'mei-node-disposition-v1', action, reason: view.reason,
    task_ids: [...view.task_ids], collaboration: view.collaboration ?? null,
    evidence_refs: [...view.evidence_refs]};
}

export function evaluateState(plan, states, options = {}) {
  if (options.requiresApproval && options.approvalGranted !== true) {
    return disposition({action: 'review', reason: 'approval_required',
      task_ids: plan.tasks.filter(task => (states[task.id] ?? 'pending') === 'pending').map(task => task.id),
      collaboration: 'human', evidence_refs: options.evidenceRefs ?? []});
  }
  const ready = readyTasks(plan, states);
  if (ready.length) return disposition({action: 'execute', reason: 'ready_tasks', task_ids: ready, evidence_refs: []});
  const required = plan.tasks.filter(task => task.required);
  if (required.length && required.every(task => states[task.id] === 'passed')) {
    return disposition({action: 'complete', reason: 'all_required_passed',
      task_ids: required.map(task => task.id), evidence_refs: options.evidenceRefs ?? []});
  }
  if (required.some(task => states[task.id] === 'failed')) return disposition({action: 'escalate',
    reason: 'required_task_failed', task_ids: required.filter(task => states[task.id] === 'failed').map(task => task.id),
    collaboration: 'human_or_large_model', evidence_refs: options.evidenceRefs ?? []});
  if (required.some(task => states[task.id] === 'cancelled')) return disposition({action: 'stop',
    reason: 'required_task_cancelled', task_ids: required.filter(task => states[task.id] === 'cancelled').map(task => task.id), evidence_refs: []});
  return disposition({action: options.offline ? 'wait' : 'review',
    reason: options.offline ? 'offline_assistance_queued' : 'no_ready_task',
    task_ids: required.filter(task => !TERMINAL.has(states[task.id])).map(task => task.id),
    collaboration: options.offline ? 'queued_coordinator' : 'human_or_large_model', evidence_refs: options.evidenceRefs ?? []});
}

export function narrate(view) {
  if (!view || view.verified !== true || !Array.isArray(view.results)) throw Error('unverified_narration_input');
  const completed = view.results.length;
  const issues = view.results.reduce((sum, item) => sum + Number(item.issue_count ?? 0), 0);
  const failed = view.results.filter(item => item.status === 'failed').length;
  if (failed) return `已完成 ${completed - failed} 项检查，${failed} 项执行失败，需要复核。`;
  if (issues) return `已完成 ${completed} 项检查，共发现 ${issues} 个问题，请修正后重新检查。`;
  return `已完成 ${completed} 项检查，未发现问题。`;
}
