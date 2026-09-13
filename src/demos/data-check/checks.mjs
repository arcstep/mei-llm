import {TOOLS} from './catalog.mjs';
export function validateCall(name,args) {
 const t=TOOLS.find(t=>t.name===name);if(!t)throw Error('未知检查工具：'+name);
 if(!args||typeof args!=='object'||Array.isArray(args))throw Error('工具参数必须是对象');
 const props=t.parameters.properties;
 if(Object.keys(args).length!==Object.keys(props).length)throw Error('工具参数不完整');
 for(const [k,s] of Object.entries(props)) {
  if(typeof args[k]!==s.type || (s.type==='number'&&!Number.isFinite(args[k])) || (s.type==='string'&&(!args[k].trim()||args[k].length>200)))throw Error('无效参数：'+k);
 }
 if(name==='check_number'&&args.min>args.max)throw Error('数字上下界颠倒');
 return t;
}
export function validatePlan(plan) {
 if(!plan||!Array.isArray(plan.rules)||!Array.isArray(plan.unresolved)||plan.rules.length>12)throw Error('规划格式无效（最多12项检查）');
 const ids=new Set();
 for(const r of plan.rules) {
  if(typeof r.id!=='string'||!r.id||ids.has(r.id))throw Error('规则编号无效');ids.add(r.id);
  if(typeof r.instruction!=='string'||!r.instruction.trim()||r.instruction.length>400)throw Error('检查要求无效');
  if(!['block','warn'].includes(r.severity))throw Error('处置级别无效');validateCall(r.tool,r.args);
 }
 if(plan.unresolved.some(x=>typeof x!=='string'||x.length>500))throw Error('未解决要求格式无效');
 if(!plan.rules.length&&!plan.unresolved.length)throw Error('没有可执行的检查要求');return plan;
}
const blank=v=>v===null||v===undefined||typeof v==='string'&&!v.trim();
const number=v=>typeof v==='number'&&Number.isFinite(v)?v:typeof v==='string'&&/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(v.trim())?Number(v):null;
export function runCheck(sheet,name,args) {
 validateCall(name,args);
 const headers=sheet.rows[0]?.map(x=>String(x??'').trim())||[];
 const cols=Object.entries(args).filter(([k])=>!['min','max','allowed'].includes(k));
 const ix={};
 for(const [k,v] of cols){const all=headers.flatMap((x,i)=>x===v?[i]:[]);if(all.length!==1)throw Error(`列「${v}」${all.length?'重复，无法定位':'不存在'}，需要修改要求或表头`);ix[k]=all[0];}
 const issues=[],seen=new Map();let checked=0;
 const issue=(row,column,value,message)=>issues.push({sheet:sheet.name,row,column,value,message});
 for(let i=1;i<sheet.rows.length;i++){
  const row=sheet.rows[i];if(row.every(blank))continue;checked++;const n=(sheet.rowNumbers?.[i]??i+1);
  const v=row[ix.column];
  if(name==='check_required'&&blank(v))issue(n,args.column,v,'必填内容为空');
  if(name==='check_unique'&&!blank(v)) {const key=JSON.stringify([typeof v,typeof v==='string'?v.trim():v]);if(seen.has(key))issue(n,args.column,v,`与第 ${seen.get(key)} 行重复`);else seen.set(key,n);}
  if(name==='check_number'){const num=number(v);if(num===null||num<args.min||num>args.max)issue(n,args.column,v,`应为 ${args.min} 至 ${args.max} 的数字`);}
  if(name==='check_enum'&&!args.allowed.split(',').map(x=>x.trim()).includes(String(v??'').trim()))issue(n,args.column,v,`允许值：${args.allowed}`);
  if(name==='check_product') {const q=number(row[ix.quantity]),p=number(row[ix.price]),total=number(row[ix.total]);if(q===null||p===null||total===null)issue(n,args.total,row[ix.total],'数量、单价或金额不是有效数字，无法核算');else if(Math.abs(q*p-total)>0.010000001)issue(n,args.total,total,`数量 × 单价 = ${Number((q*p).toFixed(6))}，与金额不符`);}
 }
 if(!checked)throw Error('工作表没有可检查的数据行');
 return {tool:name,args,checked,issue_count:issues.length,issues};
}
export const sameCall=(r,c)=>r.tool===c.name&&Object.keys(r.args).length===Object.keys(c.arguments||{}).length&&Object.entries(r.args).every(([k,v])=>c.arguments[k]===v);
export function summarize(results,plan) {
 const done=results.filter(r=>r.check),pending=plan.rules.length-done.length;
 const blocked=done.some(r=>r.rule.severity==='block'&&r.check.issue_count>0);
 const warnings=done.some(r=>r.check.issue_count>0);
 return {status:pending||plan.unresolved.length?'incomplete':blocked?'blocked':warnings?'warning':'passed',pending,issue_count:done.reduce((n,r)=>n+r.check.issue_count,0),mei_completed:done.filter(r=>r.executor==='mei').length,total:plan.rules.length};
}
export function describeRule(r){const a=r.args;switch(r.tool){case 'check_required':return `「${a.column}」不能为空`;case 'check_unique':return `「${a.column}」不能重复`;case 'check_number':return `「${a.column}」为 ${a.min} 至 ${a.max} 的数字（含边界）`;case 'check_enum':return `「${a.column}」只允许：${a.allowed}`;case 'check_product':return `「${a.quantity}」×「${a.price}」=「${a.total}」，误差不超过0.01`;}}
