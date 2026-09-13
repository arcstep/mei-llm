import test from 'node:test';import assert from 'node:assert/strict';
import {runCheck,validatePlan,summarize,sameCall} from './checks.mjs';import {SAMPLE} from './catalog.mjs';
const sheet={name:'订单',rows:SAMPLE};
test('real sample catches empty, duplicate, invalid numbers, enum and totals',()=>{
 const cases=[['check_required',{column:'订单号'},1],['check_unique',{column:'订单号'},1],['check_number',{column:'数量',min:1,max:1000},2],['check_product',{quantity:'数量',price:'单价',total:'金额'},2],['check_enum',{column:'状态',allowed:'待付款,已付款,已取消'},1]];
 for(const [name,args,count]of cases)assert.equal(runCheck(sheet,name,args).issue_count,count);
});
test('missing/duplicate columns and empty datasets are not passes',()=>{assert.throws(()=>runCheck(sheet,'check_required',{column:'客户'}));assert.throws(()=>runCheck({name:'x',rows:[['a','a'],[1,2]]},'check_required',{column:'a'}));assert.throws(()=>runCheck({name:'x',rows:[['a']]},'check_required',{column:'a'}));});
test('zero is not missing, row numbers remain physical',()=>{const x=runCheck({name:'x',rows:[['a'],[0],[null]],rowNumbers:[5,6,8]},'check_number',{column:'a',min:1,max:5});assert.equal(x.issues[0].row,6);assert.equal(runCheck({name:'x',rows:[['a'],[0]]},'check_required',{column:'a'}).issue_count,0);});
test('unsupported requirements and refused inference cannot pass',()=>{const rule={id:'r',instruction:'检查订单号',tool:'check_required',args:{column:'订单号'},severity:'block'};assert.equal(summarize([],{rules:[rule],unresolved:[]}).status,'incomplete');assert.equal(summarize([{rule,executor:'human-plan',check:{issue_count:0}}],{rules:[rule],unresolved:['日期未明确']}).status,'incomplete');assert.equal(summarize([{rule,executor:'human-plan',check:{issue_count:0}}],{rules:[rule],unresolved:[]}).mei_completed,0);assert.equal(sameCall(rule,{name:'check_required',arguments:{column:'金额'}}),false);});
test('malformed plans fail closed',()=>{assert.throws(()=>validatePlan({rules:[],unresolved:[]}));assert.throws(()=>validatePlan({rules:[{id:'a',instruction:'x',severity:'block',tool:'shell',args:{}}],unresolved:[]}));assert.throws(()=>runCheck(sheet,'check_number',{column:'数量',min:5,max:1}));});
import {runNode} from './node-task.mjs';
test('node success submits real evidence, then completes before narration',()=>{
 const rule={id:'r',instruction:'检查订单号为空',tool:'check_required',args:{column:'订单号'},severity:'block'};
 const events=[];let step=0;
 const model={createSession(){return {complete(req){assert.equal('candidate_text' in req,false);events.push('complete');return step++?{kind:'respond'}:{kind:'call',call:{name:rule.tool,arguments:rule.args,call_id:'c1'}};},submitToolResult(r){assert.equal(r.payload.issue_count,1);assert.equal(r.provenance.verified,true);events.push('result');},narrate(){events.push('narrate');return {mode:'deterministic',fallback_used:true,text:'测试摘要'};},close(){events.push('close');}};}};
 const result=runNode(model,rule,sheet);assert.equal(result.executor,'mei');assert.deepEqual(events,['complete','result','complete','narrate','close']);assert.equal(result.narration.fallback_used,true);
});
test('wrong model call cannot trigger a tool or narrated result',()=>{
 const rule={id:'r',instruction:'检查订单号',tool:'check_required',args:{column:'订单号'}};let closed=false;
 const result=runNode({createSession:()=>({complete:()=>({kind:'call',call:{name:'check_required',arguments:{column:'金额'}}}),submitToolResult(){throw Error('must not execute');},close(){closed=true;}})},rule,sheet);
 assert.equal(result.check,null);assert.equal(result.executor,null);assert.equal(closed,true);
});
