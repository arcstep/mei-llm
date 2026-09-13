import {runCheck,sameCall} from './checks.mjs';
export function runNode(model,rule,sheet){
 const session=model.createSession({max_steps:2,runtime_profile:'compact'}),start=performance.now();
  const report={rule,executor:null,check:null,turn:null,narration:null,attempted:true};
  try {
   // The model sees the written assignment, not an injected candidate or expected tool call.
   const turn=session.complete({wire_version:'mei-runtime-wire-v2',query:rule.instruction,context:{columns:sheet.rows[0]},evidence:[],history:[],tool_results:[],permissions:{},state:{},decode_mode:'constrained',max_new:128});
   report.turn=turn;
   if(turn.kind==='call'&&turn.call&&sameCall(rule,turn.call)){
    let payload;
    try {report.check=runCheck(sheet,turn.call.name,turn.call.arguments);report.executor='mei';payload={checked:report.check.checked,issue_count:report.check.issue_count,summary:report.check.issue_count?'发现数据问题':'检查通过'};}
    catch(e){report.error=e.message;}
    session.submitToolResult({wire_version:'mei-runtime-wire-v2',call_id:turn.call.call_id,status:payload?'ok':'error',payload:payload||null,...(!payload?{error:{code:'check_failed',message:report.error}}:{}),provenance:{source:'browser-js-check-tool',verified:true}});
    if(payload)report.followup=session.complete({wire_version:'mei-runtime-wire-v2',query:rule.instruction,context:{},evidence:[],history:[],tool_results:[],permissions:{},state:{},decode_mode:'constrained',max_new:128});
    try{report.narration=session.narrate({mode:'adapter',locale:'zh-CN'});}catch(e){report.narration_error=e.message;}
   }else report.error=turn.kind==='call'?'模型调用与确认的规则不一致，未执行':`Mei 未执行：${turn.refusal?.reason||turn.error?.id||turn.kind}`;
   report.elapsed_ms=performance.now()-start;
  }catch(e){report.error=e.message;report.runtime_error_code=e.code||null;}finally{report.elapsed_ms=performance.now()-start;session.close();}
 return report;
}
