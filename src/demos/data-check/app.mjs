import {EXAMPLE,SAMPLE} from './catalog.mjs';
import {validatePlan,runCheck,summarize,describeRule} from './checks.mjs';
import {save,load} from './storage.mjs';
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let config,plan,confirmed=false,book=[],fileMeta=null,fileBytes=null,results=[],audit=null,worker=null,parser=null,busy=false,epoch=0,requestId=0,activeRule=null,planHash=null;
const trace=[];let toastTimer;
const notify=text=>{$('notice').textContent=text;$('notice').style.display='block';clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('notice').style.display='none',6000);};
const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(x=>x.toString(16).padStart(2,'0')).join('');
const selected=()=>book[Number($('sheet').value)||0];
function log(kind,value){trace.push({time:new Date().toISOString(),kind,value});$('trace').textContent=JSON.stringify(trace,null,2);}
function rpc(w,data,timeout=120000){return new Promise((resolve,reject)=>{const id=++requestId;const timer=setTimeout(()=>{cleanup();w.terminate();if(w===worker)worker=null;if(w===parser)parser=null;reject(Error('运行超时，Worker 已停止；没有将未完成项算作通过'));},timeout);const handler=e=>{if(e.data.id!==id)return;cleanup();e.data.error?reject(Error(e.data.error)):resolve(e.data);};const err=e=>{cleanup();reject(Error(e.message||'Worker 异常'));};function cleanup(){clearTimeout(timer);w.removeEventListener('message',handler);w.removeEventListener('error',err);}w.addEventListener('message',handler);w.addEventListener('error',err);w.postMessage({...data,id});});}
function parseRPC(data){parser??=new Worker('/parse-worker.js');return rpc(parser,data,30000);}
function setBusy(value){busy=value;for(const id of ['plan','example','confirm','file','sheet','assist','sample','clean-sample'])$(id).disabled=value;$('requirement').disabled=value;$('cancel').hidden=!value;$('run').disabled=value||!confirmed||!selected();$('export').disabled=value;$('submit').disabled=value;}
function updatePlan(){
 $('plan-view').innerHTML=plan?`<h3>${esc(plan.title||'检查安排')}</h3>`+plan.rules.map(r=>`<div class="rule"><b>${esc(describeRule(r))}</b><small>${r.severity==='warn'?'发现问题时提示':'发现问题时阻止提交'}</small></div>`).join('')+plan.unresolved.map(x=>`<div class="issue">待明确：${esc(x)}</div>`).join(''):'';
 $('confirm').hidden=!plan||confirmed;$('confirm').textContent=plan?.unresolved.length?'确认已明确部分（其余仍待处理）':'确认安排';render();
}
function render(){
 const summary=plan?summarize(results,plan):null;
 const formulaPending=!!selected()?.formulas.length;
 const status=summary?.status==='passed'&&formulaPending?'incomplete':summary?.status;
 const labels={incomplete:'仍有检查未完成',blocked:'发现需要修正的问题',passed:'本轮检查通过',warning:'检查完成，有提示项'};
 $('summary').className='summary '+(status||'');
 $('summary').textContent=busy?`Mei 正在执行 ${activeRule??'准备'}，页面可继续响应…`:results.length?`${labels[status]} · ${summary.mei_completed}/${summary.total} 项由 Mei 完成 · ${summary.issue_count} 条问题记录${formulaPending?' · 存在公式，缓存值需人工确认':''}`:'确认检查安排后，选择文件即可自动开始。';
 $('results').innerHTML=results.map(r=>`<article class="result"><h4>${esc(r.rule.instruction)}</h4><span class="badge">${r.executor==='mei'?'Mei 调用工具':r.executor==='human-plan'?'用户确认的计划补做':'Mei 未完成'}</span>${r.elapsed_ms?`<span class="badge">${(r.elapsed_ms/1000).toFixed(1)} 秒</span>`:''}<p>${esc(r.check?`检查了 ${r.check.checked} 行，发现 ${r.check.issue_count} 条问题。`:r.error)}</p>${r.check?`<small>以上为工具证据的确定性摘要</small>`:''}${r.narration?`<p>${esc(r.narration.text)}</p><small>SDK 解读：${r.narration.mode}${r.narration.fallback_used?'（已回退为确定性文本）':''}</small>`:''}${r.check?.issues.slice(0,30).map(x=>`<div class="issue">${esc(x.sheet)} · 第 ${x.row} 行 · ${esc(x.column)}：${esc(x.message)}${x.value!=null?`（当前：${esc(x.value)}）`:''}</div>`).join('')||''}${r.check?.issues.length>30?'<small>页面展示前30条，完整问题见导出报告。</small>':''}</article>`).join('');
 $('nodes').innerHTML=(plan?.rules||[]).map((r,i)=>{const result=results.find(x=>x.rule.id===r.id);return `<div class="node ${activeRule===r.id?'running':result?.check?'done':result?'pending':''}"><span class="dot"></span><div>检查员 ${String(i+1).padStart(2,'0')}<br>${esc(r.instruction)}<small>${activeRule===r.id?'推理中':result?.check?'已完成':result?'等待协助':'等待文件'}</small></div></div>`;}).join('');
 $('assist').hidden=!results.length||!(plan.rules.length>results.filter(r=>r.check).length);$('export').hidden=!results.length;
 $('submit').hidden=status!=='passed'||!results.length;
 $('run').disabled=busy||!confirmed||!selected();
 $('metrics').innerHTML=plan?`<p>主节点规划：${audit?'1 次':'已恢复配置'}${audit?` · ${(audit.ms/1000).toFixed(1)} 秒`:''}</p><p>本轮 Mei 推理：${results.reduce((n,x)=>n+Number(!!x.attempted||!!x.turn)+Number(!!x.followup),0)} 次</p><p>计划补做：${results.filter(x=>x.executor==='human-plan').length} 项</p><p>表格发送给云模型：0 字节</p>`:'等待配置';
}
async function persist(){await save('workspace',{plan,confirmed,book,fileMeta,fileBytes,results,audit,planHash,requirement:$('requirement').value,package:config?.package,sheetIndex:Number($('sheet').value)||0});}
function fileView(){
 $('sheet').innerHTML=book.map((s,i)=>`<option value="${i}">${esc(s.name)}</option>`).join('');$('sheet-label').hidden=!book.length;
 $('file-info').textContent=fileMeta?`${fileMeta.name} · ${book.length} 个工作表 · ${Math.max(0,selected()?.rows.length-1)} 行数据`:'未选择文件';
}
$('example').onclick=()=>{$('requirement').value=EXAMPLE;invalidatePlan();};
function invalidatePlan(){epoch++;confirmed=false;plan=null;results=[];planHash=null;updatePlan();void persist();}
$('requirement').oninput=invalidatePlan;
$('plan').onclick=async()=>{
 setBusy(true);$('cancel').hidden=true;$('plan-error').hidden=true;notify('主节点正在把书面要求拆成检查岗位…');
 try {config=await fetch('/api/config').then(r=>r.json());const r=await fetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json','X-Demo-Token':config.token},body:JSON.stringify({requirement:$('requirement').value})});const data=await r.json();if(!r.ok)throw Error(data.error);plan=validatePlan(data.plan);audit=data.audit;confirmed=false;results=[];log('planning',data);updatePlan();await persist();notify('检查安排已生成，请确认。');}
 catch(e){$('plan-error').textContent=e.message;$('plan-error').hidden=false;notify(e.message);}finally{setBusy(false);render();}
};
$('confirm').onclick=async()=>{try{validatePlan(plan);planHash=await hash(new TextEncoder().encode(JSON.stringify(plan)));confirmed=true;updatePlan();await persist();notify('安排已确认；上传后将自动体检。');if(selected())await run();}catch(e){notify(e.message);}};
async function ingest(file){
 if(!file||busy)return;const current=++epoch;setBusy(true);$('cancel').hidden=true;results=[];book=[];fileMeta=null;fileBytes=null;render();
 try{if(!/\.(xlsx|xls|csv)$/i.test(file.name))throw Error('请选择 xlsx、xls 或 csv 文件');if(file.size>10*1024*1024)throw Error('演示版文件上限10 MiB');const buffer=await file.arrayBuffer();const data=await parseRPC({buffer});if(epoch!==current)return;book=data.sheets;fileBytes=buffer;fileMeta={name:file.name,size:file.size,sha256:await hash(buffer)};fileView();await persist();notify('文件已在浏览器内解析。');}
 catch(e){notify(e.message);}finally{setBusy(false);render();}
 if(confirmed&&selected())await run();
}
$('file').onchange=e=>ingest(e.target.files[0]);
$('drop').ondragover=e=>e.preventDefault();$('drop').ondrop=e=>{e.preventDefault();ingest(e.dataTransfer.files[0]);};
$('sheet').onchange=()=>{results=[];$('file-info').textContent=`${fileMeta.name} · 当前工作表 ${selected().name} · ${Math.max(0,selected().rows.length-1)} 行数据`;render();void persist();};
async function downloadSample(clean=false){try{const rows=clean?[SAMPLE[0],...Array.from({length:8},(_,i)=>['B'+String(i+1).padStart(3,'0'),i+1,20,(i+1)*20,'已付款'])]:SAMPLE;const data=await parseRPC({sample:true,rows});download(data.buffer,clean?'修正示例.xlsx':'问题示例.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');}catch(e){notify(e.message);}}
$('sample').onclick=()=>downloadSample();$('clean-sample').onclick=()=>downloadSample(true);
function toBase64(buffer){if(!buffer)throw Error('原文件未保存，请重新选择文件');const bytes=new Uint8Array(buffer);let text='';for(let i=0;i<bytes.length;i+=32768)text+=String.fromCharCode(...bytes.subarray(i,i+32768));return btoa(text);}
function download(data,name,type='application/json'){const url=URL.createObjectURL(new Blob([data],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function run(){
 if(busy||!confirmed||!selected())return;const current=++epoch;results=[];setBusy(true);render();
 try{
  worker??=new Worker('/mei-worker.mjs',{type:'module'});$('runtime-status').textContent='加载真实 900M SFT CQ2 / WASM…';await rpc(worker,{type:'init'});if(current!==epoch)return;$('runtime-status').textContent='Mei 已加载 · 单 Worker 串行 · 2048 token 契约';
  for(const rule of plan.rules){if(current!==epoch)break;activeRule=rule.id;render();let report;try{const data=await rpc(worker,{rule,sheet:selected()});report=data.report;}catch(e){report={rule,executor:null,check:null,error:e.message};}if(current!==epoch)break;results.push(report);log('mei-node',report);await persist();if(!worker)break;}
 }catch(e){if(current===epoch){notify(e.message);for(const rule of plan.rules)if(!results.some(r=>r.rule.id===rule.id))results.push({rule,executor:null,check:null,error:e.message});}}
 finally{if(current===epoch){activeRule=null;setBusy(false);render();await persist();}}
}
$('run').onclick=run;
$('cancel').onclick=()=>{epoch++;worker?.terminate();worker=null;activeRule=null;setBusy(false);render();notify('已停止。已完成记录保留，其余仍未完成。');void persist();};
$('assist').onclick=async()=>{
 for(const rule of plan.rules){let r=results.find(x=>x.rule.id===rule.id);if(r?.check)continue;if(!r){r={rule};results.push(r);}try{r.check=runCheck(selected(),rule.tool,rule.args);r.executor='human-plan';r.assistance='explicit-user-plan-execution';}catch(e){r.error=e.message;}}
 log('human-plan-assistance',{rules:results.filter(x=>x.executor==='human-plan').map(x=>x.rule.id)});render();await persist();
};
$('export').onclick=()=>download(JSON.stringify({schema:'mei-data-check-report-v1',file:fileMeta,plan_sha256:planHash,plan,planning:audit,package:config?.package,summary:{...summarize(results,plan),...(selected().formulas.length?{status:'incomplete',formula_review_required:true}:{})},scope:{sheet:selected().name,formula_cache_pending:selected().formulas.length>0},results,trace},null,2),'mei-体检报告.json');
$('submit').onclick=async()=>{try{const summary=summarize(results,plan);if(summary.status!=='passed'||selected().formulas.length)throw Error('体检未通过');config=await fetch('/api/config').then(r=>r.json());const response=await fetch('/api/submit',{method:'POST',headers:{'Content-Type':'application/json','X-Demo-Token':config.token},body:JSON.stringify({file_sha256:fileMeta.sha256,plan_sha256:planHash,status:'passed',sheet:selected().name,file_base64:toBase64(fileBytes)})});const receipt=await response.json();if(!response.ok)throw Error(receipt.error);log('local-demo-receipt',receipt);notify('本机收集端复检通过，文件已接收：'+receipt.id);$('submit-info').textContent=`回执 ${receipt.id}；本机收集端已复检并保存文件。`;}catch(e){notify('回执未发送，报告已保留本地：'+e.message);}};
async function boot(){
 try{config=await fetch('/api/config').then(r=>r.json());$('planner-info').textContent=`规划模型：${config.planner.model}。只发送书面要求，表格留在本地。`;}catch{$('connection').textContent='离线工作空间';}
 try{const old=await load('workspace');if(old){if(!config&&old.package)config={package:old.package};plan=old.plan;confirmed=old.confirmed;book=old.book||[];fileMeta=old.fileMeta;fileBytes=old.fileBytes;results=old.package?.wasm_sha256&&config?.package?.wasm_sha256&&old.package.wasm_sha256!==config.package.wasm_sha256?[]:old.results||[];audit=old.audit;planHash=old.planHash;$('requirement').value=old.requirement||'';fileView();$('sheet').value=old.sheetIndex||0;updatePlan();notify(old.package?.wasm_sha256&&config?.package?.wasm_sha256&&old.package.wasm_sha256!==config.package.wasm_sha256?'运行时已切换，已保留文件与规则，请重新检查。':'已恢复本地任务与检查记录。');}else $('requirement').value=EXAMPLE;}catch(e){notify('本地记录无法恢复：'+e.message);}
 if(config?.package?.runtime_experiment)$('runtime-profile').textContent='速度实验版 · 近似计算 · 固定前缀缓存 · 质量未验收';
 if('serviceWorker'in navigator){try{await navigator.serviceWorker.register('/sw.js');await navigator.serviceWorker.ready;$('connection').textContent='本地工作空间 · 离线资源已缓存';}catch{$('connection').textContent='本地工作空间 · 离线缓存未完成';}}
 render();
}
window.addEventListener('online',()=>$('connection').textContent='本地工作空间 · 已联网');window.addEventListener('offline',()=>$('connection').textContent='离线工作空间 · 使用已确认安排');
boot();
