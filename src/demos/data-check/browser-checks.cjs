const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/xuehongwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{const b=await chromium.launch({headless:true});const p=await b.newPage();await p.goto('http://127.0.0.1:8765');
const result=await p.evaluate(async()=>{
 const w=new Worker('/parse-worker.js');let seq=0;
 const rpc=data=>new Promise((resolve,reject)=>{const id=++seq;w.onmessage=e=>e.data.error?reject(Error(e.data.error)):resolve(e.data);w.postMessage({...data,id});});
 const checks=[];
 for(const format of ['xlsx','biff8']){const rows=[['姓名','金额'],['小明',123.45],['小红',0]];const x=await rpc({sample:true,rows,format});const y=await rpc({buffer:x.buffer});checks.push({format,rows:y.sheets[0].rows});}
 const csv=await rpc({buffer:new TextEncoder().encode('\ufeff姓名,备注\r\n小明,"含,逗号"\r\n小红,"第一行\n第二行"').buffer});
 checks.push({format:'csv',rows:csv.sheets[0].rows});w.terminate();return checks;
});
assert.deepEqual(result[0].rows,[['姓名','金额'],['小明',123.45],['小红',0]]);assert.deepEqual(result[1].rows,result[0].rows);assert.equal(result[2].rows[1][1],'含,逗号');assert.equal(result[2].rows[2][1],'第一行\n第二行');
const api=await p.evaluate(async()=>({env:(await fetch('/.env')).status,traversal:(await fetch('/assets/../server.mjs')).status,post:(await fetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).status,config:await(await fetch('/api/config')).json()}));
assert.equal(api.env,404);assert.equal(api.traversal,404);assert.equal(api.post,403);assert.equal('key' in api.config.planner,false);assert.equal('url' in api.config.planner,false);
await p.setViewportSize({width:390,height:844});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
console.log(JSON.stringify({ok:true,parsers:result.map(r=>r.format),local_server_boundary:true,mobile_no_overflow:true}));await b.close();})();
