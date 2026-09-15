// Samples the unchanged delivery binary. No provider requests, model edits or gate overrides.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/xuehongwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs/promises');
const os=require('node:os');
function stages(profile){
 const nodes=new Map(profile.nodes.map(n=>[n.id,n])),parents=new Map();
 for(const n of profile.nodes)for(const c of n.children||[])parents.set(c,n.id);
 const us={retrieval:0,forward_bounded:0,grammar:0,other_wasm:0,host_and_profiler:0};
 for(let i=0;i<profile.samples.length;i++){
  let id=profile.samples[i];const names=[];
  while(nodes.has(id)){names.push(nodes.get(id).callFrame.functionName);id=parents.get(id);}
  const stack=names.join(' ');
  const key=stack.includes('search_ranked')?'retrieval':stack.includes('forward_bounded_last_logits')?'forward_bounded':/mask_grammar|byte_grammar/.test(stack)?'grammar':stack.includes('mei_sdk_wasm_complete')?'other_wasm':'host_and_profiler';
  us[key]+=profile.timeDeltas[i];
 }
 return Object.fromEntries(Object.entries(us).map(([k,v])=>[k,v/1000]));
}
(async()=>{
 const out=`.local/cache/data-check/profile-${Date.now()}`;await fs.mkdir(out,{recursive:true});
 const browser=await chromium.launch({headless:true});
 try {
  const context=await browser.newContext({serviceWorkers:'block'});
  if(process.env.MEI_BENCH_WASM){
   const bytes=await fs.readFile(process.env.MEI_BENCH_WASM);const sha=require('node:crypto').createHash('sha256').update(bytes).digest('hex');
   await context.route('**/assets/runtime.wasm',route=>route.fulfill({body:bytes,contentType:'application/wasm'}));
   await context.route('**/assets/receipt.json',async route=>{const response=await route.fetch();await route.fulfill({json:{...await response.json(),wasm_sha256:sha,diagnostic_override:true}});});
  }
  const page=await context.newPage();await page.goto('http://127.0.0.1:8765');
  const setup=await page.evaluate(async()=>{
   const {loadModel}=await import('/sdk/wasm-abi.mjs');
   const {TOOLS,SAMPLE}=await import('/catalog.mjs');
   const manifest=await fetch('/assets/mei-model.json').then(r=>r.json());
   const [wasm,weights,vocab,toolIndex]=await Promise.all(['runtime.wasm','tensors.bin','tokenizer.model','tool-index.json'].map(async f=>new Uint8Array(await fetch('/assets/'+f).then(r=>r.arrayBuffer()))));
   const hash=async b=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',b)),x=>x.toString(16).padStart(2,'0')).join('');
   const t=performance.now();const {instance}=await WebAssembly.instantiate(wasm,{});
   globalThis.benchModel=loadModel({manifest,weights,vocab,toolIndex,wasm:instance});benchModel.registerTools(TOOLS);
   globalThis.benchMemory=instance.exports.memory;
   globalThis.benchRun=query=>{const s=benchModel.createSession({max_steps:2,runtime_profile:'compact'});const start=performance.now();try{return {elapsed_ms:performance.now()-start,result:s.complete({wire_version:'mei-runtime-wire-v2',query,context:{columns:SAMPLE[0]},evidence:[],history:[],tool_results:[],permissions:{},state:{},decode_mode:'constrained',max_new:128})};}catch(e){return {error:e.message,code:e.code};}finally{globalThis.lastElapsed=performance.now()-start;s.close();}};
   return {load_ms:performance.now()-t,wasm_sha256:await hash(wasm),weights_sha256:await hash(weights),columns:SAMPLE[0]};
  });
  const cdp=await page.context().newCDPSession(page);await cdp.send('Profiler.enable');await cdp.send('Profiler.setSamplingInterval',{interval:1000});
  const runs=[];
  for(const [i,query] of ['检查订单号列是否存在空值','检查订单号列是否存在空值','检查数量列是否为数字且在1到1000之间'].entries()){
   await cdp.send('Profiler.start');
   const run=await page.evaluate(q=>{const r=benchRun(q);r.elapsed_ms=lastElapsed;return r;},query);
   run.wasm_heap_bytes=await page.evaluate(()=>benchMemory.buffer.byteLength);
   const {profile}=await cdp.send('Profiler.stop');await fs.writeFile(`${out}/${i}.cpuprofile`,JSON.stringify(profile));
   const hits=new Map();for(const n of profile.nodes)hits.set(n.callFrame.functionName,(hits.get(n.callFrame.functionName)||0)+(n.hitCount||0));
   run.sampled_stage_ms=stages(profile);run.top_samples=[...hits].sort((a,b)=>b[1]-a[1]).slice(0,15);run.query=query;runs.push(run);
   console.log(JSON.stringify({i,elapsed_ms:run.elapsed_ms,stats:run.result?.stats,error:run.error,top:run.top_samples}));
  }
  await page.evaluate(()=>benchModel.unload());
  const worker=await page.evaluate(async()=>{
   const {SAMPLE}=await import('/catalog.mjs');const w=new Worker('/mei-worker.mjs',{type:'module'});
   const send=data=>new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(Error('Worker timeout')),60000);w.onmessage=({data:r})=>{clearTimeout(timeout);resolve(r);};w.onerror=e=>{clearTimeout(timeout);reject(Error(e.message));};w.postMessage(data);});
   try {
    const start=performance.now();const init=await send({id:'init',type:'init'});if(init.error||!init.ready)throw Error(init.error||'Worker not ready');const init_ms=performance.now()-start;const requests=[];
    for(let i=0;i<3;i++){
     const start=performance.now();const result=await send({id:i,rule:{id:'r1',instruction:'检查订单号列是否存在空值',tool:'check_required',args:{column:'订单号'},severity:'block'},sheet:{name:'Sheet1',rows:SAMPLE,rowNumbers:SAMPLE.map((_,i)=>i+1)}});
     if(result.error||!result.report?.attempted)throw Error(result.error||'Worker inference not attempted');
     requests.push({roundtrip_ms:performance.now()-start,...result});
    }
    return {init_ms,requests};
   }finally{w.terminate();}
  });
  console.log(JSON.stringify({worker_init_ms:worker.init_ms,worker_request_ms:worker.requests.map(r=>r.roundtrip_ms)}));
  await fs.writeFile(`${out}/report.json`,JSON.stringify({setup,runs,worker,cpu:os.cpus()[0].model,browser:browser.version(),date:new Date().toISOString(),measurement:'WASM identified by setup.wasm_sha256 sampled at 1 ms. forward_bounded is prefill for one-output-token [] requests; errors may also include later forward passes. Stage times are sampling estimates, not exact instrumentation.'},null,2));console.log(out);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
