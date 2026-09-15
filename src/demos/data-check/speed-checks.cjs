// Real WASM checks: private request state, prefix reuse, resource bounds, honest profile.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/xuehongwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs/promises'),assert=require('node:assert/strict');
(async()=>{
 const bytes=await fs.readFile(process.env.MEI_BENCH_WASM);
 const browser=await chromium.launch({headless:true});
 try {
  const context=await browser.newContext({serviceWorkers:'block'});
  await context.route('**/assets/runtime.wasm',route=>route.fulfill({body:bytes,contentType:'application/wasm'}));
  const page=await context.newPage();await page.goto('http://127.0.0.1:8765');
  const report=await page.evaluate(async()=>{
   const {loadModel}=await import('/sdk/wasm-abi.mjs');const {TOOLS,SAMPLE}=await import('/catalog.mjs');
   const manifest=await fetch('/assets/mei-model.json').then(r=>r.json());
   const [wasm,weights,vocab,toolIndex]=await Promise.all(['runtime.wasm','tensors.bin','tokenizer.model','tool-index.json'].map(async f=>new Uint8Array(await fetch('/assets/'+f).then(r=>r.arrayBuffer()))));
   const {instance}=await WebAssembly.instantiate(wasm,{});
   const load=()=>{const model=loadModel({manifest,weights,vocab,toolIndex,wasm:instance});model.registerTools(TOOLS);return model;};
   let model=load();const capabilities=model.capabilities();const runs=[];
   const query='检查订单号列是否存在空值';
   const run=(label,columns,q=query)=>{
    const s=model.createSession({max_steps:2,runtime_profile:'compact'});const start=performance.now();
    let result,error;try{result=s.complete({wire_version:'mei-runtime-wire-v2',query:q,context:{columns},evidence:[],history:[],tool_results:[],permissions:{},state:{},decode_mode:'constrained',max_new:128});}catch(e){error={code:e.code,message:e.message};}finally{s.close();}
    const signature=result?{kind:result.kind,raw_text:result.raw_text,generated:result.generated_token_ids,topk:result.prefill_topk_ids,confidence:result.confidence,mw:result.mw_disposition,call:result.call?{name:result.call.name,arguments:result.call.arguments}:null,refusal:result.refusal}:error;
    const row={label,ms:performance.now()-start,signature,cache:result?.runtime_cache,compute_profile:result?.compute_profile,heap_bytes:instance.exports.memory.buffer.byteLength};runs.push(row);return row;
   };
   try {
    run('cold A',SAMPLE[0]);run('warm A',SAMPLE[0]);
    run('changed context B with cached prefix',['订单号','审核备注','数量','金额']);
    model.unload();model=load();run('fresh model B',['订单号','审核备注','数量','金额']);
    run('changed task','订单号,数量,单价,金额,状态'.split(','),'检查订单号列是否有重复值');
    run('A after task change',SAMPLE[0]);
    return {capabilities,runs};
   }finally{model.unload();}
  });
  assert.deepEqual(report.runs[0].signature,report.runs[1].signature,'cached prefix changes identical request');
  assert.deepEqual(report.runs[2].signature,report.runs[3].signature,'dynamic context leaked across requests');
  assert.deepEqual(report.runs[0].signature,report.runs[5].signature,'task switching leaks KV state');
  assert.equal(report.runs[1].cache.prefix_cache.hits,1);
  assert.equal(report.capabilities.release_eligible,false);
  assert.equal(report.capabilities.compute_profile.experimental,true);
  assert(report.runs.every(r=>r.heap_bytes<=96*1024*1024),'tested WASM heap exceeds 96 MiB');
  report.ok=true;report.wasm_sha256=require('node:crypto').createHash('sha256').update(bytes).digest('hex');report.browser=browser.version();
  const out=process.env.MEI_BENCH_OUTPUT||`.local/cache/data-check/speed-checks-${Date.now()}.json`;
  await fs.writeFile(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({out,ok:report.ok,runs:report.runs.map(r=>({label:r.label,ms:r.ms,heap_mib:r.heap_bytes/1048576,hits:r.cache?.prefix_cache?.hits}))},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
